import torch
import random
import numpy as np
import torch.nn.functional as F
from utils.loss_utils import lncc

# Geman-McClure loss function (special case of Barron's robust loss with alpha = -2)
# This loss is robust to outliers and is often used for regression tasks.
# The formula is:
#     loss = (error^2 / 2) / ( (error^2 / 2) + scale^2 )
# - 'error' is the difference between prediction and target.
# - 'scale' controls how strongly outliers are down-weighted (higher = more tolerant).
# This loss smoothly limits the influence of large errors, making optimization more stable.
def geman_mcclure_loss(error, scale = 0.1):
    error_sq = error**2
    return (error_sq / 2.0) / (error_sq / 2.0 + scale**2)

def patch_offsets(h_patch_size, device):
    offsets = torch.arange(-h_patch_size, h_patch_size + 1, device=device)
    return torch.stack(torch.meshgrid(offsets, offsets, indexing='xy')[::-1], dim=-1).view(1, -1, 2)

def patch_warp(H, uv):
    B, P = uv.shape[:2]
    H = H.view(B, 3, 3)
    ones = torch.ones((B,P,1), device=uv.device)
    homo_uv = torch.cat((uv, ones), dim=-1)

    grid_tmp = torch.einsum("bik,bpk->bpi", H, homo_uv)
    grid_tmp = grid_tmp.reshape(B, P, 3)
    grid = grid_tmp[..., :2] / (grid_tmp[..., 2:] + 1e-10)
    return grid


def compute_multi_view_losses(viewpoint_cam, nearest_cam, scene, gaussians, render_pkg, nearest_render_pkg, gt_image_gray, opt, use_virtul_cam=False):
    """
    Compute geometry consistency (geo_loss) and normalized cross-correlation (ncc_loss).
    Also returns candidate indices and supporting arrays for TGPC triangulation.

    Returns a dict with keys:
      - geo_loss: tensor or None
      - ncc_loss: tensor or None
      - tgpc_candidates: 1D tensor of indices into flattened point arrays (or empty tensor)
      - pts: (N,3) tensor of 3D reference points (flattened)
      - pixels: (N,2) tensor of pixel coordinates (flattened)
      - weights: (N,) tensor of weights (flattened)
      - d_mask: (N,) boolean mask flattened
    """
    H, W = render_pkg['median_depth'].squeeze().shape
    ix, iy = torch.meshgrid(
        torch.arange(W), torch.arange(H), indexing='xy')
    pixels = torch.stack([ix, iy], dim=-1).float().to(render_pkg['median_depth'].device)

    # get nearest render and depth mapping
    pts = gaussians.get_points_from_depth(viewpoint_cam, render_pkg['median_depth'])
    pts_in_nearest_cam = pts @ nearest_cam.world_view_transform[:3,:3] + nearest_cam.world_view_transform[3,:3]
    map_z, d_mask = gaussians.get_points_depth_in_depth_map(nearest_cam, nearest_render_pkg['median_depth'], pts_in_nearest_cam)

    pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:,2:3])
    pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[...,None]
    R = torch.tensor(nearest_cam.R).float().cuda()
    T = torch.tensor(nearest_cam.T).float().cuda()
    pts_ = (pts_in_nearest_cam-T)@R.transpose(-1,-2)
    pts_in_view_cam = pts_ @ viewpoint_cam.world_view_transform[:3,:3] + viewpoint_cam.world_view_transform[3,:3]
    pts_projections = torch.stack(
                [pts_in_view_cam[:,0] * viewpoint_cam.Fx / pts_in_view_cam[:,2] + viewpoint_cam.Cx,
                pts_in_view_cam[:,1] * viewpoint_cam.Fy / pts_in_view_cam[:,2] + viewpoint_cam.Cy], -1).float()
    pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
    d_mask = d_mask & (pixel_noise < opt.multi_view_pixel_noise_th)
    weights = (1.0 / torch.exp(pixel_noise)).detach()
    weights[~d_mask] = 0

    geo_loss = None
    ncc_loss = None

    final_d_mask_flat = d_mask.reshape(-1)
    candidate_indices_for_tgpc = torch.arange(final_d_mask_flat.shape[0], device=final_d_mask_flat.device)[final_d_mask_flat]
    final_valid_indices_for_tgpc = candidate_indices_for_tgpc

    if d_mask.sum() > 0:
        geo_weight = opt.multi_view_geo_weight
        geo_loss = geo_weight * ((weights * pixel_noise)[d_mask]).mean()

        # photometric (NCC) loss: sample patches and compute lncc
        # We follow original sampling logic: pick up to sample_num points
        d_mask_flat = d_mask.reshape(-1)
        valid_indices = torch.arange(d_mask_flat.shape[0], device=d_mask_flat.device)[d_mask_flat]
        num_valid = int(d_mask.sum().item())
        if num_valid > opt.multi_view_sample_num:
            chosen = np.random.choice(num_valid, opt.multi_view_sample_num, replace=False)
            # chosen indexes are into the array of valid indices
            chosen_t = torch.from_numpy(chosen).to(valid_indices.device, dtype=torch.long)
            valid_indices = valid_indices[chosen_t]

        weights_valid = weights.reshape(-1)[valid_indices]
        pixels_valid = pixels.reshape(-1,2)[valid_indices]
        offsets = patch_offsets(opt.multi_view_patch_size, pixels_valid.device)
        ori_pixels_patch = pixels_valid.reshape(-1, 1, 2) / viewpoint_cam.ncc_scale + offsets.float()

        H_img, W_img = gt_image_gray.squeeze().shape
        pixels_patch = ori_pixels_patch.clone()
        pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W_img - 1) - 1.0
        pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H_img - 1) - 1.0
        ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2), align_corners=True)
        ref_gray_val = ref_gray_val.reshape(-1, (opt.multi_view_patch_size * 2 + 1) ** 2)

        # compute homography and sample from nearest image
        ref_local_n = render_pkg["normal"].permute(1,2,0)
        ref_local_n = ref_local_n.reshape(-1,3)[valid_indices]

        # Render distance / depth for homography
        rays_d = viewpoint_cam.get_rays()
        rendered_normal2 = render_pkg["normal"].permute(1,2,0).reshape(-1,3)
        ref_local_d = render_pkg['median_depth'].view(-1) * ((rendered_normal2 * rays_d.reshape(-1,3)).sum(-1).abs())
        ref_local_d = ref_local_d.reshape(*render_pkg['median_depth'].shape)
        ref_local_d = ref_local_d.reshape(-1)[valid_indices]

        ref_to_neareast_r = nearest_cam.world_view_transform[:3,:3].transpose(-1,-2) @ viewpoint_cam.world_view_transform[:3,:3]
        ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3,:3] + nearest_cam.world_view_transform[3,:3]

        H_ref_to_neareast = ref_to_neareast_r[None] - \
            torch.matmul(ref_to_neareast_t[None,:,None].expand(ref_local_d.shape[0],3,1),
                        ref_local_n[:,:,None].expand(ref_local_d.shape[0],3,1).permute(0, 2, 1))/ref_local_d[...,None,None]
        H_ref_to_neareast = torch.matmul(nearest_cam.get_k(nearest_cam.ncc_scale)[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_neareast)
        H_ref_to_neareast = H_ref_to_neareast @ viewpoint_cam.get_inv_k(viewpoint_cam.ncc_scale)

        grid = patch_warp(H_ref_to_neareast.reshape(-1,3,3), ori_pixels_patch)
        grid[:, :, 0] = 2 * grid[:, :, 0] / (W_img - 1) - 1.0
        grid[:, :, 1] = 2 * grid[:, :, 1] / (H_img - 1) - 1.0
        _, nearest_image_gray = nearest_cam.get_image()
        sampled_gray_val = F.grid_sample(nearest_image_gray[None], grid.reshape(1, -1, 1, 2), align_corners=True)
        sampled_gray_val = sampled_gray_val.reshape(-1, (opt.multi_view_patch_size * 2 + 1) ** 2)

        ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)
        mask = ncc_mask.reshape(-1)
        ncc_vals = ncc.reshape(-1) * weights_valid
        ncc_vals = ncc_vals[mask].squeeze()

        if mask.sum() > 0:
            ncc_loss = opt.multi_view_ncc_weight * ncc_vals.mean()

    # prepare return values for TGPC
    return {
        'geo_loss': geo_loss,
        'ncc_loss': ncc_loss,
        'tgpc_candidates': final_valid_indices_for_tgpc if 'final_valid_indices_for_tgpc' in locals() else torch.tensor([], device=render_pkg['median_depth'].device, dtype=torch.long),
        'pts': pts.reshape(-1,3),
        'pixels': pixels.reshape(-1,2),
        'weights': weights.reshape(-1),
        'd_mask': d_mask.reshape(-1)
    }


def compute_tgpc_loss(final_valid_indices_for_tgpc, pts_flat, pixels_flat, weights_flat, viewpoint_cam, scene, gaussians, render_pkg, nearest_cam, opt, tgpc_num_neighbors=1):
    """
    Compute TGPC triangulation loss for the provided candidate indices. Returns the scalar TGPC loss tensor.
    """
    if final_valid_indices_for_tgpc is None or final_valid_indices_for_tgpc.numel() == 0:
        return None

    # sample the points used for TGPC (use all candidates as in original code)
    X_r_sampled_tgpc = pts_flat.reshape(-1,3)[final_valid_indices_for_tgpc]
    p_r_sampled_coords_tgpc = pixels_flat.reshape(-1,2)[final_valid_indices_for_tgpc]
    u_r_tgpc, v_r_tgpc = p_r_sampled_coords_tgpc[:,0], p_r_sampled_coords_tgpc[:,1]

    K_r_tgpc = viewpoint_cam.get_k()
    RT_r_w2c_tgpc = viewpoint_cam.world_view_transform.transpose(0,1)[:3,:]
    P_r_tgpc = K_r_tgpc @ RT_r_w2c_tgpc

    A_rows_list_tgpc = []
    row0_r = u_r_tgpc.unsqueeze(-1) * P_r_tgpc[2,:] - P_r_tgpc[0,:]
    row1_r = v_r_tgpc.unsqueeze(-1) * P_r_tgpc[2,:] - P_r_tgpc[1,:]
    Ac_r = torch.stack([row0_r, row1_r], dim=1)
    norm_Ac_r = torch.linalg.norm(Ac_r, ord='fro', dim=(1,2), keepdim=True)
    Ac_r_normalized = (Ac_r / norm_Ac_r)
    A_rows_list_tgpc.append(Ac_r_normalized[:,0,:].unsqueeze(1))
    A_rows_list_tgpc.append(Ac_r_normalized[:,1,:].unsqueeze(1))

    # select neighbor cameras
    selected_neighbor_cams_for_tgpc = []
    neighbor_ids = viewpoint_cam.nearest_id
    if neighbor_ids:
        num_to_sample = min(tgpc_num_neighbors, len(neighbor_ids))
        chosen_indices = random.sample(neighbor_ids, num_to_sample)
        selected_neighbor_cams_for_tgpc = [scene.getTrainCameras()[idx] for idx in chosen_indices]

    if nearest_cam is not None and nearest_cam not in selected_neighbor_cams_for_tgpc:
        selected_neighbor_cams_for_tgpc.append(nearest_cam)
        selected_neighbor_cams_for_tgpc = selected_neighbor_cams_for_tgpc[:tgpc_num_neighbors]

    for neighbor_cam_for_tgpc in selected_neighbor_cams_for_tgpc:
        X_r_in_current_neighbor_space = X_r_sampled_tgpc @ neighbor_cam_for_tgpc.world_view_transform[:3,:3] + neighbor_cam_for_tgpc.world_view_transform[3,:3]
        z_curr_n_proj_tgpc = X_r_in_current_neighbor_space[:, 2:3]
        p_curr_n_sampled_coords_tgpc = torch.stack(
            [X_r_in_current_neighbor_space[:,0] * neighbor_cam_for_tgpc.Fx / z_curr_n_proj_tgpc.squeeze(-1) + neighbor_cam_for_tgpc.Cx,
             X_r_in_current_neighbor_space[:,1] * neighbor_cam_for_tgpc.Fy / z_curr_n_proj_tgpc.squeeze(-1) + neighbor_cam_for_tgpc.Cy], dim=-1)
        u_curr_n_tgpc, v_curr_n_tgpc = p_curr_n_sampled_coords_tgpc[:,0], p_curr_n_sampled_coords_tgpc[:,1]

        K_curr_n_tgpc = neighbor_cam_for_tgpc.get_k()
        RT_curr_n_w2c_tgpc = neighbor_cam_for_tgpc.world_view_transform.transpose(0,1)[:3,:]
        P_curr_n_tgpc = K_curr_n_tgpc @ RT_curr_n_w2c_tgpc

        row0_n_curr = u_curr_n_tgpc.unsqueeze(-1) * P_curr_n_tgpc[2,:] - P_curr_n_tgpc[0,:]
        row1_n_curr = v_curr_n_tgpc.unsqueeze(-1) * P_curr_n_tgpc[2,:] - P_curr_n_tgpc[1,:]
        Ac_n_curr = torch.stack([row0_n_curr, row1_n_curr], dim=1)
        norm_Ac_n_curr = torch.linalg.norm(Ac_n_curr, ord='fro', dim=(1,2), keepdim=True)
        Ac_n_curr_normalized = Ac_n_curr / norm_Ac_n_curr
        A_rows_list_tgpc.append(Ac_n_curr_normalized[:,0,:].unsqueeze(1))
        A_rows_list_tgpc.append(Ac_n_curr_normalized[:,1,:].unsqueeze(1))

    assert (2 * len(selected_neighbor_cams_for_tgpc) + 2) == len(A_rows_list_tgpc)
    assert len(A_rows_list_tgpc) >= 4, "Not enough equations for triangulation in TGPC loss computation"

    A_Xr_tgpc = torch.cat(A_rows_list_tgpc, dim=1)

    try:
        _U_svd, S_svd, Vh_svd = torch.linalg.svd(A_Xr_tgpc, full_matrices=False)
        X_triangulated_homo_tgpc = Vh_svd[:, -1, :]
        X_triangulated_homo_tgpc = X_triangulated_homo_tgpc / (X_triangulated_homo_tgpc[:, 3:4].clone().abs().clamp(min=1e-8) * torch.sign(X_triangulated_homo_tgpc[:, 3:4].clone().clamp(min=1e-8)))
        X_triangulated_cartesian_tgpc = X_triangulated_homo_tgpc[:, :3]

        ref_depths_sampled = render_pkg['median_depth'].squeeze().reshape(-1)[final_valid_indices_for_tgpc]
        inverse_depths = (1.0 / ref_depths_sampled)
        depth_weights = inverse_depths / torch.max(inverse_depths).detach()
        depth_weights = depth_weights.detach()

        scale = 0.1
        error_3d = X_r_sampled_tgpc - X_triangulated_cartesian_tgpc
        gm_loss_per_dimension = geman_mcclure_loss(error_3d, scale=scale)
        threeD_distance = gm_loss_per_dimension.sum(dim=1)

        # reprojection weights (use the weights sampled in compute_multi_view_losses)
        weights_for_tgpc = weights_flat.reshape(-1)[final_valid_indices_for_tgpc]
        weights_final = depth_weights * weights_for_tgpc
        current_tgpc_loss = (weights_final * threeD_distance).mean()
        return current_tgpc_loss
    except torch.linalg.LinAlgError:
        raise RuntimeError("SVD failed during TGPC loss computation")
