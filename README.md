# TriaGS: Differentiable Triangulation-Guided Geometric Consistency for 3D Gaussian Splatting

Accepted at WACV 2026 : [The IEEE/CVF Winter Conference on Applications of Computer Vision 2026](https://wacv.thecvf.com/)

## Authors:
- [Quan Tran]()
- [Tuan Dang](https://tuandang.info)
> All authors are with [**Cognitive Robotics Laboratory**](https://www.cogniboticslab.org/)
>
> **Department of Electrical Engineering and Computer Science (EECS)**
>
> **University of Arkansas**, Fayetteville, AR 72701, USA.

## Astract
3D Gaussian Splatting is crucial for real-time novel view synthesis due to its efficiency and ability to render photorealistic images. However, building a 3D Gaussian is guided solely by photometric loss, which can result in inconsistencies in reconstruction. This under-constrained process often results in "floater" artifacts and an unstructured geometry, preventing the extraction of high-fidelity surfaces. To address this issue, our paper introduces a novel method that improves reconstruction by enforcing global geometry consistency through constrained multi-view triangulation. Our approach aims to achieve a consensus on 3D representation in the physical world by utilizing various estimated views. We optimize this process by evaluating a 3D point against a robust consensus point, which is re-triangulated from a bundle of neighboring views in a self-supervised fashion. We demonstrate the effectiveness of our method across multiple datasets, achieving state-of-the-art results. On the DTU dataset, our method attains a mean Chamfer Distance of $0.50$ mm, outperforming comparable explicit methods. We will make our code open-source to facilitate community validation and ensure reproducibility.

# Overview
<p align="center">
    <img src="images/overview.png"  width="640"/><br/>
</p>


## Environment and dependencies setup
``` shell
# 1. Clone the repository
git clone [[your-github-repo-url]](https://github.com/cogniboticslab/triags)
cd TriaGS

# 2. Create and activate a Conda environment
conda create -n triags python=3.10
conda activate triags

# 3. Install PyTorch
pip install torch torchvision

# 4. Install other Python dependencies
pip install -r requirements.txt

# 5. Install custom CUDA extensions
# Ensure the CUDA toolkit is installed and nvcc is in your PATH
# For example:
# export PATH=/usr/local/cuda-12.8/bin:${PATH}
# export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH}

pip install submodules/diff-gaussian-rasterization
pip install git+https://github.com/camenduru/simple-knn

# tetra-nerf for Marching Tetrahedra
cd submodules/tetra_triangulation
conda install cmake
conda install conda-forge::gmp
conda install conda-forge::cgal
cmake .
make
pip install -e .
```
## Dataset
### DTU
Download preprocessed data from [2DGS](https://drive.google.com/drive/folders/1SJFgt8qhQomHX55Q4xSvYE2C6-8tFll9) and ground truth point clouds from the [official DTU website](https://roboimagedata.compute.dtu.dk/). The data should be organized as follows:
```
dtu_dataset
├── dtu
│   ├── scan24
│   │   ├── images
│   │   ├── mask
│   │   ├── sparse
│   │   ├── cameras_sphere.npz
│   │   └── cameras.npz
│   └── ...
├── dtu_eval
│   ├── Points
│   │   └── stl
│   └── ObsMask
```

### TNT dataset
Download the training dataset from the [TNT website](https://www.tanksandtemples.org/download/). You will need the image set, Camera Poses, Alignment, Cropfiles and ground truth. The dataset should be organized as follows
```
TNT
|---Barn
|	|---images_raw
|	|	|---000001.jpg
|	|	|---...
|	|---Barn_COLMAP_SfM.log
|	|---Barn_trans.txt
|	|---Barn.json
|	|---Barn.ply
|---...
```

Afterwards, the dataset should look as follows:
```
TNT
|---Barn
|	|---images
|	|	|---000001.jpg
|	|	|---...
|	|---sparse
|	|	|---0
|	|	|	|---cameras.bin
|	|	|	|---cameras.txt
|	|	|	|---images.bin
|	|	|	|---images.txt
|	|	|	|---points3D.bin
|	|	|	|---points3D.txt
|	|---Barn_COLMAP_SfM.log
|	|---Barn_trans.txt
|	|---Barn.json
|	|---Barn.ply
|	|---database.db
|---...
```

The TNT evaluation code requires an older version of Open3D (9.0). create a separate Conda environment with the following libraries:
```
matplotlib>=1.3
open3d==0.9
```
### NeRF Synthetic
Download from the original [NeRF repository](https://www.matthewtancik.com/nerf).

## Training and Evaluation
### DTU Dataset
```
# 1. Train the model on a specific DTU scan
python train.py -s <path_to_dtu_scan_data> -m <output_folder> -r 2 --use_decoupled_appearance

# 2. Extract a mesh using TSDF fusion
python mesh_extract.py -s <path_to_dtu_scan_data> -m <output_folder>

# 3. Evaluate the extracted mesh
python eval/eval_dtu/evaluate_single_scene.py \
    --input_mesh <output_folder>/recon.ply \
    --scan_id <scan_id> \
    --output_dir <output_folder>/evaluation_results \
    --mask_dir <path_to_dtu_dataset_root>/dtu \
    --DTU <path_to_dtu_eval_data>
```

### Tanks and Temples Dataset
```
# 1. Train the model on a scene
python train.py -s <path_to_tnt_scene> -m <output_folder> -r 2 --eval --use_decoupled_appearance

# 2. Extract a mesh using marching tetrahedra
python mesh_extract_tetrahedra.py -s <path_to_tnt_scene> -m <output_folder> --iteration 30000

# 3. Evaluate the mesh (requires open3d==0.9.0)
python eval/eval_tnt/run.py \
    --dataset-dir <path_to_gt_tnt_scene> \
    --traj-path <path_to_tnt_scene>/<scene_name>_COLMAP_SfM.log \
    --ply-path <output_folder>/recon.ply \
    --out-dir <output_folder>/evaluation_results
```
### Novel View Synthesis (e.g., NeRF Synthetic)
```
# 1. Train the model
python train.py -s <path_to_nerf_synthetic_scene> -m <output_folder> --eval

# 2. Render test views
python render.py -m <output_folder>

# 3. Compute image quality metrics (PSNR, SSIM, LPIPS)
python metrics.py -m <output_folder>
```

# Results

See more results in the paper and supplemental documents (will be released soon).

## Quatitative Results
<p align="center">
    <img src="images/results.png"  width="640"/><br/>
</p>

## Qualitative Results
<p align="center">
    <img src="images/results-1.png"  width="640"/><br/>
</p>

## Acknowledgement
The implementation of this project is built upon the official PyTorch implementation of **[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)**. Our methodology was influenced by the work of **[PGSR](https://zju3dv.github.io/pgsr/)** and **[Radegs](https://baowenz.github.io/radegs/)**, which provided valuable insights into surface reconstruction and regularization techniques. For data, the preprocessing scripts for the DTU and Tanks and Temples datasets were adapted from the **[Neuralangelo project](https://github.com/NVlabs/neuralangelo/blob/main/DATA_PROCESSING.md)**. The evaluation protocols are based on **[DTUeval-python](https://github.com/jzhangbs/DTUeval-python)** and the official **[TanksAndTemples evaluation scripts](https://github.com/isl-org/TanksAndTemples/tree/master/python_toolbox/evaluation)**. We are grateful to all the authors for their excellent work and for making their code publicly available to the research community.

## Citing
```
@inproceedings{tran2025triags,
  title        = {TriaGS: Differentiable Triangulation-Guided Geometric Consistency for 3D Gaussian Splatting},
  author       = {Tran, Quan and Dang, Tuan},
  booktitle    = {The IEEE/CVF Winter Conference on Applications of Computer Vision 2026 (WACV)},
  year         = {2026},
  month        = March,
  note         = {},
  url          = {},
}
```
