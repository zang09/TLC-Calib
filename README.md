<p align="center">

<h1 align="center">Targetless LiDAR-Camera Calibration with Neural Gaussian Splatting</h1>
  <p align="center">
    <a href="https://haebeom.com/" target="_blank">Haebeom Jung</a><sup>1</sup>
    &middot;
    Namtae Kim<sup>1</sup>
    &middot;
    Jungwoo Kim<sup>2</sup>
    &middot;
    <a href="https://jaesik.info/" target="_blank">Jaesik Park</a><sup>1</sup>
  </p>
  <p align="center">
    <sup>1</sup>Seoul National University &middot; <sup>2</sup>Yonsei University
  </p>

<h2 align="center">IEEE RA-L 2026 (ICRA 2026)</h2>

<h3 align="center">
  <a href="https://arxiv.org/abs/2504.04597" target="_blank">arXiv</a>
  |
  <a href="https://ieeexplore.ieee.org/document/11397170" target="_blank">Paper</a>
  |
  <a href="https://www.haebeom.com/tlc-calib-site/" target="_blank">Project Page</a>
  |
  <a href="https://drive.google.com/drive/folders/1P9EcXuyUL9NZpgj-IU44-UfJUDiEZ7zg?usp=drive_link" target="_blank">Dataset</a>
</h3>
  <p align="center">
    <a href="https://arxiv.org/abs/2504.04597"><img src="https://img.shields.io/badge/arXiv-2504.04597-b31b1b.svg"></a>
    <a href="https://ieeexplore.ieee.org/document/11397170"><img src="https://img.shields.io/badge/IEEE-11397170-00629B.svg"></a>
    <a href="https://www.haebeom.com/tlc-calib-site/"><img src="https://img.shields.io/badge/Project%20Page-TLC--Calib-00B894.svg"></a>
    <a href="https://drive.google.com/drive/folders/1P9EcXuyUL9NZpgj-IU44-UfJUDiEZ7zg?usp=drive_link"><img src="https://img.shields.io/badge/Dataset-Google%20Drive-green.svg"></a>
  </p>
</p>

<p align="center">
  <a href="https://www.youtube.com/watch?v=HxDTiVllGe0" target="_blank">
    <img src="https://img.youtube.com/vi/HxDTiVllGe0/maxresdefault.jpg" alt="TLC-Calib demo video" width="90%">
  </a>
</p>

<p align="center">
TLC-Calib is a targetless LiDAR-camera calibration framework based on Neural Gaussian Splatting. Given multi-camera images, LiDAR poses, and a LiDAR scan, TLC-Calib jointly optimizes a neural scene representation and the LiDAR-camera extrinsics without requiring calibration boards or manually designed targets. We also provide a web viewer for real-time inspection of the calibration process.</p>
<br>

**Updates:**

- 🎉 May 23, 2026: Release the full codebase.
- 📦 Apr 15, 2026: Release processed public datasets in the TLC-Calib format.
- 📝 Jan 13, 2026: Paper accepted to IEEE RA-L 2026.

## Installation

Clone the repository with submodules:

```bash
git clone https://github.com/zang09/TLC-Calib.git --recursive
cd TLC-Calib
```

If you already cloned the repository without `--recursive`, initialize the submodules with:

```bash
git submodule update --init --recursive
```

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate tlc_calib
```

The provided environment has been tested with PyTorch 2.1.2 and CUDA 11.8. The CUDA rasterizer and simple-knn extensions are installed from the local submodules through `environment.yml`.

## Data Preparation

We provide processed versions of public datasets, including Waymo, KITTI-360, and FAST-LIVO2, converted into the TLC-Calib format. Each scene contains synchronized camera images, LiDAR poses, camera intrinsics, LiDAR-camera extrinsics, per-frame point clouds, and scene-level LiDAR maps.

The dataset root follows this structure:

```text
<dataset_scene>/
├── images/          # image_00, image_01, ...
│   ├── image_00/
│   ├── image_01/
│   ├── ...
│   └── image_XX/
├── lidar/           # rgb_map.ply, map.ply
├── params/          # poses, intrinsics, extrinsics, timestamps
├── pcds/            # per-frame LiDAR point clouds
├── README.md
└── valid_frame.txt
```

All modalities use the same zero-based local frame index. For example, `images/image_00/000123.png`, `pcds/000123.pcd`, and line `123` in `params/lidars.txt` refer to the same sample.

Important files in `params/` include `intrinsics.txt`, `lidars.txt`, `cams_to_lidar_gt.txt`, `cams_to_lidar_init.txt`, optional `timestamps.txt`, and camera-indexed files such as `cam0.txt` and `cam0_to_lidar.txt`. Supported dataset names include `kitti-360`, `waymo`, and `fast-livo2`; custom data can use the same layout. For more details, please refer to the `README.md` included in the [dataset link](https://drive.google.com/drive/folders/1P9EcXuyUL9NZpgj-IU44-UfJUDiEZ7zg?usp=drive_link).

## Calibration

Run the default TLC-Calib optimization with:

```bash
python train.py -s <dataset_scene_folder> -m <output_path> \
  --eval --from_lidar --use_rig --opt_pose --pose_scheduler --adaptive_voxel \
  --dataset <dataset_name>
```

Example:

```bash
python train.py -s data/TLC-Calib/kitti-360/large_rotation \
  -m outputs/kitti-360/large_rotation/eval \
  --eval --from_lidar --use_rig --opt_pose --pose_scheduler --adaptive_voxel \
  --dataset kitti-360
```

Useful options:

- `--from_lidar`: initialize camera poses from LiDAR poses. If disabled, camera poses are initialized from the ground-truth LiDAR-camera extrinsics.
- `--use_rig`: optimize one shared rig transform per camera.
- `--opt_pose`: enable calibration pose optimization.
- `--pose_scheduler`: schedule pose optimization during training.
- `--adaptive_voxel`: compute the LiDAR voxel size from trajectory length.
- `--refine`: run an additional NVS refinement stage after calibration.
- `--viewer --port 8080`: launch the web viewer to monitor the calibration process in real time.
- `--viewer_camera_step`: show every N-th camera frame in the viewer to keep visualization lightweight for long sequences.

## Evaluation

Evaluate calibration pose errors for one trained output:

```bash
python metrics_pose.py -m <output_path>
```

This writes pose calibration results to:

```text
<output_path>/rig_results.json
```

Evaluate NVS quality using the calibrated LiDAR-camera poses:

```bash
python metrics_nvs.py -m <output_path>
```

`metrics_nvs.py` reads `<output_path>/config.yml` to recover the source scene, then runs `nvs_eval/train.py` with `<output_path>/point_cloud/iteration_30000/cams_to_lidar.txt` as the prior pose. The NVS run and summary are saved inside the calibration output:

```text
<output_path>/nvs_eval/ours_30000/
<output_path>/nvs_results.json
```

Run full evaluation over a dataset root:

```bash
python full_eval.py \
  --data_path <dataset_root> \
  --output_path outputs \
  --datasets kitti-360 waymo fast-livo2 \
  --repeat 1
```

If `--data_path` is omitted, update `DEFAULT_DATA_PATH` in `full_eval.py` to the location where you downloaded the processed TLC-Calib dataset.

For each scene/run, `full_eval.py` runs calibration training, pose metrics, NVS metrics, and scene-level aggregation. Extra training options are forwarded to `train.py`.

Aggregate existing outputs without retraining. The `-a` path can be any output level:

```bash
python full_eval.py -a <scene_output_path>    # e.g., outputs/kitti-360/straight/
python full_eval.py -a <dataset_output_path>  # e.g., outputs/kitti-360/
python full_eval.py -a <output_root>          # e.g., outputs/
```

- `<scene_output_path>` aggregates one scene and writes `exp_rig_results.json`, `exp_nvs_results.json`, and `train_results.json` inside that scene directory.
- `<dataset_output_path>` aggregates every scene under one dataset output directory, then writes `<dataset_output_path>/full_eval_results.json`.
- `<output_root>` aggregates every dataset output directory under the root, then writes one `full_eval_results.json` inside each dataset output directory.

For each scene output, aggregation averages pose, NVS, and training results across run directories such as `eval_01` or `eval_02`.

## Rendering (Optional)

<details>
<summary>Show optional rendering instructions</summary>

**`render.py` directly visualizes the trained TLC-Calib calibration model.** It is useful for inspecting the calibration training output itself, but it does not retrain an NVS model or compute PSNR/SSIM/LPIPS.

**For NVS benchmark images and metrics, use `metrics_nvs.py`.** `metrics_nvs.py` retrains the NVS model with the calibrated poses and writes rendered test images plus `PSNR`, `SSIM`, and `LPIPS`.

Render train or test views from a saved calibration model:

```bash
python render.py -m <output_path> --iteration 30000
```

Rendered images, ground truth images, error maps, and comparison images are stored under `train/ours_<iteration>/` and `test/ours_<iteration>/`.

</details>

## Outputs

A standard run saves:

- `config.yml`: training and dataset configuration for reproduction.
- `point_cloud/iteration_<N>/`: optimized Gaussian representation and calibrated rig files.
- `train_info.json`: training time and memory statistics.
- `outputs.log`: training and evaluation logs.
- `rig_results.json`: rotation and translation calibration errors for each camera plus an `Average` entry.
- `train/` and `test/`: rendered evaluation views when rendering is requested.

## Acknowledgement

This repository builds on the excellent open-source projects [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), [Scaffold-GS](https://github.com/city-super/Scaffold-GS), and [MonoGS](https://github.com/muskie82/MonoGS). We thank the authors for making their code available to the research community.

------

If you use our paper, code, dataset, or any part of this repository in your research, please cite our work:

```bibtex
@article{jung2026targetless,
  title   = {{Targetless LiDAR-Camera Calibration with Neural Gaussian Splatting}},
  author  = {Jung, Haebeom and Kim, Namtae and Kim, Jungwoo and Park, Jaesik},
  journal = {IEEE Robotics and Automation Letters (RA-L)},
  year    = {2026}
}
```
