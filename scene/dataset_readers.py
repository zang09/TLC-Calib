#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# Modifications Copyright (C) 2026, SNU
# SNU VGI lab
# Modified for TLC-Calib: targetless LiDAR-camera calibration,
# including LiDAR/camera pose handling, time-offset modeling, and dataset loading.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact george.drettakis@inria.fr and haebeom.jung@snu.ac.kr
#

import os
import sys
from PIL import Image
from typing import NamedTuple, Optional
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3d_binary, read_points3d_text
from utils.graphics_utils import get_world_to_view_scaled, focal2fov
import numpy as np
from plyfile import PlyData, PlyElement
from utils.pose_utils import get_c2l
from utils.pose_utils import apply_time_offset_uniform, apply_time_offset_with_timestamps
from utils.noise_utils import make_each_cam_noise
from scene.gaussian_model import BasicPointCloud
import open3d as o3d

class CameraInfo(NamedTuple):
    cam_id: int
    uid: int
    cam2lidar: np.array
    lidar_R: np.array
    lidar_t: np.array
    GT_cam_R: np.array
    GT_cam_t: np.array
    cam_R: np.array
    cam_t: np.array
    fx: np.array
    fy: np.array
    cx: np.array
    cy: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    timestamp: float

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    cam_num: int
    opt_cam_num: int
    voxel_size: int
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    time_offsets: np.array
    lidar_poses: np.array
    lidar_timestamps: np.array
    ply_path: str
    rig_path: str

def get_traj_length(lidar_info):
    length = 0.0
    for i in range(1, len(lidar_info)):
        prev_t = lidar_info[i-1][:3, 3]
        curr_t = lidar_info[i][:3, 3]
        length += np.linalg.norm(curr_t - prev_t)
    return length

def get_nerfpp_norm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = get_world_to_view_scaled(cam.cam_R, cam.cam_t)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def count_voxels(data, voxel_size):
    down = data.voxel_down_sample(voxel_size)
    return np.asarray(down.points).shape[0]

def compute_voxel_size(pc=None, target_voxels=1000000, tolerance=0.05, max_iter=30):
    if len(pc.points) < target_voxels: return 0.1
    data = pc

    lo, hi = 0.1, 0.5
    mid = (lo + hi) / 2.0
    best_size, best_count = mid, count_voxels(data, mid)

    # binary search
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        n_vox = count_voxels(data, mid)

        if best_count is None or abs(n_vox - target_voxels) < abs(best_count - target_voxels):
            best_size, best_count = mid, n_vox

        if n_vox > target_voxels:
            lo = mid
        else:
            hi = mid

        if abs(n_vox - target_voxels) / target_voxels < tolerance:
            break
    return round(best_size, 2)

def voxelize_sample(pc=None, voxel_size=0.01):
    points = np.asarray(pc.points)
    min_bound = pc.get_min_bound() - voxel_size * 0.5
    max_bound = pc.get_max_bound() + voxel_size * 0.5
    _, trace, _ = pc.voxel_down_sample_and_trace(voxel_size, min_bound, max_bound, False)

    selected_indices = trace.max(axis=1)
    selected_indices = selected_indices[selected_indices >= 0]

    down = o3d.geometry.PointCloud()
    down.points = o3d.utility.Vector3dVector(points[selected_indices])

    colors = np.asarray(pc.colors)
    if len(colors) > 0:
        down.colors = o3d.utility.Vector3dVector(colors[selected_indices])

    normals = np.asarray(pc.normals)
    if len(normals) > 0:
        down.normals = o3d.utility.Vector3dVector(normals[selected_indices])

    return down

def apply_time_offset(lidar_extrinsics: np.ndarray, time_offset: float, lidar_timestamps: Optional[np.ndarray] = None, frame_rate: Optional[float] = None):
    if lidar_timestamps is not None:
        return apply_time_offset_with_timestamps(lidar_extrinsics, lidar_timestamps, time_offset)
    if frame_rate is None:
        raise ValueError("frame_rate is required if timestamps are not provided.")

    return apply_time_offset_uniform(lidar_extrinsics, time_offset, frame_rate)

def read_colmap_cameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.array(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        if intr.model=="SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def read_custom_cameras(dataset_type, cam_id, cam_extr, lidar_timestamps, lidar_extrinsics, lidar_extrinsics_offset, cam2lidar, cam_intrinsics, images_folder, from_lidar=False, use_rig=False, noises=None):
    """
    - from_lidar: If true, cameras' poses start from LiDAR poses
    - use_rig: If true, transform the camera poses together for cameras that have the same cam ID
    """
    img_prefix = os.listdir(images_folder)[0].split(".")[-1]

    cam_infos = []
    for idx, lidar_extr in enumerate(lidar_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(lidar_extrinsics)))
        sys.stdout.flush()

        timestamp = lidar_timestamps[idx]
        lidar_init = lidar_extr if lidar_extrinsics_offset is None else lidar_extrinsics_offset[idx]

        # Extrinsics
        ## GT_cam: w2c
        GT_cam = np.linalg.inv(lidar_extr @ cam2lidar)
        GT_cam_R = np.array(GT_cam[:3, :3])
        GT_cam_t = np.array(GT_cam[:3, 3])

        ## GT_lidar: l2w
        # GT_lidar_R = np.array(lidar_extr[:3, :3])
        # GT_lidar_t = np.array(lidar_extr[:3, 3])

        # From-LiDAR initialize
        if from_lidar:
            init_cam2lidar = get_c2l(cam_id, dataset_type)
            lc2w = lidar_init @ init_cam2lidar
            w2lc = np.linalg.inv(lc2w)
        # From-Blueprint initialize
        else:
            init_cam2lidar = cam2lidar
            lc2w = lidar_init @ init_cam2lidar
            w2lc = np.linalg.inv(lc2w)

            if noises is not None:
                noise_lc2w = lc2w @ noises[cam_id]
                w2lc = np.linalg.inv(noise_lc2w)

        uid = idx
        cam_R = np.array(w2lc[:3, :3])
        cam_t = np.array(w2lc[:3, 3])
        lidar_R = np.array(lidar_init[:3, :3])
        lidar_t = np.array(lidar_init[:3, 3])

        # Image Data
        image_path = os.path.join(images_folder, f"{idx:06d}.{img_prefix}")
        if not os.path.exists(image_path):
            return None
        image_name = os.path.basename(image_path).split(".")[0]
        image_fs = Image.open(image_path)
        image = image_fs.copy()
        image_fs.close()

        # Intrinsics
        width, height = image.size
        focal_length_x = cam_intrinsics[0][0]
        focal_length_y = cam_intrinsics[1][1]
        cx, cy = cam_intrinsics[0][2], cam_intrinsics[1][2]
        # FovY = focal2fov(focal_length_y, height)
        # FovX = focal2fov(focal_length_x, width)

        cam_info = CameraInfo(cam_id=cam_id, uid=uid, cam2lidar=init_cam2lidar.astype(np.float32), lidar_R=lidar_R, lidar_t=lidar_t, GT_cam_R=GT_cam_R, GT_cam_t=GT_cam_t, cam_R=cam_R, cam_t=cam_t, fx=focal_length_x, fy=focal_length_y, cx=cx, cy=cy,
                              image=image, image_path=image_path, image_name=image_name, width=width, height=height, timestamp=timestamp)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetch_ply(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    except:
        colors = np.random.rand(positions.shape[0], positions.shape[1])
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def store_ply(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def read_custom_intrinsics(path):
    intrinsics = {}
    current_cam = None

    with open(path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("# CAM"):
                current_cam = line.split(":")[0][2:].strip()
                intrinsics[int(current_cam[-1])] = []
            elif current_cam:
                intrinsics[int(current_cam[-1])].append([float(x) for x in line.split()])

    # Convert lists to matrices
    for cam in intrinsics:
        intrinsics[cam] = np.array([
            intrinsics[cam][0],
            intrinsics[cam][1],
            intrinsics[cam][2],
        ])
    return intrinsics

def read_custom_extrinsics(path):
    interval = 1
    poses = []
    with open(path, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if i % interval == 0:
                pose = np.array([float(x) for x in line.split()]).reshape(4, 4)
                poses.append(pose)
    poses = np.asarray(poses).reshape(-1, 4, 4)
    return poses

def read_custom_rigs(path):
    rigs = {}
    with open(path, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            cam_id = int(values[0])                         # First is ID
            rig = np.array(values[1:]).reshape(4, 4)        # Next 16 are 4x4 matrix
            rigs[cam_id] = rig
    return list(rigs.values())

def read_colmap_scene_info(path, images, eval, lod, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = read_colmap_cameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        if lod>0:
            print(f'using lod, using eval')
            if lod < 50:
                train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx > lod]
                test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx <= lod]
                print(f'test_cam_infos: {len(test_cam_infos)}')
            else:
                train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx <= lod]
                test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx > lod]

        else:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]

    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = get_nerfpp_norm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")

    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3d_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3d_text(txt_path)
        store_ply(ply_path, xyz, rgb)

    try:
        print(f'Start fetching data from ply file')
        pcd = fetch_ply(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def read_custom_scene_info(path, eval, dataset_type, voxel_size, llffhold=2, from_lidar=False, use_rig=False, adaptive_voxel=False, avc_beta=5000, t_noise=0.0, r_noise=0.0, time_offset=0):
    print(f"\n\033[92mREADING SCENE: {dataset_type.upper()}\033[0m")
    if dataset_type.upper() == "NONE":
        raise ValueError("Dataset type is not specified!")
        exit(1)

    if from_lidar:
        print("\n\033[92mCamera starts from correspondent LiDAR poses\033[0m")

    if use_rig:
        print("\n\033[92mUse rig optimization strategies\033[0m")

    params_dir = os.path.join(path, "params")
    check_file = sorted(os.listdir(params_dir))

    camera_extrinsic_files, cams_lidar_extrinsic_file = [], None
    cams2lidar, cam_intrinsics, lidar_extrinsics, lidar_timestamps = None, None, None, None
    for file in check_file:
        if "cams_to_lidar_gt" in file:
            cams_lidar_extrinsic_file = os.path.join(params_dir, file)
            cams2lidar = read_custom_rigs(cams_lidar_extrinsic_file) #c2l
        elif "lidars" in file:
            lidar_extrinsics = read_custom_extrinsics(os.path.join(params_dir, file))
        elif "cam" in file and not "lidar" in file:
            camera_extrinsic_files.append(os.path.join(params_dir, file))
        elif "intrinsics" in file:
            cam_intrinsics = read_custom_intrinsics(os.path.join(params_dir, file))
        elif "timestamp" in file:
            lidar_timestamps = np.loadtxt(os.path.join(params_dir, file))

    # Check valid cam number
    valid_cam_id = [i for i in range(len(cams2lidar))]

    cam_number = len(cams2lidar) # or len(cam_intrinsics)

    lidar_poses = lidar_extrinsics.copy()
    traj_length = get_traj_length(lidar_poses)

    # [n, 4, 4] -> [m, n, 4, 4]
    lidar_extrinsics = np.repeat(lidar_extrinsics[None, :, :, :], cam_number, axis=0)
    lidar_extrinsics_offset = np.array([None for _ in range(cam_number)])

    # Time offset configuration (for lidar init pose)
    frame_rate = 10.0 # lidar interval (Hz)
    time_offset_list = [0.0 for _ in range(cam_number)]

    if time_offset != 0:
        np.random.seed()
        time_offset_list = [time_offset * np.random.choice([-1, 1]) for _ in range(cam_number)]
        print(f"\n\033[92mTime offset: {time_offset_list} ms\033[0m")

        for cam_id in range(cam_number):
            if lidar_timestamps is not None:
                lidar_extrinsics_offset[cam_id] = apply_time_offset(lidar_extrinsics[cam_id], time_offset_list[cam_id] * 0.001, lidar_timestamps=lidar_timestamps)
            else:
                lidar_timestamps = np.arange(len(lidar_poses)) * (1.0 / frame_rate)
                lidar_extrinsics_offset[cam_id] = apply_time_offset(lidar_extrinsics[cam_id], time_offset_list[cam_id] * 0.001, frame_rate=frame_rate)
    else:
        if lidar_timestamps is None:
            lidar_timestamps = np.arange(len(lidar_poses)) * (1.0 / frame_rate)

    # From cams_to_lidar_gt.txt, get camera pose
    cam_extrinsics = {}
    for cam_id, c2l in enumerate(cams2lidar):
        cam_extrinsics[cam_id] = lidar_extrinsics[cam_id] @ c2l # c2w = l2w @ c2l

    # Noise configuration (cam noise)
    if use_rig and from_lidar: t_noise, r_noise = 0.0, 0.0
    if t_noise > 0.0 or r_noise > 0.0:
        print(f"\n\033[96mCamera noise added: t_noise={t_noise} m, r_noise={r_noise} deg\033[0m")
    noises = make_each_cam_noise(cam_num=cam_number, t_noise_bound=t_noise, r_noise_bound=r_noise)

    # Get camera infos
    cam_infos_dict = {}
    for cam_id, cam2lidar in enumerate(cams2lidar):
        if cam_id not in valid_cam_id: continue
        print(f"\n\033[92mReading camera {cam_id}\033[0m")
        reading_dir = os.path.join(path, "images", "image_%02d" % cam_id)

        cam_infos_unsorted = read_custom_cameras(dataset_type, cam_id, cam_extrinsics[cam_id], lidar_timestamps, lidar_extrinsics[cam_id], lidar_extrinsics_offset[cam_id], cam2lidar, cam_intrinsics[cam_id], \
                                               reading_dir, from_lidar=from_lidar, use_rig=use_rig, noises=noises)
        cam_infos_dict[cam_id] = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_path)

    train_cam_infos = []
    test_cam_infos = []
    for cam_id, cam_infos in cam_infos_dict.items():
        if eval:
            train_cam_infos += [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            test_cam_infos += [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
        else:
            train_cam_infos += cam_infos
            test_cam_infos = []

    # Re-id sequentially, per camera image to whole images
    for idx, _ in enumerate(train_cam_infos):
        train_cam_infos[idx] = train_cam_infos[idx]._replace(uid=idx)

    opt_cam_number = len(valid_cam_id) if use_rig else len(train_cam_infos)

    nerf_normalization = get_nerfpp_norm(train_cam_infos)

    pcd_path = os.path.join(path, "lidar/map.ply")
    if not os.path.exists(pcd_path):
        raise FileNotFoundError("Lidar point cloud not found!")
    else:
        pc = o3d.io.read_point_cloud(pcd_path)

    if adaptive_voxel:
        print(f"\nLiDAR trajectory length, Beta value: {traj_length:.2f}, {avc_beta}")
        voxel_size = compute_voxel_size(pc, target_voxels=int(traj_length*avc_beta))
        print(f"\033[92m\033[1mAdaptive voxel size: {voxel_size}\033[0m")

    ply_path = os.path.join(path, f"lidar/input_{voxel_size}.ply")
    if not os.path.exists(ply_path):
        print("Converting .pcd to .ply")
        pc = voxelize_sample(pc, voxel_size=voxel_size)

        xyz = np.asarray(pc.points)
        rgb = np.asarray(pc.colors)

        if len(rgb) == 0:
            rgb = np.random.rand(xyz.shape[0], 3) # [0~1]

        store_ply(ply_path, xyz, rgb * 255)

    try:
        pcd = fetch_ply(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           cam_num=cam_number,
                           opt_cam_num=opt_cam_number,
                           voxel_size=voxel_size,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           time_offsets=np.array(time_offset_list),
                           lidar_poses=lidar_poses,
                           lidar_timestamps=lidar_timestamps,
                           ply_path=ply_path,
                           rig_path=cams_lidar_extrinsic_file)
    return scene_info


scene_load_type_callbacks = {
    "Colmap": read_colmap_scene_info,
    "Custom": read_custom_scene_info,
}
