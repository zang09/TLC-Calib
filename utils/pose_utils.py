#
# Copyright (C) 2026, SNU
# SNU VGI lab
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact haebeom.jung@snu.ac.kr
#

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp


BASE_CAMERA_TO_LIDAR = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

DATASET_CAMERA_YAW_DEGREES = {
    "kitti-360": {2: 90.0, 3: -90.0},
    "kitti": {},
    "waymo": {1: 45.0, 2: -45.0, 3: 90.0, 4: -90.0},
    "fast-livo2": {},
}


def _interpolate_pose(extr0, extr1, t0, t1, t_query):
    rotations = Rotation.from_matrix([extr0[:3, :3], extr1[:3, :3]])
    rotation = Slerp([t0, t1], rotations)([t_query])[0]

    alpha = (t_query - t0) / (t1 - t0)
    translation = (1.0 - alpha) * extr0[:3, 3] + alpha * extr1[:3, 3]

    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation.as_matrix()
    pose[:3, 3] = translation
    return pose


def _extrapolate_pose(extr0, extr1, t0, t1, t_query):
    dt = t1 - t0
    if dt <= 0:
        return extr0.copy()

    rot0 = Rotation.from_matrix(extr0[:3, :3])
    rot1 = Rotation.from_matrix(extr1[:3, :3])
    angular_velocity = (rot0.inv() * rot1).as_rotvec() / dt
    linear_velocity = (extr1[:3, 3] - extr0[:3, 3]) / dt

    if t_query >= t1:
        delta_t = t_query - t1
        rotation = rot1 * Rotation.from_rotvec(angular_velocity * delta_t)
        translation = extr1[:3, 3] + linear_velocity * delta_t
    else:
        delta_t = t_query - t0
        rotation = rot0 * Rotation.from_rotvec(angular_velocity * delta_t)
        translation = extr0[:3, 3] + linear_velocity * delta_t

    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation.as_matrix()
    pose[:3, 3] = translation
    return pose


def apply_time_offset_with_timestamps(lidar_extrinsics, lidar_timestamps, time_offset):
    pose_count = len(lidar_timestamps)
    assert lidar_extrinsics.shape[0] == pose_count and pose_count >= 2

    timestamps = lidar_timestamps.astype(np.float64)
    target_timestamps = timestamps + time_offset
    poses = np.empty_like(lidar_extrinsics, dtype=np.float64)

    for i, target_time in enumerate(target_timestamps):
        left = np.searchsorted(timestamps, target_time, side="right") - 1
        if left < 0:
            poses[i] = _extrapolate_pose(
                lidar_extrinsics[0], lidar_extrinsics[1], timestamps[0], timestamps[1], target_time
            )
        elif left >= pose_count - 1:
            poses[i] = _extrapolate_pose(
                lidar_extrinsics[-2], lidar_extrinsics[-1], timestamps[-2], timestamps[-1], target_time
            )
        else:
            poses[i] = _interpolate_pose(
                lidar_extrinsics[left],
                lidar_extrinsics[left + 1],
                timestamps[left],
                timestamps[left + 1],
                target_time,
            )

    return poses


def apply_time_offset_uniform(lidar_extrinsics, time_offset, frame_rate):
    pose_count = lidar_extrinsics.shape[0]
    assert pose_count >= 2 and frame_rate > 0

    timestamps = np.arange(pose_count, dtype=np.float64) / frame_rate
    return apply_time_offset_with_timestamps(lidar_extrinsics, timestamps, time_offset)


def _yaw_transform(degrees):
    radians = np.deg2rad(degrees)
    cos, sin = np.cos(radians), np.sin(radians)
    return np.array(
        [
            [cos, -sin, 0.0, 0.0],
            [sin, cos, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def get_c2l(idx, dataset="kitti-360"):
    if dataset not in DATASET_CAMERA_YAW_DEGREES:
        raise ValueError(f"Unknown dataset: {dataset}")

    c2l = BASE_CAMERA_TO_LIDAR.copy()
    yaw_degrees = DATASET_CAMERA_YAW_DEGREES[dataset].get(idx)
    if yaw_degrees is not None:
        c2l = _yaw_transform(yaw_degrees) @ c2l
    return c2l


def _skew_symmetric(vector):
    x, y, z = vector
    matrix = torch.zeros(3, 3, device=vector.device, dtype=vector.dtype)
    matrix[0, 1] = -z
    matrix[0, 2] = y
    matrix[1, 0] = z
    matrix[1, 2] = -x
    matrix[2, 0] = -y
    matrix[2, 1] = x
    return matrix


def _SO3_exp(theta):
    omega = _skew_symmetric(theta)
    omega2 = omega @ omega
    angle = torch.norm(theta)
    identity = torch.eye(3, device=theta.device, dtype=theta.dtype)

    if angle < 1e-5:
        return identity + omega + 0.5 * omega2

    return (
        identity
        + (torch.sin(angle) / angle) * omega
        + ((1.0 - torch.cos(angle)) / (angle**2)) * omega2
    )


def _SO3_left_jacobian(theta):
    omega = _skew_symmetric(theta)
    omega2 = omega @ omega
    angle = torch.norm(theta)
    identity = torch.eye(3, device=theta.device, dtype=theta.dtype)

    if angle < 1e-5:
        return identity + 0.5 * omega + (1.0 / 6.0) * omega2

    return (
        identity
        + ((1.0 - torch.cos(angle)) / (angle**2)) * omega
        + ((angle - torch.sin(angle)) / (angle**3)) * omega2
    )


def SE3_exp(tau):
    rho = tau[:3]
    theta = tau[3:]

    transform = torch.eye(4, device=tau.device, dtype=tau.dtype)
    transform[:3, :3] = _SO3_exp(theta)
    transform[:3, 3] = _SO3_left_jacobian(theta) @ rho
    return transform


def make_transformation(R=np.eye(3), t=np.zeros(3), batch=False):
    if batch:
        rotations = np.asarray(R)
        translations = np.asarray(t).reshape(-1, 3)
        dtype = np.result_type(rotations, translations, np.float64)
        poses = np.zeros((len(rotations), 4, 4), dtype=dtype)
        poses[:, :3, :3] = rotations
        poses[:, :3, 3] = translations
        poses[:, 3, 3] = 1.0
        return poses

    dtype = np.result_type(R, t, np.float64)
    pose = np.eye(4, dtype=dtype)
    pose[:3, :3] = R
    pose[:3, 3] = t
    return pose


def _translation_errors(pred_poses, gt_poses):
    return torch.norm(pred_poses[:, :3, 3] - gt_poses[:, :3, 3], dim=1)


def _normalize_rotation_matrix(rotation):
    u, _, vt = torch.linalg.svd(rotation.double())
    rotation_ortho = u @ vt
    if torch.det(rotation_ortho) < 0:
        u[:, -1] *= -1
        rotation_ortho = u @ vt
    return rotation_ortho


def _rotation_errors(pred_poses, gt_poses):
    errors = []
    for pred_pose, gt_pose in zip(pred_poses, gt_poses):
        pred_rotation = _normalize_rotation_matrix(pred_pose[:3, :3])
        gt_rotation = _normalize_rotation_matrix(gt_pose[:3, :3])
        rotation_diff = pred_rotation @ gt_rotation.T
        cos_theta = torch.clamp((torch.trace(rotation_diff) - 1.0) / 2.0, -1.0, 1.0)
        errors.append(torch.rad2deg(torch.arccos(cos_theta)))
    return torch.stack(errors)


def compute_ape_metrics(pred_poses, gt_poses):
    pred_poses = torch.as_tensor(pred_poses)
    gt_poses = torch.as_tensor(gt_poses)

    if pred_poses.dim() == 2:
        pred_poses = pred_poses.unsqueeze(0)
    if gt_poses.dim() == 2:
        gt_poses = gt_poses.unsqueeze(0)

    return _rotation_errors(pred_poses, gt_poses), _translation_errors(pred_poses, gt_poses)
