#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# Modifications Copyright (C) 2026, SNU
# SNU VGI lab
# Modified for TLC-Calib: added LiDAR-camera pose state,
# intrinsic projection, timestamps, and mutable camera updates.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact george.drettakis@inria.fr and haebeom.jung@snu.ac.kr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import focal2fov, get_world_to_view_scaled_torch, get_projection_matrix_intrinsics


class Camera(nn.Module):
    def __init__(self, cam_id, uid, cam2lidar, lidar_R, lidar_t, GT_cam_R, GT_cam_t, cam_R, cam_t, fx, fy, cx, cy,
                 image, alpha_mask, image_name, timestamp,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, use_rig=False, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.opt_id = cam_id if use_rig else uid
        self.cam_id = cam_id
        self.uid = uid

        self.lidar_R = lidar_R
        self.lidar_t = lidar_t
        self.GT_cam_R = GT_cam_R
        self.GT_cam_t = GT_cam_t
        self.timestamp = timestamp

        # For update camera pose
        self.init_cam2lidar = torch.tensor(cam2lidar, device="cuda")
        self.init_cam_R = torch.tensor(cam_R, device="cuda")
        self.init_cam_t = torch.tensor(cam_t, device="cuda")
        self.cam_R = torch.tensor(cam_R, device="cuda")
        self.cam_t = torch.tensor(cam_t, device="cuda")

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        self.FoVy = focal2fov(self.fy, self.image_height)
        self.FoVx = focal2fov(self.fx, self.image_width)

        if alpha_mask is not None:
            self.original_image *= alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

    @property
    def projection_matrix(self):
        return get_projection_matrix_intrinsics(znear=self.znear, zfar=self.zfar, fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy, W=self.image_width, H=self.image_height).transpose(0,1).cuda()

    @property
    def world_view_transform(self):
        return get_world_to_view_scaled_torch(self.cam_R, self.cam_t).transpose(0, 1)

    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]

    def update_Rt(self, R, t):
        self.cam_R = R.cuda()
        self.cam_t = t.cuda()

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
