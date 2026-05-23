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

from numpy import identity
import torch

from utils.pose_utils import *
from scene.gaussian_model import GaussianModel


def load_trajectory(path):
    data = open(path, 'r').read()

    # Split the data by "Camera"
    cam_data = data.strip().split('# CAM')[1:]
    cam_num = len(cam_data)

    poses = []
    for idx in range(cam_num):
        lines = cam_data[idx].strip().split('\n')
        pose = np.array([list(map(float, row.split(' '))) for row in lines[1:]])
        poses.append(np.linalg.inv(pose))

    return poses

def update_pred_rig(camera, pc : GaussianModel):
    l2w = make_transformation(camera.lidar_R, camera.lidar_t)
    w2c = make_transformation(camera.init_cam_R.detach().cpu().numpy(), camera.init_cam_t.detach().cpu().numpy())

    extr = pc.get_delta_pose(camera.opt_id)
    new_w2c = extr @ w2c

    c2l = np.linalg.inv(l2w) @ np.linalg.inv(new_w2c)

    pc.update_pred_rig(camera.opt_id, c2l)

@torch.no_grad()
def update_pred_pose(camera, pc : GaussianModel, base=None):
    extr = torch.tensor(pc.get_delta_pose(camera.opt_id), dtype=torch.float32, device=camera.data_device) # 4x4

    if base is not None:
        T_w2c = base
    else:
        T_w2c = torch.eye(4, device=extr.device)
        T_w2c[:3, :3] = camera.init_cam_R
        T_w2c[:3, 3] = camera.init_cam_t

    new_w2c = extr @ T_w2c
    new_R = new_w2c[:3, :3]
    new_t = new_w2c[:3, 3]

    camera.update_Rt(new_R, new_t)

@torch.no_grad()
def update_delta_pose(camera, pc : GaussianModel, converged_threshold=1e-4):
    # Update delta pose
    geo_rot_delta = pc.cam_rot_deltas[camera.opt_id]
    geo_trans_delta = pc.cam_trans_deltas[camera.opt_id]

    tau_geo_step = torch.cat([geo_trans_delta, geo_rot_delta])
    pc.update_delta_pose(camera.opt_id, SE3_exp(tau_geo_step).detach().cpu().numpy())

    pc.cam_rot_deltas[camera.opt_id].data.fill_(0)
    pc.cam_trans_deltas[camera.opt_id].data.fill_(0)

    converged_score = torch.norm(tau_geo_step)
    return converged_score.cpu().numpy()
