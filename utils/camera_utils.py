#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
from utils.general_utils import pil_to_torch

WARNED = False

def load_cam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        scale = resolution_scale * args.resolution
        resolution = round(orig_w / scale), round(orig_h / scale)
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = pil_to_torch(cam_info.image, resolution)
    gt_image = resized_image_rgb[:3, ...]

    loaded_mask = None
    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(cam_id=cam_info.cam_id, uid=cam_info.uid, cam2lidar=cam_info.cam2lidar, lidar_R=cam_info.lidar_R, lidar_t=cam_info.lidar_t, GT_cam_R=cam_info.GT_cam_R, GT_cam_t=cam_info.GT_cam_t, cam_R=cam_info.cam_R, cam_t=cam_info.cam_t,
                  fx=cam_info.fx/scale, fy=cam_info.fy/scale, cx=cam_info.cx/scale, cy=cam_info.cy/scale,
                  image=gt_image, alpha_mask=loaded_mask, timestamp=cam_info.timestamp,
                  image_name=cam_info.image_name, use_rig=args.use_rig, data_device=args.data_device)

def camera_list_from_cam_infos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(load_cam(args, id, c, resolution_scale))

    return camera_list

def camera_to_json(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.cam_R.transpose()
    Rt[:3, 3] = camera.cam_t
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : camera.fy,
        'fx' : camera.fx
    }
    return camera_entry

