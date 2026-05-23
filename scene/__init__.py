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

import os
import random
import json
import torch
import shutil
import numpy as np
from utils.system_utils import search_for_max_iteration
from scene.dataset_readers import scene_load_type_callbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.pose_utils import make_transformation
from utils.camera_utils import camera_list_from_cam_infos, camera_to_json

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], ply_path=None):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = search_for_max_iteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration

            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = scene_load_type_callbacks["Colmap"](args.source_path, args.images, args.eval, args.lod)
        elif os.path.exists(os.path.join(args.source_path, "params")):
            print("Found 'params' folder, assuming Custom data set!")
            scene_info = scene_load_type_callbacks["Custom"](args.source_path, args.eval, args.dataset, args.voxel_size, args.llffhold, \
                                                          args.from_lidar, args.use_rig, args.adaptive_voxel, args.avc_beta, args.t_noise_bound, args.r_noise_bound, args.time_offset)
        else:
            assert False, "Could not recognize scene type!"

        # Save GT Rig
        if scene_info.rig_path is not None:
            self.gt_rigs = self.gaussians.load_gt_extrinsic(scene_info.rig_path) # cams_to_lidar_gt.txt
            shutil.copy2(scene_info.rig_path, os.path.join(self.model_path, scene_info.rig_path.split("/")[-1]))

        self.init_time = scene_info.time_offsets * (0.001)
        self.gaussians.gt_time_offset(-self.init_time)

        # Appearance embedding
        self.gaussians.set_appearance(len(scene_info.train_cameras))

        if not self.loaded_iter:
            if ply_path is not None:
                with open(ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                    dest_file.write(src_file.read())
            else:
                with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                    dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_json(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.cam_num = scene_info.cam_num
        self.opt_cam_num = scene_info.opt_cam_num

        for resolution_scale in resolution_scales:
            print("\nLoading Training Cameras")
            self.train_cameras[resolution_scale] = camera_list_from_cam_infos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = camera_list_from_cam_infos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            print("Pre-trained model loaded. Replace the previous gaussians.")
            self.gaussians.load_ply_sparse_gaussian(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            self.gaussians.load_mlp_checkpoints(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter)))
            self.gaussians.load_extrinsic(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "cams_to_lidar.txt"), args.dataset)
        else:
            self.init_pointcloud = scene_info.point_cloud
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.voxel_size, self.cameras_extent)

        self.train_info_dict = {"Initial point number": self.gaussians.get_point_num()}

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.train_info_dict.update({f"Point number at iteration {iteration}": self.gaussians.get_point_num()})
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_offset_ply(os.path.join(point_cloud_path, "offset.ply"))
        self.gaussians.save_mlp_checkpoints(point_cloud_path)
        self.gaussians.save_extrinsic(os.path.join(point_cloud_path, "cams_to_lidar.txt"))
        self.gaussians.save_trajectory(os.path.join(point_cloud_path), self.get_test_cameras().copy(), self.get_train_cameras().copy())

    def get_train_cameras(self, scale=1.0):
        return self.train_cameras[scale]

    def get_test_cameras(self, scale=1.0):
        return self.test_cameras[scale]

    def get_gt_rigs(self, idx=None):
        if idx == None:
            return self.gt_rigs
        else:
            return self.gt_rigs[idx]

    def get_gt_poses(self, type="train"):
        cam_infos = self.get_train_cameras() if type == "train" else self.get_test_cameras()
        return np.stack([make_transformation(R=cam_info.GT_cam_R, t=cam_info.GT_cam_t) for cam_info in cam_infos])

    def get_pred_poses(self, type="train"):
        cam_infos = self.get_train_cameras() if type == "train" else self.get_test_cameras()
        return np.stack([make_transformation(R=cam_info.cam_R.detach().cpu().numpy(), t=cam_info.cam_t.detach().cpu().numpy()) for cam_info in cam_infos])
