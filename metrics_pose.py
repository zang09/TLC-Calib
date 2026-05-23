#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# Modifications Copyright (C) 2026, SNU
# SNU VGI lab
# Modified for TLC-Calib: added rig and camera pose error
# evaluation for LiDAR-camera calibration.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact george.drettakis@inria.fr and haebeom.jung@snu.ac.kr
#

import os
import json
import torch
import numpy as np

from pathlib import Path
from utils.pose_utils import compute_ape_metrics
from argparse import ArgumentParser
from arguments import ModelParams, get_combined_args
from scene.dataset_readers import read_custom_rigs

def read_cam_poses(cam_path):
    poses = {}
    with open(cam_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            cam_id = int(parts[0])
            pose = np.array(parts[2:], dtype=np.float32).reshape(4, 4)
            poses.setdefault(cam_id, []).append(pose)
    return poses


def evaluate(model_path, logger=None):
    if logger is None:
        logger = get_logger(model_path)
    pose_dict = {}
    print("")

    scene_dir = model_path
    pose_dict[scene_dir] = {}

    cloud_dir = Path(scene_dir) / "point_cloud"

    for method in os.listdir(cloud_dir):
        key = "ours_" + method.split("_")[-1]
        print("")
        logger.info(f"  Method: \033[1;35m{key}\033[0m")
        pose_dict[scene_dir][key] = {}

        pred_rig = np.array(read_custom_rigs(os.path.join(cloud_dir, method, 'cams_to_lidar.txt')))
        gt_rig = np.array(read_custom_rigs(os.path.join(scene_dir, 'cams_to_lidar_gt.txt')))

        if len(pred_rig) != len(gt_rig):
            print("\033[1;32m\nAveraging the poses with the camera id to make rig\033[0m")
            pred_pose = read_cam_poses(os.path.join(cloud_dir, method, 'pred_poses.txt'))
            gt_pose = read_cam_poses(os.path.join(cloud_dir, method, 'gt_poses.txt'))
            are_errors, ate_errors = [], []
            for cam_id in sorted(pred_pose.keys()):
                are_error, ate_error = compute_ape_metrics(np.linalg.inv(pred_pose[cam_id]), np.linalg.inv(gt_pose[cam_id]))
                are_errors.append(are_error.detach().cpu().numpy().mean())
                ate_errors.append(ate_error.detach().cpu().numpy().mean())
            are_errors, ate_errors = torch.tensor(are_errors), torch.tensor(ate_errors)
        else:
            print("\033[1;32m\nDirectly computing the rig error\033[0m")
            are_errors, ate_errors = compute_ape_metrics(np.linalg.inv(pred_rig), np.linalg.inv(gt_rig))

        for idx, (are_error, ate_error) in enumerate(zip(are_errors, ate_errors)):
            cam_id = f'CAM {idx:02}'
            pose_dict[scene_dir][key][cam_id] = {}

            logger.info("  [{}] Rot Err   : \033[1;35m{:>12.7f}\033[0m".format(cam_id, are_error, ".5"))
            logger.info("  [{}] Trans Err : \033[1;35m{:>12.7f}\033[0m".format(cam_id, ate_error, ".5"))
            print("")

            pose_dict[scene_dir][key][cam_id].update({"Rot Err": are_error.item(), "Trans Err": ate_error.item()})
        pose_dict[scene_dir][key]["Average"] = {"Rot Err": are_errors.mean().item(), "Trans Err": ate_errors.mean().item()}

        logger.info("  Avg Rot Err   : \033[1;35m{:>12.7f}\033[0m".format(are_errors.mean(), ".5"))
        logger.info("  Avg Trans Err : \033[1;35m{:>12.7f}\033[0m".format(ate_errors.mean(), ".5"))

    with open(scene_dir + "/rig_results.json", 'w') as fp:
        json.dump(pose_dict[scene_dir], fp, indent=True)

def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO)
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Metrics script parameters")
    ModelParams(parser, sentinel=True)
    args = get_combined_args(parser)

    logger = get_logger(args.model_path)

    evaluate(args.model_path, logger=logger)
