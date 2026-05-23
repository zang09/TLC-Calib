#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# Modifications Copyright (C) 2026, SNU
# SNU VGI lab
# Modified for TLC-Calib: added targetless LiDAR-camera
# calibration training, pose optimization.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact george.drettakis@inria.fr and haebeom.jung@snu.ac.kr
#

import warnings
warnings.filterwarnings("ignore")

import torch
import torchvision
import os
import sys
import numpy as np
import json
import yaml
import time
import datetime
import logging
from os import makedirs
import shutil, pathlib
from pathlib import Path
from random import randint
from utils.loss_utils import photometric_loss, ssim
from gaussian_renderer import prefilter_voxel, render
from scene import Scene, GaussianModel
from scene.poses import update_pred_rig, update_pred_pose, update_delta_pose
from scene.dataset_readers import read_custom_rigs
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from collections import defaultdict
from utils.image_utils import psnr
from utils.pose_utils import make_transformation, compute_ape_metrics
import utils.viser_utils as viser_utils
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, logger=None, viewer_enabled=False, viewer_port=8080, viewer_camera_step=3, ply_path=None):
    main_iterations = opt.iterations
    total_iterations = main_iterations + opt.refine_iterations if opt.refine else main_iterations
    opt.refine_start = main_iterations
    opt.refine_end = total_iterations
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, pipe, opt)
    gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank,
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
    scene = Scene(dataset, gaussians, ply_path=ply_path, shuffle=True)

    init_cam = scene.cam_num if scene.cam_num > scene.opt_cam_num else scene.opt_cam_num
    gaussians.init_cam(cam_num=init_cam)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    viewer = None
    if viewer_enabled:
        server = viser_utils.viser.ViserServer(port=viewer_port, verbose=False)
        viewer_handles = viser_utils.setup_viewer(server, scene, camera_frame_step=viewer_camera_step)
        viewer_renderer = viser_utils.ViewerRenderer(scene, pipe, dataset, viewer_handles, server)

        viewer = viser_utils.nerfview.Viewer(
            server=server,
            render_fn=viewer_renderer,
            mode="training",
        )
        # viewer.state.status = "paused"

        print(f"\n[viser] Viewer running... Go to http://localhost:{viewer_port} to connect.")
        time.sleep(7) # Wait for the viewer to be ready

    train_img_num = len(scene.get_train_cameras())
    print(f"\nNumber of CAMERAS for optimization: {scene.opt_cam_num}")
    print(f"Number of IMAGES for training: {train_img_num}")
    print(f"Checkpoint: {checkpoint}")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    warmup = True
    viewpoint_cycle = 0
    viewpoint_stack = None
    ema_loss_for_log, convergence_score = 0.0, 0.0
    are_errors, ate_errors = torch.tensor(0.0), torch.tensor(0.0)
    first_iter += 1

    # Reset CUDA memory
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    memory_usage_total = 0
    memory_usage_count = 0

    start_time = time.time()

    training_phases = [("Training progress", first_iter, main_iterations, False)]
    if opt.refine:
        refine_start = max(first_iter, opt.refine_start + 1)
        training_phases.append(("Refine progress", refine_start, opt.refine_end, True))

    for progress_desc, phase_start, phase_end, refine_phase in training_phases:
        if phase_start > phase_end:
            continue

        if refine_phase:
            print(f"\033[96m[ITER {phase_start}] Refine Start!\033[0m")
            opt.start_stat = opt.refine_start
            opt.update_from = opt.refine_start + 500
            opt.update_until = opt.refine_end - 1000
            opt.opt_pose = False

        progress_bar = tqdm(range(phase_start - 1, phase_end), desc=progress_desc)

        for iteration in range(phase_start, phase_end + 1):
            viewer_locked = False
            if viewer:
                while viewer.state.status == "paused":
                    time.sleep(0.01)
                viewer.lock.acquire()
                viewer_locked = True
                tic = time.time()
            try:
                iter_start.record()
                # if iteration % 2000 == 0: print(f"[ITER {iteration}] Debugging...")
                bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
                background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

                # Pick a random Camera
                if not viewpoint_stack:
                    viewpoint_stack = scene.get_train_cameras().copy()
                    viewpoint_cycle += 1
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

                # Update the previous pose
                with torch.no_grad():
                    all_cameras = scene.get_train_cameras().copy()
                    for cam in all_cameras:
                        if cam.cam_id == viewpoint_cam.cam_id:
                            update_pred_pose(cam, gaussians)
                    update_pred_rig(viewpoint_cam, gaussians)

                # Update LR
                gaussians.update_learning_rate(iteration)
                gaussians.update_pose_learning_rate(viewpoint_cam.opt_id, iteration)

                # Render
                if (iteration - 1) == debug_from:
                    pipe.debug = True

                voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)
                retain_grad = (iteration < opt.update_until and iteration >= 0)
                render_pkg = render(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad)

                image, viewspace_point_tensor, visibility_filter, offset_selection_mask, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["scaling"], render_pkg["neural_opacity"]

                # Loss
                gt_image = viewpoint_cam.original_image.cuda()

                photo_loss = photometric_loss(image, gt_image, type=opt.loss_type)
                ssim_loss = (1.0 - ssim(image, gt_image))

                if scaling.shape[0] > 0:
                    scaling_reg = (torch.clamp(scaling.max() / scaling.min() - opt.scale_regularizer, min=0).view(-1, 1).sum()) / scaling.shape[0]
                else:
                    scaling_reg = torch.tensor(0.0, device="cuda")

                loss = (1.0 - opt.lambda_dssim) * photo_loss + opt.lambda_dssim * ssim_loss + opt.lambda_scale * scaling_reg
                loss.backward()

                # Memory usage tracking
                current_memory = torch.cuda.memory_reserved()
                memory_usage_total += current_memory
                memory_usage_count += 1

                iter_end.record()

                # Wait for the events to complete
                torch.cuda.synchronize()

                with torch.no_grad():
                    # Progress bar
                    ema_loss_for_log = 0.4 * loss + 0.6 * ema_loss_for_log

                    if iteration % 10 == 0:
                        pred_poses, gt_poses = scene.get_pred_poses(), scene.get_gt_poses()
                        are_errors, ate_errors = compute_ape_metrics(pred_poses, gt_poses)
                        progress_bar.set_postfix({
                            "Loss": f"{ema_loss_for_log.item():.4f}",
                            # "Converge": f"{convergence_score:.4f}",
                            "are": f"{are_errors.mean().item():.4f}",
                            "ate": f"{ate_errors.mean().item():.4f}",
                            "N": f"{gaussians.get_point_num()}",
                        })
                        progress_bar.update(10)
                    if iteration == main_iterations:
                        peak_memory = torch.cuda.max_memory_reserved()
                        avg_memory = memory_usage_total / memory_usage_count if memory_usage_count > 0 else 0
                        scene.train_info_dict.update({
                            "peak_memory": f"{peak_memory / (1024 ** 3):.2f} GB",
                            "avg_memory": f"{avg_memory / (1024 ** 3):.2f} GB",
                        })
                    if iteration == phase_end:
                        progress_bar.close()
                        if not refine_phase:
                            end_time = time.time()

                    # Log and save
                    training_report(tb_writer, dataset_name, iteration, photo_loss, loss, are_errors.mean(), ate_errors.mean(), gaussians.get_point_num(), \
                                    iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), logger)
                    if (iteration in saving_iterations):
                        logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                        scene.save(iteration)

                    # Densification
                    if iteration < opt.update_until and iteration > opt.start_stat:
                        if gaussians.n_offsets > 0:
                            # add statis
                            gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)

                            # Densification
                            if iteration > opt.update_from and iteration % opt.update_interval == 0:
                                gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, \
                                                        grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity, refine=(opt.refine and iteration > opt.refine_start))
                    elif iteration == opt.update_until:
                        gaussians.reset_gradient()
                        torch.cuda.empty_cache()

                    # Optimize the scene
                    if iteration == opt.scene_decay_until:
                        gaussians.update_scene_decay(0.0)
                    if iteration < phase_end:
                        gaussians.optimizer.step()
                        gaussians.optimizer.zero_grad(set_to_none = True)

                    # Optimize the pose
                    if opt.opt_pose:
                        if viewpoint_cycle > opt.min_viewpoint_cycle:
                            if warmup:
                                warmup = False
                                print(f"\033[96m[ITER {iteration}] Pose Released!\033[0m")

                            gaussians.calib_optimizer[viewpoint_cam.opt_id].step()
                            gaussians.calib_optimizer[viewpoint_cam.opt_id].zero_grad(set_to_none=True)

                            convergence_score = float(update_delta_pose(viewpoint_cam, gaussians))
                        else:
                            gaussians.calib_optimizer[viewpoint_cam.opt_id].zero_grad(set_to_none=True)
                            gaussians.calib_optimizer[viewpoint_cam.opt_id].state = defaultdict(dict)

                    if viewer:
                        viewer.lock.release()
                        viewer_locked = False
                        num_train_rays_per_step = viewpoint_cam.image_height * viewpoint_cam.image_width
                        num_train_steps_per_sec = 1.0 / (time.time() - tic)
                        num_train_rays_per_sec = (
                            num_train_rays_per_step * num_train_steps_per_sec
                        )
                        # Update the viewer state.
                        viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                        # Update the scene.
                        viewer.update(iteration, num_train_rays_per_step)

                    # Save Checkpoints
                    if (iteration in checkpoint_iterations):
                        logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                        torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            finally:
                if viewer_locked:
                    viewer.lock.release()

    train_time = end_time - start_time
    scene.train_info_dict.update({"train_time": str(datetime.timedelta(seconds=round(train_time)))})

    return scene.train_info_dict, viewer


def training_report(tb_writer, dataset_name, iteration, photo_loss, loss, are, ate, pointN, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, logger=None):
    if tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train/photo_loss', photo_loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train/rot_error', are.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train/trans_error', ate.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/# of GS', pointN, iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        scene.gaussians.eval()
        torch.cuda.empty_cache()

        validation_configs = ({'name': 'test', 'cameras' : scene.get_test_cameras()},
                              {'name': 'train', 'cameras' : scene.get_train_cameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                test_count = 0
                rgb_test = 0.0
                psnr_test = 0.0

                for idx, viewpoint in enumerate(config['cameras']):
                    if idx % 5 != 0:
                        continue
                    if config['name'] == 'test':
                        update_pred_pose(viewpoint, scene.gaussians)

                    voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 30):
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    rgb_test += photometric_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    test_count += 1

                psnr_test /= test_count
                rgb_test /= test_count

                # Pose Evaluation
                trans_error = {}
                rot_error = {}
                cam_list = []
                for idx, viewpoint in enumerate(config['cameras']):
                    if config['name'] == 'test':
                        update_pred_pose(viewpoint, scene.gaussians)

                    cam_id = viewpoint.cam_id
                    if cam_id not in cam_list:
                        trans_error[cam_id] = 0.0
                        rot_error[cam_id] = 0.0
                        cam_list.append(cam_id)

                    pred_pose = make_transformation(R=viewpoint.cam_R.detach().cpu().numpy(), t=viewpoint.cam_t.detach().cpu().numpy())
                    gt_pose = make_transformation(R=viewpoint.GT_cam_R, t=viewpoint.GT_cam_t)
                    r_err, t_err = compute_ape_metrics(pred_pose, gt_pose)

                    trans_error[cam_id] += t_err.item()
                    rot_error[cam_id] += r_err.item()

                for cam_id in cam_list:
                    trans_error[cam_id] /= (len(config['cameras'])/len(cam_list))
                    rot_error[cam_id] /= (len(config['cameras'])/len(cam_list))

                mean_trans_error = sum(trans_error.values()) / len(cam_list)
                mean_rot_error = sum(rot_error.values()) / len(cam_list)

                logger.info("\n[ITER {}] Evaluating {}: Photo {:.6f} PSNR {:.6f}, Rot_Err: {:.6f}[deg], Trans_Err: {:.6f}[m]".format(
                    iteration, config['name'], rgb_test, psnr_test, mean_rot_error, mean_trans_error))
                for cam_id in sorted(cam_list):
                    logger.info(f"[CAM {cam_id}] Evaluating {config['name']}: Rot_Err: {rot_error[cam_id]:.6f}[deg], Trans_Err: {trans_error[cam_id]:.6f}[m]")

                if tb_writer:
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - photo_loss', rgb_test, iteration)
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - trans_error', mean_trans_error, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - rot_error', mean_rot_error, iteration)
                    for cam_id in cam_list:
                        tb_writer.add_scalar(config['name'] + f'/loss_viewpoint - trans_error({cam_id})', trans_error[cam_id], iteration)
                        tb_writer.add_scalar(config['name'] + f'/loss_viewpoint - rot_error({cam_id})', rot_error[cam_id], iteration)
        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/'+'total_points', scene.gaussians.get_anchor.shape[0], iteration)

        torch.cuda.empty_cache()
        scene.gaussians.train()


def read_cam_poses(cam_path):
    poses = {}
    with open(cam_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            cam_id = int(parts[0])
            uid = int(parts[1])
            pose = np.array(parts[2:], dtype=np.float32).reshape(4, 4)
            if cam_id not in poses:
                poses[cam_id] = []
            poses[cam_id].append(pose)
    return poses

def save_runtime_code(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = pathlib.Path(__file__).parent.resolve()


    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))

    print('Backup Finished!')

def ensure_model_path(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])


def prepare_output_and_logger(args, pipe, opt):
    ensure_model_path(args)

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    config = {
        'Model Params': args,
        'Pipeline Params': pipe,
        'Optimization Params': opt
    }
    with open(os.path.join(args.model_path, 'config.yml'), 'w') as f:
        yaml.dump(config, f)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def get_logger(path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO)
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)
    logger.propagate = False

    return logger

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--viewer', action='store_true', default=False)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--viewer_camera_step', type=int, default=3)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--backup', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    if args.refine:
        args.save_iterations = [iteration for iteration in args.save_iterations if iteration != args.iterations]
        if (args.iterations + args.refine_iterations) not in args.save_iterations:
            args.save_iterations.append(args.iterations + args.refine_iterations)
    elif args.iterations not in args.save_iterations:
        args.save_iterations.append(args.iterations)
    
    # enable logging
    ensure_model_path(args)
    os.makedirs(args.model_path, exist_ok=True)

    logger = get_logger(args.model_path)

    if args.backup:
        try:
            save_runtime_code(os.path.join(args.model_path, 'backup'))
        except:
            logger.info(f'Save Code Failed ..')

    dataset = args.source_path.split('/')[-1]
    logger.info("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # training
    train_info_dict, viewer = training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, logger, args.viewer, args.port, args.viewer_camera_step)

    print(f"\n[ INFO ] Training complete {train_info_dict['train_time']}")

    # Save training info
    with open(os.path.join(args.model_path, "train_info.json"), "w") as f:
        json.dump(train_info_dict, f, indent=4)

    # Maintain viewer after training
    if viewer is not None:
        print("Process complete. Viewer is still running. Ctrl+C to exit.")
        time.sleep(1000000)
