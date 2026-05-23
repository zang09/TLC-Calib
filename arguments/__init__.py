#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# Modifications Copyright (C) 2026, SNU
# SNU VGI lab
# Modified for TLC-Calib: added calibration, pose optimization,
# and rasterizer-related configuration parameters.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact george.drettakis@inria.fr and haebeom.jung@snu.ac.kr
#

from argparse import ArgumentParser, Namespace
import sys
import os
import yaml

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 0
        self.llffhold = 2
        self.voxel_size = 0.1 # if voxel_size<=0, using 1nn dist
        self.feat_dim = 32
        self.n_offsets = 10
        self.update_depth = 3
        self.update_init_factor = 16
        self.update_hierachy_factor = 4

        self.use_feat_bank = False
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.dataset = "none" # [kitti-360, kitti, waymo, fast-livo2]
        self.eval = False
        self.lod = 0 # with image number

        self.adaptive_voxel = False
        self.avc_beta = 5000 # for adaptive voxel computation

        # Camera noise
        self.from_lidar = False
        self.use_rig = False
        self.time_offset = 0     # [ms]
        self.t_noise_bound = 0.0 # [cm]
        self.r_noise_bound = 0.0 # [deg]

        self.appearance_dim = 0 # 32
        self.lowpoly = False
        self.ds = 1
        self.ratio = 1 # sampling the input point cloud
        self.undistorted = False

        # In the Bungeenerf dataset, we propose to set the following three parameters to True,
        # Because there are enough dist variations.
        self.add_opacity_dist = False
        self.add_cov_dist = False
        self.add_color_dist = False

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.iterations = 30_000
        self.min_viewpoint_cycle = 5
        self.loss_type = "l1" # l1, l2, robust_l1, robust_l2

        ## Refine quality ##
        self.refine = False
        self.refine_iterations = 10_000
        ####################

        ####### Pose #######
        self.opt_pose = False
        self.pose_scheduler = False

        self.calib_rot_lr_init = 0.002
        self.calib_rot_lr_final = 0.0002
        self.calib_trans_lr_init = 0.005
        self.calib_trans_lr_final = 0.0005
        self.calib_lr_delay_mult = 0.01

        self.scene_weight_decay = 1e-2
        self.scene_decay_until = 15_000
        ####################

        self.position_lr_init = 0.0
        self.position_lr_final = 0.0
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000

        self.offset_lr_init = 0.01
        self.offset_lr_final = 0.0001
        self.offset_lr_delay_mult = 0.01
        self.offset_lr_max_steps = 30_000

        self.feature_lr = 0.0075
        self.opacity_lr = 0.02
        self.scaling_lr = 0.007
        self.rotation_lr = 0.002

        self.mlp_opacity_lr_init = 0.002
        self.mlp_opacity_lr_final = 0.00002
        self.mlp_opacity_lr_delay_mult = 0.01
        self.mlp_opacity_lr_max_steps = 30_000

        self.mlp_cov_lr_init = 0.004
        self.mlp_cov_lr_final = 0.004
        self.mlp_cov_lr_delay_mult = 0.01
        self.mlp_cov_lr_max_steps = 30_000

        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000

        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000

        self.mlp_featurebank_lr_init = 0.01
        self.mlp_featurebank_lr_final = 0.00001
        self.mlp_featurebank_lr_delay_mult = 0.01
        self.mlp_featurebank_lr_max_steps = 30_000

        self.appearance_lr_init = 0.05
        self.appearance_lr_final = 0.0005
        self.appearance_lr_delay_mult = 0.01
        self.appearance_lr_max_steps = 30_000

        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_scale = 1.0
        self.scale_regularizer = 10.0

        # for anchor densification
        self.start_stat = 500
        self.update_from = 1500
        self.update_interval = 100
        self.update_until = 15_000

        self.min_opacity = 0.005
        self.success_threshold = 0.8
        self.densify_grad_threshold = 0.0002

        super().__init__(parser, "Optimization Parameters", sentinel)

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    args_cmdline = parser.parse_args(cmdlne_string)
    cfg_dict = {}

    if args_cmdline.model_path is not None:
        config_path = os.path.join(args_cmdline.model_path, "config.yml")
        print("Looking for config file in", config_path)
        try:
            with open(config_path) as cfg_file:
                print("Config file found: {}".format(config_path))
                config = yaml.unsafe_load(cfg_file) or {}
            for section in config.values():
                if isinstance(section, dict):
                    cfg_dict.update(section)
                elif hasattr(section, "__dict__"):
                    cfg_dict.update(vars(section))
        except FileNotFoundError:
            print("Config file not found at", config_path)

    merged_dict = cfg_dict.copy()
    for k, v in vars(args_cmdline).items():
        if v is not None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
