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

import json
import logging
import os
import shlex
import subprocess
from argparse import ArgumentParser
from pathlib import Path

from arguments import ModelParams, get_combined_args


DEFAULT_VOXEL_SIZE = 0.1
DEFAULT_LLFFHOLD = 2
DEFAULT_POSE_NAME = "cams_to_lidar.txt"
DEFAULT_TIME_NAME = "time_offset.txt"


def get_logger(path):
    logger = logging.getLogger("metrics_nvs")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO)
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)
    logger.propagate = False
    return logger


def run_command(command, cwd, logger):
    printable = " ".join(shlex.quote(str(part)) for part in command)
    logger.info(f"Command: {printable}")
    subprocess.run(command, cwd=cwd, check=True)


def result_key(iteration):
    return f"ours_{iteration}"


def find_prior_paths(model_path, model_iter, pose_name=DEFAULT_POSE_NAME, time_name=DEFAULT_TIME_NAME):
    iteration_dir = Path(model_path) / "point_cloud" / f"iteration_{model_iter}"
    pose_path = iteration_dir / pose_name
    time_path = iteration_dir / time_name

    if not pose_path.exists():
        raise FileNotFoundError(f"Pose path does not exist: {pose_path}")

    if not time_path.exists():
        time_path = None

    return pose_path, time_path


def evaluate(model_path, source_path, model_iter=30000, logger=None):
    model_path = Path(model_path).resolve()
    source_path = Path(source_path).resolve()
    nvs_root = model_path / "nvs_eval" / result_key(model_iter)
    nvs_eval_dir = Path(__file__).resolve().parent / "nvs_eval"

    if logger is None:
        logger = get_logger(model_path)

    pose_path, time_path = find_prior_paths(model_path, model_iter)
    result_path = nvs_root / "test" / result_key(model_iter) / "results.json"

    if result_path.exists():
        logger.info(f"NVS results already exist: {result_path}")
    else:
        command = [
            "python",
            "train.py",
            "-s",
            str(source_path),
            "-m",
            str(nvs_root),
            "--prior_pose_path",
            str(pose_path),
            "--prior_time_path",
            str(time_path) if time_path is not None else "None",
            "--eval",
            "--voxel",
            str(DEFAULT_VOXEL_SIZE),
            "--llffhold",
            str(DEFAULT_LLFFHOLD),
            "--iterations",
            str(model_iter),
        ]
        run_command(command, cwd=nvs_eval_dir, logger=logger)

    if not result_path.exists():
        raise FileNotFoundError(f"No NVS results found after evaluation: {result_path}")

    with open(result_path, "r") as fp:
        metrics = json.load(fp)

    summary = {
        result_key(model_iter): metrics,
        "source_path": str(source_path),
        "nvs_output_path": str(nvs_root),
        "prior_pose_path": str(pose_path),
        "prior_time_path": str(time_path) if time_path is not None else None,
    }

    save_path = model_path / "nvs_results.json"
    with open(save_path, "w") as fp:
        json.dump(summary, fp, indent=True)

    logger.info("  NVS PSNR : \033[1;35m{:>12.7f}\033[0m".format(metrics["PSNR"]))
    logger.info("  NVS SSIM : \033[1;35m{:>12.7f}\033[0m".format(metrics["SSIM"]))
    logger.info("  NVS LPIPS: \033[1;35m{:>12.7f}\033[0m".format(metrics["LPIPS"]))
    logger.info(f"NVS summary saved to {save_path}")

    return summary


if __name__ == "__main__":
    parser = ArgumentParser(description="NVS metrics script parameters")
    ModelParams(parser, sentinel=True)
    parser.add_argument("--model_iter", "-it", type=int, default=30000, help="Calibration iteration to use.")
    args = get_combined_args(parser)

    logger = get_logger(args.model_path)
    evaluate(args.model_path, args.source_path, model_iter=args.model_iter, logger=logger)
