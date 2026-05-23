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

import argparse
import json
import os
import re
import shlex
import subprocess
from pathlib import Path


DEFAULT_DATA_PATH = "/home/haebeom/data/TLC-Calib-Test"
DEFAULT_TRAIN_ARGS = [
    "--eval",
    "--from_lidar",
    "--use_rig",
    "--opt_pose",
    "--pose_scheduler",
    "--adaptive_voxel",
]
DATASET_ALIASES = {
    "kitti-360": "kitti-360",
    "fast-livo2": "fast-livo2",
    "waymo": "waymo",
}


def get_directories(path):
    try:
        return sorted(
            name for name in os.listdir(path)
            if os.path.isdir(os.path.join(path, name))
        )
    except FileNotFoundError:
        print(f"The path {path} does not exist.")
        return []


def normalize_dataset_name(dataset_name):
    return DATASET_ALIASES.get(dataset_name.lower(), dataset_name.lower())


def discover_scenes(data_path, datasets=None, scenes=None):
    data_path = Path(data_path)
    requested_datasets = {normalize_dataset_name(name) for name in datasets or []}
    requested_scenes = set(scenes or [])

    discovered = []
    for dataset_dir in sorted(path for path in data_path.iterdir() if path.is_dir()):
        dataset_key = normalize_dataset_name(dataset_dir.name)
        if requested_datasets and dataset_key not in requested_datasets and dataset_dir.name not in requested_datasets:
            continue

        for scene_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
            if requested_scenes and scene_dir.name not in requested_scenes:
                continue
            discovered.append((dataset_key, dataset_dir.name, scene_dir.name, scene_dir))

    return discovered


def run_command(command, cwd):
    printable = " ".join(shlex.quote(str(part)) for part in command)
    print(f"\n\033[1m[CMD]\033[0m {printable}")
    return subprocess.run(command, cwd=cwd, check=True).returncode


def make_run_name(repeat_idx, repeat_count):
    if repeat_count == 1:
        return "eval"
    return f"eval_{repeat_idx + 1:02d}"


def run_full_eval(args, train_extra_args):
    repo_root = Path(__file__).resolve().parent
    data_path = Path(args.data_path or DEFAULT_DATA_PATH)
    output_root = Path(args.output_path)
    if not output_root.is_absolute():
        output_root = repo_root / output_root

    scenes = discover_scenes(data_path, args.datasets, args.scenes)
    if not scenes:
        raise RuntimeError(f"No scenes found under {data_path}")

    for dataset_key, dataset_dir_name, scene_name, source_path in scenes:
        scene_output_root = output_root / dataset_key / scene_name
        print(f"\n\033[1;34m[{dataset_dir_name}/{scene_name}]\033[0m")

        for repeat_idx in range(args.repeat):
            run_name = make_run_name(repeat_idx, args.repeat)
            model_path = scene_output_root / run_name
            model_path.parent.mkdir(parents=True, exist_ok=True)

            train_command = [
                "python",
                "train.py",
                "-s",
                str(source_path),
                "-m",
                str(model_path),
                "--dataset",
                dataset_key,
                *DEFAULT_TRAIN_ARGS,
                *train_extra_args,
            ]
            run_command(train_command, cwd=repo_root)

            pose_metrics_command = [
                "python",
                "metrics_pose.py",
                "-m",
                str(model_path),
            ]
            run_command(pose_metrics_command, cwd=repo_root)

            nvs_metrics_command = [
                "python",
                "metrics_nvs.py",
                "-m",
                str(model_path),
                "--model_iter",
                str(args.model_iter),
            ]
            run_command(nvs_metrics_command, cwd=repo_root)

        aggregate_results(scene_output_root, args.model_iter)

    write_all_dataset_summaries(output_root)


def aggregate_pose_results(input_path, model_iter):
    directories = get_directories(input_path)
    print("Directories:", directories)

    valid_idx = 0
    result_dict = {}
    key = f"ours_{model_iter}"

    for directory in directories:
        rig_results_path = os.path.join(input_path, directory, "rig_results.json")
        if os.path.exists(rig_results_path):
            with open(rig_results_path, "r") as fp:
                data = json.load(fp)
        else:
            print(f"\033[91mNo rig results found for {directory}.\033[0m")
            continue

        if key not in data:
            print(f"\033[91mNo {key} results found for {directory}.\033[0m")
            continue

        valid_idx += 1
        for cam, errors in data[key].items():
            if cam not in result_dict:
                result_dict[cam] = {"rot": 0.0, "trans": 0.0}

            result_dict[cam]["rot"] += errors["Rot Err"]
            result_dict[cam]["trans"] += errors["Trans Err"]

    if valid_idx == 0:
        print(f"\033[91mNo valid rig results found under {input_path}.\033[0m")
        return {}

    for cam, errors in result_dict.items():
        result_dict[cam]["rot"] /= valid_idx
        result_dict[cam]["trans"] /= valid_idx

    if "Average" in result_dict:
        result_dict["Average"] = result_dict.pop("Average")

    for cam, errors in result_dict.items():
        print(f"Average errors for {cam}:", errors)

    with open(os.path.join(input_path, "exp_rig_results.json"), "w") as fp:
        json.dump(result_dict, fp, indent=True)

    return result_dict


def parse_memory_gb(value):
    match = re.match(r"([\d\.]+)", value)
    if match is None:
        raise ValueError(f"Could not parse memory value: {value}")
    return float(match.group(1))


def parse_train_time_seconds(value):
    h, m, s = map(int, value.split(":"))
    return h * 3600 + m * 60 + s


def aggregate_train_results(input_path):
    directories = get_directories(input_path)
    valid_idx = 0
    peak_mem, avg_mem, train_time = 0.0, 0.0, 0.0

    for directory in directories:
        train_info_path = os.path.join(input_path, directory, "train_info.json")
        if os.path.exists(train_info_path):
            with open(train_info_path, "r") as fp:
                data = json.load(fp)
        else:
            print(f"\033[91mNo train info found for {directory}.\033[0m")
            continue

        peak_mem += parse_memory_gb(data["peak_memory"])
        avg_mem += parse_memory_gb(data["avg_memory"])
        train_time += parse_train_time_seconds(data["train_time"])
        valid_idx += 1

    result_dict = {}
    if valid_idx > 0:
        result_dict["Peak Memory"] = peak_mem / valid_idx
        result_dict["Average Memory"] = avg_mem / valid_idx
        result_dict["Train Time"] = train_time / valid_idx
    else:
        print(f"\033[91mNo valid train info found under {input_path}.\033[0m")

    with open(os.path.join(input_path, "train_results.json"), "w") as fp:
        json.dump(result_dict, fp, indent=True)

    return result_dict


def load_nvs_metrics(run_path, model_iter):
    key = f"ours_{model_iter}"
    summary_path = os.path.join(run_path, "nvs_results.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as fp:
            data = json.load(fp)
        if key in data:
            return data[key]

    result_path = os.path.join(run_path, "nvs_eval", key, "test", key, "results.json")
    if os.path.exists(result_path):
        with open(result_path, "r") as fp:
            return json.load(fp)

    return None


def aggregate_nvs_results(input_path, model_iter):
    directories = get_directories(input_path)
    valid_idx = 0
    sums = {"PSNR": 0.0, "SSIM": 0.0, "LPIPS": 0.0}
    result_dict = {}

    for directory in directories:
        metrics = load_nvs_metrics(os.path.join(input_path, directory), model_iter)
        if metrics is None:
            print(f"\033[91mNo NVS results found for {directory}.\033[0m")
            continue

        result_dict[directory] = {metric: metrics[metric] for metric in sums}
        for metric in sums:
            sums[metric] += metrics[metric]
        valid_idx += 1

    if valid_idx > 0:
        result_dict["Average"] = {metric: value / valid_idx for metric, value in sums.items()}
        print(f"\033[1m\033[94mAverage NVS results: {result_dict['Average']}\033[0m\033[0m\n")
    else:
        result_dict["Average"] = {}
        print(f"\033[91mNo valid NVS results found under {input_path}.\033[0m")

    with open(os.path.join(input_path, "exp_nvs_results.json"), "w") as fp:
        json.dump(result_dict, fp, indent=True)

    return result_dict


def aggregate_results(input_path, model_iter):
    input_path = str(input_path)
    pose_results = aggregate_pose_results(input_path, model_iter)
    nvs_results = aggregate_nvs_results(input_path, model_iter)
    train_results = aggregate_train_results(input_path)
    return {"pose": pose_results, "nvs": nvs_results, "train": train_results}


def has_run_results(path):
    path = Path(path)
    return any((path / name).exists() for name in ["rig_results.json", "train_info.json", "nvs_results.json"]) or (path / "nvs_eval").exists()


def is_scene_output(path):
    path = Path(path)
    return path.is_dir() and any(child.is_dir() and has_run_results(child) for child in path.iterdir())


def is_dataset_output(path):
    path = Path(path)
    return path.is_dir() and any(child.is_dir() and is_scene_output(child) for child in path.iterdir())


def aggregate_dataset_output(dataset_path, model_iter):
    dataset_path = Path(dataset_path)
    for scene_dir in sorted(path for path in dataset_path.iterdir() if path.is_dir()):
        if is_scene_output(scene_dir):
            print(f"\n\033[1;34mAggregating scene: {scene_dir}\033[0m")
            aggregate_results(scene_dir, model_iter)


def aggregate_existing_outputs(aggregate_path, model_iter):
    aggregate_path = Path(aggregate_path)
    if not aggregate_path.exists():
        raise FileNotFoundError(f"Aggregate path does not exist: {aggregate_path}")

    if is_scene_output(aggregate_path):
        aggregate_results(aggregate_path, model_iter)
        write_dataset_summary(aggregate_path.parent)
        return

    if is_dataset_output(aggregate_path):
        aggregate_dataset_output(aggregate_path, model_iter)
        write_dataset_summary(aggregate_path)
        return

    dataset_dirs = [path for path in sorted(aggregate_path.iterdir()) if path.is_dir() and is_dataset_output(path)]
    if dataset_dirs:
        for dataset_dir in dataset_dirs:
            aggregate_dataset_output(dataset_dir, model_iter)
        write_all_dataset_summaries(aggregate_path)
        return

    raise RuntimeError(f"Could not find scene or dataset outputs under {aggregate_path}")


def collect_dataset_summary(dataset_dir):
    dataset_dir = Path(dataset_dir)
    summary = {}

    for scene_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        scene_summary = {}
        rig_path = scene_dir / "exp_rig_results.json"
        train_path = scene_dir / "train_results.json"
        nvs_path = scene_dir / "exp_nvs_results.json"

        if rig_path.exists():
            with open(rig_path, "r") as fp:
                scene_summary["pose"] = json.load(fp)
        if nvs_path.exists():
            with open(nvs_path, "r") as fp:
                scene_summary["nvs"] = json.load(fp)
        if train_path.exists():
            with open(train_path, "r") as fp:
                scene_summary["train"] = json.load(fp)

        if scene_summary:
            summary[scene_dir.name] = scene_summary

    return summary


def write_dataset_summary(dataset_dir):
    dataset_dir = Path(dataset_dir)
    summary = collect_dataset_summary(dataset_dir)
    with open(dataset_dir / "full_eval_results.json", "w") as fp:
        json.dump(summary, fp, indent=True)


def write_all_dataset_summaries(output_root):
    output_root = Path(output_root)
    for dataset_dir in sorted(path for path in output_root.iterdir() if path.is_dir() and is_dataset_output(path)):
        write_dataset_summary(dataset_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="TLC-Calib full evaluation script.")
    parser.add_argument(
        "--aggregate_path",
        "-a",
        type=str,
        help="Existing scene, dataset, or output root path to aggregate without training.",
    )
    parser.add_argument(
        "--data_path",
        "-d",
        type=str,
        default=None,
        help=f"Dataset root to evaluate. Defaults to {DEFAULT_DATA_PATH}.",
    )
    parser.add_argument("--output_path", "-o", type=str, default="./outputs", help="Root path for full-eval outputs.")
    parser.add_argument("--model_iter", "-n", type=int, default=30000, help="The model iteration to aggregate.")
    parser.add_argument("--datasets", nargs="+", help="Datasets to run, e.g. kitti-360 waymo fast-livo2.")
    parser.add_argument("--scenes", nargs="+", help="Scene names to run.")
    parser.add_argument("--repeat", type=int, default=1, help="Number of runs per scene.")
    args, train_extra_args = parser.parse_known_args()
    return args, train_extra_args


if __name__ == "__main__":
    args, train_extra_args = parse_args()

    if args.aggregate_path:
        aggregate_existing_outputs(args.aggregate_path, args.model_iter)
    else:
        run_full_eval(args, train_extra_args)
