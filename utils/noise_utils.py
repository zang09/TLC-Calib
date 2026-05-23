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


def _axis_angle_transform(axis, translation, angle_degrees):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    translation = np.asarray(translation, dtype=np.float64)

    angle = np.radians(angle_degrees)
    c, s = np.cos(angle), np.sin(angle)
    one_minus_c = 1.0 - c
    x, y, z = axis

    return np.array(
        [
            [one_minus_c * x * x + c, one_minus_c * x * y - s * z, one_minus_c * x * z + s * y, translation[0]],
            [one_minus_c * x * y + s * z, one_minus_c * y * y + c, one_minus_c * y * z - s * x, translation[1]],
            [one_minus_c * x * z - s * y, one_minus_c * y * z + s * x, one_minus_c * z * z + c, translation[2]],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _sample_unit_vector():
    vector = np.random.normal(size=3)
    return vector / np.linalg.norm(vector)


def make_each_cam_noise(cam_num, t_noise_bound=0.0, r_noise_bound=0.0, same=False):
    np.random.seed()

    noises = []
    for _ in range(cam_num):
        if same and noises:
            noises.append(noises[0].copy())
            continue

        if t_noise_bound > 0.0:
            translation = _sample_unit_vector() * t_noise_bound * np.random.rand()
        else:
            translation = np.zeros(3, dtype=np.float64)

        if r_noise_bound > 0.0:
            rotation_axis = _sample_unit_vector()
            rotation_degrees = r_noise_bound * np.random.rand()
        else:
            rotation_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            rotation_degrees = 0.0

        noises.append(_axis_angle_transform(rotation_axis, translation, rotation_degrees))

    return noises
