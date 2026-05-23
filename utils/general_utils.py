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

import torch
import sys
from datetime import datetime
import numpy as np
import random

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def pil_to_torch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def get_cosine_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Cosine annealing learning rate decay function.

    The learning rate starts at lr_init when step=0 and anneals to lr_final when step=max_steps,
    following a cosine curve. Optionally, the initial learning rate can be scaled by lr_delay_mult
    during the first lr_delay_steps.

    :param lr_init: float, initial learning rate.
    :param lr_final: float, final learning rate.
    :param lr_delay_steps: int, number of steps to delay the learning rate ramp-up.
    :param lr_delay_mult: float, multiplier for the learning rate during the delay period.
    :param max_steps: int, total number of steps during optimization.
    :return: function that takes step as input and returns the learning rate.
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable learning rate
            return 0.0

        if lr_delay_steps > 0:
            # Reverse cosine decay for the delay period
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0

        t = np.clip(step / max_steps, 0, 1)
        # Cosine annealing formula
        cosine_decay = 0.5 * (1 + np.cos(np.pi * t))
        lr = lr_final + (lr_init - lr_final) * cosine_decay

        return delay_rate * lr

    return helper

def get_cosine_annealing_warmup_restarts_lr_func(
    first_cycle_steps, cycle_mult=1.0, max_lr=0.1, min_lr=0.001, warmup_steps=0, gamma=1.0,
):
    """
    Cosine annealing learning rate function with warm restarts and warmup.

    The learning rate starts at min_lr, increases linearly to max_lr during warmup_steps,
    then decreases following a cosine curve back to min_lr over the course of the cycle.
    After each cycle, the cycle length is multiplied by cycle_mult and max_lr is multiplied by gamma.

    :param first_cycle_steps: int, number of steps in the first cycle.
    :param cycle_mult: float, factor to increase the cycle length after each cycle.
    :param max_lr: float, initial maximum learning rate.
    :param min_lr: float, minimum learning rate.
    :param warmup_steps: int, number of steps for linear warmup at the start of each cycle.
    :param gamma: float, multiplicative factor to decrease max_lr after each cycle.
    :return: function that takes step as input and returns the learning rate.
    """

    assert warmup_steps < first_cycle_steps, "warmup_steps must be less than first_cycle_steps"

    def helper(step):
        if step < 0:
            return min_lr

        # Initialize variables for the current cycle
        cycle = 0
        cur_cycle_steps = first_cycle_steps
        step_in_cycle = step

        # Determine the current cycle and step within the cycle
        while step_in_cycle >= cur_cycle_steps:
            step_in_cycle -= cur_cycle_steps
            cycle += 1
            cur_cycle_steps = int((cur_cycle_steps - warmup_steps) * cycle_mult) + warmup_steps

        # Update max_lr for the current cycle
        current_max_lr = max_lr * (gamma ** cycle)

        if step_in_cycle < warmup_steps:
            # Linear warmup phase
            lr = min_lr + (current_max_lr - min_lr) * step_in_cycle / warmup_steps
        else:
            # Cosine annealing phase
            cosine_steps = cur_cycle_steps - warmup_steps
            t = (step_in_cycle - warmup_steps) / cosine_steps
            lr = min_lr + 0.5 * (current_max_lr - min_lr) * (1 + np.cos(np.pi * t))

        return lr

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))