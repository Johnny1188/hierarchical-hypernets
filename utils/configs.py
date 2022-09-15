import os
import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import time


def finish_arch_config(cell, root_level=False):
    cell["is_root"] = root_level
    cell["hnet"]["model"]["no_uncond_weights"] = not root_level # only root cells maintain
    cell["hnet"]["model"]["no_cond_weights"] = False # all maintain conditional weights
    cell["hnet"]["model"]["no_cond_weights"] = False # all maintain conditional weights
    cell["hnet"]["model"]["num_cond_embs"] = (len(cell["children"]) + 1) * cell["num_tasks"] # TODO - only considers children at one level below (not deeper)
    for child_i, child in enumerate(cell["children"]):
        cell["children"][child_i] = finish_arch_config(child, root_level=False)
    return cell
