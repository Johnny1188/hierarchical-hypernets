import os
import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import time


def get_cell_config(global_config):
    cell_config = {
        "hnet": global_config["hnet"],
        "solver": global_config["solver"],
        "device": global_config["device"],
        "num_tasks": global_config["data"]["num_tasks"]
    }
    return deepcopy(cell_config)


def finish_arch_config(cell, root_level=False):
    cell["is_root"] = root_level
    cell["hnet"]["model"]["no_uncond_weights"] = not root_level # only root cells maintain
    cell["hnet"]["model"]["no_cond_weights"] = False # all maintain conditional weights
    
    n_cells_visited_below = 0
    for child_i, child in enumerate(cell["children"]):
        cell["children"][child_i], child_n_cells_visited_below = finish_arch_config(child, root_level=False)
        n_cells_visited_below += child_n_cells_visited_below

    cell["hnet"]["model"]["num_cond_embs"] = (n_cells_visited_below + 1) * cell["num_tasks"] # +1 for the cell itself
    return cell, n_cells_visited_below + 1
