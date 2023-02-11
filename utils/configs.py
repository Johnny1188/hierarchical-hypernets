import os
import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import time
from configs.solver_specs import get_solver_specs


def get_cell_config(global_config):
    cell_config = {
        "hnet": global_config["hnet"],
        "solver": global_config["solver"],
        "device": global_config["device"],
        "num_tasks": global_config["data"]["num_tasks"]
    }
    return deepcopy(cell_config)


def finish_config_from_data(config, data_handlers):
    assert len(data_handlers) > 0
    config["solver"]["task_heads"] = []
    start_idx = 0
    for context in data_handlers.values():
        for tasks in context.values():
            for task in tasks:
                config["solver"]["task_heads"].append((start_idx, start_idx + len(task["classes_in_experience"])))
                start_idx += len(task["classes_in_experience"])
            # config["solver"]["task_heads"].extend(get_task_separation_idxs(
            #     n_tasks=b["num_tasks"],
            #     n_classes_per_task=b["num_classes_per_task"],
            #     start_idx=start_idx
            # ))
            # start_idx = start_idx + b["num_tasks"] * b["num_classes_per_task"]
    in_shape = None
    for benchmark in [*config["data"]["benchmark_specs_seen_before"], *config["data"]["benchmark_specs_seen_now"]]:
        if in_shape is None:
            in_shape = benchmark["in_shape"]
        else:
            assert in_shape == benchmark["in_shape"], "All benchmarks must have the same input shape"
    config["solver"]["specs"] = get_solver_specs("zenkenet", in_shape=in_shape, num_outputs=config["solver"]["task_heads"][-1][-1])
    return config


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

def get_task_separation_idxs(n_tasks, n_classes_per_task, start_idx=0):
    sep_idxs = list(range(start_idx, start_idx + n_tasks * n_classes_per_task + 1, n_classes_per_task))
    return [(s, e) for s, e in zip(sep_idxs[:-1], sep_idxs[1:])]
