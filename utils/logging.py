import os
import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import gc
from IPython.display import clear_output
import time
from datetime import datetime
import wandb
import json
from hypnettorch.data import FashionMNISTData, MNISTData
from hypnettorch.data.special.split_mnist import get_split_mnist_handlers
from hypnettorch.data.special.split_cifar import get_split_cifar_handlers
from hypnettorch.mnets import LeNet, ZenkeNet, ResNet
from hypnettorch.hnets import HMLP, StructuredHMLP, ChunkedHMLP

from utils.data import get_mnist_data_loaders, get_emnist_data_loaders, randomize_targets, select_from_classes, get_data_handlers
from utils.visualization import show_imgs, get_model_dot
from utils.others import measure_alloc_mem, count_parameters
from utils.timing import func_timer
from utils.metrics import get_accuracy, calc_accuracy, print_arch_summary
from utils.hypnettorch_utils import correct_param_shapes, calc_delta_theta, get_reg_loss_for_cond, get_reg_loss, \
    infer, print_stats, print_metrics, clip_grads, take_training_step, init_hnet_unconditionals, remove_hnet_uncondtionals, \
    validate_cells_training_inputs, train_cells
from utils.models import get_target_nets, get_hnets

torch.set_printoptions(precision=3, linewidth=180)
wandb.login()


def init_run_logging(config, arch_config, cli_args, cells_to_watch=[], logs_dir="logs"):
    ### init wandb
    wandb_run = None
    if config["wandb_logging"] is True:
        wandb_notes = "" if cli_args.description is None else cli_args.description
        wandb_run = wandb.init(
            project="Hypernets", entity="johnny1188",
            config={"config": config, "arch": arch_config},
            group=config["data"]["name"],
            tags=[], notes=wandb_notes
        )
        for c in cells_to_watch:
            wandb.watch(c, log="all", log_freq=100)
    
    ### save config and arch_config locally
    tm = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    run_name = f"run_{tm}"
    path_to_run_dir = os.path.join(logs_dir, run_name)
    os.makedirs(path_to_run_dir, exist_ok=True)
    with open(os.path.join(path_to_run_dir, f"full_config.json"), "w") as f:
        json.dump({"config": config, "arch_config": arch_config}, f, default=str, indent=4)

    return run_name, path_to_run_dir, tm, wandb_run


def print_metrics(metrics, indent=0):
   for key, value in metrics.items():
      if isinstance(value, dict):
         print('  ' * indent + str(key))
         print_metrics(value, indent+1)
      else:
         print('  ' * indent, end='')
         if type(value) in (list, tuple) and len(value) > 0 and type(value[0]) in (float, int):
            print(f"{key:>25}\t{[round(n, 3) for n in value]}")
         else:
            print(f"{key:>25}\t{value}")


def log_wandb(metrics, wandb_run):
    if wandb_run is None:
        return
    wandb_run.log(metrics)


def log_eval(root, data_handlers, config, prefix, wandb_run, additional_metrics):
    # set the models to eval mode and return them to their original mode after
    cell_modes = []
    for hnet, solver in cells:
        cell_modes.append((hnet.training, solver.training))
        hnet.eval()
        solver.eval()
    wandb_metrics = {}
    
    print(prefix)
    with torch.no_grad():
        for task_i, task_data in data_handlers.items():
            print(f"[TASK {task_i + 1}/{len(data_handlers)}]")

            # prepare a test batch for calculating loss & getting solver params
            X = task_data.input_to_torch_tensor(task_data.get_test_inputs(), config["device"], mode="inference")
            y = task_data.output_to_torch_tensor(task_data.get_test_outputs(), config["device"], mode="inference")

            phase = "hnet->"
            wb_phase = "h->"
            wandb_metrics[str(task_i + 1)] = {}

            for c_i, cell in cells:
                hnet, solver = cell
                y_hat, params_solver = infer(X, config=config, cells=cells, cell_i=c_i)
                loss = F.cross_entropy(y_hat, y).item()
                acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean() * 100.
                print(f"    {phase}->solver Loss: {loss:.3f} | Accuracy: {acc:.3f}")
                wandb_metrics[str(task_i + 1)][f"{wb_phase}->s loss"] = loss
                wandb_metrics[str(task_i + 1)][f"{wb_phase}->s acc"] = acc
                phase += "hnet->"
                wb_phase += "h->"

    if additional_metrics:
        wandb_metrics.update(additional_metrics)
        for n, v in additional_metrics.items():
            print(f"{n}: {v:.3f}")

    if wandb_run is not None:
        wandb_run.log(wandb_metrics)
    
    for (hnet_training, solver_training), (hnet, solver) in zip(cell_modes, cells):
        hnet.train(mode=hnet_training)
        solver.train(mode=solver_training)
