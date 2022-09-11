import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
from copy import deepcopy
import time
import json
import os
from datetime import datetime
import wandb

from utils.data import get_data_handlers
from utils.visualization import show_imgs, get_model_dot
from utils.others import measure_alloc_mem, count_parameters
from utils.timing import func_timer
from utils.metrics import get_accuracy, calc_accuracy, print_arch_summary
from utils.logging import print_metrics
from utils.hypnettorch_utils import correct_param_shapes, calc_delta_theta, get_reg_loss_for_cond, get_reg_loss, \
    infer, print_stats, clip_grads, take_training_step, init_hnet_unconditionals, remove_hnet_uncondtionals, \
    validate_cells_training_inputs, train_cells
from utils.models import get_target_nets, get_hnets, create_tree, init_arch
from utils.cli_args import get_args
# from configs.single import get_config, get_arch_config
# from configs.multiple import get_config, get_arch_config
from configs.hypercl_zenke_splitcifar100 import get_config, get_arch_config
from train import train
from evaluate import evaluate

torch.set_printoptions(precision=3, linewidth=180)
wandb.login()


def main(cli_args=None):
    print(f"[INFO] Reading config")
    config = get_config(cli_args)
    arch_config = get_arch_config(config)
    
    print(f"[INFO] Initializing architecture")
    [root_cell] = init_arch(arch_config, config)
    
    print(f"[INFO] Creating data handlers")
    data_handlers = get_data_handlers(config)

    print(f"[INFO] Running on {config['device']}")
    wandb_run = None
    if config["wandb_logging"] is True:
        wandb_run = wandb.init(
            project="Hypernets", entity="johnny1188", config={"config": config, "arch": arch_config}, group=config["data"]["name"],
            tags=[], notes=f""
        )
        wandb.watch(root_cell, log="all", log_freq=100)
    
    ### save config and arch_config
    os.makedirs("logs", exist_ok=True)
    with open(os.path.join("logs", f"config_{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.json"), 'w') as f:
        json.dump({"config": config, "arch_config": arch_config}, f)

    ### train all possible branches of the cells' tree
    paths = root_cell.get_available_paths()
    print(f"[INFO] Training starts with the following paths:\n{paths}")
    train(
        data_handlers=data_handlers,
        root_cell=root_cell,
        config=config,
        paths=paths, # a single root cell
        wandb_run=wandb_run,
        loss_fn=F.cross_entropy,
    )

    ### final evaluation
    print(f"[INFO] Final evaluation")
    metrics = evaluate(root_cell=root_cell, data_handlers=data_handlers, config=config, paths=paths, loss_fn=F.cross_entropy)
    print_metrics(metrics)


if __name__ == "__main__":
    cli_args = get_args()
    main(cli_args=cli_args)
