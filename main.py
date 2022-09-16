import torch
import torch.nn.functional as F
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import wandb

from utils.visualization import get_summary_plots
from utils.data import get_data_handlers
from utils.timing import func_timer
from utils.metrics import print_arch_summary
from utils.logging import print_metrics
from utils.models import init_arch
from utils.cli_args import get_args
from train import train
from evaluate import evaluate

from configs.single import get_config as s_get_config, get_arch_config as s_get_arch_config
from configs.multiple import get_config as m_get_config, get_arch_config as m_get_arch_config
from configs.hypercl_zenke_splitcifar100 import get_config as zenke_cifar_get_config, get_arch_config as zenke_cifar_get_arch_config

torch.set_printoptions(precision=3, linewidth=180)
wandb.login()

LOGS_DIR = "logs"


def get_full_config(config_name, cli_args=None):
    config, arch_config = dict(), dict()
    if config_name == "single":
        config = s_get_config(cli_args=cli_args)
        arch_config = s_get_arch_config(config)
    elif config_name == "multiple":
        config = m_get_config(cli_args=cli_args)
        arch_config = m_get_arch_config(config)
    elif config_name == "zenke_cifar":
        config = zenke_cifar_get_config(cli_args=cli_args)
        arch_config = zenke_cifar_get_arch_config(config)
    else:
        raise ValueError(f"Unknown config name: {config_name}")
    return config, arch_config


@func_timer
def main(cli_args):
    ### prepare config for the run
    print(f"[INFO] Reading config")
    config, arch_config = get_full_config(
        config_name="multiple" if cli_args.config is None else cli_args.config,
        cli_args=cli_args
    )
    
    ### init architecture
    print(f"[INFO] Initializing architecture")
    [root_cell] = init_arch(arch_config, config)

    ### init logging
    wandb_run = None
    if config["wandb_logging"] is True:
        wandb_notes = "" if cli_args.description is None else cli_args.description
        wandb_run = wandb.init(
            project="Hypernets", entity="johnny1188",
            config={"config": config, "arch": arch_config},
            group=config["data"]["name"],
            tags=[], notes=wandb_notes
        )
        wandb.watch(root_cell, log="all", log_freq=100)
    
    print(f"[INFO] Running on {config['device']}")
    print_arch_summary(root_cell)
    
    ### prep data
    print(f"[INFO] Creating data handlers")
    data_handlers = get_data_handlers(config)
    
    ### save config and arch_config locally
    os.makedirs(LOGS_DIR, exist_ok=True)
    tm = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    with open(os.path.join(LOGS_DIR, f"config_{tm}.json"), "w") as f:
        json.dump({"config": config, "arch_config": arch_config}, f, default=str, indent=4)

    ### train all possible branches of the cells' tree
    paths = root_cell.get_available_paths([], [])
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

    ### generate summary plot
    fig, axes = get_summary_plots(metrics, with_baselines=True)
    plt.savefig(os.path.join(LOGS_DIR, f"summary_{tm}.png"))
    plt.show()
    if wandb_run is not None:
        wandb_run.log({"summary": wandb.Image(fig)})
    
    wandb_run.finish() if wandb_run is not None else None


if __name__ == "__main__":
    cli_args = get_args()
    main(cli_args=cli_args)
