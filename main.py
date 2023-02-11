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
from utils.logging import print_metrics, init_run_logging
from utils.models import init_arch
from utils.cli_args import get_args
from utils.configs import finish_config_from_data
from train import train
from evaluate import evaluate

from configs.single import get_config as s_get_config, get_arch_config as s_get_arch_config
from configs.multiple import get_config as m_get_config, get_arch_config as m_get_arch_config
from configs.hypercl_zenke_splitcifar100 import get_config as zenke_cifar_get_config, get_arch_config as zenke_cifar_get_arch_config

torch.set_printoptions(precision=3, linewidth=180)
wandb.login()

ARTIFACTS_DIR = "artifacts"


def get_global_config(config_name, cli_args=None):
    config = dict()
    if config_name == "single":
        config = s_get_config(cli_args=cli_args)
    elif config_name == "multiple":
        config = m_get_config(cli_args=cli_args)
    elif config_name == "zenke_cifar":
        config = zenke_cifar_get_config(cli_args=cli_args)
    else:
        raise ValueError(f"Unknown config name: {config_name}")
    return config


def get_arch_configs(config_name, global_config):
    arch_config = dict()
    if config_name == "single":
        arch_config = s_get_arch_config(global_config)
    elif config_name == "multiple":
        arch_config = m_get_arch_config(global_config)
    elif config_name == "zenke_cifar":
        arch_config = zenke_cifar_get_arch_config(global_config)
    else:
        raise ValueError(f"Unknown config name: {config_name}")
    return arch_config


@func_timer
def main(cli_args):
    ### prepare config for the run
    print(f"[INFO] Reading config")
    global_config = get_global_config(
        config_name="single" if cli_args.config is None else cli_args.config,
        cli_args=cli_args
    )

    ### prep data
    print(f"[INFO] Creating data handlers")
    data_handlers = get_data_handlers(global_config)
    
    ### prepare architecutre config
    global_config = finish_config_from_data(global_config, data_handlers)
    arch_configs = get_arch_configs(
        config_name="single" if cli_args.config is None else cli_args.config,
        global_config=global_config
    )
    
    ### init architecture
    print(f"[INFO] Initializing architecture")
    [root_cell] = init_arch(arch_configs, global_config)
    root_hnet_cond_ids_trained, hnet_root_prev_params = None, None
    # load parameters if specified
    if cli_args.checkpoint_file is not None:
        if os.path.exists(cli_args.checkpoint_file) is False:
            raise ValueError(f"Checkpoint path does not exist: {cli_args.checkpoint_file}")
        print(f"[INFO] Loading checkpoint from: {cli_args.checkpoint_file}")
        checkpoint = root_cell.load_tree(curr_path=[], check_dict=None, path_to_checkpoint_file=cli_args.checkpoint_file)
        root_hnet_cond_ids_trained = checkpoint["root_hnet_cond_ids_trained"]
        hnet_root_prev_params = checkpoint["hnet_root_prev_params"]

    ### init logging
    run_name, path_to_run_dir, tm, wandb_run = init_run_logging(
        config=global_config,
        arch_configs=arch_configs,
        cli_args=cli_args,
        cells_to_watch=[root_cell],
        logs_dir=ARTIFACTS_DIR
    )

    print(f"[INFO] Running on {global_config['device']}")
    print_arch_summary(root_cell)

    ### train all possible branches of the cells' tree
    paths = root_cell.get_available_paths([], [])
    print(f"[INFO] Training starts with the following paths:\n{paths}")
    hnet_root_prev_params, root_hnet_cond_ids_trained = train(
        data_handlers=data_handlers,
        root_cell=root_cell,
        config=global_config,
        paths=paths, # a single root cell
        root_hnet_cond_ids_trained=root_hnet_cond_ids_trained,
        hnet_root_prev_params=hnet_root_prev_params,
        wandb_run=wandb_run,
        path_to_run_dir=path_to_run_dir,
        loss_fn=F.cross_entropy,
    )

    ### save the whole cells tree together with some additional training progress info
    root_cell.save_tree(
        curr_path=[],
        dict_to_save={
            "hnet_root_prev_params": hnet_root_prev_params,
            "root_hnet_cond_ids_trained": root_hnet_cond_ids_trained,
        },
        path_to_checkpoint_file=os.path.join(path_to_run_dir, "tree.tar"),
        is_root=True
    )

    ### final evaluation
    print(f"[INFO] Final evaluation of run {run_name}")
    metrics, _ = evaluate(
        root_cell=root_cell,
        data_handlers=data_handlers,
        config=global_config,
        paths=paths,
        loss_fn=F.cross_entropy
    )
    print_metrics(metrics)

    ### generate summary plot
    fig, axes = get_summary_plots(metrics, with_baselines=False)
    plt.savefig(os.path.join(path_to_run_dir, f"summary_plot.png"))
    plt.show()
    if wandb_run is not None:
        wandb_run.log({"summary": wandb.Image(fig)})
    
    wandb_run.finish() if wandb_run is not None else None


if __name__ == "__main__":
    cli_args = get_args()
    main(cli_args=cli_args)
