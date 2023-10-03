import os
import torch

from utils.configs import finish_arch_config, get_cell_config
from configs.data_specs import get_data_specs
from configs.solver_specs import get_solver_specs


def get_arch_config(config):
    ### ability to have more root cells
    arch_config = [
        {
            **get_cell_config(config),
            "children": []
        }
    ]

    final_arch_config = [finish_arch_config(root_cell, root_level=True)[0] for root_cell in arch_config]
    return final_arch_config


def get_config(cli_args=None):
    config = {
        "epochs": 80,
        "use_early_stopping": False,
        "early_stopping": {
            "patience": 15,
            "min_delta": 0.01,
        },
        "data": {
            # "benchmark_specs_seen_before": get_data_specs(benchmarks=["splitcifar110"], in_shape=[32,32,3]),
            "benchmark_specs_seen_before": dict(),
            # "benchmark_specs_seen_now": get_data_specs(benchmarks=["splitmnist", "splitcifar110"], in_shape=[32,32,3]),
            "benchmark_specs_seen_now": get_data_specs(benchmarks=["splitcifar110"], in_shape=[32,32,3]),
            "batch_size": 256,
            "data_dir": "data" if os.environ.get("DATA_PATH") is None else os.path.join(os.environ.get("DATA_PATH"), "cl"),
            "validation_size": 0,
        },
        "solver": {
            "use": "zenkenet", # ["lenet", "zenkenet", "resnet"]
        },
        "hnet": {
            "model": {
                "layers": [100,150,200],
                # "layers": [80,80],
                "dropout_rate": -1, # hmlp doesn't get images -> need to be added to resnet
                "chunk_emb_size": 80,
                "chunk_size": 5500,
                "num_cond_embs": None, # specified later
                "cond_in_size": 48,
                "cond_chunk_embs": False,
                "root_no_uncond_weights": False,
                "root_no_cond_weights": False,
                "children_no_uncond_weights": True,
                "children_no_cond_weights": False,
                "act_func": torch.nn.LeakyReLU(negative_slope=0.05),
            },
            "lr": 0.0001,
            "reg_lr": 0.0001,
            "reg_alpha": 0, # L2 regularization of solvers' parameters
            "reg_beta": 0.01, # regularization against forgetting other contexts (tasks)
            "adam_beta_1": 0.5,
            "adam_beta_2": 0.999,
            "weight_decay": 0,
            "detach_d_theta": True,
            "reg_clip_grads_max_norm": None,
            "reg_clip_grads_max_value": None,
            "init": {
                "method": "xavier",
                "std_normal_init_params": 0.02,
                "std_normal_init_chunk_embs": 1.,
                "std_normal_init_task_embs": 1.,
            }
        },
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "wandb_logging": False,
    }

    ### Update config from cli args
    if cli_args is not None:
        if hasattr(cli_args, "wandb") and cli_args.wandb is True:
            config["wandb_logging"] = True
        if hasattr(cli_args, "epochs") and cli_args.epochs is not None:
            config["epochs"] = cli_args.epochs
        if hasattr(cli_args, "data") and cli_args.data is not None:
            config["data"].update(**get_data_specs(cli_args.data))
        if hasattr(cli_args, "solver") and cli_args.solver is not None:
            raise NotImplementedError("TODO - keep multihead from config if specified")
            assert cli_args.solver in ["lenet", "zenkenet", "resnet"], "Unknown solver"
            config["solver"]["use"] = cli_args.solver
            config["solver"]["specs"].update(get_solver_specs(cli_args.solver, in_shape=config["data"]["in_shape"], num_outputs=config["data"]["num_classes_per_task"]))
        if hasattr(cli_args, "multihead") and cli_args.data is not None:
            raise NotImplementedError

    config["data"]["num_tasks"] = sum([
        *[b["num_tasks"] for b in config["data"]["benchmark_specs_seen_before"]],
        *[b["num_tasks"] for b in config["data"]["benchmark_specs_seen_now"]]
    ])

    return config
