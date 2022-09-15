import torch
from copy import deepcopy

from utils.configs import finish_arch_config
from configs.data_specs import get_data_specs
from configs.solver_specs import get_solver_specs


def get_arch_config(config):
    cell_config = {"hnet": config["hnet"], "solver": config["solver"], "device": config["device"]}
    get_c = lambda: deepcopy(cell_config)

    ### ability to have more root cells
    arch_config = [
        {
            **get_c(),
            "children": []
        }
    ]

    final_arch_config = [finish_arch_config(root_cell, root_level=True) for root_cell in arch_config]
    return final_arch_config


def get_config(cli_args=None):
    config = {
        "epochs": 50,
        "data": {
            # **get_data_specs("mnist|fmnist"),
            # **get_data_specs("splitmnist"),
            **get_data_specs("splitcifar10"),
            # **get_data_specs("splitcifar100"),
            "batch_size": 32,
            "data_dir": "data",
            "validation_size": 0,
        },
        "solver": {
            # "use": "lenet",
            # "specs": get_solver_specs("lenet", in_shape=[32, 32, 3], num_outputs=10),

            "use": "zenkenet",
            "specs": get_solver_specs("zenkenet", in_shape=[32, 32, 3], num_outputs=10),

            # "use": "resnet",
            # "specs": get_solver_specs("resnet", in_shape=[32, 32, 3], num_outputs=10),

            "task_heads": None,
            # "task_heads": [(0,10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60)],
        },
        "hnet": {
            "model": {
                "layers": [25, 25],
                "dropout_rate": -1, # hmlp doesn't get images -> need to be added to resnet
                "chunk_emb_size": 80,
                "chunk_size": 30_000,
                "num_cond_embs": None, # specified later
                "cond_in_size": 48,
                "cond_chunk_embs": False,
                "root_no_uncond_weights": False,
                "root_no_cond_weights": False,
                "children_no_uncond_weights": True,
                "children_no_cond_weights": False,
            },
            "lr": 0.0005,
            "reg_lr": 0.0005,
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
                "std_normal_init_chunk_embs": 1.0,
                "std_normal_init_task_embs": 1.0,
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

    config["solver"]["in_shape"] = config["data"]["in_shape"]
    config["hnet"]["model"]["num_cond_embs"] = config["data"]["num_tasks"] * 2

    return config
