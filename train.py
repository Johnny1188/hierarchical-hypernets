import os
import torch
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import wandb
from hypnettorch.data import FashionMNISTData, MNISTData
from hypnettorch.data.special.split_mnist import get_split_mnist_handlers
from hypnettorch.data.special.split_cifar import get_split_cifar_handlers

from utils.visualization import show_imgs, get_model_dot
from utils.others import get_optimizer, measure_alloc_mem, count_parameters, EarlyStopper
from utils.timing import func_timer
from utils.metrics import get_metrics_path_key, get_metrics_task_key
from utils.hypnettorch_utils import get_reg_loss
from utils.logging import print_metrics, log_wandb
from evaluate import evaluate

torch.set_printoptions(precision=3, linewidth=180)


@func_timer
def train_task(data_handlers, context_name, context_idx, task_i, root_cell, config, paths, path_i, hnet_root_prev_params,
        root_hnet_cond_ids_already_trained, wandb_run=None, loss_fn=F.cross_entropy):
    task_data = data_handlers["new"][context_name][task_i]
    path = paths[path_i]
    root_hnet_cond_ids_now_trained = set()
    
    root_cell.reinit_uncond_params_optim()
    root_cell.reinit_cond_embs_optims(path=path, context_name=context_name, task_i=task_i, reinit_all_children=False) # reinit only for cells on the path
    early_stopper = None
    if config["use_early_stopping"] is True and "early_stopping" in config.keys():
        early_stopper = EarlyStopper(patience=config["early_stopping"]["patience"], min_delta=config["early_stopping"]["min_delta"])

    for epoch in range(config["epochs"]):
        for X, y, _ in task_data["train_loader"]:
            X = X.to(config["device"])
            _, y = torch.unique(y, return_inverse=True)
            y = y.to(config["device"])

            root_cell.zero_grad() # zero gradients of all cells in this root's tree

            # generate theta and predict
            y_hat, theta_solver, path_trace = root_cell(
                X,
                context_name=context_name,
                task_i=task_i,
                path=path,
                theta_hnet=None,
                path_trace={"cond_ids": []},
                save_last_fc_acts=True
            )
            solving_cell = path_trace["solving_cell"]
            root_hnet_cond_ids_now_trained.add(path_trace["cond_ids"][0]) # add the root's hnet cond_id

            # task loss
            loss_class = loss_fn(y_hat, y)
            # solvers' params regularization
            loss_theta_solver_reg = solving_cell.config["hnet"]["reg_alpha"] * (sum([p.norm(p=2) for p in theta_solver]) / len(theta_solver))
            
            total_loss = loss_class + loss_theta_solver_reg
            apply_forgetting_reg = (len(root_hnet_cond_ids_already_trained) > 0) and root_cell.config["hnet"]["reg_beta"] is not None and root_cell.config["hnet"]["reg_beta"] > 0.
            total_loss.backward(retain_graph=apply_forgetting_reg, create_graph=not root_cell.config["hnet"]["detach_d_theta"])

            # clip gradients of all cells in this root's tree
            root_cell.clip_grads()
            
            # regularization against forgetting other contexts
            if apply_forgetting_reg is True:
                loss_reg = solving_cell.config["hnet"]["reg_beta"] * get_reg_loss(
                    hnet=root_cell.hnet,
                    hnet_prev_params=hnet_root_prev_params,
                    theta_targets=None, # TODO
                    reg_cond_ids=root_hnet_cond_ids_already_trained,
                    lr=root_cell.config["hnet"]["reg_lr"],
                    detach_d_theta=root_cell.config["hnet"]["detach_d_theta"],
                    device=config["device"],
                )
                loss_reg.backward()
                # clip gradients of all cells in this root's tree
                root_cell.clip_grads()
            
            root_cell.step(path=path, context_name=context_name, task_i=task_i, skip_my_hnet_theta=False) # optimize only the current task emb (+ unconditional params)
            root_cell.zero_grad()

        ### evaluation & logging between epochs
        # get hidden activations of the solver
        solver_acts_mean, solver_acts_std = [], []
        if len(solving_cell.saved_solver_acts) > 0:
            solver_acts_mean = [layer_acts.mean(dim=0) for layer_acts in solving_cell.saved_solver_acts]
            solver_acts_std = [layer_acts.std(dim=0) for layer_acts in solving_cell.saved_solver_acts]
        # evaluate
        metrics, curr_task_metrics = evaluate(
            root_cell=root_cell,
            data_handlers=data_handlers,
            config=config,
            paths=paths,
            loss_fn=loss_fn,
            task_to_return=("new", context_name, task_i),
        )
        metrics["loss_class"] = loss_class.item()
        metrics["acc_class"] = (y_hat.argmax(dim=1) == y).float().mean().item() * 100.
        metrics["loss_theta_solver_reg"] = loss_theta_solver_reg.item()
        metrics["loss_reg"] = loss_reg.item() if apply_forgetting_reg is True else 0.
        metrics["y_hat_std"] = y_hat.std(dim=0).detach().clone().tolist()
        # logging
        # print(f"[P{path_i + 1}/{len(paths)}-{''.join([str(i) for i in path])} | T{task_i + 1}/{len(data_handlers)} | E{epoch + 1}/{config['epochs']}]")
        print(f"[P{path_i + 1}/{len(paths)}-{''.join([str(i) for i in path])} | {context_name} | T{task_i + 1} | E{epoch + 1}/{config['epochs']}]")
        print_metrics(metrics)
        print("---")        
        if wandb_run is not None:
            metrics["y_hat_std"] = wandb.Histogram(metrics["y_hat_std"])
            for l_i, (layer_acts_mean, layer_acts_std) in enumerate(zip(solver_acts_mean, solver_acts_std)):
                metrics[f"solver_acts_mean_l{l_i}"] = wandb.Histogram(layer_acts_mean.detach().clone().tolist())
                metrics[f"solver_acts_std_l{l_i}"] = wandb.Histogram(layer_acts_std.detach().clone().tolist())
            log_wandb(metrics, wandb_run)
        
        # early stopping (using test/val data)
        if early_stopper is not None and early_stopper.early_stop(loss=curr_task_metrics["total_loss"]):
            break

    ### add the root's hnet cond_ids that were used during this task - used for regularization against forgetting
    return root_hnet_cond_ids_now_trained


@func_timer
def train(data_handlers, root_cell, config, paths, root_hnet_cond_ids_trained=None, hnet_root_prev_params=None,
    wandb_run=None, path_to_run_dir=None, loss_fn=F.cross_entropy):
    """
    :param paths: list of lists specifying the paths through the hypernetwork hierarchy
        example: [[], [0], [1], [0,0]] (1. hnet -> solver; 2. hnet -0> hnet -> solver; 3. hnet -1> hnet -> solver; 4. hnet -0> hnet -0> hnet -> solver)
    """
    torch.manual_seed(0)
    np.random.seed(0)
    if root_hnet_cond_ids_trained is None:
        root_hnet_cond_ids_trained = set()
    # if root_cell_history is None:
    #     root_cell_history = {
    #         tuple(p):[
    #             {
    #                 "task_idx": task_idx,
    #                 "task_name": task["name"],
    #                 "cond_ids": set(),
    #             } for task_idx, task in enumerate(data_handlers)
    #         ] for p in paths
    #     }
    #     # root_cell_history = {tuple(p): dict() for p in paths}
    #     # for data_time_info, contexts in data_handlers.items():
    #     #         for context_idx, (context_name, tasks) in enumerate(contexts.items()):
    #     #             for task_i, task in enumerate(tasks):
    # else:
    #     # get the root hypernet's cond_ids that were already trained
    #     for path, path_tasks in root_cell_history.values():
    #         for task in path_tasks:
    #             root_hnet_cond_ids_trained.update(task["cond_ids"])
    root_cell.toggle_mode(mode="train")

    for p_i, path in enumerate(paths):
        # sequential training on tasks
        for context_idx, (context_name, tasks) in enumerate(data_handlers["new"].items()):
            for task_i, task in enumerate(tasks):
            # for task_i in range(len(data_handlers)):
                root_hnet_cond_ids_now_trained = train_task(
                    data_handlers=data_handlers,
                    context_name=context_name,
                    context_idx=context_idx,
                    task_i=task_i,
                    root_cell=root_cell,
                    config=config,
                    paths=paths,
                    path_i=p_i,
                    hnet_root_prev_params=hnet_root_prev_params,
                    root_hnet_cond_ids_already_trained=root_hnet_cond_ids_trained,
                    wandb_run=wandb_run,
                    loss_fn=loss_fn,
                )

                # update training history            
                # root_cell_history[tuple(path)][task_i]["cond_ids"].update(root_hnet_cond_ids_now_trained)
                root_hnet_cond_ids_trained.update(root_hnet_cond_ids_now_trained)

                # save parameters for regularization against forgetting (doesn't apply to the first task)
                hnet_root_prev_params = {"uncond_weights": [p.detach().clone() for p in root_cell.hnet.unconditional_params]}

                print(f"[INFO] Training on [P{p_i + 1}/{len(paths)}-{''.join([str(i) for i in path])} | T{task_i + 1}/{len(data_handlers)}] has finished")
                metrics, _ = evaluate(root_cell=root_cell, data_handlers=data_handlers, config=config, paths=paths, loss_fn=loss_fn)
                print_metrics(metrics)
                print("\n" * 3)

                ### save a checkpoint together with some additional training progress info
                if path_to_run_dir is not None:
                    print(f"[INFO] Saving a checkpoint to {path_to_run_dir}")
                    root_cell.save_tree(
                        curr_path=[],
                        dict_to_save={
                            "hnet_root_prev_params": hnet_root_prev_params,
                            "root_hnet_cond_ids_trained": root_hnet_cond_ids_trained,
                        },
                        path_to_checkpoint_file=os.path.join(path_to_run_dir, "tree.tar"),
                        is_root=True
                    )
    return hnet_root_prev_params, root_hnet_cond_ids_trained
