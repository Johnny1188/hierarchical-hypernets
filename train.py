import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import gc
import time
import wandb
from hypnettorch.data import FashionMNISTData, MNISTData
from hypnettorch.data.special.split_mnist import get_split_mnist_handlers
from hypnettorch.data.special.split_cifar import get_split_cifar_handlers
from hypnettorch.mnets import LeNet, ZenkeNet, ResNet
from hypnettorch.hnets import HMLP, StructuredHMLP, ChunkedHMLP

from utils.data import get_mnist_data_loaders, get_emnist_data_loaders, randomize_targets, select_from_classes
from utils.visualization import show_imgs, get_model_dot
from utils.others import get_optimizer, measure_alloc_mem, count_parameters
from utils.timing import func_timer
from utils.metrics import get_accuracy, calc_accuracy
from utils.hypnettorch_utils import get_reg_loss
from utils.logging import print_metrics, log_wandb
from utils.cli_args import get_args
from evaluate import evaluate

torch.set_printoptions(precision=3, linewidth=180)


# def take_training_step(X, y, parent, child, phase, hnet_parent_prev_params, config, loss_fn=F.cross_entropy):
#     """
#     parent and child structure: tuple (hnet, solver, hnet_optimizer, hnet_cond_id)
#         child can be None if phase is "hnet->solver"
#     """
#     hnet_parent, solver_parent, hnet_parent_optim, hnet_parent_cond_id = parent
#     hnet_child, solver_child, hnet_child_optim, hnet_child_cond_id = child
#     for m in (hnet_parent, solver_parent, hnet_child, solver_child):
#         if m is not None:
#             m.train(mode=True)
    
#     hnet_parent_optim.zero_grad()
#     if hnet_child_optim is not None:
#         hnet_child_optim.zero_grad()
#     hnet_parent_optim.zero_grad()
#     if hnet_child_optim is not None:
#         hnet_child_optim.zero_grad()

#     # generate theta and predict
#     y_hat, params_solver = infer(X, phase, hnet_parent_cond_id=hnet_parent_cond_id, hnet_child_cond_id=hnet_child_cond_id,
#         hnet_parent=hnet_parent, hnet_child=hnet_child, solver_parent=solver_parent, solver_child=solver_child, config=config)
    
#     # task loss
#     loss_class = loss_fn(y_hat, y)
#     loss = loss_class
#     # solvers' params regularization
#     loss_solver_params_reg = torch.tensor(0., device=config["device"])
#     if config["hnet"]["reg_alpha"] is not None and config["hnet"]["reg_alpha"] > 0.:
#         loss_solver_params_reg = config["hnet"]["reg_alpha"] * sum([p.norm(p=2) for p in params_solver]) / len(params_solver)
#     loss += loss_solver_params_reg
#     perform_forgetting_reg = config["hnet"]["reg_beta"] is not None and config["hnet"]["reg_beta"] > 0.
#     loss.backward(retain_graph=perform_forgetting_reg, create_graph=not config["hnet"]["detach_d_theta"])
#     # gradient clipping
#     clip_grads([m for m in (hnet_parent, hnet_child) if m is not None], config["hnet"]["reg_clip_grads_max_norm"], config["hnet"]["reg_clip_grads_max_value"])
    
#     # regularization against forgetting other contexts
#     loss_reg = torch.tensor(0., device=config["device"])
#     if config["hnet"]["reg_beta"] is not None and config["hnet"]["reg_beta"] > 0.:
#         loss_reg = config["hnet"]["reg_beta"] * get_reg_loss(hnet_parent, hnet_parent_prev_params, curr_cond_id=hnet_parent_cond_id, lr=config["hnet"]["reg_lr"], detach_d_theta=config["hnet"]["detach_d_theta"])
#         loss_reg.backward()
#         # gradient clipping
#         clip_grads([m for m in (hnet_parent, hnet_child) if m is not None], config["hnet"]["reg_clip_grads_max_norm"], config["hnet"]["reg_clip_grads_max_value"])
    
#     hnet_parent_optim.step()
#     if hnet_child_optim is not None:
#         hnet_child_optim.step()
#     hnet_parent_optim.zero_grad()
#     if hnet_child_optim is not None:
#         hnet_child_optim.zero_grad()

#     return loss_class.detach().clone(), loss_solver_params_reg.detach().clone(), loss_reg.detach().clone(), y_hat.var(dim=0).detach().clone()


# def train_subtree(data_handlers, root_cell, config, loss_fn=F.cross_entropy):
#     torch.manual_seed(0)
#     np.random.seed(0)
#     hnet_root_prev_params = None
#     log_step = 0
#     phases = config["phases"]

#     for p_i, phase in enumerate(phases):
#         for d_i, data in enumerate(data_handlers):
#             # save parameters before solving the task for regularization against forgetting
#             hnet_root_prev_params = [p.detach().clone() for p_idx, p in enumerate(hnet_root.unconditional_params)]
#             for epoch in range(config["epochs"]):
#                 for i, (batch_size, X, y) in enumerate(data.train_iterator(config["data"]["batch_size"])):
#                     X = data.input_to_torch_tensor(X, config["device"], mode="train")
#                     y = data.output_to_torch_tensor(y, config["device"], mode="train")

#                     hnet_root_optim.zero_grad()
#                     hnet_child_optim.zero_grad()

#                     # select cond_ids for hypernets
#                     hnet_root_cond_id = hnets_cond_ids[d_i][phase]["hnet_root"]
#                     hnet_child_cond_id = hnets_cond_ids[d_i][phase]["hnet_child"]
#                     # generate theta and predict
#                     y_hat, params_solver = infer(X, phase, hnet_parent_cond_id=hnet_root_cond_id, hnet_child_cond_id=hnet_child_cond_id,
#                         hnet_parent=hnet_root, hnet_child=hnet_child, solver_parent=solver_root, solver_child=solver_child, config=config)
                    
#                     # solvers' params regularization
#                     loss_solver_params_reg = sum([p.norm(p=2) for p in params_solver]) / len(params_solver)
#                     # task loss
#                     loss_class = loss_fn(y_hat, y)
#                     loss = loss_class + config["hnet"]["reg_alpha"] * loss_solver_params_reg
#                     loss.backward(retain_graph=True, create_graph=not config["hnet"]["detach_d_theta"])
#                     # gradient clipping
#                     clip_grads([hnet_child, hnet_root], config["hnet"]["reg_clip_grads_max_norm"], config["hnet"]["reg_clip_grads_max_value"])
                    
#                     # regularization against forgetting other contexts
#                     loss_reg = config["hnet"]["reg_beta"] * get_reg_loss(hnet_root, hnet_root_prev_params, curr_cond_id=hnet_root_cond_id, lr=config["hnet"]["reg_lr"], detach_d_theta=config["hnet"]["detach_d_theta"])
#                     loss_reg.backward()
#                     clip_grads([hnet_child, hnet_root], config["hnet"]["reg_clip_grads_max_norm"], config["hnet"]["reg_clip_grads_max_value"])
                    
#                     hnet_root_optim.step()
#                     hnet_child_optim.step()
#                     hnet_root_optim.zero_grad()
#                     hnet_child_optim.zero_grad()

#                     if i % 100 == 99:
#                         acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean() * 100.
#                         print_metrics(
#                             datasets_for_eval, config=config, hnet_root=hnet_root, hnet_child=hnet_child, solver_root=solver_root, solver_child=solver_child,
#                             prefix=f"[{p_i + 1}:{phase} | {d_i}/{len(data_handlers) - 1} | {epoch + 1}/{config['epochs']} | {i + 1}]",
#                             skip_phases=[],
#                             wandb_run=wandb_run, additional_metrics={
#                                 "loss_class": loss_class.item(),
#                                 "acc_class": acc,
#                                 "loss_solver_params_reg": loss_solver_params_reg.item(),
#                                 "loss_reg": loss_reg.item(),
#                             }
#                         )
#                         print("---")
#                         log_step += 1

#     hnet, solver, hnet_optimizer, hnet_cond_id = root_cell
#     for m in (hnet, solver):
#         if m is not None:
#             m.train(mode=True)
    
#     hnet_optimizer.zero_grad()
#     hnet_optimizer.zero_grad()

#     # generate theta and predict
#     y_hat, params_solver = infer(X, "hnet->solver", hnet_parent_cond_id=hnet_cond_id, hnet_parent=hnet, solver_parent=solver, config=config)
    
#     # task loss
#     loss_class = loss_fn(y_hat, y)
#     loss = loss_class
#     # solvers' params regularization
#     loss_solver_params_reg = torch.tensor(0., device=config["device"])
#     if config["hnet"]["reg_alpha"] is not None and config["hnet"]["reg_alpha"] > 0.:
#         loss_solver_params_reg = config["hnet"]["reg_alpha"] * sum([p.norm(p=2) for p in params_solver]) / len(params_solver)
#     loss += loss_solver_params_reg
#     loss.backward()
#     # gradient clipping
#     clip_grads([m for m in (hnet, solver) if m is not None], config["hnet"]["reg_clip_grads_max_norm"], config["hnet"]["reg_clip_grads_max_value"])
    
#     hnet_optimizer.step()
#     hnet_optimizer.zero_grad()

#     return loss_class.detach().clone(), loss_solver_params_reg.detach().clone(), y_hat.var(dim=0).detach().clone()


@func_timer
def train_task(data_handlers, task_i, root_cell, config, paths, path_i, hnet_root_prev_params,
        root_hnet_cond_ids_already_trained, wandb_run=None, loss_fn=F.cross_entropy):
    task_data = data_handlers[task_i]
    path = paths[path_i]
    root_hnet_cond_ids_now_trained = set()
    
    root_cell.reinit_uncond_params_optim()
    root_cell.reinit_cond_embs_optims(path=path, task_i=task_i, reinit_all_children=False) # reinit only for cells on the path
    
    for epoch in range(config["epochs"]):
        for _, X, y in task_data.train_iterator(config["data"]["batch_size"]):
            X = task_data.input_to_torch_tensor(X, config["device"], mode="train")
            y = task_data.output_to_torch_tensor(y, config["device"], mode="train")
            if hasattr(task_data, "transform"): # additional transformation
                X = task_data.transform(X, batched=True)

            root_cell.zero_grad() # zero gradients of all cells in this root's tree

            # generate theta and predict
            y_hat, theta_solver, path_trace = root_cell(X, task_i=task_i, path=path, theta_hnet=None, path_trace={"cond_ids": []}, save_last_fc_acts=True)
            solving_cell = path_trace["solving_cell"]
            root_hnet_cond_ids_now_trained.add(path_trace["cond_ids"][0]) # add the root's hnet cond_id

            # task loss
            loss_class = loss_fn(y_hat, y)
            # solvers' params regularization
            loss_theta_solver_reg = solving_cell.config["hnet"]["reg_alpha"] * (sum([p.norm(p=2) for p in theta_solver]) / len(theta_solver))
            
            total_loss = loss_class + loss_theta_solver_reg
            apply_forgetting_reg = (task_i > 0 or path_i > 0) and root_cell.config["hnet"]["reg_beta"] is not None and root_cell.config["hnet"]["reg_beta"] > 0.
            total_loss.backward(retain_graph=apply_forgetting_reg, create_graph=not root_cell.config["hnet"]["detach_d_theta"])

            # clip gradients of all cells in this root's tree
            root_cell.clip_grads()
            
            # regularization against forgetting other contexts
            if apply_forgetting_reg is True:
                loss_reg = solving_cell.config["hnet"]["reg_beta"] * get_reg_loss(
                    hnet=root_cell.hnet,
                    hnet_prev_params=hnet_root_prev_params,
                    reg_cond_ids=root_hnet_cond_ids_already_trained,
                    lr=root_cell.config["hnet"]["reg_lr"],
                    detach_d_theta=root_cell.config["hnet"]["detach_d_theta"],
                    device=config["device"],
                )
                loss_reg.backward()
                # clip gradients of all cells in this root's tree
                root_cell.clip_grads()
            
            root_cell.step(path=path, task_i=task_i) # optimize only the current task emb (+ unconditional params)
            root_cell.zero_grad()

        ### evaluation & logging between epochs
        # get hidden activations of the solver
        solver_acts_mean, solver_acts_std = [], []
        if len(solving_cell.saved_solver_acts) > 0:
            solver_acts_mean = [layer_acts.mean(dim=0) for layer_acts in solving_cell.saved_solver_acts]
            solver_acts_std = [layer_acts.std(dim=0) for layer_acts in solving_cell.saved_solver_acts]
        # evaluate
        metrics = evaluate(root_cell=root_cell, data_handlers=data_handlers, config=config, paths=paths, loss_fn=loss_fn)
        metrics["loss_class"] = loss_class.item()
        metrics["acc_class"] = (y_hat.argmax(dim=1) == y.argmax(dim=-1)).float().mean().item() * 100.
        metrics["loss_theta_solver_reg"] = loss_theta_solver_reg.item()
        metrics["loss_reg"] = loss_reg.item() if apply_forgetting_reg is True else 0.
        metrics["y_hat_std"] = y_hat.std(dim=0).detach().clone().tolist()
        # logging
        print(f"[P{path_i + 1}/{len(paths)}-{''.join([str(i) for i in path])} | T{task_i + 1}/{len(data_handlers)} | E{epoch + 1}/{config['epochs']}]")
        print_metrics(metrics)
        print("---")        
        if wandb_run is not None:
            metrics["y_hat_std"] = wandb.Histogram(metrics["y_hat_std"])
            for l_i, (layer_acts_mean, layer_acts_std) in enumerate(zip(solver_acts_mean, solver_acts_std)):
                metrics[f"solver_acts_mean_l{l_i}"] = wandb.Histogram(layer_acts_mean.detach().clone().tolist())
                metrics[f"solver_acts_std_l{l_i}"] = wandb.Histogram(layer_acts_std.detach().clone().tolist())
            log_wandb(metrics, wandb_run)

    ### add the root's hnet cond_ids that were used during this task - used for regularization against forgetting
    root_hnet_cond_ids_already_trained.update(root_hnet_cond_ids_now_trained)
    return root_hnet_cond_ids_already_trained


@func_timer
def train(data_handlers, root_cell, config, paths, wandb_run=None, loss_fn=F.cross_entropy):
    """
    :param paths: list of lists specifying the paths through the hypernetwork hierarchy
        example: [[], [0], [1], [0,0]] (1. hnet -> solver; 2. hnet -0> hnet -> solver; 3. hnet -1> hnet -> solver; 4. hnet -0> hnet -0> hnet -> solver)
    """
    torch.manual_seed(0)
    np.random.seed(0)
    hnet_root_prev_params = None
    root_hnet_cond_ids_already_trained = set()
    root_cell.toggle_mode(mode="train")

    for p_i, path in enumerate(paths):
        # sequential training on tasks
        for task_i in range(len(data_handlers)):
            # save parameters before solving the task for regularization against forgetting
            hnet_root_prev_params = {"uncond_weights": [p.detach().clone() for p_idx, p in enumerate(root_cell.hnet.unconditional_params)]}
            
            root_hnet_cond_ids_already_trained = train_task(
                data_handlers=data_handlers,
                task_i=task_i,
                root_cell=root_cell,
                config=config,
                paths=paths,
                path_i=p_i,
                hnet_root_prev_params=hnet_root_prev_params,
                root_hnet_cond_ids_already_trained=root_hnet_cond_ids_already_trained,
                wandb_run=wandb_run,
                loss_fn=loss_fn,
            )

            print(f"[INFO] Training on [P{p_i + 1}/{len(paths)}-{''.join([str(i) for i in path])} | T{task_i + 1}/{len(data_handlers)}] has finished")
            metrics = evaluate(root_cell=root_cell, data_handlers=data_handlers, config=config, paths=paths, loss_fn=loss_fn)
            print_metrics(metrics)
            print("\n" * 3)
