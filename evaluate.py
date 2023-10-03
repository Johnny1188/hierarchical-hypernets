import torch
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import time
import wandb

from utils.visualization import show_imgs, get_model_dot
from utils.timing import func_timer
from utils.metrics import get_metrics_path_key, get_metrics_task_key, get_accuracy, print_arch_summary

torch.set_printoptions(precision=3, linewidth=180)
wandb.login()


def eval_task(root_cell, task_i, task, context_name, config, path, loss_fn=F.cross_entropy):
    num_correct = 0
    loss_class = 0
    loss_theta_solver_reg = 0
    total_loss = 0
    b_i = 0

    for X, y, _ in task["test_loader"]:
        X = X.to(config["device"])
        _, y = torch.unique(y, return_inverse=True)
        y = y.to(config["device"])

        # generate theta and predict # TODO: not necessary to run the hypernetwork for each batch
        y_hat, theta_solver, path_trace = root_cell(X, context_name=context_name, task_i=task_i, path=path, theta_hnet=None, path_trace={"cond_ids": []})

        # task loss and accuracy
        curr_loss_class = loss_fn(y_hat, y)
        loss_class += curr_loss_class
        num_correct += (y_hat.argmax(dim=-1) == y).sum().item()
        
        # solvers' params regularization
        curr_loss_theta_solver_reg = \
            path_trace["solving_cell"].config["hnet"]["reg_alpha"] \
            * (sum([p.norm(p=2) for p in theta_solver]) / len(theta_solver))
        loss_theta_solver_reg += curr_loss_theta_solver_reg
        
        # total loss
        total_loss += curr_loss_class + curr_loss_theta_solver_reg
        
        b_i += 1

    acc = (num_correct / task["num_test_samples"]) * 100.
    loss_class = loss_class / b_i
    loss_theta_solver_reg = loss_theta_solver_reg / b_i
    total_loss = total_loss / b_i

    return {
        "loss_class": loss_class.item(),
        "loss_theta_solver_reg": loss_theta_solver_reg.item(),
        "total_loss": total_loss.item(),
        "acc": acc,
    }


@func_timer
def evaluate(root_cell, data_handlers, config, paths, loss_fn=F.cross_entropy, task_to_return=None):
    metrics = {}
    root_cell.toggle_mode(mode="eval")
    single_metrics_out = dict()
    
    with torch.no_grad():
        for p_i, path in enumerate(paths):
            m_key_p = get_metrics_path_key(p_i, path)
            metrics[m_key_p] = {}
            logging_task_i = 0
            for data_time_info, contexts in data_handlers.items():
                for context_idx, (context_name, tasks) in enumerate(contexts.items()):
                    for task_i, task in enumerate(tasks):
                        curr_task_metrics = eval_task(
                            root_cell=root_cell,
                            task_i=task_i,
                            task=task,
                            context_name=context_name,
                            config=config,
                            path=path,
                            loss_fn=loss_fn
                        )
                        
                        # add to metrics
                        # m_key_t = get_metrics_task_key(context_idx, task_i)
                        m_key_t = get_metrics_task_key(logging_task_i)
                        metrics[m_key_p][m_key_t] = curr_task_metrics
                        if task_to_return is not None \
                            and task_to_return[0] == data_time_info \
                            and task_to_return[1] == context_name \
                            and task_to_return[2] == task_i:
                            single_metrics_out = curr_task_metrics                        
                        logging_task_i += 1

    root_cell.toggle_mode(mode="train")
    
    return metrics, single_metrics_out
