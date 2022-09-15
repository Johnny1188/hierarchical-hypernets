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
from utils.metrics import get_accuracy, calc_accuracy, print_arch_summary

torch.set_printoptions(precision=3, linewidth=180)
wandb.login()


def eval_task(root_cell, task_i, task_data, config, path, loss_fn=F.cross_entropy):
    num_correct = 0
    loss_class = 0
    loss_theta_solver_reg = 0
    total_loss = 0
    b_i = 0

    for batch_size, X, y in task_data.test_iterator(config["data"]["batch_size"]):
        X = task_data.input_to_torch_tensor(X, config["device"], mode="inference")
        y = task_data.output_to_torch_tensor(y, config["device"], mode="inference")

        # generate theta and predict # TODO: not necessary to run the hypernetwork for each batch
        y_hat, theta_solver, path_trace = root_cell(X, task_i=task_i, path=path, theta_hnet=None, path_trace={"cond_ids": []})

        # task loss and accuracy
        curr_loss_class = loss_fn(y_hat, y)
        loss_class += curr_loss_class
        num_correct += (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).sum().item()
        
        # solvers' params regularization
        curr_loss_theta_solver_reg = \
            path_trace["solving_cell"].config["hnet"]["reg_alpha"] \
            * (sum([p.norm(p=2) for p in theta_solver]) / len(theta_solver))
        loss_theta_solver_reg += curr_loss_theta_solver_reg
        
        # total loss
        total_loss += curr_loss_class + curr_loss_theta_solver_reg
        
        b_i += 1

    acc = (num_correct / task_data.num_test_samples) * 100.
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
def evaluate(root_cell, data_handlers, config, paths, loss_fn=F.cross_entropy):
    metrics = {}
    root_cell.toggle_mode(mode="eval")
    
    with torch.no_grad():
        for p_i, path in enumerate(paths):
            m_key_p = f"[P{p_i + 1}-{''.join([str(i) for i in path])}]"
            metrics[m_key_p] = {}
            for task_i, task_data in enumerate(data_handlers):
                curr_task_metrics = eval_task(
                    root_cell=root_cell,
                    task_i=task_i,
                    task_data=task_data,
                    config=config,
                    path=path,
                    loss_fn=loss_fn
                )
                
                # add to metrics
                m_key_t = f"[T{task_i + 1}]"
                metrics[m_key_p][m_key_t] = curr_task_metrics

    root_cell.toggle_mode(mode="train")
    
    return metrics
