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
import wandb
from hypnettorch.data import FashionMNISTData, MNISTData
from hypnettorch.data.special.split_mnist import get_split_mnist_handlers
from hypnettorch.data.special.split_cifar import get_split_cifar_handlers
from hypnettorch.mnets import LeNet, ZenkeNet, ResNet
from hypnettorch.hnets import HMLP, StructuredHMLP, ChunkedHMLP

from utils.data import get_mnist_data_loaders, get_emnist_data_loaders, randomize_targets, select_from_classes
from utils.visualization import show_imgs, get_model_dot
from utils.others import measure_alloc_mem, count_parameters
from utils.timing import func_timer
from utils.metrics import get_accuracy, calc_accuracy

from IPython.display import clear_output

torch.set_printoptions(precision=3, linewidth=180)


def correct_param_shapes(solver, params, config):
    """Correct the shapes of the parameters for the solver"""
    params_solver = []
    src_param_i = 0
    src_param_start_idx = 0

    for target_param_i, p_shape in enumerate(solver.param_shapes):
        curr_available_src_params = params[src_param_i].flatten()[src_param_start_idx:].numel()
        if curr_available_src_params >= math.prod(p_shape):
            params_solver.append(params[src_param_i].flatten()[src_param_start_idx:src_param_start_idx + math.prod(p_shape)].view(p_shape))
            src_param_start_idx += math.prod(p_shape)
        else:
            new_param = torch.zeros(math.prod(p_shape), device=config["device"])
            s = 0

            while math.prod(p_shape) > s:
                curr_available_src_params = params[src_param_i].flatten().numel()
                to_add = params[src_param_i].flatten()[src_param_start_idx:min(curr_available_src_params, src_param_start_idx + (math.prod(p_shape) - s))]
                new_param[s:s + to_add.numel()] = to_add
                s += to_add.numel()

                if s < math.prod(p_shape):
                    src_param_i += 1
                    src_param_start_idx = 0
                else:
                    src_param_start_idx += to_add.numel()

            params_solver.append(new_param.view(p_shape))
    return params_solver


def calc_delta_theta(hnet, lr, detach=False):
    ret = []
    for p in hnet.internal_params:
        if p.grad is None:
            ret.append(None)
            continue
        if detach:
            ret.append(-lr * p.grad.detach().clone())
        else:
            ret.append(-lr * p.grad.clone())
    return ret


def get_reg_loss_for_cond(hnet, hnet_prev_params, lr, reg_cond_id, detach_d_theta=False):
    # prepare targets (theta for child nets predicted by previous hnet)
    hnet_mode = hnet.training
    hnet.eval()
    with torch.no_grad():
        theta_child_target = hnet(cond_id=reg_cond_id, weights={"uncond_weights": hnet_prev_params} if hnet_prev_params is not None else None)
    # detaching target below is important!
    theta_child_target = torch.cat([p.detach().clone().view(-1) for p in theta_child_target])
    hnet.train(mode=hnet_mode)
    
    d_theta = calc_delta_theta(hnet, lr, detach=detach_d_theta)
    theta_parent_for_pred = []
    for _theta, _d_theta in zip(hnet.internal_params, d_theta):
        if _d_theta is None:
            theta_parent_for_pred.append(_theta)
        else:
            theta_parent_for_pred.append(_theta + _d_theta if detach_d_theta is False else _theta + _d_theta.detach())
    theta_child_predicted = hnet(cond_id=reg_cond_id, weights=theta_parent_for_pred)
    theta_child_predicted = torch.cat([p.view(-1) for p in theta_child_predicted])

    return (theta_child_target - theta_child_predicted).pow(2).sum()


def get_reg_loss(hnet, hnet_prev_params, curr_cond_id, lr=1e-3, clip_grads_max_norm=1., detach_d_theta=False):
    reg_loss = 0
    for c_i in range(hnet._num_cond_embs):
        if curr_cond_id is not None and c_i == curr_cond_id:
            continue
        reg_loss += get_reg_loss_for_cond(hnet, hnet_prev_params, lr, c_i, detach_d_theta)
    return reg_loss / (hnet._num_cond_embs - (curr_cond_id is not None))


def infer(X, scenario, hnet_parent_cond_id, hnet_child_cond_id, hnet_parent, hnet_child, solver_parent, solver_child, config):
    assert scenario != "hnet->hnet->solver" or hnet_child_cond_id is not None, f"Scenario {scenario} requires hnet_child_cond_id to be set"
    
    if scenario == "hnet->solver":
        params_solver = hnet_parent.forward(cond_id=hnet_parent_cond_id) # parent hnet -> theta parent solver
        y_hat = solver_parent.forward(X, weights=correct_param_shapes(solver_parent, params_solver, config))
    elif scenario == "hnet->hnet->solver":
        params_hnet_child = hnet_parent.forward(cond_id=hnet_parent_cond_id) # parent hnet -> theta child hnet (only the unconditional ones) -> solver child
        params_solver = hnet_child.forward(cond_id=hnet_child_cond_id, weights=params_hnet_child)
        y_hat = solver_child.forward(X, weights=params_solver)
    else:
        raise ValueError(f"Unknown inference scenario {scenario}")
    return y_hat, params_solver


def print_metrics(datasets : dict, config, hnet_root, hnet_child, solver_root, solver_child, prefix="", skip_phases=[], wandb_run=None, additional_metrics=None):
    # set the models to eval mode and return them to their original mode after
    ms_modes = []
    for m in [hnet_root, hnet_child, solver_root, solver_child]:
        ms_modes.append([m, m.training])
        m.eval()
    wandb_metrics = {}
    
    print(prefix)
    with torch.no_grad():
        for data_name, (hnet_root_cond_id_hnet_solver, hnet_root_cond_id_hnet_hnet_solver, hnet_child_cond_id, dataset) in datasets.items():
            print(data_name)

            # prepare a test batch for calculating loss & getting solver params
            X = dataset.input_to_torch_tensor(dataset.get_test_inputs(), config["device"], mode="inference")
            y = dataset.output_to_torch_tensor(dataset.get_test_outputs(), config["device"], mode="inference")

            hnet_solver_loss, hnet_solver_acc, hnet_hnet_solver_loss, hnet_hnet_solver_acc = np.nan, np.nan, np.nan, np.nan
            if "hnet->solver" not in skip_phases:
                print("    hnet->solver")
                y_hat, params_solver = infer(X, "hnet->solver", hnet_parent_cond_id=hnet_root_cond_id_hnet_solver, hnet_child_cond_id=None, hnet_parent=hnet_root, hnet_child=hnet_child, solver_parent=solver_root, solver_child=solver_child, config=config)
                hnet_solver_loss = F.cross_entropy(y_hat, y).item()
                hnet_solver_acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean() * 100.
                print(f"        Loss: {hnet_solver_loss:.3f} | Accuracy: {hnet_solver_acc:.3f}")
            
            if "hnet->hnet->solver" not in skip_phases:
                print("    hnet->hnet->solver")
                y_hat, params_solver = infer(X, "hnet->hnet->solver", hnet_parent_cond_id=hnet_root_cond_id_hnet_hnet_solver, hnet_child_cond_id=hnet_child_cond_id, hnet_parent=hnet_root, hnet_child=hnet_child, solver_parent=solver_root, solver_child=solver_child, config=config)
                hnet_hnet_solver_loss = F.cross_entropy(y_hat, y).item()
                hnet_hnet_solver_acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean() * 100.
                print(f"        Loss: {hnet_hnet_solver_loss:.3f} | Accuracy: {hnet_hnet_solver_acc:.3f}")
            
            wandb_metrics[str(data_name)] = {
                "h->s loss": hnet_solver_loss,
                "h->s acc": hnet_solver_acc,
                "h->h->s loss": hnet_hnet_solver_loss,
                "h->h->s acc": hnet_hnet_solver_acc,
            }
    
    if additional_metrics:
        wandb_metrics.update(additional_metrics)
        for n, v in additional_metrics.items():
            print(f"{n}: {v:.3f}")

    if wandb_run is not None:
        wandb_run.log(wandb_metrics)
    
    for m, mode in ms_modes:
        m.train(mode=mode)
    

def print_stats(stats):
    for c_i, lh in enumerate(stats):
        print(f"{c_i if c_i != 0 else f'{c_i} (root)'}:")
        print('\n'.join([f'{k:>30}\t{f"{v.item():.4f}" if v.numel() == 1 else v.tolist()}' for k,v in dict(sorted(lh.items())).items()]))


def clip_grads(models, reg_clip_grads_max_norm, reg_clip_grads_max_value):
    if reg_clip_grads_max_norm is not None and reg_clip_grads_max_value is not None:
        print("Warning: both reg_clip_grads_max_norm and reg_clip_grads_max_value are set. Using reg_clip_grads_max_norm.")
    for m in models:
        if reg_clip_grads_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(m.parameters(), reg_clip_grads_max_norm)
        elif reg_clip_grads_max_value is not None:
            torch.nn.utils.clip_grad_value_(m.parameters(), reg_clip_grads_max_value)


def take_training_step(X, y, parent, child, phase, hnet_parent_prev_params, config, loss_fn=F.cross_entropy):
    """
    parent and child structure: tuple (hnet, solver, hnet_optimizer, hnet_cond_id)
        child can be None if phase is "hnet->solver"
    """
    hnet_parent, solver_parent, hnet_parent_optim, hnet_parent_cond_id = parent
    hnet_child, solver_child, hnet_child_optim, hnet_child_cond_id = child
    for m in (hnet_parent, solver_parent, hnet_child, solver_child):
        if m is not None:
            m.train(mode=True)
    
    hnet_parent_optim.zero_grad()
    if hnet_child_optim is not None:
        hnet_child_optim.zero_grad()
    hnet_parent_optim.zero_grad()
    if hnet_child_optim is not None:
        hnet_child_optim.zero_grad()

    # generate theta and predict
    y_hat, params_solver = infer(X, phase, hnet_parent_cond_id=hnet_parent_cond_id, hnet_child_cond_id=hnet_child_cond_id,
        hnet_parent=hnet_parent, hnet_child=hnet_child, solver_parent=solver_parent, solver_child=solver_child, config=config)
    
    # task loss
    loss_class = loss_fn(y_hat, y)
    loss = loss_class
    # solvers' params regularization
    loss_solver_params_reg = torch.tensor(0., device=config["device"])
    if config["hnet"]["reg_alpha"] is not None and config["hnet"]["reg_alpha"] > 0.:
        loss_solver_params_reg = config["hnet"]["reg_alpha"] * sum([p.norm(p=2) for p in params_solver]) / len(params_solver)
    loss += loss_solver_params_reg
    perform_forgetting_reg = config["hnet"]["reg_beta"] is not None and config["hnet"]["reg_beta"] > 0.
    loss.backward(retain_graph=perform_forgetting_reg, create_graph=not config["hnet"]["detach_d_theta"])
    # gradient clipping
    clip_grads([m for m in (hnet_parent, hnet_child) if m is not None], config["hnet"]["reg_clip_grads_max_norm"], config["hnet"]["reg_clip_grads_max_value"])
    
    # regularization against forgetting other contexts
    loss_reg = torch.tensor(0., device=config["device"])
    if config["hnet"]["reg_beta"] is not None and config["hnet"]["reg_beta"] > 0.:
        loss_reg = config["hnet"]["reg_beta"] * get_reg_loss(hnet_parent, hnet_parent_prev_params, curr_cond_id=hnet_parent_cond_id, lr=config["hnet"]["reg_lr"], detach_d_theta=config["hnet"]["detach_d_theta"])
        loss_reg.backward()
        # gradient clipping
        clip_grads([m for m in (hnet_parent, hnet_child) if m is not None], config["hnet"]["reg_clip_grads_max_norm"], config["hnet"]["reg_clip_grads_max_value"])
    
    hnet_parent_optim.step()
    if hnet_child_optim is not None:
        hnet_child_optim.step()
    hnet_parent_optim.zero_grad()
    if hnet_child_optim is not None:
        hnet_child_optim.zero_grad()

    return loss_class.detach().clone(), loss_solver_params_reg.detach().clone(), loss_reg.detach().clone(), y_hat.var(dim=0).detach().clone()


def init_hnet_unconditionals(hnet, uncond_theta):
    assert [s for s in hnet.unconditional_param_shapes] == [list(p.shape) for p in uncond_theta], f"uncond_theta shapes don't match hnet.unconditional_param_shapes"
    
    params_before = [p.clone() for p in hnet.internal_params]
    params_final = [None] * len(hnet.param_shapes)
    # add conditional params
    for p_idx, p in zip(hnet.conditional_param_shapes_ref, hnet.conditional_params):
        params_final[p_idx] = p
    # add unconditional params
    hnet._unconditional_params_ref = hnet.unconditional_param_shapes_ref
    for p_idx, p in zip(hnet.unconditional_param_shapes_ref, uncond_theta):
        params_final[p_idx] = nn.Parameter(p.detach().clone(), requires_grad=True)
    # set internal params
    hnet._hnet._internal_params = nn.ParameterList(params_final)
    hnet._internal_params = nn.ParameterList(hnet._hnet._internal_params)

    return params_before


def remove_hnet_uncondtionals(hnet, prev_params=None):
    # store the unconditional parameters
    unconditionals = []
    for p_idx in hnet.unconditional_param_shapes_ref:
        unconditionals.append(hnet.internal_params[p_idx].detach().clone())
    
    # restore previous state of parameters
    hnet._unconditional_params_ref = None
    
    hnet._hnet._internal_params = nn.ParameterList([
        p for p_idx, p in enumerate(hnet.internal_params) if p_idx not in hnet.unconditional_param_shapes_ref
    ])
    hnet._internal_params = nn.ParameterList(hnet._hnet._internal_params) # TODO: chunked_mlp_hnet line 201
    
    # append additional conditional chunk embeddings
    if hnet._cemb_shape is not None and prev_params is not None:
        for c_i in range(hnet._num_cond_embs):
            param_to_add = prev_params[-hnet._num_cond_embs + c_i]
            if type(param_to_add) == nn.Parameter:
                param_to_add = param_to_add.detach().clone()
            elif type(param_to_add) == torch.Tensor:
                param_to_add = nn.Parameter(param_to_add.detach().clone(), requires_grad=True)
            else:
                raise ValueError(f"prev_params includes a value of type {type(param_to_add)}")
            hnet._internal_params.append(param_to_add)
    return unconditionals


def validate_cells_training_inputs(X, y, cells, config):
    assert X.shape[0] == y.shape[0], f"X and y have different number of samples"
    assert y.shape[1] == config["data"]["num_classes_per_task"], f"y has incorrect number of features"
    assert cells[0]["hnet_to_hnet_cond_id"] is None and cells[0]["hnet_theta_out_target"] is None and cells[0]["n_training_iters_hnet"] in (None, 0), \
        f"The last cell should have no child cells (list of cells sorted from the furthest from the root to the closest to the root)"
    assert cells[-1]["hnet_init_theta"] is None, f"The root cell (last in the cells list) should have no initial theta - its parameters are being learned"
    
    for c_i, c in enumerate(cells):
        assert set(("hnet", "solver", "hnet_optim", "hnet_to_hnet_cond_id", "hnet_to_solver_cond_id", "hnet_init_theta", "hnet_prev_params",
            "hnet_theta_out_target", "n_training_iters_solver", "n_training_iters_hnet")).issubset(set(c.keys())), f"Cell {c_i} is missing some of the required keys"
        if c_i + 1 < len(cells) - 1:
            assert sum([np.prod(p) for p in c["hnet"].unconditional_param_shapes]) == cells[c_i + 1]["hnet"].num_outputs, \
                f"Number of outputs of the {c_i + 1}-th cell's hnet should be equal to the number of unconditional parameters of the {c_i}-th cell's hnet"
    return None


def train_cells(X, y, cells, config, stats):
    """
    cells: list of dictionaries with the following keys (and corresponding values):
        {
            "hnet", "solver", "hnet_optim", "hnet_to_hnet_cond_id", "hnet_to_solver_cond_id", "hnet_init_theta", "hnet_prev_params",
            "hnet_theta_out_target", "n_training_iters_solver", "n_training_iters_hnet"
        }
        List of cells sorted from the furthest from the root to the closest to the root.
    """
    if len(cells) == 0:
        return stats

    # pop the first cell
    hnet, solver, hnet_optim, hnet_to_hnet_cond_id, hnet_to_solver_cond_id, hnet_init_theta, hnet_prev_params, hnet_theta_out_target, \
        n_training_iters_solver, n_training_iters_hnet = cells.pop(0).values()

    # initialize statistics - logging purposes
    c_stats = {l:torch.tensor(0.) for l in ("loss_hnet_hnet", "loss_hnet_solver_class", "loss_hnet_solver_theta_reg", "loss_hnet_forgetting_reg", "y_hat_var")}
    
    # train the hnet -> solver on the given X, y => create theta target for parent hnet
    if n_training_iters_solver is not None and n_training_iters_solver > 0:
        if hnet_init_theta is not None: # is None for the root hnet
            init_hnet_unconditionals(hnet, hnet_init_theta)
            # init optimizer of the initialized  unconditional parameters
            hnet_optim = torch.optim.Adam([*hnet.unconditional_params, *hnet.conditional_params], lr=config["hnet"]["lr"])
        for iter_i in range(n_training_iters_solver):
            curr_cell = (hnet, solver, hnet_optim, hnet_to_solver_cond_id)
            c_stats["loss_hnet_solver_class"], c_stats["loss_hnet_solver_theta_reg"], c_stats["loss_hnet_forgetting_reg"], c_stats["y_hat_var"] = take_training_step(
                X, y, parent=curr_cell, child=(None, None, None, None), phase="hnet->solver",
                hnet_parent_prev_params=hnet_prev_params, config=config, loss_fn=F.cross_entropy
            )
        # set the trained theta as the target for parent hnet
        if hnet_init_theta is not None: # is None for the root hnet
            cells[0]["hnet_theta_out_target"] = remove_hnet_uncondtionals(hnet)
    
    # train the hnet -> hnet on the given target theta
    if n_training_iters_hnet is not None and n_training_iters_hnet > 0:
        if hnet_init_theta is not None: # is None for the root hnet
            assert len(cells) == 0, "hnet_init_theta is not None for a non-root hnet"
            init_hnet_unconditionals(hnet, hnet_init_theta)
            # init optimizer of the initialized unconditional parameters
            hnet_optim = torch.optim.Adam([*hnet.unconditional_params, *hnet.conditional_params], lr=config["hnet"]["lr"])
        perform_forgetting_reg = config["hnet"]["reg_beta"] is not None and config["hnet"]["reg_beta"] > 0.
        
        for iter_i in range(n_training_iters_hnet):
            theta_target = torch.cat([p.detach().clone().view(-1) for p in hnet_theta_out_target])
            
            theta_hat = hnet(cond_id=hnet_to_hnet_cond_id)
            theta_hat = torch.cat([p.view(-1) for p in theta_hat])
            
            loss_hnet_hnet = torch.sqrt(F.mse_loss(theta_hat, theta_target))
            # loss_hnet_hnet = (theta_hat - theta_target).pow(2).sum()
            loss_hnet_hnet.backward(retain_graph=perform_forgetting_reg, create_graph=not config["hnet"]["detach_d_theta"])
            # gradient clipping
            clip_grads([hnet], config["hnet"]["reg_clip_grads_max_norm"], config["hnet"]["reg_clip_grads_max_value"])

            # regularization against forgetting other contexts
            loss_reg = torch.tensor(0., device=config["device"])
            if perform_forgetting_reg:
                loss_reg = config["hnet"]["reg_beta"] * get_reg_loss(hnet, hnet_prev_params, curr_cond_id=hnet_to_hnet_cond_id, lr=config["hnet"]["reg_lr"], detach_d_theta=config["hnet"]["detach_d_theta"])
                loss_reg.backward()
                # gradient clipping
                clip_grads([hnet], config["hnet"]["reg_clip_grads_max_norm"], config["hnet"]["reg_clip_grads_max_value"])

            hnet_optim.step()
            hnet_optim.zero_grad()
            c_stats["loss_hnet_hnet"] = loss_hnet_hnet.detach().clone()
        # set the trained theta as the target for parent hnet
        if hnet_init_theta is not None: # is None for the root hnet
            cells[0]["hnet_theta_out_target"] = remove_hnet_uncondtionals(hnet)


    # one step deeper (onto the parents of the current cell)
    stats.append(c_stats)
    return train_cells(X, y, cells, config, stats)
