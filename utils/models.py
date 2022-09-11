import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
from copy import deepcopy
import gc
import time
from hypnettorch.mnets import LeNet, ZenkeNet, ResNet
from hypnettorch.hnets import HMLP, StructuredHMLP, ChunkedHMLP

from utils.data import get_mnist_data_loaders, get_emnist_data_loaders, randomize_targets, select_from_classes
from utils.visualization import show_imgs, get_model_dot
from utils.others import measure_alloc_mem, count_parameters
from utils.timing import func_timer
from utils.metrics import get_accuracy, calc_accuracy
from utils.hypnettorch_utils import correct_param_shapes, calc_delta_theta, get_reg_loss_for_cond, get_reg_loss, \
    infer, print_stats, print_metrics, clip_grads, take_training_step, init_hnet_unconditionals, remove_hnet_uncondtionals, \
    validate_cells_training_inputs, train_cells
from Cell import Cell

torch.set_printoptions(precision=3, linewidth=180)


def init_arch(arch_config, config):
    root_cells = []
    for root_cell_config in arch_config:
        root_cell = create_tree(root_cell_config, config)
        root_cells.append(root_cell)

    return root_cells


def create_tree(cell_config, config):
    # create target network (solver)
    config["num_cells"] = 1
    [solver] = get_target_nets(config=config)

    # find out the this cell's hypernetwork's output shape
    hnet_out_shape = solver.param_shapes
    max_num_of_params = sum([np.prod(s) for s in hnet_out_shape])
    children = []
    for child_cell_config in cell_config["children"]:
        child_cell = create_tree(child_cell_config, config)
        children.append(child_cell)
        # find out whether the child hypernetwork requires more parameters to be generated than the solver of this cell
        if child_cell.hnet._no_uncond_weights is True and child_cell.hnet._no_cond_weights is True:
            if sum([np.prod(s) for s in child_cell.hnet.param_shapes]) > max_num_of_params:
                hnet_out_shape = child_cell.hnet.param_shapes
                max_num_of_params = sum([np.prod(s) for s in hnet_out_shape])
        elif child_cell.hnet._no_uncond_weights is True and child_cell.hnet._no_cond_weights is False:
            if sum([np.prod(s) for s in child_cell.hnet.unconditional_param_shapes]) > max_num_of_params:
                hnet_out_shape = child_cell.hnet.unconditional_param_shapes
                max_num_of_params = sum([np.prod(s) for s in hnet_out_shape])
        elif child_cell.hnet._no_uncond_weights is False and child_cell.hnet._no_cond_weights is True:
            if sum([np.prod(s) for s in child_cell.hnet.unconditional_param_shapes]) > max_num_of_params:
                hnet_out_shape = child_cell.hnet.unconditional_param_shapes
                max_num_of_params = sum([np.prod(s) for s in hnet_out_shape])
        else:
            print(f"[WARNING] Non-root hypernetwork maintains all its parameters")

    # create hypernetwork and package everything into a cell
    hnet = ChunkedHMLP(
        hnet_out_shape,
        layers=cell_config["hnet"]["model"]["layers"],
        chunk_size=cell_config["hnet"]["model"]["chunk_size"],
        chunk_emb_size=cell_config["hnet"]["model"]["chunk_emb_size"],
        cond_chunk_embs=cell_config["hnet"]["model"]["cond_chunk_embs"],
        cond_in_size=cell_config["hnet"]["model"]["cond_in_size"],
        num_cond_embs=cell_config["hnet"]["model"]["num_cond_embs"],
        no_uncond_weights=cell_config["hnet"]["model"]["no_uncond_weights"],
        no_cond_weights=cell_config["hnet"]["model"]["no_cond_weights"],
    ).to(cell_config["device"])

    # package into cell
    if cell_config["hnet"]["model"]["no_uncond_weights"] is True and cell_config["hnet"]["model"]["no_cond_weights"] is True:
        needs_theta_type = "all"
    elif cell_config["hnet"]["model"]["no_uncond_weights"] is True and cell_config["hnet"]["model"]["no_cond_weights"] is False:
        needs_theta_type = "uncond_weights"
    elif cell_config["hnet"]["model"]["no_uncond_weights"] is False and cell_config["hnet"]["model"]["no_cond_weights"] is True:
        needs_theta_type = "cond_weights"
    else:
        needs_theta_type = "none"
    cell = Cell(
        hnet=hnet,
        solver=solver,
        children=children,
        num_tasks=config["data"]["num_tasks"],
        needs_theta_type=needs_theta_type,
        config=cell_config,
        init_hnet_optim=True,
        device=config["device"]
    )

    return cell


def get_hnets(config, target_nets_shapes):
    torch.manual_seed(0)
    np.random.seed(0)

    assert "num_cells" in config.keys() and config["num_cells"] == len(target_nets_shapes), \
        "Number of cells must be equal to number of target nets' parameter shapes"

    # create hypernetworks
    hnets = []
    for c_i, hnet_out_shape in enumerate(reversed(target_nets_shapes)):
        if hnet_out_shape == -1: # generate params for the previously created hypernetwork (child hypernet)
            if config["hnet"]["model"]["children_no_uncond_weights"] is True and config["hnet"]["model"]["children_no_cond_weights"] is True:
                hnet_out_shape = hnets[0].param_shapes
            elif config["hnet"]["model"]["children_no_uncond_weights"] is True and config["hnet"]["model"]["children_no_cond_weights"] is False:
                hnet_out_shape = hnets[0].unconditional_param_shapes
            elif config["hnet"]["model"]["children_no_uncond_weights"] is False and config["hnet"]["model"]["children_no_cond_weights"] is True:
                hnet_out_shape = hnets[0].conditional_param_shapes
            else:
                raise ValueError(f"hnet_out_shape is -1 (== get param shapes of previous hypernet) but children_no_uncond_weights and children_no_cond_weights are False")

        hnets.insert(0, ChunkedHMLP(
                hnet_out_shape,
                layers=config["hnet"]["model"]["layers"],
                chunk_size=config["hnet"]["model"]["chunk_size"],
                chunk_emb_size=config["hnet"]["model"]["chunk_emb_size"],
                cond_chunk_embs=config["hnet"]["model"]["cond_chunk_embs"],
                cond_in_size=config["hnet"]["model"]["cond_in_size"],
                num_cond_embs=config["hnet"]["model"]["num_cond_embs"],
                no_uncond_weights=config["hnet"]["model"]["root_no_uncond_weights"] if c_i == config["num_cells"] - 1 else config["hnet"]["model"]["children_no_uncond_weights"],
                no_cond_weights=config["hnet"]["model"]["root_no_cond_weights"] if c_i == config["num_cells"] - 1 else config["hnet"]["model"]["children_no_cond_weights"],
            ).to(config["device"])
        )
        # hnet_root.apply_chunked_hyperfan_init(mnet=hnet_child)
    return hnets


def get_target_nets(config):
    torch.manual_seed(0)
    np.random.seed(0)

    assert "num_cells" in config.keys() and "solver" in config.keys() and "use" in config["solver"].keys(), \
        f"Missing configurations to generate target networks: {config}"


    # create target networks (solvers)
    target_nets = []
    for _ in range(config["num_cells"]):
        if config["solver"]["use"] == "lenet":
            target_nets.append(
                LeNet(
                    **config["solver"]["specs"]
                ).to(config["device"])
            )
        elif config["solver"]["use"] == "zenkenet":
            target_nets.append(
                ZenkeNet(
                    **config["solver"]["specs"]
                ).to(config["device"])
            )
        elif config["solver"]["use"] == "resnet":
            target_nets.append(
                ResNet(
                    **config["solver"]["specs"]
                ).to(config["device"])
            )
        else:
            raise ValueError(f"Unknown solver: {config['solver']['use']}")
    return target_nets
