import torch
import numpy as np


def get_params_info(cell, path, n_maintained_params, n_solver_params):
    name = "[" + "".join([str(i) for i in path]) + "]"
    n_maintained_hnet = cell.hnet.num_internal_params
    n_maintained_solver = cell.solver.num_internal_params
    # num_maintained = sum(p.numel() for p in cell.parameters())
    max_maintained_hnet = cell.hnet.num_params
    max_maintained_solver = cell.solver.num_params
    # max_maintained = sum([np.prod(p) for p in [*cell.hnet.param_shapes, *cell.solver.param_shapes]])

    print(f"- {name} hypernet:\t{n_maintained_hnet}\t({max_maintained_hnet} possible)")
    print(f"- {name} solver: \t{n_maintained_solver}\t({max_maintained_solver} possible)")

    n_maintained_params += n_maintained_hnet + n_maintained_solver
    n_solver_params += max_maintained_solver
    
    for child_idx, child in enumerate(cell.children):
        n_maintained_params, n_solver_params = get_params_info(
            child, path + [child_idx], n_maintained_params, n_solver_params)
    return n_maintained_params, n_solver_params


def print_arch_summary(root_cell):
    print("\nSummary of parameters:")
    n_maintained_params, n_solver_params = get_params_info(root_cell, [], 0, 0)
    print(f"---\nTotal available solver parameters:\t{n_solver_params}")
    print(f"Parameters maintained:\t\t{n_maintained_params}")
    print(f"-> Coefficient of compression:\t{(n_maintained_params / n_solver_params):.5f}")


def get_metrics_path_key(path_idx, path):
    m_key_p = f"[P{path_idx + 1}-{''.join([str(i) for i in path])}]"
    return m_key_p


def get_metrics_task_key(task_idx):
    m_key_t = f"[T{task_idx + 1}]"
    return m_key_t


def get_accuracy(y_hat, y, perc=True):
    acc = (y_hat.argmax(-1) == y).float().mean()
    return acc if not perc else acc * 100


def calc_accuracy(X, y, solver, solver_weights, use_data_from="test"):
    """Compute the test accuracy for a given dataset (validation)"""
    # assert use_data_from == "test" or data.num_val_samples > 0, "No validation data available."
    solver_train = solver.training
    solver.eval()
    acc = None

    with torch.no_grad():
        if use_data_from == "validation":
            raise NotImplementedError()
            # num_correct = 0

            # for batch_size, X, y, ids in data.val_iterator(config["data"]["batch_size"], return_ids=True):
            #     X = data.input_to_torch_tensor(X, config["device"], mode='inference')
            #     y = data.output_to_torch_tensor(y, config["device"], mode='inference')
            #     y_hat = solver.forward(X, weights=solver_weights)
            #     num_correct += int(torch.sum(y_hat.argmax(dim=1) == y.argmax(dim=1)).detach().cpu())

            # acc = num_correct / data.num_val_samples * 100.
        elif use_data_from == "test":
                if solver_weights is not None:
                    y_hat = solver(X, weights=solver_weights)
                else:
                    y_hat = solver(X)

                acc = (y_hat.argmax(dim=-1) == y.argmax(dim=-1)).float().mean() * 100.
        else:
            raise ValueError("Unknown data source (use 'test' or 'validation').")

    solver.train(mode=solver_train)
    return acc
