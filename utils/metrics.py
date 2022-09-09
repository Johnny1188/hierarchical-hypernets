import torch


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
