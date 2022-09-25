

def get_solver_specs(solver_name, in_shape=None, num_outputs=None):
    available_solvers = {
        "lenet": {
            "in_shape": in_shape,
            "num_classes": num_outputs,
            "arch": "mnist_large",
            "no_weights": True,
        },
        "zenkenet": {
            "in_shape": in_shape,
            "num_classes": num_outputs,
            "arch": "cifar",
            "no_weights": True,
            "dropout_rate": 0.25,
        },
        "resnet": {
            "in_shape": in_shape,
            "num_classes": num_outputs,
            "n": 5,
            "k": 1,
            "no_weights": True,
        },
    }
    assert solver_name in available_solvers.keys(), "Invalid solver name"
    return available_solvers[solver_name]
