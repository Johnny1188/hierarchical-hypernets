

def get_data_specs(data_name):
    assert data_name in ["mnist|fmnist", "splitmnist", "splitcifar10", "splitcifar100"], "Invalid data name"
    if data_name == "mnist|fmnist":
        return {
            "name": "mnist|fmnist",
            "in_shape": [28, 28, 1],
            "num_tasks": 5,
            "num_classes_per_task": 2,
        }
    elif data_name == "splitmnist":
        return {
            "name": "splitmnist",
            "in_shape": [28, 28, 1],
            "num_tasks": 5,
            "num_classes_per_task": 2,
        }
    elif data_name == "splitcifar10":
        return {
            "name": "splitcifar10",
            "in_shape": [32, 32, 3],
            "num_tasks": 5,
            "num_classes_per_task": 2,
        }
    elif data_name == "splitcifar100":
        return {
            "name": "splitcifar100",
            "in_shape": [32, 32, 3],
            "num_tasks": 6,
            "num_classes_per_task": 10,
        }
