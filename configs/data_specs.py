
def get_task_separation_idxs(n_tasks, n_classes_per_task):
    return [
        (t_i * n_classes_per_task, (t_i + 1) * n_classes_per_task)
        for t_i in range(n_tasks)
    ]


def get_data_specs(data_name, **kwargs):
    if data_name == "mnist|fmnist":
        return {
            "name": "mnist|fmnist",
            "in_shape": [28, 28, 1],
            "num_tasks": 5,
            "num_classes_per_task": 2,
            "task_separation_idxs": [(0,2), (2,4), (4,6), (6,8), (8,10)],
        }
    elif data_name == "splitmnist":
        return {
            "name": "splitmnist",
            "in_shape": [28, 28, 1],
            "num_tasks": 5,
            "num_classes_per_task": 2,
            "task_separation_idxs": [(0,2), (2,4), (4,6), (6,8), (8,10)],
        }
    elif data_name == "permutedmnist":
        data_config = {
            "name": "permutedmnist",
            "in_shape": [28, 28, 1],
            "num_tasks": 25,
            "num_classes_per_task": 10,
            "task_separation_idxs": None, # specified below
        }
        data_config["task_separation_idxs"] = get_task_separation_idxs(data_config["num_tasks"], data_config["num_classes_per_task"])
        return data_config
    elif data_name == "splitcifar10":
        return {
            "name": "splitcifar10",
            "in_shape": [32, 32, 3],
            "num_tasks": 5,
            "num_classes_per_task": 2,
            "task_separation_idxs": [(0,2), (2,4), (4,6), (6,8), (8,10)],
        }
    elif data_name == "splitcifar100":
        return {
            "name": "splitcifar100",
            "in_shape": [32, 32, 3],
            "num_tasks": 6,
            "num_classes_per_task": 10,
            "task_separation_idxs": [(0,10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60)],
        }
    elif data_name == "splitmnist,splitcifar100":
        return {
            "name": "splitmnist,splitcifar100",
            "in_shape": [32, 32, 3],
            "num_tasks": 11,
            "splitmnist": {
                "num_tasks": 5,
                "num_classes_per_task": 2,
            },
            "splitcifar100": {
                "num_tasks": 6,
                "num_classes_per_task": 10,
            },
            "task_separation_idxs": [(0,2), (2,4), (4,6), (6,8), (8,10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70)],
        }
    elif data_name == "permutedmnist,splitcifar100,splitmnist":
        n_permutations = 25
        if "n_permuatations" in kwargs:
            n_permutations = kwargs.get("n_permuatations")
        
        data_config = {
            "name": "permutedmnist,splitcifar100,splitmnist",
            "in_shape": [32, 32, 3],
            "num_tasks": 11 + n_permutations,
            "permutedmnist": {
                "num_tasks": n_permutations,
                "num_classes_per_task": 10,
            },
            "splitcifar100": {
                "num_tasks": 6,
                "num_classes_per_task": 10,
            },
            "splitmnist": {
                "num_tasks": 5,
                "num_classes_per_task": 2,
            },
            "task_separation_idxs": None,
        }
        data_config["task_separation_idxs"] = [
            *get_task_separation_idxs(data_config["permutedmnist"]["num_tasks"], data_config["permutedmnist"]["num_classes_per_task"]),
            *get_task_separation_idxs(data_config["splitcifar100"]["num_tasks"], data_config["splitcifar100"]["num_classes_per_task"]),
            *get_task_separation_idxs(data_config["splitmnist"]["num_tasks"], data_config["splitmnist"]["num_classes_per_task"]),
        ]
        return data_config
    else:
        raise ValueError(f"Unknown data name: {data_name}")
