
def get_task_separation_idxs(n_tasks, n_classes_per_task):
    return [
        (t_i * n_classes_per_task, (t_i + 1) * n_classes_per_task)
        for t_i in range(n_tasks)
    ]


def get_data_specs(benchmarks, **kwargs):
    data_specs = []
    for bmark in benchmarks:
        if bmark == "mnist|fmnist":
            data_specs.append({
                "name": "mnist|fmnist",
                "in_shape": [28, 28, 1],
                "num_tasks": 5,
                "num_classes_per_task": 2,
                "task_separation_idxs": [(0,2), (2,4), (4,6), (6,8), (8,10)],
            })
        elif bmark == "splitmnist":
            data_specs.append({
                "name": "splitmnist",
                "in_shape": [28, 28, 1] if "in_shape" not in kwargs else kwargs["in_shape"],
                "num_tasks": 5,
                "num_classes_per_task": 2,
                "task_separation_idxs": [(0,2), (2,4), (4,6), (6,8), (8,10)],
            })
        elif bmark == "splitfmnist":
            data_specs.append({
                "name": "splitfmnist",
                "in_shape": [28, 28, 1] if "in_shape" not in kwargs else kwargs["in_shape"],
                "num_tasks": 5,
                "num_classes_per_task": 2,
                "task_separation_idxs": [(0,2), (2,4), (4,6), (6,8), (8,10)],
            })
        elif bmark == "permutedmnist":
            n_permutations = 15
            if "n_permuatations" in kwargs:
                n_permutations = kwargs.get("n_permuatations")

            data_spec = {
                "name": "permutedmnist",
                "in_shape": [28, 28, 1] if "in_shape" not in kwargs else kwargs["in_shape"],
                "num_tasks": n_permutations,
                "num_classes_per_task": 10,
                "task_separation_idxs": None, # specified below
            }
            data_spec["task_separation_idxs"] = get_task_separation_idxs(data_spec["num_tasks"], data_spec["num_classes_per_task"])
            data_specs.append(data_spec)
        elif bmark == "splitcifar10":
            data_specs.append({
                "name": "splitcifar10",
                "in_shape": [32, 32, 3] if "in_shape" not in kwargs else kwargs["in_shape"],
                "num_tasks": 5,
                "num_classes_per_task": 2,
                "task_separation_idxs": [(0,2), (2,4), (4,6), (6,8), (8,10)],
            })
        elif bmark == "splitcifar100":
            data_specs.append({
                "name": "splitcifar100",
                "in_shape": [32, 32, 3] if "in_shape" not in kwargs else kwargs["in_shape"],
                "num_tasks": 5,
                "num_classes_per_task": 10,
                "task_separation_idxs": [(0,10), (10, 20), (20, 30), (30, 40), (40, 50)],
            })
        elif bmark == "splitcifar110":
            data_specs.append({
                "name": "splitcifar110",
                "in_shape": [32, 32, 3] if "in_shape" not in kwargs else kwargs["in_shape"],
                "num_tasks": 6,
                "num_classes_per_task": 10,
                "task_separation_idxs": [(0,10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60)],
            })
        elif bmark == "splittinyimagenet":
            data_specs.append({
                "name": "splittinyimagenet",
                "in_shape": [64, 64, 3] if "in_shape" not in kwargs else kwargs["in_shape"],
                "num_tasks": 10,
                "num_classes_per_task": 10,
                "task_separation_idxs": None, # specified below
            })
            data_specs[-1]["task_separation_idxs"] = get_task_separation_idxs(data_specs[-1]["num_tasks"], data_specs[-1]["num_classes_per_task"])
        elif bmark == "splitmnist,splitcifar100":
            data_specs.append({
                "name": "splitmnist,splitcifar100",
                "in_shape": [32, 32, 3] if "in_shape" not in kwargs else kwargs["in_shape"],
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
            })
        elif bmark == "permutedmnist,splitcifar100,splitmnist":
            n_permutations = 25
            if "n_permuatations" in kwargs:
                n_permutations = kwargs.get("n_permuatations")
            
            data_spec = {
                "name": "permutedmnist,splitcifar100,splitmnist",
                "in_shape": [32, 32, 3] if "in_shape" not in kwargs else kwargs["in_shape"],
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
            data_spec["task_separation_idxs"] = [
                *get_task_separation_idxs(data_spec["permutedmnist"]["num_tasks"], data_spec["permutedmnist"]["num_classes_per_task"]),
                *get_task_separation_idxs(data_spec["splitcifar100"]["num_tasks"], data_spec["splitcifar100"]["num_classes_per_task"]),
                *get_task_separation_idxs(data_spec["splitmnist"]["num_tasks"], data_spec["splitmnist"]["num_classes_per_task"]),
            ]
            data_specs.append(data_spec)
        else:
            raise ValueError(f"Unknown benchmark name: {bmark}")
    return data_specs
