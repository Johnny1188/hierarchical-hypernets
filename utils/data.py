import os
import torch
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import numpy as np
import random
from hypnettorch.data import FashionMNISTData, MNISTData
from hypnettorch.data.special.split_mnist import get_split_mnist_handlers
from hypnettorch.data.special.split_cifar import get_split_cifar_handlers
from hypnettorch.data.special.permuted_mnist import PermutedMNISTList

from avalanche.benchmarks.classic import SplitMNIST, SplitCIFAR10, SplitCIFAR100, SplitCIFAR110,\
    PermutedMNIST, SplitFMNIST, SplitTinyImageNet

DATA_PATH = os.path.join(os.getenv("DATA_PATH"), "cl")
SEED = 0


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(SEED)


def get_transforms(dataset_name, img_shape):
    norm_transform = None
    if dataset_name == "mnist":
        norm_transform = transforms.Normalize((0.1307,), (0.3081,))
    elif dataset_name == "fmnist":
        norm_transform = transforms.Normalize((0.286,), (0.353,))
    else:
        print(f"[WARNING] No normalization transform for dataset {dataset_name}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(img_shape[:2]),
    ])

    # add normalization
    if norm_transform is not None:
        transform.transforms.append(norm_transform)

    # repeat across 3 channels
    if img_shape[-1] == 3:
        transform.transforms.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x))

    return transform


def get_data_handlers(config):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    to_load = {
        "old": config['data']['benchmark_specs_seen_before'],
        "new": config['data']['benchmark_specs_seen_now']
    }
    data_handlers = {k:dict() for k in to_load.keys()}
    
    for data_time_info, benchmark_specs in to_load.items():
        print(f"[DATA] Loading data handlers for '{data_time_info}' contexts: {','.join([b['name'] for b in benchmark_specs])}")
        for benchmark_spec in benchmark_specs:
            if benchmark_spec["name"] == "splitmnist":
                transform = get_transforms("mnist", benchmark_spec["in_shape"])
                benchmark = SplitMNIST(
                    n_experiences=benchmark_spec["num_tasks"],
                    train_transform=transform,
                    eval_transform=transform,
                    return_task_id=False,
                    dataset_root=config["data"]["data_dir"],
                    seed=SEED
                )
                # data_handlers.append(SplitMNIST(n_experiences=5, return_task_id=True, dataset_root=DATA_PATH, seed=SEED))
                # data_handlers.extend(get_split_mnist_handlers(config["data"]["data_dir"], use_one_hot=True, num_tasks=benchmark_spec["num_tasks"], num_classes_per_task=benchmark_spec["num_classes_per_task"], validation_size=config["data"]["validation_size"]))
            elif benchmark_spec["name"] == "splitfmnist":
                transform = get_transforms("fmnist", benchmark_spec["in_shape"])
                benchmark = SplitFMNIST(
                    n_experiences=benchmark_spec["num_tasks"],
                    train_transform=transform,
                    eval_transform=transform,
                    return_task_id=False,
                    dataset_root=config["data"]["data_dir"],
                    seed=SEED
                )
            elif benchmark_spec["name"] == "permutedmnist":
                transform = get_transforms("mnist", benchmark_spec["in_shape"])
                benchmark = PermutedMNIST(
                    n_experiences=benchmark_spec["num_tasks"],
                    train_transform=transform,
                    eval_transform=transform,
                    dataset_root=config["data"]["data_dir"],
                    seed=SEED
                )
                # rand = np.random.RandomState(0) # ensure reproducibility
                # permutations = [None] + [rand.permutation(28*28) for _ in range(benchmark_spec["num_tasks"] - 1)]
                # data_handlers.extend(PermutedMNISTList(permutations, config["data"]["data_dir"],
                #     padding=0, trgt_padding=None, show_perm_change_msg=False, validation_size=config["data"]["validation_size"]))
            elif benchmark_spec["name"] == "splitcifar10":
                benchmark = SplitCIFAR10(
                    n_experiences=benchmark_spec["num_tasks"],
                    return_task_id=False,
                    dataset_root=config["data"]["data_dir"],
                    seed=SEED
                )
            elif benchmark_spec["name"] == "splitcifar100":
                benchmark = SplitCIFAR100(
                    n_experiences=benchmark_spec["num_tasks"],
                    return_task_id=False,
                    dataset_root=config["data"]["data_dir"],
                    seed=SEED
                )
            elif benchmark_spec["name"] == "splitcifar110":
                benchmark = SplitCIFAR110(
                    n_experiences=benchmark_spec["num_tasks"],
                    dataset_root_cifar10=config["data"]["data_dir"],
                    dataset_root_cifar100=config["data"]["data_dir"],
                    seed=SEED
                )
                # data_handlers.extend(get_split_cifar_handlers(config["data"]["data_dir"], use_one_hot=True, num_tasks=benchmark_spec["num_tasks"], num_classes_per_task=benchmark_spec["num_classes_per_task"], validation_size=config["data"]["validation_size"]))
            elif benchmark_spec["name"] == "splittinyimagenet":
                benchmark = SplitTinyImageNet(
                    n_experiences=benchmark_spec["num_tasks"],
                    return_task_id=False,
                    dataset_root=config["data"]["data_dir"],
                    seed=SEED
                )
            elif benchmark_spec["name"] == "splitmnist,splitcifar100":
                raise NotImplementedError("TODO")
                splitmnist = get_split_mnist_handlers(config["data"]["data_dir"], use_one_hot=True, num_tasks=config["data"]["splitmnist"]["num_tasks"], num_classes_per_task=config["data"]["splitmnist"]["num_classes_per_task"], validation_size=config["data"]["validation_size"])
                # transform mnist data to match cifar
                for data_handler in splitmnist:
                    # pad with zeros to match cifar
                    data_handler._data["in_data"] = np.pad(
                        data_handler._data["in_data"].reshape(-1,28,28,1),
                        pad_width=((0,0),(2,2),(2,2),(0,0)),
                        mode="constant",
                        constant_values=0
                    ).reshape(-1, 32 * 32)
                    # convert to rgb
                    data_handler._data["in_data"] = np.repeat(data_handler._data["in_data"], repeats=3, axis=-1)
                    data_handler._data["in_shape"] = [32, 32, 3]

                splitcifar100 = get_split_cifar_handlers(config["data"]["data_dir"], use_one_hot=True, num_tasks=config["data"]["splitcifar100"]["num_tasks"], num_classes_per_task=config["data"]["splitcifar100"]["num_classes_per_task"], validation_size=config["data"]["validation_size"])
                data_handlers = [*splitmnist, *splitcifar100]
            elif benchmark_spec["name"] == "permutedmnist,splitcifar100,splitmnist":
                raise NotImplementedError("TODO")
                # Permuted MNIST
                rand = np.random.RandomState(0) # ensure reproducibility
                permutations = [None] + [rand.permutation(32*32) for _ in range(config["data"]["permutedmnist"]["num_tasks"] - 1)]
                perm_mnist_list = PermutedMNISTList(permutations, config["data"]["data_dir"], padding=2, trgt_padding=None,
                    show_perm_change_msg=False, validation_size=config["data"]["validation_size"], use_3_channels=True)
                # convert to 3 channels
                channels_tr = transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)
                for data_handler in perm_mnist_list:
                    data_handler._transform.transforms.append(channels_tr)
                    data_handler._using_3_channels = True
                
                # SplitCIFAR100
                splitcifar100 = get_split_cifar_handlers(config["data"]["data_dir"], use_one_hot=True, num_tasks=config["data"]["splitcifar100"]["num_tasks"], num_classes_per_task=config["data"]["splitcifar100"]["num_classes_per_task"], validation_size=config["data"]["validation_size"])

                # SplitMNIST
                splitmnist = get_split_mnist_handlers(config["data"]["data_dir"], use_one_hot=True, num_tasks=config["data"]["splitmnist"]["num_tasks"], num_classes_per_task=config["data"]["splitmnist"]["num_classes_per_task"], validation_size=config["data"]["validation_size"])
                # transform mnist data to match cifar
                for data_handler in splitmnist:
                    # pad with zeros to match cifar
                    data_handler._data["in_data"] = np.pad(
                        data_handler._data["in_data"].reshape(-1,28,28,1),
                        pad_width=((0,0),(2,2),(2,2),(0,0)),
                        mode="constant",
                        constant_values=0
                    ).reshape(-1, 32 * 32)
                    # convert to rgb
                    data_handler._data["in_data"] = np.repeat(data_handler._data["in_data"], repeats=3, axis=-1)
                    data_handler._data["in_shape"] = [32, 32, 3]

                data_handlers = [*perm_mnist_list, *splitcifar100, *splitmnist]
            else:
                raise NotImplementedError(f"Unknown dataset: {benchmark_spec['name']}")

            ### add data handler for each experience in the benchmark
            data_handlers[data_time_info][benchmark_spec["name"]] = []
            for experience_train, experience_test in zip(benchmark.train_stream, benchmark.test_stream):
                data_handlers[data_time_info][benchmark_spec["name"]].append({
                    "classes_in_experience": experience_train.classes_in_this_experience,
                    "classes_in_orig_dataset": benchmark.n_classes,
                    "num_train_samples": len(experience_train.dataset),
                    "num_test_samples": len(experience_test.dataset),
                    "train_loader": torch.utils.data.DataLoader(
                        experience_train.dataset,
                        batch_size=config["data"]["batch_size"],
                        shuffle=True
                    ),
                    "test_loader": torch.utils.data.DataLoader(
                        experience_test.dataset,
                        batch_size=config["data"]["batch_size"],
                        shuffle=False
                    )
                })

    return data_handlers


def randomize_targets(X, y, noise=0.1, n_classes=10):
    X_noisy = X.clone()
    y_noisy = y.clone()
    for i in range(len(y)):
        if random.random() < noise:
            y_noisy[i] = random.choice([c_i for c_i in range(0, n_classes) if c_i is not y_noisy[i].item()])
    return X_noisy, y_noisy


def select_from_classes(x, y, classes_to_select):
    samples_mask = np.array([s in classes_to_select for s in y])
    return x[samples_mask,:], y[samples_mask]


# Data transformations and loading - MNIST
def get_mnist_data_loaders(batch_size=32, flatten=False, drop_last=True, only_classes=None, img_size=28):
    # build transforms
    img_transformation = transforms.Compose([
        transforms.ToTensor(),
    ])
    if img_size < 28 and img_size >= 24:
        img_transformation.transforms.append(transforms.Resize(img_size))
    elif img_size < 24:
        img_transformation.transforms.append(transforms.CenterCrop(24))
        img_transformation.transforms.append(transforms.Resize(img_size))
    if flatten:
        img_transformation.transforms.append(transforms.Lambda(lambda x: torch.flatten(x)))

    train_dataset = datasets.MNIST(DATA_PATH, train=True, download=False, transform=img_transformation)
    if only_classes != None: # list of classes to select from the dataset (0,1,...)
        idx = torch.isin(train_dataset.targets, only_classes if type(only_classes) == torch.Tensor else torch.tensor(only_classes))
        train_dataset.targets = train_dataset.targets[idx]
        train_dataset.data = train_dataset.data[idx]
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, worker_init_fn=seed_worker, generator=g
    )
    
    test_dataset = datasets.MNIST(DATA_PATH, train=False, download=False, transform=img_transformation)
    if only_classes != None: # list of classes to select from the dataset (0,1,...)
        idx = torch.isin(test_dataset.targets, only_classes if type(only_classes) == torch.Tensor else torch.tensor(only_classes))
        test_dataset.targets = test_dataset.targets[idx]
        test_dataset.data = test_dataset.data[idx]
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, worker_init_fn=seed_worker, generator=g
    )
    return train_loader, test_loader, datasets.MNIST.classes


# Data transformations and loading - EMNIST
def get_emnist_data_loaders(batch_size=32, drop_last=True):
    img_transformation = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.EMNIST(DATA_PATH, train=True, download=False, transform=img_transformation)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, worker_init_fn=seed_worker, generator=g
    )
    test_dataset = datasets.EMNIST(DATA_PATH, train=False, download=False, transform=img_transformation)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, worker_init_fn=seed_worker, generator=g
    )
    return train_loader, test_loader, datasets.EMNIST.classes
