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

DATA_PATH = os.getenv("DATA_PATH")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)


def get_data_handlers(config):
    torch.manual_seed(0)
    np.random.seed(0)

    print(f"[DATA] Loading data handlers for {config['data']['name']}")

    if config["data"]["name"] == "mnist|fmnist":
        mnist = MNISTData(config["data"]["data_dir"], use_one_hot=True, validation_size=config["data"]["validation_size"])
        fmnist = FashionMNISTData(config["data"]["data_dir"], use_one_hot=True, validation_size=config["data"]["validation_size"])
        data_handlers = [mnist, fmnist]
    elif config["data"]["name"] == "splitmnist":
        data_handlers = get_split_mnist_handlers(config["data"]["data_dir"], use_one_hot=True, num_tasks=config["data"]["num_tasks"], num_classes_per_task=config["data"]["num_classes_per_task"], validation_size=config["data"]["validation_size"])
    elif config["data"]["name"] == "permutedmnist":
        rand = np.random.RandomState(0) # ensure reproducibility
        permutations = [None] + [rand.permutation(28*28) for _ in range(config["data"]["num_tasks"] - 1)]
        data_handlers = PermutedMNISTList(permutations, config["data"]["data_dir"],
            padding=0, trgt_padding=None, show_perm_change_msg=False, validation_size=config["data"]["validation_size"])
    elif config["data"]["name"] in ("splitcifar10", "splitcifar100"):
        data_handlers = get_split_cifar_handlers(config["data"]["data_dir"], use_one_hot=True, num_tasks=config["data"]["num_tasks"], num_classes_per_task=config["data"]["num_classes_per_task"], validation_size=config["data"]["validation_size"])
    elif config["data"]["name"] == "splitmnist,splitcifar100":
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
    elif config["data"]["name"] == "permutedmnist,splitcifar100,splitmnist":
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
        raise NotImplementedError(f"Unknown dataset: {config['data']['name']}")

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
