import torch
import numpy as np


def measure_alloc_mem():
    torch.cuda.synchronize()
    max_mem = torch.cuda.max_memory_allocated() / 1024**2
    mem = torch.cuda.memory_allocated() / 1024**2
    return max_mem, mem


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_optimizer(params, lr, weight_decay=0, beta_1=0.9, beta_2=0.999):
    optimizer = torch.optim.Adam(params, lr=lr, betas=[beta_1, beta_2], weight_decay=weight_decay)
    return optimizer


class EarlyStopper:
    # modified from https://stackoverflow.com/a/73704579
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss_seen = np.inf

    def early_stop(self, loss):
        if loss < self.min_loss_seen:
            self.min_loss_seen = loss
            self.counter = 0
        elif loss > (self.min_loss_seen - self.min_delta):
            self.counter += 1
        return self.counter >= self.patience
