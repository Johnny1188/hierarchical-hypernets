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
