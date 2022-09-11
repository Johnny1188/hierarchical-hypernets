import torch
from torch import nn
import numpy as np
import math

from utils.others import get_optimizer


class Cell(nn.Module):
    def __init__(self, hnet, solver, num_tasks, needs_theta_type="none", children=[], config={}, init_hnet_optim=True, device="cpu"):
        super().__init__()
        self.hnet = hnet
        self.solver = solver
        self.children = children
        self.num_tasks = num_tasks
        self.needs_theta_type = needs_theta_type
        self.config = config
        self.device = device
        self.hnet_theta_optim = None
        self.task_emb_optims = None
        self._init_params()

        if init_hnet_optim is True:
            self.reinit_hnet_theta_optim()
            self.reinit_task_emb_optims()

    def reinit_hnet_theta_optim(self):
        self.hnet_theta_optim = get_optimizer(
            params=self.hnet.unconditional_params,
            lr=self.config["hnet"]["lr"],
            beta_1=self.config["hnet"]["adam_beta_1"],
            beta_2=self.config["hnet"]["adam_beta_2"],
            weight_decay=self.config["hnet"]["weight_decay"],
        )
        for child in self.children:
            child.reinit_hnet_theta_optim()
    
    def reinit_task_emb_optims(self, task_i=None):
        if task_i is not None and self.task_emb_optims is not None:
            ### reinitialize only a single task embedding optimizer
            self.task_emb_optims[task_i] = get_optimizer(
                params=[self.hnet.conditional_params[task_i]],
                lr=self.config["hnet"]["lr"],
                beta_1=self.config["hnet"]["adam_beta_1"],
                beta_2=self.config["hnet"]["adam_beta_2"],
                weight_decay=self.config["hnet"]["weight_decay"],
            )
        else:
            ### reinitialize all task embedding optimizers
            self.task_emb_optims = [
                get_optimizer(
                    params=[cond_param],
                    lr=self.config["hnet"]["lr"],
                    beta_1=self.config["hnet"]["adam_beta_1"],
                    beta_2=self.config["hnet"]["adam_beta_2"],
                    weight_decay=self.config["hnet"]["weight_decay"],
                ) for cond_param in self.hnet.conditional_params
            ]
        for child in self.children:
            child.reinit_hnet_theta_optim(task_i=task_i)        

    def forward(self, x, task_i, path, theta_hnet=None, path_trace={"cond_ids": []}):
        assert type(path) == list, "path must be a list of integers"
        
        if len(path) == 0: # hnet -> solver
            cond_id = self.get_cond_id(task_i, None)
            
            ### hypernet -> solver
            theta_solver = self.hnet.forward(cond_id=cond_id, weights=theta_hnet) # hnet -> solver
            y_hat = self.solver.forward(x, weights=Cell.correct_param_shapes(theta_solver, self.solver.param_shapes))
            
            ### get task head indices
            task_head_idxs = self.get_task_head_idxs(task_i)
            y_hat = y_hat[:, task_head_idxs[0]:task_head_idxs[1]]

            ### save path trace to this cell
            path_trace["solving_cell"] = self
            path_trace["cond_ids"].append(cond_id)
        else: # hnet -> hnet
            ### generate params (theta) for child cell's hypernet
            child_idx = path.pop(0)
            cond_id = self.get_cond_id(task_i, child_idx)
            theta_child_hnet = self.hnet.forward(cond_id=cond_id, weights=theta_hnet)
            path_trace["cond_ids"].append(cond_id)
            
            if self.children[child_idx].needs_theta_type == "uncond_weights":
                theta_child_hnet = {"uncond_weights": theta_child_hnet}
            elif self.children[child_idx].needs_theta_type == "cond_weights":
                theta_child_hnet = {"cond_weights": theta_child_hnet}
            elif self.children[child_idx].needs_theta_type == "none":
                theta_child_hnet = None

            y_hat, theta_solver, path_trace = self.children[child_idx].forward(x, task_i=task_i, path=path, theta_hnet=theta_child_hnet)
        
        return y_hat, theta_solver, path_trace

    def get_task_head_idxs(self, task_i):
        """ Get solver's head indices for a given task """
        if "task_heads" not in self.config["solver"] or self.config["solver"]["task_heads"] is None:
            return [0, self.solver.num_classes]
        
        return [self.config["solver"]["task_heads"][task_i][0], self.config["solver"]["task_heads"][task_i][1]]

    def get_cond_id(self, task_i, child_idx):
        if child_idx is None:
            return task_i
        else:
            return task_i + ((child_idx + 1) * self.num_tasks) # children indexed from 0

    def toggle_mode(self, mode):
        assert mode in ("train", "eval"), "Unknown mode (only 'train' and 'eval')."
        
        self.hnet.train(mode == "train")
        self.solver.train(mode == "train")

        for child in self.children:
            child.toggle_mode(mode=mode)
    
    def zero_grad(self):
        assert self.hnet_theta_optim is not None and self.task_emb_optims is not None, \
            "Cell's optimizers not fully initialized"

        self.hnet_theta_optim.zero_grad()
        for task_emb_optim in self.task_emb_optims:
            task_emb_optim.zero_grad()

        for child in self.children:
            child.zero_grad()
    
    def step(self, task_i=None):
        assert self.hnet_theta_optim is not None and self.task_emb_optims is not None, \
            "Cell's optimizers not fully initialized"

        self.hnet_theta_optim.step()
        if task_i is not None: # step only a single task embedding optimizer
            self.task_emb_optims[task_i].step()
        else: # step all task embedding optimizers
            for task_emb_optim in self.task_emb_optims:
                task_emb_optim.step()
        
        for child in self.children:
            child.step(task_i=task_i)
    
    def clip_grads(self):
        if self.config["hnet"]["reg_clip_grads_max_norm"] is not None and self.config["hnet"]["reg_clip_grads_max_value"] is not None:
            print("[WARNING] Both reg_clip_grads_max_norm and reg_clip_grads_max_value are set. Using reg_clip_grads_max_norm.")
        if self.config["hnet"]["reg_clip_grads_max_norm"] is not None:
            torch.nn.utils.clip_grad_norm_(self.hnet.parameters(), self.config["hnet"]["reg_clip_grads_max_norm"])
        elif self.config["hnet"]["reg_clip_grads_max_value"] is not None:
            torch.nn.utils.clip_grad_value_(self.hnet.parameters(), self.config["hnet"]["reg_clip_grads_max_value"])
        
        for child in self.children:
            child.clip_grads()

    def get_available_paths(self, paths=[], curr_path=[]):
        paths.append(curr_path)
        if len(self.children) > 0:
            for ch_i in range(len(self.children)):
                paths = self.children[ch_i].get_available_paths(paths=paths, curr_path=curr_path + [ch_i])
        return paths

    def _init_params(self, method="xavier", std_normal_init_params=0.02, std_normal_init_task_embs=1.0, std_normal_init_chunk_embs=1.0):
        if self.config is not None and "hnet" in self.config.keys() and "init" in self.config["hnet"].keys():
            if "method" in self.config["hnet"]["init"].keys():
                method = self.config["hnet"]["init"]["method"]
            if "std_normal_init_params" in self.config["hnet"]["init"].keys():
                std_normal_init_params = self.config["hnet"]["init"]["std_normal_init_params"]
            if "std_normal_init_task_embs" in self.config["hnet"]["init"].keys():
                std_normal_init_task_embs = self.config["hnet"]["init"]["std_normal_init_task_embs"]
            if "std_normal_init_chunk_embs" in self.config["hnet"]["init"].keys():
                std_normal_init_chunk_embs = self.config["hnet"]["init"]["std_normal_init_chunk_embs"]

        ### initialize the weights and biases of the network
        for w in self.hnet._layer_weight_tensors:
            if method == "xavier":
                torch.nn.init.xavier_uniform_(w)
            elif method == "normal":
                torch.nn.init.normal_(w, mean=0., std=std_normal_init_params)
            else:
                raise ValueError(f"Unknown initialization method: {method}")
        
        # biases are initialized to 0
        for b in self.hnet._layer_bias_vectors:
            torch.nn.init.constant_(b, 0)
        
        # task embeddings
        for emb in self.hnet.conditional_params[:self.hnet.num_known_conds]:
            torch.nn.init.normal_(emb, mean=0., std=std_normal_init_task_embs)

        # chunk embeddings
        for emb in self.hnet.conditional_params[self.hnet.num_known_conds:]:
            torch.nn.init.normal_(emb, mean=0., std=std_normal_init_chunk_embs)

    @staticmethod
    def correct_param_shapes(av_params, target_shapes):
        """Correct the shapes of the parameters"""
        if type(av_params) in (list, tuple):
            av_params_flattened = []
            for p in av_params:
                av_params_flattened.append(p.flatten())
            av_params = torch.cat(av_params_flattened)
        elif isinstance(av_params, torch.Tensor):
            av_params = av_params.flatten()
        else:
            raise ValueError("av_params must be a list, tuple or a torch.Tensor")

        params_out = []
        start_idx = 0
        for s in target_shapes:
            n_vals = int(np.prod(s))
            p = av_params[start_idx:start_idx + n_vals].view(*s)
            params_out.append(p)
            start_idx += n_vals

        return params_out