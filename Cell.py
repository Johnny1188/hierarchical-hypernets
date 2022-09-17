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
        self.cond_id_mapping = None
        self.saved_solver_acts = [] # debugging
        self._init_params()

        if init_hnet_optim is True:
            self.reinit_uncond_params_optim()
            self.reinit_cond_embs_optims()

    def save_tree(self, curr_path, dict_to_save, path_to_checkpoint_file, is_root=True):
        dict_to_save[f"{''.join([str(s) for s in curr_path])}_state_dict"] = self.state_dict()
        dict_to_save[f"{''.join([str(s) for s in curr_path])}_additionals"] = self.dump_cell_additionals()

        for c_idx, child_cell in enumerate(self.children):
            child_cell.save_tree(curr_path=curr_path + [c_idx], dict_to_save=dict_to_save, path_to_checkpoint_file=None, is_root=False)

        if is_root:
            torch.save(dict_to_save, path_to_checkpoint_file)

    def load_tree(self, curr_path, check_dict, path_to_checkpoint_file):
        if check_dict is None:
            check_dict = torch.load(path_to_checkpoint_file)

        path_key = ''.join([str(s) for s in curr_path])
        self.load_state_dict(check_dict[f"{path_key}_state_dict"])
        self.load_additionals(check_dict[f"{path_key}_additionals"])
        del check_dict[f"{path_key}_state_dict"]
        del check_dict[f"{path_key}_additionals"]

        for c_idx, child_cell in enumerate(self.children):
            child_cell.load_tree(curr_path=curr_path + [c_idx], check_dict=check_dict, path_to_checkpoint_file=None)

        return check_dict

    def dump_cell_additionals(self):
        c_adds = {}
        c_adds["num_tasks"] = self.num_tasks
        c_adds["needs_theta_type"] = self.needs_theta_type
        c_adds["config"] = self.config
        c_adds["device"] = self.device
        c_adds["cond_id_mapping"] = self.cond_id_mapping
        return c_adds

    def load_additionals(self, cell_additionals):
        self.num_tasks = cell_additionals["num_tasks"]
        self.needs_theta_type = cell_additionals["needs_theta_type"]
        self.config = cell_additionals["config"]
        self.device = cell_additionals["device"]
        self.cond_id_mapping = cell_additionals["cond_id_mapping"]
        return self

    def init_cond_id_mapping(self, init_children=True):
        if init_children is True:
            for child in self.children:
                child.init_cond_id_mapping(init_children=True)
        
        av_paths = self.get_available_paths([], [])
        self.cond_id_mapping = {}

        starting_cond_id = 0
        for path in av_paths:
            last_child_n_tasks = self.get_child_at_path(path).num_tasks
            self.cond_id_mapping[tuple(path)] = list(range(starting_cond_id, starting_cond_id + last_child_n_tasks))
            starting_cond_id += last_child_n_tasks

    def get_child_at_path(self, path):
        if len(path) == 0:
            return self
        return self.children[path[0]].get_child_at_path(path[1:])

    def reinit_uncond_params_optim(self):
        ### note: unconditional params include chunk embeddings when they are not conditional (self.hnet.cond_chunk_embs == False)
        if self.hnet.unconditional_params is not None:
            self.hnet_theta_optim = get_optimizer(
                params=self.hnet.unconditional_params,
                lr=self.config["hnet"]["lr"],
                beta_1=self.config["hnet"]["adam_beta_1"],
                beta_2=self.config["hnet"]["adam_beta_2"],
                weight_decay=self.config["hnet"]["weight_decay"],
            )
        else:
            self.hnet_theta_optim = None
        
        for child in self.children:
            child.reinit_uncond_params_optim()
    
    def reinit_cond_embs_optims(self, path=[], task_i=None, reinit_all_children=False):
        if self.hnet.conditional_params is None: # no conditional params
            self.task_emb_optims = None
        else:
            if task_i is not None and self.task_emb_optims is not None:
                ### reinitialize only a single path---task/context embedding optimizer
                cond_id = self.get_cond_id(task_i, path)
                params = [self.hnet.conditional_params[cond_id]]
                if self.hnet.cond_chunk_embs is True: # chunk embeddings may also be conditional
                    params.append(self.hnet.get_chunk_emb(cond_id=cond_id))
                self.task_emb_optims[cond_id] = get_optimizer(
                    params=params,
                    lr=self.config["hnet"]["lr"],
                    beta_1=self.config["hnet"]["adam_beta_1"],
                    beta_2=self.config["hnet"]["adam_beta_2"],
                    weight_decay=self.config["hnet"]["weight_decay"],
                )
            else:
                ### reinitialize all conditional embedding optimizers
                params_per_optim = []
                for cond_id in range(self.hnet.num_known_conds):
                    params = [self.hnet.conditional_params[cond_id]]
                    if self.hnet.cond_chunk_embs is True: # chunk embeddings may also be conditional
                        params.append(self.hnet.get_chunk_emb(cond_id=cond_id))
                    params_per_optim.append(params)
                self.task_emb_optims = [
                    get_optimizer(
                        params=params_for_cond,
                        lr=self.config["hnet"]["lr"],
                        beta_1=self.config["hnet"]["adam_beta_1"],
                        beta_2=self.config["hnet"]["adam_beta_2"],
                        weight_decay=self.config["hnet"]["weight_decay"],
                    ) for params_for_cond in params_per_optim
                ]

        if reinit_all_children is True: # reinit all children
            for child in self.children:
                child.reinit_cond_embs_optims(path=[], task_i=None, reinit_all_children=True)
        elif len(path) > 0:
            self.children[path[0]].reinit_cond_embs_optims(path=path[1:], task_i=task_i, reinit_all_children=reinit_all_children)

    def forward(self, x, task_i, path, theta_hnet=None, path_trace={"cond_ids": []}, save_last_fc_acts=False):
        assert type(path) == list, "path must be a list of integers"
        
        cond_id = self.get_cond_id(task_i, path)
        
        if len(path) == 0: # hnet -> solver
            ### hypernet -> solver
            theta_solver = self.hnet.forward(cond_id=cond_id, weights=theta_hnet) # hnet -> solver
            out = self.solver.forward(x, weights=Cell.correct_param_shapes(theta_solver, self.solver.param_shapes), save_last_fc_acts=save_last_fc_acts)
            # save activations for debugging
            if type(out) in (list, tuple) and len(out) == 2:
                y_hat, self.saved_solver_acts = out
            else:
                y_hat, self.saved_solver_acts = out, []

            ### get task head indices
            task_head_idxs = self.get_task_head_idxs(task_i)
            y_hat = y_hat[:, task_head_idxs[0]:task_head_idxs[1]]

            ### save path trace to this cell
            path_trace["solving_cell"] = self
            path_trace["cond_ids"].append(cond_id)
        else: # hnet -> hnet
            ### generate params (theta) for child cell's hypernet
            child_idx = path[0]
            theta_child_hnet = self.hnet.forward(cond_id=cond_id, weights=theta_hnet)
            path_trace["cond_ids"].append(cond_id)
            
            theta_child_hnet = self._package_child_theta(theta_child_hnet, child_idx)

            y_hat, theta_solver, path_trace = self.children[child_idx](
                x, task_i=task_i, path=path[1:], theta_hnet=theta_child_hnet,
                path_trace=path_trace, save_last_fc_acts=save_last_fc_acts
            )
        
        return y_hat, theta_solver, path_trace

    def _package_child_theta(self, theta_child_hnet, child_idx):
        if self.children[child_idx].needs_theta_type == "uncond_weights":
            theta_child_hnet = {
                "uncond_weights": Cell.correct_param_shapes(theta_child_hnet, self.children[child_idx].hnet.unconditional_param_shapes)
            }
        elif self.children[child_idx].needs_theta_type == "cond_weights":
            theta_child_hnet = {
                "cond_weights": Cell.correct_param_shapes(theta_child_hnet, self.children[child_idx].hnet.conditional_param_shapes)
            }
        elif self.children[child_idx].needs_theta_type == "all":
            theta_child_hnet = Cell.correct_param_shapes(theta_child_hnet, self.children[child_idx].hnet.param_shapes)
        elif self.children[child_idx].needs_theta_type == "none":
            theta_child_hnet = None
        else:
            raise RuntimeError("Child cell doesn't have a valid needs_theta_type property")
        return theta_child_hnet

    def get_task_head_idxs(self, task_i):
        """ Get solver's head indices for a given task """
        if "task_heads" not in self.config["solver"] or self.config["solver"]["task_heads"] is None:
            return [0, self.solver.num_classes]
        
        return [self.config["solver"]["task_heads"][task_i][0], self.config["solver"]["task_heads"][task_i][1]]

    def get_cond_id(self, task_i, path):
        return self.cond_id_mapping[tuple(path)][task_i]

    def toggle_mode(self, mode):
        assert mode in ("train", "eval"), "Unknown mode (only 'train' and 'eval')."
        
        self.hnet.train(mode == "train")
        self.solver.train(mode == "train")

        for child in self.children:
            child.toggle_mode(mode=mode)
    
    def zero_grad(self):
        if self.hnet_theta_optim is not None:
            self.hnet_theta_optim.zero_grad()
        if self.task_emb_optims is not None:
            for task_emb_optim in self.task_emb_optims: # TODO: step only with the one for the current task
                task_emb_optim.zero_grad()

        for child in self.children:
            child.zero_grad()
    
    def step(self, path, task_i=None):
        assert type(path) == list, "path must be a list of integers"

        ### step hnet's unconditional parameters (theta - everything except task embeddings)
        if self.hnet_theta_optim is not None:
            self.hnet_theta_optim.step()
            
        if self.task_emb_optims is not None:
            if task_i is not None: # step only a single task embedding optimizer
                self.task_emb_optims[self.get_cond_id(task_i, path)].step()
            else: # step with all task embedding optimizers
                for task_emb_optim in self.task_emb_optims:
                    task_emb_optim.step()
        
        if len(path) > 0:
            self.children[path[0]].step(path=path[1:], task_i=task_i)
    
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
        
        # check if the shapes are already correct
        if [list(p.shape) for p in av_params] == target_shapes:
            return av_params
        
        if type(av_params) in (list, tuple):
            # flatten available params
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
