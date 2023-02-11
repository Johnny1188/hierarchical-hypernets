import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from torchviz import make_dot


def get_summary_plots(metrics, max_cols=2, fig_w=16, fig_h_mul=4, with_baselines=False):
    ### create plots of individual accuracies
    n_rows = math.ceil((len(metrics.keys()) + 1) / max_cols)
    n_cols = min(max_cols, (len(metrics.keys()) + 1))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, n_rows * fig_h_mul))
    
    accs_all_paths = []
    task_names = []
    for p_i, (path, tasks) in enumerate(metrics.items()):
        accs = [task_metrics["acc"] for task_metrics in tasks.values()]
        task_names = list(tasks.keys())
        if n_rows == 1:
            axis = axes[p_i]
        else:
            axis = axes[p_i // n_cols, p_i % n_cols]
            # axes[p_i // n_cols][p_i % n_cols].bar(task_names, accs)
            # axes[p_i // n_cols][p_i % n_cols].set_title(f"{path} - acc")
            # axes[p_i // n_cols][p_i % n_cols].set_ylim(0, 100)
        axis.bar(task_names, accs)
        axis.set_title(f"{path} - acc", fontweight="bold")
        axis.set_xlabel("task", fontweight="bold")
        axis.set_ylabel("accuracy", fontweight="bold")
        axis.set_ylim(0, 100)
        
        accs_all_paths.append(accs)

    ### mean accuracies across all paths - TODO: assuming all paths have the same tasks
    mean_accs = list(np.mean(accs_all_paths, axis=0))
    if n_rows == 1:
        if with_baselines:
            axes[-1] = get_final_w_baselines(axes[-1], task_names, mean_accs)
        else:
            axes[-1].bar(task_names, mean_accs)
        axes[-1].set_title("Final result")
        axes[-1].set_ylim(0, 100)
    else:
        if with_baselines:
            axes[-1][-1] = get_final_w_baselines(axes[-1][-1], task_names, mean_accs)
        else:
            axes[-1][-1].bar(task_names, mean_accs)
        axes[-1][-1].set_title("Final result", fontweight="bold")
        axes[-1][-1].set_ylim(0, 100)

    fig.tight_layout()
    return fig, axes


def get_final_w_baselines(axis, task_names, mean_accs, bar_width=0.25):
    assert len(task_names) == len(mean_accs), "Task names and mean accuracies must have the same length."
    assert len(task_names) == 6, "Baselines only available for SplitCIFAR100 having 6 tasks."

    # baselines from Johannes von Oswald et. al. (2019): https://arxiv.org/abs/1906.00695
    to_plot = {
        "multi-hnet": mean_accs,
        "single-hnet": [73, 69, 66, 74, 69, 75],
        "from-scratch": [77, 63, 57, 69, 58, 65],
        "synaptic-intelligence": [74, 72, 71, 76, 70, 77],
    }

    # positions on the x-axis
    # x_pos = np.arange(start=0, stop=len(mean_accs), step=bar_width)
    init_x_pos = np.array([x_i * (bar_width * len(to_plot) + 0.3) for x_i in range(len(task_names))])
    x_pos = init_x_pos.copy()

    for name, results in to_plot.items():
        rects = axis.bar(x_pos, results, width=bar_width, label=name)
        axis.bar_label(rects, padding=3)
        x_pos = x_pos + bar_width

    # Add xticks on the middle of the group bars
    axis.set_xlabel("task", fontweight="bold")
    axis.set_ylabel("accuracy", fontweight="bold")
    axis.set_xticks([x_pos + bar_width * (len(to_plot) // 2) for x_pos in init_x_pos], task_names)
    
    # Create legend & Show graphic
    axis.legend(loc="lower center", ncol=len(task_names), bbox_to_anchor=(0.5, -0.3))
    return axis


def resize_dot_graph(dot, size_per_element=0.15, min_size=12):
    """Resize the graph according to how much content it contains.
    Modify the graph in place.

    Author: ucalyptus (https://github.com/ucalyptus): https://github.com/szagoruyko/pytorchviz/issues/41#issuecomment-699061964
    """
    # Get the approximate number of nodes and edges
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)
    return dot


def get_model_dot(model, model_out, show_detailed_grad_info=True, output_filepath=None):
    if show_detailed_grad_info:
        dot = make_dot(model_out, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
    else:
        dot = make_dot(model_out, params=dict(model.named_parameters()))
    resize_dot_graph(dot, size_per_element=1, min_size=20)

    if output_filepath:
        dot.format = "png"
        dot.render(output_filepath)

    return dot


def show_imgs(imgs, titles=None):
    if len(imgs.shape) == 1: 
        imgs = imgs.reshape(1, int(math.sqrt(imgs.shape[0])), int(math.sqrt(imgs.shape[0]))) # flattened img -> square img
    if len(imgs.shape) == 2:
        imgs = imgs.reshape(imgs.shape[0], 1, int(math.sqrt(imgs.shape[1])), int(math.sqrt(imgs.shape[1]))) # flattened imgs -> square img
    
    fig, axes = plt.subplots(
        ((imgs.shape[0]-1) // 5) + 1, 5 if (len(imgs.shape) == 4 and imgs.shape[0] > 1) else 1,
        squeeze=False,
        figsize=(20, 4 * ((imgs.shape[0]-1) // 5 + 1))
    )
    curr_img_i = 0
    curr_img = imgs if len(imgs.shape) == 3 else imgs[curr_img_i]
    while curr_img != None:
        axes[curr_img_i // 5, curr_img_i % 5].imshow(np.transpose(curr_img, (1, 2, 0)))
        if type(titles) == str: axes[curr_img_i // 5, curr_img_i % 5].set_title(titles)
        if type(titles) in (list, tuple, set, torch.Tensor, np.ndarray): axes[curr_img_i // 5, curr_img_i % 5].set_title(titles[curr_img_i])
        curr_img_i += 1
        curr_img = None if (len(imgs.shape) == 3 or curr_img_i >= imgs.shape[0]) else imgs[curr_img_i]
    plt.show()


def get_umap_emb(x, n_umap_components=3):
    assert n_umap_components in (2, 3), "Use either 2 or 3 UMAP components (2D or 3D visualization)"
    import umap.umap_ as umap
    mapper = umap.UMAP(random_state=42, n_components=n_umap_components).fit(x)
    embedding = mapper.transform(x)
    return embedding


def scatter(x, y, dim=3, width=1000, height=600, color_label="digit", marker_size=2):
    assert dim in (2, 3), "Scatter plot supports only 2D and 3D."
    import plotly.express as px
    if dim == 2:
        fig = px.scatter(
            x,
            x=0, y=1,
            color=y,
            labels={'color': color_label},
            width=width,
            height=height
        )
    elif dim == 3:
        fig = px.scatter_3d(
            x, 
            x=0, y=1, z=2,
            color=y, 
            labels={"color": color_label},
            width=width,
            height=height
        )
    fig.update_traces(marker_size=marker_size)
    fig.show()
