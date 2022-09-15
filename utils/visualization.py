import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from torchviz import make_dot


def get_summary_plots(metrics, max_cols=3, fig_w=14, fig_h_mul=3):
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
            axes[p_i].bar(task_names, accs)
            axes[p_i].set_title(f"{path} - acc")
            axes[p_i].set_ylim(0, 100)
        else:
            axes[p_i // n_cols][p_i % n_cols].bar(task_names, accs)
            axes[p_i // n_cols][p_i % n_cols].set_title(f"{path} - acc")
            axes[p_i // n_cols][p_i % n_cols].set_ylim(0, 100)
        accs_all_paths.append(accs)

    ### mean accuracies across all paths - TODO: assuming all paths have the same tasks
    mean_accs = np.mean(accs_all_paths, axis=0)
    if n_rows == 1:
        axes[-1].bar(task_names, mean_accs)
        axes[-1].set_title("mean acc")
        axes[-1].set_ylim(0, 100)
    else:
        axes[-1][-1].bar(task_names, mean_accs)
        axes[-1][-1].set_title("mean acc")
        axes[-1][-1].set_ylim(0, 100)

    fig.tight_layout()
    return fig, axes


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
