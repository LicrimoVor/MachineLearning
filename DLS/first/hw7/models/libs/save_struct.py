import os
from pathlib import Path

import torch
from torchview import draw_graph
from torchviz import make_dot

os.environ["PATH"] += os.pathsep + "C:\\Program Files\\Graphviz\\bin\\"
BASE_PATH = Path(__file__).parent.parent.joinpath("assets")


def save_struct(model: torch.nn.Module, batch: torch.Tensor, filename: str, view_viz=False):

    if not view_viz:
        model_graph = draw_graph(model, batch[0], depth=3, expand_nested=True)
        model_graph.visual_graph.render(BASE_PATH.joinpath(filename), format="pdf")
        model_graph.visual_graph.save(filename + "_graph", Path(__file__).parent)
    else:
        make_dot(model(batch), params=dict(model.named_parameters())).render(
            BASE_PATH.joinpath(filename), format="pdf"
        )
