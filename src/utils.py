import numpy as np
import pydot
from pydrake.all import (
    ExternallyAppliedSpatialForce,
    GeometrySet,
    CollisionFilterDeclaration
)
from IPython.display import display, Image

def notebook_plot_plant(plant):
    display(
        Image(
            pydot.graph_from_dot_data(plant.GetTopologyGraphvizString())[0].create_png()
        )
    )

def notebook_plot_diagram(diagram):
    display(
        Image(
            pydot.graph_from_dot_data(diagram.GetGraphvizString())[0].create_png()
        )
    )