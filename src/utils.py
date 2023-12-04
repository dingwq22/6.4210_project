import numpy as np
import pydot
from pydrake.all import (
    ExternallyAppliedSpatialForce,
    GeometrySet,
    CollisionFilterDeclaration
)
from IPython.display import display, Image
import os

import numpy as np
from pydrake.all import (
    ConstantVectorSource,
    DiagramBuilder,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    MultibodyPlant,
    Parser,
    PiecewisePolynomial,
    PiecewiseQuaternionSlerp,
    RigidTransform,
    RollPitchYaw,
    RotationMatrix,
    Simulator,
    Solve,
    StartMeshcat,
    TrajectorySource,
)
from pydrake.multibody import inverse_kinematics
from pydrake.trajectories import PiecewisePolynomial

from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.scenarios import AddMultibodyTriad, MakeManipulationStation
from manipulation.utils import running_as_notebook

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

def find_project_path():
    current_file_path = os.path.abspath(__file__)
    parent_dir = os.path.dirname(current_file_path)
    return os.path.dirname(parent_dir)

def notebook_plot_diagram_svg(diagram):
    SVG(
    pydot.graph_from_dot_data(diagram.GetGraphvizString(max_depth=1))[
        0
    ].create_svg()
)
