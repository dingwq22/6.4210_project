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

def get_mountain_yaml(obstables):
    path = find_project_path()
    mountains = ''
    i = 0
    for (r,c) in obstables:
        i+=1
      
        x = 2*c-6
        y = -2*r+6
        path = find_project_path()
        mountains+= f'''
- add_model:
    name: mountain{i}
    file: file://{path}/objects/mountain_OBJ.sdf
- add_weld:
    parent: world
    child: mountain{i}::mountain_OBJ
    X_PC:
        translation: [{x}, {y}, 0]       
        '''
    return mountains


def build_scenario_data(obstacles, base_pos):
    path = find_project_path()
    print(path)
    degrees = '{ deg: [90, 0, 90]}'
    driver1 = '!InverseDynamicsDriver {}'
    driver2 = '!SchunkWsgDriver {}'
    degstr = '{ deg: [0.0, 0.0, 180.0 ]}'
    camera0deg = '{ deg: [-125., 0, 130]}' #[45, -125, 0]
    camera1deg = '{ deg: [-125., 0, 0]}'
    camera2deg = '{ deg: [-125.0, 0, -130]}'
    scenario_data = f"""
directives:
- add_model:
    name: iiwa
    file: package://manipulation/mobile_iiwa14_primitive_collision.urdf
    default_joint_positions:
        iiwa_joint_1: [0.81737796]
        iiwa_joint_2: [1.26516111]
        iiwa_joint_3: [0.28494653]
        iiwa_joint_4: [-1.38298128]
        iiwa_joint_5: [0.85251895]
        iiwa_joint_6: [-1.36210458]
        iiwa_joint_7: [0.98480138]
        # iiwa_joint_1: [-1.57]
        # iiwa_joint_2: [0.1]
        # iiwa_joint_3: [0]
        # iiwa_joint_4: [-1.2]
        # iiwa_joint_5: [0]
        # iiwa_joint_6: [ 1.6]
        # iiwa_joint_7: [0]
        iiwa_base_x: [-4]
        iiwa_base_y: [4]
        iiwa_base_z: [0]


- add_model:
    name: wsg
    file: package://drake/manipulation/models/wsg_50_description/sdf/schunk_wsg_50_no_tip.sdf
- add_weld:
    parent: iiwa::iiwa_link_7
    child: wsg::body
    X_PC:
        translation: [0, 0, 0.114]
        rotation: !Rpy {degrees}
- add_model:
    name: ground
    file: file://{path}/objects/ground.sdf
- add_weld:
    parent: world
    child: ground::base
    X_PC:
        translation: [0, 0, 0]
# - add_model:
#     name: rock2
#     file: file://{path}/objects/Cliff_Rock_One_OBJ.sdf
# - add_weld:
#     parent: world
#     child: rock2::Cliff_Rock_One_OBJ
#     X_PC:
#         # translation: [-1, -1, 0]
#         translation: [-1.97, -2.43, 0.01]
- add_model:
    name: bin1
    file: package://manipulation/hydro/bin.sdf
- add_weld:
    parent: iiwa::iiwa_link_0
    child: bin1::bin_base
    X_PC:
        translation: [-.5, 0, 0]
"""
    # add mountains
    scenario_data += get_mountain_yaml(obstacles)
    #add the cylinder & 3 mountains near it
    for i, b in enumerate(base_pos):
        scenario_data += f'''
- add_model:
    name: object{i}
    file: file://{path}/objects/obstacle_boxes.sdf
    default_free_body_pose:
        obstacles:
            translation: [{b[0]}, {b[1]}, 0.01]
- add_frame:
    name: camera{3*i}_origin
    X_PF:
        base_frame: world
        rotation: !Rpy {camera0deg}
        translation: [{b[0]+0.5}, {b[1]+0.5}, .5]

- add_model:
    name: camera{3*i}
    file: package://manipulation/camera_box.sdf

- add_weld:
    parent: camera{3*i}_origin
    child: camera{3*i}::base

- add_frame:
    name: camera{3*i+1}_origin
    X_PF:
        base_frame: world
        rotation: !Rpy {camera1deg}
        translation: [{b[0]}, {b[1]-0.6}, .5] 

- add_model:
    name: camera{3*i+1}
    file: package://manipulation/camera_box.sdf

- add_weld:
    parent: camera{3*i+1}_origin
    child: camera{3*i+1}::base

- add_frame:
    name: camera{3*i+2}_origin
    X_PF:
        base_frame: world
        rotation: !Rpy {camera2deg}
        translation: [{b[0]-0.5}, {b[1]+0.5}, .5]

- add_model:
    name: camera{3*i+2}
    file: package://manipulation/camera_box.sdf

- add_weld:
    parent: camera{3*i+2}_origin
    child: camera{3*i+2}::base
    '''
    # cameras section
    scenario_data += f'''
cameras:
'''
    for i in range(len(base_pos)*3):
       scenario_data += f'''
    camera{i}:
      name: camera{i}
      depth: True
      X_PB:
        base_frame: camera{i}::base
     '''
    # model_drivers section
    scenario_data += f'''
model_drivers:
    iiwa: {driver1}
    wsg: {driver2}
    '''
    return scenario_data
