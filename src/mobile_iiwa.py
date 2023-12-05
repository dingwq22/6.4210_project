import numpy as np
from pydrake.all import (
    ConstantVectorSource,
    InverseDynamicsController,
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    PiecewisePolynomial,
    Parser,
    MeshcatVisualizer,
    StartMeshcat,
    MeshcatVisualizerParams,
    Simulator,
    JointSliders,
    InverseKinematics,
    RotationMatrix,
    Solve,
    ContactVisualizerParams,
    ContactVisualizer,
    GeometrySet,
    CollisionFilterDeclaration,
    Role,
    eq,
    RigidTransform,
    InverseDynamicsController,
    TrajectorySource,
)
from manipulation.scenarios import AddMultibodyTriad, MakeManipulationStation
from manipulation.station import MakeHardwareStation
from manipulation import ConfigureParser, running_as_notebook
# from manipulation.scenarios import AddMultibodyTriad, MakeManipulationStation
from manipulation.station import MakeHardwareStation, load_scenario

from utils import find_project_path, notebook_plot_diagram
from convert_pose import create_q_knots
from mountain_building import get_mountain_yaml
from trajectory_planning import trajectory_plan

import pydot
from IPython.display import display, Image, SVG
import math
import os


def setup_hardware_station(meshcat, goal, gripper_poses, obstables = [(1, 4), (1, 5), (2, 2), (2, 3), (4, 1), (4, 2), (4, 4), (5, 1), (5, 2)]):

    builder = DiagramBuilder()
    path = find_project_path()
    print(path)
    degrees = '{ deg: [90, 0, 90]}'
    driver1 = '!InverseDynamicsDriver {}'
    driver2 = '!SchunkWsgDriver {}'
    degstr = '{ deg: [0.0, 0.0, 180.0 ]}'
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

- add_model:
    name: object1
    file: file://{path}/objects/obstacle_boxes.sdf
    default_free_body_pose:
        obstacles:
            translation: [-1.97, -2.43, 0.01]

model_drivers:
    iiwa: {driver1}
    wsg: {driver2}


"""
    # print(scenario_data)
    #add mountains
    # scenario_data += get_mountain_yaml(obstables)
    
    scenario = load_scenario(data=scenario_data)
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat))

    plant = station.GetSubsystemByName("plant")
    print(plant.GetStateNames())
    scene_graph = station.GetSubsystemByName("scene_graph")
    # AddMultibodyTriad(plant.GetFrameByName("body"), scene_graph)

    visualizer = MeshcatVisualizer.AddToBuilder(
        builder,
        station.GetOutputPort("query_object"),
        meshcat,
        MeshcatVisualizerParams(delete_on_initialization_event=False),
    )

    # wsg_position = builder.AddSystem(ConstantVectorSource([0.1]))
    # builder.Connect(
    #     wsg_position.get_output_port(), station.GetInputPort("wsg.position")
    # )

    context = plant.CreateDefaultContext()
    gripper = plant.GetBodyByName("body")

    initial_pose = plant.EvalBodyPoseInWorld(context, gripper)

    traj = trajectory_plan(goal)
    traj_len = len(traj)
    print(traj_len)

    duration = 10
    gripper_traj_len = len(gripper_poses)
    unit_time = duration / (traj_len + gripper_traj_len)
    print("unit_time=", unit_time)

    t_lst = np.linspace(0, duration, traj_len + gripper_traj_len)
    q_poses = np.zeros((20, traj_len + gripper_traj_len))
    # set iiwa to the last traj pos when gripper is moving 
    q_poses[0:3, :]= np.array(traj + [traj[-1]]*gripper_traj_len).T #np.array([[-5.0, -5.0, 0], [-5.0, -5.0, 0], [-5.0, -5.0, 0], [-2.5, -5.0, 0], [0.0, -5.0, 0], [0.0, -5.0, 0], [0.0, -2.5, 0], [0.0, 0.0, 0], [0.0, 2.5, 0], [0.0, 5.0, 0], [0.0, 5.0, 0], [2.5, 5.0, 0], [5.0, 5.0, 0], [5.0, 5.0, 0], [5.0, 5.0, 0]]).T
    # find joint positions from end-effector pose
    joint_pos_lst = create_q_knots(gripper_poses)  # shape=(gripper_traj_len, 7)
    
    for i in range(gripper_traj_len):
        q_poses[3:10, traj_len+i] = joint_pos_lst[i][:7]
    q_traj = PiecewisePolynomial.CubicShapePreserving(t_lst, q_poses)
    q_traj_system = builder.AddSystem(TrajectorySource(q_traj)) 
    print("q_traj:", q_traj)
    
    
    gripper_t_lst = np.array([0.0, unit_time*(traj_len), unit_time*(traj_len+2), duration])
    gripper_knots = np.array([0.8, 0.8, 0.2, 0]).reshape(1, 4)
    #test:
    # gripper_t_lst = np.array([0.0, unit_time*(traj_len), unit_time*(traj_len+1), unit_time*(traj_len+6), duration])
    # gripper_knots = np.array([0, 0, 0, 0, 0]).reshape(1, 5)
    g_traj = PiecewisePolynomial.CubicShapePreserving(gripper_t_lst, gripper_knots)
    #working code:
    # gripper_knots = np.array([0.05, 0.05, 0.0, 0.0]).reshape(1, 4)
    # g_traj = PiecewisePolynomial.FirstOrderHold(gripper_t_lst, gripper_knots)
    g_traj_system = builder.AddSystem(TrajectorySource(g_traj)) 

    builder.Connect(
        q_traj_system.get_output_port(), station.GetInputPort("iiwa.desired_state")
    )
    builder.Connect(
        g_traj_system.get_output_port(), station.GetInputPort("wsg.position")
    )

    diagram = builder.Build()
    simulator = Simulator(diagram)
    visualizer.StartRecording(False)
    
    simulator.AdvanceTo(duration)
    visualizer.PublishRecording()

    return station, plant, scene_graph, diagram



def set_position(meshcat, X_WG, goal = (5,5), max_tries=10, fix_base=False, base_pose=np.zeros(3), gripper_poses=[RigidTransform([0.5, 0.5, 0.1])]):
    # diagram, plant, scene_graph = build_env(meshcat)

    # using hardware station
    plant, station, scene_graph, diagram = setup_hardware_station(meshcat, gripper_poses=gripper_poses, goal=goal)
    return diagram
