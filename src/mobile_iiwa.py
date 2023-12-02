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

from utils import find_project_path, notebook_plot_diagram, create_q_knots
from mountain_building import get_mountain_yaml
import pydot
from IPython.display import display, Image, SVG
import math
import os

def notebook_plot_diagram_svg(diagram):
    return SVG(
    pydot.graph_from_dot_data(diagram.GetGraphvizString(max_depth=1))[
        0
    ].create_svg()
)

def setup_hardware_station(meshcat, final_pose = RigidTransform([0.5, 0.5, 0.1]), obstables = [(1, 4), (1, 5), (2, 2), (2, 3), (4, 1), (4, 2), (4, 4), (5, 1), (5, 2)]):

    builder = DiagramBuilder()
    path = find_project_path()
    print(path)
    degrees = '{ deg: [90, 0, 90]}'
    driver1 = '!InverseDynamicsDriver {}'
    driver2 = '!SchunkWsgDriver {}'
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
        iiwa_base_x: [-5]
        iiwa_base_y: [-5]
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
    name: rock2
    file: file://{path}/objects/Cliff_Rock_One_OBJ.sdf
- add_weld:
    parent: world
    child: rock2::Cliff_Rock_One_OBJ
    X_PC:
        translation: [-1, -1, 0]
model_drivers:
    iiwa: {driver1}
    wsg: {driver2}
"""
    # print(scenario_data)
    #add mountains
    # scenario_data += get_mountain_yaml(obstables)
    
    scenario = load_scenario(data=scenario_data)
    # station = builder.AddSystem(MfakeManipulationStation(filename="file:///workspaces/6.4210_project/drake_obstacles_nopkg.dmd.yaml"))
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat))

    plant = station.GetSubsystemByName("plant")
    print(plant.GetStateNames())
    scene_graph = station.GetSubsystemByName("scene_graph")
    AddMultibodyTriad(plant.GetFrameByName("body"), scene_graph)

    visualizer = MeshcatVisualizer.AddToBuilder(
        builder,
        station.GetOutputPort("query_object"),
        meshcat,
        MeshcatVisualizerParams(delete_on_initialization_event=False),
    )

    wsg_position = builder.AddSystem(ConstantVectorSource([0.1]))
    # builder.Connect(
    #     wsg_position.get_output_port(), station.GetInputPort("wsg_position")
    # )

    context = plant.CreateDefaultContext()
    gripper = plant.GetBodyByName("body")

    initial_pose = plant.EvalBodyPoseInWorld(context, gripper)

    t_lst = np.linspace(0, 10, 15)
    q_poses = np.zeros((20,15))
    q_poses[0:3, :]= np.array([[-5.0, -5.0, 0], [-5.0, -5.0, 0], [-5.0, -5.0, 0], [-2.5, -5.0, 0], [0.0, -5.0, 0], [0.0, -5.0, 0], [0.0, -2.5, 0], [0.0, 0.0, 0], [0.0, 2.5, 0], [0.0, 5.0, 0], [0.0, 5.0, 0], [2.5, 5.0, 0], [5.0, 5.0, 0], [5.0, 5.0, 0], [5.0, 5.0, 0]]).T
    #joint pose trial
    q_final = create_q_knots([final_pose])[0][:7]

    for i in [-1, -2]:
        q_poses[3:10, i] = q_final #np.array([0.07080026,  1.47270634,  0.77810903, -1.28555459,  1.2570383 , 0.79680724,  0.4843345])
    q_traj = PiecewisePolynomial.CubicShapePreserving(t_lst, q_poses)
    q_traj_system = builder.AddSystem(TrajectorySource(q_traj)) 
    
    gripper_t_lst = np.array([0.0, 5.0, 6.0, 10.0])
    #test:
    gripper_knots = np.array([0.0, 0.0, 0.0, 0.0]).reshape(1, 4)
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
    # notebook_plot_diagram(diagram)
    simulator = Simulator(diagram)
    visualizer.StartRecording(False)
    duration = 10
    simulator.AdvanceTo(duration)
    visualizer.PublishRecording()

    # q_traj_system = builder.AddSystem(TrajectorySource(q_traj)) 

    # simulator = Simulator(diagram)
    # simulator.set_target_realtime_rate(1.0)
    # simulator.AdvanceTo(0.01)
    # context = station.CreateDefaultContext()

    # # station.GetInputPort("wsg.position").FixValue(context, [0.1])
    # station.ForcedPublish(context)
    # # print(station.get_input_port(1))
   
    # plant = station.GetSubsystemByName("plant")

    print('plotting')
    
    
    # print(notebook_plot_diagram_svg(diagram))
    return station, plant, scene_graph, diagram



def set_position(meshcat, X_WG, max_tries=10, fix_base=False, base_pose=np.zeros(3), final_pose = RigidTransform([0.5, 0.5, 0.1])):
    # diagram, plant, scene_graph = build_env(meshcat)

    # using hardware station
    plant, station, scene_graph, diagram = setup_hardware_station(meshcat, final_pose =  final_pose)
    return diagram
