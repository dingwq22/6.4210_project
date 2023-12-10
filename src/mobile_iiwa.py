import numpy as np
from pydrake.all import (
    AbstractValue,
    PointCloud,
    LeafSystem,
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
    RigidTransform,
    InverseDynamicsController,
    TrajectorySource,
)
from manipulation.scenarios import AddMultibodyTriad
from manipulation.station import (
    AddPointClouds,
    MakeHardwareStation,
    add_directives,
    load_scenario,
)
from manipulation import ConfigureParser, running_as_notebook
from manipulation.station import MakeHardwareStation, load_scenario

from utils import find_project_path, notebook_plot_diagram, build_scenario_data
from convert_pose import create_q_knots
from mountain_building import get_mountain_yaml
from trajectory_planning import trajectory_plan
from perception import grasp_selection, make_internal_model

import pydot
from IPython.display import display, Image, SVG

def plan_trajectory():
    # TODO: move trajectory code here 
    return 

def setup_hardware_station(meshcat, base_pos, gripper_poses, obstacles = [(1, 4), (1, 5), (2, 2), (2, 3), (4, 1), (4, 2), (4, 4), (5, 1), (5, 2)]):
    builder = DiagramBuilder()
    scenario_data = build_scenario_data(obstacles)
    scenario = load_scenario(data=scenario_data)
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat))

    plant = station.GetSubsystemByName("plant")
    # print('plant states', plant.GetStateNames())
    scene_graph = station.GetSubsystemByName("scene_graph")

    AddMultibodyTriad(plant.GetFrameByName("camera0_origin"), scene_graph)

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

    ### camera
    to_point_cloud = AddPointClouds(
        scenario=scenario, station=station, builder=builder
    )

    #export
    for i in range(3):
        builder.ExportOutput(
            to_point_cloud[f"camera{i}"].get_output_port(), f"camera{i}_point_cloud"
        )
        builder.ExportOutput(
            station.GetOutputPort(f"camera{i}.rgb_image"), f"camera{i}_rgb_image"
        )
        builder.ExportOutput(
            station.GetOutputPort(f"camera{i}.depth_image"), f"camera{i}_depth_image"
        )

    initial_pose = plant.EvalBodyPoseInWorld(context, gripper)

    # # find antipodal grasps
    # TODO: fix builder issue. Specifically, using builder.Build() in grasp_selection will cause "Build() called twice error" 
    # cost, X_G = grasp_selection(builder, plant, station, meshcat)

    # trajectory plan
    x, y = base_pos
    goal = (-0.55 + x, -0.07 + y)
    traj = trajectory_plan(goal, obstacles = obstacles)
    traj_len = len(traj)
    

    duration = 10
    gripper_traj_len = len(gripper_poses)
    unit_time = duration / (traj_len + gripper_traj_len)
    print("trag_len:", traj_len)
    print('gripper_traj_len:', gripper_traj_len)
    print("unit_time:", unit_time)

    t_lst = np.linspace(0, duration, traj_len + gripper_traj_len)
    q_poses = np.zeros((20, traj_len + gripper_traj_len))
    # set iiwa to the last traj pos when gripper is moving 
    q_poses[0:3, :]= np.array(traj + [traj[-1]]*gripper_traj_len).T 
    
    # find joint positions from end-effector pose
    joint_pos_lst = [np.zeros(9) for _ in range(gripper_traj_len)]
    ######## to do - convert back[hard coding for debugging speed] ######3
    # joint_pos_lst = create_q_knots(gripper_poses)  # shape=(gripper_traj_len, 7)
    print('joint_pos_lst', joint_pos_lst)
    
    # iiwa joints trajectory 
    for i in range(gripper_traj_len):
        q_poses[3:10, traj_len+i] = joint_pos_lst[i][:7]
    print('q poses', q_poses[3:10, traj_len+1:])
    q_traj = PiecewisePolynomial.CubicShapePreserving(t_lst, q_poses)
    q_traj_system = builder.AddSystem(TrajectorySource(q_traj)) 
     
    # gripper trajectory 
    gripper_close_t = unit_time*(traj_len+2)
    gripper_open_t = duration - unit_time*1
    gripper_t_lst = np.array([0.0, unit_time*(traj_len), gripper_close_t,  gripper_open_t, duration])
    gripper_knots = np.array([1, 1, 0.1, 0, 1]).reshape(1, len(gripper_t_lst))
    g_traj = PiecewisePolynomial.CubicShapePreserving(gripper_t_lst, gripper_knots)
    g_traj_system = builder.AddSystem(TrajectorySource(g_traj)) 

    # duration = 10
    # q_traj_system = builder.AddSystem(ConstantVectorSource([0]*20))
    # g_traj_system = builder.AddSystem(ConstantVectorSource([0.1]))
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



def set_position(meshcat, base_pos=(0,0), gripper_poses=[RigidTransform([0.5, 0.5, 0.1])]):
    # using hardware station
    station, plant, scene_graph, diagram = setup_hardware_station(meshcat, gripper_poses=gripper_poses, base_pos=base_pos)

    cost, X_G = grasp_selection(diagram, plant, station, meshcat)
   
    return diagram, station
