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

from utils import notebook_plot_diagram, build_scenario_data
from convert_pose import create_q_knots
from mountain_building import get_mountain_yaml
from trajectory_planning import trajectory_plan
from perception import grasp_selection

import pydot
from IPython.display import display, Image, SVG

def setup_hardware_station(meshcat, base_pos=None, gripper_poses=None, obstacles = [(1, 4), (1, 5), (2, 2), (2, 3), (4, 1), (4, 2), (4, 4), (5, 1), (5, 2)]):
    builder = DiagramBuilder()
    scenario_data = build_scenario_data()
    scenario = load_scenario(data=scenario_data)
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat))

    plant = station.GetSubsystemByName("plant")
    print(plant.GetStateNames())
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

    # trajectory plan
    x, y = base_pos
    goal = (-0.5 + x, -0.0 + y) 
    traj = trajectory_plan(goal)
    traj_len = len(traj)

    # time 
    duration = 10
    if gripper_poses is None:
        gripper_traj_len = 9
    else:
        gripper_traj_len = len(gripper_poses)
    unit_time = duration / (traj_len + gripper_traj_len)
    print("trag_len:", traj_len)
    print('gripper_traj_len:', gripper_traj_len)
    print("unit_time:", unit_time)
    t_lst = np.linspace(0, duration, traj_len + gripper_traj_len)

    # make dummy trajectory sources with the right size for the output port 
    # update trajectory later 

    # iiwa joints trajectory
    q_poses = np.zeros((20, traj_len + gripper_traj_len))
    q_traj = PiecewisePolynomial.CubicShapePreserving(t_lst, q_poses)
    q_traj_source = TrajectorySource(q_traj)
    q_traj_system = builder.AddSystem(q_traj_source) 
     
    # gripper trajectory
    gripper_t_lst = np.array([0, 1, 2, 3, duration])
    gripper_knots = np.array([1, 1, 1, 1, 1]).reshape(1, len(gripper_t_lst))
    g_traj = PiecewisePolynomial.CubicShapePreserving(gripper_t_lst, gripper_knots)
    g_traj_source = TrajectorySource(g_traj)
    g_traj_system = builder.AddSystem(g_traj_source) 

    builder.Connect(
        q_traj_system.get_output_port(), station.GetInputPort("iiwa.desired_state")
    )
    builder.Connect(
        g_traj_system.get_output_port(), station.GetInputPort("wsg.position")
    )

    diagram = builder.Build()
    # find antipodal grasps
    cost, X_G = grasp_selection(diagram, plant, station, meshcat)
    print("grasp selection done")


    # construct gripper poses 
    last_iiwa_pos = traj[-1]
    print("last_iiwa_pos", last_iiwa_pos)

    # gripper_1 = X_G.translation() - last_iiwa_pos
    gripper_1 = X_G.translation() - last_iiwa_pos + [-0.02, -0.07, -0.2] # magic number 
    print("gripper_1", gripper_1)
    new_pos = lambda delta: [x+y for x,y in zip(gripper_1, delta)]
    gripper_poses = [RigidTransform(new_pos([0, 0, 0.3])), 
                     RigidTransform(gripper_1), 
                     RigidTransform(gripper_1), 
                     RigidTransform(gripper_1), 
                     RigidTransform(new_pos([0, 0, 0.3])), 
                     RigidTransform([0, -0.3, 0.5]), 
                     RigidTransform([-0.5, -0.3, 0.5]), 
                     RigidTransform([-.5, 0, 0.5]), 
                     RigidTransform([-.5, 0, 0.5])] 



    q_poses = np.zeros((20, traj_len + gripper_traj_len))
    # set iiwa to the last traj pos when gripper is moving 
    q_poses[0:3, :]= np.array(traj + [traj[-1]]*gripper_traj_len).T 
    
    # find joint positions from end-effector pose
    joint_pos_lst = create_q_knots(gripper_poses)  # shape=(gripper_traj_len, 7)
    
    # iiwa joints trajectory 
    for i in range(gripper_traj_len):
        q_poses[3:10, traj_len+i] = joint_pos_lst[i][:7]
    print('q poses', q_poses[3:10, traj_len+1:])
    q_traj = PiecewisePolynomial.CubicShapePreserving(t_lst, q_poses)
    q_traj_source.UpdateTrajectory(q_traj)
     
    # gripper trajectory 
    gripper_close_t = unit_time*(traj_len+2)
    gripper_open_t = duration - unit_time*1
    gripper_t_lst = np.array([0.0, unit_time*(traj_len), gripper_close_t,  gripper_open_t, duration])
    gripper_knots = np.array([1, 1, 0.1, 0, 1]).reshape(1, len(gripper_t_lst))
    g_traj = PiecewisePolynomial.CubicShapePreserving(gripper_t_lst, gripper_knots)
    g_traj_source.UpdateTrajectory(g_traj)

    simulator = Simulator(diagram)
    visualizer.StartRecording(False)
    
    simulator.AdvanceTo(duration)
    visualizer.PublishRecording()

    return station, plant, scene_graph, diagram



def set_position(meshcat, base_pos=None, gripper_poses=None):
    # using hardware station
    station, plant, scene_graph, diagram = setup_hardware_station(meshcat, gripper_poses=gripper_poses, base_pos=base_pos)

    return diagram, station
