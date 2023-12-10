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
from traj2 import trajectory_plan
from perception import grasp_selection

import pydot
from IPython.display import display, Image, SVG

def setup_hardware_station(meshcat, base_pos=None, gripper_poses=None, obstacles  = None):
    builder = DiagramBuilder()
    scenario_data = build_scenario_data(obstacles, base_pos)
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
    for i in range(3*len(base_pos)):
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
    goal = []
    for b in base_pos:
        x, y = b
        goal.append((-0.5 + x, -0.0 + y))   # (-0.5, 0) is the offset of iiwa from object
    traj = trajectory_plan(goal, obstacles = obstacles)
    traj_len = len(traj)

    # time 
    duration = 20
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
    total_q_poses = np.zeros((20, 0))
    total_gripper_t_lst = np.zeros(0)
    total_gripper_knots = np.zeros((1, 0))
    last_timestamp = 0
    unit_time = 1.5
    for traj_seg in traj:
        last_pose = traj_seg[-1][:2]
        index = base_pos.index((last_pose[0]+0.5, last_pose[1]))
        traj_len = len(traj_seg)
        print('traj_seg', traj_seg)
        print('traj_len', traj_len)
        # find antipodal grasps
        print('camera', list(np.array([0, 1, 2])+index*3), 'base', [*base_pos[index], 0])
        cost, X_G = grasp_selection(diagram, plant, station, meshcat, camera_indices=list(np.array([0, 1, 2])*(index+1)), object_pose=[*base_pos[index], 0])
        print("grasp selection done, X_G", X_G)
        # construct gripper poses 
        last_iiwa_pos = traj_seg[-1]
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
                        RigidTransform([-.5, 0, 0.5]),
                        RigidTransform([-0.5, 0, 0.5]), 
                        RigidTransform([-0.2, 0, 1]),
                        RigidTransform([0, 0, 0])
                        ] 


        gripper_traj_len = len(gripper_poses)
        q_poses = np.zeros((20, traj_len + gripper_traj_len))
        # set iiwa to the last traj pos when gripper is moving 
        q_poses[0:3, :]= np.array(traj_seg + [traj_seg[-1]]*gripper_traj_len).T 
    
        # find joint positions from end-effector pose
        # joint_pos_lst = [np.zeros(9) for _ in range(gripper_traj_len)]
        ######## to do - convert back[hard coding for debugging speed] ######3
        joint_pos_lst = create_q_knots(gripper_poses)  # shape=(gripper_traj_len, 7)
        # print('joint_pos_lst', joint_pos_lst)
        # iiwa joints trajectory 
        for i in range(gripper_traj_len):
            q_poses[3:10, traj_len+i] = joint_pos_lst[i][:7]
        print('q poses', q_poses[0:10, :].T)
        total_q_poses = np.concatenate((total_q_poses, q_poses), axis=1)

        # gripper trajectory 
        segment_total_time = unit_time * q_poses.shape[1]
        print('segment total length', q_poses.shape[1])
        gripper_close_t = unit_time*(traj_len+2)
        gripper_open_t = segment_total_time  - unit_time*4
        gripper_t_lst = np.array([0.1, unit_time*(traj_len), gripper_close_t,  gripper_open_t, segment_total_time]) + last_timestamp
        total_gripper_t_lst = np.concatenate((total_gripper_t_lst, gripper_t_lst))
        last_timestamp = segment_total_time
        gripper_knots = np.array([1, 1, 0.1, 0, 1]).reshape(1, len(gripper_t_lst))
        total_gripper_knots = np.concatenate((total_gripper_knots, gripper_knots), axis=1)
        
    
    ####TO DO: change into a parameter
    
    total_len = total_q_poses.shape[1]
    print('total_len', total_len)
    total_time = unit_time * total_len
    t_lst = np.linspace(0, unit_time * total_len, total_len)
    print('t list', t_lst)
    print('total_q_poses', total_q_poses[0:10, :].T)
    q_traj = PiecewisePolynomial.CubicShapePreserving(t_lst, total_q_poses)
    q_traj_source.UpdateTrajectory(q_traj)
     
    print('total_gripper_t_lst', total_gripper_t_lst)
    print('total_gripper_knots', total_gripper_knots)
    g_traj = PiecewisePolynomial.CubicShapePreserving(total_gripper_t_lst, total_gripper_knots)
    g_traj_source.UpdateTrajectory(g_traj)

    simulator = Simulator(diagram)
    visualizer.StartRecording(False)
    
    simulator.AdvanceTo(total_time)
    visualizer.PublishRecording()

    return station, plant, scene_graph, diagram



def set_position(meshcat, base_pos=None, gripper_poses=None, obstacles = [(1, 4), (1, 5), (2, 2), (2, 3), (4, 1), (4, 2), (4, 4), (5, 1), (5, 2)]):
    # using hardware station
    station, plant, scene_graph, diagram = setup_hardware_station(meshcat, gripper_poses=gripper_poses, base_pos=base_pos, obstacles = obstacles)

    return diagram, station

# create q knots test
# gripper_poses = [
#                         RigidTransform([0, -0.3, 0.5]), 
#                         RigidTransform([-0.5, -0.3, 0.5]), 
#                         RigidTransform([-.5, 0, 0.5]), 
#                         RigidTransform([-.5, 0, 0.5]),
#                         RigidTransform([-0.5, 0, 0.5]), 
#                         RigidTransform([-0.2, 0, 1]),
#                         RigidTransform([0, 0, 0])
#                         ]

# joint_pos_lst = create_q_knots(gripper_poses)

# print(joint_pos_lst)

gripper_1 = [ 0.48111933, -0.0695355, 0.02945262]
# gripper_1 = X_G.translation() - last_iiwa_pos + [-0.02, -0.07, -0.2] # magic number 
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
                RigidTransform([-.5, 0, 0.5]),
                RigidTransform([-0.5, 0, 0.5]), 
                RigidTransform([-0.2, 0, 1]),
                RigidTransform([0, 0, 0])
                ] 
joint_pos_lst = create_q_knots(gripper_poses)
print(joint_pos_lst)
