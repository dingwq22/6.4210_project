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
from manipulation.station import (
    AddPointClouds,
    MakeHardwareStation,
    add_directives,
    load_scenario,
)
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

def make_internal_model():
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    ConfigureParser(parser)
    parser.AddModelsFromUrl("package://manipulation/clutter_planning.dmd.yaml")
    plant.Finalize()
    return builder.Build()

# Takes 3 point clouds (in world coordinates) as input, and outputs and estimated pose for the mustard bottle.
class GraspSelector():
    def __init__(self, plant, bin_instance, camera_body_indices):
        # LeafSystem.__init__(self)
        model_point_cloud = AbstractValue.Make(PointCloud(0))
        # self.DeclareAbstractInputPort("cloud0_W", model_point_cloud)
        # self.DeclareAbstractInputPort("cloud1_W", model_point_cloud)
        # self.DeclareAbstractInputPort("cloud2_W", model_point_cloud)
        # self.DeclareAbstractInputPort(
        #     "body_poses", AbstractValue.Make([RigidTransform()])
        # )

        # port = self.DeclareAbstractOutputPort(
        #     "grasp_selection",
        #     lambda: AbstractValue.Make((np.inf, RigidTransform())),
        #     self.SelectGrasp,
        # )
        # port.disable_caching_by_default()

        # Compute crop box.
        context = plant.CreateDefaultContext()
        bin_body = plant.GetBodyByName("bin_base", bin_instance)
        X_B = plant.EvalBodyPoseInWorld(context, bin_body)
        margin = 0.001  # only because simulation is perfect!
        a = X_B.multiply(
            [-0.22 + 0.025 + margin, -0.29 + 0.025 + margin, 0.015 + margin]
        )
        b = X_B.multiply([0.22 - 0.1 - margin, 0.29 - 0.025 - margin, 2.0])
        self._crop_lower = np.minimum(a, b)
        self._crop_upper = np.maximum(a, b)

        # self._internal_model = make_internal_model()
        # self._internal_model_context = (
        #     self._internal_model.CreateDefaultContext()
        # )
        self._internal_model_context = context
        self._rng = np.random.default_rng()
        self._camera_body_indices = camera_body_indices

    def SelectGrasp(self, context, output):
        body_poses = self.get_input_port(3).Eval(context)
        pcd = []
        for i in range(3):
            cloud = self.get_input_port(i).Eval(context)
            pcd.append(cloud.Crop(self._crop_lower, self._crop_upper))
            pcd[i].EstimateNormals(radius=0.1, num_closest=30)

            # Flip normals toward camera
            X_WC = body_poses[self._camera_body_indices[i]]
            pcd[i].FlipNormalsTowardPoint(X_WC.translation())
        merged_pcd = Concatenate(pcd)
        down_sampled_pcd = merged_pcd.VoxelizedDownSample(voxel_size=0.005)

        costs = []
        X_Gs = []
        # TODO(russt): Take the randomness from an input port, and re-enable
        # caching.
        for i in range(100 if running_as_notebook else 2):
            cost, X_G = GenerateAntipodalGraspCandidate(
                self._internal_model,
                self._internal_model_context,
                down_sampled_pcd,
                self._rng,
            )
            if np.isfinite(cost):
                costs.append(cost)
                X_Gs.append(X_G)

        if len(costs) == 0:
            # Didn't find a viable grasp candidate
            print('fail')
            X_WG = RigidTransform(
                RollPitchYaw(-np.pi / 2, 0, np.pi / 2), [0.5, 0, 0.22]
            )
            output.set_value((np.inf, X_WG))
        else:
            best = np.argmin(costs)
            output.set_value((costs[best], X_Gs[best]))
            print('output')
            print((costs[best], X_Gs[best]))

def setup_hardware_station(meshcat, goal, gripper_poses, obstacles = [(1, 4), (1, 5), (2, 2), (2, 3), (4, 1), (4, 2), (4, 4), (5, 1), (5, 2)]):
    x, y = goal
    goal = (-0.545 + x, -0.07 + y)
    builder = DiagramBuilder()
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

- add_model:
    name: object1
    file: file://{path}/objects/obstacle_boxes.sdf
    default_free_body_pose:
        obstacles:
            translation: [-2, -2.43, 0.01]
- add_model:
    name: object2
    file: file://{path}/objects/obstacle_boxes.sdf
    default_free_body_pose:
        obstacles:
            translation: [0, 0, 0.01]
    
- add_frame:
    name: camera0_origin
    X_PF:
        base_frame: world
        rotation: !Rpy {camera0deg}
        translation: [0.5, 0.5, .5] #[.25, -.5, .4]

- add_model:
    name: camera0
    file: package://manipulation/camera_box.sdf

- add_weld:
    parent: camera0_origin
    child: camera0::base

- add_frame:
    name: camera1_origin
    X_PF:
        base_frame: world
        rotation: !Rpy {camera1deg}
        translation: [0, -.7, .5] #-0.05

- add_model:
    name: camera1
    file: package://manipulation/camera_box.sdf

- add_weld:
    parent: camera1_origin
    child: camera1::base

- add_frame:
    name: camera2_origin
    X_PF:
        base_frame: world
        rotation: !Rpy {camera2deg}
        translation: [-0.5, 0.5, .5] #[-.35, -.25, .45]

- add_model:
    name: camera2
    file: package://manipulation/camera_box.sdf

- add_weld:
    parent: camera2_origin
    child: camera2::base

cameras:
    camera0:
      name: camera0
      depth: True
      X_PB:
        base_frame: camera0::base

    camera1:
      name: camera1
      depth: True
      X_PB:
        base_frame: camera1::base

    camera2:
      name: camera2
      depth: True
      X_PB:
        base_frame: camera2::base
"""
    # print(scenario_data)
    # add mountains
    # scenario_data += get_mountain_yaml(obstables)
    scenario_data += f'''
model_drivers:
    iiwa: {driver1}
    wsg: {driver2}
    '''
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

    traj = trajectory_plan(goal)
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
    joint_pos_lst = create_q_knots(gripper_poses)  # shape=(gripper_traj_len, 7)
    
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
    return diagram, station
