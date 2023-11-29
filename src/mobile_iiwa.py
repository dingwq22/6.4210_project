import numpy as np
from pydrake.all import (
    ConstantVectorSource,
    InverseDynamicsController,
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
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
)
from manipulation.scenarios import AddMultibodyTriad, MakeManipulationStation
from manipulation.station import MakeHardwareStation
from manipulation import ConfigureParser, running_as_notebook
# from manipulation.scenarios import AddMultibodyTriad, MakeManipulationStation
from manipulation.station import MakeHardwareStation, load_scenario

from utils import find_project_path, notebook_plot_diagram
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

def setup_hardware_station(meshcat, obstables = [(1, 4), (1, 5), (2, 2), (2, 3), (4, 1), (4, 2), (4, 4), (5, 1), (5, 2)]):

    builder = DiagramBuilder()
    path = find_project_path()
    print(path)
    degrees = '{ deg: [90, 0, 90]}'
    scenario_data = f"""
directives:
- add_model:
    name: iiwa
    file: package://manipulation/mobile_iiwa14_primitive_collision.urdf
    default_joint_positions:
        iiwa_joint_1: [-1.57]
        iiwa_joint_2: [0.1]
        iiwa_joint_3: [0]
        iiwa_joint_4: [-1.2]
        iiwa_joint_5: [0]
        iiwa_joint_6: [ 1.6]
        iiwa_joint_7: [0]
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
"""
    print(scenario_data)
    #add mountains
    # scenario_data += get_mountain_yaml(obstables)
    
    scenario = load_scenario(data=scenario_data)
    station = builder.AddSystem(MakeManipulationStation(filename="file:///workspaces/6.4210_project/drake_obstacles_nopkg.dmd.yaml"))
    # station = builder.AddSystem(MakeHardwareStation(scenario, meshcat))

    plant = station.GetSubsystemByName("plant")
    scene_graph = station.GetSubsystemByName("scene_graph")
    AddMultibodyTriad(plant.GetFrameByName("body"), scene_graph)

    MeshcatVisualizer.AddToBuilder(
        builder,
        station.GetOutputPort("query_object"),
        meshcat,
        MeshcatVisualizerParams(delete_on_initialization_event=False),
    )

    wsg_position = builder.AddSystem(ConstantVectorSource([0.1]))
    # builder.Connect(
    #     wsg_position.get_output_port(), station.GetInputPort("wsg_position")
    # )

    diagram = builder.Build()

    context = plant.CreateDefaultContext()
    gripper = plant.GetBodyByName("body")

    initial_pose = plant.EvalBodyPoseInWorld(context, gripper)

    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(0.01)
    # context = station.CreateDefaultContext()

    # # station.GetInputPort("wsg.position").FixValue(context, [0.1])
    # station.ForcedPublish(context)
    # # print(station.get_input_port(1))
   
    # plant = station.GetSubsystemByName("plant")

    # scene_graph = station.GetSubsystemByName("scene_graph")
    # MeshcatVisualizer.AddToBuilder(
    #     builder,
    #     station.GetOutputPort("query_object"),
    #     meshcat,
    #     MeshcatVisualizerParams(delete_on_initialization_event=False),
    # )
    # diagram = builder.Build()
    print('plotting')
    # print(notebook_plot_diagram_svg(diagram))
    return station, plant, scene_graph, diagram




def filterCollsionGeometry(scene_graph, context=None):
    """Some robot models may appear to have self collisions due to overlapping collision geometries.
    This function filters out such problems for our PR2 model."""
    if context is None:
        filter_manager = scene_graph.collision_filter_manager()
    else:
        filter_manager = scene_graph.collision_filter_manager(context)
    inspector = scene_graph.model_inspector()

    pr2 = {}
    shelves = []
    tables = []

    for gid in inspector.GetGeometryIds(
        GeometrySet(inspector.GetAllGeometryIds()), Role.kProximity
    ):
        gid_name = inspector.GetName(inspector.GetFrameId(gid))
        if "pr2" in gid_name:
            link_name = gid_name.split("::")[1]
            pr2[link_name] = [gid]

    def add_exclusion(set1, set2=None):
        if set2 is None:
            filter_manager.Apply(
                CollisionFilterDeclaration().ExcludeWithin(GeometrySet(set1))
            )
        else:
            filter_manager.Apply(
                CollisionFilterDeclaration().ExcludeBetween(
                    GeometrySet(set1), GeometrySet(set2)
                )
            )

    # Robot-to-self collisions
    add_exclusion(
        pr2["base_link"],
        pr2["l_shoulder_pan_link"]
        + pr2["r_shoulder_pan_link"]
        + pr2["l_upper_arm_link"]
        + pr2["r_upper_arm_link"]
        + pr2["head_pan_link"]
        + pr2["head_tilt_link"],
    )
    add_exclusion(
        pr2["torso_lift_link"], pr2["head_pan_link"] + pr2["head_tilt_link"]
    )
    add_exclusion(
        pr2["l_shoulder_pan_link"] + pr2["torso_lift_link"],
        pr2["l_upper_arm_link"],
    )
    add_exclusion(
        pr2["r_shoulder_pan_link"] + pr2["torso_lift_link"],
        pr2["r_upper_arm_link"],
    )
    add_exclusion(pr2["l_forearm_link"], pr2["l_gripper_palm_link"])
    add_exclusion(pr2["r_forearm_link"], pr2["r_gripper_palm_link"])




def set_position(meshcat, X_WG, max_tries=10, fix_base=False, base_pose=np.zeros(3)):
    # diagram, plant, scene_graph = build_env(meshcat)

    # using hardware station
    plant, station, scene_graph, diagram = setup_hardware_station(meshcat)
    return diagram

    # initialize context
    # station_context = station.CreateDefaultContext()
    # plant_context = plant.GetMyContextFromRoot(station_context)

    # previous code 
    # world_frame = plant.world_frame()
    # gripper_frame = plant.GetFrameByName("l_gripper_palm_link")

    # context = diagram.CreateDefaultContext()
    # plant_context = plant.GetMyContextFromRoot(context)
    # sg_context = scene_graph.GetMyContextFromRoot(context)
    # filterCollsionGeometry(scene_graph, sg_context)

    # ik = InverseKinematics(plant, plant_context)
    # q_variables = ik.q()  # Get variables for MathematicalProgram
    # q_len = len(q_variables)
    # print("q_var len: ", q_len)
    # # q_nominal = np.zeros(len(q_variables))

    # goal_position = X_WG.translation()
    # q_variables = np.concatenate((np.array(goal_position), np.zeros(q_len - 3)))
    # # q_variables = q_nominal
    # print("q_varibles: " , q_variables)


    # if running_as_notebook:
    #     plant.SetPositions(
    #         plant_context,
    #         q_variables
    #     )
    #     diagram.ForcedPublish(context)