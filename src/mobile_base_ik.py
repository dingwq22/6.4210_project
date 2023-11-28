import numpy as np
from pydrake.all import (
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
)
from manipulation import ConfigureParser, running_as_notebook
import pydot
from IPython.display import display, Image, SVG

import math

def build_env(meshcat):
    """Load in models and build the diagram."""
    print("build env start")
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    ConfigureParser(parser)
    # load local pkg 
    parser.package_map().PopulateFromFolder("..")
    parser.AddModels('../drake_obstacles.dmd.yaml')
    plant.Finalize()

    MeshcatVisualizer.AddToBuilder(
        builder,
        scene_graph.get_query_output_port(),
        meshcat,
        MeshcatVisualizerParams(delete_on_initialization_event=False),
    )

    diagram = builder.Build()
    print("build env done")
    return diagram, plant, scene_graph


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



def solve_ik(meshcat, X_WG, max_tries=10, fix_base=False, base_pose=np.zeros(3)):
    diagram, plant, scene_graph = build_env(meshcat)

    world_frame = plant.world_frame()
    gripper_frame = plant.GetFrameByName("l_gripper_palm_link")

    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    sg_context = scene_graph.GetMyContextFromRoot(context)
    filterCollsionGeometry(scene_graph, sg_context)

    # Note: passing in a plant_context is necessary for collision-free constraints!
    ik = InverseKinematics(plant, plant_context)
    q_variables = ik.q()  # Get variables for MathematicalProgram
    prog = ik.prog()  # Get MathematicalProgram
    q_nominal = np.zeros(len(q_variables))
    q_nominal[0:3] = base_pose
    prog.AddQuadraticErrorCost(
        np.eye(len(q_variables)), q_nominal, q_variables
    )

    def AddOrientationConstraint(ik, R_WG, bounds):
        """Add orientation constraint to the ik problem. Implements an inequality
        constraint where the axis-angle difference between f_R(q) and R_WG must be
        within bounds. Can be translated to:
        ik.prog().AddBoundingBoxConstraint(angle_diff(f_R(q), R_WG), -bounds, bounds)
        """
        ik.AddOrientationConstraint(
            frameAbar=world_frame,
            R_AbarA=R_WG,
            frameBbar=gripper_frame,
            R_BbarB=RotationMatrix(),
            theta_bound=bounds,
        )

    def AddPositionConstraint(ik, p_WG_lower, p_WG_upper):
        """Add position constraint to the ik problem. Implements an inequality
        constraint where f_p(q) must lie between p_WG_lower and p_WG_upper. Can be
        translated to
        ik.prog().AddBoundingBoxConstraint(f_p(q), p_WG_lower, p_WG_upper)
        """
        ik.AddPositionConstraint(
            frameA=world_frame,
            frameB=gripper_frame,
            p_BQ=np.zeros(3),
            p_AQ_lower=p_WG_lower,
            p_AQ_upper=p_WG_upper,
        )


    # Add your constraints here
    if fix_base:
        AddPositionConstraint(ik, base_pose, base_pose)
    else:
        AddPositionConstraint(ik, X_WG.translation() - 0.001, X_WG.translation() + 0.001)
    AddOrientationConstraint(ik, X_WG.rotation(), np.pi/180)
    ik.AddMinimumDistanceLowerBoundConstraint(0.01)

    for count in range(max_tries):
        # Compute a random initial guess here
        lower_limit = plant.GetPositionLowerLimits()
        upper_limit = plant.GetPositionUpperLimits()
        q_guess = []
        for l, u in zip(lower_limit, upper_limit):
            if (math.isinf(l) or math.isinf(u)):
                q_guess.append(np.random.uniform(-np.pi, np.pi))
            else:
                q_guess.append(np.random.uniform(l, u))
        print(q_guess)
        prog.SetInitialGuess(q_variables, q_guess)
        

        result = Solve(prog)
        # print("result: ", result)
        print("solution: ", result.GetSolution(q_variables))

        if running_as_notebook:
            render_context = diagram.CreateDefaultContext()
            plant.SetPositions(
                plant.GetMyContextFromRoot(render_context),
                result.GetSolution(q_variables),
            )
            diagram.ForcedPublish(context)

        if result.is_success():
            print("Succeeded in %d tries!" % (count + 1))
            return result.GetSolution(q_variables)

    print("Failed!")
    return None