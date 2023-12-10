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


def CreateIiwaControllerPlant():
    """creates plant that includes only the robot and gripper, used for controllers."""
    robot_sdf_path = "package://drake/manipulation/models/iiwa_description/iiwa7/iiwa7_no_collision.sdf"
    gripper_sdf_path = "package://drake/manipulation/models/wsg_50_description/sdf/schunk_wsg_50_no_tip.sdf"
    sim_timestep = 1e-3
    plant_robot = MultibodyPlant(sim_timestep)
    parser = Parser(plant=plant_robot)
    parser.AddModelsFromUrl(robot_sdf_path)
    parser.AddModelsFromUrl(gripper_sdf_path)
    plant_robot.WeldFrames(
        frame_on_parent_F=plant_robot.world_frame(),
        frame_on_child_M=plant_robot.GetFrameByName("iiwa_link_0"),
    )
    plant_robot.WeldFrames(
        frame_on_parent_F=plant_robot.GetFrameByName("iiwa_link_7"),
        frame_on_child_M=plant_robot.GetFrameByName("body"),
        X_FM=RigidTransform(
            RollPitchYaw(np.pi / 2, 0, np.pi / 2), np.array([0, 0, 0.114])
        ),
    )
    plant_robot.mutable_gravity_field().set_gravity_vector([0, 0, 0])
    plant_robot.Finalize()

    link_frame_indices = []
    for i in range(8):
        link_frame_indices.append(
            plant_robot.GetFrameByName("iiwa_link_" + str(i)).index()
        )

    return plant_robot, link_frame_indices
    

gripper_q_mapping =  {(0, -0.3, 0.5): np.array([-0.71238261,  1.55645414, -1.97399239, -1.72756634, -0.39117848,
        2.0944    , -2.04579181,  0.        ,  0.        ]), 
       (-0.5, -0.3, 0.5) : np.array([-1.96966727,  1.01300565, -0.87795721, -0.96760784, -0.77066757,
        1.95153448, -1.36202724,  0.        ,  0.        ]), 
        (-.5, 0, 0.5): np.array([-1.99919397,  0.7687324 , -1.13992482, -1.63819168, -0.89933682,
        1.35496273, -1.14798109,  0.        ,  0.        ]), 
        (-0.2, 0, 1) : np.array([-0.99014668,  0.40692406, -2.21711633, -1.00909656, -1.27710566,
        1.68597137, -2.2554369 ,  0.        ,  0.        ]),
        (0, 0, 0): np.zeros(9)
        }

plant = None

def create_q_knots(pose_lst):
    """Convert end-effector pose list to joint position list using series of
    InverseKinematics problems. Note that q is 9-dimensional because the last 2 dimensions
    contain gripper joints, but these should not matter to the constraints.
    @param: pose_lst (python list): post_lst[i] contains keyframe X_WG at index i.
    @return: q_knots (python_list): q_knots[i] contains IK solution that will give f(q_knots[i]) \approx pose_lst[i].
    """
    q_knots = []
    global plant 
    if plant is None:
        print('first time')
        
        plant = CreateIiwaControllerPlant()[0]
    
    world_frame = plant.world_frame()
    gripper_frame = plant.GetFrameByName("body")
    q_nominal = np.array(
        [0.0, 0.6, 0.0, -1.75, 0.0, 1.0, 0.0, 0.0, 0.0]
    )  # nominal joint angles for joint-centering.

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
            # p_BQ = [0, 0.12, 0], # p_GO
            p_AQ_lower=p_WG_lower,
            p_AQ_upper=p_WG_upper,
        )
    num_trials = 10
    print(pose_lst)
    for i in range(len(pose_lst)):
        print('enter loop', i)
        ik = inverse_kinematics.InverseKinematics(plant)
        q_variables = ik.q()  # Get variables for MathematicalProgram
        p, r = pose_lst[i].translation(), pose_lst[i].rotation()
        print('create q', i, p)
        print('mapping keys', list(gripper_q_mapping.keys()))

        if tuple(p) in gripper_q_mapping:
            q_knots.append(list(gripper_q_mapping[tuple(p)]))
            print('found in cache')
            continue

        for trial in range(num_trials):
            prog = ik.prog()  # Get MathematicalProgram
            
            #If i==0, set the initial guess to be nominal configuration using prog.SetInitialGuess. 
            #Otherwise, set the initial guess to be the solution you obtained on the previous IK problem.
            if i==0:
                init = q_nominal
            else:
                init = q_knots[-1]
            prog.SetInitialGuess(q_variables, init)

            # Implement constraints on ik using AddOrientationConstraint and AddPositionConstraint. 
            # AddOrientationConstraint implements an inequality constraint (see the docstring for more information).
            AddPositionConstraint(ik, p+np.array([0, 0, -0.01]),  p+np.array([0, 0, 0.01]))
            AddOrientationConstraint(ik, r, 0.1)
            
            # Add a joint-centering cost on q_nominal.
            prog.AddQuadraticErrorCost(np.eye(len(q_variables)), q_nominal, q_variables)


            ################################################

            result = Solve(prog)

            if (result.is_success()):
                break

        q_knots.append(result.GetSolution(q_variables))
        q0 = result.GetSolution(q_variables)
        print(i, q0)

    return q_knots