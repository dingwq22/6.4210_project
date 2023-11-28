import numpy as np
from pydrake.all import (
    AbstractValue,
    ConstantVectorSource,
    DiagramBuilder,
    LeafSystem,
    PiecewisePose,
    RigidTransform,
    RotationMatrix,
    Simulator,
    StartMeshcat,
)

from manipulation import running_as_notebook
from manipulation.exercises.grader import Grader
from manipulation.exercises.pick.test_robot_painter import TestRobotPainter
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.scenarios import AddIiwaDifferentialIK
from manipulation.station import MakeHardwareStation, load_scenario


class PoseTrajectorySource(LeafSystem):
    def __init__(self, pose_trajectory):
        LeafSystem.__init__(self)
        self._pose_trajectory = pose_trajectory
        self.DeclareAbstractOutputPort(
            "pose", lambda: AbstractValue.Make(RigidTransform()), self.CalcPose
        )

    def CalcPose(self, context, output):
        output.set_value(self._pose_trajectory.GetPose(context.get_time()))


class IIWA_Painter:
    def __init__(self, meshcat=None, traj=None):
        builder = DiagramBuilder()
        scenario_data = """
        directives:
        - add_directives:
            file: package://manipulation/clutter.dmd.yaml
        model_drivers:
            iiwa: !IiwaDriver
                hand_model_name: wsg
            wsg: !SchunkWsgDriver {}
        """
        scenario = load_scenario(data=scenario_data)
        if meshcat is not None:
          meshcat = meshcat
        else:
          raise Exception("cannot find meshcat")
        self.station = builder.AddSystem(
            MakeHardwareStation(scenario, meshcat=meshcat)
        )
        self.plant = self.station.GetSubsystemByName("plant")
        # Remove joint limits from the wrist joint.
        self.plant.GetJointByName("iiwa_joint_7").set_position_limits(
            [-np.inf], [np.inf]
        )
        controller_plant = self.station.GetSubsystemByName(
            "iiwa.controller"
        ).get_multibody_plant_for_control()

        # optionally add trajectory source
        if traj is not None:
            traj_source = builder.AddSystem(PoseTrajectorySource(traj))
            self.controller = AddIiwaDifferentialIK(
                builder,
                controller_plant,
                frame=controller_plant.GetFrameByName("body"),
            )
            builder.Connect(
                traj_source.get_output_port(),
                self.controller.get_input_port(0),
            )
            builder.Connect(
                self.station.GetOutputPort("iiwa.state_estimated"),
                self.controller.GetInputPort("robot_state"),
            )

            builder.Connect(
                self.controller.get_output_port(),
                self.station.GetInputPort("iiwa.position"),
            )

        wsg_position = builder.AddSystem(ConstantVectorSource([0.1]))
        builder.Connect(
            wsg_position.get_output_port(),
            self.station.GetInputPort("wsg.position"),
        )

        self.diagram = builder.Build()
        self.gripper_frame = self.plant.GetFrameByName("body")
        self.world_frame = self.plant.world_frame()

        context = self.CreateDefaultContext()
        self.diagram.ForcedPublish(context)

    def visualize_frame(self, name, X_WF, length=0.15, radius=0.006):
        """
        visualize imaginary frame that are not attached to existing bodies

        Input:
            name: the name of the frame (str)
            X_WF: a RigidTransform to from frame F to world.

        Frames whose names already exist will be overwritten by the new frame
        """
        AddMeshcatTriad(
            meshcat, "painter/" + name, length=length, radius=radius, X_PT=X_WF
        )

    def CreateDefaultContext(self):
        context = self.diagram.CreateDefaultContext()
        plant_context = self.diagram.GetMutableSubsystemContext(
            self.plant, context
        )

        # provide initial states
        q0 = np.array(
            [
                1.40666193e-05,
                1.56461165e-01,
                -3.82761069e-05,
                -1.32296976e00,
                -6.29097287e-06,
                1.61181157e00,
                -2.66900985e-05,
            ]
        )
        # set the joint positions of the kuka arm
        iiwa = self.plant.GetModelInstanceByName("iiwa")
        self.plant.SetPositions(plant_context, iiwa, q0)
        self.plant.SetVelocities(plant_context, iiwa, np.zeros(7))
        wsg = self.plant.GetModelInstanceByName("wsg")
        self.plant.SetPositions(plant_context, wsg, [-0.05, 0.05])
        self.plant.SetVelocities(plant_context, wsg, [0, 0])

        return context

    def get_X_WG(self, context=None):
        if not context:
            context = self.CreateDefaultContext()
        plant_context = self.plant.GetMyMutableContextFromRoot(context)
        X_WG = self.plant.CalcRelativeTransform(
            plant_context, frame_A=self.world_frame, frame_B=self.gripper_frame
        )
        return X_WG

    def paint(self, sim_duration=20.0):
        context = self.CreateDefaultContext()
        simulator = Simulator(self.diagram, context)

        meshcat.StartRecording(set_visualizations_while_recording=False)
        duration = sim_duration if running_as_notebook else 0.01
        simulator.AdvanceTo(duration)
        meshcat.PublishRecording()


def run_painter(meshcat):
  # define center and radius
  radius = 0.1
  p0 = [0.45, 0.0, 0.4]
  R0 = RotationMatrix(np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]).T)
  X_WCenter = RigidTransform(R0, p0)

  num_key_frames = 10
  """
  you may use different thetas as long as your trajectory starts
  from the Start Frame above and your rotation is positive
  in the world frame about +z axis
  thetas = np.linspace(0, 2*np.pi, num_key_frames)
  """
  thetas = np.linspace(0, 2 * np.pi, num_key_frames)

  painter = IIWA_Painter(meshcat=meshcat)