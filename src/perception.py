import numpy as np
from pydrake.all import (
    AbstractValue,
    PointCloud,
    LeafSystem,
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    Parser,
    RigidTransform,
    RollPitchYaw,
    Concatenate,
)
from manipulation import ConfigureParser, running_as_notebook
from manipulation.clutter import GenerateAntipodalGraspCandidate

def make_internal_model():
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    ConfigureParser(parser)
    # parser.AddModelsFromUrl("package://manipulation/clutter_planning.dmd.yaml")
    parser.package_map().AddPackageXml(filename="../package.xml")
    parser.AddModels('../camera_sys.dmd.yaml')
    plant.Finalize()
    return builder.Build()

# Takes 3 point clouds (in world coordinates) as input, and outputs and estimated pose for the mustard bottle.
class GraspSelector():
    def __init__(self, plant, bin_instance, camera_body_indices):
        # LeafSystem.__init__(self)
        model_point_cloud = AbstractValue.Make(PointCloud(0))
        self.DeclareAbstractInputPort("cloud0_W", model_point_cloud)
        self.DeclareAbstractInputPort("cloud1_W", model_point_cloud)
        self.DeclareAbstractInputPort("cloud2_W", model_point_cloud)
        self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])
        )

        port = self.DeclareAbstractOutputPort(
            "grasp_selection",
            lambda: AbstractValue.Make((np.inf, RigidTransform())),
            self.SelectGrasp,
        )
        port.disable_caching_by_default()

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

        self._internal_model = make_internal_model()
        self._internal_model_context = (
            self._internal_model.CreateDefaultContext()
        )
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


def grasp_selection(diagram, plant, station, meshcat, builder=None):
#   system = builder.Build()
    system = diagram
    context = system.CreateDefaultContext()
    system.ForcedPublish(context)
    plant_context = plant.GetMyContextFromRoot(context)
    
    # body poses
    body_poses = station.GetOutputPort("body_poses")

    # object pose
    # TODO: eval object pose, remove hard-coded pose
    # object = plant.GetBodyByName("box")
    # object_pose = plant.EvalBodyPoseInWorld(context, object)
    object_pose = [1.5, 1, 0]
    crop_min = [x + y for x, y in zip([-0.3, -0.3, 0.01], object_pose)]
    crop_max = [x + y for x, y in zip([0.3, 0.3, 1], object_pose)]

    pcd = []
    for i in range(3):
        cloud = system.GetOutputPort(f"camera{i}_point_cloud").Eval(context)
        meshcat.SetObject(f"pointcloud{i}", cloud, point_size=0.001)
        meshcat.SetProperty(f"pointcloud{i}", "visible", False)

        # Crop to region of interest.
        pcd.append(
            cloud.Crop(lower_xyz=crop_min, upper_xyz=crop_max)
        )
        meshcat.SetObject(f"pointcloud{i}_cropped", pcd[i], point_size=0.001)
        meshcat.SetProperty(f"pointcloud{i}_cropped", "visible", False)

        pcd[i].EstimateNormals(radius=0.1, num_closest=30)

        # Flip normals toward camera
        camera = plant.GetModelInstanceByName(f"camera{i}")
        body = plant.GetBodyByName("base", camera)
        X_C = plant.EvalBodyPoseInWorld(plant_context, body)
        pcd[i].FlipNormalsTowardPoint(X_C.translation())

    # Merge point clouds
    merged_pcd = Concatenate(pcd)
    meshcat.SetObject("merged", merged_pcd, point_size=0.001)

    # Voxelize down-sample.  (Note that the normals still look reasonable)
    down_sampled_pcd = merged_pcd.VoxelizedDownSample(voxel_size=0.005)
    meshcat.SetObject("down_sampled", down_sampled_pcd, point_size=0.001)
    meshcat.SetLineSegments(
        "down_sampled_normals",
        down_sampled_pcd.xyzs(),
        down_sampled_pcd.xyzs() + 0.01 * down_sampled_pcd.normals(),
    )

    print("down_sampled_normals done")


    # select antipodal grasps 
    costs = []
    X_Gs = []
    # TODO(russt): Take the randomness from an input port, and re-enable
    # caching.
    internal_model = make_internal_model()
    internal_model_context = internal_model.CreateDefaultContext() # TODO: add collision filter 
    for i in range(100 if running_as_notebook else 20):
        cost, X_G = GenerateAntipodalGraspCandidate(
            # station,
            # context,
            internal_model,
            internal_model_context,
            down_sampled_pcd,
            np.random.default_rng()
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
        return np.inf, X_WG
        # output.set_value((np.inf, X_WG))
    else:
        best = np.argmin(costs)
        print('output')
        print((costs[best], X_Gs[best]))
        return costs[best], X_Gs[best]
        # output.set_value((costs[best], X_Gs[best]))
        