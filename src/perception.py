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
    parser.package_map().AddPackageXml(filename="../package.xml")
    parser.AddModels('../camera_sys.dmd.yaml')
    plant.Finalize()
    return builder.Build()


def grasp_selection(diagram, plant, station, meshcat, camera_indices=[], object_pose=[]):
    print('camera', camera_indices, 'base', object_pose)
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
    object_pose = object_pose
    crop_min = [x + y for x, y in zip([-0.3, -0.3, 0.001], object_pose)]
    crop_max = [x + y for x, y in zip([0.3, 0.3, 1], object_pose)]

    pcd = []
    for i, ci in enumerate(camera_indices):
        cloud = system.GetOutputPort(f"camera{ci}_point_cloud").Eval(context)
        meshcat.SetObject(f"pointcloud{ci}", cloud, point_size=0.001)
        meshcat.SetProperty(f"pointcloud{ci}", "visible", False)

        # Crop to region of interest.
        pcd.append(
            cloud.Crop(lower_xyz=crop_min, upper_xyz=crop_max)
        )
        meshcat.SetObject(f"pointcloud{ci}_cropped", pcd[i], point_size=0.001)
        meshcat.SetProperty(f"pointcloud{ci}_cropped", "visible", False)

        pcd[i].EstimateNormals(radius=0.1, num_closest=30)

        # Flip normals toward camera
        camera = plant.GetModelInstanceByName(f"camera{ci}")
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
    else:
        best = np.argmin(costs)
        print('output')
        print((costs[best], X_Gs[best]))
        return costs[best], X_Gs[best]
        