import open3d as o3d
import os
from os.path import join
import numpy as np
from tqdm import tqdm
from utils import load_image

# This script is used to make a raw 3D reconstruction

BASE_LINE = 8  # centimeters
FOCAL_LENGTH_X = 454.18860566  # pixel
image_width, image_height = 440, 276
# intrinsics = np.array([[454.18860566, 0, 314.37282313], [0, 445.34632249, 194.6299758], [0, 0, 1]],
#                       dtype=float)  # ov9218 original
intrinsics = np.array([[454.18860566, 0, 314.37282313 - 100], [0, 445.34632249, 194.6299758 - 62.5], [0, 0, 1]],
                      dtype=float)  # ov9218 resized after rectification

volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=20 / 512.0,
    sdf_trunc=0.04,  # default 0.04.
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.Gray32)

color_path = f'/home/user5/WORKSPACE/STEREO_DEPTH/calib_stereo/ov9281_23_may_slam_rect2/left/'
depth_path = f'/home/user5/WORKSPACE/STEREO_DEPTH/RAFT-Stereo/output/v9281_23_may_slam_rect2_disp/'
pose_path = f'pose_dv_outside_1_orbslam3_rgbd/'

color_list = sorted(os.listdir(color_path))

for i, f in enumerate(tqdm(color_list[1451:])):  # Modify the indexing to visualize different parts of frames
    color_raw = o3d.geometry.Image(load_image(join(color_path, f)))
    disp = np.load(join(depth_path, f[:-4] + '.npy')).astype(np.float32)
    depth = (BASE_LINE * FOCAL_LENGTH_X / disp)
    depth_raw = o3d.geometry.Image(depth)
    pose = np.loadtxt(join(pose_path, f[:-4] + '.txt'))

    # Scale smaller number to faster calculation
    # pose[:3, 3] = pose[:3, 3]/10

    # pose = np.loadtxt(join(pose_path, str(i+1).zfill(4) + '.txt'))

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw,
        depth_raw,
        depth_scale=100,
        depth_trunc=20,  # max depth
        convert_rgb_to_intensity=True)

    volume.integrate(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            image_width, image_height,
            intrinsics[0, 0], intrinsics[1, 1],
            intrinsics[0, 2], intrinsics[1, 2]),
        np.linalg.inv(pose))

# pcd = volume.extract_point_cloud()


mesh = volume.extract_triangle_mesh()
print('Saving mesh ...')
o3d.io.write_triangle_mesh("dv-outside-1-mesh-rgbd.ply", mesh)
print('Done!')

mesh.compute_vertex_normals()
pcd = o3d.geometry.PointCloud()
pcd.points = mesh.vertices
pcd.colors = mesh.vertex_colors
# pcd = pcd.voxel_down_sample(0.02)
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
# o3d.io.write_point_cloud('pcd3.pcd', pcd)
# o3d.visualization.draw_geometries([pcd])

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(height=540, width=960)
vis.add_geometry(pcd)
vis.run()
vis.destroy_window()
