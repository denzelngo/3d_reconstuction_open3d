import open3d as o3d
import os
from os.path import join
import numpy as np
from tqdm import tqdm
from utils import load_image

# Config, has to be the same with one in fragment step
BASE_LINE = 8  # centimeter
FOCAL_LENGTH_X = 454.18860566  # pixel
FRAGMENT_SIZE = 10
image_width, image_height = 440, 276
intrinsics = np.array([[454.18860566, 0, 314.37282313 - 100], [0, 445.34632249, 194.6299758 - 62.5], [0, 0, 1]],
                      dtype=float)  # ov9218 resize

# 3D volume of the scene
volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=20.0 / 512.0,
    sdf_trunc=0.04,  # default 0.04.
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.Gray32)

# Paths to folders: image, depth and pose
color_path = f'/home/user5/WORKSPACE/STEREO_DEPTH/RAFT-Stereo/calib/ov9281_slam_rect1/left/'
depth_path = f'/home/user5/WORKSPACE/STEREO_DEPTH/RAFT-Stereo/ov9281_slam_rect1_disp/'
pose_path = f'pose_forest_orbslam3/'

color_list = sorted(os.listdir(color_path))[:500]
end_id = (len(color_list) // FRAGMENT_SIZE) * FRAGMENT_SIZE

# Path to load the registration transformation
trans_save = 'fragment_transformation'
trans_list = []
for t in sorted(os.listdir(trans_save), key=lambda x: float(x[9:-4])):
    print(t)
    trans_list.append(np.load(join(trans_save, t)))

# 3D reconstruction
for i, f in enumerate(tqdm(color_list[:500])):
    print('Fragment: ', i // FRAGMENT_SIZE)
    trans = trans_list[i // FRAGMENT_SIZE]
    color_raw = o3d.geometry.Image(load_image(join(color_path, f)))
    disp = np.load(join(depth_path, f[:-4] + '.npy')).astype(np.float32)
    depth = (BASE_LINE * FOCAL_LENGTH_X / disp)
    depth_raw = o3d.geometry.Image(depth)
    pose = np.loadtxt(join(pose_path, f[:-4] + '.txt'))
    # Scale smaller number to faster calculation
    # pose[:3, 3] = pose[:3, 3] / 10

    # pose = np.dot(pose, np.linalg.inv(trans))
    # pose = np.dot(trans, pose)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw,
        depth_raw,
        depth_scale=100,
        depth_trunc=10,  # max depth
        convert_rgb_to_intensity=True)

    volume.integrate(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            image_width, image_height,
            intrinsics[0, 0], intrinsics[1, 1],
            intrinsics[0, 2], intrinsics[1, 2]),
        np.linalg.inv(pose))
mesh = volume.extract_triangle_mesh()
o3d.io.write_triangle_mesh("mesh_fined.ply", mesh)

mesh.compute_vertex_normals()
pcd = o3d.geometry.PointCloud()
pcd.points = mesh.vertices
pcd.colors = mesh.vertex_colors
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(height=540, width=960)
vis.add_geometry(pcd)
vis.run()
vis.destroy_window()
