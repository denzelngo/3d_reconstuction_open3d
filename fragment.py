import open3d as o3d
import os
from os.path import join
import numpy as np
from tqdm import tqdm
from utils import load_image

# Config
BASE_LINE = 8  # centimeter
FOCAL_LENGTH_X = 454.18860566  # pixel
image_width, image_height = 440, 276
intrinsics = np.array([[454.18860566, 0, 314.37282313 - 100], [0, 445.34632249, 194.6299758 - 62.5], [0, 0, 1]],
                      dtype=float)  # ov9218 resize

FRAGMENT_SIZE = 50

# Paths to folders: image, depth and pose
color_path = f'/home/user5/WORKSPACE/STEREO_DEPTH/RAFT-Stereo/calib/ov9281_28_april_slam_rect3/left/'
depth_path = f'/home/user5/WORKSPACE/STEREO_DEPTH/RAFT-Stereo/ov9281_28_april_slam_rect3_disp/'
pose_path = f'pose_rd_room_orbslam3_rgbd/'
color_list = sorted(os.listdir(color_path))[:500]
n_fragment = int(len(color_list) / FRAGMENT_SIZE)

# Path to save fragments
fragment_pcd_save = 'fragment_pcd'
if not os.path.isdir(fragment_pcd_save):
    os.mkdir(fragment_pcd_save)

# Make fragments
for n in range(n_fragment):
    print('Processing fragment ', n)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=10 / 512.0,
        sdf_trunc=0.04,  # default 0.04.
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.Gray32)
    for i, f in enumerate(tqdm(color_list[n * FRAGMENT_SIZE:n * FRAGMENT_SIZE + FRAGMENT_SIZE])):
        color_raw = o3d.geometry.Image(load_image(join(color_path, f)))
        disp = np.load(join(depth_path, f[:-4] + '.npy')).astype(np.float32)
        depth = (BASE_LINE * FOCAL_LENGTH_X / disp)
        depth_raw = o3d.geometry.Image(depth)
        pose = np.loadtxt(join(pose_path, f[:-4] + '.txt'))

        # Scale smaller number to faster calculation
        # pose[:3, 3] = pose[:3, 3] / 10

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw,
            depth_raw,
            depth_scale=100,
            depth_trunc=3,  # max depth
            convert_rgb_to_intensity=True)

        volume.integrate(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                image_width, image_height,
                intrinsics[0, 0], intrinsics[1, 1],
                intrinsics[0, 2], intrinsics[1, 2]),
            np.linalg.inv(pose))

    mesh = volume.extract_triangle_mesh()

    mesh.compute_vertex_normals()
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors
    o3d.io.write_point_cloud(join(fragment_pcd_save, f'frag_{n}.pcd'), pcd)
