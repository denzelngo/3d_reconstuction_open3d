import open3d as o3d
import numpy as np
import os
from os.path import join
import time

# This script is used to visualize in animation the camera pose in a reconstructed 3D scene

# Camera object to be plotted
CAM_POINTS = np.array([
    [0, 0, 0],
    [-1, -1, 1.5],
    [1, -1, 1.5],
    [1, 1, 1.5],
    [-1, 1, 1.5],
    [-0.5, 1, 1.5],
    [0.5, 1, 1.5],
    [0, 1.2, 1.5]])
CAM_LINES = np.array([
    [1, 2], [2, 3], [3, 4], [4, 1], [1, 0], [0, 2], [3, 0], [0, 4], [5, 7], [7, 6]])

# Path to pose
pose_path = 'pose_exp_room_orbslam3_rgbd'

# To visualize all poses at the same time?
keep_traj = False

# To save the drawn frames?
save_frame = False
if save_frame:
    save_path = 'demo_save_dv_rd_room'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)


# Animation function
def animation(vis):
    for i, f in enumerate(sorted(os.listdir(pose_path))[:]):
        cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
        print(i)
        pose = np.loadtxt(join(pose_path, f))

        scale = 0.2
        camera_actor = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
            lines=o3d.utility.Vector2iVector(CAM_LINES))
        g = 1
        color = (g * 1.0, 0.5 * (1 - g), 0.9 * (1 - g))
        camera_actor.paint_uniform_color(color)
        camera_actor.transform(pose)
        camera_actor.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # camera_actor.transform([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        vis.add_geometry(camera_actor)
        cam = vis.get_view_control().convert_from_pinhole_camera_parameters(cam)
        vis.poll_events()
        vis.update_renderer()
        if save_frame:
            vis.capture_screen_image(join(save_path, str(i).zfill(5) + '.jpg'))
        time.sleep(0.01)
        if not keep_traj:
            cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
            vis.remove_geometry(camera_actor)
            cam = vis.get_view_control().convert_from_pinhole_camera_parameters(cam)
            vis.poll_events()
            vis.update_renderer()


# Load mesh
mesh1 = o3d.io.read_triangle_mesh('dv-exp-room-mesh.ply')
# Flip it, otherwise the pointcloud will be upside down
mesh1.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
vis = o3d.visualization.VisualizerWithKeyCallback()

vis.create_window(height=540, width=960)
vis.add_geometry(mesh1)
vis.register_animation_callback(animation)
vis.run()
vis.destroy_window()
