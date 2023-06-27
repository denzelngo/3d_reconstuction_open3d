import numpy as np
import os
from scipy.spatial.transform import Rotation as R

# Pose calculated from SLAM or VO methods (in form of quaternion)
post_txt_path = '/home/user5/WORKSPACE/SLAM/DV_RGBD_data/DV_outside_1_RGBD/pose_orbslam3.txt'

# Path to save output
save_dir = 'pose_dv_outside_1_orbslam3_rgbd'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
with open(post_txt_path, 'r') as f:
    poses = f.read().splitlines()
count = 0
for pose in poses:
    count += 1
    p_mat = np.zeros((4, 4))
    p_mat[3, 3] = 1
    _, *p = pose.split(' ')
    n_frame = str(int(float(count))).zfill(4)
    p = list(map(float, p))
    p_mat[:3, :3] = R.from_quat(p[3:]).as_matrix()
    # p_mat[:3, :3] = R.from_quat([p[6], p[3], p[4], p[5]]).as_matrix  # DPVO output from lietorch, has to be reordered
    p_mat[:3, 3] = np.array(p[:3])

    np.savetxt(os.path.join(save_dir, n_frame + '.txt'), p_mat)
