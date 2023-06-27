import open3d as o3d
import numpy as np
import os
from os.path import join

# Path to load fragments and to save the registration transformation
fragment_pcd_save = 'fragment_pcd'
pcd_f_list = sorted(os.listdir(fragment_pcd_save), key=lambda x: float(x[5:-4]))
pcd_list = []

trans_save = 'fragment_transformation'
if not os.path.isdir(trans_save):
    os.mkdir(trans_save)

# Config
voxel_size = 0.03
thresh_distance = 1
trans_init = np.eye(4)  # for ICP
trans = np.eye(4)  # use to transform the pcd[n+1]

np.save(join(trans_save, f'fragment_0.npy'), trans)

# Load point clouds
for p in pcd_f_list:
    print(p)
    pcd = o3d.io.read_point_cloud(join(fragment_pcd_save, p))
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    pcd_list.append(pcd)

# Registration
for i in range(len(pcd_list) - 1):
    print(f'ICP between fragment {i} and fragment {i + 1}')
    loss = o3d.pipelines.registration.TukeyLoss(k=0.1)
    reg_p2l = o3d.pipelines.registration.registration_icp(
        pcd_list[i + 1], pcd_list[i].transform(trans), thresh_distance, trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
        # o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20000))
    print(reg_p2l)
    trans = reg_p2l.transformation
    np.save(join(trans_save, f'fragment_{i + 1}.npy'), trans)
