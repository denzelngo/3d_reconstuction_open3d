import open3d as o3d
from random import random
import numpy as np
import copy

voxel_size = 0.01

trans_init = np.eye(4)

i = 0
j = 2
pcd1 = o3d.io.read_point_cloud(f'fragment_pcd/frag_{i}.pcd')
pcd3 = o3d.io.read_point_cloud(f'fragment_pcd/frag_{j}.pcd')

pcd1_down = copy.deepcopy(pcd1).voxel_down_sample(voxel_size)
pcd3_down = copy.deepcopy(pcd3).voxel_down_sample(voxel_size)

pcd1_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
pcd3_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

loss = o3d.pipelines.registration.TukeyLoss(k=0.01)
reg_p2l = o3d.pipelines.registration.registration_icp(
    pcd3_down, pcd1_down, 0.07, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    # o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20000))
pcd3.transform(reg_p2l.transformation)

print(reg_p2l)
print(reg_p2l.transformation)

pcd1.paint_uniform_color([0.1, 0.1, 0.7])  # red
pcd3.paint_uniform_color([0.5, 0.1, 0.2])  # blue

pcd1.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
pcd3.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(height=540, width=960)
vis.add_geometry(pcd1)
vis.add_geometry(pcd3)
vis.run()
vis.destroy_window()
