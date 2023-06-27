import cv2
import os
from os.path import join
import time
import threading
import queue

render_path = 'demo_save_dv_rd_room'
# rgb_path = '/home/user5/WORKSPACE/STEREO_DEPTH/RAFT-Stereo/calib/ov9281_slam_rect1/left'
rgb_path = '/home/user5/Downloads/ov9281_28_april_slam_video3/left'


# render_path = 'demo_save'
# rgb_path = '/home/user5/WORKSPACE/DATASETS/7scenes/office/seq-01/rgb'

# for ren_f, img_f in zip(sorted(os.listdir(render_path))[:40], sorted(os.listdir(rgb_path))[:40]):
for ren_f, img_f in zip(sorted(os.listdir(render_path)), sorted(os.listdir(rgb_path)[::])):
    # print(ren_f, img_f)
    ren = cv2.imread(join(render_path, ren_f))
    img = cv2.imread(join(rgb_path, img_f))

    cv2.imshow('RGB', img)

    cv2.imshow('3D View', ren)

    if cv2.waitKey(1) == 27:
        break

    time.sleep(0.03)
