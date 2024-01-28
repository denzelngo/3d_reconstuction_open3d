# 3D Reconstruction using open3D
_written by Damien_

The code in this repository is designed to utilize RGB-D data for generating 3D scenes, using open3D framework.

The required data include RGB images, depth images and camera pose of each image. The depth data can be obtained by using RGB-D camera or calculated by processing rectified stereo images using RAFT-Stereo (or other stereo depth estimation methods). The pose data can be obtained by using SLAM or Visual Odometry methods (ORBSLAM3 (monocular, stereo or RGB-D) or DPVO (monocular)).

## Convert quaternion to rotation matrix
Normally, the estimated camera orientation of SLAM or VO methods are in form of quaternion. They have to be converted into rotation matrix. Modify in the script **quat_to_rot.py** the path to load the pose from SLAM or VO methods and the path to save the output pose matrix. scipy is used for the conversion. Note that quaternion calculated by lietorch (DPVO for example) has to be reordered to be converted correctly by scipy.
## 3D reconstruction pipeline
### Make fragments
The first step is to build local geometric surfaces (referred to as fragments) from short subsequences of the input RGBD sequence. Modify in the script **fragment.py** the camera config and the path to RGB images, depth images and poses as well as path to save output fragments. Run:
```
python fragment.py
```

### Register fragments
 The next step is that the fragments are aligned in a global space to decrease the accumulated error in estimated pose caused by SLAM or VO methods. Modify in the script **registration.py** the path to load fragments and the path to save the transformation matrix calculated by registration methods (ICP). Run:
```
python registration.py
```

### Integrate scene
The final step is to integrate RGB-D images to generate a mesh model for the scene.Modify in the script **reconstruction.py** the camera config and the path to RGB images, depth images and poses as well as path to load the registration matrix from step 2. Run: 
```
python reconstruction.py
```

## Experimental scripts
- **demo_open3d_registration.py**: Test and visualize the registration between 2 fragments.
- **demo_open3d_3d_reconstruction.py**: Test and visualize the integrated 3D volume of a scene
- **demo_open3d_load3d.py**: Visualize a 3D volume with camera trajectory
