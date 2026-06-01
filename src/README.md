# Visual Localization Pipeline

Estimates the AUV camera pose (position + orientation) relative to a known underwater
board, using XFeat sparse feature detection and LightGlue feature matching accelerated
via TensorRT.

---

## Algorithm Overview

```
Query image (camera)
    │
    ├─► XFeat TRT ──► query keypoints + descriptors
    │
    │   Reference image (front-facing, loaded once at startup)
    ├─► XFeat TRT ──► reference keypoints + descriptors
    │
    ├─► LightGlue TRT ──► matched (ref_kp_i  ↔  query_kp_i) pairs
    │
    ├─► perspectiveTransform(H_ref→board)
    │       ref pixel (x, y)  ──►  board coordinate (X_m, Y_m, 0)
    │
    └─► solvePnPRansac
            3D board points  +  2D query pixels  ──►  camera pose
```

### Step 1 — Feature extraction

XFeat runs on both images and returns sparse keypoints with 64-D descriptors.
The reference image is processed once at node startup and cached.

### Step 2 — Feature matching

LightGlue matches the reference descriptors against the query descriptors and
returns a list of confident correspondence pairs `(idx_ref, idx_query)`.

### Step 3 — 2D-to-3D lifting

The reference image is captured front-facing and undistorted, so each pixel maps
linearly to a physical coordinate on the board. A homography `H_ref→board` is
pre-computed from the four image corners to the four board corners (in metres).
For every matched reference keypoint, `cv::perspectiveTransform` applies this
homography to produce a 3D board point `(X_m, Y_m, Z=0)`.

### Step 4 — Pose estimation

`cv::solvePnPRansac` receives the list of `(3D board point, 2D query pixel)` pairs
and solves for the rotation `R` and translation `t` that map board coordinates into
the camera frame:

```
p_camera = R * p_board + t
```

The inverse is then computed to express the camera pose in the board frame:

```
R_cam_in_board = R^T
t_cam_in_board = -R^T * t
```

The rotation is converted to a quaternion via Eigen and published as a
`geometry_msgs/PoseStamped`.

---

## Prerequisites

| Dependency | Version |
|---|---|
| NVIDIA JetPack | 5.1.3 |
| CUDA | 11.4 |
| TensorRT | 8.5.2 |
| OpenCV | 4.x |
| LibTorch (PyTorch C++) | 2.1.0 |
| ROS 2 | Humble |
| Eigen3 | any |
| yaml-cpp | any |

---

## Setup

### 1. Export ONNX models

```bash
cd XFeat-Lightglue-TRT

# XFeat — 480x640 grayscale input, up to 512 keypoints
python3 scripts/export.py \
    --xfeat_only_model \
    --height 480 --width 640 \
    --top_k 512 \
    --split_instance_norm \
    --export_path ./weights/xfeat_1_480_640.onnx

# LightGlue — 480x640, up to 512 match pairs
python3 scripts/export.py \
    --xfeat_only_lighterglue \
    --height 480 --width 640 \
    --top_k 512 \
    --export_path ./weights/lightglue_L6_1_480_640.onnx
```

### 2. Convert ONNX to TensorRT engines

```bash
# XFeat engine (FP16 recommended on Jetson)
/usr/src/tensorrt/bin/trtexec \
    --onnx=weights/xfeat_1_480_640.onnx \
    --saveEngine=weights/xfeat_1_480_640.engine \
    --fp16

# LightGlue engine
/usr/src/tensorrt/bin/trtexec \
    --onnx=weights/lightglue_L6_1_480_640.onnx \
    --saveEngine=weights/lightglue_L6_1_480_640.engine \
    --fp16
```

### 3. Prepare the reference image

Place a front-facing, undistorted photo of the board at:

```
config/reference/board_reference.png
```

The image must be captured with the camera perpendicular to the board surface.
The four corners of the image are assumed to correspond exactly to the four
corners of the physical board.

### 4. Build the ROS 2 package

```bash
cd ~/auv_ws
colcon build --packages-select xfeat_lightglue_trt
source install/setup.bash
```

---

## Configuration

Edit `config/localizer.param.yaml` and `config/xfeat_lightglue.yaml` to match
your setup.

### `config/localizer.param.yaml`

| Parameter | Default | Description |
|---|---|---|
| `image_width` | `640` | Camera image width (pixels) — must match engine |
| `image_height` | `480` | Camera image height (pixels) — must match engine |
| `board_width_m` | `0.60` | Physical board width (metres) |
| `board_height_m` | `0.40` | Physical board height (metres) |
| `pnp_reproj_thresh` | `3.0` | solvePnP RANSAC reprojection error threshold (pixels) |
| `min_inliers` | `10` | Minimum RANSAC inliers required to accept a pose estimate |

### `config/xfeat_lightglue.yaml`

| Parameter | Default | Description |
|---|---|---|
| `image_width` | `640` | Must match the value used during ONNX export |
| `image_height` | `480` | Must match the value used during ONNX export |
| `max_keypoints` | `512` | Maximum keypoints returned by XFeat |
| `feat_threshold` | `0.05` | NMS keypoint confidence threshold |
| `match_threshold` | `0.7` | LightGlue minimum match confidence |

---

## Launch

```bash
ros2 launch xfeat_lightglue_trt localizer.launch.py \
    camera_topic:=/camera/image_raw \
    camera_info_topic:=/camera/camera_info \
    pose_topic:=/auv/board_pose
```

---

## ROS 2 Interface

### Subscribed topics

| Topic | Type | Description |
|---|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` | Live camera frames (BGR8) |
| `/camera/camera_info` | `sensor_msgs/CameraInfo` | Camera intrinsics and distortion |

### Published topics

| Topic | Type | Description |
|---|---|---|
| `/auv/board_pose` | `geometry_msgs/PoseStamped` | Camera pose in the board frame |

### Coordinate frames

- **Board frame origin**: top-left corner of the physical board.
- **X axis**: points right along the board width.
- **Y axis**: points down along the board height.
- **Z axis**: points out of the board surface toward the camera.
- The published pose gives the camera position and orientation expressed in this frame.
