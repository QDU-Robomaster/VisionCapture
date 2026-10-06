# VisionRecorder

录像：把同步帧写成统一录像格式（逐帧 PGM 与 frames.csv），供回放与离线标定 / Recorder that writes synced frames in the unified recording format (per-frame PGM and frames.csv) for replay and offline calibration

## 1. 模块作用 / Purpose

VisionRecorder 订阅 `<相机名>_synced`，把每帧的原始 BayerRG8 图像、帧几何和同步的 IMU 写到磁盘。回调只把字节复制进模块自己的环形缓冲，写盘在单独的线程里进行；缓冲满时丢帧并计数，不会拖慢采集。模块不持有图像句柄，不占相机的图像槽。录下的目录可以直接交给 CaptureFileCamera 回放，也是离线标定工具的输入。

VisionRecorder subscribes to `<camera>_synced` and writes each frame's raw BayerRG8 image, frame geometry and synced IMU to disk. The callback only copies the bytes into the Module's own ring buffer and a separate thread writes them; when the buffer is full frames are dropped and counted, so capture is never slowed. No image handle is kept, so no camera image slot is held. A recorded directory replays directly in CaptureFileCamera and is the input of the offline calibration tools.

## 2. 录像格式 / Recording Format

每次启动在 `output_dir` 下新建一个以本地时间命名的目录（如 `20261006-222401`）：

Each start creates a directory named by local time (such as `20261006-222401`) under `output_dir`:

```text
session.txt   相机名与第一帧携带的标定（回放不读）/ camera name and the calibration of the first frame (not read by replay)
frames.csv    frame,timestamp_us,frame_counter,roi_x,roi_y,decimation,qw,qx,qy,qz,gx,gy,gz,ax,ay,az
000000.pgm    640×512 P5 8 位 BayerRG8，(0,0) 为 R / 8-bit BayerRG8, R at (0,0)
000001.pgm
…
```

`frame` 从 0 连续编号，对应 PGM 文件名；`timestamp_us` 是图像的传感器时间，IMU 是 CFS 配给这一帧的样本（四元数为本体系到世界系，角速度 rad/s，加速度 m/s²，本体系 x 右、y 前、z 上）。浮点按 9 位有效数字写，读回与原值逐位相同。

`frame` counts from 0 without gaps and names the PGM file; `timestamp_us` is the image's sensor time, and the IMU is the sample CFS paired with the frame (body-to-world quaternion, rad/s, m/s², body x right, y forward, z up). Floats are written with nine significant digits and read back bit for bit.

## 3. 配置示例 / Configuration Example

```yaml
modules:
  - module: QDU-Robomaster/VisionRecorder
    id: recorder
    args:
      - settings:
          camera_name: "gimbal"
          record: true
          output_dir: "/home/robot/recordings"
          max_fps: 0
          max_frames: 0
          buffer_frames: 32
```

`record` 为 false 时模块不订阅也不写盘。`max_fps` 为 0 时每帧都存，否则按图像时间限速（例如 100 Hz 输入、上限 50 时隔帧保存）。`max_frames` 为 0 时不限帧数。每个缓冲帧约 0.33 MB。

With `record` false the Module neither subscribes nor writes. With `max_fps` 0 every frame is kept; otherwise frames are thinned by image time (100 Hz in with a cap of 50 keeps every other frame). `max_frames` 0 means no limit. Each buffered frame takes about 0.33 MB.

## 4. 离线标定 / Offline Calibration

`tools/` 下的脚本在录像上做标定，需要 Python 3.10+、NumPy 与 OpenCV（含 aruco）。标定板为 GShang 板：8×6 个棋盘格，白格中印 DICT_ARUCO_ORIGINAL marker，marker 边长 25 mm（棋盘格 32.14 mm）。工具用 marker 识别出板，再取棋盘格的内角点（鞍点）求解；每帧按自己的几何换算到原生像素，所以 WIDE 与 NARROW 的录像可以混用。

The scripts under `tools/` calibrate on recordings and need Python 3.10+, NumPy and OpenCV (with aruco). The board is the GShang board: 8×6 chessboard squares with DICT_ARUCO_ORIGINAL markers printed in the white squares, 25 mm markers (32.14 mm squares). The tools identify the board by its markers and solve with the inner chessboard corners (saddle points); each frame is mapped to native pixels by its own geometry, so WIDE and NARROW recordings can be mixed.

### 4.1 内参 / Intrinsics

录一段手持标定板在相机前移动的录像：板要覆盖画面四角与中间，并有 30–45° 的倾斜；距离以 marker 在画面上不小于约 15 像素为宜。

Record the board held in front of the camera and moved around: cover the corners and the centre of the view and include tilts of 30–45°; keep markers at least about 15 pixels wide in the image.

```bash
python tools/calibrate_intrinsics.py <录像目录> --out intrinsics.yaml
```

输出 `intrinsics.yaml`（`native_width`、`native_height`、`fx`、`fy`、`cx`、`cy`、`distortion`）并打印可粘贴的 `CameraTypes::CameraCalibration` 初始化式，以及 fx、fy、cx、cy 的标准差。焦距标准差超过 0.5% 时会给出警告，说明视角不够多样，需要补录倾斜的视角。8 mm 等窄视场镜头可加 `--fix-k3`。

It writes `intrinsics.yaml` (`native_width`, `native_height`, `fx`, `fy`, `cx`, `cy`, `distortion`) and prints a `CameraTypes::CameraCalibration` initializer to paste, together with the standard deviations of fx, fy, cx and cy. A focal standard deviation above 0.5% produces a warning that the views lack variety and tilted views should be added. Narrow lenses such as 8 mm can add `--fix-k3`.

### 4.2 相机安装（手眼）/ Camera Mounting (Hand-Eye)

标定板固定不动；云台依次转到不同的偏航、俯仰（彼此相差 5° 以上，总跨度 10° 以上），每个姿态停住约半秒。录像需要带 IMU 列（CFS 的同步帧都带）。

Fix the board; turn the gimbal through different yaw and pitch attitudes (at least 5° apart, spanning more than 10°) and hold each for about half a second. The recording needs the IMU columns, which every CFS synced frame carries.

```bash
python tools/calibrate_handeye.py <录像目录> --intrinsics intrinsics.yaml --out mount.yaml
```

工具按陀螺找出静止段，每段取中间一帧求板的位姿，先用 `cv2.calibrateRobotWorldHandEye` 得到初值，再把所有姿态的重投影误差一起最小化（单帧位姿的倾角误差约 1°，在这一步被平均掉）。输出 ArmorTracker 的 `mount_rotation_wxyz` 与 `mount_translation`；云台绕固定点转动时平移只能粗略确定，旋转是主要结果。

The tool finds still segments from the gyro and solves the board pose in the middle frame of each, takes `cv2.calibrateRobotWorldHandEye` as the start, then minimises the reprojection error of all poses together (the tilt error of a single-frame pose, about 1°, averages out there). It writes ArmorTracker's `mount_rotation_wxyz` and `mount_translation`; with the gimbal rotating about a fixed point the translation is only roughly determined, and the rotation is the main result.

### 4.3 合成检查 / Synthetic Check

`python tools/test_tools.py` 按已知内参与安装渲染标定板录像，再用两个工具解回来：内参 60 个视角（WIDE 与 NARROW 混合）解出的焦距误差约 0.2%、主点误差约 3 像素、畸变在画面内造成的位移小于 0.4 像素；手眼 15 个姿态解出的安装旋转误差约 0.02°。

`python tools/test_tools.py` renders board recordings with known intrinsics and mounting and solves them back with both tools: 60 views (WIDE and NARROW mixed) give the focal length within about 0.2%, the principal point within about 3 pixels and a distortion displacement below 0.4 pixels over the view; 15 poses give the mount rotation within about 0.02°.

## 5. 测试 / Tests

`tests/recorder_test.cpp` 检查 frames.csv 的表头与内容、PGM 的头与像素、session.txt、50 fps 上限（100 Hz 输入存一半）与帧数上限。

`tests/recorder_test.cpp` checks the frames.csv header and rows, the PGM header and pixels, session.txt, the 50 fps cap (half of a 100 Hz input) and the frame limit.

## 6. 依赖 / Dependencies

AutoAimTypes（含 CameraBase）、LibXR。离线工具另需 Python、NumPy、OpenCV。

AutoAimTypes (with CameraBase), LibXR. The offline tools also need Python, NumPy and OpenCV.
