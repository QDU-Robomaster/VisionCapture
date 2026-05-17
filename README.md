# VisionCapture

`VisionCapture` 用于采集 `CameraFrameSync` 输出的同步图像和 IMU 数据，并在需要时完成标定采样。它通常只出现在采集或标定配置中，不参与常规自瞄运行配置。

## 运行模式

- `record`：保存同步图像和每帧元数据。
- `calibrate_camera`：判稳后保存标定样本，并用同一批图像求解相机内参。
- `calibrate_handeye`：判稳后保存手眼标定样本，并在 `solve` 时求解手眼外参。
- `calibrate`：同时保存标定样本，并在 `solve` 时运行相机内参和手眼标定。

`camera_calibration.enabled: true` 时，即使 `mode` 仍为 `record`，也会进入标定采样保存流程。

## 记录内容

记录目录由 `output_dir` 和 `session_name` 决定。`session_name` 为空时，模块自动生成带时间戳的目录名。

`record` 模式按 `record.*` 配置保存。所有标定模式会固定保存通过判稳的样本，不依赖 YAML 里是否打开 `record.enabled`。

典型输出包括：

- `samples.csv`：同步图像时间戳、IMU 时间戳、四元数、角速度、加速度、图像文件名、标定板检测结果和采样判定。
- `frames/`：保存的原始图像。
- `camera_info.txt`：本次运行使用的编译期 CameraInfo。
- `preview`：可选窗口或 Web 预览。

标定模式只保存通过判稳的样本；被判稳拒绝的帧不会写入 `frames/` 和 `samples.csv`。

## 相机内参标定

相机内参标定使用 GShang 标定板和 ArUco original 字典。相机标定器只接收通过判稳的图像，因此它和手眼标定使用同一批已保存样本。配置项：

- `camera_calibration.marker_size_mm`：marker 黑色码区边长，单位 mm。
- `camera_calibration.cols`：标定板棋盘列数。
- `camera_calibration.rows`：标定板棋盘行数。
- `camera_calibration.auto_save_views`：达到该数量后自动求解并保存结果。

输出目录格式：

```text
runs/camera_calib/<timestamp>_<session>_<marker>mm_<cols>x<rows>/
```

标定结果包括：

- `calibration.yml`
- `views.csv`
- `quality_report.txt`
- `camera_info_snippet.txt`
- 调试图像

`camera_info_snippet.txt` 会打印可粘贴到 `CameraInfo` 配置中的内参、畸变和投影矩阵。写入配置前应检查焦距、主点、畸变系数和重投影 RMS 是否合理。

## 手眼标定

手眼标定继续使用当前配置的 ArUco/GShang 标定板检测和 PnP 结果，不切换到圆点板。`solve` 命令会用已接受样本调用 OpenCV `calibrateRobotWorldHandEye`，假设每帧 `t_world2body = 0`。同步 IMU 四元数默认按公开本体系 `B` 表达：`x` 向右、`y` 向前、`z` 向上；如果实机发布的四元数仍有 IMU 安装轴差，可通过 `handeye_calibration.R_body2imu` 配置。

输出文件位于当前采集目录：

- `handeye.yml`：OpenCV FileStorage 格式结果，包含 `R_camera2body`、`t_camera2body`，并保留兼容字段 `R_camera2gimbal`、`t_camera2gimbal`，平移单位为 m。
- `handeye_report.txt`：便于人工检查的文本报告。

报告中同时给出可填入 `ArmorTracker.extrinsic.camera_mount_to_body` 的 `rotation(wxyz)` 和 `translation`。`handeye_calibration.min_samples` 控制求解所需的最少稳定样本数，默认 `6`。

## 采样判稳

`calibration_sampling` 用于筛选相机和 IMU 都稳定的样本。它会检查：

- PnP 重投影 RMS。
- PnP 平移和旋转抖动。
- IMU 四元数抖动。
- 陀螺仪模长。
- 加速度模长、模长抖动和方向抖动。
- 样本之间的位移和角度变化。

该采样结果直接决定 `frames/`、`samples.csv` 和手眼求解输入中保存哪些样本。

## 本地命令

`control.stdin_enabled: true` 时，标准输入可使用以下命令：

- `start`：开始采样。
- `pause` / `stop`：暂停采样。
- `reset`：清空本轮采样状态。
- `snapshot`：请求保存一帧。
- `status`：打印当前采样状态。
- `solve`：立即尝试相机内参求解和手眼标定求解。
- `help`：打印命令列表。
