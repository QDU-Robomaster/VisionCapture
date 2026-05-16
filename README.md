# VisionCapture

`VisionCapture` 用于采集 `CameraFrameSync` 输出的同步图像和 IMU 数据，并在需要时完成标定采样。它通常只出现在采集或标定配置中，不参与常规自瞄运行配置。

## 运行模式

- `record`：保存同步图像和每帧元数据。
- `calibrate_camera`：判稳后保存标定样本，并用同一批图像求解相机内参。
- `calibrate_handeye`：判稳后保存手眼标定样本；当前只保存数据，不求解手眼结果。
- `calibrate`：同时保存标定样本并运行相机内参标定，数据也可用于后续手眼标定。

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

## 采样判稳

`calibration_sampling` 用于筛选相机和 IMU 都稳定的样本。它会检查：

- PnP 重投影 RMS。
- PnP 平移和旋转抖动。
- IMU 四元数抖动。
- 陀螺仪模长。
- 加速度模长、模长抖动和方向抖动。
- 样本之间的位移和角度变化。

该采样结果直接决定 `frames/` 和 `samples.csv` 中保存哪些样本。当前模块会采集和保存手眼标定数据，但不求解手眼标定结果。

## 本地命令

`control.stdin_enabled: true` 时，标准输入可使用以下命令：

- `start`：开始采样。
- `pause` / `stop`：暂停采样。
- `reset`：清空本轮采样状态。
- `snapshot`：请求保存一帧。
- `status`：打印当前采样状态。
- `solve`：立即尝试相机内参求解，并打印当前手眼样本数。
- `help`：打印命令列表。
