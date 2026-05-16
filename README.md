# VisionCapture

`VisionCapture` 用于采集 `CameraFrameSync` 输出的同步图像和 IMU 数据，并在需要时完成相机内参标定采样。它通常只出现在采集或标定配置中，不参与常规自瞄运行配置。

## 运行模式

- `record`：保存同步图像和每帧元数据。
- `calibrate_camera`：检测 GShang/ArUco 标定板，自动筛选有效视角，样本足够后求解相机内参。

`camera_calibration.enabled: true` 时，即使 `mode` 仍为 `record`，也会启用相机内参标定流程。

## 记录内容

记录目录由 `output_dir` 和 `session_name` 决定。`session_name` 为空时，模块自动生成带时间戳的目录名。

典型输出包括：

- `metadata.csv`：同步图像时间戳、IMU 时间戳、时间差和图像文件名。
- `raw_imu.csv`：每帧附近的原始 IMU 数据。
- `images/`：按配置保存的图像文件。
- `preview`：可选窗口或 Web 预览。

`record.flush_every_n` 控制 CSV 刷盘间隔。实车采集建议保持默认值 `1`，避免异常退出后丢失太多元数据。

## 相机内参标定

相机内参标定使用 GShang 标定板和 ArUco original 字典。配置项：

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

该采样结果可用于后续手眼标定数据整理。当前模块会采集和保存数据，但不求解手眼标定结果。

## 本地命令

`control.stdin_enabled: true` 时，标准输入可使用以下命令：

- `start`：开始采样。
- `pause` / `stop`：暂停采样。
- `reset`：清空本轮采样状态。
- `snapshot`：请求保存一帧。
- `status`：打印当前采样状态。
- `solve`：相机内参模式下立即求解；手眼数据模式下只打印当前样本数。
- `help`：打印命令列表。
