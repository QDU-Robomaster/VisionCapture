# VisionCapture

`VisionCapture` 用于采集 `CameraFrameSync` 输出的同步图像和 IMU 数据，并在需要时完成标定采样。它通常只出现在采集或标定配置中，不参与常规自瞄运行配置。

## 运行模式

- `record`：保存同步图像和每帧元数据。
- `calibrate_camera`：判稳后保存标定样本，并用同一批图像求解相机内参。
- `calibrate_handeye`：判稳后保存手眼标定样本；当前只保存数据，不求解手眼结果。
- `calibrate`：`calibrate_camera` 的旧名称，只运行相机内参标定。

`record` 配合 `camera_calibration.enabled: true` 也会进入内参标定。显式
`calibrate_handeye` 的优先级最高，不会因该开关同时启动内参求解。

## 记录内容

记录目录由 `output_dir` 和 `session_name` 决定。`session_name` 为空时，模块自动生成带时间戳的目录名。

`record` 模式按 `record.*` 配置保存。所有标定模式会固定保存通过判稳的样本，不依赖 YAML 里是否打开 `record.enabled`。

典型输出包括：

- `samples.csv`：同步图像时间戳、IMU 时间戳、可选原始 IMU、图像文件名、标定板检测结果和采样判定。
- `frame_geometry.csv`：每个成功记录帧对应的完整 ROI、下采样、翻转、保留字段和采样相位，可直接区分混合 WIDE/NARROW 记录。
- `frames/`：保存的原始图像。
- `frame_layout.txt`：编译期固定的帧存储布局。
- `camera_calibration.txt`：原生传感器坐标系下的固定标定。
- `frame_geometry.txt`：首个有效帧携带的 ROI、下采样、翻转和采样相位，保留给旧记录工具。
- `camera_info.txt`：由上述三项派生的帧坐标兼容快照，保留给旧记录工具。
- `preview`：可选窗口或 Web 预览，显示拒绝帧、检测/重投影点、判定原因、视觉覆盖、
  采样/求解进度以及当前帧 geometry/profile。

标定模式只保存通过判稳的样本；被判稳拒绝的帧不会写入 `frames/` 和 `samples.csv`。
标定模式若配置 `calibration_sampling.enabled: false` 会以 `sampling_disabled` 拒绝，
不会退化为无条件记录。手眼模式始终写入原始 IMU，即使 YAML 将 `save_raw_imu`
设为 `false`；内参和普通记录模式仍按该配置决定是否写入原始 IMU。
图像写盘失败的帧不会写入两个 CSV，也不会计入已保存帧数。`save_raw_imu: false`
会保留 `samples.csv` 固定列结构并将十个原始 IMU 数值单元格留空；`flush_every_n: 0`
表示仅在文件流关闭时刷盘，否则每成功记录指定行数后同时刷新两个 CSV。
相机和 MCU 时间戳属于不同时间域，`samples.csv` 分别保留两者，旧 `dt_us` 列恒为空；
原始加速度沿用 `CameraBase::ImuStamped` 的固定 `m/s^2` 契约并在末列明确记录单位。

## 相机内参标定

两个标定模式都固定使用 GShang 25 mm、8x6 标定板和 ArUco original 字典。内参
标定器只接收通过纯视觉门限的图像，不读取旧 K/D、PnP 或 IMU。配置项：

- `camera_calibration.marker_size_mm/cols/rows`：为旧 YAML 保留；标定模式会规范为
  `25.0/8/6`。
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

`camera_info_snippet.txt` 文件名为兼容旧产物而保留，内容是可粘贴到 xrobot YAML 的
`MainFrameLayout` 与原生 `CameraCalibration` 片段。写入配置前应检查原生尺寸、焦距、
主点、畸变系数和重投影 RMS 是否合理。离群阈值、`rms`、`views.csv` 和质量报告中的
重投影误差均使用当前帧像素；`calibration.yml` 另存 `native_rms` 供原生坐标诊断。
只有视角数、中心/尺度覆盖、内参合理性和全局/逐视角重投影误差全部通过
`quality_ok`，且所有输出逐字节写后读回成功时，求解才返回成功并生成
`calibration.yml` 与 `camera_info_snippet.txt`。质量失败仍保留
`views.csv` 和 `quality_report.txt`，但不会打印 PASS 或生成可粘贴配置。
自动保存和 stdin `solve` 共用一次求解/写盘 claim；并发重复请求等待并复用同一结果，
不会同时截断或覆盖同一路径。

## 采样判稳

内参模式只检查 GShang marker 数、单应性 RMS、清晰度、相机时间戳间隔，以及中心、
尺度和角度的视觉重复；不会被 IMU 运动或跨时钟差值拒绝。

手眼模式使用冻结的原生 K/D 执行畸变感知 PnP，并检查：

- PnP 重投影 RMS。
- PnP 平移和旋转抖动。
- IMU 四元数抖动。
- 陀螺仪模长。
- 加速度模长、模长抖动和方向抖动。
- 样本之间的位移和角度变化。

手眼输入必须有非零 MCU 时间戳、有限且可归一化的四元数、有限角速度和有限
`m/s^2` 加速度。约为 `1.0` 的历史 `g` 量纲输入会以 `acc_unit_or_scale` 明确拒绝，
不会在本模块静默换算。该采样结果直接决定 `frames/` 和 `samples.csv` 保存哪些样本；
当前模块只采集手眼数据，不求解手眼结果。

## 本地命令

`control.stdin_enabled: true` 时，标准输入可使用以下命令：

- `start`：开始采样。
- `pause` / `stop`：暂停采样。
- `reset`：清空本轮采样状态。
- `snapshot`：请求保存一帧。
- `status`：打印当前采样状态。
- `solve`：内参模式立即尝试求解；手眼模式只打印当前样本数。
- `help`：打印命令列表。
