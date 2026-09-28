# VisionCapture

`VisionCapture` 用于采集 `CameraFrameSync` 输出的同步图像和 IMU 数据，并在需要时完成标定
采样。它通常只出现在采集或标定配置中，不参与常规自瞄运行配置。

## 运行模式

- `record`：保存同步图像和每帧元数据。
- `calibrate_camera`：判稳后保存标定样本，并用同一批图像求解相机内参。
- `calibrate_handeye`：判稳后保存手眼标定样本；当前只保存数据，不求解手眼结果。
- `calibrate`：`calibrate_camera` 的别名，只运行相机内参标定。

`record` 配合 `camera_calibration.enabled: true` 也会进入内参标定。显式
`calibrate_handeye` 的优先级最高，不会因该开关同时启动内参求解。

## 记录内容

记录目录为 `<output_dir>/<session_name>`。`session_name` 为空时，模块自动生成
`vision_capture_<YYYYmmdd_HHMMSS>` 目录名。

`record` 模式按 `record.*` 配置保存。所有标定模式会固定保存通过判稳的样本（强制打开记录、
图像和元数据，不限帧率），不依赖配置里是否打开 `record.enabled`。

典型输出包括：

- `samples.csv`：同步图像时间戳、IMU 时间戳、可选原始 IMU、图像文件名、标定板检测结果和
  采样判定。
- `frame_geometry.csv`：每个成功记录帧对应的完整 ROI、下采样、翻转、保留字段和采样相位，
  可直接区分混合 WIDE/NARROW 记录。
- `frames/`：保存的原始图像（格式由 `record.image_format` 决定）。
- `frame_layout.txt`：编译期固定的帧存储布局。
- `camera_calibration.txt`：原生传感器坐标系下的固定标定。
- `frame_geometry.txt`：首个有效帧携带的 ROI、下采样、翻转和采样相位，保留给旧记录工具。
- `camera_info.txt`：由上述三项派生的帧坐标兼容快照，保留给旧记录工具。
- `preview`：可选窗口或 Web 预览（见 VisionPreview），显示拒绝帧、检测/重投影点、判定原因、
  视觉覆盖、采样/求解进度以及当前帧 geometry/profile。

标定模式只保存通过判稳的样本；被判稳拒绝的帧不会写入 `frames/` 和 `samples.csv`。
标定模式若配置 `calibration_sampling.enabled: false` 会以 `sampling_disabled` 拒绝，
不会退化为无条件记录。手眼模式始终写入原始 IMU，即使配置将 `save_raw_imu`
设为 `false`；内参和普通记录模式仍按该配置决定是否写入原始 IMU。
图像写盘失败的帧不会写入两个 CSV，也不会计入已保存帧数。`save_raw_imu: false`
会保留 `samples.csv` 固定列结构并将十个原始 IMU 数值单元格留空；`flush_every_n: 0`
表示仅在文件流关闭时刷盘，否则每成功记录指定行数后同时刷新两个 CSV。
相机和 MCU 时间戳属于不同时间域，`samples.csv` 分别保留两者，`dt_us` 列恒为空；
原始加速度沿用 `CameraBase::ImuStamped` 的固定 `m/s^2` 契约并在末列明确记录单位。
任一必需记录或 CSV 刷盘失败都会锁存会话失败、清空内参求解视角并阻止 `solve`
返回 PASS；只有图像和元数据完整持久化后，该帧才会提交给内参求解器。

同步 Topic 回调只 retain `SharedFrame` 并写入容量为 2 的 drop-oldest 队列，OpenCV、
预览深拷贝和磁盘 I/O 全部在对象拥有的 worker 中执行，不反压 CameraFrameSync 发布
线程。析构会先停止并 join 可取消的 stdin 控制读取器，再停止并 join 帧 worker；由于
LibXR Topic 当前没有回调注销接口，调用方必须先停止上游发布再析构 `VisionCapture`。
支持 BGR8、RGB8、BGRA8、RGBA8 和 MONO8 输入；RGB8/RGBA8 会在 worker 内分别转换为
OpenCV 的 BGR/BGRA 约定后再检测、预览和写图。worker 处理异常会关闭队列、释放
待处理帧并锁存会话失败，不会让异常越过线程入口终止进程。

`OnMonitor()` 打印自上次调用以来的帧数、保存数、检测到标定板的帧数、PnP 成功数、接受/拒绝
样本数、队列丢弃数，以及当前模式和采样状态。

## 相机内参标定

两个标定模式都固定使用 GShang 25 mm、8x6 标定板和 ArUco original 字典（`board.*` 只用于
普通记录模式的检测）。内参标定器只接收通过纯视觉门限的图像，不读取旧 K/D、PnP 或 IMU。
配置项：

- `camera_calibration.marker_size_mm/cols/rows`：标定模式会规范为 `25.0/8/6`，
  配置不同时打印警告。
- `camera_calibration.auto_save_views`：达到该数量后自动求解并保存结果；`0` 表示只由
  `solve` 命令触发。

输出目录格式（相对进程工作目录）：

```text
runs/camera_calib/<timestamp>_<session>_<marker>mm_<cols>x<rows>/
```

标定结果包括：

- `calibration.yml`
- `views.csv`
- `quality_report.txt`
- `camera_info_snippet.txt`
- `debug/` 调试图像

`camera_info_snippet.txt` 的内容是可粘贴到 BSP `User/xrobot.yaml` 的 `constexprs` 片段：
`MainFrameLayout`（`CameraTypes::FrameLayout`）与原生 `MainCameraCalibration`
（`CameraTypes::CameraCalibration`），值以 YAML map 写出。写入配置前应检查原生尺寸、焦距、
主点、畸变系数和重投影 RMS 是否合理。离群阈值、`rms`、`views.csv` 和质量报告中的
重投影误差均使用当前帧像素；`calibration.yml` 另存 `native_rms` 供原生坐标诊断。
只有视角数、中心/尺度覆盖、最终外参恢复出的双轴标定板倾斜跨度、内参合理性和
全局/逐视角重投影误差全部通过 `quality_ok`，且所有输出逐字节写后读回成功时，求解才返回
成功并生成 `calibration.yml` 与 `camera_info_snippet.txt`。质量失败仍保留
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
- 加速度模长（与 9.80665 m/s^2 比较）、模长抖动和方向抖动。
- 样本之间的时间间隔、位移和角度变化。

手眼输入必须有非零 MCU 时间戳、有限且可归一化的四元数、有限角速度和有限
`m/s^2` 加速度。约为 `1.0` 的 `g` 量纲输入会以 `acc_unit_or_scale` 明确拒绝，
不会在本模块静默换算。该采样结果直接决定 `frames/` 和 `samples.csv` 保存哪些样本；
当前模块只采集手眼数据，不求解手眼结果。

## 本地命令

`control.stdin_enabled: true` 时，标准输入可使用以下命令：

- `start`：开始采样（`calibration_sampling.auto_start: true` 时启动即开始）。
- `pause` / `stop`：暂停采样。
- `reset`：清空判稳窗口、已接受样本、求解器视角和完成状态。
- `snapshot`：请求强制接受下一帧通过质量门限的样本（跳过间隔和重复检查）；被拒绝帧不会
  消费请求，并发新请求不会丢失。
- `status`：打印当前采样状态。
- `solve`：内参模式立即尝试求解；手眼模式只打印当前样本数。
- `help`：打印命令列表。

## 依赖

- `QDU-Robomaster/CameraFrameSync`：同步帧输入（`SyncedFrame`）和原生标定。
- `QDU-Robomaster/VisionPreview`：预览输出。
- `QDU-Robomaster/CameraBase`：帧布局、geometry 与共享图像类型。
- 外部：OpenCV 4（`core`、`imgproc`、`imgcodecs`、`calib3d`、`aruco`）。

## 构造接口

```cpp
template <CameraTypes::FrameLayout FrameLayoutV>
class VisionCapture;

VisionCapture(
    Sync& sync,
    Config cfg = DefaultConfig());
```

模板参数：

- `FrameLayoutV`：帧布局，必须与上游 CameraFrameSync 和相机相同。

依赖：

- `sync`：`CameraFrameSync<FrameLayoutV>&`，即前面的 CameraFrameSync 实例；模块订阅其
  `SyncedFrameTopicName()`，并复制其原生标定。

配置 `cfg`（`Config`，`DefaultConfig()` 即全部默认值）：

- `mode`：`"record"`（默认）、`"calibrate_camera"`、`"calibrate_handeye"` 或 `"calibrate"`。
- `output_dir`：输出根目录，默认 `"runs/vision_capture"`。
- `session_name`：会话名，默认为空（自动生成）。
- `record`（`RecordParams`）：`enabled = true`、`image_format = "bmp"`、
  `max_fps = 30.0`（`0` 不限）、`max_frames = 0`（不限）、`save_images = true`、
  `save_metadata = true`、`save_raw_imu = true`、`flush_every_n = 1`。
- `preview`（`VisionPreview::RuntimeParam`）：默认关闭，字段见 VisionPreview。
- `board`（`BoardParams`）：普通记录模式的检测参数，`type = "aruco"`、
  `dictionary = "DICT_5X5_100"`、`marker_length_m = 0.04`。
- `camera_calibration`（`CameraCalibrationParams`）：`enabled = false`、
  `marker_size_mm = 25.0`、`cols = 8`、`rows = 6`、`auto_save_views = 120`。
- `calibration_sampling`（`CalibrationSamplingParams`）：`enabled = true`、
  `auto_start = true`、`window_size = 8`、`min_accept_interval_us = 500000`，以及手眼门限
  `max_pnp_reprojection_rms_px = 2.0`、`max_pnp_translation_jitter_m = 0.005`、
  `max_pnp_rotation_jitter_deg = 1.0`、`max_imu_rotation_jitter_deg = 0.8`、
  `max_gyro_norm_dps = 2.0`、`max_acc_norm_error_mps2 = 1.5`、
  `max_acc_norm_jitter_mps2 = 0.5`、`max_acc_direction_jitter_deg = 2.0`、
  `min_sample_translation_delta_m = 0.03`、`min_sample_rotation_delta_deg = 5.0`。
- `control`（`ControlParams`）：`stdin_enabled = false`。
- `filter`（`FilterParams`）：`require_synced_imu`、`max_image_imu_dt_us` 仅为兼容保留，
  当前实现不使用。

## 使用

```sh
xrobot module add QDU-Robomaster/VisionCapture
xrobot setup
xrobot instance add QDU-Robomaster/VisionCapture
```

`xrobot instance add` 在 `User/xrobot.yaml` 中写入一个实例，依赖项留空，默认值按源码写出；
把 `sync` 填为前面 CameraFrameSync 实例的 id。帧布局用 constexpr 定义，必须与相机输出一致：

```yaml
constexpr_includes:
  - CameraBase.hpp
constexprs:
  FrameLayout:
    type: CameraTypes::FrameLayout
    value: '{.width = 640, .height = 480, .step = 1920, .encoding = CameraTypes::Encoding::BGR8}'
modules:
  - module: QDU-Robomaster/VisionCapture
    id: visioncapture_0
    template_args:
      - ProjectConstexpr::FrameLayout
    args:
      - sync: cameraframesync_0
      - cfg: VisionCapture<ProjectConstexpr::FrameLayout>::DefaultConfig()
```

`cameraframesync_0` 是 CameraFrameSync 实例的 id，必须在 `modules:` 中列在本实例之前，
并使用同一个 `template_args`。本模块不直接使用 BSP 对象，不需要额外的 `XR_REGISTER`。

`cfg` 写成 YAML map 时，`Config` 带构造函数，键必须按顺序写出某个构造函数的参数名
（`mode_in`、`output_dir_in`、`session_name_in`、`record_in`、`preview_in`、`board_in`、
`camera_calibration_in`、[`calibration_sampling_in`、[`control_in`、]]`filter_in`）；
子结构按字段名写 map，或写 `'{}'` 使用默认值。内参标定示例：

```yaml
cfg:
  mode_in: '"calibrate_camera"'
  output_dir_in: '"runs/vision_capture"'
  session_name_in: '""'
  record_in: '{}'
  preview_in:
    enabled: true
    preview_window_name: '"vision_capture_preview"'
    preview_scale: 0.5
    preview_wait_key_ms: 1
    queue_capacity: 1
    output_mode: '"web"'
    web_bind_address: '"0.0.0.0"'
    web_port: 8080
    web_stream_name: '"vision_capture"'
    max_fps: 15.0
  board_in: '{}'
  camera_calibration_in:
    enabled: true
    marker_size_mm: 25.0
    cols: 8
    rows: 6
    auto_save_views: 120
  calibration_sampling_in: '{}'
  control_in:
    stdin_enabled: true
  filter_in: '{}'
```

填好后再次运行 `xrobot setup`，生成 `User/xrobot_main.hpp`。

`xrobot module show .`（在本仓库中）或 `xrobot module show Modules/QDU-Robomaster/VisionCapture`
（在 BSP 中）打印当前的构造函数。

## 测试

在打开 `BUILD_TESTING` 的 BSP 构建中，本模块加入 `vision_capture_calibration_geometry_test`，
用 `ctest` 运行。
