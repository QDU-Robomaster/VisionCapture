# VisionCapture

同步图像与 IMU 采集模块：记录 CameraFrameSync 输出，并完成相机内参标定与手眼标定数据采样 / Synchronized image and IMU capture Module that records the CameraFrameSync output and performs camera intrinsic calibration and hand-eye calibration sampling

## 1. 模块作用 / Purpose

VisionCapture 订阅 CameraFrameSync 发布的同步图像与 IMU 数据，按运行模式保存图像和每帧元数据，并在标定模式下完成标定采样。模块用于采集与标定配置。

运行模式：

- `record`：保存同步图像和每帧元数据。
- `calibrate_camera`：判稳后保存标定样本，并用同一批图像求解相机内参。
- `calibrate_handeye`：判稳后保存手眼标定样本，仅采集手眼标定数据。
- `calibrate`：`calibrate_camera` 的别名，运行相机内参标定。

`record` 配合 `camera_calibration.enabled: true` 也进入内参标定。`calibrate_handeye` 的优先级高于该开关，只运行手眼数据采集。

同步 Topic 回调只 retain `SharedFrame` 并写入容量为 2 的 drop-oldest 队列，OpenCV、预览深拷贝和磁盘 I/O 都在模块拥有的 worker 中执行，CameraFrameSync 的发布线程不受影响。析构时先停止并 join 可取消的 stdin 控制读取器，再停止并 join 帧 worker；LibXR Topic 当前没有回调注销接口，因此上游停止发布之后才能析构 `VisionCapture`。输入图像支持 BGR8、RGB8、BGRA8、RGBA8 和 MONO8，RGB8 / RGBA8 在 worker 内转换为 OpenCV 的 BGR / BGRA 约定后再检测、预览和写图。worker 处理异常时关闭队列、释放待处理帧并锁存会话失败。

`OnMonitor()` 打印自上次调用以来的帧数、保存数、检测到标定板的帧数、PnP 成功数、接受与拒绝的样本数、队列丢弃数，以及当前模式和采样状态。

VisionCapture subscribes to the synchronized image and IMU data published by CameraFrameSync, saves the images and the per-frame metadata according to the run mode, and performs the calibration sampling in the calibration modes. The Module is used in capture and calibration configurations.

Run modes:

- `record`: saves the synchronized images and the per-frame metadata.
- `calibrate_camera`: saves calibration samples after the stability check and solves the camera intrinsics from the same images.
- `calibrate_handeye`: saves hand-eye calibration samples after the stability check and only collects the hand-eye calibration data.
- `calibrate`: alias of `calibrate_camera`, runs the camera intrinsic calibration.

`record` together with `camera_calibration.enabled: true` also enters the intrinsic calibration. `calibrate_handeye` takes priority over that switch and runs the hand-eye data collection only.

The synchronized Topic callback only retains the `SharedFrame` and writes it into a drop-oldest queue of capacity 2. OpenCV, the preview deep copy and the disk I/O all run in a worker owned by the Module, so the publishing thread of CameraFrameSync is unaffected. The destructor first stops and joins the cancellable stdin control reader and then stops and joins the frame worker; LibXR Topic currently has no callback deregistration, so `VisionCapture` is destroyed after the upstream publishing has stopped. The input images support BGR8, RGB8, BGRA8, RGBA8 and MONO8; RGB8 / RGBA8 are converted in the worker to the OpenCV BGR / BGRA convention before detection, preview and image writing. When the worker raises an exception, the queue is closed, the pending frames are released and the session failure is latched.

`OnMonitor()` prints the frame count, the saved count, the number of frames with a detected board, the PnP success count, the accepted and rejected sample counts and the queue drop count since the previous call, together with the current mode and sampling status.

## 2. 记录内容 / Records

记录目录为 `<output_dir>/<session_name>`。`session_name` 为空时，模块生成 `vision_capture_<YYYYmmdd_HHMMSS>` 作为目录名。

`record` 模式按 `record.*` 配置保存。标定模式固定保存通过判稳的样本：记录、图像和元数据强制打开，帧率不受限，与 `record.enabled` 无关；被判稳拒绝的帧不写入 `frames/` 和 `samples.csv`。标定模式下 `calibration_sampling.enabled` 为 `false` 时，帧以 `sampling_disabled` 拒绝。

输出内容：

- `samples.csv`：同步图像时间戳、IMU 时间戳、可选原始 IMU、图像文件名、标定板检测结果和采样判定。
- `frame_geometry.csv`：每个成功记录帧的完整 ROI、下采样、翻转、保留字段和采样相位，可区分混合 WIDE / NARROW 记录。
- `frames/`：保存的原始图像，格式由 `record.image_format` 决定。
- `frame_layout.txt`：编译期固定的帧存储布局。
- `camera_calibration.txt`：原生传感器坐标系下的固定标定。
- `frame_geometry.txt`：首个有效帧的 ROI、下采样、翻转和采样相位。
- `camera_info.txt`：由 `frame_layout.txt`、`camera_calibration.txt` 和 `frame_geometry.txt` 派生的帧坐标快照。
- 预览：可选的窗口或 Web 预览（见 VisionPreview），显示被拒绝的帧、检测与重投影点、判定原因、视觉覆盖、采样与求解进度以及当前帧的 geometry / profile。

记录细节：

- 手眼模式写入原始 IMU，与 `save_raw_imu` 无关；内参模式和 `record` 模式按 `save_raw_imu` 决定。
- 图像写盘失败的帧不写入两个 CSV，也不计入已保存帧数。
- `save_raw_imu: false` 时 `samples.csv` 保持固定列结构，十个原始 IMU 数值单元格留空。
- `flush_every_n: 0` 表示只在文件流关闭时刷盘，否则每成功记录指定行数后同时刷新两个 CSV。
- 相机和 MCU 时间戳属于不同时间域，`samples.csv` 分别保存两者，`dt_us` 列为空；原始加速度遵循 `CameraBase::ImuStamped` 的 `m/s^2` 约定，单位记录在末列。
- 任一必需记录或 CSV 刷盘失败时，会话失败被锁存，内参求解视角被清空，`solve` 不再返回 PASS；图像和元数据完整持久化后，该帧才提交给内参求解器。

The record directory is `<output_dir>/<session_name>`. When `session_name` is empty the Module generates `vision_capture_<YYYYmmdd_HHMMSS>` as the directory name.

The `record` mode saves according to the `record.*` configuration. The calibration modes always save the samples that pass the stability check: recording, images and metadata are forced on and the frame rate is unlimited, independent of `record.enabled`; frames rejected by the stability check are not written to `frames/` and `samples.csv`. In the calibration modes, frames are rejected with `sampling_disabled` when `calibration_sampling.enabled` is `false`.

Outputs:

- `samples.csv`: synchronized image timestamp, IMU timestamp, optional raw IMU, image file name, board detection result and sampling decision.
- `frame_geometry.csv`: the complete ROI, downsampling, flip, reserved fields and sampling phase of every successfully recorded frame, which distinguishes mixed WIDE / NARROW records.
- `frames/`: the saved raw images, in the format given by `record.image_format`.
- `frame_layout.txt`: the frame storage layout fixed at compile time.
- `camera_calibration.txt`: the fixed calibration in the native sensor coordinate system.
- `frame_geometry.txt`: ROI, downsampling, flip and sampling phase of the first valid frame.
- `camera_info.txt`: frame-coordinate snapshot derived from `frame_layout.txt`, `camera_calibration.txt` and `frame_geometry.txt`.
- Preview: optional window or Web preview (see VisionPreview), showing rejected frames, detected and reprojected points, the decision reason, visual coverage, sampling and solving progress and the geometry / profile of the current frame.

Record details:

- The hand-eye mode writes the raw IMU regardless of `save_raw_imu`; the intrinsic mode and the `record` mode follow `save_raw_imu`.
- Frames whose image write fails are not written to the two CSV files and are not counted as saved frames.
- With `save_raw_imu: false`, `samples.csv` keeps its fixed column layout and the ten raw IMU value cells are left empty.
- `flush_every_n: 0` flushes only when the file stream is closed; otherwise both CSV files are flushed after every given number of successfully recorded rows.
- Camera and MCU timestamps belong to different time domains; `samples.csv` stores both and the `dt_us` column is empty. The raw acceleration follows the `m/s^2` convention of `CameraBase::ImuStamped`, and the unit is recorded in the last column.
- When any required record or CSV flush fails, the session failure is latched, the intrinsic solver views are cleared and `solve` no longer returns PASS; a frame is submitted to the intrinsic solver after its image and metadata are completely persisted.

## 3. 相机内参标定 / Camera Intrinsic Calibration

两个标定模式使用 GShang 25 mm、8x6 标定板和 ArUco original 字典；`board.*` 用于 `record` 模式的检测。内参标定器接收通过纯视觉门限的图像，仅使用图像上的视觉观测。

配置项：

- `camera_calibration.marker_size_mm`、`cols`、`rows`：标定模式把它们规范为 `25.0` / `8` / `6`，取值不同时打印警告。
- `camera_calibration.auto_save_views`：接受的视角数达到该值后自动求解并保存结果；`0` 表示由 `solve` 命令触发。

输出目录（相对进程工作目录）：

```text
runs/camera_calib/<timestamp>_<session>_<marker>mm_<cols>x<rows>/
```

输出内容：

- `calibration.yml`
- `views.csv`
- `quality_report.txt`
- `camera_info_snippet.txt`
- `debug/`：调试图像

`camera_info_snippet.txt` 是可粘贴到 BSP `User/xrobot.yaml` 的 `constexprs` 片段，包含 `MainFrameLayout`（`CameraTypes::FrameLayout`）与原生 `MainCameraCalibration`（`CameraTypes::CameraCalibration`），值以 YAML map 写出。原生尺寸、焦距、主点、畸变系数与 `quality_report.txt` 中的重投影 RMS 用于判断结果是否可用。离群阈值、`rms`、`views.csv` 和质量报告中的重投影误差使用当前帧像素，`calibration.yml` 另存 `native_rms` 供原生坐标诊断。

`quality_ok` 要求视角数、中心与尺度覆盖、最终外参恢复出的双轴标定板倾斜跨度、内参合理性以及全局与逐视角重投影误差全部通过；在此基础上所有输出逐字节写后读回成功，求解才返回成功并生成 `calibration.yml` 与 `camera_info_snippet.txt`。质量未通过时只保留 `views.csv` 和 `quality_report.txt`。自动保存与 stdin `solve` 共用一次求解与写盘 claim，并发的重复请求等待并复用同一结果。

Both calibration modes use the GShang 25 mm, 8x6 board and the ArUco original dictionary; `board.*` is used for the detection in the `record` mode. The intrinsic calibrator receives the images that pass the pure vision gates and uses only the visual observations in the images.

Configuration:

- `camera_calibration.marker_size_mm`, `cols` and `rows`: the calibration modes normalize them to `25.0` / `8` / `6` and print a warning when the values differ.
- `camera_calibration.auto_save_views`: once the number of accepted views reaches this value, the result is solved and saved automatically; `0` means the `solve` command triggers it.

Output directory (relative to the process working directory) as in the code block above.

Outputs:

- `calibration.yml`
- `views.csv`
- `quality_report.txt`
- `camera_info_snippet.txt`
- `debug/`: debug images

`camera_info_snippet.txt` is a `constexprs` snippet that can be pasted into the BSP `User/xrobot.yaml`. It contains `MainFrameLayout` (`CameraTypes::FrameLayout`) and the native `MainCameraCalibration` (`CameraTypes::CameraCalibration`), with the values written as YAML maps. The native size, focal lengths, principal point, distortion coefficients and the reprojection RMS in `quality_report.txt` serve to judge whether the result is usable. The outlier thresholds, `rms`, `views.csv` and the reprojection errors in the quality report use current-frame pixels, and `calibration.yml` additionally stores `native_rms` for diagnostics in native coordinates.

`quality_ok` requires the view count, the center and scale coverage, the dual-axis board tilt span recovered from the final extrinsics, the intrinsic plausibility and the global and per-view reprojection errors all to pass; in addition every output has to be written and read back byte for byte, and only then does the solve return success and generate `calibration.yml` and `camera_info_snippet.txt`. When the quality check fails, only `views.csv` and `quality_report.txt` are kept. The automatic save and the stdin `solve` share one solve and write claim, and concurrent duplicate requests wait for and reuse the same result.

## 4. 采样判稳 / Sampling Stability

内参模式检查 GShang marker 数、单应性 RMS、清晰度、相机时间戳间隔，以及中心、尺度和角度的视觉重复。

手眼模式使用冻结的原生 K / D 执行畸变感知 PnP，并检查：

- PnP 重投影 RMS。
- PnP 平移和旋转抖动。
- IMU 四元数抖动。
- 陀螺仪模长。
- 加速度模长（与 9.80665 m/s^2 比较）、模长抖动和方向抖动。
- 样本之间的时间间隔、位移和角度变化。

手眼输入需要非零 MCU 时间戳、有限且可归一化的四元数、有限角速度和有限的 `m/s^2` 加速度。约为 `1.0` 的 `g` 量纲输入以 `acc_unit_or_scale` 拒绝。采样结果决定 `frames/` 和 `samples.csv` 保存哪些样本；手眼模式采集手眼数据。

The intrinsic mode checks the number of GShang markers, the homography RMS, the sharpness, the camera timestamp interval and the visual repetition of center, scale and angle.

The hand-eye mode runs a distortion-aware PnP with the frozen native K / D and checks:

- the PnP reprojection RMS;
- the PnP translation and rotation jitter;
- the IMU quaternion jitter;
- the gyroscope norm;
- the acceleration norm (compared with 9.80665 m/s^2), its jitter and its direction jitter;
- the time interval, displacement and angle change between samples.

Hand-eye input requires a non-zero MCU timestamp, a finite and normalizable quaternion, a finite angular velocity and a finite acceleration in `m/s^2`. Input in `g` units, with a norm of about `1.0`, is rejected with `acc_unit_or_scale`. The sampling result decides which samples are saved to `frames/` and `samples.csv`; the hand-eye mode collects the hand-eye data.

## 5. 本地命令 / Local Commands

`control.stdin_enabled: true` 时，标准输入接受以下命令：

- `start`：开始采样（`calibration_sampling.auto_start: true` 时启动即开始）。
- `pause` / `stop`：暂停采样。
- `reset`：清空判稳窗口、已接受样本、求解器视角和完成状态。
- `snapshot`：请求强制接受下一帧通过质量门限的样本（跳过间隔和重复检查）；被拒绝的帧不消费请求，并发的新请求不会丢失。
- `status`：打印当前采样状态。
- `solve`：内参模式立即尝试求解；手眼模式打印当前样本数。
- `help`：打印命令列表。

With `control.stdin_enabled: true` the standard input accepts the following commands:

- `start`: starts sampling (sampling starts at launch when `calibration_sampling.auto_start: true`).
- `pause` / `stop`: pauses sampling.
- `reset`: clears the stability window, the accepted samples, the solver views and the completion state.
- `snapshot`: requests that the next frame passing the quality gates is accepted by force (the interval and duplicate checks are skipped); rejected frames do not consume the request, and concurrent new requests are not lost.
- `status`: prints the current sampling status.
- `solve`: the intrinsic mode tries to solve immediately; the hand-eye mode prints the current sample count.
- `help`: prints the command list.

## 6. 构造接口 / Constructor

```cpp
template <CameraTypes::FrameLayout FrameLayoutV>
class VisionCapture;

VisionCapture(Sync& sync, Config cfg = DefaultConfig());
```

模板参数：

- `FrameLayoutV`：帧布局，与上游 CameraFrameSync 和相机的帧布局相同。

依赖：

- `sync`：`CameraFrameSync<FrameLayoutV>&`，上游的 CameraFrameSync 实例；模块订阅其 `SyncedFrameTopicName()` 并复制其原生标定。

配置参数 `cfg`（`Config`，`DefaultConfig()` 为全部默认值）：

- `mode`：`"record"`（默认）、`"calibrate_camera"`、`"calibrate_handeye"` 或 `"calibrate"`。
- `output_dir`：输出根目录，默认 `"runs/vision_capture"`。
- `session_name`：会话名，默认为空（自动生成）。
- `record`（`RecordParams`）：`enabled = true`、`image_format = "bmp"`、`max_fps = 30.0`（0 为不限）、`max_frames = 0`（0 为不限）、`save_images = true`、`save_metadata = true`、`save_raw_imu = true`、`flush_every_n = 1`。
- `preview`（`VisionPreview::RuntimeParam`）：默认关闭，字段见 VisionPreview。
- `board`（`BoardParams`）：`record` 模式的检测参数，`type = "aruco"`、`dictionary = "DICT_5X5_100"`、`marker_length_m = 0.04`。
- `camera_calibration`（`CameraCalibrationParams`）：`enabled = false`、`marker_size_mm = 25.0`、`cols = 8`、`rows = 6`、`auto_save_views = 120`。
- `calibration_sampling`（`CalibrationSamplingParams`）：`enabled = true`、`auto_start = true`、`window_size = 8`、`min_accept_interval_us = 500000`，以及手眼门限 `max_pnp_reprojection_rms_px = 2.0`、`max_pnp_translation_jitter_m = 0.005`、`max_pnp_rotation_jitter_deg = 1.0`、`max_imu_rotation_jitter_deg = 0.8`、`max_gyro_norm_dps = 2.0`、`max_acc_norm_error_mps2 = 1.5`、`max_acc_norm_jitter_mps2 = 0.5`、`max_acc_direction_jitter_deg = 2.0`、`min_sample_translation_delta_m = 0.03`、`min_sample_rotation_delta_deg = 5.0`。
- `control`（`ControlParams`）：`stdin_enabled = false`。
- `filter`（`FilterParams`）：`require_synced_imu = true`、`max_image_imu_dt_us = 2000`。

Template parameter:

- `FrameLayoutV`: the frame layout, equal to the frame layout of the upstream CameraFrameSync and the camera.

Dependencies:

- `sync`: `CameraFrameSync<FrameLayoutV>&`, the upstream CameraFrameSync instance; the Module subscribes to its `SyncedFrameTopicName()` and copies its native calibration.

Configuration parameters `cfg` (`Config`, `DefaultConfig()` holds all defaults):

- `mode`: `"record"` (default), `"calibrate_camera"`, `"calibrate_handeye"` or `"calibrate"`.
- `output_dir`: output root directory, default `"runs/vision_capture"`.
- `session_name`: session name, default empty (generated automatically).
- `record` (`RecordParams`): `enabled = true`, `image_format = "bmp"`, `max_fps = 30.0` (0 is unlimited), `max_frames = 0` (0 is unlimited), `save_images = true`, `save_metadata = true`, `save_raw_imu = true`, `flush_every_n = 1`.
- `preview` (`VisionPreview::RuntimeParam`): disabled by default, fields see VisionPreview.
- `board` (`BoardParams`): detection parameters of the `record` mode, `type = "aruco"`, `dictionary = "DICT_5X5_100"`, `marker_length_m = 0.04`.
- `camera_calibration` (`CameraCalibrationParams`): `enabled = false`, `marker_size_mm = 25.0`, `cols = 8`, `rows = 6`, `auto_save_views = 120`.
- `calibration_sampling` (`CalibrationSamplingParams`): `enabled = true`, `auto_start = true`, `window_size = 8`, `min_accept_interval_us = 500000`, and the hand-eye gates `max_pnp_reprojection_rms_px = 2.0`, `max_pnp_translation_jitter_m = 0.005`, `max_pnp_rotation_jitter_deg = 1.0`, `max_imu_rotation_jitter_deg = 0.8`, `max_gyro_norm_dps = 2.0`, `max_acc_norm_error_mps2 = 1.5`, `max_acc_norm_jitter_mps2 = 0.5`, `max_acc_direction_jitter_deg = 2.0`, `min_sample_translation_delta_m = 0.03`, `min_sample_rotation_delta_deg = 5.0`.
- `control` (`ControlParams`): `stdin_enabled = false`.
- `filter` (`FilterParams`): `require_synced_imu = true`, `max_image_imu_dt_us = 2000`.

## 7. Topic

| Topic | 方向 | 类型 | 说明 |
| --- | --- | --- | --- |
| `sync.SyncedFrameTopicName()` | 订阅 | `SyncedFrameTopicPayload` | CameraFrameSync 发布的同步图像与 IMU 数据 |

| Topic | Direction | Type | Meaning |
| --- | --- | --- | --- |
| `sync.SyncedFrameTopicName()` | Subscribe | `SyncedFrameTopicPayload` | Synchronized image and IMU data published by CameraFrameSync |

## 8. 配置示例 / Configuration Example

`xrobot instance add QDU-Robomaster/VisionCapture` 写入含空 `template_args` 的实例；`template_args` 填写为 `constexprs` 中定义的帧布局，`sync` 填写为 CameraFrameSync 实例的 id。`Config` 带有多个构造函数，`cfg` 写成 YAML map 时，键为其中一个构造函数的参数名（`mode_in`、`output_dir_in`、`session_name_in`、`record_in`、`preview_in`、`board_in`、`camera_calibration_in`、`calibration_sampling_in`、`control_in`、`filter_in`，其中 `calibration_sampling_in` 与 `control_in` 可成组省略），子结构按字段名写 map：

`xrobot instance add QDU-Robomaster/VisionCapture` writes an instance with an empty `template_args`. `template_args` is set to a frame layout defined in `constexprs`, and `sync` to the id of a CameraFrameSync instance. `Config` has several constructors; when `cfg` is written as a YAML map, the keys are the parameter names of one of them (`mode_in`, `output_dir_in`, `session_name_in`, `record_in`, `preview_in`, `board_in`, `camera_calibration_in`, `calibration_sampling_in`, `control_in`, `filter_in`, where `calibration_sampling_in` and `control_in` can be omitted as a group), and sub-structures are written as maps by field name:

```yaml
constexpr_namespace: AutoAimRunConfig
constexpr_includes:
  - CameraBase.hpp
constexprs:
  HikFrameLayout:
    type: CameraTypes::FrameLayout
    value: '{.width = 720, .height = 540, .step = 2160, .encoding = CameraTypes::Encoding::BGR8}'
modules:
  - module: QDU-Robomaster/VisionCapture
    id: vision_capture
    template_args:
      - AutoAimRunConfig::HikFrameLayout
    args:
      - sync: camera_frame_sync
      - cfg:
          mode_in: "record"
          output_dir_in: "runs/vision_capture"
          session_name_in: "hik_capture"
          record_in:
            enabled: true
            image_format: "bmp"
            max_fps: 30.0
            max_frames: 0
            save_images: true
            save_metadata: true
            save_raw_imu: true
            flush_every_n: 1
          preview_in:
            enabled: true
            preview_window_name: "vision_capture"
            preview_scale: 0.5
            preview_wait_key_ms: 1
            queue_capacity: 1
            output_mode: "web"
            web_bind_address: "0.0.0.0"
            web_port: 8080
            web_stream_name: "vision_capture"
            max_fps: 30.0
          board_in:
            type: "aruco"
            dictionary: "DICT_5X5_100"
            marker_length_m: 0.04
          camera_calibration_in:
            enabled: true
            marker_size_mm: 25.0
            cols: 8
            rows: 6
            auto_save_views: 120
          filter_in:
            require_synced_imu: true
            max_image_imu_dt_us: 2000
```

被引用的 CameraFrameSync 实例列在本实例之前，并使用相同的 `template_args`。

The referenced CameraFrameSync instance is listed before this instance and uses the same `template_args`.

## 9. 依赖与硬件 / Dependencies and Hardware

依赖：

- `QDU-Robomaster/CameraFrameSync`：同步帧输入（`SyncedFrame`）和原生标定。
- `QDU-Robomaster/VisionPreview`：预览输出。
- `QDU-Robomaster/CameraBase`：帧布局、geometry 与共享图像类型。
- OpenCV 4（`core`、`imgproc`、`imgcodecs`、`calib3d`、`aruco`）。
- LibXR。

硬件：由 CameraFrameSync 同步的相机与 IMU。

测试：在启用 `BUILD_TESTING` 的 BSP 构建中，模块加入 `vision_capture_calibration_geometry_test`，用 `ctest` 运行。

Dependencies:

- `QDU-Robomaster/CameraFrameSync`: synchronized frame input (`SyncedFrame`) and the native calibration.
- `QDU-Robomaster/VisionPreview`: preview output.
- `QDU-Robomaster/CameraBase`: frame layout, geometry and shared image types.
- OpenCV 4 (`core`, `imgproc`, `imgcodecs`, `calib3d`, `aruco`).
- LibXR.

Hardware: the camera and the IMU synchronized by CameraFrameSync.

Tests: in a BSP build with `BUILD_TESTING` enabled, the Module adds `vision_capture_calibration_geometry_test`, run with `ctest`.
