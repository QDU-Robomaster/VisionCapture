#pragma once

// clang-format off
/* === MODULE MANIFEST V2 ===
module_description: 同步图像/IMU 采集与标定数据记录模块
constructor_args:
  cfg:
    mode: "record"
    output_dir: "runs/vision_capture"
    session_name: ""
    record:
      enabled: true
      image_format: "bmp"
      max_fps: 30.0
      max_frames: 0
      save_images: true
      save_metadata: true
      save_raw_imu: true
      flush_every_n: 1
    preview:
      enabled: false
      preview_window_name: "vision_capture"
      preview_scale: 0.5
      preview_wait_key_ms: 1
      queue_capacity: 1
      output_mode: "window"
      web_bind_address: "0.0.0.0"
      web_port: 8080
      web_stream_name: "vision_capture"
      max_fps: 30.0
    board:
      type: "aruco"
      dictionary: "DICT_5X5_100"
      marker_length_m: 0.04
    camera_calibration:
      enabled: false
      marker_size_mm: 25.0
      cols: 8
      rows: 6
      auto_save_views: 120
    calibration_sampling:
      enabled: true
      auto_start: true
      window_size: 8
      min_accept_interval_us: 500000
      max_pnp_reprojection_rms_px: 2.0
      max_pnp_translation_jitter_m: 0.005
      max_pnp_rotation_jitter_deg: 1.0
      max_imu_rotation_jitter_deg: 0.8
      max_gyro_norm_dps: 2.0
      max_acc_norm_error_mps2: 1.5
      max_acc_norm_jitter_mps2: 0.5
      max_acc_direction_jitter_deg: 2.0
      min_sample_translation_delta_m: 0.03
      min_sample_rotation_delta_deg: 5.0
    handeye_calibration:
      min_samples: 6
      R_body2imu: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    control:
      stdin_enabled: false
    filter:
      require_synced_imu: true
      max_image_imu_dt_us: 2000
  sync: '@camera_frame_sync'
template_args:
  - Info:
      width: 1280
      height: 720
      step: 3840
      encoding: CameraTypes::Encoding::BGR8
      camera_matrix: [800.0, 0.0, 640.0, 0.0, 800.0, 360.0, 0.0, 0.0, 1.0]
      distortion_model: CameraTypes::DistortionModel::PLUMB_BOB
      distortion_coefficients: [0.0, 0.0, 0.0, 0.0, 0.0]
      rectification_matrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
      projection_matrix: [800.0, 0.0, 640.0, 0.0, 0.0, 800.0, 360.0, 0.0, 0.0, 0.0, 1.0, 0.0]
required_hardware: []
depends:
  - qdu-future/CameraFrameSync
  - qdu-future/VisionPreview
=== END MANIFEST === */
// clang-format on

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "CameraFrameSync.hpp"
#include "VisionPreview.hpp"
#include "VisionCaptureCalibrationBoard.hpp"
#include "VisionCaptureCameraCalibration.hpp"
#include "app_framework.hpp"
#include "libxr.hpp"
#include "logger.hpp"

namespace VisionCaptureDetail
{
/// 同步帧消费线程栈大小。
inline constexpr size_t kWorkerStackSize = 8192;
/// 标准输入命令线程栈大小。
inline constexpr size_t kControlStackSize = 4096;
/// 等待 CameraFrameSync 同步帧的超时时间，单位 ms。
inline constexpr uint32_t kSyncWaitTimeoutMs = 1000;

/**
 * @brief 将 string_view 复制为拥有存储的 std::string。
 */
inline std::string ToString(std::string_view value)
{
  return std::string(value.data(), value.size());
}

/**
 * @brief 生成默认采集会话名。
 */
inline std::string MakeTimestampSessionName()
{
  const auto now = std::chrono::system_clock::now();
  const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
#if defined(_WIN32)
  localtime_s(&tm, &now_time);
#else
  localtime_r(&now_time, &tm);
#endif
  std::ostringstream out;
  out << "vision_capture_" << std::put_time(&tm, "%Y%m%d_%H%M%S");
  return out.str();
}

/**
 * @brief 将配置中的 ArUco 字典名转换为 OpenCV 枚举值。
 */
inline int ArucoDictionaryId(std::string_view name)
{
  if (name == "DICT_4X4_50") return cv::aruco::DICT_4X4_50;
  if (name == "DICT_4X4_100") return cv::aruco::DICT_4X4_100;
  if (name == "DICT_5X5_50") return cv::aruco::DICT_5X5_50;
  if (name == "DICT_5X5_100") return cv::aruco::DICT_5X5_100;
  if (name == "DICT_6X6_50") return cv::aruco::DICT_6X6_50;
  if (name == "DICT_6X6_100") return cv::aruco::DICT_6X6_100;
  if (name == "DICT_ARUCO_ORIGINAL") return cv::aruco::DICT_ARUCO_ORIGINAL;
  return cv::aruco::DICT_5X5_100;
}

/**
 * @brief 将 CameraFrameSync 图像帧封装为 OpenCV Mat。
 *
 * 返回的 Mat 不拥有像素内存，只在传入图像帧有效期间使用。
 */
template <CameraTypes::CameraInfo CameraInfoV>
cv::Mat MakeImageView(
    const typename CameraFrameSync<CameraInfoV>::ImageFrame& image)
{
  const int width = static_cast<int>(CameraInfoV.width);
  const int height = static_cast<int>(CameraInfoV.height);
  const size_t step = static_cast<size_t>(CameraInfoV.step);
  auto* data = const_cast<uint8_t*>(image.data.data());
  if constexpr (CameraInfoV.encoding == CameraTypes::Encoding::BGR8 ||
                CameraInfoV.encoding == CameraTypes::Encoding::RGB8)
  {
    return cv::Mat(height, width, CV_8UC3, data, step);
  }
  else if constexpr (CameraInfoV.encoding == CameraTypes::Encoding::BGRA8 ||
                     CameraInfoV.encoding == CameraTypes::Encoding::RGBA8)
  {
    return cv::Mat(height, width, CV_8UC4, data, step);
  }
  else if constexpr (CameraInfoV.encoding == CameraTypes::Encoding::MONO8)
  {
    return cv::Mat(height, width, CV_8UC1, data, step);
  }
  else
  {
    return {};
  }
}

/**
 * @brief 将输入图像转换为 BGR 预览图。
 */
inline cv::Mat MakeBgrForPreview(const cv::Mat& image)
{
  if (image.empty())
  {
    return {};
  }
  cv::Mat bgr;
  if (image.channels() == 1)
  {
    cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
  }
  else if (image.channels() == 4)
  {
    cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
  }
  else
  {
    bgr = image.clone();
  }
  return bgr;
}

/**
 * @brief 弧度转角度。
 */
inline double RadToDeg(double rad) { return rad * 180.0 / CV_PI; }

/**
 * @brief 计算三维数组的二范数。
 */
inline double Norm3(const std::array<float, 3>& value)
{
  const double x = value[0];
  const double y = value[1];
  const double z = value[2];
  return std::sqrt(x * x + y * y + z * z);
}

/**
 * @brief 将三维数组转换为 OpenCV Vec3d。
 */
inline cv::Vec3d ToVec3d(const std::array<float, 3>& value)
{
  return {static_cast<double>(value[0]), static_cast<double>(value[1]),
          static_cast<double>(value[2])};
}

/**
 * @brief 归一化 wxyz 顺序四元数。
 */
inline cv::Vec4d NormalizeQuatWxyz(const std::array<float, 4>& q)
{
  const double w = q[0];
  const double x = q[1];
  const double y = q[2];
  const double z = q[3];
  const double norm = std::sqrt(w * w + x * x + y * y + z * z);
  if (norm <= 1e-9)
  {
    return {1.0, 0.0, 0.0, 0.0};
  }
  return {w / norm, x / norm, y / norm, z / norm};
}

/**
 * @brief 将 wxyz 顺序四元数转换为旋转矩阵。
 */
inline cv::Mat QuatWxyzToRotationMatrix(const cv::Vec4d& q)
{
  const double w = q[0];
  const double x = q[1];
  const double y = q[2];
  const double z = q[3];
  return (cv::Mat_<double>(3, 3)
          << 1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
          2.0 * (x * z + y * w), 2.0 * (x * y + z * w),
          1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
          2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
          1.0 - 2.0 * (x * x + y * y));
}

/**
 * @brief 将 OpenCV 相机系 C 转到相机安装系 M。
 *
 * C: x 向右、y 向下、z 向前。M: x 向右、y 向前、z 向上。
 */
inline cv::Mat CameraToMountRotation()
{
  return (cv::Mat_<double>(3, 3) << 1.0, 0.0, 0.0,
          0.0, 0.0, 1.0,
          0.0, -1.0, 0.0);
}

/**
 * @brief 将旋转矩阵转换为 wxyz 四元数。
 */
inline cv::Vec4d RotationMatrixToQuatWxyz(const cv::Mat& rotation)
{
  const double m00 = rotation.at<double>(0, 0);
  const double m11 = rotation.at<double>(1, 1);
  const double m22 = rotation.at<double>(2, 2);
  const double trace = m00 + m11 + m22;
  double w = 1.0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  if (trace > 0.0)
  {
    const double s = std::sqrt(trace + 1.0) * 2.0;
    w = 0.25 * s;
    x = (rotation.at<double>(2, 1) - rotation.at<double>(1, 2)) / s;
    y = (rotation.at<double>(0, 2) - rotation.at<double>(2, 0)) / s;
    z = (rotation.at<double>(1, 0) - rotation.at<double>(0, 1)) / s;
  }
  else if (m00 > m11 && m00 > m22)
  {
    const double s = std::sqrt(1.0 + m00 - m11 - m22) * 2.0;
    w = (rotation.at<double>(2, 1) - rotation.at<double>(1, 2)) / s;
    x = 0.25 * s;
    y = (rotation.at<double>(0, 1) + rotation.at<double>(1, 0)) / s;
    z = (rotation.at<double>(0, 2) + rotation.at<double>(2, 0)) / s;
  }
  else if (m11 > m22)
  {
    const double s = std::sqrt(1.0 + m11 - m00 - m22) * 2.0;
    w = (rotation.at<double>(0, 2) - rotation.at<double>(2, 0)) / s;
    x = (rotation.at<double>(0, 1) + rotation.at<double>(1, 0)) / s;
    y = 0.25 * s;
    z = (rotation.at<double>(1, 2) + rotation.at<double>(2, 1)) / s;
  }
  else
  {
    const double s = std::sqrt(1.0 + m22 - m00 - m11) * 2.0;
    w = (rotation.at<double>(1, 0) - rotation.at<double>(0, 1)) / s;
    x = (rotation.at<double>(0, 2) + rotation.at<double>(2, 0)) / s;
    y = (rotation.at<double>(1, 2) + rotation.at<double>(2, 1)) / s;
    z = 0.25 * s;
  }

  const double norm = std::sqrt(w * w + x * x + y * y + z * z);
  if (norm <= 1e-9)
  {
    return {1.0, 0.0, 0.0, 0.0};
  }
  return {w / norm, x / norm, y / norm, z / norm};
}

/**
 * @brief 计算两个四元数之间的最小旋转角。
 */
inline double QuatAngularDistanceDeg(const cv::Vec4d& lhs, const cv::Vec4d& rhs)
{
  const double dot = std::fabs(lhs[0] * rhs[0] + lhs[1] * rhs[1] +
                              lhs[2] * rhs[2] + lhs[3] * rhs[3]);
  return RadToDeg(2.0 * std::acos(VisionCaptureCalibrationBoard::ClampUnit(dot)));
}

/**
 * @brief 计算两个 Rodrigues 旋转向量之间的最小旋转角。
 */
inline double RotationDistanceDeg(const cv::Mat& lhs_rvec,
                                  const cv::Mat& rhs_rvec)
{
  cv::Mat lhs;
  cv::Mat rhs;
  cv::Rodrigues(lhs_rvec, lhs);
  cv::Rodrigues(rhs_rvec, rhs);
  const cv::Mat delta = lhs * rhs.t();
  const double trace = delta.at<double>(0, 0) + delta.at<double>(1, 1) +
                       delta.at<double>(2, 2);
  return RadToDeg(std::acos(VisionCaptureCalibrationBoard::ClampUnit((trace - 1.0) * 0.5)));
}
}  // namespace VisionCaptureDetail

/**
 * @brief 同步图像和 IMU 采集模块。
 *
 * 模块从 CameraFrameSync 读取同步帧，可保存图像/IMU 元数据、显示预览、
 * 执行相机内参标定采样，并筛选后续手眼标定所需的稳定样本。
 */
template <CameraTypes::CameraInfo CameraInfoV>
class VisionCapture : public LibXR::Application
{
 public:
  /// 对应的 CameraFrameSync 类型。
  using Sync = CameraFrameSync<CameraInfoV>;
  /// 同步后的图像帧类型。
  using ImageFrame = typename Sync::ImageFrame;
  /// 同步后的 IMU 数据类型。
  using ImuStamped = typename Sync::ImuStamped;
  /// 图像和 IMU 合包类型。
  using SyncedFrame = typename Sync::SyncedFrame;

  /**
   * @brief 同步帧记录配置。
   */
  struct RecordParams
  {
    /// 是否保存同步帧。
    bool enabled = true;
    /// 图像文件扩展名，常用 `bmp` 或 `png`。
    std::string_view image_format = "bmp";
    /// 记录帧率上限，0 表示不限制。
    double max_fps = 30.0;
    /// 最多保存多少帧，0 表示不限制。
    uint32_t max_frames = 0;
    /// 是否保存图像文件。
    bool save_images = true;
    /// 是否保存 samples.csv。
    bool save_metadata = true;
    /// 是否在 samples.csv 中写入同步 IMU 数据。
    bool save_raw_imu = true;
    /// CSV 每写入多少行刷盘一次。
    uint32_t flush_every_n = 1;
  };

  /**
   * @brief 标定板检测配置。
   */
  struct BoardParams
  {
    /// 当前支持 `aruco`。
    std::string_view type = "aruco";
    /// OpenCV ArUco 字典名称。
    std::string_view dictionary = "DICT_5X5_100";
    /// 单个 marker 边长，单位 m。
    double marker_length_m = 0.04;
  };

  /**
   * @brief 同步帧过滤配置。
   */
  struct FilterParams
  {
    /// true 表示没有同步 IMU 的图像不保存。
    bool require_synced_imu = true;
    /// 图像与 IMU 时间戳允许的最大差值，单位 us。
    uint32_t max_image_imu_dt_us = 2000;
  };

  /**
   * @brief 相机内参标定配置。
   */
  struct CameraCalibrationParams
  {
    /// 是否启用相机内参标定。
    bool enabled = false;
    /// marker 黑码区域边长，单位 mm。
    double marker_size_mm = 25.0;
    /// GShang 标定板棋盘列数。
    int cols = 8;
    /// GShang 标定板棋盘行数。
    int rows = 6;
    /// 自动求解前需要接受的视角数量。
    uint32_t auto_save_views = 120;
  };

  /**
   * @brief 标定采样判稳配置。
   */
  struct CalibrationSamplingParams
  {
    /// 是否启用判稳采样。
    bool enabled = true;
    /// true 表示启动后立即开始采样。
    bool auto_start = true;
    /// 计算稳定性时保留的最近样本数。
    uint32_t window_size = 8;
    /// 两个接受样本之间的最小时间间隔，单位 us。
    uint64_t min_accept_interval_us = 500000;
    /// PnP 重投影 RMS 上限，单位像素。
    double max_pnp_reprojection_rms_px = 2.0;
    /// PnP 平移抖动上限，单位 m。
    double max_pnp_translation_jitter_m = 0.005;
    /// PnP 旋转抖动上限，单位 deg。
    double max_pnp_rotation_jitter_deg = 1.0;
    /// IMU 四元数抖动上限，单位 deg。
    double max_imu_rotation_jitter_deg = 0.8;
    /// 陀螺仪模长上限，单位 deg/s。
    double max_gyro_norm_dps = 2.0;
    /// 加速度模长与重力加速度差值上限，单位 m/s^2。
    double max_acc_norm_error_mps2 = 1.5;
    /// 加速度模长抖动上限，单位 m/s^2。
    double max_acc_norm_jitter_mps2 = 0.5;
    /// 加速度方向抖动上限，单位 deg。
    double max_acc_direction_jitter_deg = 2.0;
    /// 与已接受样本相比需要达到的最小平移变化，单位 m。
    double min_sample_translation_delta_m = 0.03;
    /// 与已接受样本相比需要达到的最小姿态变化，单位 deg。
    double min_sample_rotation_delta_deg = 5.0;
  };

  /**
   * @brief 手眼标定求解配置。
   */
  struct HandEyeCalibrationParams
  {
    /// 触发求解需要的最少稳定样本数。
    uint32_t min_samples = 6;
    /// 公开本体系 B 到 IMU 四元数本体系的旋转，行优先 3x3。
    std::array<double, 9> R_body2imu{
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0};
  };

  /**
   * @brief 本地命令配置。
   */
  struct ControlParams
  {
    /// true 时从标准输入读取 start/pause/reset/status 等命令。
    bool stdin_enabled = false;
  };

  /**
   * @brief VisionCapture 总配置。
   */
  struct Config
  {
    /**
     * @brief 默认配置。
     */
    Config() = default;

    /**
     * @brief 供只填写基础配置项的生成代码使用。
     */
    Config(std::string_view mode_in, std::string_view output_dir_in,
           std::string_view session_name_in, RecordParams record_in,
           VisionPreview::RuntimeParam preview_in, BoardParams board_in,
           FilterParams filter_in)
        : mode(mode_in),
          output_dir(output_dir_in),
          session_name(session_name_in),
          record(record_in),
          preview(preview_in),
          board(board_in),
          filter(filter_in)
    {
    }

    /**
     * @brief 供填写相机标定配置项的生成代码使用。
     */
    Config(std::string_view mode_in, std::string_view output_dir_in,
           std::string_view session_name_in, RecordParams record_in,
           VisionPreview::RuntimeParam preview_in, BoardParams board_in,
           CameraCalibrationParams camera_calibration_in, FilterParams filter_in)
        : mode(mode_in),
          output_dir(output_dir_in),
          session_name(session_name_in),
          record(record_in),
          preview(preview_in),
          board(board_in),
          camera_calibration(camera_calibration_in),
          filter(filter_in)
    {
    }

    /**
     * @brief 供填写相机标定和手眼标定配置项的生成代码使用。
     */
    Config(std::string_view mode_in, std::string_view output_dir_in,
           std::string_view session_name_in, RecordParams record_in,
           VisionPreview::RuntimeParam preview_in, BoardParams board_in,
           CameraCalibrationParams camera_calibration_in,
           HandEyeCalibrationParams handeye_calibration_in,
           FilterParams filter_in)
        : mode(mode_in),
          output_dir(output_dir_in),
          session_name(session_name_in),
          record(record_in),
          preview(preview_in),
          board(board_in),
          camera_calibration(camera_calibration_in),
          handeye_calibration(handeye_calibration_in),
          filter(filter_in)
    {
    }

    /**
     * @brief 供填写 calibration_sampling 的生成代码使用。
     */
    Config(std::string_view mode_in, std::string_view output_dir_in,
           std::string_view session_name_in, RecordParams record_in,
           VisionPreview::RuntimeParam preview_in, BoardParams board_in,
           CameraCalibrationParams camera_calibration_in,
           CalibrationSamplingParams calibration_sampling_in,
           FilterParams filter_in)
        : mode(mode_in),
          output_dir(output_dir_in),
          session_name(session_name_in),
          record(record_in),
          preview(preview_in),
          board(board_in),
          camera_calibration(camera_calibration_in),
          calibration_sampling(calibration_sampling_in),
          filter(filter_in)
    {
    }

    /**
     * @brief 完整配置。
     */
    Config(std::string_view mode_in, std::string_view output_dir_in,
           std::string_view session_name_in, RecordParams record_in,
           VisionPreview::RuntimeParam preview_in, BoardParams board_in,
           CameraCalibrationParams camera_calibration_in,
           CalibrationSamplingParams calibration_sampling_in,
           ControlParams control_in, FilterParams filter_in)
        : mode(mode_in),
          output_dir(output_dir_in),
          session_name(session_name_in),
          record(record_in),
          preview(preview_in),
          board(board_in),
          camera_calibration(camera_calibration_in),
          calibration_sampling(calibration_sampling_in),
          control(control_in),
          filter(filter_in)
    {
    }

    /**
     * @brief 完整配置，包含手眼标定参数。
     */
    Config(std::string_view mode_in, std::string_view output_dir_in,
           std::string_view session_name_in, RecordParams record_in,
           VisionPreview::RuntimeParam preview_in, BoardParams board_in,
           CameraCalibrationParams camera_calibration_in,
           CalibrationSamplingParams calibration_sampling_in,
           HandEyeCalibrationParams handeye_calibration_in,
           FilterParams filter_in)
        : mode(mode_in),
          output_dir(output_dir_in),
          session_name(session_name_in),
          record(record_in),
          preview(preview_in),
          board(board_in),
          camera_calibration(camera_calibration_in),
          calibration_sampling(calibration_sampling_in),
          handeye_calibration(handeye_calibration_in),
          filter(filter_in)
    {
    }

    /**
     * @brief 完整配置，包含手眼标定和本地控制参数。
     */
    Config(std::string_view mode_in, std::string_view output_dir_in,
           std::string_view session_name_in, RecordParams record_in,
           VisionPreview::RuntimeParam preview_in, BoardParams board_in,
           CameraCalibrationParams camera_calibration_in,
           CalibrationSamplingParams calibration_sampling_in,
           HandEyeCalibrationParams handeye_calibration_in,
           ControlParams control_in, FilterParams filter_in)
        : mode(mode_in),
          output_dir(output_dir_in),
          session_name(session_name_in),
          record(record_in),
          preview(preview_in),
          board(board_in),
          camera_calibration(camera_calibration_in),
          calibration_sampling(calibration_sampling_in),
          handeye_calibration(handeye_calibration_in),
          control(control_in),
          filter(filter_in)
    {
    }

    /// 运行模式：`record`、`calibrate_camera`、`calibrate_handeye` 或 `calibrate`。
    std::string_view mode = "record";
    /// 输出根目录。
    std::string_view output_dir = "runs/vision_capture";
    /// 会话名称；为空时自动使用时间戳。
    std::string_view session_name = "";
    /// 同步帧记录配置。
    RecordParams record{};
    /// 预览配置。
    VisionPreview::RuntimeParam preview{};
    /// 标定板检测配置。
    BoardParams board{};
    /// 相机内参标定配置。
    CameraCalibrationParams camera_calibration{};
    /// 标定采样判稳配置。
    CalibrationSamplingParams calibration_sampling{};
    /// 手眼标定求解配置。
    HandEyeCalibrationParams handeye_calibration{};
    /// 本地命令配置。
    ControlParams control{};
    /// 同步帧过滤配置。
    FilterParams filter{};
  };

  /**
   * @brief 构造同步采集模块并启动工作线程。
   */
  VisionCapture(LibXR::HardwareContainer&, LibXR::ApplicationManager& app,
                Config cfg, Sync& sync)
      : cfg_(cfg),
        sync_(sync),
        dictionary_(cv::aruco::getPredefinedDictionary(
            VisionCaptureDetail::ArucoDictionaryId(cfg_.board.dictionary)))
  {
    NormalizeCalibrationConfig();
    detector_params_.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    preview_.Start(cfg_.preview);
    sampling_running_.store(cfg_.calibration_sampling.auto_start,
                            std::memory_order_release);
    PrepareOutput();
    StartCameraCalibrationIfNeeded();
    worker_thread_.Create(this, WorkerThreadFun, "VisionCapture",
                          VisionCaptureDetail::kWorkerStackSize,
                          LibXR::Thread::Priority::MEDIUM);
    if (cfg_.control.stdin_enabled)
    {
      control_thread_.Create(this, ControlThreadFun, "VisionCaptureCtl",
                             VisionCaptureDetail::kControlStackSize,
                             LibXR::Thread::Priority::LOW);
    }
    app.Register(*this);
  }

  /**
   * @brief 周期输出采集、检测和采样计数。
   */
  void OnMonitor() override
  {
    const uint64_t frames = frames_seen_.exchange(0);
    const uint64_t saved = frames_saved_.exchange(0);
    const uint64_t boards = boards_detected_.exchange(0);
    const uint64_t pnp_ok = sampling_pnp_ok_.exchange(0);
    const uint64_t accepted = sampling_accepted_.exchange(0);
    const uint64_t rejected = sampling_rejected_.exchange(0);
    const std::string status = BuildStatusLine();
    XR_LOG_INFO("VisionCapture monitor: frames=%llu saved=%llu boards=%llu "
                "pnp_ok=%llu accepted=%llu rejected=%llu %s",
                static_cast<unsigned long long>(frames),
                static_cast<unsigned long long>(saved),
                static_cast<unsigned long long>(boards),
                static_cast<unsigned long long>(pnp_ok),
                static_cast<unsigned long long>(accepted),
                static_cast<unsigned long long>(rejected), status.c_str());
  }

 private:
  /**
   * @brief 从 CameraFrameSync 阻塞读取同步帧并交给 ProcessFrame()。
   */
  static void WorkerThreadFun(VisionCapture* self)
  {
    XR_LOG_INFO("VisionCapture worker starting: image=%s imu=%s",
                self->sync_.ImageTopicName(), self->sync_.ImuTopicName());

    typename Sync::Subscriber subscriber(self->sync_);
    if (!subscriber.Valid())
    {
      XR_LOG_ERROR("VisionCapture failed to attach sync stream: image=%s",
                   self->sync_.ImageTopicName());
      return;
    }

    SyncedFrame frame;
    while (true)
    {
      const auto wait_ans =
          subscriber.Wait(frame, VisionCaptureDetail::kSyncWaitTimeoutMs);
      if (wait_ans == LibXR::ErrorCode::TIMEOUT)
      {
        continue;
      }
      if (wait_ans != LibXR::ErrorCode::OK)
      {
        XR_LOG_ERROR("VisionCapture sync wait failed err=%d",
                     static_cast<int>(wait_ans));
        return;
      }
      self->ProcessFrame(frame);
    }
  }

  /**
   * @brief 从标准输入读取采样控制命令。
   */
  static void ControlThreadFun(VisionCapture* self)
  {
    XR_LOG_INFO("VisionCapture stdin control ready: help/status/start/pause/reset/solve/snapshot");
    std::string line;
    while (std::getline(std::cin, line))
    {
      self->HandleControlCommand(line);
    }
    XR_LOG_INFO("VisionCapture stdin control stopped: stdin closed");
  }

  /**
   * @brief 执行一条标准输入控制命令。
   */
  void HandleControlCommand(std::string_view command)
  {
    while (!command.empty() &&
           (command.front() == ' ' || command.front() == '\t' ||
            command.front() == '\r' || command.front() == '\n'))
    {
      command.remove_prefix(1);
    }
    while (!command.empty() &&
           (command.back() == ' ' || command.back() == '\t' ||
            command.back() == '\r' || command.back() == '\n'))
    {
      command.remove_suffix(1);
    }
    if (command == "start")
    {
      sampling_running_.store(true, std::memory_order_release);
      XR_LOG_PASS("VisionCapture control: sampling started");
    }
    else if (command == "pause" || command == "stop")
    {
      sampling_running_.store(false, std::memory_order_release);
      XR_LOG_INFO("VisionCapture control: sampling paused");
    }
    else if (command == "reset")
    {
      ResetCalibrationSampling();
      XR_LOG_PASS("VisionCapture control: sampling reset");
    }
    else if (command == "status")
    {
      const std::string status = BuildStatusLine();
      XR_LOG_INFO("VisionCapture status: %s", status.c_str());
    }
    else if (command == "solve")
    {
      SolveCurrentCalibration();
    }
    else if (command == "snapshot")
    {
      force_snapshot_.store(true, std::memory_order_release);
      XR_LOG_INFO("VisionCapture control: snapshot requested");
    }
    else if (command == "help")
    {
      XR_LOG_INFO("VisionCapture commands: start pause reset solve status snapshot help");
    }
    else if (!command.empty())
    {
      const std::string text(command);
      XR_LOG_WARN("VisionCapture control: unknown command '%s'", text.c_str());
    }
  }

  /**
   * @brief 清空判稳窗口和已接受样本。
   */
  void ResetCalibrationSampling()
  {
    std::lock_guard<std::mutex> lock(sampling_mutex_);
    stability_window_.clear();
    accepted_calibration_samples_.clear();
    last_accepted_sample_ = StableSample{};
    {
      std::lock_guard<std::mutex> status_lock(status_mutex_);
      last_reject_reason_ = "reset";
      last_pnp_rms_px_ = 0.0;
      last_gyro_norm_dps_ = 0.0;
      last_acc_norm_mps2_ = 0.0;
    }
  }

  /**
   * @brief 根据当前模式触发求解或打印样本状态。
   */
  void SolveCurrentCalibration()
  {
    bool solved_camera = false;
    if (ShouldRunCameraCalibration())
    {
      if (camera_calibration_.SaveAndStop())
      {
        XR_LOG_PASS("VisionCapture control: camera calibration solved");
        solved_camera = true;
      }
      else
      {
        XR_LOG_WARN("VisionCapture control: camera calibration solve failed");
      }
    }

    std::size_t samples = 0;
    {
      std::lock_guard<std::mutex> lock(sampling_mutex_);
      samples = accepted_calibration_samples_.size();
    }
    if (IsCalibrationDatasetMode())
    {
      if (ShouldRunHandEyeCalibration())
      {
        SolveHandEyeCalibration();
      }
      else
      {
        XR_LOG_INFO("VisionCapture control: handeye solver skipped in mode=%s accepted_samples=%u",
                    VisionCaptureDetail::ToString(cfg_.mode).c_str(),
                    static_cast<unsigned>(samples));
      }
    }
    else if (!solved_camera)
    {
      XR_LOG_WARN("VisionCapture control: no calibration solver active");
    }
  }

  /**
   * @brief 用已接受样本求解相机到公开本体系 B 的手眼外参。
   */
  bool SolveHandEyeCalibration()
  {
    std::vector<StableSample> samples;
    {
      std::lock_guard<std::mutex> lock(sampling_mutex_);
      samples = accepted_calibration_samples_;
    }

    const uint32_t min_samples =
        std::max<uint32_t>(3, cfg_.handeye_calibration.min_samples);
    if (samples.size() < min_samples)
    {
      XR_LOG_WARN("VisionCapture handeye: need at least %u samples, got %u",
                  static_cast<unsigned>(min_samples),
                  static_cast<unsigned>(samples.size()));
      return false;
    }

    try
    {
      std::vector<cv::Mat> rvec_board2camera;
      std::vector<cv::Mat> t_board2camera;
      std::vector<cv::Mat> R_world2body;
      std::vector<cv::Mat> t_world2body;
      rvec_board2camera.reserve(samples.size());
      t_board2camera.reserve(samples.size());
      R_world2body.reserve(samples.size());
      t_world2body.reserve(samples.size());

      const auto& rb2i = cfg_.handeye_calibration.R_body2imu;
      const cv::Mat R_body2imu =
          (cv::Mat_<double>(3, 3) << rb2i[0], rb2i[1], rb2i[2],
           rb2i[3], rb2i[4], rb2i[5], rb2i[6], rb2i[7], rb2i[8]);

      for (const StableSample& sample : samples)
      {
        const cv::Mat R_imu2world =
            VisionCaptureDetail::QuatWxyzToRotationMatrix(sample.quat);
        const cv::Mat R_body2world = R_imu2world * R_body2imu;

        R_world2body.push_back(R_body2world.t());
        t_world2body.push_back(cv::Mat::zeros(3, 1, CV_64F));
        rvec_board2camera.push_back(sample.rvec.clone());
        t_board2camera.push_back(sample.tvec.clone());
      }

      cv::Mat R_world2board;
      cv::Mat t_world2board;
      cv::Mat R_body2camera;
      cv::Mat t_body2camera;
      cv::calibrateRobotWorldHandEye(
          rvec_board2camera, t_board2camera, R_world2body, t_world2body,
          R_world2board, t_world2board, R_body2camera, t_body2camera);

      cv::Mat R_camera2body;
      cv::transpose(R_body2camera, R_camera2body);
      const cv::Mat t_camera2body = -R_camera2body * t_body2camera;
      const cv::Mat R_mount2body =
          R_camera2body * VisionCaptureDetail::CameraToMountRotation().t();
      const cv::Vec4d q_mount2body =
          VisionCaptureDetail::RotationMatrixToQuatWxyz(R_mount2body);

      const std::filesystem::path yaml_path = output_dir_ / "handeye.yml";
      const std::filesystem::path report_path = output_dir_ / "handeye_report.txt";
      WriteHandEyeYaml(yaml_path, R_body2imu, R_camera2body, t_camera2body,
                       R_mount2body, q_mount2body, R_world2board, t_world2board,
                       static_cast<uint32_t>(samples.size()));
      WriteHandEyeReport(report_path, R_camera2body, t_camera2body,
                         R_mount2body, q_mount2body,
                         R_world2board, t_world2board,
                         static_cast<uint32_t>(samples.size()));

      XR_LOG_PASS("VisionCapture handeye solved: samples=%u result=%s",
                  static_cast<unsigned>(samples.size()),
                  yaml_path.string().c_str());
      return true;
    }
    catch (const cv::Exception& e)
    {
      XR_LOG_ERROR("VisionCapture handeye solve failed: %s", e.what());
    }
    catch (const std::exception& e)
    {
      XR_LOG_ERROR("VisionCapture handeye write failed: %s", e.what());
    }
    return false;
  }

  /**
   * @brief 写出手眼标定 YAML，单位为 m。
   */
  void WriteHandEyeYaml(const std::filesystem::path& path,
                        const cv::Mat& R_body2imu,
                        const cv::Mat& R_camera2body,
                        const cv::Mat& t_camera2body,
                        const cv::Mat& R_mount2body,
                        const cv::Vec4d& q_mount2body,
                        const cv::Mat& R_world2board,
                        const cv::Mat& t_world2board,
                        uint32_t sample_count) const
  {
    cv::FileStorage fs(path.string(), cv::FileStorage::WRITE);
    fs << "sample_count" << static_cast<int>(sample_count);
    fs << "method" << "calibrateRobotWorldHandEye";
    fs << "translation_unit" << "m";
    fs << "body_frame" << "B: x right, y forward, z up";
    fs << "camera_frame" << "C: OpenCV, x right, y down, z forward";
    fs << "mount_frame" << "M: x right, y forward, z up";
    fs << "R_body2imu" << R_body2imu;
    fs << "R_camera2body" << R_camera2body;
    fs << "t_camera2body" << t_camera2body;
    fs << "R_camera2gimbal" << R_camera2body;
    fs << "t_camera2gimbal" << t_camera2body;
    fs << "R_mount2body" << R_mount2body;
    fs << "camera_mount_to_body_rotation_wxyz"
       << std::vector<double>{q_mount2body[0], q_mount2body[1],
                              q_mount2body[2], q_mount2body[3]};
    fs << "camera_mount_to_body_translation" << t_camera2body;
    fs << "R_world2board" << R_world2board;
    fs << "t_world2board" << t_world2board;
  }

  /**
   * @brief 写出便于人工检查的手眼标定报告。
   */
  void WriteHandEyeReport(const std::filesystem::path& path,
                          const cv::Mat& R_camera2body,
                          const cv::Mat& t_camera2body,
                          const cv::Mat& R_mount2body,
                          const cv::Vec4d& q_mount2body,
                          const cv::Mat& R_world2board,
                          const cv::Mat& t_world2board,
                          uint32_t sample_count) const
  {
    std::ofstream out(path, std::ios::out);
    out << "method=calibrateRobotWorldHandEye\n";
    out << "sample_count=" << sample_count << "\n";
    out << "translation_unit=m\n";
    out << "body_frame=B x_right y_forward z_up\n";
    out << "camera_frame=C OpenCV x_right y_down z_forward\n";
    out << "mount_frame=M x_right y_forward z_up\n";
    out << "assumption=t_world2body is zero for every sample\n";
    out << "R_camera2body=\n" << R_camera2body << "\n";
    out << "t_camera2body=\n" << t_camera2body << "\n";
    out << "ArmorTracker camera_mount_to_body.rotation(wxyz)=["
        << q_mount2body[0] << ", " << q_mount2body[1] << ", "
        << q_mount2body[2] << ", " << q_mount2body[3] << "]\n";
    out << "ArmorTracker camera_mount_to_body.translation=\n"
        << t_camera2body << "\n";
    out << "R_mount2body=\n" << R_mount2body << "\n";
    out << "R_world2board=\n" << R_world2board << "\n";
    out << "t_world2board=\n" << t_world2board << "\n";
  }

  /**
   * @brief 标定模式固定保存通过判稳的图像和同步 IMU 元数据。
   */
  void NormalizeCalibrationConfig()
  {
    if (!IsCalibrationDatasetMode())
    {
      return;
    }
    cfg_.record.enabled = true;
    cfg_.record.max_fps = 0.0;
    cfg_.record.save_images = true;
    cfg_.record.save_metadata = true;
    cfg_.record.save_raw_imu = true;
    cfg_.record.flush_every_n = 1;
  }

  /**
   * @brief 生成一行当前采样状态文本。
   */
  std::string BuildStatusLine() const
  {
    std::size_t samples = 0;
    {
      std::lock_guard<std::mutex> lock(sampling_mutex_);
      samples = accepted_calibration_samples_.size();
    }
    std::lock_guard<std::mutex> status_lock(status_mutex_);
    std::ostringstream out;
    out << "mode=" << cfg_.mode
        << " sampling=" << (sampling_running_.load(std::memory_order_acquire) ? 1 : 0)
        << " accepted_total=" << samples
        << " last_reason=" << last_reject_reason_
        << " last_pnp_rms_px=" << last_pnp_rms_px_
        << " last_gyro_norm_dps=" << last_gyro_norm_dps_
        << " last_acc_norm_mps2=" << last_acc_norm_mps2_;
    return out.str();
  }

  /**
   * @brief 创建输出目录并打开元数据文件。
   */
  void PrepareOutput()
  {
    session_name_ = cfg_.session_name.empty()
                        ? VisionCaptureDetail::MakeTimestampSessionName()
                        : VisionCaptureDetail::ToString(cfg_.session_name);
    output_dir_ = std::filesystem::path(
                      VisionCaptureDetail::ToString(cfg_.output_dir)) /
                  session_name_;
    frames_dir_ = output_dir_ / "frames";

    if (cfg_.record.enabled)
    {
      std::filesystem::create_directories(frames_dir_);
      if (cfg_.record.save_metadata)
      {
        metadata_csv_.open(output_dir_ / "samples.csv", std::ios::out);
        metadata_csv_
            << "frame_id,image_timestamp_us,imu_timestamp_us,dt_us,"
               "qw,qx,qy,qz,gx,gy,gz,ax,ay,az,image_path,"
               "board_detected,marker_count,marker_ids,"
               "accepted,reject_reason,pnp_ok,pnp_rms_px,"
               "pnp_t_jitter_m,pnp_r_jitter_deg,imu_r_jitter_deg,"
               "gyro_norm_dps,acc_norm_mps2,acc_norm_error_mps2,"
               "acc_norm_jitter_mps2,acc_dir_jitter_deg\n";
      }
      WriteCameraInfoSnapshot();
    }

    XR_LOG_PASS("VisionCapture output session=%s dir=%s",
                session_name_.c_str(), output_dir_.string().c_str());
  }

  /**
   * @brief 将当前 CameraInfoV 写入采集目录，方便复盘配置来源。
   */
  void WriteCameraInfoSnapshot()
  {
    std::ofstream out(output_dir_ / "camera_info.txt", std::ios::out);
    out << "width=" << CameraInfoV.width << "\n";
    out << "height=" << CameraInfoV.height << "\n";
    out << "step=" << CameraInfoV.step << "\n";
    out << "encoding=" << static_cast<int>(CameraInfoV.encoding) << "\n";
    out << "camera_matrix=";
    for (size_t i = 0; i < CameraInfoV.camera_matrix.size(); ++i)
    {
      if (i != 0) out << ",";
      out << CameraInfoV.camera_matrix[i];
    }
    out << "\ndistortion_coefficients=";
    for (size_t i = 0; i < CameraInfoV.distortion_coefficients.size(); ++i)
    {
      if (i != 0) out << ",";
      out << CameraInfoV.distortion_coefficients[i];
    }
    out << "\n";
  }

  /**
   * @brief 按配置启动相机内参标定流程。
   */
  void StartCameraCalibrationIfNeeded()
  {
    if (!ShouldRunCameraCalibration())
    {
      return;
    }

    std::ostringstream marker;
    marker << std::setprecision(10) << cfg_.camera_calibration.marker_size_mm;
    const bool started = camera_calibration_.Start(
        marker.str(), cfg_.camera_calibration.cols,
        cfg_.camera_calibration.rows, session_name_);
    if (!started)
    {
      XR_LOG_ERROR("VisionCapture camera calibration failed to start");
      return;
    }
    XR_LOG_INFO("VisionCapture camera calibration enabled: marker=%.3fmm board=%dx%d auto_save_views=%u",
                static_cast<float>(cfg_.camera_calibration.marker_size_mm),
                cfg_.camera_calibration.cols, cfg_.camera_calibration.rows,
                static_cast<unsigned>(cfg_.camera_calibration.auto_save_views));
  }

  /**
   * @brief 检查当前帧是否满足记录帧率限制。
   */
  bool AcceptByRate(uint64_t timestamp_us)
  {
    if (cfg_.record.max_fps <= 0.0)
    {
      return true;
    }
    const uint64_t min_period_us =
        static_cast<uint64_t>(1000000.0 / cfg_.record.max_fps);
    if (last_saved_timestamp_us_ != 0 &&
        timestamp_us > last_saved_timestamp_us_ &&
        timestamp_us - last_saved_timestamp_us_ < min_period_us)
    {
      return false;
    }
    return true;
  }

  /**
   * @brief 处理一帧同步图像和 IMU。
   */
  void ProcessFrame(const SyncedFrame& frame)
  {
    const ImageFrame* image_frame = frame.GetImageFrame();
    if (image_frame == nullptr)
    {
      return;
    }
    frames_seen_.fetch_add(1);

    const uint64_t image_ts = static_cast<uint64_t>(image_frame->timestamp_us);
    const uint64_t imu_ts = static_cast<uint64_t>(frame.imu.timestamp_us);
    const uint64_t dt_us = image_ts > imu_ts ? image_ts - imu_ts : imu_ts - image_ts;
    if (cfg_.filter.require_synced_imu && dt_us > cfg_.filter.max_image_imu_dt_us)
    {
      return;
    }

    const cv::Mat image =
        VisionCaptureDetail::MakeImageView<CameraInfoV>(*image_frame);
    if (image.empty())
    {
      if (!unsupported_encoding_logged_)
      {
        unsupported_encoding_logged_ = true;
        XR_LOG_ERROR("VisionCapture unsupported image encoding=%u",
                     static_cast<unsigned>(CameraInfoV.encoding));
      }
      return;
    }

    BoardObservation detection = DetectBoard(image);
    if (detection.observed)
    {
      boards_detected_.fetch_add(1);
    }
    SamplingDecision sampling = EvaluateCalibrationSampling(detection, frame.imu, dt_us);
    SubmitPreview(image, detection);
    ProcessCameraCalibration(image, image_ts, sampling.accepted);

    if (!ShouldSaveFrames())
    {
      return;
    }
    const bool calibration_recording = IsCalibrationDatasetMode();
    if (calibration_recording && !sampling.accepted)
    {
      return;
    }
    if (!calibration_recording && !AcceptByRate(image_ts))
    {
      return;
    }
    if (cfg_.record.max_frames != 0 && total_saved_frames_ >= cfg_.record.max_frames)
    {
      return;
    }

    ++total_saved_frames_;
    last_saved_timestamp_us_ = image_ts;
    const std::string image_path = SaveImage(image, total_saved_frames_);
    WriteMetadata(total_saved_frames_, image_ts, imu_ts, dt_us, frame.imu,
                  image_path, detection, sampling);
    frames_saved_.fetch_add(1);
  }

  /**
   * @brief 当前模式是否按标定样本保存。
   */
  bool IsCalibrationDatasetMode() const
  {
    return cfg_.mode == "calibrate" || cfg_.mode == "calibrate_handeye" ||
           cfg_.mode == "calibrate_camera" || cfg_.camera_calibration.enabled;
  }

  /**
   * @brief 当前模式是否需要求解手眼标定。
   */
  bool ShouldRunHandEyeCalibration() const
  {
    return cfg_.mode == "calibrate" || cfg_.mode == "calibrate_handeye";
  }

  /**
   * @brief 当前配置是否需要运行相机内参标定器。
   */
  bool ShouldRunCameraCalibration() const
  {
    return cfg_.mode == "calibrate" || cfg_.mode == "calibrate_camera" ||
           cfg_.camera_calibration.enabled;
  }

  /**
   * @brief 当前配置是否保存图像和元数据。
   */
  bool ShouldSaveFrames() const
  {
    return cfg_.record.enabled || IsCalibrationDatasetMode();
  }

  /**
   * @brief 将通过判稳的标定样本交给相机内参标定器。
   */
  void ProcessCameraCalibration(const cv::Mat& image, uint64_t image_ts,
                                bool sample_accepted)
  {
    if (!ShouldRunCameraCalibration() || !sample_accepted)
    {
      return;
    }
    camera_calibration_.ProcessFrame(image.data, image_ts);
    if (cfg_.camera_calibration.auto_save_views != 0 &&
        camera_calibration_.SaveAndStopIfReady(
            cfg_.camera_calibration.auto_save_views))
    {
      XR_LOG_PASS("VisionCapture camera calibration auto-saved");
    }
  }

  /// 标定板检测结果类型。
  using BoardObservation = VisionCaptureCalibrationBoard::Observation;

  /**
   * @brief 一帧标定采样判定结果。
   */
  struct SamplingDecision
  {
    /// 当前帧是否通过判稳并保存为标定样本。
    bool accepted = false;
    /// 判定原因。
    std::string reason = "not_evaluated";
    /// true 表示 PnP 求解成功。
    bool pnp_ok = false;
    /// PnP 重投影 RMS，单位像素。
    double pnp_rms_px = 0.0;
    /// 判稳窗口内 PnP 平移最大抖动，单位 m。
    double pnp_t_jitter_m = 0.0;
    /// 判稳窗口内 PnP 旋转最大抖动，单位 deg。
    double pnp_r_jitter_deg = 0.0;
    /// 判稳窗口内 IMU 姿态最大抖动，单位 deg。
    double imu_r_jitter_deg = 0.0;
    /// 当前帧陀螺仪模长，单位 deg/s。
    double gyro_norm_dps = 0.0;
    /// 当前帧加速度模长，单位 m/s^2。
    double acc_norm_mps2 = 0.0;
    /// 当前帧加速度模长与重力加速度差值，单位 m/s^2。
    double acc_norm_error_mps2 = 0.0;
    /// 判稳窗口内加速度模长抖动，单位 m/s^2。
    double acc_norm_jitter_mps2 = 0.0;
    /// 判稳窗口内加速度方向最大抖动，单位 deg。
    double acc_dir_jitter_deg = 0.0;
    /// solvePnP 得到的 Rodrigues 旋转向量。
    cv::Mat rvec{};
    /// solvePnP 得到的平移向量。
    cv::Mat tvec{};
  };

  /**
   * @brief 进入稳定窗口的单帧状态。
   */
  struct StableSample
  {
    /// IMU 时间戳，单位 us。
    uint64_t timestamp_us = 0;
    /// 当前帧标定板到相机的旋转向量。
    cv::Mat rvec{};
    /// 当前帧标定板到相机的平移向量。
    cv::Mat tvec{};
    /// 当前帧 IMU 姿态四元数，wxyz 顺序。
    cv::Vec4d quat{1.0, 0.0, 0.0, 0.0};
    /// 当前帧角速度，单位 rad/s。
    cv::Vec3d gyro{};
    /// 当前帧线加速度，单位 m/s^2。
    cv::Vec3d acc{};
  };

  /**
   * @brief 检测当前图像中的标定板。
   */
  BoardObservation DetectBoard(const cv::Mat& image)
  {
    BoardObservation detection;
    if (cfg_.board.type != "aruco")
    {
      return detection;
    }

    cv::Mat ids;
    const cv::aruco::Dictionary& dictionary = SamplingDictionary();
#if CV_VERSION_MAJOR >= 4 && CV_VERSION_MINOR >= 7
    cv::aruco::ArucoDetector detector(dictionary, detector_params_);
    detector.detectMarkers(image, detection.marker_corners, ids);
#else
    cv::aruco::detectMarkers(image, dictionary, detection.marker_corners, ids,
                             detector_params_);
#endif
    if (!ids.empty())
    {
      detection.marker_ids = ids.clone();
      for (int i = 0; i < ids.rows; ++i)
      {
        detection.marker_ids_vec.push_back(ids.at<int>(i, 0));
      }
      const auto board = SamplingBoard();
      VisionCaptureCalibrationBoard::CollectBoardPoints(detection.marker_corners,
                                             detection.marker_ids, board,
                                             detection);
      detection.homography_rms =
          VisionCaptureCalibrationBoard::HomographyRms(detection.object_points,
                                            detection.image_points);
      VisionCaptureCalibrationBoard::FillQuality(image, CameraInfoV.width, CameraInfoV.height,
                                      detection);
    }
    return detection;
  }

  /**
   * @brief 当前采样使用的 ArUco 字典。
   */
  const cv::aruco::Dictionary& SamplingDictionary() const
  {
    return ShouldRunCameraCalibration() ? camera_sampling_dictionary_ : dictionary_;
  }

  /**
   * @brief 当前采样使用的标定板三维点，单位 m。
   */
  VisionCaptureCalibrationBoard::BoardMap SamplingBoard() const
  {
    if (!ShouldRunCameraCalibration())
    {
      return VisionCaptureCalibrationBoard::MakeSingleArucoBoard(
          cfg_.board.marker_length_m);
    }

    auto board = VisionCaptureCalibrationBoard::MakeGShangBoard(
        cfg_.camera_calibration.marker_size_mm, cfg_.camera_calibration.cols,
        cfg_.camera_calibration.rows);
    for (auto& item : board)
    {
      for (auto& point : item.second)
      {
        point.x *= 0.001F;
        point.y *= 0.001F;
        point.z *= 0.001F;
      }
    }
    return board;
  }

  /**
   * @brief 根据 CameraInfoV 构造 OpenCV 相机内参矩阵。
   */
  cv::Mat CameraMatrix() const
  {
    return (cv::Mat_<double>(3, 3) << CameraInfoV.camera_matrix[0],
            CameraInfoV.camera_matrix[1], CameraInfoV.camera_matrix[2],
            CameraInfoV.camera_matrix[3], CameraInfoV.camera_matrix[4],
            CameraInfoV.camera_matrix[5], CameraInfoV.camera_matrix[6],
            CameraInfoV.camera_matrix[7], CameraInfoV.camera_matrix[8]);
  }

  /**
   * @brief 根据 CameraInfoV 构造 OpenCV 畸变系数矩阵。
   */
  cv::Mat DistortionCoefficients() const
  {
    cv::Mat distortion(1, static_cast<int>(CameraInfoV.distortion_coefficients.size()),
                       CV_64F);
    for (int i = 0; i < distortion.cols; ++i)
    {
      distortion.at<double>(0, i) = CameraInfoV.distortion_coefficients[i];
    }
    return distortion;
  }

  /**
   * @brief 对当前标定板观测执行 PnP 并写入采样判定。
   */
  bool SolveMarkerPnp(const BoardObservation& detection,
                      SamplingDecision& decision) const
  {
    if (!detection.observed)
    {
      decision.reason = "board_not_detected";
      return false;
    }
    try
    {
      const int method = detection.object_points.size() == 4
                             ? cv::SOLVEPNP_IPPE_SQUARE
                             : cv::SOLVEPNP_ITERATIVE;
      const auto pose =
          VisionCaptureCalibrationBoard::EstimatePose(detection, CameraMatrix(),
                                       DistortionCoefficients(), method);
      if (!pose.ok)
      {
        decision.reason = "pnp_failed";
        return false;
      }
      decision.pnp_rms_px = pose.reprojection_rms_px;
      decision.rvec = pose.rvec;
      decision.tvec = pose.tvec;
    }
    catch (const cv::Exception& e)
    {
      decision.reason = "pnp_exception";
      return false;
    }
    if (decision.pnp_rms_px > cfg_.calibration_sampling.max_pnp_reprojection_rms_px)
    {
      decision.reason = "pnp_rms";
      return false;
    }
    decision.pnp_ok = true;
    return true;
  }

  /**
   * @brief 对一帧图像和 IMU 执行判稳采样。
   */
  SamplingDecision EvaluateCalibrationSampling(const BoardObservation& detection,
                                               const ImuStamped& imu,
                                               uint64_t dt_us)
  {
    SamplingDecision decision;
    if (!IsCalibrationDatasetMode() || !cfg_.calibration_sampling.enabled)
    {
      decision.accepted = true;
      decision.reason = "record_all";
      SetLastSamplingStatus(decision);
      return decision;
    }
    if (!sampling_running_.load(std::memory_order_acquire))
    {
      decision.reason = "sampling_paused";
      sampling_rejected_.fetch_add(1);
      SetLastSamplingStatus(decision);
      return decision;
    }
    if (dt_us > cfg_.filter.max_image_imu_dt_us)
    {
      decision.reason = "sync_dt";
      sampling_rejected_.fetch_add(1);
      SetLastSamplingStatus(decision);
      return decision;
    }
    if (!SolveMarkerPnp(detection, decision))
    {
      sampling_rejected_.fetch_add(1);
      SetLastSamplingStatus(decision);
      return decision;
    }
    sampling_pnp_ok_.fetch_add(1);

    StableSample sample;
    sample.timestamp_us = static_cast<uint64_t>(imu.timestamp_us);
    sample.rvec = decision.rvec.clone();
    sample.tvec = decision.tvec.clone();
    sample.quat = VisionCaptureDetail::NormalizeQuatWxyz(imu.rotation_wxyz);
    sample.gyro = VisionCaptureDetail::ToVec3d(imu.angular_velocity_xyz);
    sample.acc = VisionCaptureDetail::ToVec3d(imu.linear_acceleration_xyz);
    decision.gyro_norm_dps = VisionCaptureDetail::RadToDeg(cv::norm(sample.gyro));
    decision.acc_norm_mps2 = cv::norm(sample.acc);
    decision.acc_norm_error_mps2 =
        std::fabs(decision.acc_norm_mps2 - kGravityMps2);

    const bool force_snapshot = force_snapshot_.exchange(false, std::memory_order_acq_rel);
    {
      std::lock_guard<std::mutex> lock(sampling_mutex_);
      stability_window_.push_back(sample);
      while (stability_window_.size() > cfg_.calibration_sampling.window_size)
      {
        stability_window_.pop_front();
      }
      if (stability_window_.size() < cfg_.calibration_sampling.window_size)
      {
        decision.reason = "stability_window";
        sampling_rejected_.fetch_add(1);
        SetLastSamplingStatus(decision);
        return decision;
      }

      ComputeWindowStabilityLocked(sample, decision);
    }
    if (decision.pnp_t_jitter_m >
        cfg_.calibration_sampling.max_pnp_translation_jitter_m)
    {
      decision.reason = "pnp_translation_unstable";
      sampling_rejected_.fetch_add(1);
      SetLastSamplingStatus(decision);
      return decision;
    }
    if (decision.pnp_r_jitter_deg >
        cfg_.calibration_sampling.max_pnp_rotation_jitter_deg)
    {
      decision.reason = "pnp_rotation_unstable";
      sampling_rejected_.fetch_add(1);
      SetLastSamplingStatus(decision);
      return decision;
    }
    if (decision.imu_r_jitter_deg >
        cfg_.calibration_sampling.max_imu_rotation_jitter_deg)
    {
      decision.reason = "imu_rotation_unstable";
      sampling_rejected_.fetch_add(1);
      SetLastSamplingStatus(decision);
      return decision;
    }
    if (decision.gyro_norm_dps > cfg_.calibration_sampling.max_gyro_norm_dps)
    {
      decision.reason = "gyro_moving";
      sampling_rejected_.fetch_add(1);
      SetLastSamplingStatus(decision);
      return decision;
    }
    if (decision.acc_norm_error_mps2 >
        cfg_.calibration_sampling.max_acc_norm_error_mps2)
    {
      decision.reason = "acc_not_gravity";
      sampling_rejected_.fetch_add(1);
      SetLastSamplingStatus(decision);
      return decision;
    }
    if (decision.acc_norm_jitter_mps2 >
        cfg_.calibration_sampling.max_acc_norm_jitter_mps2)
    {
      decision.reason = "acc_vibration";
      sampling_rejected_.fetch_add(1);
      SetLastSamplingStatus(decision);
      return decision;
    }
    if (decision.acc_dir_jitter_deg >
        cfg_.calibration_sampling.max_acc_direction_jitter_deg)
    {
      decision.reason = "acc_direction_unstable";
      sampling_rejected_.fetch_add(1);
      SetLastSamplingStatus(decision);
      return decision;
    }
    {
      std::lock_guard<std::mutex> lock(sampling_mutex_);
      if (!force_snapshot && last_accepted_sample_.timestamp_us != 0 &&
          sample.timestamp_us > last_accepted_sample_.timestamp_us &&
          sample.timestamp_us - last_accepted_sample_.timestamp_us <
              cfg_.calibration_sampling.min_accept_interval_us)
      {
        decision.reason = "accept_interval";
        sampling_rejected_.fetch_add(1);
        SetLastSamplingStatus(decision);
        return decision;
      }
      if (!force_snapshot && IsDuplicateCalibrationSampleLocked(sample))
      {
        decision.reason = "duplicate_pose";
        sampling_rejected_.fetch_add(1);
        SetLastSamplingStatus(decision);
        return decision;
      }
      accepted_calibration_samples_.push_back(sample);
      last_accepted_sample_ = sample;
    }
    decision.accepted = true;
    decision.reason = force_snapshot ? "snapshot" : "accepted";
    sampling_accepted_.fetch_add(1);
    SetLastSamplingStatus(decision);
    return decision;
  }

  /**
   * @brief 计算当前稳定窗口的 PnP、IMU 和加速度抖动。
   */
  void ComputeWindowStabilityLocked(const StableSample& reference,
                                    SamplingDecision& decision) const
  {
    double translation_max = 0.0;
    double rotation_max = 0.0;
    double imu_rotation_max = 0.0;
    double acc_norm_sum = 0.0;
    double acc_norm_sum2 = 0.0;
    cv::Vec3d acc_dir_sum{};
    for (const StableSample& sample : stability_window_)
    {
      translation_max =
          std::max(translation_max, cv::norm(sample.tvec - reference.tvec));
      rotation_max = std::max(rotation_max,
                              VisionCaptureDetail::RotationDistanceDeg(
                                  sample.rvec, reference.rvec));
      imu_rotation_max =
          std::max(imu_rotation_max, VisionCaptureDetail::QuatAngularDistanceDeg(
                                         sample.quat, reference.quat));
      const double acc_norm = cv::norm(sample.acc);
      acc_norm_sum += acc_norm;
      acc_norm_sum2 += acc_norm * acc_norm;
      if (acc_norm > 1e-6)
      {
        acc_dir_sum += sample.acc * (1.0 / acc_norm);
      }
    }
    const double count = static_cast<double>(stability_window_.size());
    const double acc_mean = acc_norm_sum / count;
    const double acc_var = std::max(0.0, acc_norm_sum2 / count - acc_mean * acc_mean);
    cv::Vec3d acc_dir_mean = acc_dir_sum * (1.0 / count);
    const double acc_dir_mean_norm = cv::norm(acc_dir_mean);
    if (acc_dir_mean_norm > 1e-6)
    {
      acc_dir_mean *= 1.0 / acc_dir_mean_norm;
    }
    double acc_dir_max = 0.0;
    for (const StableSample& sample : stability_window_)
    {
      const double acc_norm = cv::norm(sample.acc);
      if (acc_norm <= 1e-6 || acc_dir_mean_norm <= 1e-6)
      {
        continue;
      }
      const cv::Vec3d dir = sample.acc * (1.0 / acc_norm);
      acc_dir_max = std::max(
          acc_dir_max,
          VisionCaptureDetail::RadToDeg(std::acos(
              VisionCaptureCalibrationBoard::ClampUnit(dir.dot(acc_dir_mean)))));
    }
    decision.pnp_t_jitter_m = translation_max;
    decision.pnp_r_jitter_deg = rotation_max;
    decision.imu_r_jitter_deg = imu_rotation_max;
    decision.acc_norm_jitter_mps2 = std::sqrt(acc_var);
    decision.acc_dir_jitter_deg = acc_dir_max;
  }

  /**
   * @brief 判断样本是否与已经接受的样本过近。
   */
  bool IsDuplicateCalibrationSampleLocked(const StableSample& sample) const
  {
    for (const StableSample& accepted : accepted_calibration_samples_)
    {
      const double translation_delta = cv::norm(sample.tvec - accepted.tvec);
      const double rotation_delta =
          VisionCaptureDetail::RotationDistanceDeg(sample.rvec, accepted.rvec);
      if (translation_delta <
              cfg_.calibration_sampling.min_sample_translation_delta_m &&
          rotation_delta < cfg_.calibration_sampling.min_sample_rotation_delta_deg)
      {
        return true;
      }
    }
    return false;
  }

  /**
   * @brief 更新最近一次采样状态，供 monitor/status 输出。
   */
  void SetLastSamplingStatus(const SamplingDecision& decision)
  {
    std::lock_guard<std::mutex> lock(status_mutex_);
    last_reject_reason_ = decision.reason;
    last_pnp_rms_px_ = decision.pnp_rms_px;
    last_gyro_norm_dps_ = decision.gyro_norm_dps;
    last_acc_norm_mps2_ = decision.acc_norm_mps2;
  }

  /**
   * @brief 提交预览图并绘制标定板角点。
   */
  void SubmitPreview(const cv::Mat& image, const BoardObservation& detection)
  {
    if (!preview_.Running())
    {
      return;
    }

    cv::Mat preview_image = VisionCaptureDetail::MakeBgrForPreview(image);
    if (preview_image.empty())
    {
      return;
    }
    preview_.Submit(preview_image,
                    [detection](cv::Mat& frame)
                    {
                      if (detection.observed)
                      {
                        cv::aruco::drawDetectedMarkers(frame,
                                                       detection.marker_corners,
                                                       detection.marker_ids);
                      }
                    });
  }

  /**
   * @brief 按帧号保存图像并返回文件路径。
   */
  std::string SaveImage(const cv::Mat& image, uint64_t frame_id)
  {
    if (!cfg_.record.save_images)
    {
      return {};
    }
    std::ostringstream name;
    name << std::setw(8) << std::setfill('0') << frame_id << "."
         << cfg_.record.image_format;
    const std::filesystem::path path = frames_dir_ / name.str();
    cv::imwrite(path.string(), image);
    return path.string();
  }

  /**
   * @brief 将 marker id 列表格式化为 CSV 单元格。
   */
  static std::string JoinIds(const std::vector<int>& ids)
  {
    std::ostringstream out;
    for (size_t i = 0; i < ids.size(); ++i)
    {
      if (i != 0)
      {
        out << "|";
      }
      out << ids[i];
    }
    return out.str();
  }

  /**
   * @brief 写入一行同步帧和采样状态元数据。
   */
  void WriteMetadata(uint64_t frame_id, uint64_t image_ts, uint64_t imu_ts,
                     uint64_t dt_us, const ImuStamped& imu,
                     const std::string& image_path, const BoardObservation& detection,
                     const SamplingDecision& sampling)
  {
    if (!metadata_csv_.is_open())
    {
      return;
    }
    metadata_csv_ << frame_id << "," << image_ts << "," << imu_ts << ","
                  << dt_us << "," << imu.rotation_wxyz[0] << ","
                  << imu.rotation_wxyz[1] << "," << imu.rotation_wxyz[2] << ","
                  << imu.rotation_wxyz[3] << "," << imu.angular_velocity_xyz[0]
                  << "," << imu.angular_velocity_xyz[1] << ","
                  << imu.angular_velocity_xyz[2] << ","
                  << imu.linear_acceleration_xyz[0] << ","
                  << imu.linear_acceleration_xyz[1] << ","
                  << imu.linear_acceleration_xyz[2] << "," << image_path << ","
                  << (detection.observed ? 1 : 0) << ","
                  << detection.marker_ids_vec.size() << ","
                  << JoinIds(detection.marker_ids_vec) << ","
                  << (sampling.accepted ? 1 : 0) << "," << sampling.reason << ","
                  << (sampling.pnp_ok ? 1 : 0) << "," << sampling.pnp_rms_px
                  << "," << sampling.pnp_t_jitter_m << ","
                  << sampling.pnp_r_jitter_deg << ","
                  << sampling.imu_r_jitter_deg << ","
                  << sampling.gyro_norm_dps << "," << sampling.acc_norm_mps2
                  << "," << sampling.acc_norm_error_mps2 << ","
                  << sampling.acc_norm_jitter_mps2 << ","
                  << sampling.acc_dir_jitter_deg << "\n";
    metadata_csv_.flush();
  }

 private:
  /// 模块运行配置。
  Config cfg_{};
  /// 同步帧来源。
  Sync& sync_;
  /// 可选预览输出。
  VisionPreview preview_{};
  /// 同步帧消费线程。
  LibXR::Thread worker_thread_{};
  /// 标准输入命令线程。
  LibXR::Thread control_thread_{};

  /// OpenCV ArUco 字典。
  cv::aruco::Dictionary dictionary_{};
  /// 相机内参标定采样使用的 ArUco original 字典。
  cv::aruco::Dictionary camera_sampling_dictionary_{
      cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_ORIGINAL)};
  /// OpenCV ArUco 检测参数。
  cv::aruco::DetectorParameters detector_params_{};
  /// 相机内参标定流程对象。
  VisionCaptureCameraCalibration<CameraInfoV> camera_calibration_{};

  /// 标准重力加速度，单位 m/s^2。
  static constexpr double kGravityMps2 = 9.80665;
  /// 标定采样是否正在运行。
  std::atomic<bool> sampling_running_{true};
  /// 是否强制接受下一帧有效 PnP 样本。
  std::atomic<bool> force_snapshot_{false};
  /// 保护稳定窗口和已接受样本。
  mutable std::mutex sampling_mutex_{};
  /// 最近若干帧稳定性计算窗口。
  std::deque<StableSample> stability_window_{};
  /// 已保存的判稳样本。
  std::vector<StableSample> accepted_calibration_samples_{};
  /// 最近一次通过判稳的样本。
  StableSample last_accepted_sample_{};
  /// 保护最近一次采样状态文本。
  mutable std::mutex status_mutex_{};
  /// 最近一次采样判定原因。
  std::string last_reject_reason_{"init"};
  /// 最近一次 PnP RMS，单位像素。
  double last_pnp_rms_px_{0.0};
  /// 最近一次陀螺仪模长，单位 deg/s。
  double last_gyro_norm_dps_{0.0};
  /// 最近一次加速度模长，单位 m/s^2。
  double last_acc_norm_mps2_{0.0};

  /// 当前采集会话名。
  std::string session_name_{};
  /// 当前采集输出目录。
  std::filesystem::path output_dir_{};
  /// 当前图像输出目录。
  std::filesystem::path frames_dir_{};
  /// 同步帧元数据 CSV。
  std::ofstream metadata_csv_{};

  /// 已保存帧总数。
  uint64_t total_saved_frames_{0};
  /// 最近一次保存图像的时间戳，单位 us。
  uint64_t last_saved_timestamp_us_{0};
  /// 是否已经打印过不支持图像编码错误。
  bool unsupported_encoding_logged_{false};

  /// monitor 周期内看到的同步帧数。
  std::atomic<uint64_t> frames_seen_{0};
  /// monitor 周期内保存的同步帧数。
  std::atomic<uint64_t> frames_saved_{0};
  /// monitor 周期内检测到标定板的帧数。
  std::atomic<uint64_t> boards_detected_{0};
  /// monitor 周期内 PnP 成功次数。
  std::atomic<uint64_t> sampling_pnp_ok_{0};
  /// monitor 周期内接受的标定样本数。
  std::atomic<uint64_t> sampling_accepted_{0};
  /// monitor 周期内拒绝的标定样本数。
  std::atomic<uint64_t> sampling_rejected_{0};
};
