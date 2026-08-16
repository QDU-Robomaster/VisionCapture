#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>

#include "CameraBase.hpp"

namespace VisionCaptureSampling
{
/**
 * @brief 请求代次大于已消费代次时，仍有 snapshot 等待有效样本。
 */
inline bool SnapshotPending(uint64_t request_generation, uint64_t consumed_generation)
{
  return request_generation > consumed_generation;
}

/**
 * @brief 只有成功接受样本才消费本帧观察到的 snapshot 代次。
 */
inline uint64_t ConsumeSnapshotGeneration(uint64_t request_generation,
                                          uint64_t consumed_generation,
                                          bool sample_accepted)
{
  return sample_accepted && SnapshotPending(request_generation, consumed_generation)
             ? request_generation
             : consumed_generation;
}

inline constexpr double kStandardGravityMps2 = 9.80665;
inline constexpr double kGShangMarkerSizeMm = 25.0;
inline constexpr int kGShangColumns = 8;
inline constexpr int kGShangRows = 6;

/**
 * @brief VisionCapture 当前运行采用的标定数据契约。
 */
enum class DatasetMode : uint8_t
{
  NONE = 0,
  INTRINSIC,
  HAND_EYE,
};

/**
 * @brief 按显式模式选择唯一的数据采样契约。
 *
 * `calibrate` 是 `calibrate_camera` 的兼容别名。手眼模式优先级最高，避免同时
 * 运行内参求解和手眼数据采集。
 */
inline DatasetMode ClassifyDatasetMode(std::string_view mode,
                                       bool camera_calibration_enabled)
{
  if (mode == "calibrate_handeye")
  {
    return DatasetMode::HAND_EYE;
  }
  if (mode == "calibrate_camera" || mode == "calibrate" ||
      (mode == "record" && camera_calibration_enabled))
  {
    return DatasetMode::INTRINSIC;
  }
  return DatasetMode::NONE;
}

inline const char* DatasetModeName(DatasetMode mode)
{
  switch (mode)
  {
    case DatasetMode::INTRINSIC:
      return "INTRINSIC";
    case DatasetMode::HAND_EYE:
      return "HAND_EYE";
    default:
      return "RECORD";
  }
}

inline bool RequiresImuStability(DatasetMode mode)
{
  return mode == DatasetMode::HAND_EYE;
}

inline bool UsesGShangBoard(DatasetMode mode)
{
  return mode == DatasetMode::INTRINSIC || mode == DatasetMode::HAND_EYE;
}

/**
 * @brief 手眼数据集必须保留原始 IMU；其它模式沿用记录配置。
 */
inline bool ShouldSaveRawImu(DatasetMode mode, bool configured)
{
  return configured || mode == DatasetMode::HAND_EYE;
}

/**
 * @brief 内参求解只要求有效原生图像尺寸，不依赖构造期旧 K/D。
 */
inline bool NativeSensorSizeUsable(const CameraTypes::CameraCalibration& calibration)
{
  return calibration.native_width != 0U && calibration.native_height != 0U;
}

/**
 * @brief 返回 GShang 棋盘中实际放置的 marker 数量。
 */
inline int GShangMarkerCount(int cols = kGShangColumns, int rows = kGShangRows)
{
  if (cols <= 0 || rows <= 0)
  {
    return 0;
  }
  return (cols * rows) / 2;
}

/**
 * @brief 按板上 marker 总数和可见比例计算单帧最低 marker 数。
 */
inline int RequiredGShangMarkerCount(int cols = kGShangColumns, int rows = kGShangRows,
                                     double minimum_ratio = 2.0 / 3.0)
{
  const int marker_count = GShangMarkerCount(cols, rows);
  if (marker_count == 0 || !std::isfinite(minimum_ratio) || minimum_ratio <= 0.0)
  {
    return 0;
  }
  const int by_ratio = static_cast<int>(
      std::ceil(static_cast<double>(marker_count) * std::min(1.0, minimum_ratio)));
  return std::max(1, std::min(marker_count, std::max(4, by_ratio)));
}

/**
 * @brief 当前帧几何对应的可读 profile 分类。
 */
enum class FrameProfile : uint8_t
{
  UNKNOWN = 0,
  FULL_NATIVE,
  WIDE,
  NARROW,
};

inline const char* FrameProfileName(FrameProfile profile)
{
  switch (profile)
  {
    case FrameProfile::FULL_NATIVE:
      return "FULL_NATIVE";
    case FrameProfile::WIDE:
      return "WIDE";
    case FrameProfile::NARROW:
      return "NARROW";
    default:
      return "UNKNOWN";
  }
}

/**
 * @brief 只依据当前帧和原生尺寸分类，不读取或缓存当前相机挡位。
 */
inline FrameProfile ClassifyFrameProfile(uint32_t native_width, uint32_t native_height,
                                         const CameraTypes::FrameGeometry& geometry)
{
  if (native_width == 0U || native_height == 0U || geometry.width == 0U ||
      geometry.height == 0U || geometry.decimation_x == 0U || geometry.decimation_y == 0U)
  {
    return FrameProfile::UNKNOWN;
  }

  if (geometry.decimation_x == 1U && geometry.decimation_y == 1U &&
      geometry.roi_offset_x_native == 0U && geometry.roi_offset_y_native == 0U &&
      geometry.width == native_width && geometry.height == native_height)
  {
    return FrameProfile::FULL_NATIVE;
  }

  const uint64_t covered_width =
      static_cast<uint64_t>(geometry.width) * geometry.decimation_x;
  const uint64_t covered_height =
      static_cast<uint64_t>(geometry.height) * geometry.decimation_y;
  if ((geometry.decimation_x > 1U || geometry.decimation_y > 1U) &&
      geometry.roi_offset_x_native == 0U && geometry.roi_offset_y_native == 0U &&
      covered_width == native_width && covered_height == native_height)
  {
    return FrameProfile::WIDE;
  }

  if (geometry.width > native_width || geometry.height > native_height ||
      geometry.roi_offset_x_native > native_width - geometry.width ||
      geometry.roi_offset_y_native > native_height - geometry.height)
  {
    return FrameProfile::UNKNOWN;
  }
  const uint64_t right_margin =
      static_cast<uint64_t>(native_width) - geometry.width - geometry.roi_offset_x_native;
  const uint64_t bottom_margin = static_cast<uint64_t>(native_height) - geometry.height -
                                 geometry.roi_offset_y_native;
  if (geometry.decimation_x == 1U && geometry.decimation_y == 1U &&
      geometry.width < native_width && geometry.height < native_height &&
      geometry.roi_offset_x_native == right_margin &&
      geometry.roi_offset_y_native == bottom_margin)
  {
    return FrameProfile::NARROW;
  }
  return FrameProfile::UNKNOWN;
}

/**
 * @brief 不依赖既有内参或 IMU 的单帧视觉标定观测。
 */
struct VisualSample
{
  bool observed{false};
  uint64_t image_timestamp_us{0};
  int used_markers{0};
  double homography_rms{0.0};
  double sharpness_score{0.0};
  double center_x_norm{0.0};
  double center_y_norm{0.0};
  double scale_norm{0.0};
  double angle_deg{0.0};
};

/**
 * @brief 内参采样的纯视觉准入门限。
 */
struct IntrinsicLimits
{
  int minimum_markers{RequiredGShangMarkerCount()};
  uint64_t minimum_interval_us{500000};
  double max_homography_rms{2.5};
  double min_sharpness_score{30.0};
  double min_sharpness_best_ratio{0.20};
  double min_center_delta_norm{0.020};
  double min_scale_delta_log{0.05};
  double min_angle_delta_deg{3.0};
};

struct AdmissionResult
{
  bool accepted{false};
  const char* reason{"not_evaluated"};
};

/**
 * @brief 标定模式的采样控制必须 fail-closed；普通记录模式始终保存。
 */
inline AdmissionResult EvaluateSamplingControl(DatasetMode mode, bool enabled,
                                               bool running)
{
  if (mode == DatasetMode::NONE)
  {
    return {true, "record_all"};
  }
  if (!enabled)
  {
    return {false, "sampling_disabled"};
  }
  if (!running)
  {
    return {false, "sampling_paused"};
  }
  return {true, "evaluate"};
}

inline bool VisualMetricsFinite(const VisualSample& sample)
{
  return std::isfinite(sample.homography_rms) && sample.homography_rms >= 0.0 &&
         std::isfinite(sample.sharpness_score) && sample.sharpness_score >= 0.0 &&
         std::isfinite(sample.center_x_norm) && std::isfinite(sample.center_y_norm) &&
         std::isfinite(sample.scale_norm) && sample.scale_norm > 0.0 &&
         std::isfinite(sample.angle_deg);
}

inline double VisualAngleDeltaDeg(double lhs, double rhs)
{
  double delta = std::fmod(std::fabs(lhs - rhs), 180.0);
  return delta > 90.0 ? 180.0 - delta : delta;
}

inline bool IsVisualDuplicate(const VisualSample& sample,
                              std::span<const VisualSample> accepted,
                              const IntrinsicLimits& limits = {})
{
  for (const VisualSample& previous : accepted)
  {
    const double center_delta = std::hypot(sample.center_x_norm - previous.center_x_norm,
                                           sample.center_y_norm - previous.center_y_norm);
    const double scale_delta =
        std::fabs(std::log(sample.scale_norm / previous.scale_norm));
    const double angle_delta = VisualAngleDeltaDeg(sample.angle_deg, previous.angle_deg);
    if (center_delta < limits.min_center_delta_norm &&
        scale_delta < limits.min_scale_delta_log &&
        angle_delta < limits.min_angle_delta_deg)
    {
      return true;
    }
  }
  return false;
}

/**
 * @brief 只依据 GShang 视觉观测和相机时间戳判定内参样本。
 */
inline AdmissionResult EvaluateIntrinsicObservation(
    const VisualSample& sample, std::span<const VisualSample> accepted,
    uint64_t last_accepted_timestamp_us, double best_sharpness_score,
    const IntrinsicLimits& limits = {}, bool force_snapshot = false)
{
  if (!sample.observed)
  {
    return {false, "board_not_detected"};
  }
  if (limits.minimum_markers <= 0 || sample.used_markers < limits.minimum_markers)
  {
    return {false, "marker_count"};
  }
  if (!VisualMetricsFinite(sample))
  {
    return {false, "visual_metrics_invalid"};
  }
  if (!std::isfinite(limits.max_homography_rms) || limits.max_homography_rms <= 0.0 ||
      sample.homography_rms > limits.max_homography_rms)
  {
    return {false, "homography_rms"};
  }
  const double sharpness_threshold = std::max(
      limits.min_sharpness_score, best_sharpness_score * limits.min_sharpness_best_ratio);
  if (!std::isfinite(sharpness_threshold) || sample.sharpness_score < sharpness_threshold)
  {
    return {false, "sharpness"};
  }
  if (sample.image_timestamp_us == 0U)
  {
    return {false, "image_timestamp_invalid"};
  }
  if (last_accepted_timestamp_us != 0U &&
      sample.image_timestamp_us <= last_accepted_timestamp_us)
  {
    return {false, "image_timestamp_non_monotonic"};
  }
  if (!force_snapshot && last_accepted_timestamp_us != 0U &&
      sample.image_timestamp_us - last_accepted_timestamp_us < limits.minimum_interval_us)
  {
    return {false, "accept_interval"};
  }
  if (!force_snapshot && IsVisualDuplicate(sample, accepted, limits))
  {
    return {false, "duplicate_visual"};
  }
  return {true, force_snapshot ? "snapshot" : "accepted"};
}

struct VisualCoverage
{
  double center_span_x{0.0};
  double center_span_y{0.0};
  double scale_ratio{0.0};
};

inline VisualCoverage ComputeVisualCoverage(std::span<const VisualSample> samples)
{
  if (samples.empty())
  {
    return {};
  }
  double center_x_min = std::numeric_limits<double>::infinity();
  double center_x_max = -std::numeric_limits<double>::infinity();
  double center_y_min = std::numeric_limits<double>::infinity();
  double center_y_max = -std::numeric_limits<double>::infinity();
  double scale_min = std::numeric_limits<double>::infinity();
  double scale_max = 0.0;
  for (const VisualSample& sample : samples)
  {
    if (!VisualMetricsFinite(sample))
    {
      continue;
    }
    center_x_min = std::min(center_x_min, sample.center_x_norm);
    center_x_max = std::max(center_x_max, sample.center_x_norm);
    center_y_min = std::min(center_y_min, sample.center_y_norm);
    center_y_max = std::max(center_y_max, sample.center_y_norm);
    scale_min = std::min(scale_min, sample.scale_norm);
    scale_max = std::max(scale_max, sample.scale_norm);
  }
  if (!std::isfinite(center_x_min) || !std::isfinite(scale_min))
  {
    return {};
  }
  return {center_x_max - center_x_min, center_y_max - center_y_min,
          scale_min > 0.0 ? scale_max / scale_min : 0.0};
}

template <std::size_t Size>
inline bool AllFinite(const std::array<float, Size>& values)
{
  for (const float value : values)
  {
    if (!std::isfinite(value))
    {
      return false;
    }
  }
  return true;
}

inline bool QuaternionUsable(const std::array<float, 4>& quaternion)
{
  if (!AllFinite(quaternion))
  {
    return false;
  }
  double norm2 = 0.0;
  for (const float value : quaternion)
  {
    norm2 += static_cast<double>(value) * value;
  }
  return std::isfinite(norm2) && norm2 > 1e-12;
}

/**
 * @brief 检查手眼采样所需的同步 IMU 字段是否完整可用。
 */
inline bool ImuSampleUsable(uint64_t timestamp_us,
                            const std::array<float, 4>& rotation_wxyz,
                            const std::array<float, 3>& angular_velocity_xyz,
                            const std::array<float, 3>& linear_acceleration_xyz)
{
  return timestamp_us != 0U && QuaternionUsable(rotation_wxyz) &&
         AllFinite(angular_velocity_xyz) && AllFinite(linear_acceleration_xyz);
}

inline double AccelerationNormMps2(const std::array<float, 3>& acceleration)
{
  if (!AllFinite(acceleration))
  {
    return std::numeric_limits<double>::infinity();
  }
  double norm2 = 0.0;
  for (const float value : acceleration)
  {
    norm2 += static_cast<double>(value) * value;
  }
  return std::sqrt(norm2);
}

/**
 * @brief 在稳定窗口前检查当前手眼帧的绝对运动和重力量纲。
 */
inline AdmissionResult EvaluateHandEyeAbsoluteImu(double gyro_norm_dps,
                                                  double acc_norm_mps2,
                                                  double max_gyro_norm_dps,
                                                  double max_acc_norm_error_mps2)
{
  if (!std::isfinite(gyro_norm_dps) || !std::isfinite(max_gyro_norm_dps) ||
      max_gyro_norm_dps < 0.0)
  {
    return {false, "gyro_gate_invalid"};
  }
  if (gyro_norm_dps > max_gyro_norm_dps)
  {
    return {false, "gyro_moving"};
  }
  if (!std::isfinite(acc_norm_mps2) || !std::isfinite(max_acc_norm_error_mps2) ||
      max_acc_norm_error_mps2 < 0.0)
  {
    return {false, "acc_gate_invalid"};
  }
  if (std::fabs(acc_norm_mps2 - kStandardGravityMps2) > max_acc_norm_error_mps2)
  {
    return {false, "acc_unit_or_scale"};
  }
  return {true, "absolute_imu_ok"};
}
}  // namespace VisionCaptureSampling
