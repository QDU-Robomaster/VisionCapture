#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>

namespace VisionCaptureCalibrationQuality
{
/**
 * @brief SaveAndStop() 在锁内观察到的下一步动作。
 */
enum class SolveClaimAction
{
  CLAIM,
  WAIT,
  RETURN_SUCCESS,
  RETURN_FAILURE,
};

/**
 * @brief 为自动和手动保存选择唯一求解者，并让重复调用复用结果。
 */
inline SolveClaimAction DecideSolveClaim(bool active, bool finished, bool solving)
{
  if (solving)
  {
    return SolveClaimAction::WAIT;
  }
  if (finished)
  {
    return SolveClaimAction::RETURN_SUCCESS;
  }
  if (!active)
  {
    return SolveClaimAction::RETURN_FAILURE;
  }
  return SolveClaimAction::CLAIM;
}

/**
 * @brief 最终标定视角在帧内的覆盖范围。
 */
struct Coverage
{
  double center_span_x{0.0};    ///< 标定板中心横向覆盖，按帧宽归一化。
  double center_span_y{0.0};    ///< 标定板中心纵向覆盖，按帧高归一化。
  double scale_ratio{0.0};      ///< 最大标定板尺度与最小尺度之比。
  std::size_t pose_views{0};    ///< 成功恢复出标定板法向的最终视角数。
  double tilt_span_x_deg{0.0};  ///< 标定板法向在相机 x-z 平面的倾斜跨度。
  double tilt_span_y_deg{0.0};  ///< 标定板法向在相机 y-z 平面的倾斜跨度。
};

/**
 * @brief 相机坐标系中的标定板单位法向。
 */
struct BoardNormal
{
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

/**
 * @brief 从求解外参的标定板法向计算双轴透视倾斜覆盖。
 *
 * 法向统一翻转到相机前向半球，避免同一平面法向符号不确定造成虚假的大跨度。
 * 非有限或零长度法向不计入 pose_views，随后由质量门 fail closed。
 */
inline Coverage ComputeTiltCoverage(std::span<const BoardNormal> normals)
{
  Coverage coverage{};
  double min_x = 0.0;
  double max_x = 0.0;
  double min_y = 0.0;
  double max_y = 0.0;
  bool initialized = false;
  for (const BoardNormal& normal : normals)
  {
    const double norm =
        std::sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
    if (!std::isfinite(norm) || norm <= 1.0e-12)
    {
      continue;
    }
    const double sign = normal.z < 0.0 ? -1.0 : 1.0;
    const double x = sign * normal.x / norm;
    const double y = sign * normal.y / norm;
    const double z = sign * normal.z / norm;
    const double tilt_x = std::atan2(x, z) * 180.0 / 3.14159265358979323846;
    const double tilt_y = std::atan2(y, z) * 180.0 / 3.14159265358979323846;
    if (!std::isfinite(tilt_x) || !std::isfinite(tilt_y))
    {
      continue;
    }
    if (!initialized)
    {
      min_x = max_x = tilt_x;
      min_y = max_y = tilt_y;
      initialized = true;
    }
    else
    {
      min_x = std::min(min_x, tilt_x);
      max_x = std::max(max_x, tilt_x);
      min_y = std::min(min_y, tilt_y);
      max_y = std::max(max_y, tilt_y);
    }
    ++coverage.pose_views;
  }
  if (initialized)
  {
    coverage.tilt_span_x_deg = max_x - min_x;
    coverage.tilt_span_y_deg = max_y - min_y;
  }
  return coverage;
}

/**
 * @brief 可交付内参结果的最低质量门。
 */
struct Limits
{
  std::size_t minimum_views{30};
  double max_global_reprojection_rms{1.5};
  double max_per_view_reprojection_rms{3.0};
  double min_center_span_x{0.35};
  double min_center_span_y{0.30};
  double min_scale_ratio{1.5};
  double min_tilt_span_x_deg{15.0};
  double min_tilt_span_y_deg{15.0};
};

/**
 * @brief 标定候选结果的质量门判定。
 */
struct Result
{
  bool views_ok{false};         ///< 最终视角数是否达到求解下限。
  bool intrinsics_ok{false};    ///< 内参和畸变参数是否通过合理性检查。
  bool reprojection_ok{false};  ///< 全局和逐视角重投影误差是否通过。
  bool coverage_ok{false};      ///< 标定板中心和尺度覆盖是否通过。
  bool pose_ok{false};          ///< 双轴透视倾斜覆盖是否通过。
  bool quality_ok{false};       ///< 所有质量门是否同时通过。
  double per_view_max{-1.0};    ///< 最大逐视角重投影 RMS，单位帧像素。
};

/**
 * @brief 按视角数、内参和重投影误差判定标定候选是否可交付。
 */
inline Result Evaluate(std::size_t views, bool intrinsics_ok, double global_frame_rms,
                       std::span<const double> per_view_rms, const Coverage& coverage,
                       const Limits& limits = {})
{
  Result result{};
  result.views_ok = views >= limits.minimum_views;
  result.intrinsics_ok = intrinsics_ok;

  bool per_view_ok = !per_view_rms.empty() && per_view_rms.size() == views;
  for (const double rms : per_view_rms)
  {
    if (!std::isfinite(rms) || rms < 0.0)
    {
      per_view_ok = false;
      continue;
    }
    result.per_view_max = std::max(result.per_view_max, rms);
  }

  result.reprojection_ok =
      std::isfinite(limits.max_global_reprojection_rms) &&
      limits.max_global_reprojection_rms > 0.0 &&
      std::isfinite(limits.max_per_view_reprojection_rms) &&
      limits.max_per_view_reprojection_rms > 0.0 && std::isfinite(global_frame_rms) &&
      global_frame_rms >= 0.0 && global_frame_rms <= limits.max_global_reprojection_rms &&
      per_view_ok && result.per_view_max <= limits.max_per_view_reprojection_rms;
  result.coverage_ok = std::isfinite(coverage.center_span_x) &&
                       std::isfinite(coverage.center_span_y) &&
                       std::isfinite(coverage.scale_ratio) &&
                       coverage.center_span_x >= limits.min_center_span_x &&
                       coverage.center_span_y >= limits.min_center_span_y &&
                       coverage.scale_ratio >= limits.min_scale_ratio;
  result.pose_ok = coverage.pose_views == views &&
                   std::isfinite(coverage.tilt_span_x_deg) &&
                   std::isfinite(coverage.tilt_span_y_deg) &&
                   std::isfinite(limits.min_tilt_span_x_deg) &&
                   std::isfinite(limits.min_tilt_span_y_deg) &&
                   limits.min_tilt_span_x_deg > 0.0 && limits.min_tilt_span_y_deg > 0.0 &&
                   coverage.tilt_span_x_deg >= limits.min_tilt_span_x_deg &&
                   coverage.tilt_span_y_deg >= limits.min_tilt_span_y_deg;
  result.quality_ok = result.views_ok && result.intrinsics_ok && result.reprojection_ok &&
                      result.coverage_ok && result.pose_ok;
  return result;
}
}  // namespace VisionCaptureCalibrationQuality
