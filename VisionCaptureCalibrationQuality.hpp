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
  double center_span_x{0.0};  ///< 标定板中心横向覆盖，按帧宽归一化。
  double center_span_y{0.0};  ///< 标定板中心纵向覆盖，按帧高归一化。
  double scale_ratio{0.0};    ///< 最大标定板尺度与最小尺度之比。
};

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
  result.quality_ok = result.views_ok && result.intrinsics_ok && result.reprojection_ok &&
                      result.coverage_ok;
  return result;
}
}  // namespace VisionCaptureCalibrationQuality
