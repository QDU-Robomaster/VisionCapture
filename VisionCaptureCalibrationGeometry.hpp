#pragma once

#include <cmath>
#include <cstddef>
#include <span>
#include <string_view>

#include "CameraBase.hpp"

namespace VisionCaptureCalibrationGeometry
{
inline constexpr std::string_view kFrameLayoutConstexprName = "MainFrameLayout";

[[nodiscard]] inline double FrameResidualSquared(
    const CameraTypes::FrameGeometry& geometry, double observed_native_x,
    double observed_native_y, double projected_native_x, double projected_native_y)
{
  const auto observed_frame =
      CameraTypes::NativeToFrame(geometry, observed_native_x, observed_native_y);
  const auto projected_frame =
      CameraTypes::NativeToFrame(geometry, projected_native_x, projected_native_y);
  const double delta_x = projected_frame[0] - observed_frame[0];
  const double delta_y = projected_frame[1] - observed_frame[1];
  return delta_x * delta_x + delta_y * delta_y;
}

[[nodiscard]] inline double WeightedRms(std::span<const double> rms,
                                        std::span<const std::size_t> sample_counts)
{
  if (rms.size() != sample_counts.size())
  {
    return -1.0;
  }

  double squared_error_sum = 0.0;
  std::size_t sample_count = 0;
  for (std::size_t i = 0; i < rms.size(); ++i)
  {
    if (!std::isfinite(rms[i]) || rms[i] < 0.0)
    {
      return -1.0;
    }
    squared_error_sum += rms[i] * rms[i] * static_cast<double>(sample_counts[i]);
    sample_count += sample_counts[i];
  }

  return sample_count == 0
             ? -1.0
             : std::sqrt(squared_error_sum / static_cast<double>(sample_count));
}
}  // namespace VisionCaptureCalibrationGeometry
