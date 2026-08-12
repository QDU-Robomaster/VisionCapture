#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include "VisionCaptureCalibrationGeometry.hpp"

namespace
{
void Expect(bool condition, const char* message)
{
  if (!condition)
  {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

void ExpectNear(double actual, double expected, const char* message)
{
  if (std::abs(actual - expected) > 1.0e-9)
  {
    std::cerr << message << ": actual=" << actual << " expected=" << expected << '\n';
    std::exit(1);
  }
}

CameraTypes::FrameGeometry MakeGeometry(uint16_t decimation_x, uint16_t decimation_y,
                                        uint16_t flags = 0)
{
  return {1, 720, 540, 2160, 11, 13, decimation_x, decimation_y, flags, 0, 0.0F, 0.0F};
}

void TestFrameResidualPreservesFramePixelUnits()
{
  const auto geometry = MakeGeometry(2, 2);
  const auto observed = CameraTypes::FrameToNative(geometry, 10.0, 20.0);
  const auto projected = CameraTypes::FrameToNative(geometry, 13.0, 24.0);
  ExpectNear(VisionCaptureCalibrationGeometry::FrameResidualSquared(
                 geometry, observed[0], observed[1], projected[0], projected[1]),
             25.0, "2x2 residual must be measured in frame pixels");
}

void TestResidualHandlesAnisotropicReversal()
{
  const auto geometry = MakeGeometry(
      2, 4,
      CameraTypes::FRAME_GEOMETRY_REVERSE_X | CameraTypes::FRAME_GEOMETRY_REVERSE_Y);
  const auto observed = CameraTypes::FrameToNative(geometry, 100.0, 50.0);
  const auto projected = CameraTypes::FrameToNative(geometry, 103.0, 54.0);
  ExpectNear(VisionCaptureCalibrationGeometry::FrameResidualSquared(
                 geometry, observed[0], observed[1], projected[0], projected[1]),
             25.0, "reversal and anisotropic sampling must preserve frame residual");
}

void TestWeightedGlobalRms()
{
  const std::array<double, 2> rms{3.0, 4.0};
  const std::array<std::size_t, 2> counts{4, 12};
  ExpectNear(VisionCaptureCalibrationGeometry::WeightedRms(rms, counts), std::sqrt(14.25),
             "global RMS must be weighted by observed points");
  Expect(VisionCaptureCalibrationGeometry::WeightedRms(
             std::span<const double>{}, std::span<const std::size_t>{}) < 0.0,
         "empty RMS input must be rejected");
}

void TestGeneratedLayoutName()
{
  Expect(VisionCaptureCalibrationGeometry::kFrameLayoutConstexprName ==
             std::string_view{"MainFrameLayout"},
         "calibration snippet must use the active preset layout name");
}
}  // namespace

int main()
{
  TestFrameResidualPreservesFramePixelUnits();
  TestResidualHandlesAnisotropicReversal();
  TestWeightedGlobalRms();
  TestGeneratedLayoutName();
  return 0;
}
