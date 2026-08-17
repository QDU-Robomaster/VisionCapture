#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <mutex>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

#include "VisionCapture.hpp"
#include "VisionCaptureCalibrationGeometry.hpp"
#include "VisionCaptureCalibrationQuality.hpp"
#include "VisionCaptureCameraCalibration.hpp"
#include "VisionCaptureRecording.hpp"
#include "VisionCaptureSampling.hpp"

namespace
{
inline constexpr CameraTypes::FrameLayout kFullNativeLayout{1440, 1080, 4320,
                                                            CameraTypes::Encoding::BGR8};

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

template <typename Predicate>
bool WaitUntil(Predicate predicate, std::chrono::milliseconds timeout)
{
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!predicate() && std::chrono::steady_clock::now() < deadline)
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return predicate();
}

#if !defined(_WIN32)
bool WriteAll(int fd, std::string_view data)
{
  std::size_t written = 0;
  while (written < data.size())
  {
    const ssize_t count = ::write(fd, data.data() + written, data.size() - written);
    if (count < 0 && errno == EINTR)
    {
      continue;
    }
    if (count <= 0)
    {
      return false;
    }
    written += static_cast<std::size_t>(count);
  }
  return true;
}
#endif

CameraTypes::FrameGeometry MakeGeometry(uint16_t decimation_x, uint16_t decimation_y,
                                        uint16_t flags = 0)
{
  return {720, 540, 2160, 11, 13, decimation_x, decimation_y, flags, 0, 0.0F, 0.0F};
}

VisionCaptureSampling::VisualSample MakeVisualSample(uint64_t timestamp_us)
{
  return {
      .observed = true,
      .image_timestamp_us = timestamp_us,
      .used_markers = VisionCaptureSampling::RequiredGShangMarkerCount(),
      .homography_rms = 0.5,
      .sharpness_score = 100.0,
      .center_x_norm = 0.5,
      .center_y_norm = 0.5,
      .scale_norm = 0.2,
      .angle_deg = 20.0,
  };
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

void TestDatasetModePolicy()
{
  using VisionCaptureSampling::ClassifyDatasetMode;
  using VisionCaptureSampling::DatasetMode;

  Expect(ClassifyDatasetMode("calibrate_camera", false) == DatasetMode::INTRINSIC,
         "calibrate_camera must select intrinsic mode");
  Expect(ClassifyDatasetMode("calibrate", false) == DatasetMode::INTRINSIC,
         "legacy calibrate must select intrinsic mode");
  Expect(ClassifyDatasetMode("record", true) == DatasetMode::INTRINSIC,
         "record plus camera_calibration.enabled must select intrinsic mode");
  Expect(ClassifyDatasetMode("calibrate_handeye", true) == DatasetMode::HAND_EYE,
         "hand-eye mode must win conflicting intrinsic flags");
  Expect(ClassifyDatasetMode("other", true) == DatasetMode::NONE,
         "camera calibration flag must not reinterpret an unknown mode");
  Expect(VisionCaptureSampling::UsesGShangBoard(DatasetMode::INTRINSIC) &&
             VisionCaptureSampling::UsesGShangBoard(DatasetMode::HAND_EYE),
         "both calibration modes must use the GShang board");
  Expect(!VisionCaptureSampling::RequiresImuStability(DatasetMode::INTRINSIC) &&
             VisionCaptureSampling::RequiresImuStability(DatasetMode::HAND_EYE),
         "only hand-eye mode may require IMU stability");
  Expect(
      !VisionCaptureSampling::EvaluateSamplingControl(DatasetMode::INTRINSIC, false, true)
              .accepted &&
          !VisionCaptureSampling::EvaluateSamplingControl(DatasetMode::HAND_EYE, false,
                                                          true)
               .accepted,
      "calibration modes must fail closed when sampling is disabled");
  Expect(VisionCaptureSampling::ShouldSaveRawImu(DatasetMode::HAND_EYE, false) &&
             !VisionCaptureSampling::ShouldSaveRawImu(DatasetMode::INTRINSIC, false),
         "hand-eye must force raw IMU while intrinsic preserves the record "
         "option");
  Expect(VisionCaptureSampling::kGShangMarkerSizeMm == 25.0 &&
             VisionCaptureSampling::kGShangColumns == 8 &&
             VisionCaptureSampling::kGShangRows == 6 &&
             VisionCaptureSampling::GShangMarkerCount() == 24 &&
             VisionCaptureSampling::RequiredGShangMarkerCount() == 16,
         "GShang policy must stay fixed at 25 mm 8x6 with 16 visible markers");
}

void TestIntrinsicVisualAdmission()
{
  const VisionCaptureSampling::IntrinsicLimits limits{};
  const std::span<const VisionCaptureSampling::VisualSample> none{};
  const auto sample = MakeVisualSample(1000000);
  const auto good = VisionCaptureSampling::EvaluateIntrinsicObservation(
      sample, none, 0, sample.sharpness_score, limits);
  Expect(good.accepted,
         "intrinsic admission must pass using only visual and "
         "image-time observables");

  auto marker_failure = sample;
  marker_failure.used_markers = limits.minimum_markers - 1;
  Expect(std::string_view(VisionCaptureSampling::EvaluateIntrinsicObservation(
                              marker_failure, none, 0, 100.0, limits)
                              .reason) == "marker_count",
         "insufficient marker count must fail intrinsic admission");

  auto homography_failure = sample;
  homography_failure.homography_rms = limits.max_homography_rms + 0.1;
  Expect(std::string_view(VisionCaptureSampling::EvaluateIntrinsicObservation(
                              homography_failure, none, 0, 100.0, limits)
                              .reason) == "homography_rms",
         "high homography RMS must fail intrinsic admission");

  auto sharpness_failure = sample;
  sharpness_failure.sharpness_score = limits.min_sharpness_score - 1.0;
  Expect(std::string_view(VisionCaptureSampling::EvaluateIntrinsicObservation(
                              sharpness_failure, none, 0, 100.0, limits)
                              .reason) == "sharpness",
         "blurred views must fail intrinsic admission");

  const std::array<VisionCaptureSampling::VisualSample, 1> accepted{sample};
  auto interval_failure = sample;
  interval_failure.image_timestamp_us =
      sample.image_timestamp_us + limits.minimum_interval_us - 1U;
  Expect(std::string_view(
             VisionCaptureSampling::EvaluateIntrinsicObservation(
                 interval_failure, accepted, sample.image_timestamp_us, 100.0, limits)
                 .reason) == "accept_interval",
         "too-close image timestamps must fail intrinsic admission");

  auto duplicate_failure = sample;
  duplicate_failure.image_timestamp_us =
      sample.image_timestamp_us + limits.minimum_interval_us;
  Expect(std::string_view(
             VisionCaptureSampling::EvaluateIntrinsicObservation(
                 duplicate_failure, accepted, sample.image_timestamp_us, 100.0, limits)
                 .reason) == "duplicate_visual",
         "visually duplicate views must fail intrinsic admission");

  auto moved = duplicate_failure;
  moved.center_x_norm += limits.min_center_delta_norm * 2.0;
  Expect(VisionCaptureSampling::EvaluateIntrinsicObservation(
             moved, accepted, sample.image_timestamp_us, 100.0, limits)
             .accepted,
         "a sufficient visual change must admit the next intrinsic view");

  auto forced_duplicate = duplicate_failure;
  forced_duplicate.image_timestamp_us = sample.image_timestamp_us + 1U;
  const auto forced = VisionCaptureSampling::EvaluateIntrinsicObservation(
      forced_duplicate, accepted, sample.image_timestamp_us, 100.0, limits, true);
  Expect(forced.accepted && std::string_view(forced.reason) == "snapshot",
         "snapshot may bypass interval and visual duplicate gates");

  auto non_monotonic_snapshot = sample;
  non_monotonic_snapshot.image_timestamp_us = sample.image_timestamp_us;
  Expect(std::string_view(VisionCaptureSampling::EvaluateIntrinsicObservation(
                              non_monotonic_snapshot, accepted, sample.image_timestamp_us,
                              100.0, limits, true)
                              .reason) == "image_timestamp_non_monotonic",
         "snapshot must not bypass monotonic image timestamps");
}

void TestSnapshotGenerationConsumption()
{
  Expect(VisionCaptureSampling::SnapshotPending(2U, 1U),
         "new snapshot generation must remain pending");
  Expect(VisionCaptureSampling::ConsumeSnapshotGeneration(2U, 1U, false) == 1U,
         "a rejected frame must not consume snapshot generation");
  Expect(VisionCaptureSampling::ConsumeSnapshotGeneration(2U, 1U, true) == 2U,
         "an accepted frame must consume exactly the generation it observed");
  const uint64_t consumed =
      VisionCaptureSampling::ConsumeSnapshotGeneration(2U, 1U, true);
  Expect(VisionCaptureSampling::SnapshotPending(3U, consumed),
         "a request racing after an observed generation must remain for the "
         "next frame");
}

void TestControlInputLineFraming()
{
#if defined(_WIN32)
  return;
#else
  int descriptors[2]{-1, -1};
  Expect(::pipe(descriptors) == 0, "control input pipe must be created");

  std::mutex lines_mutex;
  std::vector<std::string> lines;
  VisionCaptureDetail::StoppableLineInput input;
  Expect(input.Start(descriptors[0],
                     [&](std::string_view line)
                     {
                       std::lock_guard<std::mutex> lock(lines_mutex);
                       lines.emplace_back(line);
                     }),
         "control input must start on a pipe");

  Expect(WriteAll(descriptors[1], " sta") &&
             WriteAll(descriptors[1], "rt \r\nsnapshot\nstatus"),
         "fragmented control input must be writable");
  Expect(::close(descriptors[1]) == 0, "control input write end must close");
  descriptors[1] = -1;
  Expect(WaitUntil(
             [&]()
             {
               std::lock_guard<std::mutex> lock(lines_mutex);
               return lines.size() == 3U;
             },
             std::chrono::seconds(2)),
         "control input must deliver newline and EOF-terminated commands");
  input.Stop();

  {
    std::lock_guard<std::mutex> lock(lines_mutex);
    Expect(lines[0] == " start \r" && lines[1] == "snapshot" && lines[2] == "status",
           "control input must preserve getline-compatible command text");
  }
  Expect(::close(descriptors[0]) == 0, "control input read end must close");
#endif
}

void TestControlInputStopsWithoutEof()
{
#if defined(_WIN32)
  return;
#else
  int descriptors[2]{-1, -1};
  Expect(::pipe(descriptors) == 0, "stoppable control input pipe must be created");

  std::atomic<uint32_t> lines{0};
  VisionCaptureDetail::StoppableLineInput input;
  Expect(input.Start(descriptors[0], [&](std::string_view) { lines.fetch_add(1U); }),
         "stoppable control input must start");
  Expect(WriteAll(descriptors[1], "pause\npartial"),
         "control input with a partial line must be writable");
  Expect(WaitUntil([&]() { return lines.load() == 1U; }, std::chrono::seconds(2)),
         "complete control command must be delivered before stop");

  const auto start = std::chrono::steady_clock::now();
  input.Stop();
  const auto elapsed = std::chrono::steady_clock::now() - start;
  Expect(elapsed < std::chrono::milliseconds(500),
         "control input stop must not wait for newline or EOF");
  Expect(lines.load() == 1U, "shutdown must not dispatch an incomplete command");

  Expect(::close(descriptors[1]) == 0, "stoppable control write end must close");
  Expect(::close(descriptors[0]) == 0, "stoppable control read end must close");
#endif
}

void TestDropOldestWorkerLifecycle()
{
  VisionCaptureDetail::DropOldestWorkerQueue<int, 2> queue;
  std::atomic<bool> first_started{false};
  std::atomic<bool> release_first{false};
  std::atomic<uint32_t> processed{0};
  queue.Start(
      [&](int value)
      {
        if (value == 1)
        {
          first_started.store(true, std::memory_order_release);
          while (!release_first.load(std::memory_order_acquire))
          {
            std::this_thread::yield();
          }
        }
        processed.fetch_add(1, std::memory_order_acq_rel);
      });
  Expect(queue.Enqueue(1), "running worker queue must accept the first item");
  const auto start_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!first_started.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < start_deadline)
  {
    std::this_thread::yield();
  }
  Expect(first_started.load(std::memory_order_acquire),
         "worker must begin processing the first item");
  Expect(queue.Enqueue(2) && queue.Enqueue(3) && queue.Enqueue(4),
         "running queue must accept bounded pending items");
  Expect(queue.TakeDroppedCount() == 1U,
         "third pending item must drop exactly the oldest queued item");
  release_first.store(true, std::memory_order_release);
  const auto drain_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (processed.load(std::memory_order_acquire) != 3U &&
         std::chrono::steady_clock::now() < drain_deadline)
  {
    std::this_thread::yield();
  }
  Expect(processed.load(std::memory_order_acquire) == 3U,
         "worker must process the active item and two retained newest items");
  queue.Stop();
  Expect(!queue.Enqueue(5), "stopped worker queue must reject enqueue");

  queue.Start([&](int) { processed.fetch_add(1, std::memory_order_acq_rel); });
  Expect(queue.Enqueue(6), "worker queue must support a clean restart");
  const auto restart_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (processed.load(std::memory_order_acquire) != 4U &&
         std::chrono::steady_clock::now() < restart_deadline)
  {
    std::this_thread::yield();
  }
  queue.Stop();
  Expect(processed.load(std::memory_order_acquire) == 4U,
         "restarted worker must process and join cleanly");
}

void TestWorkerExceptionStopsQueue()
{
  VisionCaptureDetail::DropOldestWorkerQueue<int, 2> queue;
  std::atomic<bool> failure_reported{false};
  queue.Start([](int) { throw std::runtime_error("injected worker failure"); },
              [&](std::exception_ptr error)
              {
                try
                {
                  std::rethrow_exception(error);
                }
                catch (const std::runtime_error& exception)
                {
                  failure_reported.store(
                      std::string_view(exception.what()) == "injected worker failure",
                      std::memory_order_release);
                }
              });
  Expect(queue.Enqueue(1), "worker queue must accept the injected failure item");
  const auto failure_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while ((queue.Accepting() || !failure_reported.load(std::memory_order_acquire)) &&
         std::chrono::steady_clock::now() < failure_deadline)
  {
    std::this_thread::yield();
  }
  Expect(!queue.Accepting() && failure_reported.load(std::memory_order_acquire),
         "worker exception must stop admission and report the failure");
  Expect(!queue.Enqueue(2), "failed worker queue must reject later items");
  queue.Stop();
}

void TestImuContract()
{
  const std::array<float, 4> quaternion{1.0F, 0.0F, 0.0F, 0.0F};
  const std::array<float, 3> gyro{0.0F, 0.0F, 0.0F};
  const std::array<float, 3> acceleration{0.0F, 0.0F, 9.80665F};
  Expect(VisionCaptureSampling::ImuSampleUsable(1, quaternion, gyro, acceleration),
         "finite nonzero-timestamp m/s2 IMU data must be usable");
  Expect(!VisionCaptureSampling::ImuSampleUsable(0, quaternion, gyro, acceleration),
         "zero IMU timestamp must fail closed");

  auto zero_quaternion = quaternion;
  zero_quaternion.fill(0.0F);
  Expect(!VisionCaptureSampling::ImuSampleUsable(1, zero_quaternion, gyro, acceleration),
         "zero quaternion must fail closed");

  auto nan_gyro = gyro;
  nan_gyro[1] = std::numeric_limits<float>::quiet_NaN();
  Expect(!VisionCaptureSampling::ImuSampleUsable(1, quaternion, nan_gyro, acceleration),
         "non-finite gyro must fail closed");

  auto infinite_acceleration = acceleration;
  infinite_acceleration[2] = std::numeric_limits<float>::infinity();
  Expect(
      !VisionCaptureSampling::ImuSampleUsable(1, quaternion, gyro, infinite_acceleration),
      "non-finite acceleration must fail closed");

  const double norm = VisionCaptureSampling::AccelerationNormMps2(acceleration);
  Expect(std::abs(norm - VisionCaptureSampling::kStandardGravityMps2) < 1.0e-5,
         "CameraBase acceleration must be interpreted directly as m/s2");
  const std::array<float, 3> wrong_g_units{0.0F, 0.0F, 1.0F};
  const auto wrong_unit_gate = VisionCaptureSampling::EvaluateHandEyeAbsoluteImu(
      0.0, VisionCaptureSampling::AccelerationNormMps2(wrong_g_units), 2.0, 1.5);
  Expect(!wrong_unit_gate.accepted &&
             std::string_view(wrong_unit_gate.reason) == "acc_unit_or_scale",
         "historical approximately-one-g-unit input must fail immediately");
  const auto moving_gate =
      VisionCaptureSampling::EvaluateHandEyeAbsoluteImu(2.1, norm, 2.0, 1.5);
  Expect(!moving_gate.accepted && std::string_view(moving_gate.reason) == "gyro_moving",
         "absolute gyro motion must fail before the stability window");
}

void TestIntrinsicConstructionAndSolveClaim()
{
  CameraTypes::CameraCalibration dimensions_only{};
  dimensions_only.native_width = 1440;
  dimensions_only.native_height = 1080;
  Expect(VisionCaptureSampling::NativeSensorSizeUsable(dimensions_only),
         "intrinsic solver must accept native dimensions without old K/D");
  VisionCaptureCameraCalibration<kFullNativeLayout> calibration(dimensions_only);
  (void)calibration;

  using VisionCaptureCalibrationQuality::DecideSolveClaim;
  using VisionCaptureCalibrationQuality::SolveClaimAction;
  Expect(DecideSolveClaim(true, false, false) == SolveClaimAction::CLAIM,
         "one active caller must claim solve ownership");
  Expect(DecideSolveClaim(false, false, true) == SolveClaimAction::WAIT,
         "a concurrent save must wait for the current solver");
  Expect(DecideSolveClaim(false, true, false) == SolveClaimAction::RETURN_SUCCESS,
         "a completed save must be idempotently successful");
  Expect(DecideSolveClaim(false, false, false) == SolveClaimAction::RETURN_FAILURE,
         "an inactive failed solve must not start a second writer");
}

void TestCalibrationResetAndAbortState()
{
  CameraTypes::CameraCalibration dimensions_only{};
  dimensions_only.native_width = 1440;
  dimensions_only.native_height = 1080;
  VisionCaptureCameraCalibration<kFullNativeLayout> calibration(dimensions_only);

  const std::string unique =
      "vision_capture_reset_" +
      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  const std::filesystem::path root = std::filesystem::temp_directory_path() / unique;
  Expect(std::filesystem::create_directory(root), "reset test directory must be created");
  const std::filesystem::path previous = std::filesystem::current_path();
  std::filesystem::current_path(root);
  Expect(calibration.Start("25mm", 8, 6, "reset-test"),
         "calibration reset test session must start");
  Expect(calibration.Reset() && calibration.AcceptedViewCount() == 0U &&
             !calibration.Finished(),
         "reset must clear solver views and derived completion state");

  const CameraTypes::FrameGeometry geometry{
      1440, 1080, 4320, 0, 0, 1, 1, CameraTypes::FRAME_GEOMETRY_NONE, 0, 0.0F, 0.0F};
  Expect(calibration.ProcessFrame(nullptr, geometry, 1U),
         "reset must leave the calibration sampler active");
  calibration.AbortAndClear();
  Expect(!calibration.ProcessFrame(nullptr, geometry, 2U) &&
             calibration.AcceptedViewCount() == 0U && !calibration.Finished(),
         "record abort must clear views and stop solver admission");
  std::filesystem::current_path(previous);
  Expect(std::filesystem::remove_all(root) != 0U,
         "reset test directory must be removable");
}

void TestFrameProfiles()
{
  using VisionCaptureSampling::ClassifyFrameProfile;
  using VisionCaptureSampling::FrameProfile;

  const CameraTypes::FrameGeometry full{
      1440, 1080, 4320, 0, 0, 1, 1, CameraTypes::FRAME_GEOMETRY_NONE, 0, 0.0F, 0.0F};
  const CameraTypes::FrameGeometry wide{
      720, 540, 2160, 0, 0, 2, 2, CameraTypes::FRAME_GEOMETRY_NONE, 0, 0.5F, 0.5F};
  const CameraTypes::FrameGeometry narrow{
      720, 540, 2160, 360, 270, 1, 1, CameraTypes::FRAME_GEOMETRY_NONE, 0, 0.0F, 0.0F};
  auto reversed_wide = wide;
  reversed_wide.flags =
      CameraTypes::FRAME_GEOMETRY_REVERSE_X | CameraTypes::FRAME_GEOMETRY_REVERSE_Y;
  auto reversed_narrow = narrow;
  reversed_narrow.flags = CameraTypes::FRAME_GEOMETRY_REVERSE_X;
  auto unknown = narrow;
  unknown.roi_offset_x_native = 100;

  Expect(ClassifyFrameProfile(1440, 1080, full) == FrameProfile::FULL_NATIVE,
         "full native geometry must be classified");
  Expect(ClassifyFrameProfile(1440, 1080, wide) == FrameProfile::WIDE &&
             ClassifyFrameProfile(1440, 1080, reversed_wide) == FrameProfile::WIDE,
         "wide profile classification must preserve reversal flags");
  Expect(ClassifyFrameProfile(1440, 1080, narrow) == FrameProfile::NARROW &&
             ClassifyFrameProfile(1440, 1080, reversed_narrow) == FrameProfile::NARROW,
         "narrow profile classification must preserve reversal flags");
  Expect(ClassifyFrameProfile(1440, 1080, unknown) == FrameProfile::UNKNOWN,
         "off-center unsupported ROI must be unknown");
}

void TestCalibrationQualityGate()
{
  const std::array<double, 8> good_rms{0.8, 0.9, 1.0, 1.1, 0.7, 0.8, 0.9, 1.0};
  const std::array<VisionCaptureCalibrationQuality::BoardNormal, 8> good_normals{
      {{-0.25, -0.20, 1.0},
       {-0.20, 0.20, 1.0},
       {0.25, -0.20, 1.0},
       {0.20, 0.20, 1.0},
       {-0.30, -0.12, 1.0},
       {0.30, 0.12, 1.0},
       {-0.12, 0.25, 1.0},
       {0.12, -0.25, 1.0}}};
  const auto good_pose =
      VisionCaptureCalibrationQuality::ComputeTiltCoverage(good_normals);
  VisionCaptureCalibrationQuality::Limits limits;
  limits.minimum_views = good_rms.size();
  const VisionCaptureCalibrationQuality::Coverage good_coverage{
      .center_span_x = 0.40,
      .center_span_y = 0.35,
      .scale_ratio = 1.7,
      .pose_views = good_pose.pose_views,
      .tilt_span_x_deg = good_pose.tilt_span_x_deg,
      .tilt_span_y_deg = good_pose.tilt_span_y_deg,
  };
  const auto good = VisionCaptureCalibrationQuality::Evaluate(
      good_rms.size(), true, 0.95, good_rms, good_coverage, limits);
  Expect(good.views_ok && good.intrinsics_ok && good.reprojection_ok &&
             good.coverage_ok && good.pose_ok && good.quality_ok,
         "well-distributed centers, scales, and board tilts must pass");

  const auto global_rms_failure = VisionCaptureCalibrationQuality::Evaluate(
      good_rms.size(), true, limits.max_global_reprojection_rms + 0.1, good_rms,
      good_coverage, limits);
  Expect(!global_rms_failure.reprojection_ok && !global_rms_failure.quality_ok,
         "high global RMS must fail the calibration quality gate");

  auto poor_coverage = good_coverage;
  poor_coverage.center_span_x = 0.10;
  poor_coverage.center_span_y = 0.10;
  poor_coverage.scale_ratio = 1.1;
  const auto coverage_failure = VisionCaptureCalibrationQuality::Evaluate(
      good_rms.size(), true, 0.95, good_rms, poor_coverage, limits);
  Expect(!coverage_failure.coverage_ok && !coverage_failure.quality_ok,
         "insufficient center and scale coverage must fail calibration quality");

  std::array<VisionCaptureCalibrationQuality::BoardNormal, 8> degenerate_normals{};
  degenerate_normals.fill({0.01, -0.01, 1.0});
  const auto degenerate_pose =
      VisionCaptureCalibrationQuality::ComputeTiltCoverage(degenerate_normals);
  auto degenerate_coverage = good_coverage;
  degenerate_coverage.pose_views = degenerate_pose.pose_views;
  degenerate_coverage.tilt_span_x_deg = degenerate_pose.tilt_span_x_deg;
  degenerate_coverage.tilt_span_y_deg = degenerate_pose.tilt_span_y_deg;
  const auto pose_failure = VisionCaptureCalibrationQuality::Evaluate(
      good_rms.size(), true, 0.95, good_rms, degenerate_coverage, limits);
  Expect(pose_failure.coverage_ok && !pose_failure.pose_ok && !pose_failure.quality_ok,
         "fronto-parallel views must fail even with good center, scale, and "
         "RMS metrics");

  const auto view_failure = VisionCaptureCalibrationQuality::Evaluate(
      good_rms.size() - 1U, true, 0.95,
      std::span<const double>(good_rms).first(good_rms.size() - 1U), good_coverage,
      limits);
  Expect(!view_failure.views_ok && !view_failure.quality_ok,
         "insufficient views must fail the calibration quality gate");
}

void TestMixedProfileGeometryCsv()
{
  const CameraTypes::FrameGeometry wide{
      720, 540, 2160, 0, 0, 2, 2, CameraTypes::FRAME_GEOMETRY_NONE, 0, 0.5F, 0.5F};
  const CameraTypes::FrameGeometry narrow_reverse{
      720,  540,
      2160, 360,
      270,  1,
      1,    CameraTypes::FRAME_GEOMETRY_REVERSE_X | CameraTypes::FRAME_GEOMETRY_REVERSE_Y,
      0,    0.25F,
      0.75F};

  std::ostringstream csv;
  VisionCaptureRecording::WriteFrameGeometryHeader(csv);
  VisionCaptureRecording::WriteFrameGeometryRow(csv, 1, "frames/wide.bmp", wide);
  VisionCaptureRecording::WriteFrameGeometryRow(csv, 2, "frames/narrow,reverse.bmp",
                                                narrow_reverse);

  const std::string expected =
      "frame_id,image_path,width,height,step,roi_offset_x_native,"
      "roi_offset_y_native,decimation_x,decimation_y,flags,reserved,"
      "sample_phase_x_native,sample_phase_y_native\n"
      "1,frames/wide.bmp,720,540,2160,0,0,2,2,0,0,0.5,0.5\n"
      "2,\"frames/"
      "narrow,reverse.bmp\",720,540,2160,360,270,1,1,3,0,0.25,0.75\n";
  Expect(csv.str() == expected,
         "WIDE, NARROW, and reversed geometry must round-trip per recorded frame");

  CameraTypes::FrameGeometry precise_phase = wide;
  precise_phase.sample_phase_x_native = std::nextafter(0.1F, 1.0F);
  precise_phase.sample_phase_y_native = std::nextafter(0.9F, 0.0F);
  std::ostringstream precise_row;
  VisionCaptureRecording::WriteFrameGeometryRow(precise_row, 3, "phase.bmp",
                                                precise_phase);
  const std::string serialized = precise_row.str();
  const std::size_t last_comma = serialized.rfind(',');
  const std::size_t previous_comma = serialized.rfind(',', last_comma - 1U);
  Expect(last_comma != std::string::npos && previous_comma != std::string::npos,
         "geometry row must contain both sample phase fields");
  const float phase_x =
      std::stof(serialized.substr(previous_comma + 1U, last_comma - previous_comma - 1U));
  const float phase_y = std::stof(serialized.substr(last_comma + 1U));
  Expect(phase_x == precise_phase.sample_phase_x_native &&
             phase_y == precise_phase.sample_phase_y_native,
         "sample phases must survive CSV serialization exactly");
}

void TestRecordOptionsAndCheckedWrite()
{
  const std::array<float, 4> rotation{1.0F, 0.0F, 0.0F, 0.0F};
  const std::array<float, 3> angular_velocity{1.0F, 2.0F, 3.0F};
  const std::array<float, 3> acceleration{4.0F, 5.0F, 6.0F};

  std::ostringstream disabled;
  VisionCaptureRecording::WriteRawImuCells(disabled, false, rotation, angular_velocity,
                                           acceleration);
  Expect(disabled.str() == ",,,,,,,,,,",
         "save_raw_imu=false must retain ten empty CSV cells");

  std::ostringstream enabled;
  VisionCaptureRecording::WriteRawImuCells(enabled, true, rotation, angular_velocity,
                                           acceleration);
  Expect(enabled.str() == ",1,0,0,0,1,2,3,4,5,6",
         "save_raw_imu=true must write all ten raw IMU values");

  Expect(!VisionCaptureRecording::ShouldFlush(1, 3),
         "flush_every_n must not flush early");
  Expect(VisionCaptureRecording::ShouldFlush(3, 3),
         "flush_every_n must flush on the configured interval");
  Expect(!VisionCaptureRecording::ShouldFlush(3, 0),
         "flush_every_n=0 must disable periodic flushes");
  Expect(VisionCaptureRecording::CalibrationViewMayCommit(true, true, true, false),
         "persisted accepted samples may enter the solver");
  Expect(!VisionCaptureRecording::CalibrationViewMayCommit(true, true, false, false) &&
             !VisionCaptureRecording::CalibrationViewMayCommit(true, true, true, true),
         "unpersisted or failed records must not enter the solver");
  Expect(VisionCaptureRecording::CalibrationViewMayCommit(true, false, false, false),
         "non-recording calibration paths must not require a persistence receipt");

  const std::string unique =
      "vision_capture_checked_write_" +
      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  const std::filesystem::path root = std::filesystem::temp_directory_path() / unique;
  Expect(std::filesystem::create_directory(root),
         "checked-write test directory must be created");
  const std::filesystem::path output = root / "output.txt";
  const std::string content{"calibration\nreadback\n"};
  Expect(VisionCaptureRecording::WriteTextFile(output, content),
         "checked text output must write and read back successfully");
  std::ifstream in(output, std::ios::binary);
  const std::string readback((std::istreambuf_iterator<char>(in)),
                             std::istreambuf_iterator<char>());
  Expect(readback == content, "checked text output must preserve exact bytes");
  Expect(!VisionCaptureRecording::WriteTextFile(root, content),
         "writing text to an existing directory must fail");
  Expect(std::filesystem::remove(output), "checked-write output must be removable");
  Expect(std::filesystem::remove(root), "checked-write directory must be removable");
}
}  // namespace

int main()
{
  TestFrameResidualPreservesFramePixelUnits();
  TestResidualHandlesAnisotropicReversal();
  TestWeightedGlobalRms();
  TestGeneratedLayoutName();
  TestDatasetModePolicy();
  TestIntrinsicVisualAdmission();
  TestSnapshotGenerationConsumption();
  TestControlInputLineFraming();
  TestControlInputStopsWithoutEof();
  TestDropOldestWorkerLifecycle();
  TestWorkerExceptionStopsQueue();
  TestImuContract();
  TestIntrinsicConstructionAndSolveClaim();
  TestCalibrationResetAndAbortState();
  TestFrameProfiles();
  TestCalibrationQualityGate();
  TestMixedProfileGeometryCsv();
  TestRecordOptionsAndCheckedWrite();
  return 0;
}
