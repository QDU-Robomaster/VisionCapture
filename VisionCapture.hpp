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
inline constexpr size_t kWorkerStackSize = 8192;
inline constexpr size_t kControlStackSize = 4096;
inline constexpr uint32_t kSyncWaitTimeoutMs = 1000;

inline std::string ToString(std::string_view value)
{
  return std::string(value.data(), value.size());
}

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

inline double RadToDeg(double rad) { return rad * 180.0 / CV_PI; }

inline double Norm3(const std::array<float, 3>& value)
{
  const double x = value[0];
  const double y = value[1];
  const double z = value[2];
  return std::sqrt(x * x + y * y + z * z);
}

inline cv::Vec3d ToVec3d(const std::array<float, 3>& value)
{
  return {static_cast<double>(value[0]), static_cast<double>(value[1]),
          static_cast<double>(value[2])};
}

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

inline double QuatAngularDistanceDeg(const cv::Vec4d& lhs, const cv::Vec4d& rhs)
{
  const double dot = std::fabs(lhs[0] * rhs[0] + lhs[1] * rhs[1] +
                              lhs[2] * rhs[2] + lhs[3] * rhs[3]);
  return RadToDeg(2.0 * std::acos(VisionCaptureCalibrationBoard::ClampUnit(dot)));
}

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

template <CameraTypes::CameraInfo CameraInfoV>
class VisionCapture : public LibXR::Application
{
 public:
  using Sync = CameraFrameSync<CameraInfoV>;
  using ImageFrame = typename Sync::ImageFrame;
  using ImuStamped = typename Sync::ImuStamped;
  using SyncedFrame = typename Sync::SyncedFrame;

  struct RecordParams
  {
    bool enabled = true;
    std::string_view image_format = "bmp";
    double max_fps = 30.0;
    uint32_t max_frames = 0;
    bool save_images = true;
    bool save_metadata = true;
    bool save_raw_imu = true;
    uint32_t flush_every_n = 1;
  };

  struct BoardParams
  {
    std::string_view type = "aruco";
    std::string_view dictionary = "DICT_5X5_100";
    double marker_length_m = 0.04;
  };

  struct FilterParams
  {
    bool require_synced_imu = true;
    uint32_t max_image_imu_dt_us = 2000;
  };

  struct CameraCalibrationParams
  {
    bool enabled = false;
    double marker_size_mm = 25.0;
    int cols = 8;
    int rows = 6;
    uint32_t auto_save_views = 120;
  };

  struct CalibrationSamplingParams
  {
    bool enabled = true;
    bool auto_start = true;
    uint32_t window_size = 8;
    uint64_t min_accept_interval_us = 500000;
    double max_pnp_reprojection_rms_px = 2.0;
    double max_pnp_translation_jitter_m = 0.005;
    double max_pnp_rotation_jitter_deg = 1.0;
    double max_imu_rotation_jitter_deg = 0.8;
    double max_gyro_norm_dps = 2.0;
    double max_acc_norm_error_mps2 = 1.5;
    double max_acc_norm_jitter_mps2 = 0.5;
    double max_acc_direction_jitter_deg = 2.0;
    double min_sample_translation_delta_m = 0.03;
    double min_sample_rotation_delta_deg = 5.0;
  };

  struct ControlParams
  {
    bool stdin_enabled = false;
  };

  struct Config
  {
    std::string_view mode = "record";
    std::string_view output_dir = "runs/vision_capture";
    std::string_view session_name = "";
    RecordParams record{};
    VisionPreview::RuntimeParam preview{};
    BoardParams board{};
    CameraCalibrationParams camera_calibration{};
    CalibrationSamplingParams calibration_sampling{};
    ControlParams control{};
    FilterParams filter{};
  };

  VisionCapture(LibXR::HardwareContainer&, LibXR::ApplicationManager& app,
                Config cfg, Sync& sync)
      : cfg_(cfg),
        sync_(sync),
        dictionary_(cv::aruco::getPredefinedDictionary(
            VisionCaptureDetail::ArucoDictionaryId(cfg_.board.dictionary)))
  {
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

  void SolveCurrentCalibration()
  {
    const bool camera_mode = cfg_.mode == "calibrate_camera" ||
                             cfg_.camera_calibration.enabled;
    if (camera_mode)
    {
      if (camera_calibration_.SaveAndStop())
      {
        XR_LOG_PASS("VisionCapture control: camera calibration solved");
      }
      else
      {
        XR_LOG_WARN("VisionCapture control: camera calibration solve failed");
      }
      return;
    }

    std::size_t samples = 0;
    {
      std::lock_guard<std::mutex> lock(sampling_mutex_);
      samples = accepted_calibration_samples_.size();
    }
    XR_LOG_WARN("VisionCapture control: handeye solver is not implemented yet, accepted_samples=%u",
                static_cast<unsigned>(samples));
  }

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

  void StartCameraCalibrationIfNeeded()
  {
    const bool calibration_mode = cfg_.mode == "calibrate_camera";
    if (!calibration_mode && !cfg_.camera_calibration.enabled)
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
    ProcessCameraCalibration(image, image_ts);

    if (!cfg_.record.enabled)
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

  bool IsCalibrationDatasetMode() const
  {
    return cfg_.mode == "calibrate" || cfg_.mode == "calibrate_handeye";
  }

  void ProcessCameraCalibration(const cv::Mat& image, uint64_t image_ts)
  {
    const bool calibration_mode = cfg_.mode == "calibrate_camera";
    if (!calibration_mode && !cfg_.camera_calibration.enabled)
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

  using BoardObservation = VisionCaptureCalibrationBoard::Observation;

  struct SamplingDecision
  {
    bool accepted = false;
    std::string reason = "not_evaluated";
    bool pnp_ok = false;
    double pnp_rms_px = 0.0;
    double pnp_t_jitter_m = 0.0;
    double pnp_r_jitter_deg = 0.0;
    double imu_r_jitter_deg = 0.0;
    double gyro_norm_dps = 0.0;
    double acc_norm_mps2 = 0.0;
    double acc_norm_error_mps2 = 0.0;
    double acc_norm_jitter_mps2 = 0.0;
    double acc_dir_jitter_deg = 0.0;
    cv::Mat rvec{};
    cv::Mat tvec{};
  };

  struct StableSample
  {
    uint64_t timestamp_us = 0;
    cv::Mat rvec{};
    cv::Mat tvec{};
    cv::Vec4d quat{1.0, 0.0, 0.0, 0.0};
    cv::Vec3d gyro{};
    cv::Vec3d acc{};
  };

  BoardObservation DetectBoard(const cv::Mat& image)
  {
    BoardObservation detection;
    if (cfg_.board.type != "aruco")
    {
      return detection;
    }

    cv::Mat ids;
#if CV_VERSION_MAJOR >= 4 && CV_VERSION_MINOR >= 7
    cv::aruco::ArucoDetector detector(dictionary_, detector_params_);
    detector.detectMarkers(image, detection.marker_corners, ids);
#else
    cv::aruco::detectMarkers(image, dictionary_, detection.marker_corners, ids,
                             detector_params_);
#endif
    if (!ids.empty())
    {
      detection.marker_ids = ids.clone();
      for (int i = 0; i < ids.rows; ++i)
      {
        detection.marker_ids_vec.push_back(ids.at<int>(i, 0));
      }
      const auto board = VisionCaptureCalibrationBoard::MakeSingleArucoBoard(
          cfg_.board.marker_length_m);
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

  cv::Mat CameraMatrix() const
  {
    return (cv::Mat_<double>(3, 3) << CameraInfoV.camera_matrix[0],
            CameraInfoV.camera_matrix[1], CameraInfoV.camera_matrix[2],
            CameraInfoV.camera_matrix[3], CameraInfoV.camera_matrix[4],
            CameraInfoV.camera_matrix[5], CameraInfoV.camera_matrix[6],
            CameraInfoV.camera_matrix[7], CameraInfoV.camera_matrix[8]);
  }

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
      const auto pose =
          VisionCaptureCalibrationBoard::EstimatePose(detection, CameraMatrix(),
                                       DistortionCoefficients(),
                                       cv::SOLVEPNP_IPPE_SQUARE);
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

  void SetLastSamplingStatus(const SamplingDecision& decision)
  {
    std::lock_guard<std::mutex> lock(status_mutex_);
    last_reject_reason_ = decision.reason;
    last_pnp_rms_px_ = decision.pnp_rms_px;
    last_gyro_norm_dps_ = decision.gyro_norm_dps;
    last_acc_norm_mps2_ = decision.acc_norm_mps2;
  }

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
  Config cfg_{};
  Sync& sync_;
  VisionPreview preview_{};
  LibXR::Thread worker_thread_{};
  LibXR::Thread control_thread_{};

  cv::aruco::Dictionary dictionary_{};
  cv::aruco::DetectorParameters detector_params_{};
  VisionCaptureCameraCalibration<CameraInfoV> camera_calibration_{};

  static constexpr double kGravityMps2 = 9.80665;
  std::atomic<bool> sampling_running_{true};
  std::atomic<bool> force_snapshot_{false};
  mutable std::mutex sampling_mutex_{};
  std::deque<StableSample> stability_window_{};
  std::vector<StableSample> accepted_calibration_samples_{};
  StableSample last_accepted_sample_{};
  mutable std::mutex status_mutex_{};
  std::string last_reject_reason_{"init"};
  double last_pnp_rms_px_{0.0};
  double last_gyro_norm_dps_{0.0};
  double last_acc_norm_mps2_{0.0};

  std::string session_name_{};
  std::filesystem::path output_dir_{};
  std::filesystem::path frames_dir_{};
  std::ofstream metadata_csv_{};

  uint64_t total_saved_frames_{0};
  uint64_t last_saved_timestamp_us_{0};
  bool unsupported_encoding_logged_{false};

  std::atomic<uint64_t> frames_seen_{0};
  std::atomic<uint64_t> frames_saved_{0};
  std::atomic<uint64_t> boards_detected_{0};
  std::atomic<uint64_t> sampling_pnp_ok_{0};
  std::atomic<uint64_t> sampling_accepted_{0};
  std::atomic<uint64_t> sampling_rejected_{0};
};
