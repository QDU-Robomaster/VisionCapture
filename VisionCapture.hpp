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
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <opencv2/aruco.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "CameraFrameSync.hpp"
#include "VisionPreview.hpp"
#include "app_framework.hpp"
#include "libxr.hpp"
#include "logger.hpp"

namespace VisionCaptureDetail
{
inline constexpr size_t kWorkerStackSize = 8192;
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

  struct Config
  {
    std::string_view mode = "record";
    std::string_view output_dir = "runs/vision_capture";
    std::string_view session_name = "";
    RecordParams record{};
    VisionPreview::RuntimeParam preview{};
    BoardParams board{};
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
    PrepareOutput();
    worker_thread_.Create(this, WorkerThreadFun, "VisionCapture",
                          VisionCaptureDetail::kWorkerStackSize,
                          LibXR::Thread::Priority::MEDIUM);
    app.Register(*this);
  }

  void OnMonitor() override
  {
    const uint64_t frames = frames_seen_.exchange(0);
    const uint64_t saved = frames_saved_.exchange(0);
    const uint64_t boards = boards_detected_.exchange(0);
    XR_LOG_INFO("VisionCapture monitor: frames=%llu saved=%llu boards=%llu",
                static_cast<unsigned long long>(frames),
                static_cast<unsigned long long>(saved),
                static_cast<unsigned long long>(boards));
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
               "board_detected,marker_count,marker_ids\n";
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

    Detection detection = DetectBoard(image);
    if (detection.detected)
    {
      boards_detected_.fetch_add(1);
    }
    SubmitPreview(image, detection);

    if (!cfg_.record.enabled || !AcceptByRate(image_ts))
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
                  image_path, detection);
    frames_saved_.fetch_add(1);
  }

  struct Detection
  {
    bool detected = false;
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
  };

  Detection DetectBoard(const cv::Mat& image)
  {
    Detection detection;
    if (cfg_.board.type != "aruco")
    {
      return detection;
    }

#if CV_VERSION_MAJOR >= 4 && CV_VERSION_MINOR >= 7
    cv::aruco::ArucoDetector detector(dictionary_, detector_params_);
    detector.detectMarkers(image, detection.corners, detection.ids);
#else
    cv::aruco::detectMarkers(image, dictionary_, detection.corners, detection.ids,
                             detector_params_);
#endif
    detection.detected = !detection.ids.empty();
    return detection;
  }

  void SubmitPreview(const cv::Mat& image, const Detection& detection)
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
                      if (detection.detected)
                      {
                        cv::aruco::drawDetectedMarkers(frame, detection.corners,
                                                       detection.ids);
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
                     const std::string& image_path, const Detection& detection)
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
                  << (detection.detected ? 1 : 0) << ","
                  << detection.ids.size() << "," << JoinIds(detection.ids) << "\n";
    if (cfg_.record.flush_every_n != 0 &&
        frame_id % cfg_.record.flush_every_n == 0)
    {
      metadata_csv_.flush();
    }
  }

 private:
  Config cfg_{};
  Sync& sync_;
  VisionPreview preview_{};
  LibXR::Thread worker_thread_{};

  cv::aruco::Dictionary dictionary_{};
  cv::aruco::DetectorParameters detector_params_{};

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
};
