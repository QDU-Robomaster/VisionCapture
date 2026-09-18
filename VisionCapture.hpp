#pragma once

// clang-format off
/* === MODULE MANIFEST V2 ===
module_description: 同步图像/IMU 采集与标定数据记录模块
depends:
- id: QDU-Robomaster/CameraFrameSync
  ref: same-or-dev
- id: QDU-Robomaster/VisionPreview
  ref: same-or-dev
- id: QDU-Robomaster/CameraBase
  ref: same-or-dev
=== END MANIFEST === */
// clang-format on

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <mutex>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "CameraFrameSync.hpp"
#include "VisionCaptureCalibrationBoard.hpp"
#include "VisionCaptureCameraCalibration.hpp"
#include "VisionCaptureControlInput.hpp"
#include "VisionCaptureRecording.hpp"
#include "VisionCaptureSampling.hpp"
#include "VisionPreview.hpp"
#include "libxr.hpp"
#include "libxr_def.hpp"
#include "logger.hpp"

namespace VisionCaptureDetail
{
/**
 * @brief 有界 drop-oldest 队列及其拥有的单消费 worker。
 */
template <typename T, std::size_t Capacity>
class DropOldestWorkerQueue
{
  static_assert(Capacity > 0U);

 public:
  using Handler = std::function<void(T)>;
  using FailureHandler = std::function<void(std::exception_ptr)>;

  DropOldestWorkerQueue() = default;
  DropOldestWorkerQueue(const DropOldestWorkerQueue&) = delete;
  DropOldestWorkerQueue& operator=(const DropOldestWorkerQueue&) = delete;

  ~DropOldestWorkerQueue() { Stop(); }

  /** @brief 启动或在 Stop() 后重启 worker；handler 异常会关闭队列并上报。 */
  void Start(Handler handler, FailureHandler failure_handler = {})
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ASSERT(!worker_.joinable());
    ASSERT(static_cast<bool>(handler));
    pending_.clear();
    handler_ = std::move(handler);
    failure_handler_ = std::move(failure_handler);
    stop_ = false;
    accepting_.store(true, std::memory_order_release);
    try
    {
      worker_ = std::thread([this] { Run(); });
    }
    catch (...)
    {
      accepting_.store(false, std::memory_order_release);
      stop_ = true;
      handler_ = nullptr;
      failure_handler_ = nullptr;
      throw;
    }
  }

  /** @brief 非阻塞入队；stop 后返回 false，队满时丢弃最旧待处理项。 */
  bool Enqueue(T item)
  {
    if (!Accepting())
    {
      return false;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stop_ || !Accepting())
      {
        return false;
      }
      while (pending_.size() >= Capacity)
      {
        pending_.pop_front();
        dropped_.fetch_add(1, std::memory_order_relaxed);
      }
      pending_.push_back(std::move(item));
    }
    cv_.notify_one();
    return true;
  }

  /** @brief 拒绝新项、释放待处理项并等待当前 handler 返回。 */
  void Stop()
  {
    accepting_.store(false, std::memory_order_release);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
      pending_.clear();
    }
    cv_.notify_all();
    if (worker_.joinable())
    {
      ASSERT(worker_.get_id() != std::this_thread::get_id());
      worker_.join();
    }
    std::lock_guard<std::mutex> lock(mutex_);
    handler_ = nullptr;
    failure_handler_ = nullptr;
  }

  [[nodiscard]] bool Accepting() const
  {
    return accepting_.load(std::memory_order_acquire);
  }

  uint64_t TakeDroppedCount() { return dropped_.exchange(0, std::memory_order_acq_rel); }

 private:
  void Run()
  {
    while (true)
    {
      std::optional<T> item;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return stop_ || !pending_.empty(); });
        if (stop_)
        {
          pending_.clear();
          return;
        }
        item.emplace(std::move(pending_.front()));
        pending_.pop_front();
      }
      try
      {
        handler_(std::move(*item));
      }
      catch (...)
      {
        const std::exception_ptr error = std::current_exception();
        accepting_.store(false, std::memory_order_release);
        {
          std::lock_guard<std::mutex> lock(mutex_);
          stop_ = true;
          pending_.clear();
        }
        try
        {
          if (failure_handler_)
          {
            failure_handler_(error);
          }
        }
        catch (...)
        {
          // failure handler 也不得让异常越过 std::thread 入口。
        }
        return;
      }
    }
  }

  mutable std::mutex mutex_{};
  std::condition_variable cv_{};
  std::deque<T> pending_{};
  Handler handler_{};
  FailureHandler failure_handler_{};
  std::thread worker_{};
  bool stop_{true};
  std::atomic<bool> accepting_{false};
  std::atomic<uint64_t> dropped_{0};
};

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
template <CameraTypes::FrameLayout FrameLayoutV>
cv::Mat MakeImageView(const typename CameraFrameSync<FrameLayoutV>::ImageFrame& image)
{
  const int width = static_cast<int>(image.geometry.width);
  const int height = static_cast<int>(image.geometry.height);
  const size_t step = static_cast<size_t>(image.geometry.step);
  auto* data = const_cast<uint8_t*>(image.data.data());
  if constexpr (FrameLayoutV.encoding == CameraTypes::Encoding::BGR8 ||
                FrameLayoutV.encoding == CameraTypes::Encoding::RGB8)
  {
    return cv::Mat(height, width, CV_8UC3, data, step);
  }
  else if constexpr (FrameLayoutV.encoding == CameraTypes::Encoding::BGRA8 ||
                     FrameLayoutV.encoding == CameraTypes::Encoding::RGBA8)
  {
    return cv::Mat(height, width, CV_8UC4, data, step);
  }
  else if constexpr (FrameLayoutV.encoding == CameraTypes::Encoding::MONO8)
  {
    return cv::Mat(height, width, CV_8UC1, data, step);
  }
  else
  {
    return {};
  }
}

/**
 * @brief 将 RGB/RGBA 输入转换为 OpenCV 约定的 BGR/BGRA，其他编码保持零拷贝。
 */
template <CameraTypes::FrameLayout FrameLayoutV>
cv::Mat MakeCanonicalImage(const cv::Mat& image)
{
  if constexpr (FrameLayoutV.encoding == CameraTypes::Encoding::RGB8)
  {
    cv::Mat bgr;
    cv::cvtColor(image, bgr, cv::COLOR_RGB2BGR);
    return bgr;
  }
  else if constexpr (FrameLayoutV.encoding == CameraTypes::Encoding::RGBA8)
  {
    cv::Mat bgra;
    cv::cvtColor(image, bgra, cv::COLOR_RGBA2BGRA);
    return bgra;
  }
  else
  {
    return image;
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

inline void DrawOutlinedText(cv::Mat& image, std::string_view text, cv::Point origin,
                             const cv::Scalar& color, double scale = 0.62)
{
  const std::string owned(text);
  cv::putText(image, owned, origin, cv::FONT_HERSHEY_SIMPLEX, scale, {0, 0, 0}, 4,
              cv::LINE_AA);
  cv::putText(image, owned, origin, cv::FONT_HERSHEY_SIMPLEX, scale, color, 1,
              cv::LINE_AA);
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
 * @brief 计算两个四元数之间的最小旋转角。
 */
inline double QuatAngularDistanceDeg(const cv::Vec4d& lhs, const cv::Vec4d& rhs)
{
  const double dot =
      std::fabs(lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2] + lhs[3] * rhs[3]);
  return RadToDeg(2.0 * std::acos(VisionCaptureCalibrationBoard::ClampUnit(dot)));
}

/**
 * @brief 计算两个 Rodrigues 旋转向量之间的最小旋转角。
 */
inline double RotationDistanceDeg(const cv::Mat& lhs_rvec, const cv::Mat& rhs_rvec)
{
  cv::Mat lhs;
  cv::Mat rhs;
  cv::Rodrigues(lhs_rvec, lhs);
  cv::Rodrigues(rhs_rvec, rhs);
  const cv::Mat delta = lhs * rhs.t();
  const double trace =
      delta.at<double>(0, 0) + delta.at<double>(1, 1) + delta.at<double>(2, 2);
  return RadToDeg(
      std::acos(VisionCaptureCalibrationBoard::ClampUnit((trace - 1.0) * 0.5)));
}
}  // namespace VisionCaptureDetail

/**
 * @brief 同步图像和 IMU 采集模块。
 *
 * 模块从 CameraFrameSync 读取同步帧，可保存图像/IMU 元数据、显示预览、
 * 执行相机内参标定采样，并筛选后续手眼标定所需的稳定样本。
 *
 * Topic 回调只复制 SharedFrame 所有权并写入有界队列。析构会停止并 join worker；
 * LibXR Topic 不提供回调注销，因此调用方必须先停止上游发布，再析构本实例。
 */
template <CameraTypes::FrameLayout FrameLayoutV>
class VisionCapture
{
 public:
  /// 对应的 CameraFrameSync 类型。
  using Sync = CameraFrameSync<FrameLayoutV>;
  /// 同步后的图像帧类型。
  using ImageFrame = typename Sync::ImageFrame;
  /// 同步后的 IMU 数据类型。
  using ImuStamped = typename Sync::ImuStamped;
  /// 图像和 IMU 合包类型。
  using SyncedFrame = typename Sync::SyncedFrame;
  /// 同步帧普通 Topic 的借用 payload。
  using SyncedFrameTopicPayload = typename Sync::SyncedFrameTopicPayload;
  /// 原生传感器坐标系下的不可变相机标定。
  using CameraCalibration = typename Sync::CameraCalibration;
  /// 单帧到原生传感器坐标系的采样映射。
  using FrameGeometry = CameraTypes::FrameGeometry;

  /// 编译期固定的帧存储布局。
  static inline constexpr auto frame_layout = Sync::frame_layout;

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
    /// CSV 每写入多少行刷盘一次；0 表示仅在流关闭时刷盘。
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
    /// 旧 YAML 兼容字段；输入已由 CameraFrameSync 配对，标定模式不读取本字段。
    bool require_synced_imu = true;
    /// 旧 YAML 兼容字段；相机和 MCU 时间域不可直接相减，当前实现不使用本字段。
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
     * @brief 供填写 calibration_sampling 的生成代码使用。
     */
    Config(std::string_view mode_in, std::string_view output_dir_in,
           std::string_view session_name_in, RecordParams record_in,
           VisionPreview::RuntimeParam preview_in, BoardParams board_in,
           CameraCalibrationParams camera_calibration_in,
           CalibrationSamplingParams calibration_sampling_in, FilterParams filter_in)
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
           CalibrationSamplingParams calibration_sampling_in, ControlParams control_in,
           FilterParams filter_in)
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

    /// 运行模式：`record`、`calibrate_camera`、`calibrate_handeye` 或
    /// `calibrate`。
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
    /// 本地命令配置。
    ControlParams control{};
    /// 同步帧过滤配置。
    FilterParams filter{};
  };

  /**
   * @brief 构造同步采集模块并订阅同步帧 Topic。
   */
  static Config DefaultConfig() { return {}; }

  VisionCapture(
      Sync& sync,
      Config cfg = DefaultConfig())
      : cfg_(cfg),
        calibration_(sync.Calibration()),
        dictionary_(cv::aruco::getPredefinedDictionary(
            VisionCaptureDetail::ArucoDictionaryId(cfg_.board.dictionary))),
        camera_calibration_(calibration_)
  {

    NormalizeCalibrationConfig();
    detector_params_.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    preview_.Start(cfg_.preview);
    sampling_running_.store(cfg_.calibration_sampling.auto_start,
                            std::memory_order_release);
    PrepareOutput();
    StartCameraCalibrationIfNeeded();
    synced_frame_topic_ =
        LibXR::Topic(LibXR::Topic::FindOrCreate<SyncedFrameTopicPayload>(
            sync.SyncedFrameTopicName()));
    synced_frame_callback_ = LibXR::Topic::Callback::Create(OnSyncedFrameStatic, this);
    frame_queue_.Start([this](SyncedFrame frame) { ProcessFrame(frame); },
                       [this](std::exception_ptr error)
                       { HandleFrameProcessingFailure(error); });
    synced_frame_topic_.RegisterCallback(synced_frame_callback_);
    XR_LOG_INFO("VisionCapture subscribed: topic=%s", sync.SyncedFrameTopicName());

    StartControlInputIfNeeded();
  }

  /**
   * @brief 停止控制输入和帧处理 worker，并释放队列中 retain 的 SharedFrame。
   *
   * 上游必须已经停止发布；LibXR Topic 当前没有回调注销接口。
   */
  ~VisionCapture()
  {
    control_input_.Stop();
    frame_queue_.Stop();
    preview_.Stop();
  }


  /**
   * @brief 周期输出采集、检测和采样计数。
   */
  void OnMonitor()
  {
    const uint64_t frames = frames_seen_.exchange(0);
    const uint64_t saved = frames_saved_.exchange(0);
    const uint64_t boards = boards_detected_.exchange(0);
    const uint64_t pnp_ok = sampling_pnp_ok_.exchange(0);
    const uint64_t accepted = sampling_accepted_.exchange(0);
    const uint64_t rejected = sampling_rejected_.exchange(0);
    const uint64_t queue_dropped = frame_queue_.TakeDroppedCount();
    const std::string status = BuildStatusLine();
    XR_LOG_INFO(
        "VisionCapture monitor: frames=%llu saved=%llu boards=%llu "
        "pnp_ok=%llu accepted=%llu rejected=%llu queue_dropped=%llu %s",
        static_cast<unsigned long long>(frames), static_cast<unsigned long long>(saved),
        static_cast<unsigned long long>(boards), static_cast<unsigned long long>(pnp_ok),
        static_cast<unsigned long long>(accepted),
        static_cast<unsigned long long>(rejected),
        static_cast<unsigned long long>(queue_dropped), status.c_str());
  }

 private:
  /**
   * @brief 在普通 Topic 同步回调内 retain 并入队借用的同步帧。
   */
  static void OnSyncedFrameStatic(bool, VisionCapture* self,
                                  SyncedFrameTopicPayload borrowed)
  {
    if (borrowed == nullptr || !borrowed->Valid() || !self->frame_queue_.Accepting())
    {
      return;
    }
    SyncedFrame retained = *borrowed;
    (void)self->frame_queue_.Enqueue(std::move(retained));
  }

  /**
   * @brief 记录标准输入 worker 的异常退出原因。
   */
  static void LogControlInputFailure(std::exception_ptr error) noexcept
  {
    try
    {
      if (error)
      {
        std::rethrow_exception(error);
      }
    }
    catch (const std::exception& exception)
    {
      XR_LOG_ERROR("VisionCapture stdin control stopped after error: %s",
                   exception.what());
    }
    catch (...)
    {
      XR_LOG_ERROR("VisionCapture stdin control stopped after unknown error");
    }
  }

  /**
   * @brief 启动可取消的标准输入采样控制。
   */
  void StartControlInputIfNeeded() noexcept
  {
    if (!cfg_.control.stdin_enabled)
    {
      return;
    }

    try
    {
      const bool started = control_input_.StartStandardInput(
          [this](std::string_view line) { HandleControlCommand(line); },
          LogControlInputFailure);
      if (!started)
      {
        XR_LOG_ERROR("VisionCapture failed to start stdin control");
        return;
      }
      XR_LOG_INFO(
          "VisionCapture stdin control ready: "
          "help/status/start/pause/reset/solve/snapshot");
    }
    catch (...)
    {
      LogControlInputFailure(std::current_exception());
    }
  }

  /**
   * @brief 执行一条标准输入控制命令。
   */
  void HandleControlCommand(std::string_view command)
  {
    while (!command.empty() && (command.front() == ' ' || command.front() == '\t' ||
                                command.front() == '\r' || command.front() == '\n'))
    {
      command.remove_prefix(1);
    }
    while (!command.empty() && (command.back() == ' ' || command.back() == '\t' ||
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
      const uint64_t generation =
          snapshot_request_generation_.fetch_add(1, std::memory_order_acq_rel) + 1U;
      XR_LOG_INFO("VisionCapture control: snapshot requested generation=%llu",
                  static_cast<unsigned long long>(generation));
    }
    else if (command == "help")
    {
      XR_LOG_INFO(
          "VisionCapture commands: start pause reset solve status "
          "snapshot help");
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
    std::lock_guard<std::mutex> operation_lock(operation_mutex_);
    std::lock_guard<std::mutex> lock(sampling_mutex_);
    stability_window_.clear();
    accepted_calibration_samples_.clear();
    accepted_visual_samples_.clear();
    last_accepted_sample_ = StableSample{};
    last_intrinsic_accept_timestamp_us_ = 0;
    best_intrinsic_sharpness_score_ = 0.0;
    snapshot_consumed_generation_ =
        snapshot_request_generation_.load(std::memory_order_acquire);
    sampling_accepted_total_.store(0, std::memory_order_release);
    {
      std::lock_guard<std::mutex> status_lock(status_mutex_);
      last_reject_reason_ = "reset";
      last_pnp_rms_px_ = 0.0;
      last_gyro_norm_dps_ = 0.0;
      last_acc_norm_mps2_ = 0.0;
    }
    if (ShouldRunCameraCalibration() && !record_io_failed_ &&
        !camera_calibration_.Reset())
    {
      XR_LOG_WARN("VisionCapture reset rejected: camera calibration solve is active");
    }
  }

  /**
   * @brief 根据当前模式触发求解或打印样本状态。
   */
  void SolveCurrentCalibration()
  {
    std::lock_guard<std::mutex> operation_lock(operation_mutex_);
    const auto mode = CurrentDatasetMode();
    if (mode == VisionCaptureSampling::DatasetMode::INTRINSIC)
    {
      if (record_io_failed_)
      {
        XR_LOG_ERROR("VisionCapture control: solve blocked by record I/O failure");
        return;
      }
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

    if (mode == VisionCaptureSampling::DatasetMode::HAND_EYE)
    {
      XR_LOG_WARN(
          "VisionCapture control: handeye solver is not implemented yet, "
          "accepted_samples=%u",
          static_cast<unsigned>(
              sampling_accepted_total_.load(std::memory_order_acquire)));
      return;
    }
    XR_LOG_WARN("VisionCapture control: no calibration solver active");
  }

  /**
   * @brief 标定模式固定保存通过判稳的图像和元数据。
   */
  void NormalizeCalibrationConfig()
  {
    const auto mode = CurrentDatasetMode();
    if (mode == VisionCaptureSampling::DatasetMode::NONE)
    {
      return;
    }
    cfg_.record.enabled = true;
    cfg_.record.max_fps = 0.0;
    cfg_.record.save_images = true;
    cfg_.record.save_metadata = true;
    cfg_.record.save_raw_imu =
        VisionCaptureSampling::ShouldSaveRawImu(mode, cfg_.record.save_raw_imu);
    if (cfg_.camera_calibration.marker_size_mm !=
            VisionCaptureSampling::kGShangMarkerSizeMm ||
        cfg_.camera_calibration.cols != VisionCaptureSampling::kGShangColumns ||
        cfg_.camera_calibration.rows != VisionCaptureSampling::kGShangRows)
    {
      XR_LOG_WARN("VisionCapture calibration board forced to GShang %.0fmm %dx%d",
                  VisionCaptureSampling::kGShangMarkerSizeMm,
                  VisionCaptureSampling::kGShangColumns,
                  VisionCaptureSampling::kGShangRows);
    }
    cfg_.camera_calibration.marker_size_mm = VisionCaptureSampling::kGShangMarkerSizeMm;
    cfg_.camera_calibration.cols = VisionCaptureSampling::kGShangColumns;
    cfg_.camera_calibration.rows = VisionCaptureSampling::kGShangRows;
  }

  /**
   * @brief 生成一行当前采样状态文本。
   */
  std::string BuildStatusLine() const
  {
    const auto mode = CurrentDatasetMode();
    const uint64_t samples = sampling_accepted_total_.load(std::memory_order_acquire);
    std::lock_guard<std::mutex> status_lock(status_mutex_);
    std::ostringstream out;
    out << "mode=" << VisionCaptureSampling::DatasetModeName(mode)
        << " sampling=" << (sampling_running_.load(std::memory_order_acquire) ? 1 : 0)
        << " accepted_total=" << samples
        << " target=" << cfg_.camera_calibration.auto_save_views << " solver_views="
        << (mode == VisionCaptureSampling::DatasetMode::INTRINSIC
                ? camera_calibration_.AcceptedViewCount()
                : 0U)
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
    output_dir_ = std::filesystem::path(VisionCaptureDetail::ToString(cfg_.output_dir)) /
                  session_name_;
    frames_dir_ = output_dir_ / "frames";

    if (cfg_.record.enabled)
    {
      std::filesystem::create_directories(frames_dir_);
      if (cfg_.record.save_metadata)
      {
        metadata_csv_.open(output_dir_ / "samples.csv", std::ios::out);
        metadata_csv_ << "frame_id,image_timestamp_us,imu_timestamp_us,dt_us,"
                         "qw,qx,qy,qz,gx,gy,gz,ax,ay,az,image_path,"
                         "board_detected,marker_count,marker_ids,"
                         "accepted,reject_reason,pnp_ok,pnp_rms_px,"
                         "pnp_t_jitter_m,pnp_r_jitter_deg,imu_r_jitter_deg,"
                         "gyro_norm_dps,acc_norm_mps2,acc_norm_error_mps2,"
                         "acc_norm_jitter_mps2,acc_dir_jitter_deg,"
                         "acceleration_unit\n";
        if (!metadata_csv_)
        {
          record_io_failed_ = true;
          XR_LOG_ERROR("VisionCapture failed to open samples.csv");
        }
      }
      frame_geometry_csv_.open(output_dir_ / "frame_geometry.csv", std::ios::out);
      VisionCaptureRecording::WriteFrameGeometryHeader(frame_geometry_csv_);
      if (!frame_geometry_csv_)
      {
        record_io_failed_ = true;
        XR_LOG_ERROR("VisionCapture failed to open frame_geometry.csv");
      }
      if (!WriteStaticCameraSnapshots())
      {
        record_io_failed_ = true;
        XR_LOG_ERROR("VisionCapture failed to write static camera snapshots");
      }
    }

    if (record_io_failed_)
    {
      XR_LOG_ERROR("VisionCapture output unavailable: session=%s dir=%s",
                   session_name_.c_str(), output_dir_.string().c_str());
    }
    else
    {
      XR_LOG_PASS("VisionCapture output session=%s dir=%s", session_name_.c_str(),
                  output_dir_.string().c_str());
    }
  }

  template <std::size_t Size>
  static void WriteArrayLine(std::ostream& out, std::string_view name,
                             const std::array<double, Size>& values)
  {
    out << name << "=";
    for (std::size_t i = 0; i < values.size(); ++i)
    {
      if (i != 0)
      {
        out << ",";
      }
      out << values[i];
    }
    out << "\n";
  }

  /**
   * @brief 分别保存固定帧布局和原生相机标定。
   */
  bool WriteStaticCameraSnapshots()
  {
    std::ostringstream layout;
    layout << "width=" << frame_layout.width << "\n";
    layout << "height=" << frame_layout.height << "\n";
    layout << "step=" << frame_layout.step << "\n";
    layout << "encoding=" << static_cast<int>(frame_layout.encoding) << "\n";

    std::ostringstream calibration;
    calibration << std::setprecision(17);
    calibration << "native_width=" << calibration_.native_width << "\n";
    calibration << "native_height=" << calibration_.native_height << "\n";
    calibration << "distortion_model=" << static_cast<int>(calibration_.distortion_model)
                << "\n";
    WriteArrayLine(calibration, "camera_matrix", calibration_.camera_matrix);
    WriteArrayLine(calibration, "distortion_coefficients",
                   calibration_.distortion_coefficients);
    WriteArrayLine(calibration, "rectification_matrix",
                   calibration_.rectification_matrix);
    WriteArrayLine(calibration, "projection_matrix", calibration_.projection_matrix);
    return VisionCaptureRecording::WriteTextFile(output_dir_ / "frame_layout.txt",
                                                 layout.str()) &&
           VisionCaptureRecording::WriteTextFile(output_dir_ / "camera_calibration.txt",
                                                 calibration.str());
  }

  /**
   * @brief 从原生 K 和当前采样几何派生兼容旧记录工具的帧坐标 K。
   */
  std::array<double, 9> FrameCameraMatrix(const FrameGeometry& geometry) const
  {
    const auto& native = calibration_.camera_matrix;
    const double scale_x =
        CameraTypes::HasGeometryFlag(geometry, CameraTypes::FRAME_GEOMETRY_REVERSE_X)
            ? -1.0 / static_cast<double>(geometry.decimation_x)
            : 1.0 / static_cast<double>(geometry.decimation_x);
    const double scale_y =
        CameraTypes::HasGeometryFlag(geometry, CameraTypes::FRAME_GEOMETRY_REVERSE_Y)
            ? -1.0 / static_cast<double>(geometry.decimation_y)
            : 1.0 / static_cast<double>(geometry.decimation_y);
    const double translate_x =
        CameraTypes::HasGeometryFlag(geometry, CameraTypes::FRAME_GEOMETRY_REVERSE_X)
            ? static_cast<double>(geometry.width - 1U) +
                  (static_cast<double>(geometry.roi_offset_x_native) +
                   static_cast<double>(geometry.sample_phase_x_native)) /
                      static_cast<double>(geometry.decimation_x)
            : -(static_cast<double>(geometry.roi_offset_x_native) +
                static_cast<double>(geometry.sample_phase_x_native)) /
                  static_cast<double>(geometry.decimation_x);
    const double translate_y =
        CameraTypes::HasGeometryFlag(geometry, CameraTypes::FRAME_GEOMETRY_REVERSE_Y)
            ? static_cast<double>(geometry.height - 1U) +
                  (static_cast<double>(geometry.roi_offset_y_native) +
                   static_cast<double>(geometry.sample_phase_y_native)) /
                      static_cast<double>(geometry.decimation_y)
            : -(static_cast<double>(geometry.roi_offset_y_native) +
                static_cast<double>(geometry.sample_phase_y_native)) /
                  static_cast<double>(geometry.decimation_y);

    std::array<double, 9> frame{};
    for (std::size_t col = 0; col < 3; ++col)
    {
      frame[col] = scale_x * native[col] + translate_x * native[6 + col];
      frame[3 + col] = scale_y * native[3 + col] + translate_y * native[6 + col];
      frame[6 + col] = native[6 + col];
    }
    return frame;
  }

  /**
   * @brief 首帧到达时保存 geometry，并保留旧 camera_info.txt 派生快照。
   */
  bool WriteFrameGeometrySnapshot(const FrameGeometry& geometry)
  {
    if (frame_geometry_snapshot_written_ || !cfg_.record.enabled)
    {
      return true;
    }

    std::ostringstream frame_geometry;
    frame_geometry << std::setprecision(17);
    frame_geometry << "width=" << geometry.width << "\n";
    frame_geometry << "height=" << geometry.height << "\n";
    frame_geometry << "step=" << geometry.step << "\n";
    frame_geometry << "roi_offset_x_native=" << geometry.roi_offset_x_native << "\n";
    frame_geometry << "roi_offset_y_native=" << geometry.roi_offset_y_native << "\n";
    frame_geometry << "decimation_x=" << geometry.decimation_x << "\n";
    frame_geometry << "decimation_y=" << geometry.decimation_y << "\n";
    frame_geometry << "flags=" << geometry.flags << "\n";
    frame_geometry << "sample_phase_x_native=" << geometry.sample_phase_x_native << "\n";
    frame_geometry << "sample_phase_y_native=" << geometry.sample_phase_y_native << "\n";

    std::ostringstream camera_info;
    camera_info << std::setprecision(17);
    camera_info << "width=" << geometry.width << "\n";
    camera_info << "height=" << geometry.height << "\n";
    camera_info << "step=" << geometry.step << "\n";
    camera_info << "encoding=" << static_cast<int>(frame_layout.encoding) << "\n";
    WriteArrayLine(camera_info, "camera_matrix", FrameCameraMatrix(geometry));
    WriteArrayLine(camera_info, "distortion_coefficients",
                   calibration_.distortion_coefficients);
    if (!VisionCaptureRecording::WriteTextFile(output_dir_ / "frame_geometry.txt",
                                               frame_geometry.str()) ||
        !VisionCaptureRecording::WriteTextFile(output_dir_ / "camera_info.txt",
                                               camera_info.str()))
    {
      return false;
    }
    frame_geometry_snapshot_written_ = true;
    return true;
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
    if (record_io_failed_)
    {
      XR_LOG_ERROR("VisionCapture camera calibration disabled: output unavailable");
      return;
    }

    std::ostringstream marker;
    marker << std::setprecision(10) << cfg_.camera_calibration.marker_size_mm;
    const bool started =
        camera_calibration_.Start(marker.str(), cfg_.camera_calibration.cols,
                                  cfg_.camera_calibration.rows, session_name_);
    if (!started)
    {
      XR_LOG_ERROR("VisionCapture camera calibration failed to start");
      return;
    }
    XR_LOG_INFO(
        "VisionCapture camera calibration enabled: marker=%.3fmm board=%dx%d "
        "auto_save_views=%u",
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
    const uint64_t min_period_us = static_cast<uint64_t>(1000000.0 / cfg_.record.max_fps);
    if (last_saved_timestamp_us_ != 0 && timestamp_us > last_saved_timestamp_us_ &&
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
    std::lock_guard<std::mutex> operation_lock(operation_mutex_);
    const ImageFrame* image_frame = frame.GetImageFrame();
    if (image_frame == nullptr)
    {
      return;
    }
    frames_seen_.fetch_add(1);

    if (!CameraTypes::ValidateFrameGeometry(frame_layout, calibration_,
                                            image_frame->geometry))
    {
      if (!invalid_geometry_logged_)
      {
        invalid_geometry_logged_ = true;
        XR_LOG_ERROR("VisionCapture rejected invalid FrameGeometry");
      }
      return;
    }
    if (record_io_failed_ || !WriteFrameGeometrySnapshot(image_frame->geometry))
    {
      LatchRecordIoFailure("frame snapshot");
      return;
    }

    const uint64_t image_ts = static_cast<uint64_t>(image_frame->timestamp_us);
    const uint64_t imu_ts = static_cast<uint64_t>(frame.imu.timestamp_us);
    const cv::Mat raw_image =
        VisionCaptureDetail::MakeImageView<FrameLayoutV>(*image_frame);
    if (raw_image.empty())
    {
      if (!unsupported_encoding_logged_)
      {
        unsupported_encoding_logged_ = true;
        XR_LOG_ERROR("VisionCapture unsupported image encoding=%u",
                     static_cast<unsigned>(FrameLayoutV.encoding));
      }
      return;
    }
    const cv::Mat image =
        VisionCaptureDetail::MakeCanonicalImage<FrameLayoutV>(raw_image);

    BoardObservation detection = DetectBoard(image, image_frame->geometry);
    if (detection.observed)
    {
      boards_detected_.fetch_add(1);
    }
    SamplingDecision sampling = EvaluateCalibrationSampling(
        detection, image_frame->geometry, frame.imu, image_ts);
    SubmitPreview(image, detection, sampling, image_frame->geometry);

    if (!ShouldSaveFrames())
    {
      ProcessCameraCalibration(raw_image, image_frame->geometry, image_ts,
                               sampling.accepted, false);
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

    const uint64_t frame_id = total_saved_frames_ + 1U;
    std::string image_path;
    if (!SaveImage(image, frame_id, image_path))
    {
      LatchRecordIoFailure("image write");
      return;
    }
    if (!WriteFrameGeometry(frame_id, image_path, image_frame->geometry) ||
        !WriteMetadata(frame_id, image_ts, imu_ts, frame.imu, image_path, detection,
                       sampling))
    {
      LatchRecordIoFailure("CSV write");
      return;
    }

    if (!FlushCsvIfNeeded())
    {
      LatchRecordIoFailure("CSV flush");
      return;
    }
    total_saved_frames_ = frame_id;
    last_saved_timestamp_us_ = image_ts;
    frames_saved_.fetch_add(1);
    ProcessCameraCalibration(raw_image, image_frame->geometry, image_ts,
                             sampling.accepted, true);
  }

  /**
   * @brief 锁存记录失败并撤销全部尚未求解的内参视角。
   */
  void LatchRecordIoFailure(std::string_view stage)
  {
    const bool first_failure = !record_io_failed_;
    record_io_failed_ = true;
    camera_calibration_.AbortAndClear();
    if (first_failure)
    {
      const std::string owned_stage(stage);
      XR_LOG_ERROR("VisionCapture recording disabled after I/O failure: %s",
                   owned_stage.c_str());
    }
  }

  /**
   * @brief 关闭异常退出的异步处理路径，避免异常越过 std::thread 入口。
   */
  void HandleFrameProcessingFailure(const std::exception_ptr& error) noexcept
  {
    try
    {
      std::lock_guard<std::mutex> operation_lock(operation_mutex_);
      LatchRecordIoFailure("frame processing exception");
    }
    catch (...)
    {
      // 异常路径必须保持 noexcept，确保 worker 能正常退出并被 join。
    }

    try
    {
      if (error)
      {
        std::rethrow_exception(error);
      }
    }
    catch (const std::exception& exception)
    {
      XR_LOG_ERROR("VisionCapture frame worker stopped after exception: %s",
                   exception.what());
    }
    catch (...)
    {
      XR_LOG_ERROR("VisionCapture frame worker stopped after unknown exception");
    }
  }

  /**
   * @brief 当前模式是否按标定样本保存。
   */
  bool IsCalibrationDatasetMode() const
  {
    return CurrentDatasetMode() != VisionCaptureSampling::DatasetMode::NONE;
  }

  VisionCaptureSampling::DatasetMode CurrentDatasetMode() const
  {
    return VisionCaptureSampling::ClassifyDatasetMode(cfg_.mode,
                                                      cfg_.camera_calibration.enabled);
  }

  /**
   * @brief 当前配置是否需要运行相机内参标定器。
   */
  bool ShouldRunCameraCalibration() const
  {
    return CurrentDatasetMode() == VisionCaptureSampling::DatasetMode::INTRINSIC;
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
  void ProcessCameraCalibration(const cv::Mat& image, const FrameGeometry& geometry,
                                uint64_t image_ts, bool sample_accepted,
                                bool record_completed)
  {
    if (!ShouldRunCameraCalibration() ||
        !VisionCaptureRecording::CalibrationViewMayCommit(
            sample_accepted, ShouldSaveFrames(), record_completed, record_io_failed_))
    {
      return;
    }
    camera_calibration_.ProcessFrame(image.data, geometry, image_ts);
    if (cfg_.camera_calibration.auto_save_views != 0 &&
        camera_calibration_.SaveAndStopIfReady(cfg_.camera_calibration.auto_save_views))
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
    /// 当前相机模型将标定板角点重投影回本帧后的像素坐标。
    std::vector<cv::Point2f> projected_frame_points{};
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
   * @brief 异步预览线程绘制一帧所需的不可变状态快照。
   */
  struct PreviewSnapshot
  {
    VisionCaptureSampling::DatasetMode mode{VisionCaptureSampling::DatasetMode::NONE};
    VisionCaptureSampling::FrameProfile profile{
        VisionCaptureSampling::FrameProfile::UNKNOWN};
    FrameGeometry geometry{};
    SamplingDecision sampling{};
    VisionCaptureSampling::VisualCoverage coverage{};
    uint64_t accepted_total{0};
    uint32_t accepted_target{0};
    std::size_t solver_views{0};
    uint32_t solver_target{0};
  };

  /**
   * @brief 检测当前图像中的标定板。
   */
  BoardObservation DetectBoard(const cv::Mat& image, const FrameGeometry& geometry)
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
      VisionCaptureCalibrationBoard::CollectBoardPoints(
          detection.marker_corners, detection.marker_ids, board, detection);
      detection.homography_rms = VisionCaptureCalibrationBoard::HomographyRms(
          detection.object_points, detection.image_points,
          &detection.homography_projected_points);
      VisionCaptureCalibrationBoard::FillQuality(image, geometry.width, geometry.height,
                                                 detection);
    }
    return detection;
  }

  /**
   * @brief 当前采样使用的 ArUco 字典。
   */
  const cv::aruco::Dictionary& SamplingDictionary() const
  {
    return VisionCaptureSampling::UsesGShangBoard(CurrentDatasetMode())
               ? camera_sampling_dictionary_
               : dictionary_;
  }

  /**
   * @brief 当前采样使用的标定板三维点，单位 m。
   */
  VisionCaptureCalibrationBoard::BoardMap SamplingBoard() const
  {
    if (!VisionCaptureSampling::UsesGShangBoard(CurrentDatasetMode()))
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
   * @brief 根据原生相机标定构造 OpenCV 相机内参矩阵。
   */
  cv::Mat CameraMatrix() const
  {
    return (cv::Mat_<double>(3, 3) << calibration_.camera_matrix[0],
            calibration_.camera_matrix[1], calibration_.camera_matrix[2],
            calibration_.camera_matrix[3], calibration_.camera_matrix[4],
            calibration_.camera_matrix[5], calibration_.camera_matrix[6],
            calibration_.camera_matrix[7], calibration_.camera_matrix[8]);
  }

  /**
   * @brief 根据原生相机标定构造 OpenCV PnP 畸变系数矩阵。
   */
  cv::Mat DistortionCoefficients() const
  {
    const auto coeffs = CameraTypes::BuildPnPDistCoeffs(calibration_);
    if (coeffs.size == 0)
    {
      return {};
    }
    cv::Mat distortion(1, static_cast<int>(coeffs.size), CV_64F);
    for (int i = 0; i < static_cast<int>(coeffs.size); ++i)
    {
      distortion.at<double>(0, i) = coeffs.values[static_cast<std::size_t>(i)];
    }
    return distortion;
  }

  /**
   * @brief 将原生坐标重投影误差换回当前帧像素，保持采样阈值语义。
   */
  static double FrameReprojectionRms(const BoardObservation& detection,
                                     const std::vector<cv::Point2f>& projected_native,
                                     const FrameGeometry& geometry)
  {
    if (projected_native.size() != detection.image_points.size())
    {
      return std::numeric_limits<double>::infinity();
    }

    double sum2 = 0.0;
    for (std::size_t i = 0; i < projected_native.size(); ++i)
    {
      const auto frame_point =
          CameraTypes::NativeToFrame(geometry, static_cast<double>(projected_native[i].x),
                                     static_cast<double>(projected_native[i].y));
      const double dx = frame_point[0] - detection.image_points[i].x;
      const double dy = frame_point[1] - detection.image_points[i].y;
      sum2 += dx * dx + dy * dy;
    }
    return std::sqrt(sum2 / std::max<std::size_t>(1, projected_native.size()));
  }

  /**
   * @brief 对当前标定板观测执行 PnP 并写入采样判定。
   */
  bool SolveMarkerPnp(const BoardObservation& detection, const FrameGeometry& geometry,
                      SamplingDecision& decision) const
  {
    if (!detection.observed)
    {
      decision.reason = "board_not_detected";
      return false;
    }
    if (CameraTypes::BuildPnPDistCoeffs(calibration_).requires_undistort_first)
    {
      decision.reason = "distortion_model_unsupported";
      return false;
    }
    try
    {
      BoardObservation native_detection = detection;
      for (auto& point : native_detection.image_points)
      {
        const auto native = CameraTypes::FrameToNative(
            geometry, static_cast<double>(point.x), static_cast<double>(point.y));
        point.x = static_cast<float>(native[0]);
        point.y = static_cast<float>(native[1]);
      }
      const int method = detection.object_points.size() == 4 ? cv::SOLVEPNP_IPPE_SQUARE
                                                             : cv::SOLVEPNP_ITERATIVE;
      const cv::Mat camera_matrix = CameraMatrix();
      const cv::Mat distortion = DistortionCoefficients();
      const auto pose = VisionCaptureCalibrationBoard::EstimatePose(
          native_detection, camera_matrix, distortion, method);
      if (!pose.ok)
      {
        decision.reason = "pnp_failed";
        return false;
      }
      std::vector<cv::Point2f> projected_native;
      cv::projectPoints(detection.object_points, pose.rvec, pose.tvec, camera_matrix,
                        distortion, projected_native);
      decision.pnp_rms_px = FrameReprojectionRms(detection, projected_native, geometry);
      decision.projected_frame_points.reserve(projected_native.size());
      for (const cv::Point2f& point : projected_native)
      {
        const auto frame_point = CameraTypes::NativeToFrame(
            geometry, static_cast<double>(point.x), static_cast<double>(point.y));
        decision.projected_frame_points.emplace_back(static_cast<float>(frame_point[0]),
                                                     static_cast<float>(frame_point[1]));
      }
      decision.rvec = pose.rvec;
      decision.tvec = pose.tvec;
    }
    catch (const cv::Exception& e)
    {
      decision.reason = "pnp_exception";
      return false;
    }
    if (!std::isfinite(decision.pnp_rms_px) ||
        decision.pnp_rms_px > cfg_.calibration_sampling.max_pnp_reprojection_rms_px)
    {
      decision.reason = "pnp_rms";
      return false;
    }
    decision.pnp_ok = true;
    return true;
  }

  static VisionCaptureSampling::VisualSample MakeVisualSample(
      const BoardObservation& detection, uint64_t image_timestamp_us)
  {
    return {
        .observed = detection.observed,
        .image_timestamp_us = image_timestamp_us,
        .used_markers = detection.used_markers,
        .homography_rms = detection.homography_rms,
        .sharpness_score = detection.sharpness_score,
        .center_x_norm = detection.center_x_norm,
        .center_y_norm = detection.center_y_norm,
        .scale_norm = detection.scale_norm,
        .angle_deg = detection.angle_deg,
    };
  }

  VisionCaptureSampling::IntrinsicLimits IntrinsicSamplingLimits() const
  {
    VisionCaptureSampling::IntrinsicLimits limits;
    limits.minimum_markers = VisionCaptureSampling::RequiredGShangMarkerCount(
        cfg_.camera_calibration.cols, cfg_.camera_calibration.rows);
    limits.minimum_interval_us = cfg_.calibration_sampling.min_accept_interval_us;
    return limits;
  }

  SamplingDecision RejectSampling(SamplingDecision decision, std::string_view reason)
  {
    decision.reason = VisionCaptureDetail::ToString(reason);
    sampling_rejected_.fetch_add(1);
    SetLastSamplingStatus(decision);
    return decision;
  }

  SamplingDecision AcceptSampling(SamplingDecision decision, std::string_view reason)
  {
    decision.accepted = true;
    decision.reason = VisionCaptureDetail::ToString(reason);
    sampling_accepted_.fetch_add(1);
    sampling_accepted_total_.fetch_add(1, std::memory_order_acq_rel);
    SetLastSamplingStatus(decision);
    return decision;
  }

  /**
   * @brief 仅用 GShang 视觉观测和相机时间戳筛选内参样本。
   */
  SamplingDecision EvaluateIntrinsicSampling(const BoardObservation& detection,
                                             uint64_t image_timestamp_us)
  {
    SamplingDecision decision;
    decision.projected_frame_points = detection.homography_projected_points;
    const auto sample = MakeVisualSample(detection, image_timestamp_us);
    const auto limits = IntrinsicSamplingLimits();
    VisionCaptureSampling::AdmissionResult admission;
    bool force_snapshot = false;
    {
      std::lock_guard<std::mutex> lock(sampling_mutex_);
      const uint64_t request_generation =
          snapshot_request_generation_.load(std::memory_order_acquire);
      force_snapshot = VisionCaptureSampling::SnapshotPending(
          request_generation, snapshot_consumed_generation_);
      if (sample.observed && sample.used_markers >= limits.minimum_markers &&
          VisionCaptureSampling::VisualMetricsFinite(sample) &&
          sample.homography_rms <= limits.max_homography_rms)
      {
        best_intrinsic_sharpness_score_ =
            std::max(best_intrinsic_sharpness_score_, sample.sharpness_score);
      }
      admission = VisionCaptureSampling::EvaluateIntrinsicObservation(
          sample,
          std::span<const VisionCaptureSampling::VisualSample>(accepted_visual_samples_),
          last_intrinsic_accept_timestamp_us_, best_intrinsic_sharpness_score_, limits,
          force_snapshot);
      if (admission.accepted)
      {
        accepted_visual_samples_.push_back(sample);
        last_intrinsic_accept_timestamp_us_ = sample.image_timestamp_us;
        if (force_snapshot)
        {
          snapshot_consumed_generation_ =
              VisionCaptureSampling::ConsumeSnapshotGeneration(
                  request_generation, snapshot_consumed_generation_, true);
        }
      }
    }
    if (!admission.accepted)
    {
      return RejectSampling(std::move(decision), admission.reason);
    }
    return AcceptSampling(std::move(decision), admission.reason);
  }

  /**
   * @brief 用冻结 K/D 的 PnP 和同步 IMU 稳定性筛选手眼数据。
   */
  SamplingDecision EvaluateHandEyeSampling(const BoardObservation& detection,
                                           const FrameGeometry& geometry,
                                           const ImuStamped& imu,
                                           uint64_t image_timestamp_us)
  {
    SamplingDecision decision;
    const uint64_t imu_timestamp_us = static_cast<uint64_t>(imu.timestamp_us);
    if (image_timestamp_us == 0U)
    {
      return RejectSampling(std::move(decision), "image_timestamp_invalid");
    }
    if (!VisionCaptureSampling::ImuSampleUsable(imu_timestamp_us, imu.rotation_wxyz,
                                                imu.angular_velocity_xyz,
                                                imu.linear_acceleration_xyz))
    {
      if (imu_timestamp_us == 0U)
      {
        return RejectSampling(std::move(decision), "imu_timestamp_invalid");
      }
      if (!VisionCaptureSampling::AllFinite(imu.rotation_wxyz))
      {
        return RejectSampling(std::move(decision), "imu_quaternion_nonfinite");
      }
      if (!VisionCaptureSampling::QuaternionUsable(imu.rotation_wxyz))
      {
        return RejectSampling(std::move(decision), "imu_quaternion_invalid");
      }
      if (!VisionCaptureSampling::AllFinite(imu.angular_velocity_xyz))
      {
        return RejectSampling(std::move(decision), "imu_gyro_nonfinite");
      }
      return RejectSampling(std::move(decision), "imu_acceleration_nonfinite");
    }
    if (!SolveMarkerPnp(detection, geometry, decision))
    {
      const std::string reason = decision.reason;
      return RejectSampling(std::move(decision), reason);
    }
    sampling_pnp_ok_.fetch_add(1);

    StableSample sample;
    sample.timestamp_us = imu_timestamp_us;
    sample.rvec = decision.rvec.clone();
    sample.tvec = decision.tvec.clone();
    sample.quat = VisionCaptureDetail::NormalizeQuatWxyz(imu.rotation_wxyz);
    sample.gyro = VisionCaptureDetail::ToVec3d(imu.angular_velocity_xyz);
    sample.acc = VisionCaptureDetail::ToVec3d(imu.linear_acceleration_xyz);
    decision.gyro_norm_dps = VisionCaptureDetail::RadToDeg(cv::norm(sample.gyro));
    decision.acc_norm_mps2 =
        VisionCaptureSampling::AccelerationNormMps2(imu.linear_acceleration_xyz);
    decision.acc_norm_error_mps2 =
        std::fabs(decision.acc_norm_mps2 - VisionCaptureSampling::kStandardGravityMps2);

    const auto absolute_imu = VisionCaptureSampling::EvaluateHandEyeAbsoluteImu(
        decision.gyro_norm_dps, decision.acc_norm_mps2,
        cfg_.calibration_sampling.max_gyro_norm_dps,
        cfg_.calibration_sampling.max_acc_norm_error_mps2);
    if (!absolute_imu.accepted)
    {
      return RejectSampling(std::move(decision), absolute_imu.reason);
    }

    if (cfg_.calibration_sampling.window_size == 0U)
    {
      return RejectSampling(std::move(decision), "stability_window_invalid");
    }
    {
      std::lock_guard<std::mutex> lock(sampling_mutex_);
      stability_window_.push_back(sample);
      while (stability_window_.size() > cfg_.calibration_sampling.window_size)
      {
        stability_window_.pop_front();
      }
      if (stability_window_.size() < cfg_.calibration_sampling.window_size)
      {
        return RejectSampling(std::move(decision), "stability_window");
      }
      ComputeWindowStabilityLocked(sample, decision);
    }
    if (decision.pnp_t_jitter_m > cfg_.calibration_sampling.max_pnp_translation_jitter_m)
    {
      return RejectSampling(std::move(decision), "pnp_translation_unstable");
    }
    if (decision.pnp_r_jitter_deg > cfg_.calibration_sampling.max_pnp_rotation_jitter_deg)
    {
      return RejectSampling(std::move(decision), "pnp_rotation_unstable");
    }
    if (decision.imu_r_jitter_deg > cfg_.calibration_sampling.max_imu_rotation_jitter_deg)
    {
      return RejectSampling(std::move(decision), "imu_rotation_unstable");
    }
    if (decision.acc_norm_jitter_mps2 >
        cfg_.calibration_sampling.max_acc_norm_jitter_mps2)
    {
      return RejectSampling(std::move(decision), "acc_vibration");
    }
    if (decision.acc_dir_jitter_deg >
        cfg_.calibration_sampling.max_acc_direction_jitter_deg)
    {
      return RejectSampling(std::move(decision), "acc_direction_unstable");
    }
    bool force_snapshot = false;
    {
      std::lock_guard<std::mutex> lock(sampling_mutex_);
      const uint64_t request_generation =
          snapshot_request_generation_.load(std::memory_order_acquire);
      force_snapshot = VisionCaptureSampling::SnapshotPending(
          request_generation, snapshot_consumed_generation_);
      if (last_accepted_sample_.timestamp_us != 0U &&
          sample.timestamp_us <= last_accepted_sample_.timestamp_us)
      {
        return RejectSampling(std::move(decision), "imu_timestamp_non_monotonic");
      }
      if (!force_snapshot && last_accepted_sample_.timestamp_us != 0U &&
          sample.timestamp_us - last_accepted_sample_.timestamp_us <
              cfg_.calibration_sampling.min_accept_interval_us)
      {
        return RejectSampling(std::move(decision), "accept_interval");
      }
      if (!force_snapshot && IsDuplicateCalibrationSampleLocked(sample))
      {
        return RejectSampling(std::move(decision), "duplicate_pose");
      }
      accepted_calibration_samples_.push_back(sample);
      accepted_visual_samples_.push_back(MakeVisualSample(detection, image_timestamp_us));
      last_accepted_sample_ = sample;
      if (force_snapshot)
      {
        snapshot_consumed_generation_ = VisionCaptureSampling::ConsumeSnapshotGeneration(
            request_generation, snapshot_consumed_generation_, true);
      }
    }
    return AcceptSampling(std::move(decision), force_snapshot ? "snapshot" : "accepted");
  }

  /**
   * @brief 按运行模式选择内参视觉采样或手眼 PnP/IMU 判稳。
   */
  SamplingDecision EvaluateCalibrationSampling(const BoardObservation& detection,
                                               const FrameGeometry& geometry,
                                               const ImuStamped& imu,
                                               uint64_t image_timestamp_us)
  {
    SamplingDecision decision;
    const auto mode = CurrentDatasetMode();
    const auto control = VisionCaptureSampling::EvaluateSamplingControl(
        mode, cfg_.calibration_sampling.enabled,
        sampling_running_.load(std::memory_order_acquire));
    if (mode == VisionCaptureSampling::DatasetMode::NONE)
    {
      decision.accepted = true;
      decision.reason = control.reason;
      SetLastSamplingStatus(decision);
      return decision;
    }
    if (!control.accepted)
    {
      return RejectSampling(std::move(decision), control.reason);
    }
    if (mode == VisionCaptureSampling::DatasetMode::INTRINSIC)
    {
      return EvaluateIntrinsicSampling(detection, image_timestamp_us);
    }
    return EvaluateHandEyeSampling(detection, geometry, imu, image_timestamp_us);
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
      translation_max = std::max(translation_max, cv::norm(sample.tvec - reference.tvec));
      rotation_max = std::max(rotation_max, VisionCaptureDetail::RotationDistanceDeg(
                                                sample.rvec, reference.rvec));
      imu_rotation_max = std::max(
          imu_rotation_max,
          VisionCaptureDetail::QuatAngularDistanceDeg(sample.quat, reference.quat));
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
      acc_dir_max =
          std::max(acc_dir_max,
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
      if (translation_delta < cfg_.calibration_sampling.min_sample_translation_delta_m &&
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

  PreviewSnapshot BuildPreviewSnapshot(const SamplingDecision& sampling,
                                       const FrameGeometry& geometry) const
  {
    PreviewSnapshot snapshot;
    snapshot.mode = CurrentDatasetMode();
    snapshot.profile = VisionCaptureSampling::ClassifyFrameProfile(
        calibration_.native_width, calibration_.native_height, geometry);
    snapshot.geometry = geometry;
    snapshot.sampling = sampling;
    snapshot.accepted_total = sampling_accepted_total_.load(std::memory_order_acquire);
    snapshot.accepted_target = cfg_.camera_calibration.auto_save_views;
    if (snapshot.mode == VisionCaptureSampling::DatasetMode::INTRINSIC)
    {
      snapshot.solver_views = camera_calibration_.AcceptedViewCount();
      snapshot.solver_target = cfg_.camera_calibration.auto_save_views;
    }
    {
      std::lock_guard<std::mutex> lock(sampling_mutex_);
      snapshot.coverage = VisionCaptureSampling::ComputeVisualCoverage(
          std::span<const VisionCaptureSampling::VisualSample>(accepted_visual_samples_));
    }
    return snapshot;
  }

  static void DrawPreviewOverlay(cv::Mat& frame, const BoardObservation& detection,
                                 const PreviewSnapshot& snapshot)
  {
    if (detection.observed && !detection.marker_ids.empty())
    {
      cv::aruco::drawDetectedMarkers(frame, detection.marker_corners,
                                     detection.marker_ids);
    }
    for (const cv::Point2f& point : detection.image_points)
    {
      cv::circle(frame, point, 3, {0, 255, 255}, cv::FILLED, cv::LINE_AA);
    }
    for (const cv::Point2f& point : snapshot.sampling.projected_frame_points)
    {
      cv::drawMarker(frame, point, {255, 0, 255}, cv::MARKER_CROSS, 10, 2, cv::LINE_AA);
    }

    const cv::Scalar status_color =
        snapshot.sampling.accepted ? cv::Scalar{0, 220, 0} : cv::Scalar{0, 0, 255};
    int y = 26;
    auto draw_line = [&](const std::string& text, const cv::Scalar& color)
    {
      VisionCaptureDetail::DrawOutlinedText(frame, text, {12, y}, color, 0.55);
      y += 24;
    };

    std::ostringstream line;
    line << VisionCaptureSampling::DatasetModeName(snapshot.mode) << " "
         << (snapshot.sampling.accepted ? "ACCEPT" : "REJECT") << " "
         << snapshot.sampling.reason;
    draw_line(line.str(), status_color);

    line.str("");
    line.clear();
    line << std::fixed << std::setprecision(3) << "VIS markers=" << detection.used_markers
         << " H=" << detection.homography_rms << " S=" << std::setprecision(1)
         << detection.sharpness_score << " center=(" << std::setprecision(3)
         << detection.center_x_norm << "," << detection.center_y_norm << ")";
    draw_line(line.str(), {0, 255, 255});

    line.str("");
    line.clear();
    line << std::fixed << std::setprecision(3) << "VIS scale=" << detection.scale_norm
         << " angle=" << std::setprecision(1) << detection.angle_deg << " cover=("
         << std::setprecision(3) << snapshot.coverage.center_span_x << ","
         << snapshot.coverage.center_span_y
         << ") ratio=" << snapshot.coverage.scale_ratio;
    draw_line(line.str(), {0, 255, 255});

    line.str("");
    line.clear();
    line << "SAMPLES " << snapshot.accepted_total << "/" << snapshot.accepted_target
         << " SOLVER " << snapshot.solver_views << "/" << snapshot.solver_target;
    draw_line(line.str(), {255, 255, 255});

    line.str("");
    line.clear();
    line << "PROFILE=" << VisionCaptureSampling::FrameProfileName(snapshot.profile)
         << " FRAME=" << snapshot.geometry.width << "x" << snapshot.geometry.height
         << " STEP=" << snapshot.geometry.step;
    draw_line(line.str(), {255, 255, 255});

    line.str("");
    line.clear();
    line << std::fixed << std::setprecision(2) << "ROI=("
         << snapshot.geometry.roi_offset_x_native << ","
         << snapshot.geometry.roi_offset_y_native << ") DEC=("
         << snapshot.geometry.decimation_x << "," << snapshot.geometry.decimation_y
         << ") PHASE=(" << snapshot.geometry.sample_phase_x_native << ","
         << snapshot.geometry.sample_phase_y_native
         << ") FLAGS=" << snapshot.geometry.flags;
    draw_line(line.str(), {255, 255, 255});

    if (snapshot.mode == VisionCaptureSampling::DatasetMode::HAND_EYE)
    {
      line.str("");
      line.clear();
      line << std::fixed << std::setprecision(3)
           << "PNP rms=" << snapshot.sampling.pnp_rms_px
           << " t_jitter=" << snapshot.sampling.pnp_t_jitter_m
           << "m r_jitter=" << snapshot.sampling.pnp_r_jitter_deg << "deg";
      draw_line(line.str(), {255, 0, 255});

      line.str("");
      line.clear();
      line << std::fixed << std::setprecision(3)
           << "IMU r_jitter=" << snapshot.sampling.imu_r_jitter_deg
           << "deg gyro=" << snapshot.sampling.gyro_norm_dps
           << "deg/s acc=" << snapshot.sampling.acc_norm_mps2 << "m/s2";
      draw_line(line.str(), {255, 0, 255});
    }
  }

  /**
   * @brief 非阻塞提交检测、重投影、判定、进度和逐帧几何预览。
   */
  void SubmitPreview(const cv::Mat& image, const BoardObservation& detection,
                     const SamplingDecision& sampling, const FrameGeometry& geometry)
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
    const PreviewSnapshot snapshot = BuildPreviewSnapshot(sampling, geometry);
    preview_.Submit(preview_image, [detection, snapshot](cv::Mat& frame)
                    { DrawPreviewOverlay(frame, detection, snapshot); });
  }

  /**
   * @brief 按帧号保存图像并返回文件路径。
   *
   * @return true 表示未要求写图或图像已经成功写盘。
   */
  bool SaveImage(const cv::Mat& image, uint64_t frame_id, std::string& image_path)
  {
    image_path.clear();
    if (!cfg_.record.save_images)
    {
      return true;
    }
    std::ostringstream name;
    name << std::setw(8) << std::setfill('0') << frame_id << "."
         << cfg_.record.image_format;
    const std::filesystem::path path = frames_dir_ / name.str();
    try
    {
      if (!cv::imwrite(path.string(), image))
      {
        XR_LOG_ERROR("VisionCapture image write failed: %s", path.string().c_str());
        return false;
      }
    }
    catch (const cv::Exception& e)
    {
      XR_LOG_ERROR("VisionCapture image write failed %s: %s", path.string().c_str(),
                   e.what());
      return false;
    }
    image_path = path.string();
    return true;
  }

  /**
   * @brief 写入一帧完整采样几何。
   */
  bool WriteFrameGeometry(uint64_t frame_id, const std::string& image_path,
                          const FrameGeometry& geometry)
  {
    if (!frame_geometry_csv_.is_open())
    {
      record_io_failed_ = true;
      XR_LOG_ERROR("VisionCapture frame_geometry.csv is not open");
      return false;
    }
    VisionCaptureRecording::WriteFrameGeometryRow(frame_geometry_csv_, frame_id,
                                                  image_path, geometry);
    if (!frame_geometry_csv_)
    {
      record_io_failed_ = true;
      XR_LOG_ERROR("VisionCapture frame_geometry.csv write failed");
      return false;
    }
    return true;
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
  bool WriteMetadata(uint64_t frame_id, uint64_t image_ts, uint64_t imu_ts,
                     const ImuStamped& imu, const std::string& image_path,
                     const BoardObservation& detection, const SamplingDecision& sampling)
  {
    if (!cfg_.record.save_metadata)
    {
      return true;
    }
    if (!metadata_csv_.is_open())
    {
      record_io_failed_ = true;
      XR_LOG_ERROR("VisionCapture samples.csv is not open");
      return false;
    }
    // 相机和 MCU 时间戳属于不同时间域；保留兼容 dt_us 列但永远不相减。
    metadata_csv_ << frame_id << "," << image_ts << "," << imu_ts << ",";
    VisionCaptureRecording::WriteRawImuCells(metadata_csv_, cfg_.record.save_raw_imu,
                                             imu.rotation_wxyz, imu.angular_velocity_xyz,
                                             imu.linear_acceleration_xyz);
    metadata_csv_ << ",";
    VisionCaptureRecording::WriteCsvCell(metadata_csv_, image_path);
    metadata_csv_ << "," << (detection.observed ? 1 : 0) << ","
                  << detection.marker_ids_vec.size() << ","
                  << JoinIds(detection.marker_ids_vec) << ","
                  << (sampling.accepted ? 1 : 0) << "," << sampling.reason << ","
                  << (sampling.pnp_ok ? 1 : 0) << "," << sampling.pnp_rms_px << ","
                  << sampling.pnp_t_jitter_m << "," << sampling.pnp_r_jitter_deg << ","
                  << sampling.imu_r_jitter_deg << "," << sampling.gyro_norm_dps << ","
                  << sampling.acc_norm_mps2 << "," << sampling.acc_norm_error_mps2 << ","
                  << sampling.acc_norm_jitter_mps2 << "," << sampling.acc_dir_jitter_deg
                  << ",mps2\n";
    if (!metadata_csv_)
    {
      record_io_failed_ = true;
      XR_LOG_ERROR("VisionCapture samples.csv write failed");
      return false;
    }
    return true;
  }

  /**
   * @brief 按 flush_every_n 同步刷新本次记录的两个 CSV。
   */
  bool FlushCsvIfNeeded()
  {
    ++csv_rows_written_;
    if (!VisionCaptureRecording::ShouldFlush(csv_rows_written_,
                                             cfg_.record.flush_every_n))
    {
      return true;
    }
    if (metadata_csv_.is_open())
    {
      metadata_csv_.flush();
      if (!metadata_csv_)
      {
        record_io_failed_ = true;
        XR_LOG_ERROR("VisionCapture samples.csv flush failed");
      }
    }
    if (frame_geometry_csv_.is_open())
    {
      frame_geometry_csv_.flush();
      if (!frame_geometry_csv_)
      {
        record_io_failed_ = true;
        XR_LOG_ERROR("VisionCapture frame_geometry.csv flush failed");
      }
    }
    return !record_io_failed_;
  }

 private:
  /// 模块运行配置。
  Config cfg_{};
  /// 构造时从 CameraFrameSync 复制的原生相机标定。
  const CameraCalibration calibration_;
  /// 可选预览输出。
  VisionPreview preview_{};
  /// 同步帧进程内 Topic。
  LibXR::Topic synced_frame_topic_ = LibXR::Topic();
  /// 同步帧借用回调。
  LibXR::Topic::Callback synced_frame_callback_{};
  /// 回调 retain 后写入的双槽 drop-oldest worker 队列。
  VisionCaptureDetail::DropOldestWorkerQueue<SyncedFrame, 2> frame_queue_{};

  /// OpenCV ArUco 字典。
  cv::aruco::Dictionary dictionary_{};
  /// 相机内参标定采样使用的 ArUco original 字典。
  cv::aruco::Dictionary camera_sampling_dictionary_{
      cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_ORIGINAL)};
  /// OpenCV ArUco 检测参数。
  cv::aruco::DetectorParameters detector_params_{};
  /// 相机内参标定流程对象。
  VisionCaptureCameraCalibration<FrameLayoutV> camera_calibration_;
  /// 串行化单帧记录与 reset/solve，防止求解观察到半写会话。
  std::mutex operation_mutex_{};

  /// 标定采样是否正在运行。
  std::atomic<bool> sampling_running_{true};
  /// 每条 snapshot 命令递增一次；worker 只在成功接受样本后消费对应代次。
  std::atomic<uint64_t> snapshot_request_generation_{0};
  /// 保护稳定窗口和已接受样本。
  mutable std::mutex sampling_mutex_{};
  /// 最近若干帧稳定性计算窗口。
  std::deque<StableSample> stability_window_{};
  /// 已保存的判稳样本。
  std::vector<StableSample> accepted_calibration_samples_{};
  /// 已接受样本的视觉观测，用于内参去重和预览覆盖统计。
  std::vector<VisionCaptureSampling::VisualSample> accepted_visual_samples_{};
  /// 最近一次通过判稳的样本。
  StableSample last_accepted_sample_{};
  /// 最近一次接受内参视觉样本的相机时间戳。
  uint64_t last_intrinsic_accept_timestamp_us_{0};
  /// 本轮内参视觉样本中已见到的最佳清晰度。
  double best_intrinsic_sharpness_score_{0.0};
  /// 已由成功样本消费的最新 snapshot 命令代次，由 sampling_mutex_ 保护。
  uint64_t snapshot_consumed_generation_{0};
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
  /// 每个成功记录帧的完整采样几何 CSV。
  std::ofstream frame_geometry_csv_{};

  /// 已保存帧总数。
  uint64_t total_saved_frames_{0};
  /// 已成功写入记录 CSV 的行数。
  uint64_t csv_rows_written_{0};
  /// 记录 I/O 失败锁存；置位后停止分配帧号，避免复用半写记录的 ID。
  bool record_io_failed_{false};
  /// 最近一次保存图像的时间戳，单位 us。
  uint64_t last_saved_timestamp_us_{0};
  /// 是否已经打印过不支持图像编码错误。
  bool unsupported_encoding_logged_{false};
  /// 是否已经打印过非法 geometry 错误。
  bool invalid_geometry_logged_{false};
  /// 首帧 geometry 和兼容 camera_info 快照是否已经写出。
  bool frame_geometry_snapshot_written_{false};

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
  /// 本轮从 reset 开始累计接受的标定样本数，不随 monitor 清零。
  std::atomic<uint64_t> sampling_accepted_total_{0};
  /// monitor 周期内拒绝的标定样本数。
  std::atomic<uint64_t> sampling_rejected_{0};
  /// 可取消并由析构等待的标准输入读取器；最后声明以便构造失败时优先析构。
  VisionCaptureDetail::StoppableLineInput control_input_{};
};
