#pragma once

// clang-format off
/* === MODULE MANIFEST V2 ===
module_description: 录像：把同步帧写成统一录像格式（逐帧 PGM 与 frames.csv），供回放与离线标定 / Recorder that writes synced frames in the unified recording format (per-frame PGM and frames.csv) for replay and offline calibration
depends:
- id: QDU-Robomaster/AutoAimTypes
  ref: same-or-dev
=== END MANIFEST === */
// clang-format on

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <deque>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "AutoAimTypes.hpp"
#include "logger.hpp"
#include "message.hpp"

/// 录像设置 / Recorder settings.
struct RecorderSettings
{
  const char* camera_name;  ///< 相机名，订阅 `<相机名>_synced` / Camera name
  bool record;              ///< false 时模块什么也不做 / The Module is idle when false
  const char* output_dir;   ///< 每次启动在其下新建一个以时间命名的目录 / Parent directory
  double max_fps;           ///< 帧率上限，0 为不限 / Frame rate cap, 0 for none
  uint32_t max_frames;      ///< 最多写多少帧，0 为不限 / Frame limit, 0 for none
  uint32_t buffer_frames;   ///< 写盘前的缓冲帧数 / Frames buffered before writing
};

/**
 * @brief 录像。订阅 `<相机名>_synced`，在回调里把图像字节和 IMU 复制进自有的环形缓冲，
 *        写盘线程写出 `<frame:06>.pgm` 与 `frames.csv`；缓冲满时丢帧并计数。
 *        Recorder. Subscribes to `<camera>_synced`; the callback copies the image bytes
 *        and IMU into its own ring buffer, and a writer thread writes `<frame:06>.pgm`
 *        and `frames.csv`. Frames are dropped and counted when the buffer is full.
 *
 * 不持有图像句柄，不占相机的图像槽。输出可由 CaptureFileCamera 直接回放。
 * No image handle is kept, so no camera image slot is held. CaptureFileCamera replays the
 * output directly.
 */
class VisionRecorder
{
 public:
  explicit VisionRecorder(const RecorderSettings& settings)
      : camera_name_(settings.camera_name),
        max_frames_(settings.max_frames),
        min_interval_us_(settings.max_fps > 0 ? 1e6 / settings.max_fps : 0.0)
  {
    if (!settings.record)
    {
      return;
    }
    dir_ = std::filesystem::path(settings.output_dir) / SessionName();
    std::error_code ec;
    std::filesystem::create_directories(dir_, ec);
    csv_ = ec ? nullptr : std::fopen((dir_ / "frames.csv").string().c_str(), "w");
    if (csv_ == nullptr)
    {
      XR_LOG_ERROR("%s recorder: cannot create %s", camera_name_.c_str(),
                   dir_.string().c_str());
      return;
    }
    std::fputs(
        "frame,timestamp_us,frame_counter,roi_x,roi_y,decimation,"
        "qw,qx,qy,qz,gx,gy,gz,ax,ay,az\n",
        csv_);
    slots_.resize(std::max<uint32_t>(settings.buffer_frames, 1));
    for (std::size_t i = 0; i < slots_.size(); ++i)
    {
      free_.push_back(i);
    }
    running_.store(true);
    writer_ = std::thread([this]() { WriterLoop(); });
    auto callback = LibXR::Topic::Callback::Create(
        [](bool, VisionRecorder* self, const AutoAim::SyncedFrame* frame)
        { self->OnSynced(*frame); }, this);
    AutoAim::RequireTopic<const AutoAim::SyncedFrame*>(
        StageTopicName(camera_name_, AutoAim::STAGE_SYNCED))
        .RegisterCallback(callback);
    XR_LOG_INFO("%s recorder: writing to %s", camera_name_.c_str(),
                dir_.string().c_str());
  }

  ~VisionRecorder()
  {
    if (!writer_.joinable())
    {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      running_.store(false);
    }
    cv_.notify_all();
    writer_.join();
    std::fclose(csv_);
  }

  VisionRecorder(const VisionRecorder&) = delete;
  VisionRecorder& operator=(const VisionRecorder&) = delete;

  /// 本次录像的目录，未录像时为空 / Directory of this recording, empty when idle.
  const std::filesystem::path& Directory() const { return dir_; }

  /// 打印周期摘要 / Print the periodic summary.
  void OnMonitor()
  {
    if (writer_.joinable())
    {
      XR_LOG_INFO("%s recorder: written=%u dropped=%u", camera_name_.c_str(),
                  written_.load(), dropped_.exchange(0));
    }
  }

 private:
  struct Slot
  {
    uint64_t frame;
    AutoAim::ImuSample imu;
    uint32_t frame_counter;
    uint64_t timestamp_us;
    CameraTypes::FrameGeometry geometry;
    const CameraTypes::CameraCalibration* calibration;
    std::array<uint8_t, CameraTypes::FRAME_BYTES> data;
  };

  static std::string SessionName()
  {
    const std::time_t now = std::time(nullptr);
    std::tm local{};
    localtime_r(&now, &local);
    char name[32];
    std::strftime(name, sizeof(name), "%Y%m%d-%H%M%S", &local);
    return name;
  }

  void OnSynced(const AutoAim::SyncedFrame& frame)
  {
    const ImageFrame& image = *frame.image;
    const auto t_us = static_cast<uint64_t>(image.timestamp_us);
    std::unique_lock<std::mutex> lock(mutex_);
    if (!running_.load() || (max_frames_ > 0 && next_frame_ >= max_frames_))
    {
      return;
    }
    if (has_last_ && static_cast<double>(t_us - last_us_) < min_interval_us_ - 500.0)
    {
      return;  // 帧率上限，允许 0.5 ms 抖动 / Rate cap with 0.5 ms tolerance
    }
    if (free_.empty())
    {
      dropped_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    const std::size_t index = free_.front();
    free_.pop_front();
    has_last_ = true;
    last_us_ = t_us;
    Slot& slot = slots_[index];
    slot.frame = next_frame_++;
    lock.unlock();
    // 槽已从空闲表取出，写盘线程拿到它之前只有这里访问 / The slot left the free list,
    // so only this thread touches it until it is queued.
    slot.imu = frame.imu;
    slot.frame_counter = image.frame_counter;
    slot.timestamp_us = t_us;
    slot.geometry = image.geometry;
    slot.calibration = image.calibration;
    slot.data = image.data;
    lock.lock();
    queued_.push_back(index);
    lock.unlock();
    cv_.notify_one();
  }

  void WriterLoop()
  {
    while (true)
    {
      std::size_t index = 0;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() { return !queued_.empty() || !running_.load(); });
        if (queued_.empty())
        {
          return;  // 停止且已写完 / Stopped and drained
        }
        index = queued_.front();
        queued_.pop_front();
      }
      Write(slots_[index]);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        free_.push_back(index);
      }
    }
  }

  /// 相机名与第一帧携带的标定；回放不读 / Camera name and the calibration of the first
  /// frame; not read by replay.
  void WriteSession(const Slot& s)
  {
    std::FILE* f = std::fopen((dir_ / "session.txt").string().c_str(), "w");
    if (f == nullptr)
    {
      return;
    }
    std::fprintf(f, "camera %s\nrecorder VisionRecorder\n", camera_name_.c_str());
    if (s.calibration != nullptr)
    {
      const CameraTypes::CameraCalibration& c = *s.calibration;
      std::fprintf(f,
                   "native_width %u\nnative_height %u\nfx %.6f\nfy %.6f\ncx %.6f\n"
                   "cy %.6f\ndistortion %.8g %.8g %.8g %.8g %.8g\n",
                   c.native_width, c.native_height, c.fx, c.fy, c.cx, c.cy,
                   c.distortion[0], c.distortion[1], c.distortion[2], c.distortion[3],
                   c.distortion[4]);
    }
    std::fclose(f);
  }

  void Write(const Slot& s)
  {
    if (s.frame == 0)
    {
      WriteSession(s);
    }
    char name[32];
    std::snprintf(name, sizeof(name), "%06llu.pgm",
                  static_cast<unsigned long long>(s.frame));
    std::FILE* pgm = std::fopen((dir_ / name).string().c_str(), "wb");
    const bool ok = pgm != nullptr &&
                    std::fprintf(pgm, "P5\n%u %u\n255\n", CameraTypes::FRAME_WIDTH,
                                 CameraTypes::FRAME_HEIGHT) > 0 &&
                    std::fwrite(s.data.data(), 1, s.data.size(), pgm) == s.data.size();
    if (pgm != nullptr)
    {
      std::fclose(pgm);
    }
    if (!ok)
    {
      if (write_errors_++ == 0)
      {
        XR_LOG_ERROR("%s recorder: cannot write %s", camera_name_.c_str(), name);
      }
      return;
    }
    const auto& q = s.imu.rotation_wxyz;
    const auto& g = s.imu.angular_velocity_xyz;
    const auto& a = s.imu.linear_acceleration_xyz;
    std::fprintf(
        csv_, "%llu,%llu,%u,%u,%u,%u,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
        static_cast<unsigned long long>(s.frame),
        static_cast<unsigned long long>(s.timestamp_us), s.frame_counter,
        s.geometry.roi_x, s.geometry.roi_y, static_cast<unsigned>(s.geometry.decimation),
        q[0], q[1], q[2], q[3], g[0], g[1], g[2], a[0], a[1], a[2]);
    std::fflush(csv_);
    written_.fetch_add(1, std::memory_order_relaxed);
  }

  const std::string camera_name_;
  const uint32_t max_frames_;
  const double min_interval_us_;
  std::filesystem::path dir_;
  std::FILE* csv_ = nullptr;
  std::vector<Slot> slots_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<std::size_t> free_;
  std::deque<std::size_t> queued_;
  std::atomic<bool> running_{false};
  uint64_t next_frame_ = 0;
  bool has_last_ = false;
  uint64_t last_us_ = 0;
  std::atomic<uint32_t> written_{0};
  std::atomic<uint32_t> dropped_{0};
  uint32_t write_errors_ = 0;
  std::thread writer_;
};
