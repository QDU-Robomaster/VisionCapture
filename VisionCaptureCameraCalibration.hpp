#pragma once

#include "CameraBaseIntrinsicSanity.hpp"
#include "logger.hpp"

#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

/**
 * @brief VisionCapture 相机内参标定辅助器。
 *
 * VisionCapture 在 CameraFrameSync 后消费同步帧。本类保留旧 CameraBase 标定器的
 * GShang/ArUco 采样过滤、OpenCV 求解和质量报告输出逻辑。
 */
template <CameraTypes::CameraInfo CameraInfoV>
class VisionCaptureCameraCalibration
{
 public:
  /// GShang 生成器使用 ArUco original 字典中的 5-bit 有效码元。
  static constexpr int gshang_dictionary_bits = 5;
  /// 单个 marker 的总码元宽度：5-bit 有效区加一圈白边。
  static constexpr int gshang_marker_cells = gshang_dictionary_bits + 2;
  /// 单个棋盘方格的总码元宽度：marker 外再加棋盘留边。
  static constexpr int gshang_square_cells = gshang_dictionary_bits + 4;
  /// 保存阶段二次过滤时保留的 OpenCV 求解视角下限。
  static constexpr int solver_view_floor = 8;
  /// 经验建议视角数；达到后只提示，不自动停止采样。
  static constexpr int default_recommended_views = 120;
  /// 默认 marker 可见比例门槛；实际最少 marker 数按当前板型在 Start() 后派生。
  static constexpr double default_min_marker_ratio = 2.0 / 3.0;
  /// 极小标定板仍尽量要求至少 4 个 marker；若板上 marker 总数不足则取总数。
  static constexpr int minimum_required_markers = 4;
  /// 当前 CameraInfoV 的像素编码是否能被本标定器零拷贝封装为 cv::Mat。
  static constexpr bool supports_image_encoding =
      CameraInfoV.encoding == CameraTypes::Encoding::BGR8 ||
      CameraInfoV.encoding == CameraTypes::Encoding::RGB8 ||
      CameraInfoV.encoding == CameraTypes::Encoding::BGRA8 ||
      CameraInfoV.encoding == CameraTypes::Encoding::RGBA8 ||
      CameraInfoV.encoding == CameraTypes::Encoding::MONO8;

  /**
   * @brief 单次在线标定的参数快照。
   *
   * 这些参数在 Start() 时根据命令行和经验默认值生成，采样线程只读取快照；
   * 保存阶段也复制同一份配置，保证输出文件和采样判据一致。
   */
  struct Config
  {
    /// 标记黑码区域边长，单位 mm；对应命令 `cali 25mm 8 6` 中的 25mm。
    double marker_mm = 25.0;
    /// GShang 工具中的标定板列数，也就是棋盘方格列数。
    int cols = 8;
    /// GShang 工具中的标定板行数，也就是棋盘方格行数。
    int rows = 6;
    /// 单帧至少需要识别到的有效 marker 比例，实际 marker 数按当前板型派生。
    double min_marker_ratio = default_min_marker_ratio;
    /// 建议采满的视角数；达到后提示操作手可以保存。
    int recommended_views = default_recommended_views;
    /// 图像处理抽帧间隔；所有帧仍被吞掉，但只每 N 帧跑一次检测。
    int process_stride = 3;
    /// 两个已接受视角之间的最小时间间隔，避免一段静止画面刷满样本。
    uint64_t min_accept_interval_us = 200000;
    /// 单应性重投影 RMS 上限，用于早期剔除明显错检或严重畸变视角。
    double max_homography_rms = 2.5;
    /// 标定后单视角重投影 RMS 上限，用于保存前二次剔除离群视角。
    double max_reprojection_rms = 3.0;
    /// marker 区域拉普拉斯方差绝对下限，用于拒绝明显运动模糊。
    double min_sharpness_score = 30.0;
    /// 相对本轮最佳清晰度的下限比例，适配不同相机曝光和画面尺度。
    double min_sharpness_best_ratio = 0.20;
    /// 归一化图像中心变化阈值，小于该值可能是重复姿态。
    double min_center_delta_norm = 0.020;
    /// 标定板画面尺度对数变化阈值，小于该值可能是重复距离。
    double min_scale_delta_log = 0.05;
    /// 标定板平面内角度变化阈值，小于该值可能是重复朝向。
    double min_angle_delta_deg = 3.0;
  };

  /**
   * @brief 启动一轮在线 ChArUco 标定。
   *
   * @param marker_size_text marker 黑码边长，支持 `25mm` 或 `25`。
   * @param cols GShang 标定板列数。
   * @param rows GShang 标定板行数。
   * @param camera_name 用于输出目录命名的相机名。
   * @return true 表示配置、字典、输出目录准备完成，后续帧会被标定流程接管。
   */
  bool Start(std::string_view marker_size_text, int cols, int rows,
             std::string_view camera_name)
  {
    double marker_mm = 0.0;
    if (!ParseMarkerMm(marker_size_text, marker_mm))
    {
      const std::string marker_size(marker_size_text);
      XR_LOG_ERROR("camera calibration: invalid marker size %s",
                   marker_size.c_str());
      return false;
    }

    if (cols < 2 || rows < 2)
    {
      XR_LOG_ERROR("camera calibration: invalid board cols=%d rows=%d",
                   cols, rows);
      return false;
    }

    if constexpr (!supports_image_encoding)
    {
      XR_LOG_ERROR("camera calibration: unsupported image encoding=%u",
                   static_cast<unsigned>(CameraInfoV.encoding));
      return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    // Start() 持有锁后一次性替换本轮配置，避免采样线程看到半初始化状态。
    config_ = MakeConfig(marker_mm, cols, rows);
    board_ = MakeGShangMarkerMap(config_);
    dictionary_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_ORIGINAL);
    detector_params_ = cv::aruco::DetectorParameters();
    detector_params_.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;

    // 输出目录在启动阶段创建；后续 debug 图、CSV 和 YAML 都落在同一棵目录。
    output_dir_ = BuildOutputDir(camera_name, config_);
    debug_dir_ = (std::filesystem::path(output_dir_) / "debug").string();
    std::error_code ec;
    std::filesystem::create_directories(debug_dir_, ec);
    if (ec)
    {
      XR_LOG_ERROR("camera calibration: create output dir failed %s: %s",
                   output_dir_.c_str(), ec.message().c_str());
      return false;
    }

    accepted_views_.clear();
    accepted_views_.shrink_to_fit();
    ResetCountersLocked();
    // active_fast_ 是 VisionCapture 处理线程的无锁快速判断，active_ 是锁内权威状态。
    active_ = true;
    active_fast_.store(true, std::memory_order_release);

    XR_LOG_INFO("camera calibration started: marker=%.3fmm board=%dx%d "
                "square=%.3fmm min_marker_ratio=%.2f required_markers=%d "
                "recommended_views=%d output=%s",
                static_cast<float>(config_.marker_mm), config_.cols, config_.rows,
                static_cast<float>(SquareMm(config_)),
                static_cast<float>(config_.min_marker_ratio),
                RequiredMarkerCount(config_), config_.recommended_views,
                output_dir_.c_str());
    return true;
  }

  /**
   * @brief 处理来自 VisionCapture::ProcessFrame() 的一帧图像。
   *
   * 标定激活时，本函数无论是否真正跑检测都会吞掉图像，保证采样期间不向
   * CameraFrameSync 发布相机图像；标定未激活时返回 false 让正常发布路径继续。
   *
   * @param data 指向当前图像缓冲区的首地址，生命周期由 CameraBase 当前帧保证。
   * @param timestamp_us 当前帧的相机时间戳，单位 us。
   * @return true 表示该帧被标定流程接管，正常图像发布必须跳过。
   */
  bool ProcessFrame(const uint8_t* data, uint64_t timestamp_us)
  {
    if (!active_fast_.load(std::memory_order_acquire))
    {
      return false;
    }

    FrameSnapshot snapshot;
    // PrepareFrame() 在锁内决定发布/吞帧/处理，并复制检测所需的不可变快照。
    const FrameAction action = PrepareFrame(data, timestamp_us, snapshot);
    if (action == FrameAction::PUBLISH)
    {
      return false;
    }
    if (action == FrameAction::SWALLOW)
    {
      return true;
    }

    Detection detection;
    // MakeImageView() 只封装外部图像内存，不拥有像素数据，不在锁内跑 OpenCV。
    const cv::Mat image = MakeImageView(data);
    if (image.empty())
    {
      LogUnsupportedEncodingOnce();
      return true;
    }
    if (!DetectUsableView(image, snapshot, detection))
    {
      return true;
    }

    StoredView stored;
    if (StoreDetection(snapshot, detection, stored))
    {
      SaveDebugImage(image, detection, stored);
    }
    return true;
  }

  /**
   * @brief 停止当前标定并立即恢复 CameraFrameSync 图像发布。
   *
   * 已采到的视角仍保留在内存中，便于 status 查看；不会写出标定文件。
   */
  void Stop()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    active_ = false;
    active_fast_.store(false, std::memory_order_release);
    XR_LOG_INFO("camera calibration stopped: views=%u output=%s",
                static_cast<unsigned>(accepted_views_.size()),
                output_dir_.c_str());
  }

  /**
   * @brief 停止采样、运行 OpenCV 标定并写出 YAML/CSV/debug 摘要。
   *
   * @return true 表示数值求解和文件输出完成；false 表示 OpenCV 求解失败。
   */
  bool SaveAndStop()
  {
    std::vector<View> views;
    Config config;
    std::string output_dir;

    {
      std::lock_guard<std::mutex> lock(mutex_);
      // 保存命令会结束本次标定。即使数值求解失败，也恢复图像发布，
      // 操作手可以调整姿态后重新开始一轮。
      active_ = false;
      active_fast_.store(false, std::memory_order_release);
      // 复制视角和配置后释放锁，避免耗时的 calibrateCamera 阻塞采样入口。
      views = accepted_views_;
      config = config_;
      output_dir = output_dir_;
    }

    CalibrationOutput output;
    if (!CalibrateAndWrite(views, config, output_dir, output))
    {
      return false;
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      finished_ = true;
      last_rms_ = output.rms;
      last_saved_yaml_ = output.yaml_path;
    }
    PrintQualityReport(output.quality_report_text);
    PrintXRobotYamlSnippet(output.xrobot_yaml_snippet);
    return true;
  }

  /**
   * @brief 生成命令行 `cali status` 使用的中文状态摘要。
   *
   * @return 包含活动状态、视角数、拒绝计数、标定板尺寸、输出目录和最近 RMS 的文本。
   */
  std::string StatusString() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream os;
    os << "标定 激活=" << (active_ ? 1 : 0)
       << " 完成=" << (finished_ ? 1 : 0)
       << " 视角=" << accepted_views_.size()
       << " 建议=" << config_.recommended_views
       << " 已处理=" << processed_frames_
       << " 已检测=" << detected_frames_
       << " 模糊拒绝=" << sharpness_rejected_frames_
       << " 重复拒绝=" << duplicate_rejected_frames_
       << " 已截断发布=" << swallowed_frames_
       << " 标定板=" << config_.cols << "x" << config_.rows
       << " 标记mm=" << config_.marker_mm
       << " 方格mm=" << SquareMm(config_)
       << " 标记覆盖=" << config_.min_marker_ratio
       << " 最少标记=" << RequiredMarkerCount(config_)
       << " 最佳清晰度=" << best_sharpness_score_
       << " 输出=" << output_dir_;
    if (last_rms_ >= 0.0)
    {
      os << " rms=" << last_rms_ << " yaml=" << last_saved_yaml_;
    }
    return os.str();
  }

  /**
   * @brief 返回当前已接受的标定视角数。
   */
  std::size_t AcceptedViewCount() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return accepted_views_.size();
  }

  /**
   * @brief 当前标定是否已经成功写出结果。
   */
  bool Finished() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return finished_;
  }

  /**
   * @brief 采样达到目标视角数时自动保存。
   */
  bool SaveAndStopIfReady(std::size_t target_views)
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (finished_ || !active_ || accepted_views_.size() < target_views)
      {
        return false;
      }
    }
    return SaveAndStop();
  }


 private:
  /// 单个 ArUco marker 在标定板坐标系下的四个三维角点，顺序匹配 OpenCV 检测结果。
  using MarkerCorners = std::array<cv::Point3f, 4>;
  /// marker id 到三维角点的查表；用于把 GShang 生成器布局映射为 OpenCV object points。
  using BoardMap = std::map<int, MarkerCorners>;

  /**
   * @brief 已接受并参与后续标定求解的单视角样本。
   *
   * View 只保存求解需要的点集和质量指标，不保存整帧图像；debug 图会立即写到磁盘。
   */
  struct View
  {
    /// VisionCapture 当前标定轮次内的原始帧序号。
    uint64_t frame_index = 0;
    /// 该视角对应的相机时间戳，单位 us。
    uint64_t timestamp_us = 0;
    /// 该视角中实际参与 object/image points 构建的 marker 数。
    int used_markers = 0;
    /// 用平面单应性估计出的像素 RMS，用于早期几何一致性筛选。
    double homography_rms = 0.0;
    /// marker 区域拉普拉斯方差，数值越大通常越清晰。
    double sharpness_score = 0.0;
    /// 标定板外接框中心的归一化 x 坐标，范围大致为 0 到 1。
    double center_x_norm = 0.0;
    /// 标定板外接框中心的归一化 y 坐标，范围大致为 0 到 1。
    double center_y_norm = 0.0;
    /// 标定板外接框面积相对整幅图面积的平方根，用于近似距离变化。
    double scale_norm = 0.0;
    /// 标定板在图像平面内的归一化角度，范围为 [0, 180) 度。
    double angle_deg = 0.0;
    /// OpenCV calibrateCamera 的三维标定板点，单位 mm。
    std::vector<cv::Point3f> object_points;
    /// 与 object_points 一一对应的二维图像角点，单位 pixel。
    std::vector<cv::Point2f> image_points;
  };

  /**
   * @brief 从锁内复制到 OpenCV 检测阶段的只读帧快照。
   *
   * 检测和标定开销较大，不能长期持有 mutex_；因此 PrepareFrame() 只在锁内复制这些数据。
   */
  struct FrameSnapshot
  {
    /// 本轮标定参数快照。
    Config config;
    /// 本轮标定板三维角点查表快照。
    BoardMap board;
    /// OpenCV ArUco 字典，OpenCV 4.13 起位于 objdetect/aruco 新 API。
    cv::aruco::Dictionary dictionary;
    /// OpenCV 检测参数，OpenCV 4.13 起为普通值类型。
    cv::aruco::DetectorParameters detector_params;
    /// 当前原始帧序号。
    uint64_t frame_index = 0;
    /// 当前帧时间戳，单位 us。
    uint64_t timestamp_us = 0;
  };

  /**
   * @brief 单帧检测产生的中间结果。
   *
   * Detection 包含 OpenCV 原始 marker 结果、转换后的点集，以及用于决定是否入库的质量指标。
   */
  struct Detection
  {
    /// 当前帧中属于本标定板且角点完整的 marker 数。
    int used_markers = 0;
    /// 单应性 RMS，越大说明点集越不符合单平面标定板模型。
    double homography_rms = 0.0;
    /// marker 区域清晰度分数。
    double sharpness_score = 0.0;
    /// 标定板外接框中心归一化 x 坐标。
    double center_x_norm = 0.0;
    /// 标定板外接框中心归一化 y 坐标。
    double center_y_norm = 0.0;
    /// 标定板外接框归一化尺度。
    double scale_norm = 0.0;
    /// 标定板图像平面内角度。
    double angle_deg = 0.0;
    /// 已按 marker id 映射后的三维点。
    std::vector<cv::Point3f> object_points;
    /// 已按 marker id 映射后的二维点。
    std::vector<cv::Point2f> image_points;
    /// OpenCV detectMarkers 返回的 marker 四角点，保留用于 debug 绘制和清晰度 mask。
    std::vector<std::vector<cv::Point2f>> marker_corners;
    /// OpenCV detectMarkers 返回的 marker id 列矩阵。
    cv::Mat marker_ids;
  };

  /**
   * @brief StoreDetection() 接受样本后传给 SaveDebugImage() 的轻量元数据。
   *
   * 该结构避免 SaveDebugImage() 重新加锁读取 accepted_views_。
   */
  struct StoredView
  {
    /// debug 图输出目录。
    std::string debug_dir;
    /// 已接受视角编号，从 1 开始用于文件名和图像标注。
    std::size_t view_number = 0;
    /// 原始帧序号。
    uint64_t frame_index = 0;
    /// 接受时的 marker 数。
    int used_markers = 0;
    /// 接受时的单应性 RMS。
    double homography_rms = 0.0;
    /// 接受时的清晰度分数。
    double sharpness_score = 0.0;
  };

  /// 保存阶段回填给运行状态的最终输出摘要。
  struct CalibrationOutput
  {
    /// OpenCV calibrateCamera 返回的全局重投影 RMS。
    double rms = -1.0;
    /// 写出的 calibration.yml 路径。
    std::string yaml_path;
    /// 标定质量报告路径。
    std::string quality_report_path;
    /// 标定完成后打印到 stdout 的质量报告。
    std::string quality_report_text;
    /// 可直接粘贴到 xrobot.yaml 的 CameraInfo constexpr 片段。
    std::string xrobot_yaml_snippet;
  };

  /// PrepareFrame() 对当前帧给出的发布路径决策。
  enum class FrameAction
  {
    /// 标定未激活，交还给 CameraBase 正常发布。
    PUBLISH,
    /// 标定激活但本帧不做检测，吞掉帧以切断下游图像发布。
    SWALLOW,
    /// 标定激活且当前帧需要进入 OpenCV 检测。
    PROCESS,
  };

  /**
   * @brief 根据命令参数生成完整配置。
   *
   * marker_count 为 GShang 布局中实际存在的 ArUco marker 数；min_marker_ratio
   * 保留为比例配置，具体最少 marker 数由 RequiredMarkerCount() 按当前板型派生。
   */
  static Config MakeConfig(double marker_mm, int cols, int rows)
  {
    Config config{};
    config.marker_mm = marker_mm;
    config.cols = cols;
    config.rows = rows;
    return config;
  }

  /**
   * @brief 计算当前 GShang ChArUco 标定板上的 marker 总数。
   */
  static int MarkerCount(const Config& config)
  {
    return (config.rows * config.cols) / 2;
  }

  /**
   * @brief 按比例阈值派生当前板型的最少可见 marker 数。
   *
   * 比例阈值比固定数量更适合 4x4、8x6、11x8 等不同尺寸标定板；下限用于避免小板
   * 因比例计算过低而接受几何约束太弱的帧。
   */
  static int RequiredMarkerCount(const Config& config)
  {
    const int marker_count = MarkerCount(config);
    const int by_ratio = static_cast<int>(
        std::ceil(static_cast<double>(marker_count) * config.min_marker_ratio));
    return std::max(1, std::min(marker_count,
                                std::max(minimum_required_markers, by_ratio)));
  }

  /**
   * @brief 重置本轮标定的计数器和一次性日志标记。
   *
   * 调用方必须已持有 mutex_，因为这些字段和采样状态一起更新。
   */
  void ResetCountersLocked()
  {
    raw_frame_index_ = 0;
    swallowed_frames_ = 0;
    processed_frames_ = 0;
    detected_frames_ = 0;
    last_accept_timestamp_us_ = 0;
    last_rms_ = -1.0;
    last_saved_yaml_.clear();
    finished_ = false;
    unsupported_encoding_logged_ = false;
    recommended_views_logged_ = false;
    best_sharpness_score_ = 0.0;
    sharpness_rejected_frames_ = 0;
    duplicate_rejected_frames_ = 0;
  }

  /**
   * @brief 在锁内完成帧闸门决策，并为需要检测的帧生成快照。
   *
   * 这里是 CameraBase 发布路径和标定路径的分界点：只要 active_ 为 true，帧就被吞掉；
   * 只有抽帧命中的帧才继续跑 OpenCV 检测。
   */
  FrameAction PrepareFrame(const uint8_t* data, uint64_t timestamp_us,
                           FrameSnapshot& snapshot)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!active_)
    {
      return FrameAction::PUBLISH;
    }

    ++swallowed_frames_;
    ++raw_frame_index_;
    // data 为空或未命中抽帧间隔时仍然截断发布，但不消耗 OpenCV 检测算力。
    if (data == nullptr || config_.process_stride <= 0 ||
        (raw_frame_index_ % static_cast<uint64_t>(config_.process_stride)) != 0)
    {
      return FrameAction::SWALLOW;
    }

    ++processed_frames_;
    // 复制完整检测上下文，后续 DetectUsableView() 可以无锁运行。
    snapshot.config = config_;
    snapshot.board = board_;
    snapshot.dictionary = dictionary_;
    snapshot.detector_params = detector_params_;
    snapshot.frame_index = raw_frame_index_;
    snapshot.timestamp_us = timestamp_us;
    return FrameAction::PROCESS;
  }

  /**
   * @brief 对单帧运行 ArUco 检测，并判断是否达到基础可用条件。
   *
   * @return true 表示 marker 数、GShang id 映射、平面单应性和质量指标都已准备好。
   */
  bool DetectUsableView(const cv::Mat& image, const FrameSnapshot& snapshot,
                        Detection& detection)
  {
    std::vector<std::vector<cv::Point2f>> rejected;
    try
    {
#if CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 7)
      cv::aruco::ArucoDetector detector(snapshot.dictionary,
                                        snapshot.detector_params);
      detector.detectMarkers(image, detection.marker_corners,
                             detection.marker_ids, rejected);
#else
      const auto dictionary =
          cv::makePtr<cv::aruco::Dictionary>(snapshot.dictionary);
      const auto detector_params =
          cv::makePtr<cv::aruco::DetectorParameters>(snapshot.detector_params);
      cv::aruco::detectMarkers(image, dictionary, detection.marker_corners,
                               detection.marker_ids, detector_params, rejected);
#endif
    }
    catch (const cv::Exception& e)
    {
      XR_LOG_ERROR("camera calibration: detectMarkers failed: %s", e.what());
      return false;
    }

    // 只收集属于当前 GShang 标定板布局的 marker，画面里的其它 marker 直接忽略。
    if (!CollectPoints(detection.marker_corners, detection.marker_ids,
                       snapshot.board, detection.object_points,
                       detection.image_points, detection.used_markers))
    {
      return false;
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      ++detected_frames_;
    }

    // 单应性 RMS 是快速几何门槛，能在 calibrateCamera 前拒绝明显错配的角点集。
    detection.homography_rms =
        HomographyRms(detection.object_points, detection.image_points);
    if (detection.used_markers < RequiredMarkerCount(snapshot.config) ||
        detection.homography_rms > snapshot.config.max_homography_rms)
    {
      return false;
    }

    // 质量指标只对已经通过基础几何门槛的帧计算，减少无效帧的额外开销。
    FillDetectionQuality(image, detection);
    return true;
  }

  /**
   * @brief 把通过检测的帧按时间、容量、清晰度和重复姿态规则写入采样池。
   *
   * @return true 表示该视角已被接受，调用方应同步保存 debug 图。
   */
  bool StoreDetection(const FrameSnapshot& snapshot, const Detection& detection,
                      StoredView& stored)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!active_)
    {
      return false;
    }

    // 时间间隔门槛防止相机/手机静止时连续帧高度相关。
    if (last_accept_timestamp_us_ != 0 &&
        snapshot.timestamp_us > last_accept_timestamp_us_ &&
        snapshot.timestamp_us - last_accept_timestamp_us_ <
            config_.min_accept_interval_us)
    {
      return false;
    }

    // 使用“绝对清晰度”和“相对本轮最佳清晰度”两条线处理不同曝光/距离的视频。
    best_sharpness_score_ = std::max(best_sharpness_score_, detection.sharpness_score);
    const double sharpness_threshold =
        std::max(config_.min_sharpness_score,
                 best_sharpness_score_ * config_.min_sharpness_best_ratio);
    if (detection.sharpness_score < sharpness_threshold)
    {
      ++sharpness_rejected_frames_;
      return false;
    }

    // 近重复门槛鼓励覆盖不同中心、距离和旋转，提升内参可观测性。
    if (IsNearDuplicateLocked(detection))
    {
      ++duplicate_rejected_frames_;
      return false;
    }

    // 入库的 View 与 Detection 分离，保留标定所需点集和质量指标即可。
    View view;
    view.frame_index = snapshot.frame_index;
    view.timestamp_us = snapshot.timestamp_us;
    view.used_markers = detection.used_markers;
    view.homography_rms = detection.homography_rms;
    view.sharpness_score = detection.sharpness_score;
    view.center_x_norm = detection.center_x_norm;
    view.center_y_norm = detection.center_y_norm;
    view.scale_norm = detection.scale_norm;
    view.angle_deg = detection.angle_deg;
    view.object_points = detection.object_points;
    view.image_points = detection.image_points;
    accepted_views_.push_back(std::move(view));
    last_accept_timestamp_us_ = snapshot.timestamp_us;

    stored.debug_dir = debug_dir_;
    stored.view_number = accepted_views_.size();
    stored.frame_index = snapshot.frame_index;
    stored.used_markers = detection.used_markers;
    stored.homography_rms = detection.homography_rms;
    stored.sharpness_score = detection.sharpness_score;

    XR_LOG_PASS("camera calibration: accepted=%u frames=%u processed=%u "
                "detected=%u markers=%d H-rms=%.3f sharpness=%.1f ts_ms=%u",
                static_cast<unsigned>(accepted_views_.size()),
                static_cast<unsigned>(swallowed_frames_),
                static_cast<unsigned>(processed_frames_),
                static_cast<unsigned>(detected_frames_), detection.used_markers,
                static_cast<float>(detection.homography_rms),
                static_cast<float>(detection.sharpness_score),
                static_cast<unsigned>(snapshot.timestamp_us / 1000U));

    if (!recommended_views_logged_ &&
        static_cast<int>(accepted_views_.size()) >= config_.recommended_views)
    {
      XR_LOG_INFO("camera calibration: reached recommended views %d, keep sampling or run cali save",
                  config_.recommended_views);
      recommended_views_logged_ = true;
    }
    return true;
  }

  /**
   * @brief 对不支持的像素编码只打一条错误日志，避免每帧刷屏。
   */
  void LogUnsupportedEncodingOnce()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (unsupported_encoding_logged_)
    {
      return;
    }
    unsupported_encoding_logged_ = true;
    XR_LOG_ERROR("camera calibration: unsupported image encoding=%u",
                 static_cast<unsigned>(CameraInfoV.encoding));
  }

  /**
   * @brief 根据 GShang 几何比例计算完整棋盘方格边长。
   *
   * @return 方格边长，单位 mm。
   */
  static double SquareMm(const Config& config)
  {
    return config.marker_mm * static_cast<double>(gshang_square_cells) /
           static_cast<double>(gshang_marker_cells);
  }

  /**
   * @brief 解析 `25mm` / `25` 形式的 marker 边长。
   *
   * @return true 表示解析出正的有限数值，单位 mm。
   */
  static bool ParseMarkerMm(std::string_view text, double& marker_mm)
  {
    std::string value(text);
    if (value.size() >= 2)
    {
      const std::string suffix = value.substr(value.size() - 2);
      // 命令行允许带 mm 后缀，内部统一存为裸数值。
      if (suffix == "mm" || suffix == "MM")
      {
        value.resize(value.size() - 2);
      }
    }

    char* end = nullptr;
    const double parsed = std::strtod(value.c_str(), &end);
    if (end == value.c_str() || *end != '\0' || parsed <= 0.0 ||
        !std::isfinite(parsed))
    {
      return false;
    }

    marker_mm = parsed;
    return true;
  }

  /**
   * @brief 把相机名转成安全的目录名片段。
   *
   * 非字母数字、下划线和连字符都会替换为下划线，避免路径中出现空格或特殊字符。
   */
  static std::string Sanitize(std::string_view text)
  {
    std::string out;
    out.reserve(text.size());
    for (char ch : text)
    {
      if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
          (ch >= '0' && ch <= '9') || ch == '_' || ch == '-')
      {
        out.push_back(ch);
      }
      else
      {
        out.push_back('_');
      }
    }
    return out.empty() ? "camera" : out;
  }

  /**
   * @brief 生成用于输出目录的本地时间戳。
   */
  static std::string TimeStampString()
  {
    std::time_t now = std::time(nullptr);
    std::tm tm_buf{};
    localtime_r(&now, &tm_buf);

    char buf[32]{};
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm_buf);
    return std::string(buf);
  }

  /**
   * @brief 生成本轮标定输出目录。
   *
   * 路径格式为 `runs/camera_calib/<timestamp>_<camera>_<marker>mm_<cols>x<rows>`。
   */
  static std::string BuildOutputDir(std::string_view camera_name,
                                    const Config& config)
  {
    std::ostringstream leaf;
    leaf << TimeStampString() << "_" << Sanitize(camera_name) << "_"
         << std::fixed << std::setprecision(0) << config.marker_mm << "mm_"
         << config.cols << "x" << config.rows;
    return (std::filesystem::path("runs") / "camera_calib" / leaf.str()).string();
  }

  /**
   * @brief 生成 GShang 在线工具对应的 ChArUco 三维角点表。
   *
   * GShang 工具在 (row + col) 为奇数的棋盘格中放置 ArUco-original 5-bit 标记。
   * 该工具里的 markerSize 不是完整方格宽度：5-bit 标记外侧还有白边和棋盘边界，
   * 因此实际方格宽度为 square_mm = marker_mm * 9 / 7。
   */
  static BoardMap MakeGShangMarkerMap(const Config& config)
  {
    const double cell_mm = config.marker_mm / static_cast<double>(gshang_marker_cells);
    const double square_mm = cell_mm * static_cast<double>(gshang_square_cells);
    const double marker_offset_mm = cell_mm;

    BoardMap map;
    int marker_id = 0;
    for (int row = 0; row < config.rows; ++row)
    {
      for (int col = 0; col < config.cols; ++col)
      {
        // GShang ChArUco 只在黑白棋盘的奇偶格中放 marker，id 按扫描顺序递增。
        if (((row + col) % 2) == 0)
        {
          continue;
        }

        const float x0 = static_cast<float>(col * square_mm + marker_offset_mm);
        const float y0 = static_cast<float>(row * square_mm + marker_offset_mm);
        const float x1 = x0 + static_cast<float>(config.marker_mm);
        const float y1 = y0 + static_cast<float>(config.marker_mm);
        map[marker_id++] = {cv::Point3f{x0, y0, 0.0F},
                            cv::Point3f{x1, y0, 0.0F},
                            cv::Point3f{x1, y1, 0.0F},
                            cv::Point3f{x0, y1, 0.0F}};
      }
    }
    return map;
  }

  /**
   * @brief 把 OpenCV 检测到的 marker id/角点转换为标定用 object/image points。
   *
   * 只使用在 BoardMap 中能找到的 id；这样即使画面中出现无关 ArUco，也不会污染标定。
   *
   * @return true 表示至少收集到一个属于当前标定板的 marker。
   */
  static bool CollectPoints(const std::vector<std::vector<cv::Point2f>>& corners,
                            const cv::Mat& ids, const BoardMap& board,
                            std::vector<cv::Point3f>& object_points,
                            std::vector<cv::Point2f>& image_points,
                            int& used_markers)
  {
    object_points.clear();
    image_points.clear();
    used_markers = 0;
    if (ids.empty())
    {
      return false;
    }

    for (int i = 0; i < ids.rows; ++i)
    {
      if (i >= static_cast<int>(corners.size()) || corners[i].size() < 4)
      {
        continue;
      }

      const int id = ids.at<int>(i, 0);
      const auto marker = board.find(id);
      // 忽略字典中存在但不属于当前 GShang 尺寸/布局的 marker。
      if (marker == board.end())
      {
        continue;
      }

      for (int corner = 0; corner < 4; ++corner)
      {
        object_points.push_back(marker->second[corner]);
        image_points.push_back(corners[i][corner]);
      }
      ++used_markers;
    }
    return used_markers > 0;
  }

  /**
   * @brief 将输入图像转换为灰度视图/图像，供清晰度评估使用。
   *
   * 单通道输入直接返回原 Mat；三/四通道输入生成新的灰度 Mat。
   */
  static cv::Mat MakeGrayImage(const cv::Mat& image)
  {
    if (image.channels() == 1)
    {
      return image;
    }

    cv::Mat gray;
    if (image.channels() == 3)
    {
      cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }
    else if (image.channels() == 4)
    {
      cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    }
    return gray;
  }

  /**
   * @brief 计算 marker 区域内的拉普拉斯方差作为清晰度指标。
   *
   * 只在 marker 多边形及轻微膨胀区域统计，避免背景纹理或屏幕边框影响运动模糊判断。
   */
  static double MarkerSharpnessScore(
      const cv::Mat& image, const std::vector<std::vector<cv::Point2f>>& corners)
  {
    const cv::Mat gray = MakeGrayImage(image);
    if (gray.empty())
    {
      return 0.0;
    }

    cv::Mat marker_mask = cv::Mat::zeros(gray.size(), CV_8UC1);
    for (const auto& marker : corners)
    {
      if (marker.size() < 4)
      {
        continue;
      }

      std::array<cv::Point, 4> polygon{};
      for (int i = 0; i < 4; ++i)
      {
        // fillConvexPoly 需要整数像素点，使用四舍五入保留角点所在区域。
        polygon[static_cast<std::size_t>(i)] =
            cv::Point{static_cast<int>(std::lround(marker[static_cast<std::size_t>(i)].x)),
                      static_cast<int>(std::lround(marker[static_cast<std::size_t>(i)].y))};
      }
      cv::fillConvexPoly(marker_mask, polygon.data(), static_cast<int>(polygon.size()),
                         cv::Scalar{255});
    }

    if (cv::countNonZero(marker_mask) < 16)
    {
      return 0.0;
    }

    // 膨胀一圈把 marker 边缘纳入统计，边缘越锐利，拉普拉斯方差越高。
    cv::dilate(marker_mask, marker_mask, cv::Mat{}, {-1, -1}, 1);
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F, 3);

    cv::Scalar mean;
    cv::Scalar stddev;
    cv::meanStdDev(laplacian, mean, stddev, marker_mask);
    return stddev[0] * stddev[0];
  }

  /**
   * @brief 把 OpenCV minAreaRect 角度规整到 [0, 180)。
   *
   * 标定板正反 180 度在近重复判断中等价，因此只保留半圆范围。
   */
  static double NormalizeBoardAngle(double angle_deg, const cv::Size2f& size)
  {
    if (size.width < size.height)
    {
      angle_deg += 90.0;
    }

    while (angle_deg < 0.0)
    {
      angle_deg += 180.0;
    }
    while (angle_deg >= 180.0)
    {
      angle_deg -= 180.0;
    }
    return angle_deg;
  }

  /**
   * @brief 计算两个平面角度在 180 度周期下的最小差值。
   */
  static double AngleDeltaDeg(double lhs, double rhs)
  {
    double delta = std::fabs(lhs - rhs);
    while (delta >= 180.0)
    {
      delta -= 180.0;
    }
    return delta > 90.0 ? 180.0 - delta : delta;
  }

  /**
   * @brief 填充检测帧的清晰度、图像中心、尺度和旋转角质量指标。
   *
   * 这些指标不直接参与 OpenCV 标定，但用于在线采样去模糊、去重复，以及 views.csv 复盘。
   */
  static void FillDetectionQuality(const cv::Mat& image, Detection& detection)
  {
    detection.sharpness_score =
        MarkerSharpnessScore(image, detection.marker_corners);

    if (detection.image_points.empty() || CameraInfoV.width == 0 ||
        CameraInfoV.height == 0)
    {
      return;
    }

    const cv::Rect bounds = cv::boundingRect(detection.image_points);
    const double image_width = static_cast<double>(CameraInfoV.width);
    const double image_height = static_cast<double>(CameraInfoV.height);
    detection.center_x_norm =
        (static_cast<double>(bounds.x) + static_cast<double>(bounds.width) * 0.5) /
        image_width;
    detection.center_y_norm =
        (static_cast<double>(bounds.y) + static_cast<double>(bounds.height) * 0.5) /
        image_height;
    detection.scale_norm =
        std::sqrt(std::max(0.0, static_cast<double>(bounds.area())) /
                  std::max(1.0, image_width * image_height));

    if (detection.image_points.size() >= 4)
    {
      // 用全部角点的最小外接旋转矩形估计板面在图像内的旋转角。
      const cv::RotatedRect rect = cv::minAreaRect(detection.image_points);
      detection.angle_deg = NormalizeBoardAngle(rect.angle, rect.size);
    }
  }

  /**
   * @brief 判断当前检测是否和已有样本过近。
   *
   * 调用方必须持有 mutex_，因为本函数遍历 accepted_views_ 并读取 config_ 阈值。
   */
  bool IsNearDuplicateLocked(const Detection& detection) const
  {
    for (const View& view : accepted_views_)
    {
      // 中心、尺度和角度都很接近时才视为重复；任一维度变化足够大则保留。
      const double center_delta =
          std::hypot(detection.center_x_norm - view.center_x_norm,
                     detection.center_y_norm - view.center_y_norm);
      const double scale_delta =
          std::fabs(std::log((detection.scale_norm + 1e-6) /
                             (view.scale_norm + 1e-6)));
      const double angle_delta =
          AngleDeltaDeg(detection.angle_deg, view.angle_deg);
      if (center_delta < config_.min_center_delta_norm &&
          scale_delta < config_.min_scale_delta_log &&
          angle_delta < config_.min_angle_delta_deg)
      {
        return true;
      }
    }
    return false;
  }

  /**
   * @brief 使用平面单应性估计当前角点集的像素 RMS。
   *
   * 这是保存前标定的轻量前置检查：标定板是平面，正确角点应能被单应性较好解释。
   */
  static double HomographyRms(const std::vector<cv::Point3f>& object_points,
                              const std::vector<cv::Point2f>& image_points)
  {
    if (object_points.size() < 4 || object_points.size() != image_points.size())
    {
      return 1e9;
    }

    std::vector<cv::Point2f> object_xy;
    object_xy.reserve(object_points.size());
    for (const auto& point : object_points)
    {
      // 标定板 z 恒为 0，单应性只需要平面坐标 x/y。
      object_xy.emplace_back(point.x, point.y);
    }

    const cv::Mat homography = cv::findHomography(object_xy, image_points, 0);
    if (homography.empty())
    {
      return 1e9;
    }

    std::vector<cv::Point2f> projected;
    cv::perspectiveTransform(object_xy, projected, homography);

    double sum2 = 0.0;
    for (std::size_t i = 0; i < image_points.size(); ++i)
    {
      const cv::Point2f delta = projected[i] - image_points[i];
      sum2 += delta.dot(delta);
    }
    return std::sqrt(sum2 / static_cast<double>(image_points.size()));
  }

  /**
   * @brief 根据 CameraInfoV 把外部图像缓冲区封装为 OpenCV Mat。
   *
   * 本函数不复制像素，返回的 cv::Mat 只在当前 ProcessFrame() 调用期间有效。
   */
  cv::Mat MakeImageView(const uint8_t* data) const
  {
    constexpr int width = static_cast<int>(CameraInfoV.width);
    constexpr int height = static_cast<int>(CameraInfoV.height);
    constexpr std::size_t step = static_cast<std::size_t>(CameraInfoV.step);

    if constexpr (CameraInfoV.encoding == CameraTypes::Encoding::BGR8 ||
                  CameraInfoV.encoding == CameraTypes::Encoding::RGB8)
    {
      return cv::Mat(height, width, CV_8UC3, const_cast<uint8_t*>(data), step);
    }
    else if constexpr (CameraInfoV.encoding == CameraTypes::Encoding::BGRA8 ||
                       CameraInfoV.encoding == CameraTypes::Encoding::RGBA8)
    {
      return cv::Mat(height, width, CV_8UC4, const_cast<uint8_t*>(data), step);
    }
    else if constexpr (CameraInfoV.encoding == CameraTypes::Encoding::MONO8)
    {
      return cv::Mat(height, width, CV_8UC1, const_cast<uint8_t*>(data), step);
    }
    else
    {
      return {};
    }
  }

  /**
   * @brief 调用 OpenCV calibrateCamera 计算内参、畸变和每视角外参。
   *
   * @return true 保留为调用语义占位；OpenCV 异常由上层捕获。
   */
  static bool CalibrateViews(const std::vector<View>& views,
                             cv::Mat& camera_matrix, cv::Mat& distortion,
                             double& rms, std::vector<cv::Mat>& rvecs,
                             std::vector<cv::Mat>& tvecs)
  {
    std::vector<std::vector<cv::Point3f>> object_points;
    std::vector<std::vector<cv::Point2f>> image_points;
    object_points.reserve(views.size());
    image_points.reserve(views.size());
    for (const View& view : views)
    {
      // calibrateCamera 需要按视角分组的点集，而 View 已经保存了一一对应关系。
      object_points.push_back(view.object_points);
      image_points.push_back(view.image_points);
    }

    rms = cv::calibrateCamera(
        object_points, image_points,
        cv::Size(static_cast<int>(CameraInfoV.width),
                 static_cast<int>(CameraInfoV.height)),
        camera_matrix, distortion, rvecs, tvecs);
    return true;
  }

  /**
   * @brief 计算单个视角在给定内参/畸变/外参下的重投影 RMS。
   */
  static double ReprojectionRms(const View& view, const cv::Mat& camera_matrix,
                                const cv::Mat& distortion,
                                const cv::Mat& rvec, const cv::Mat& tvec)
  {
    std::vector<cv::Point2f> projected;
    cv::projectPoints(view.object_points, rvec, tvec,
                      camera_matrix, distortion, projected);

    double sum2 = 0.0;
    for (std::size_t i = 0; i < view.image_points.size(); ++i)
    {
      const cv::Point2f delta = projected[i] - view.image_points[i];
      sum2 += delta.dot(delta);
    }
    return std::sqrt(sum2 / static_cast<double>(view.image_points.size()));
  }

  /**
   * @brief 批量计算所有视角的重投影 RMS。
   *
   * rvecs/tvecs 必须与 views 一一对应，来源于同一次 CalibrateViews()。
   */
  static std::vector<double> PerViewRms(const std::vector<View>& views,
                                        const cv::Mat& camera_matrix,
                                        const cv::Mat& distortion,
                                        const std::vector<cv::Mat>& rvecs,
                                        const std::vector<cv::Mat>& tvecs)
  {
    std::vector<double> rms;
    rms.reserve(views.size());
    for (std::size_t i = 0; i < views.size(); ++i)
    {
      rms.push_back(ReprojectionRms(views[i], camera_matrix, distortion,
                                    rvecs[i], tvecs[i]));
    }
    return rms;
  }

  /**
   * @brief 从 OpenCV Mat 中抽取 3x3 内参数组。
   */
  static std::array<double, 9> CameraMatrixValues(const cv::Mat& camera_matrix)
  {
    std::array<double, 9> values{};
    for (int i = 0; i < 9; ++i)
    {
      values[static_cast<std::size_t>(i)] =
          camera_matrix.at<double>(i / 3, i % 3);
    }
    return values;
  }

  /**
   * @brief 从 OpenCV 畸变 Mat 中抽取 CameraInfo 固定容量畸变数组。
   */
  static std::array<double, 14> DistortionValues(const cv::Mat& distortion)
  {
    std::array<double, 14> values{};
    const cv::Mat flat = distortion.reshape(1, 1);
    for (int i = 0; i < std::min(flat.cols, static_cast<int>(values.size())); ++i)
    {
      values[static_cast<std::size_t>(i)] = flat.at<double>(0, i);
    }
    return values;
  }

  /**
   * @brief 根据内参生成 CameraInfo 使用的 3x4 投影矩阵。
   */
  static std::array<double, 12> ProjectionValues(const cv::Mat& camera_matrix)
  {
    return {camera_matrix.at<double>(0, 0), 0.0,
            camera_matrix.at<double>(0, 2), 0.0,
            0.0, camera_matrix.at<double>(1, 1),
            camera_matrix.at<double>(1, 2), 0.0,
            0.0, 0.0, 1.0, 0.0};
  }

  /**
   * @brief 计算 RMS 分布的分位数。
   */
  static double Percentile(std::vector<double> values, double percentile)
  {
    if (values.empty())
    {
      return -1.0;
    }

    std::sort(values.begin(), values.end());
    const double clamped = std::max(0.0, std::min(1.0, percentile));
    const std::size_t index = static_cast<std::size_t>(
        std::round(clamped * static_cast<double>(values.size() - 1)));
    return values[index];
  }

  /**
   * @brief 最终参与标定的样本覆盖指标。
   */
  struct CoverageMetrics
  {
    std::size_t views{0};              ///< 最终参与求解的视角数。
    double center_x_min{0.0};          ///< 标定板中心 x 最小值，图像宽度归一化。
    double center_x_max{0.0};          ///< 标定板中心 x 最大值，图像宽度归一化。
    double center_y_min{0.0};          ///< 标定板中心 y 最小值，图像高度归一化。
    double center_y_max{0.0};          ///< 标定板中心 y 最大值，图像高度归一化。
    double scale_min{0.0};             ///< 标定板尺度最小值，按图像面积归一化。
    double scale_max{0.0};             ///< 标定板尺度最大值，按图像面积归一化。
    double angle_min_deg{0.0};         ///< 平面内旋转角最小值。
    double angle_max_deg{0.0};         ///< 平面内旋转角最大值。
    double center_span_x{0.0};         ///< 中心 x 覆盖跨度。
    double center_span_y{0.0};         ///< 中心 y 覆盖跨度。
    double scale_ratio{0.0};           ///< 最大尺度 / 最小尺度。
    double angle_span_deg{0.0};        ///< 平面内旋转角覆盖跨度。
  };

  /**
   * @brief 统计最终样本覆盖范围。
   */
  static CoverageMetrics ComputeCoverage(const std::vector<View>& views)
  {
    CoverageMetrics metrics{};
    metrics.views = views.size();
    if (views.empty())
    {
      return metrics;
    }

    metrics.center_x_min = views.front().center_x_norm;
    metrics.center_x_max = views.front().center_x_norm;
    metrics.center_y_min = views.front().center_y_norm;
    metrics.center_y_max = views.front().center_y_norm;
    metrics.scale_min = views.front().scale_norm;
    metrics.scale_max = views.front().scale_norm;
    metrics.angle_min_deg = views.front().angle_deg;
    metrics.angle_max_deg = views.front().angle_deg;

    for (const View& view : views)
    {
      metrics.center_x_min = std::min(metrics.center_x_min, view.center_x_norm);
      metrics.center_x_max = std::max(metrics.center_x_max, view.center_x_norm);
      metrics.center_y_min = std::min(metrics.center_y_min, view.center_y_norm);
      metrics.center_y_max = std::max(metrics.center_y_max, view.center_y_norm);
      metrics.scale_min = std::min(metrics.scale_min, view.scale_norm);
      metrics.scale_max = std::max(metrics.scale_max, view.scale_norm);
      metrics.angle_min_deg = std::min(metrics.angle_min_deg, view.angle_deg);
      metrics.angle_max_deg = std::max(metrics.angle_max_deg, view.angle_deg);
    }

    metrics.center_span_x = metrics.center_x_max - metrics.center_x_min;
    metrics.center_span_y = metrics.center_y_max - metrics.center_y_min;
    metrics.scale_ratio =
        metrics.scale_min > 0.0 ? metrics.scale_max / metrics.scale_min : 0.0;
    metrics.angle_span_deg = metrics.angle_max_deg - metrics.angle_min_deg;
    return metrics;
  }

  /**
   * @brief 格式化标定完成后直接打印的质量报告。
   */
  static std::string MakeQualityReportText(
      const std::vector<View>& views, const std::vector<double>& per_view_rms,
      const CameraBaseIntrinsicSanity::Metrics& intrinsics, double global_rms,
      double max_reprojection_rms)
  {
    const CoverageMetrics coverage = ComputeCoverage(views);
    const double per_view_p50 = Percentile(per_view_rms, 0.50);
    const double per_view_p95 = Percentile(per_view_rms, 0.95);
    const double per_view_max = Percentile(per_view_rms, 1.00);
    const bool reprojection_ok = global_rms >= 0.0 &&
                                 global_rms <= max_reprojection_rms &&
                                 per_view_max <= max_reprojection_rms;
    const bool quality_ok = intrinsics.all_ok && reprojection_ok;

    std::ostringstream out;
    out << std::setprecision(10);
    out << "===== 标定质量报告开始 =====\n";
    out << "质量总判定(quality_ok): "
        << CameraBaseIntrinsicSanity::PassFail(quality_ok) << "\n";
    out << "最终视角数(views): " << coverage.views << "\n";
    out << "重投影判定(reprojection_ok): "
        << CameraBaseIntrinsicSanity::PassFail(reprojection_ok) << "\n";
    out << "重投影阈值(max_reprojection_rms): "
        << max_reprojection_rms << " px\n";
    out << "全局重投影 RMS(global_rms): " << global_rms << " px\n";
    out << "单视角重投影 RMS P50(per_view_rms_p50): "
        << per_view_p50 << " px\n";
    out << "单视角重投影 RMS P95(per_view_rms_p95): "
        << per_view_p95 << " px\n";
    out << "单视角重投影 RMS 最大值(per_view_rms_max): "
        << per_view_max << " px\n";
    out << "样本中心 X 覆盖(coverage_center_x): [" << coverage.center_x_min << ", "
        << coverage.center_x_max << "] span=" << coverage.center_span_x << "\n";
    out << "样本中心 Y 覆盖(coverage_center_y): [" << coverage.center_y_min << ", "
        << coverage.center_y_max << "] span=" << coverage.center_span_y << "\n";
    out << "样本尺度覆盖(coverage_scale): [" << coverage.scale_min << ", "
        << coverage.scale_max << "] ratio=" << coverage.scale_ratio << "\n";
    out << "样本平面角覆盖(coverage_angle_deg): [" << coverage.angle_min_deg << ", "
        << coverage.angle_max_deg << "] span=" << coverage.angle_span_deg << "\n";
    out << CameraBaseIntrinsicSanity::FormatReport(intrinsics);
    out << "===== 标定质量报告结束 =====\n";
    return out.str();
  }

  /**
   * @brief 根据单视角重投影 RMS 剔除离群样本。
   */
  static std::vector<View> FilterByRms(const std::vector<View>& views,
                                       const std::vector<double>& per_view_rms,
                                       double max_rms)
  {
    std::vector<View> filtered;
    for (std::size_t i = 0; i < views.size(); ++i)
    {
      if (i < per_view_rms.size() && per_view_rms[i] <= max_rms)
      {
        filtered.push_back(views[i]);
      }
    }
    return filtered;
  }

  /**
   * @brief 写出标定质量报告。
   */
  static std::string WriteQualityReport(
      const std::filesystem::path& path, const std::vector<View>& views,
      const std::vector<double>& per_view_rms, const cv::Mat& camera_matrix,
      const cv::Mat& distortion, double global_rms, double max_reprojection_rms)
  {
    const std::array<double, 9> camera_values =
        CameraMatrixValues(camera_matrix);
    const std::array<double, 14> distortion_values =
        DistortionValues(distortion);
    const CameraBaseIntrinsicSanity::Metrics intrinsics =
        CameraBaseIntrinsicSanity::Evaluate(CameraInfoV.width, CameraInfoV.height,
                                            camera_values, distortion_values);

    const std::string report =
        MakeQualityReportText(views, per_view_rms, intrinsics, global_rms,
                              max_reprojection_rms);
    std::ofstream out(path);
    out << report;
    return report;
  }

  /**
   * @brief 保存求解前再次按清晰度过滤样本。
   *
   * 在线阶段会动态更新最佳清晰度；保存阶段重新计算阈值，保证最终求解用的是整轮样本中
   * 相对清晰的一组视角。
   */
  static std::vector<View> FilterBySharpness(const std::vector<View>& views,
                                             const Config& config)
  {
    double best_score = 0.0;
    for (const View& view : views)
    {
      best_score = std::max(best_score, view.sharpness_score);
    }

    const double threshold =
        std::max(config.min_sharpness_score,
                 best_score * config.min_sharpness_best_ratio);

    std::vector<View> filtered;
    filtered.reserve(views.size());
    for (const View& view : views)
    {
      if (view.sharpness_score >= threshold)
      {
        filtered.push_back(view);
      }
    }
    return filtered;
  }

  /**
   * @brief 完成保存命令的数值求解、离群过滤和文件写出。
   *
   * 流程为：清晰度预过滤 -> 初次标定 -> 高 RMS 视角过滤 -> 可选二次标定 -> 写 YAML/CSV/snippet。
   */
  static bool CalibrateAndWrite(const std::vector<View>& input_views,
                                const Config& config,
                                const std::string& output_dir,
                                CalibrationOutput& output)
  {
    // 先用清晰度过滤处理运动模糊；若过滤后视角不足，则回退到原始输入避免误删过多。
    std::vector<View> calibration_views = FilterBySharpness(input_views, config);
    if (calibration_views.size() >= solver_view_floor &&
        calibration_views.size() < input_views.size())
    {
      XR_LOG_INFO("camera calibration: sharpness prefilter kept %u/%u views",
                  static_cast<unsigned>(calibration_views.size()),
                  static_cast<unsigned>(input_views.size()));
    }
    else
    {
      calibration_views = input_views;
    }

    cv::Mat camera_matrix;
    cv::Mat distortion;
    double rms = 0.0;
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;

    try
    {
      CalibrateViews(calibration_views, camera_matrix, distortion, rms, rvecs, tvecs);
    }
    catch (const cv::Exception& e)
    {
      XR_LOG_ERROR("camera calibration: calibrateCamera failed: %s", e.what());
      return false;
    }

    // 初次求解后根据每视角 RMS 去掉离群样本，再做一次最终求解。
    std::vector<View> final_views = calibration_views;
    std::vector<double> final_per_view_rms =
        PerViewRms(calibration_views, camera_matrix, distortion, rvecs, tvecs);

    const std::vector<View> filtered =
        FilterByRms(calibration_views, final_per_view_rms, config.max_reprojection_rms);
    if (filtered.size() >= solver_view_floor &&
        filtered.size() < calibration_views.size())
    {
      cv::Mat filtered_camera_matrix;
      cv::Mat filtered_distortion;
      double filtered_rms = 0.0;
      std::vector<cv::Mat> filtered_rvecs;
      std::vector<cv::Mat> filtered_tvecs;
      try
      {
        CalibrateViews(filtered, filtered_camera_matrix, filtered_distortion,
                       filtered_rms, filtered_rvecs, filtered_tvecs);
      }
      catch (const cv::Exception& e)
      {
        XR_LOG_WARN("camera calibration: outlier-filtered calibration failed: %s", e.what());
      }

      if (!filtered_camera_matrix.empty())
      {
        // 二次求解成功才替换最终输出；失败时保留初次求解结果。
        camera_matrix = filtered_camera_matrix;
        distortion = filtered_distortion;
        rms = filtered_rms;
        final_views = filtered;
        final_per_view_rms = PerViewRms(final_views, camera_matrix, distortion,
                                        filtered_rvecs, filtered_tvecs);
      }
    }

    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);
    if (ec)
    {
      XR_LOG_ERROR("camera calibration: create output dir failed %s: %s",
                   output_dir.c_str(), ec.message().c_str());
      return false;
    }

    const std::filesystem::path yaml_path =
        std::filesystem::path(output_dir) / "calibration.yml";
    const std::filesystem::path csv_path =
        std::filesystem::path(output_dir) / "views.csv";
    const std::filesystem::path quality_path =
        std::filesystem::path(output_dir) / "quality_report.txt";
    const std::filesystem::path snippet_path =
        std::filesystem::path(output_dir) / "camera_info_snippet.txt";
    const std::string xrobot_yaml_snippet =
        MakeXRobotCameraInfoYaml(camera_matrix, distortion);

    // 三个输出分别服务于 OpenCV 复用、视角质量复盘、xrobot.yaml 常量粘贴。
    WriteCalibrationYaml(yaml_path, config, camera_matrix, distortion, rms,
                         static_cast<int>(final_views.size()));
    WriteViewsCsv(csv_path, final_views, final_per_view_rms);
    const std::string quality_report =
        WriteQualityReport(quality_path, final_views, final_per_view_rms,
                           camera_matrix, distortion, rms,
                           config.max_reprojection_rms);
    WriteCameraInfoSnippet(snippet_path, xrobot_yaml_snippet);

    output.rms = rms;
    output.yaml_path = yaml_path.string();
    output.quality_report_path = quality_path.string();
    output.quality_report_text = quality_report;
    output.xrobot_yaml_snippet = xrobot_yaml_snippet;
    XR_LOG_PASS("camera calibration saved: views=%u rms=%.4f yaml=%s quality=%s",
                static_cast<unsigned>(final_views.size()), static_cast<float>(rms),
                output.yaml_path.c_str(), output.quality_report_path.c_str());
    return true;
  }

  /**
   * @brief 写出 OpenCV FileStorage 格式的完整标定结果。
   */
  static void WriteCalibrationYaml(const std::filesystem::path& path,
                                   const Config& config,
                                   const cv::Mat& camera_matrix,
                                   const cv::Mat& distortion,
                                   double rms, int views)
  {
    cv::FileStorage fs(path.string(), cv::FileStorage::WRITE);
    fs << "image_width" << static_cast<int>(CameraInfoV.width);
    fs << "image_height" << static_cast<int>(CameraInfoV.height);
    fs << "rows" << config.rows;
    fs << "cols" << config.cols;
    fs << "marker_mm" << config.marker_mm;
    fs << "square_mm" << SquareMm(config);
    fs << "dictionary" << "aruco_original";
    fs << "generator" << "GShang ChArUco: square = marker * 9 / 7";
    fs << "views" << views;
    fs << "rms" << rms;
    fs << "camera_matrix" << camera_matrix;
    fs << "distortion_coefficients" << distortion;
  }

  /**
   * @brief 写出每个最终视角的质量指标和重投影 RMS。
   *
   * CSV 用于离线检查自动采样是否覆盖了足够姿态，以及哪些视角被最终保留。
   */
  static void WriteViewsCsv(const std::filesystem::path& path,
                            const std::vector<View>& views,
                            const std::vector<double>& per_view_rms)
  {
    std::ofstream csv(path);
    csv << "frame_index,timestamp_us,used_markers,homography_rms,"
           "sharpness_score,center_x_norm,center_y_norm,scale_norm,angle_deg,"
           "per_view_reprojection_rms\n";
    for (std::size_t i = 0; i < views.size(); ++i)
    {
      csv << views[i].frame_index << ","
          << views[i].timestamp_us << ","
          << views[i].used_markers << ","
          << views[i].homography_rms << ","
          << views[i].sharpness_score << ","
          << views[i].center_x_norm << ","
          << views[i].center_y_norm << ","
          << views[i].scale_norm << ","
          << views[i].angle_deg << ","
          << (i < per_view_rms.size() ? per_view_rms[i] : -1.0) << "\n";
    }
  }

  /**
   * @brief 返回当前相机像素编码在 xrobot.yaml 中使用的枚举文本。
   */
  static const char* EncodingName()
  {
    if constexpr (CameraInfoV.encoding == CameraTypes::Encoding::BGR8)
    {
      return "CameraTypes::Encoding::BGR8";
    }
    else if constexpr (CameraInfoV.encoding == CameraTypes::Encoding::RGB8)
    {
      return "CameraTypes::Encoding::RGB8";
    }
    else if constexpr (CameraInfoV.encoding == CameraTypes::Encoding::BGRA8)
    {
      return "CameraTypes::Encoding::BGRA8";
    }
    else if constexpr (CameraInfoV.encoding == CameraTypes::Encoding::RGBA8)
    {
      return "CameraTypes::Encoding::RGBA8";
    }
    else if constexpr (CameraInfoV.encoding == CameraTypes::Encoding::MONO8)
    {
      return "CameraTypes::Encoding::MONO8";
    }
    else
    {
      return "CameraTypes::Encoding::INVALID";
    }
  }

  /**
   * @brief 向 xrobot.yaml 片段追加一个 double 列表字段。
   */
  static void AppendYamlDoubleList(std::ostringstream& out, std::string_view key,
                                   const double* values, std::size_t count)
  {
    out << "      " << key << ":\n";
    for (std::size_t i = 0; i < count; ++i)
    {
      out << "        - " << values[i] << "\n";
    }
  }

  /**
   * @brief 生成 xrobot.yaml 中 `constexprs.MainCameraInfo` 的可粘贴片段。
   */
  static std::string MakeXRobotCameraInfoYaml(const cv::Mat& camera_matrix,
                                              const cv::Mat& distortion)
  {
    const std::array<double, 9> camera_values =
        CameraMatrixValues(camera_matrix);
    const std::array<double, 14> distortion_values =
        DistortionValues(distortion);
    const std::array<double, 9> rectification_values{
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0};
    const std::array<double, 12> projection_values =
        ProjectionValues(camera_matrix);

    std::ostringstream out;
    out << std::setprecision(17);
    out << "constexprs:\n";
    out << "  MainCameraInfo:\n";
    out << "    type: CameraTypes::CameraInfo\n";
    out << "    value:\n";
    out << "      width: " << static_cast<unsigned>(CameraInfoV.width) << "\n";
    out << "      height: " << static_cast<unsigned>(CameraInfoV.height) << "\n";
    out << "      step: " << static_cast<unsigned>(CameraInfoV.step) << "\n";
    out << "      encoding: " << EncodingName() << "\n";
    AppendYamlDoubleList(out, "camera_matrix", camera_values.data(),
                         camera_values.size());
    out << "      distortion_model: CameraTypes::DistortionModel::PLUMB_BOB\n";
    AppendYamlDoubleList(out, "distortion_coefficients",
                         distortion_values.data(), distortion_values.size());
    AppendYamlDoubleList(out, "rectification_matrix",
                         rectification_values.data(), rectification_values.size());
    AppendYamlDoubleList(out, "projection_matrix",
                         projection_values.data(), projection_values.size());
    return out.str();
  }

  /**
   * @brief 写出可粘贴到 xrobot.yaml 的 CameraInfo 片段。
   */
  static void WriteCameraInfoSnippet(const std::filesystem::path& path,
                                     const std::string& xrobot_yaml_snippet)
  {
    std::ofstream out(path);
    out << xrobot_yaml_snippet;
  }

  /**
   * @brief 标定成功后直接把 xrobot.yaml 片段打印到 stdout。
   */
  static void PrintXRobotYamlSnippet(const std::string& xrobot_yaml_snippet)
  {
    std::fputs("\n===== xrobot.yaml CameraInfo begin =====\n", stdout);
    std::fputs(xrobot_yaml_snippet.c_str(), stdout);
    std::fputs("===== xrobot.yaml CameraInfo end =====\n", stdout);
    std::fflush(stdout);
  }

  /**
   * @brief 标定成功后直接把样本覆盖和重投影质量报告打印到 stdout。
   */
  static void PrintQualityReport(const std::string& quality_report)
  {
    std::fputc('\n', stdout);
    std::fputs(quality_report.c_str(), stdout);
    std::fflush(stdout);
  }

  /**
   * @brief 保存已接受视角的 debug 图。
   *
   * 图中绘制检测到的 marker 和该视角质量指标；写图失败只报警，不影响标定结果。
   */
  static void SaveDebugImage(const cv::Mat& image, const Detection& detection,
                             const StoredView& stored)
  {
    cv::Mat debug = image.clone();
    if (!detection.marker_ids.empty())
    {
      // 保留 OpenCV 原始角点绘制，方便人工复盘 id 和角点方向是否正确。
      cv::aruco::drawDetectedMarkers(debug, detection.marker_corners,
                                     detection.marker_ids);
    }

    std::ostringstream label;
    label << "view=" << stored.view_number
          << " markers=" << stored.used_markers
          << " H=" << std::fixed << std::setprecision(3)
          << stored.homography_rms
          << " S=" << std::setprecision(1) << stored.sharpness_score;
    cv::putText(debug, label.str(), {20, 45}, cv::FONT_HERSHEY_SIMPLEX,
                1.0, {0, 255, 0}, 2, cv::LINE_AA);

    // 文件名同时包含视角序号和原始帧号，方便和 views.csv 对齐。
    std::ostringstream name;
    name << "view_" << std::setw(4) << std::setfill('0') << stored.view_number
         << "_frame_" << stored.frame_index << ".jpg";
    const std::filesystem::path path =
        std::filesystem::path(stored.debug_dir) / name.str();

    try
    {
      cv::imwrite(path.string(), debug);
    }
    catch (const cv::Exception& e)
    {
      XR_LOG_WARN("camera calibration: write debug image failed %s: %s",
                  path.string().c_str(), e.what());
    }
  }

  /// 保护配置、采样池、输出路径和统计计数；OpenCV 重活不在锁内执行。
  mutable std::mutex mutex_;
  /// VisionCapture 处理线程的快速活动标记，用于未激活时无锁直接放行。
  std::atomic<bool> active_fast_{false};

  /// 锁内权威活动状态；true 时 VisionCapture 相机标定流程处于采样状态。
  bool active_{false};
  /// 最近一次 SaveAndStop() 是否成功写出了标定文件。
  bool finished_{false};
  /// 不支持像素编码错误是否已经记录过。
  bool unsupported_encoding_logged_{false};
  /// 达到建议视角数的提示是否已经记录过。
  bool recommended_views_logged_{false};

  /// 当前标定轮次配置。
  Config config_{};
  /// 当前 GShang 标定板 marker id 到三维角点的映射。
  BoardMap board_{};
  /// 当前 ArUco 字典实例。
  cv::aruco::Dictionary dictionary_{};
  /// 当前 ArUco 检测参数实例。
  cv::aruco::DetectorParameters detector_params_{};
  /// 已通过在线过滤并等待保存求解的点集；完整 debug 图在接受时立即写入磁盘。
  std::vector<View> accepted_views_{};

  /// 本轮标定输出根目录。
  std::string output_dir_{};
  /// 已接受视角 debug 图目录。
  std::string debug_dir_{};
  /// 最近一次成功保存的 calibration.yml 路径。
  std::string last_saved_yaml_{};

  /// 最近一次成功标定的全局 RMS；负数表示尚无结果。
  double last_rms_{-1.0};
  /// 本轮采样中已见到的最佳清晰度分数。
  double best_sharpness_score_{0.0};
  /// 标定激活期间被送入 ProcessFrame() 的原始帧计数。
  uint64_t raw_frame_index_{0};
  /// 标定激活期间被吞掉、未发布给 CameraFrameSync 的帧计数。
  uint64_t swallowed_frames_{0};
  /// 实际进入 OpenCV 检测的帧计数。
  uint64_t processed_frames_{0};
  /// detectMarkers 找到至少一个本板 marker 的帧计数。
  uint64_t detected_frames_{0};
  /// 因清晰度低被拒绝的候选视角计数。
  uint64_t sharpness_rejected_frames_{0};
  /// 因中心/尺度/角度过近被拒绝的重复视角计数。
  uint64_t duplicate_rejected_frames_{0};
  /// 最近一次接受视角的时间戳，单位 us。
  uint64_t last_accept_timestamp_us_{0};
};
