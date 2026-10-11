// 录像：逐帧 PGM 与 frames.csv 的内容、session.txt、帧率上限与帧数上限。
//
// Recorder: per-frame PGM and frames.csv content, session.txt, the frame rate cap and
// the frame limit.
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "VisionRecorder.hpp"
#include "libxr.hpp"

namespace
{
void Expect(bool condition, const char* message)
{
  if (!condition)
  {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
  }
}

const CameraTypes::CameraCalibration CALIBRATION{
    1440, 1080, 2340.5, 2338.25, 715.5, 530.25, {-0.1, 0.4, 0, 0, 0}};

std::string ReadFile(const std::filesystem::path& path)
{
  std::ifstream f(path, std::ios::binary);
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}

std::vector<std::string> Lines(const std::string& text)
{
  std::vector<std::string> lines;
  std::stringstream s(text);
  for (std::string line; std::getline(s, line);)
  {
    lines.push_back(line);
  }
  return lines;
}

/// 以 period_us 为间隔发布 n 帧，然后析构录像（写完缓冲）/ Publish n frames, then
/// destroy the recorder, which drains its buffer.
std::filesystem::path Record(const char* camera, const std::filesystem::path& parent,
                             double max_fps, uint32_t max_frames, int n,
                             uint64_t period_us)
{
  const std::string topic_name = std::string(camera) + "_synced";
  LibXR::Topic topic =
      LibXR::Topic::CreateTopic<const AutoAim::SyncedFrame*>(topic_name.c_str());
  const std::string parent_text = parent.string();
  auto* recorder =
      new VisionRecorder({camera, true, parent_text.c_str(), max_fps, max_frames, 64});
  const std::filesystem::path dir = recorder->Directory();
  ImagePool pool(2);
  for (int i = 0; i < n; ++i)
  {
    ImagePool::Handle h;
    Expect(pool.Acquire(h) == LibXR::ErrorCode::OK, "image slot");
    h->timestamp_us = LibXR::MicrosecondTimestamp(1000000 + period_us * i);
    h->geometry = {80, 24, 2};
    h->frame_counter = static_cast<uint32_t>(100 + i);
    h->calibration = &CALIBRATION;
    for (std::size_t k = 0; k < h->data.size(); ++k)
    {
      h->data[k] = static_cast<uint8_t>((k * 7 + i) & 0xFF);
    }
    AutoAim::SyncedFrame frame{
        static_cast<uint64_t>(i + 1), SharedFrame(std::move(h)), {}};
    frame.imu.rotation_wxyz = {0.5F, 0.5F, -0.5F, 0.5F};
    frame.imu.angular_velocity_xyz = {0.25F, 0.0F, -1.5F};
    frame.imu.linear_acceleration_xyz = {0.0F, 0.0F, 9.75F};
    const AutoAim::SyncedFrame* payload = &frame;
    topic.Publish(payload);
  }
  // 析构前没有新的发布，回调不会再被调用 / No more publishes, so the callback stays idle.
  delete recorder;
  return dir;
}

void TestContent(const std::filesystem::path& parent)
{
  const auto dir = Record("rec", parent, 0.0, 0, 20, 10000);
  const auto lines = Lines(ReadFile(dir / "frames.csv"));
  Expect(lines.size() == 21, "header plus 20 rows");
  Expect(lines[0] ==
             "frame,timestamp_us,frame_counter,roi_x,roi_y,decimation,"
             "qw,qx,qy,qz,gx,gy,gz,ax,ay,az",
         "header");
  Expect(lines[1] == "0,1000000,100,80,24,2,0.5,0.5,-0.5,0.5,0.25,0,-1.5,0,0,9.75",
         "first row");
  Expect(lines[20].rfind("19,1190000,119,", 0) == 0, "last row");
  const std::string pgm = ReadFile(dir / "000003.pgm");
  const std::string header = "P5\n640 512\n255\n";
  Expect(pgm.size() == header.size() + CameraTypes::FRAME_BYTES &&
             pgm.compare(0, header.size(), header) == 0,
         "PGM header and size");
  Expect(static_cast<uint8_t>(pgm[header.size() + 10]) == ((10 * 7 + 3) & 0xFF),
         "PGM pixels");
  const std::string session = ReadFile(dir / "session.txt");
  Expect(session.find("camera rec\n") != std::string::npos &&
             session.find("fx 2340.500000\n") != std::string::npos &&
             session.find("distortion -0.1 0.4 0 0 0\n") != std::string::npos,
         "session.txt");
}

void TestLimits(const std::filesystem::path& parent)
{
  // 100 Hz 输入、50 fps 上限：隔帧保存 / 100 Hz in, 50 fps cap: every other frame.
  const auto capped = Record("cap", parent / "cap", 50.0, 0, 20, 10000);
  Expect(Lines(ReadFile(capped / "frames.csv")).size() == 11,
         "50 fps cap keeps 10 of 20");
  const auto limited = Record("lim", parent / "lim", 0.0, 5, 20, 10000);
  Expect(Lines(ReadFile(limited / "frames.csv")).size() == 6, "frame limit 5");
  Expect(std::filesystem::exists(limited / "000004.pgm") &&
             !std::filesystem::exists(limited / "000005.pgm"),
         "files 0..4");
}
}  // namespace

int main()
{
  LibXR::PlatformInit();
  const auto parent = std::filesystem::temp_directory_path() / "vision_recorder_test";
  std::filesystem::remove_all(parent);
  // 同一秒启动的录像目录名相同，所以各用一个父目录 / Recordings started within one
  // second share a name, so each gets its own parent.
  TestContent(parent / "a");
  TestLimits(parent / "b");
  std::filesystem::remove_all(parent);
  std::puts("vision_recorder_test passed");
  std::fflush(stdout);
  std::_Exit(EXIT_SUCCESS);
}
