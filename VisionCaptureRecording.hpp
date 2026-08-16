#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <ostream>
#include <string>
#include <string_view>

#include "CameraBase.hpp"

namespace VisionCaptureRecording
{
/**
 * @brief 完整写出文本文件并执行逐字节读回校验。
 */
inline bool WriteTextFile(const std::filesystem::path& path, std::string_view content)
{
  if (content.size() >
      static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))
  {
    return false;
  }
  std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
  if (!out.is_open())
  {
    return false;
  }
  out.write(content.data(), static_cast<std::streamsize>(content.size()));
  out.flush();
  const bool write_ok = static_cast<bool>(out);
  out.close();
  if (!write_ok || out.fail())
  {
    return false;
  }

  std::ifstream in(path, std::ios::in | std::ios::binary);
  if (!in.is_open())
  {
    return false;
  }
  std::string readback(content.size(), '\0');
  in.read(readback.data(), static_cast<std::streamsize>(readback.size()));
  const bool exact_size = in.gcount() == static_cast<std::streamsize>(readback.size()) &&
                          in.peek() == std::char_traits<char>::eof();
  return exact_size && std::string_view(readback) == content;
}

/**
 * @brief 写入一个符合 RFC 4180 转义规则的 CSV 单元格。
 */
inline void WriteCsvCell(std::ostream& out, std::string_view value)
{
  const bool quote = value.find_first_of(",\"\r\n") != std::string_view::npos;
  if (!quote)
  {
    out << value;
    return;
  }

  out << '"';
  for (const char ch : value)
  {
    if (ch == '"')
    {
      out << '"';
    }
    out << ch;
  }
  out << '"';
}

/**
 * @brief 写入逐帧采样几何 CSV 表头。
 */
inline void WriteFrameGeometryHeader(std::ostream& out)
{
  out << "frame_id,image_path,width,height,step,roi_offset_x_native,"
         "roi_offset_y_native,decimation_x,decimation_y,flags,reserved,"
         "sample_phase_x_native,sample_phase_y_native\n";
}

/**
 * @brief 写入一帧完整采样几何。
 */
inline void WriteFrameGeometryRow(std::ostream& out, uint64_t frame_id,
                                  std::string_view image_path,
                                  const CameraTypes::FrameGeometry& geometry)
{
  out << frame_id << ',';
  WriteCsvCell(out, image_path);
  out << ',' << geometry.width << ',' << geometry.height << ',' << geometry.step << ','
      << geometry.roi_offset_x_native << ',' << geometry.roi_offset_y_native << ','
      << geometry.decimation_x << ',' << geometry.decimation_y << ',' << geometry.flags
      << ',' << geometry.reserved << ',';
  const std::streamsize previous_precision = out.precision();
  out << std::setprecision(std::numeric_limits<float>::max_digits10)
      << geometry.sample_phase_x_native << ',' << geometry.sample_phase_y_native;
  out.precision(previous_precision);
  out << '\n';
}

/**
 * @brief 按配置写入或留空十个原始 IMU 数值单元格。
 */
inline void WriteRawImuCells(std::ostream& out, bool enabled,
                             const std::array<float, 4>& rotation_wxyz,
                             const std::array<float, 3>& angular_velocity_xyz,
                             const std::array<float, 3>& linear_acceleration_xyz)
{
  if (!enabled)
  {
    out << ",,,,,,,,,,";
    return;
  }

  for (const float value : rotation_wxyz)
  {
    out << ',' << value;
  }
  for (const float value : angular_velocity_xyz)
  {
    out << ',' << value;
  }
  for (const float value : linear_acceleration_xyz)
  {
    out << ',' << value;
  }
}

/**
 * @brief 判断当前成功记录行是否需要主动刷盘；0 表示仅在流关闭时刷盘。
 */
inline bool ShouldFlush(uint64_t rows_written, uint32_t flush_every_n)
{
  return flush_every_n != 0U && rows_written != 0U && rows_written % flush_every_n == 0U;
}

/**
 * @brief 标定视角只有在本帧所需记录完整持久化后才能提交给求解器。
 */
inline bool CalibrationViewMayCommit(bool sample_accepted, bool record_required,
                                     bool record_completed, bool io_failed)
{
  return sample_accepted && !io_failed && (!record_required || record_completed);
}
}  // namespace VisionCaptureRecording
