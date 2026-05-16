#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <vector>

#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace VisionCaptureCalibrationBoard
{
using BoardMap = std::map<int, std::array<cv::Point3f, 4>>;

struct Observation
{
  bool observed = false;
  int used_markers = 0;
  double homography_rms = 0.0;
  double sharpness_score = 0.0;
  double center_x_norm = 0.0;
  double center_y_norm = 0.0;
  double scale_norm = 0.0;
  double angle_deg = 0.0;
  std::vector<cv::Point3f> object_points;
  std::vector<cv::Point2f> image_points;
  std::vector<std::vector<cv::Point2f>> marker_corners;
  std::vector<int> marker_ids_vec;
  cv::Mat marker_ids;
};

struct PoseEstimate
{
  bool ok = false;
  double reprojection_rms_px = 0.0;
  cv::Mat rvec;
  cv::Mat tvec;
};

inline double ClampUnit(double value)
{
  return std::max(-1.0, std::min(1.0, value));
}

inline double NormalizeBoardAngle(double angle_deg, const cv::Size2f& size)
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

inline double AngleDeltaDeg(double lhs, double rhs)
{
  double delta = std::fabs(lhs - rhs);
  while (delta >= 180.0)
  {
    delta -= 180.0;
  }
  return delta > 90.0 ? 180.0 - delta : delta;
}

inline cv::Mat MakeGrayImage(const cv::Mat& image)
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

inline double HomographyRms(const std::vector<cv::Point3f>& object_points,
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

inline double MarkerSharpnessScore(
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

  cv::dilate(marker_mask, marker_mask, cv::Mat{}, {-1, -1}, 1);
  cv::Mat laplacian;
  cv::Laplacian(gray, laplacian, CV_64F, 3);

  cv::Scalar mean;
  cv::Scalar stddev;
  cv::meanStdDev(laplacian, mean, stddev, marker_mask);
  return stddev[0] * stddev[0];
}

inline void FillQuality(const cv::Mat& image, uint32_t width, uint32_t height,
                        Observation& observation)
{
  observation.sharpness_score =
      MarkerSharpnessScore(image, observation.marker_corners);

  if (observation.image_points.empty() || width == 0 || height == 0)
  {
    return;
  }

  const cv::Rect bounds = cv::boundingRect(observation.image_points);
  const double image_width = static_cast<double>(width);
  const double image_height = static_cast<double>(height);
  observation.center_x_norm =
      (static_cast<double>(bounds.x) + static_cast<double>(bounds.width) * 0.5) /
      image_width;
  observation.center_y_norm =
      (static_cast<double>(bounds.y) + static_cast<double>(bounds.height) * 0.5) /
      image_height;
  observation.scale_norm =
      std::sqrt(std::max(0.0, static_cast<double>(bounds.area())) /
                std::max(1.0, image_width * image_height));

  if (observation.image_points.size() >= 4)
  {
    const cv::RotatedRect rect = cv::minAreaRect(observation.image_points);
    observation.angle_deg = NormalizeBoardAngle(rect.angle, rect.size);
  }
}

inline bool CollectBoardPoints(
    const std::vector<std::vector<cv::Point2f>>& corners, const cv::Mat& ids,
    const BoardMap& board, Observation& observation)
{
  observation.object_points.clear();
  observation.image_points.clear();
  observation.used_markers = 0;
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
    if (marker == board.end())
    {
      continue;
    }

    for (int corner = 0; corner < 4; ++corner)
    {
      observation.object_points.push_back(marker->second[corner]);
      observation.image_points.push_back(corners[i][corner]);
    }
    ++observation.used_markers;
  }
  observation.observed = observation.used_markers > 0;
  return observation.observed;
}

inline BoardMap MakeSingleArucoBoard(double marker_length_m)
{
  const float half = static_cast<float>(marker_length_m * 0.5);
  BoardMap board;
  board[0] = {cv::Point3f{-half, half, 0.0F},
              cv::Point3f{half, half, 0.0F},
              cv::Point3f{half, -half, 0.0F},
              cv::Point3f{-half, -half, 0.0F}};
  return board;
}

inline BoardMap MakeGShangBoard(double marker_mm, int cols, int rows,
                                int marker_cells = 7, int square_cells = 9)
{
  const double cell_mm = marker_mm / static_cast<double>(marker_cells);
  const double square_mm = cell_mm * static_cast<double>(square_cells);
  const double marker_offset_mm = cell_mm;

  BoardMap board;
  int marker_id = 0;
  for (int row = 0; row < rows; ++row)
  {
    for (int col = 0; col < cols; ++col)
    {
      if (((row + col) % 2) == 0)
      {
        continue;
      }
      const float x0 = static_cast<float>(col * square_mm + marker_offset_mm);
      const float y0 = static_cast<float>(row * square_mm + marker_offset_mm);
      const float x1 = x0 + static_cast<float>(marker_mm);
      const float y1 = y0 + static_cast<float>(marker_mm);
      board[marker_id++] = {cv::Point3f{x0, y0, 0.0F},
                            cv::Point3f{x1, y0, 0.0F},
                            cv::Point3f{x1, y1, 0.0F},
                            cv::Point3f{x0, y1, 0.0F}};
    }
  }
  return board;
}

inline double ReprojectionRms(const std::vector<cv::Point3f>& object_points,
                              const std::vector<cv::Point2f>& image_points,
                              const cv::Mat& rvec, const cv::Mat& tvec,
                              const cv::Mat& camera_matrix,
                              const cv::Mat& distortion)
{
  std::vector<cv::Point2f> projected;
  cv::projectPoints(object_points, rvec, tvec, camera_matrix, distortion, projected);
  double sum2 = 0.0;
  for (std::size_t i = 0; i < image_points.size(); ++i)
  {
    const cv::Point2f delta = projected[i] - image_points[i];
    sum2 += delta.dot(delta);
  }
  return std::sqrt(sum2 / std::max<std::size_t>(1, image_points.size()));
}

inline PoseEstimate EstimatePose(const Observation& observation,
                                 const cv::Mat& camera_matrix,
                                 const cv::Mat& distortion, int method)
{
  PoseEstimate result;
  if (!observation.observed || observation.object_points.size() < 4 ||
      observation.object_points.size() != observation.image_points.size())
  {
    return result;
  }
  if (!cv::solvePnP(observation.object_points, observation.image_points,
                    camera_matrix, distortion, result.rvec, result.tvec, false,
                    method))
  {
    return result;
  }
  result.ok = true;
  result.reprojection_rms_px =
      ReprojectionRms(observation.object_points, observation.image_points,
                      result.rvec, result.tvec, camera_matrix, distortion);
  return result;
}
}  // namespace VisionCaptureCalibrationBoard
