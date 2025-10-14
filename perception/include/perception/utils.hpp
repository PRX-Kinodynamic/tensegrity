#pragma once

namespace perception
{
void erode_or_dilate(const int idx, const cv::Mat& image_in, cv::Mat& image_out, const std::string operations,
                     const std::vector<cv::Mat> elements)
{
  if (idx >= operations.size())
    return;

  const char oper{ operations[idx] };
  const cv::Mat& element{ elements[idx] };
  switch (oper)
  {
    case 'D':
      cv::dilate(image_in, image_out, element);
      break;
    case 'E':
      cv::erode(image_in, image_out, element);
      break;
    default:
      ROS_ERROR_STREAM("Option " << oper << " not available");
  }
  erode_or_dilate(idx + 1, image_out, image_out, operations, elements);
}
}  // namespace perception