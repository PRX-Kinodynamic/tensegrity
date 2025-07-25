#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <tensegrity_utils/rosparams_utils.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>

namespace perception
{
template <class Base>
class dilation_nodelet_t : public Base
{
  using Derived = dilation_nodelet_t<Base>;

public:
  dilation_nodelet_t() : _publish_rgb(false), _operations("")
  {
  }

private:
  virtual void onInit()
  {
    ros::NodeHandle& private_nh{ Base::getPrivateNodeHandle() };

    std::string subscriber_topic;
    std::string publisher_topic;

    std::vector<int>& kernel_sizes{ _kernel_sizes };
    std::string& operations{ _operations };
    bool& publish_rgb{ _publish_rgb };
    std::string& encoding{ _encoding };

    PARAM_SETUP(private_nh, subscriber_topic);
    PARAM_SETUP(private_nh, publisher_topic);
    PARAM_SETUP(private_nh, kernel_sizes);
    PARAM_SETUP(private_nh, operations);
    PARAM_SETUP(private_nh, encoding);
    PARAM_SETUP_WITH_DEFAULT(private_nh, publish_rgb, publish_rgb);

    _image_subscriber = private_nh.subscribe(subscriber_topic, 1, &Derived::image_callback, this);
    _frame_publisher = private_nh.advertise<sensor_msgs::Image>(publisher_topic, 1, true);

    // ROS_ASSERT(kernel_sizes.size() == operations.size(), "kernel_sizes and operations must match");
    for (int i = 0; i < kernel_sizes.size(); ++i)
    {
      const int erosion_size{ kernel_sizes[i] };
      _elements.push_back(cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                    cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                                                    cv::Point(erosion_size, erosion_size)));
    }

    if (_publish_rgb)
      _frame_rgb_publisher = private_nh.advertise<sensor_msgs::Image>(publisher_topic + "/rgb", 1, true);
  }

  void erode_or_dilate(const int idx, const cv::Mat& image_in, cv::Mat& image_out)
  {
    if (idx >= _operations.size())
      return;

    const char oper{ _operations[idx] };
    const cv::Mat& element{ _elements[idx] };
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
        // std::err << "Option " << oper << " not available";
        // exit(-1);
        // prx_throw("Option " << oper << " not available");
    }
    erode_or_dilate(idx + 1, image_out, image_out);
  }

  void image_callback(const sensor_msgs::ImageConstPtr message)
  {
    cv_bridge::CvImageConstPtr frame{ cv_bridge::toCvShare(message) };

    // cv::dilate(frame->image, _dilation_dst, _element_erosion);
    erode_or_dilate(0, frame->image, _dilation_dst);

    _msg = cv_bridge::CvImage(std_msgs::Header(), _encoding, _dilation_dst).toImageMsg();
    _frame_publisher.publish(_msg);

    if (_publish_rgb)
    {
      cv::cvtColor(_dilation_dst, _frame_rgb, cv::COLOR_GRAY2RGB);
      _msg_rgb = cv_bridge::CvImage(std_msgs::Header(), "rgb8", _frame_rgb).toImageMsg();
      _frame_rgb_publisher.publish(_msg_rgb);
    }
  }

  bool _publish_rgb;
  std::string _encoding;

  std::vector<int> _kernel_sizes;
  std::string _operations;

  std::vector<cv::Mat> _elements;
  cv::Mat _dilation_dst, _frame_rgb;

  sensor_msgs::ImagePtr _msg, _msg_rgb;

  ros::Subscriber _image_subscriber;
  ros::Publisher _frame_publisher, _frame_rgb_publisher;
};

}  // namespace perception
