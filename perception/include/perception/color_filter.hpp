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
class color_filter_t : public Base
{
  using Derived = color_filter_t<Base>;

public:
  color_filter_t()
    : _low_H(0), _low_S(0), _low_V(0), _high_H(max_value_H), _high_S(max_value), _high_V(max_value), _publish_rgb(false)
  {
  }

private:
  virtual void onInit()
  {
    ros::NodeHandle& private_nh{ Base::getPrivateNodeHandle() };

    int& low_H{ _low_H };
    int& low_S{ _low_S };
    int& low_V{ _low_V };
    int& high_H{ _high_H };
    int& high_S{ _high_S };
    int& high_V{ _high_V };

    bool& publish_rgb{ _publish_rgb };
    std::string subscriber_topic;
    std::string publisher_topic;
    std::string& encoding{ _encoding };

    PARAM_SETUP(private_nh, subscriber_topic);
    PARAM_SETUP(private_nh, publisher_topic);
    PARAM_SETUP(private_nh, low_H);
    PARAM_SETUP(private_nh, low_S);
    PARAM_SETUP(private_nh, low_V);
    PARAM_SETUP(private_nh, high_H);
    PARAM_SETUP(private_nh, high_S);
    PARAM_SETUP(private_nh, high_V);
    PARAM_SETUP(private_nh, encoding);
    PARAM_SETUP_WITH_DEFAULT(private_nh, publish_rgb, publish_rgb);

    _image_subscriber = private_nh.subscribe(subscriber_topic, 1, &Derived::image_callback, this);
    _frame_publisher = private_nh.advertise<sensor_msgs::Image>(publisher_topic, 1, true);

    if (_publish_rgb)
      _frame_rgb_publisher = private_nh.advertise<sensor_msgs::Image>(publisher_topic + "/rgb", 1, true);
  }

  void image_callback(const sensor_msgs::ImageConstPtr message)
  {
    cv_bridge::CvImageConstPtr frame{ cv_bridge::toCvShare(message) };
    // frame = cv_bridge::toCvCopy(message);
    // Convert from BGR to HSV colorspace
    cv::cvtColor(frame->image, _frame_HSV, cv::COLOR_BGR2HSV);

    cv::inRange(_frame_HSV, cv::Scalar(_low_H, _low_S, _low_V), cv::Scalar(_high_H, _high_S, _high_V),
                _frame_threshold);
    _msg = cv_bridge::CvImage(std_msgs::Header(), _encoding, _frame_threshold).toImageMsg();
    _frame_publisher.publish(_msg);

    if (_publish_rgb)
    {
      cv::cvtColor(_frame_threshold, _frame_rgb, cv::COLOR_GRAY2RGB);
      _msg_rgb = cv_bridge::CvImage(std_msgs::Header(), "rgb8", _frame_rgb).toImageMsg();
      _frame_rgb_publisher.publish(_msg_rgb);
    }
  }
  // using namespace cv;
  const int max_value_H = 360 / 2;
  const int max_value = 255;

  bool _publish_rgb;
  int _low_H, _low_S, _low_V;
  int _high_H, _high_S, _high_V;

  std::string _encoding;

  cv::Mat _frame_HSV, _frame_threshold, _frame_rgb;

  sensor_msgs::ImagePtr _msg, _msg_rgb;

  ros::Subscriber _image_subscriber;
  ros::Publisher _frame_publisher, _frame_rgb_publisher;
};

}  // namespace perception
