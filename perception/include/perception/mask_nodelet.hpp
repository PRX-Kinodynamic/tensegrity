#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <tensegrity_utils/rosparams_utils.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>

#include <message_filters/subscriber.h>
// #include <message_filters/time_synchronizer.h>

namespace perception
{
template <class Base>
class mask_nodelet_t : public Base
{
  using Derived = mask_nodelet_t<Base>;

public:
  mask_nodelet_t() : _publish_rgb(false), _mask_received(false)
  {
  }

private:
  virtual void onInit()
  {
    ros::NodeHandle& private_nh{ Base::getPrivateNodeHandle() };

    bool& publish_rgb{ _publish_rgb };
    std::string image_topic;
    std::string mask_topic;
    std::string publisher_topic;
    std::string& encoding{ _encoding };

    PARAM_SETUP(private_nh, image_topic);
    PARAM_SETUP(private_nh, mask_topic);
    PARAM_SETUP(private_nh, publisher_topic);
    PARAM_SETUP(private_nh, encoding);
    PARAM_SETUP_WITH_DEFAULT(private_nh, publish_rgb, publish_rgb);

    _image_subscriber = private_nh.subscribe(image_topic, 1, &Derived::image_callback, this);
    _mask_subscriber = private_nh.subscribe(mask_topic, 1, &Derived::mask_callback, this);

    _frame_publisher = private_nh.advertise<sensor_msgs::Image>(publisher_topic, 1, true);

    if (_publish_rgb)
      _frame_rgb_publisher = private_nh.advertise<sensor_msgs::Image>(publisher_topic + "/rgb", 1, true);
  }

  void mask_callback(const sensor_msgs::ImageConstPtr message)
  {
    cv_bridge::CvImageConstPtr frame{ cv_bridge::toCvShare(message) };
    _mask = frame->image;
    _mask_received = true;
  }

  void image_callback(const sensor_msgs::ImageConstPtr message)
  {
    if (not _mask_received)
      return;

    cv_bridge::CvImageConstPtr frame{ cv_bridge::toCvShare(message) };

    // frame->image.copyTo(_frame_out, _mask);
    cv::bitwise_and(frame->image, _mask, _frame_out);

    _msg = cv_bridge::CvImage(std_msgs::Header(), _encoding, _frame_out).toImageMsg();
    _frame_publisher.publish(_msg);

    if (_publish_rgb)
    {
      cv::cvtColor(_frame_out, _frame_rgb, cv::COLOR_GRAY2RGB);
      _msg_rgb = cv_bridge::CvImage(std_msgs::Header(), "rgb8", _frame_rgb).toImageMsg();
      _frame_rgb_publisher.publish(_msg_rgb);
    }
  }

  bool _mask_received;

  bool _publish_rgb;

  std::string _encoding;

  cv::Mat _frame_out, _mask, _frame_rgb;

  sensor_msgs::ImagePtr _msg, _msg_rgb;

  ros::Subscriber _image_subscriber, _mask_subscriber;
  ros::Publisher _frame_publisher, _frame_rgb_publisher;
};

}  // namespace perception
