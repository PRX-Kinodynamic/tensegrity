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
  mask_nodelet_t() : _publish_rgb(false), _mask_received(false), _operation("and")
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
    std::string& operation{ _operation };
    double frequency{ 30 };

    PARAM_SETUP(private_nh, image_topic);
    PARAM_SETUP(private_nh, mask_topic);
    PARAM_SETUP(private_nh, publisher_topic);
    PARAM_SETUP(private_nh, encoding);
    PARAM_SETUP_WITH_DEFAULT(private_nh, operation, operation);
    PARAM_SETUP_WITH_DEFAULT(private_nh, publish_rgb, publish_rgb);
    PARAM_SETUP_WITH_DEFAULT(private_nh, frequency, frequency);

    _frame_publisher = private_nh.advertise<sensor_msgs::Image>(publisher_topic, 1, true);
    if (_publish_rgb)
      _frame_rgb_publisher = private_nh.advertise<sensor_msgs::Image>(publisher_topic + "/rgb", 1, true);

    _image_subscriber = private_nh.subscribe(image_topic, 1, &Derived::image_callback, this);
    _mask_subscriber = private_nh.subscribe(mask_topic, 1, &Derived::mask_callback, this);
    const ros::Duration timer(1.0 / frequency);
    _timer = private_nh.createTimer(timer, &Derived::update, this);
  }

  void update(const ros::TimerEvent& event)
  {
    if (not _mask_received)
      return;
    if (not _img_received)
      return;

    if (_operation == "and")
    {
      cv::bitwise_and(_frame->image, _mask, _frame_out);
    }
    else if (_operation == "or")
    {
      // _frame_out = _frame->image;
      cv::bitwise_or(_frame->image, _mask, _frame_out);
    }
    else if (_operation == "copy")
    {
      _frame_out.release();
      cv::copyTo(_frame->image, _frame_out, _mask);
    }
    else
    {
      TENSEGRITY_THROW("Invalid operation: only 'and' or 'or'");
      ros::shutdown();
    }

    _msg = cv_bridge::CvImage(_frame->header, _encoding, _frame_out).toImageMsg();
    _msg->header.stamp = ros::Time::now();

    _frame_publisher.publish(_msg);

    if (_publish_rgb)
    {
      cv::cvtColor(_frame_out, _frame_rgb, cv::COLOR_GRAY2RGB);
      _msg_rgb = cv_bridge::CvImage(_frame->header, "rgb8", _frame_rgb).toImageMsg();
      _msg_rgb->header.stamp = ros::Time::now();

      _frame_rgb_publisher.publish(_msg_rgb);
    }
    _mask_received = false;
    _img_received = false;
  }

  void mask_callback(const sensor_msgs::ImageConstPtr message)
  {
    cv_bridge::CvImageConstPtr frame{ cv_bridge::toCvShare(message) };
    // cv::copyTo(_frame->image, _mask);
    // _mask = cv_bridge::toCvCopy(message);
    _mask = frame->image;
    _mask_received = true;
  }

  void image_callback(const sensor_msgs::ImageConstPtr message)
  {
    // if (not _mask_received)
    //   return;

    // cv_bridge::CvImageConstPtr frame{ cv_bridge::toCvCopy(message) };
    _frame = cv_bridge::toCvCopy(message);
    // frame->image.copyTo(_frame);
    _img_received = true;

    // if (_operation == "and")
    // {
    //   cv::bitwise_and(frame->image, _mask, _frame_out);
    // }
    // else if (_operation == "or")
    // {
    //   cv::bitwise_or(frame->image, _mask, _frame_out);
    // }
    // else
    // {
    //   TENSEGRITY_THROW("Invalid operation: only 'and' or 'or'");
    //   ros::shutdown();
    // }

    // _msg = cv_bridge::CvImage(message->header, _encoding, _frame_out).toImageMsg();
    // _msg->header.stamp = ros::Time::now();

    // _frame_publisher.publish(_msg);

    // if (_publish_rgb)
    // {
    //   cv::cvtColor(_frame_out, _frame_rgb, cv::COLOR_GRAY2RGB);
    //   _msg_rgb = cv_bridge::CvImage(message->header, "rgb8", _frame_rgb).toImageMsg();
    //   _msg_rgb->header.stamp = ros::Time::now();

    //   _frame_rgb_publisher.publish(_msg_rgb);
    // }
  }

  cv_bridge::CvImageConstPtr _frame;
  bool _mask_received, _img_received;

  bool _publish_rgb;

  std::string _encoding, _operation;

  cv::Mat _frame_out, _mask, _frame_rgb;

  sensor_msgs::ImagePtr _msg, _msg_rgb;

  ros::Timer _timer;
  ros::Subscriber _image_subscriber, _mask_subscriber;
  ros::Publisher _frame_publisher, _frame_rgb_publisher;
};

}  // namespace perception
