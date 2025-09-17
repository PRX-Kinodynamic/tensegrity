#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <tensegrity_utils/rosparams_utils.hpp>
#include <tensegrity_utils/dbg_utils.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>

#include <dynamic_reconfigure/server.h>
#include <perception/ColorFilterConfig.h>

#include <interface/node_status.hpp>

namespace perception
{
template <class Base>
class color_filter_t : public Base
{
  using Derived = color_filter_t<Base>;
  using DynReconfServer = dynamic_reconfigure::Server<perception::ColorFilterConfig>;

public:
  color_filter_t()
    : _low_H(0), _low_S(0), _low_V(0), _high_H(max_value_H), _high_S(max_value), _high_V(max_value), _publish_rgb(false)
  {
  }

private:
  virtual void onInit()
  {
    ros::NodeHandle& private_nh{ Base::getPrivateNodeHandle() };

    _status = std::make_shared<interface::node_status_t>(private_nh);
    _status->status(interface::NodeStatus::PREPARING);

    dynamic_reconfigure::Server<perception::ColorFilterConfig>::CallbackType f;

    // int& low_H{ _low_H };
    // int& low_S{ _low_S };
    // int& low_V{ _low_V };
    // int& high_H{ _high_H };
    // int& high_S{ _high_S };
    // int& high_V{ _high_V };

    bool& publish_rgb{ _publish_rgb };
    std::string subscriber_topic;
    std::string publisher_topic;
    std::string& encoding{ _encoding };

    std::string filters_namespace;
    // /camera/filters/black/high
    PARAM_SETUP(private_nh, subscriber_topic);
    PARAM_SETUP(private_nh, publisher_topic);
    // PARAM_SETUP(private_nh, low_H);
    // PARAM_SETUP(private_nh, low_S);
    // PARAM_SETUP(private_nh, low_V);
    // PARAM_SETUP(private_nh, high_H);
    // PARAM_SETUP(private_nh, high_S);
    // PARAM_SETUP(private_nh, high_V);
    PARAM_SETUP(private_nh, encoding);
    PARAM_SETUP(private_nh, filters_namespace);
    PARAM_SETUP_WITH_DEFAULT(private_nh, publish_rgb, publish_rgb);

    // DEBUG_VARS(filters_namespace)
    ros::NodeHandle nh_param(filters_namespace);
    std::vector<double> low;
    std::vector<double> high;
    PARAM_SETUP(nh_param, low);
    PARAM_SETUP(nh_param, high);

    // DEBUG_VARS(low, high)
    _low_H = low[0];
    _low_S = low[1];
    _low_V = low[2];
    _high_H = high[0];
    _high_S = high[1];
    _high_V = high[2];

    const std::string name{ this->getName() };
    ros::NodeHandle nh1(private_nh, name);

    nh1.setParam("low_H", _low_H);
    nh1.setParam("low_S", _low_S);
    nh1.setParam("low_V", _low_V);
    nh1.setParam("high_H", _high_H);
    nh1.setParam("high_S", _high_S);
    nh1.setParam("high_V", _high_V);
    _server = std::make_shared<DynReconfServer>(nh1);
    // DEBUG_VARS(name);
    f = boost::bind(&Derived::cfg_callback, this, _1, _2);
    _server->setCallback(f);

    _frame_publisher = private_nh.advertise<sensor_msgs::Image>(publisher_topic, 1, true);
    _image_subscriber = private_nh.subscribe(subscriber_topic, 1, &Derived::image_callback, this);

    if (_publish_rgb)
      _frame_rgb_publisher = private_nh.advertise<sensor_msgs::Image>(publisher_topic + "/rgb", 1, true);

    _status->status(interface::NodeStatus::READY);
    _status->status(interface::NodeStatus::RUNNING);
  }

  void image_callback(const sensor_msgs::ImageConstPtr message)
  {
    cv_bridge::CvImageConstPtr frame{ cv_bridge::toCvShare(message) };
    // frame = cv_bridge::toCvCopy(message);
    // Convert from BGR to HSV colorspace
    cv::cvtColor(frame->image, _frame_HSV, cv::COLOR_BGR2HSV);

    cv::inRange(_frame_HSV, cv::Scalar(_low_H, _low_S, _low_V), cv::Scalar(_high_H, _high_S, _high_V),
                _frame_threshold);
    _msg = cv_bridge::CvImage(message->header, _encoding, _frame_threshold).toImageMsg();
    _msg->header.stamp = ros::Time::now();

    _frame_publisher.publish(_msg);

    if (_publish_rgb)
    {
      cv::cvtColor(_frame_threshold, _frame_rgb, cv::COLOR_GRAY2RGB);

      _msg_rgb = cv_bridge::CvImage(message->header, "rgb8", _frame_rgb).toImageMsg();
      _msg_rgb->header.stamp = ros::Time::now();

      _frame_rgb_publisher.publish(_msg_rgb);
    }
  }

  void cfg_callback(perception::ColorFilterConfig& config, uint32_t level)
  {
    _low_H = config.low_H;
    _low_S = config.low_S;
    _low_V = config.low_V;
    _high_H = config.high_H;
    _high_S = config.high_S;
    _high_V = config.high_V;

    DEBUG_VARS(_low_H, _low_S, _low_V);
    DEBUG_VARS(_high_H, _high_S, _high_V);
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

  std::shared_ptr<DynReconfServer> _server;

  std::shared_ptr<interface::node_status_t> _status;
};

}  // namespace perception
