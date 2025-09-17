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
#include <interface/TensegrityBars.h>

#include <estimation/bar_utilities.hpp>

namespace perception
{
template <class Base>
class tensegrity_filter_nodelet_t : public Base
{
  using Derived = tensegrity_filter_nodelet_t<Base>;

public:
  tensegrity_filter_nodelet_t() : _idx(0)
  {
  }

private:
  virtual void onInit()
  {
    ros::NodeHandle& private_nh{ Base::getPrivateNodeHandle() };

    _status = std::make_shared<interface::node_status_t>(private_nh);
    _status->status(interface::NodeStatus::PREPARING);

    dynamic_reconfigure::Server<perception::ColorFilterConfig>::CallbackType f;

    bool& publish_rgb{ _publish_rgb };
    std::string subscriber_topic;
    std::string publisher_topic;
    std::string& encoding{ _encoding };

    std::vector<int>& kernel_sizes{ _kernel_sizes };
    std::string& operations{ _operations };

    std::string filters_namespace;
    // /camera/filters/black/high
    PARAM_SETUP(private_nh, subscriber_topic);
    PARAM_SETUP(private_nh, publisher_topic);
    PARAM_SETUP(private_nh, encoding);
    PARAM_SETUP(private_nh, filters_namespace);
    PARAM_SETUP(private_nh, kernel_sizes);
    PARAM_SETUP(private_nh, operations);
    // PARAM_SETUP_WITH_DEFAULT(private_nh, publish_rgb, publish_rgb);

    set_hsv_values(filters_namespace + "/black", low_black, high_black);
    set_hsv_values(filters_namespace + "/red", low_red, high_red);
    set_hsv_values(filters_namespace + "/green", low_green, high_green);
    set_hsv_values(filters_namespace + "/blue", low_blue, high_blue);

    for (int i = 0; i < kernel_sizes.size(); ++i)
    {
      const int erosion_size{ kernel_sizes[i] };
      _elements.push_back(cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                    cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                                                    cv::Point(erosion_size, erosion_size)));
    }

    _frame_publisher = private_nh.advertise<sensor_msgs::Image>(publisher_topic, 1, true);
    _image_subscriber = private_nh.subscribe(subscriber_topic, 1, &Derived::image_callback, this);
    _tensegrity_pose_subscriber =
        private_nh.subscribe(tensegrity_pose_topic, 1, &Derived::tensegrity_pose_callback, this);

    _status->status(interface::NodeStatus::READY);
    _status->status(interface::NodeStatus::RUNNING);
  }

  void tensegrity_pose_callback(const interface::TensegrityBarsConstPtr msg)
  {
    using tensegrity::utils::convert_to;
    const gtsam::Pose3 pose_red{ convert_to<gtsam::Pose3>(msg->bar_red) };
    const gtsam::Pose3 pose_green{ convert_to<gtsam::Pose3>(msg->bar_green) };
    const gtsam::Pose3 pose_blue{ convert_to<gtsam::Pose3>(msg->bar_blue) };

    _btw_red_green = gtsam::traits<gtsam::Pose3>::Between(pose_red, pose_green);
    _btw_red_blue = gtsam::traits<gtsam::Pose3>::Between(pose_red, pose_blue);
    _btw_green_blue = gtsam::traits<gtsam::Pose3>::Between(pose_green, pose_blue);
  }

  void build_fg()
  {
    const gtsam::Key key_Xr{ estimation::rod_symbol(estimation::RodColors::RED, _idx) };
    const gtsam::Key key_Xg{ estimation::rod_symbol(estimation::RodColors::GREEN, _idx) };
    const gtsam::Key key_Xb{ estimation::rod_symbol(estimation::RodColors::BLUE, _idx) };

    graph.emplace_shared<gtsam::BetweenFactor>(key_Xr, key_Xg)
  }

  void set_hsv_values(const std::string filters_namespace, cv::Scalar& low_s, cv::Scalar& high_s)
  {
    ros::NodeHandle nh_param(filters_namespace);
    DEBUG_VARS(filters_namespace)
    std::vector<double> low;
    std::vector<double> high;
    PARAM_SETUP(nh_param, low);
    PARAM_SETUP(nh_param, high);

    low_s = cv::Scalar(low[0], low[1], low[2]);
    high_s = cv::Scalar(high[0], high[1], high[2]);
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
    // frame = cv_bridge::toCvCopy(message);
    // Convert from BGR to HSV colorspace
    cv::cvtColor(frame->image, _frame_HSV, cv::COLOR_BGR2HSV);

    cv::inRange(_frame_HSV, low_black, high_black, _frame_black);
    cv::inRange(_frame_HSV, low_red, high_red, _frame_red);
    cv::inRange(_frame_HSV, low_green, high_green, _frame_green);
    cv::inRange(_frame_HSV, low_blue, high_blue, _frame_blue);

    _mask = _frame_black + _frame_red + _frame_green + _frame_blue;
    erode_or_dilate(0, _mask, _mask);

    _frame_out.release();
    cv::copyTo(frame->image, _frame_out, _mask);

    _msg = cv_bridge::CvImage(message->header, _encoding, _frame_out).toImageMsg();

    _msg->header.stamp = ros::Time::now();
    _msg->header.seq = message->header.seq;

    _frame_publisher.publish(_msg);
  }

  // using namespace cv;
  const int max_value_H = 360 / 2;
  const int max_value = 255;

  int _idx;
  bool _publish_rgb;

  std::vector<int> _kernel_sizes;
  std::string _operations;
  std::vector<cv::Mat> _elements;

  cv::Scalar low_black, high_black;
  cv::Scalar low_red, high_red;
  cv::Scalar low_green, high_green;
  cv::Scalar low_blue, high_blue;
  std::string _encoding;

  cv::Mat _frame_HSV, _frame_threshold, _frame_rgb;
  cv::Mat _frame_black, _frame_red, _frame_green, _frame_blue;
  cv::Mat _frame_out, _mask;

  sensor_msgs::ImagePtr _msg, _msg_rgb;

  ros::Subscriber _image_subscriber, _tensegrity_pose_subscriber;
  ros::Publisher _frame_publisher, _frame_rgb_publisher;

  std::shared_ptr<interface::node_status_t> _status;

  gtsam::Pose3 _btw_red_green _btw_red_blue _btw_green_blue;
};

}  // namespace perception
