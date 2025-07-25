#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <Eigen/Dense>
#include <Eigen/Core>

#include <ros/console.h>
#include <ros/assert.h>
#include <tensegrity_utils/assert.hpp>
#include <tensegrity_utils/dbg_utils.hpp>
#include <tensegrity_utils/rosparams_utils.hpp>
#include <tensegrity_utils/std_utils.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <opencv2/core/fast_math.hpp>

#include <interface/TensegrityEndcaps.h>

namespace perception
{
template <class Base>
class endcap_position_detector_nodelet_t : public Base
{
  using Derived = endcap_position_detector_nodelet_t<Base>;

public:
  endcap_position_detector_nodelet_t()
    : _dp(1)
    , _min_dist(2)
    , _param1(100)
    , _param2(100)
    , _min_radius(0)
    , _max_radius(0)
    , _viz(false)
    , _depth_received(false)
    , _image_received(false)
    , _intrinsic(Eigen::Matrix4d::Identity())
    , _extrinsic(Eigen::Matrix4d::Identity())
    , _int_ext_inv(Eigen::Matrix4d::Identity())
    , _endcap_id(-1)
  {
  }

private:
  virtual void onInit()
  {
    ros::NodeHandle& private_nh{ Base::getPrivateNodeHandle() };

    std::string image_topic;
    std::string depth_topic;
    std::string image_publisher_topic{ "" };
    std::string& encoding{ _encoding };
    std::string viz_image_topic{ "" };
    std::string endcap_publisher_topic{ "" };

    double& dp{ _dp };
    double& min_dist{ _min_dist };
    double& param1{ _param1 };
    double& param2{ _param2 };
    double& depth_scale{ _depth_scale };
    int& min_radius{ _min_radius };
    int& max_radius{ _max_radius };
    bool& visualize{ _viz };
    double observation_freq{ 10 };
    int& endcap_id{ _endcap_id };

    std::vector<double> focal_lenght;
    std::vector<double> optical_center;
    std::vector<double> camera_extrinsics;

    PARAM_SETUP(private_nh, image_topic);
    PARAM_SETUP(private_nh, depth_topic);
    PARAM_SETUP(private_nh, encoding);
    PARAM_SETUP(private_nh, depth_scale);
    PARAM_SETUP(private_nh, focal_lenght);
    PARAM_SETUP(private_nh, optical_center);
    PARAM_SETUP(private_nh, camera_extrinsics);
    PARAM_SETUP(private_nh, endcap_id);
    PARAM_SETUP(private_nh, endcap_publisher_topic);
    PARAM_SETUP_WITH_DEFAULT(private_nh, image_publisher_topic, image_publisher_topic);
    PARAM_SETUP_WITH_DEFAULT(private_nh, dp, dp);
    PARAM_SETUP_WITH_DEFAULT(private_nh, min_dist, min_dist);
    PARAM_SETUP_WITH_DEFAULT(private_nh, param1, param1);
    PARAM_SETUP_WITH_DEFAULT(private_nh, param2, param2);
    PARAM_SETUP_WITH_DEFAULT(private_nh, min_radius, min_radius);
    PARAM_SETUP_WITH_DEFAULT(private_nh, max_radius, max_radius);
    PARAM_SETUP_WITH_DEFAULT(private_nh, visualize, visualize);
    PARAM_SETUP_WITH_DEFAULT(private_nh, observation_freq, observation_freq);
    PARAM_SETUP_WITH_DEFAULT(private_nh, viz_image_topic, viz_image_topic);

    TENSEGRITY_ASSERT(optical_center.size() == 2, "Optical center must have two values");
    TENSEGRITY_ASSERT(focal_lenght.size() == 2, "Focal length must have two values");

    _intrinsic.diagonal() = Eigen::Vector4d(focal_lenght[0], focal_lenght[1], 1.0, 1.0);
    _intrinsic.block<2, 1>(0, 3) = Eigen::Vector2d(optical_center[0], optical_center[1]);

    TENSEGRITY_ASSERT(camera_extrinsics.size() == 3 * 3 or camera_extrinsics.size() == 3 * 4,
                      "Wrong number of rows for camera_extrinsics");
    for (int i = 0; i < 3; ++i)
    {
      for (int j = 0; j < 4; ++j)
      {
        _extrinsic(i, j) = camera_extrinsics[4 * i + j];
      }
    }
    const Eigen::Matrix3d R{ _extrinsic.block<3, 3>(0, 0).transpose() };
    const Eigen::Vector3d t{ _extrinsic.col(3).head(3) };
    _int_ext_inv.block<3, 3>(0, 0) = R;
    _int_ext_inv.col(3).head(3) = -R * t;

    _int_ext_inv = _int_ext_inv * _intrinsic.inverse();
    DEBUG_VARS(_intrinsic);
    DEBUG_VARS(_extrinsic);

    // _intrinsic = _intrinsic.inverse();

    _image_subscriber = private_nh.subscribe(image_topic, 1, &Derived::image_callback, this);
    _depth_subscriber = private_nh.subscribe(depth_topic, 1, &Derived::depth_callback, this);

    if (_viz)
    {
      TENSEGRITY_ASSERT(viz_image_topic != "", "No viz_image_topic given");
      _viz_img_subscriber = private_nh.subscribe(viz_image_topic, 1, &Derived::viz_callback, this);
    }

    _frame_publisher = private_nh.advertise<sensor_msgs::Image>(image_publisher_topic, 1, true);
    _endcap_publisher = private_nh.advertise<interface::TensegrityEndcaps>(endcap_publisher_topic, 1, true);
    const ros::Duration observation_timer(1.0 / observation_freq);
    _observation_timer = private_nh.createTimer(observation_timer, &Derived::observation_timer_callback, this);

    _endcap_msg.header.seq = 0;
    _endcap_msg.barId = _endcap_id;
  }

  void observation_timer_callback(const ros::TimerEvent& event)
  {
    if (_depth_received and _image_received)
    {
      tensegrity::utils::update_header(_endcap_msg.header);

      _endcap_msg.endcaps.clear();
      _endcap_msg.covDim.clear();
      _endcap_msg.covariances.clear();

      cv::Scalar mean, stddev;
      for (size_t k = 0; k < _bounding_recs.size(); k++)
      {
        const cv::Moments& m{ _moments[k] };
        const cv::Rect& rec{ _bounding_recs[k] };

        const double x_img{ m.m10 / m.m00 };
        const double y_img{ m.m01 / m.m00 };
        cv::Mat roi{ _depth(rec) };
        cv::Mat mask{ roi != 0 };
        // DEBUG_VARS(_depth.at<>(m.m10 / m.m00, m.m01 / m.m00));
        // DEBUG_VARS(_depth);
        // DEBUG_VARS(roi);
        // DEBUG_VARS(mask);
        cv::meanStdDev(roi, mean, stddev, mask);
        const double z{ mean[0] / _depth_scale };
        const double z_sigma{ stddev[0] / _depth_scale };
        const Eigen::Vector4d uv(x_img * z, y_img * z, z, 1);
        const Eigen::Vector4d pw{ _int_ext_inv * uv };
        // DEBUG_VARS(uv);
        // DEBUG_VARS(_extrinsic_inv);
        // DEBUG_VARS(_intrinsic.inverse())
        // DEBUG_VARS(_extrinsic_inv * _intrinsic.inverse());
        // const Eigen::Vector2d xy{ ((pt - _center) / _focal) * mu };
        // const double x{ ((x_img - _center[0]) / _focal[0]) * z };
        // const double y{ ((y_img - _center[1]) / _focal[1]) * z };
        // DEBUG_VARS(pw);

        _endcap_msg.endcaps.push_back(create_point(pw));
      }

      _endcap_publisher.publish(_endcap_msg);
      _depth_received = _image_received = false;
    }
  }

  template <typename Point>
  inline geometry_msgs::Point create_point(const Point& point)
  {
    geometry_msgs::Point pt;
    pt.x = point[0];
    pt.y = point[1];
    pt.z = point[2];
    return std::move(pt);
  }

  void viz_callback(const sensor_msgs::ImageConstPtr message)
  {
    if (_viz)
    {
      cv_bridge::CvImageConstPtr frame{ cv_bridge::toCvShare(message) };

      // _contours_approx.resize(_contours_out.size());
      frame->image.copyTo(_frame_out);
      // if (_depth_received)
      //   _frame_out += _depth;
      for (size_t k = 0; k < _bounding_recs.size(); k++)
      {
        // cv::approxPolyDP(cv::Mat(_contours_out[k]), _contours_approx[k], 3, true);
        // const cv::Rect rec{ cv::boundingRect(_contours_out[k]) };
        const cv::Moments& m{ _moments[k] };
        const cv::Point center(m.m10 / m.m00, m.m01 / m.m00);
        cv::rectangle(_frame_out, _bounding_recs[k], cv::Scalar(128, 128, 128), 3, cv::LINE_8);
        cv::circle(_frame_out, center, 2, cv::Scalar(255, 255, 255), 2);
      }
      // cv::drawContours(_frame_out, _contours_approx, -1, cv::Scalar(128, 128, 128), cv::FILLED, cv::LINE_AA);

      _msg = cv_bridge::CvImage(std_msgs::Header(), _encoding, _frame_out).toImageMsg();
      _frame_publisher.publish(_msg);
    }
  }

  void depth_callback(const sensor_msgs::ImageConstPtr message)
  {
    cv_bridge::CvImageConstPtr frame{ cv_bridge::toCvShare(message) };
    frame->image.copyTo(_depth);
    _depth_received = true;
    // _depth = _depth / _depth_scale;
  }

  void image_callback(const sensor_msgs::ImageConstPtr message)
  {
    cv_bridge::CvImageConstPtr frame{ cv_bridge::toCvShare(message) };

    cv::findContours(frame->image, _contours_out, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    _moments.clear();
    _bounding_recs.clear();
    for (size_t k = 0; k < _contours_out.size(); k++)
    {
      _moments.push_back(cv::moments(_contours_out[k], false));
      _bounding_recs.push_back(cv::boundingRect(_contours_out[k]));
    }

    _image_received = true;
  }

  double _dp, _min_dist;
  double _param1, _param2;
  double _depth_scale;

  int _min_radius;
  int _max_radius;

  Eigen::Matrix4d _intrinsic;
  Eigen::Matrix4d _int_ext_inv;
  Eigen::Matrix4d _extrinsic;

  std::vector<cv::Rect> _bounding_recs;
  std::vector<cv::Moments> _moments;

  std::vector<std::vector<cv::Point>> _contours_out, _contours_approx;

  bool _viz;
  bool _depth_received, _image_received;

  std::string _encoding;

  cv::Mat _frame_out, _mask, _frame_rgb, _depth;

  sensor_msgs::ImagePtr _msg, _msg_rgb;

  ros::Subscriber _image_subscriber, _mask_subscriber, _depth_subscriber, _viz_img_subscriber;
  ros::Publisher _frame_publisher, _frame_rgb_publisher, _endcap_publisher;
  ros::Timer _observation_timer;

  int _endcap_id;
  interface::TensegrityEndcaps _endcap_msg;
};

}  // namespace perception
