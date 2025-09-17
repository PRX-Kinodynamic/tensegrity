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
#include <visualization_msgs/Marker.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
// #include <message_filters/subscriber.h>
// #include <message_filters/time_synchronizer.h>
#include <opencv2/core/fast_math.hpp>

#include <sensor_msgs/CameraInfo.h>

#include <interface/node_status.hpp>
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
    , _tf_received(false)
    , _intrinsic_received(false)
    , _intrinsic(Eigen::Matrix4d::Identity())
    , _extrinsic(Eigen::Matrix4d::Identity())
    , _int_ext_inv(Eigen::Matrix4d::Identity())
    , _endcap_id(-1)
    , _max_marker_observations(10)
    , _tf_listener(_tf_buffer)
  {
  }

private:
  virtual void onInit()
  {
    ros::NodeHandle& private_nh{ Base::getPrivateNodeHandle() };

    _status = std::make_shared<interface::node_status_t>(private_nh);
    _status->status(interface::NodeStatus::PREPARING);

    std::string image_topic;
    std::string depth_topic;
    std::string camera_info_topic;
    std::string image_publisher_topic{ "" };
    std::string& encoding{ _encoding };
    std::string viz_image_topic{ "" };
    std::string endcap_publisher_topic{ "" };
    std::string marker_topicname{ "" };
    std::string& camera_frame{ _camera_frame };
    std::string& world_frame{ _world_frame };

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
    int& max_observations_to_visualize{ _max_marker_observations };

    // std::vector<double> focal_lenght;
    // std::vector<double> optical_center;
    // // std::vector<double> camera_extrinsics;
    std::vector<double> rgba = { 0.0, 0.0, 0.0, 1.0 };

    PARAM_SETUP(private_nh, image_topic);
    PARAM_SETUP(private_nh, depth_topic);
    PARAM_SETUP(private_nh, encoding);
    PARAM_SETUP(private_nh, depth_scale);
    // PARAM_SETUP(private_nh, focal_lenght);
    // PARAM_SETUP(private_nh, optical_center);
    PARAM_SETUP(private_nh, camera_info_topic);
    // PARAM_SETUP(private_nh, endcap_id);
    PARAM_SETUP(private_nh, endcap_publisher_topic);
    PARAM_SETUP(private_nh, camera_frame);
    PARAM_SETUP(private_nh, world_frame);
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
    PARAM_SETUP_WITH_DEFAULT(private_nh, marker_topicname, marker_topicname);
    PARAM_SETUP_WITH_DEFAULT(private_nh, max_observations_to_visualize, max_observations_to_visualize);
    PARAM_SETUP_WITH_DEFAULT(private_nh, rgba, rgba);

    if (marker_topicname == "")
    {
      marker_topicname = endcap_publisher_topic + "/marker";
    }

    // TENSEGRITY_ASSERT(optical_center.size() == 2, "Optical center must have two values");
    // TENSEGRITY_ASSERT(focal_lenght.size() == 2, "Focal length must have two values");

    // _intrinsic.diagonal() = Eigen::Vector4d(focal_lenght[0], focal_lenght[1], 1.0, 1.0);
    // _intrinsic.block<2, 1>(0, 2) = Eigen::Vector2d(optical_center[0], optical_center[1]);

    // TENSEGRITY_ASSERT(camera_extrinsics.size() == 3 * 3 or camera_extrinsics.size() == 3 * 4,
    //                   "Wrong number of rows for camera_extrinsics");
    // for (int i = 0; i < 3; ++i)
    // {
    //   for (int j = 0; j < 4; ++j)
    //   {
    //     _extrinsic(i, j) = camera_extrinsics[4 * i + j];
    //   }
    // }

    // const Eigen::Matrix3d R{ _extrinsic.block<3, 3>(0, 0).transpose() };
    // const Eigen::Vector3d t{ _extrinsic.col(3).head(3) };
    // _int_ext_inv.block<3, 3>(0, 0) = R;
    // _int_ext_inv.col(3).head(3) = -R * t;

    // _int_ext_inv = _int_ext_inv * _intrinsic.inverse();
    // DEBUG_VARS(_intrinsic);

    // _intrinsic = _intrinsic.inverse();
    _frame_publisher = private_nh.advertise<sensor_msgs::Image>(image_publisher_topic, 1, true);
    _endcap_publisher = private_nh.advertise<interface::TensegrityEndcaps>(endcap_publisher_topic, 1, true);
    _observations_publisher = private_nh.advertise<visualization_msgs::Marker>(marker_topicname, 1, true);

    _image_subscriber = private_nh.subscribe(image_topic, 1, &Derived::image_callback, this);
    _depth_subscriber = private_nh.subscribe(depth_topic, 1, &Derived::depth_callback, this);

    if (_viz)
    {
      TENSEGRITY_ASSERT(viz_image_topic != "", "No viz_image_topic given");
      _viz_img_subscriber = private_nh.subscribe(viz_image_topic, 1, &Derived::viz_callback, this);
    }
    _camera_info_subscriber = private_nh.subscribe(camera_info_topic, 1, &Derived::camera_info_callback, this);

    const ros::Duration observation_timer(1.0 / observation_freq);
    _observation_timer = private_nh.createTimer(observation_timer, &Derived::observation_timer_callback, this);

    // _endcap_msg.header.seq = 0;
    _endcap_msg.barId = _endcap_id;
    tensegrity::utils::init_header(_endcap_msg.header, "world");

    tensegrity::utils::init_header(_marker.header, "world");
    _marker.action = visualization_msgs::Marker::ADD;
    _marker.type = visualization_msgs::Marker::POINTS;

    _marker.color.r = rgba[0];
    _marker.color.g = rgba[1];
    _marker.color.b = rgba[2];
    _marker.color.a = rgba[3];

    _marker.scale.x = 0.03;  // is point width,
    _marker.scale.y = 0.03;  // is point height

    _status->status(interface::NodeStatus::READY);
    _status->status(interface::NodeStatus::RUNNING);
  }

  void query_tf()
  {
    if (not _intrinsic_received)
      return;
    if (_tf_received)
      return;
    try
    {
      Eigen::Vector3d t;
      Eigen::Matrix3d R;
      Eigen::Quaterniond quat;
      const geometry_msgs::TransformStamped tf{ _tf_buffer.lookupTransform(_camera_frame, _world_frame, ros::Time(0)) };
      // const geometry_msgs::TransformStamped tf{ _tf_buffer.lookupTransform(_world_frame, _camera_frame, ros::Time(0))
      // };

      quat.w() = tf.transform.rotation.w;
      quat.x() = tf.transform.rotation.x;
      quat.y() = tf.transform.rotation.y;
      quat.z() = tf.transform.rotation.z;

      t[0] = tf.transform.translation.x;
      t[1] = tf.transform.translation.y;
      t[2] = tf.transform.translation.z;
      // interface::copy(quat, tf.transform.orientation);
      // interface::copy(t, tf.transform.position);

      R = quat.toRotationMatrix().transpose();
      // DEBUG_VARS(R)
      // DEBUG_VARS(t)

      _int_ext_inv.block<3, 3>(0, 0) = R;
      _int_ext_inv.col(3).head(3) = -R * t;

      _int_ext_inv = _int_ext_inv * _intrinsic.inverse();
      _tf_received = true;
    }
    catch (tf2::TransformException& ex)
    {
    }
  }

  void camera_info_callback(const sensor_msgs::CameraInfoConstPtr msg)
  {
    for (int i = 0; i < 3; ++i)
    {
      for (int j = 0; j < 3; ++j)
      {
        _intrinsic(i, j) = msg->K[i * 3 + j];
      }
    }
    // DEBUG_VARS(_intrinsic);
    _intrinsic_received = true;
    _tf_received = false;
    // query_tf();

    // _intrinsic.diagonal() = Eigen::Vector4d(focal_lenght[0], focal_lenght[1], 1.0, 1.0);
    // _intrinsic.block<2, 1>(0, 2) = Eigen::Vector2d(optical_center[0], optical_center[1]);
  }

  void observation_timer_callback(const ros::TimerEvent& event)
  {
    query_tf();

    if (_depth_received and _image_received and _tf_received and _intrinsic_received)
    {
      tensegrity::utils::update_header(_endcap_msg.header);

      _endcap_msg.endcaps.clear();
      _endcap_msg.covDim.clear();
      _endcap_msg.covariances.clear();

      cv::Scalar mean, stddev;
      // for (size_t k = 0; k < max_recs; k++)
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
        // const Eigen::Vector4d uv(y_img * z, x_img * z, z, 1);
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

      update_marker();
      _endcap_msg.header.seq = _img_idx;
      _endcap_msg.header.stamp = ros::Time::now();
      _endcap_publisher.publish(_endcap_msg);
      _depth_received = _image_received = false;
    }
  }

  void update_marker()
  {
    // DEBUG_VARS(_marker.points.size(), _max_marker_observations);
    if (_max_marker_observations > 0)
    {
      _marker.points.insert(_marker.points.end(), _endcap_msg.endcaps.begin(), _endcap_msg.endcaps.end());
    }
    if (_marker.points.size() >= _max_marker_observations)
    {
      int to_eliminate = _marker.points.size() - _max_marker_observations;
      _marker.points.erase(_marker.points.begin(), _marker.points.begin() + to_eliminate);
      _observations_publisher.publish(_marker);
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
      // for (size_t k = 0; k < _bounding_recs.size(); k++)
      // {
      //   // cv::approxPolyDP(cv::Mat(_contours_out[k]), _contours_approx[k], 3, true);
      //   // const cv::Rect rec{ cv::boundingRect(_contours_out[k]) };
      //   const cv::Moments& m{ _moments[k] };
      //   const cv::Point center(m.m10 / m.m00, m.m01 / m.m00);
      //   cv::rectangle(_frame_out, _bounding_recs[k], cv::Scalar(128, 128, 128), 3, cv::LINE_8);
      //   cv::circle(_frame_out, center, 2, cv::Scalar(255, 255, 255), 2);
      // }
      // // cv::drawContours(_frame_out, _contours_approx, -1, cv::Scalar(128, 128, 128), cv::FILLED, cv::LINE_AA);

      // _msg = cv_bridge::CvImage(message->header, _encoding, _frame_out).toImageMsg();
      // _msg.header.seq = _img_idx;
      // _msg.header.stamp = ros::Time::now();
      // _frame_publisher.publish(_msg);
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

    std::sort(_contours_out.begin(), _contours_out.end(),
              [](cv::InputArray a, cv::InputArray b) { return cv::contourArea(a) > cv::contourArea(b); });

    // const std::size_t max_countours{ std::min(_contours_out.size(), 2ul) };
    const std::size_t max_countours{ _contours_out.size() };
    for (size_t k = 0; k < max_countours; k++)
    {
      // double area{ cv::contourArea(_contours_out[k]) };
      // DEBUG_VARS(k, area);
      _moments.push_back(cv::moments(_contours_out[k], false));
      _bounding_recs.push_back(cv::boundingRect(_contours_out[k]));
    }

    _img_idx = message->header.seq;
    _image_received = true;

    if (_viz)
    {
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

      _msg = cv_bridge::CvImage(message->header, _encoding, _frame_out).toImageMsg();
      _msg->header.seq = _img_idx;
      _msg->header.stamp = ros::Time::now();
      _frame_publisher.publish(_msg);
    }
  }

  int _img_idx;

  double _dp, _min_dist;
  double _param1, _param2;
  double _depth_scale;

  int _min_radius;
  int _max_radius;

  int _max_marker_observations;

  Eigen::Matrix4d _intrinsic;
  Eigen::Matrix4d _int_ext_inv;
  Eigen::Matrix4d _extrinsic;

  std::vector<cv::Rect> _bounding_recs;
  std::vector<cv::Moments> _moments;

  std::vector<std::vector<cv::Point>> _contours_out, _contours_approx;

  bool _viz;
  bool _depth_received, _image_received;
  bool _tf_received, _intrinsic_received;

  std::string _encoding;

  cv::Mat _frame_out, _mask, _frame_rgb, _depth;

  sensor_msgs::ImagePtr _msg, _msg_rgb;

  ros::Subscriber _camera_info_subscriber;
  ros::Subscriber _image_subscriber, _mask_subscriber, _depth_subscriber, _viz_img_subscriber;
  ros::Publisher _frame_publisher, _frame_rgb_publisher, _endcap_publisher, _observations_publisher;
  ros::Timer _observation_timer;

  int _endcap_id;
  interface::TensegrityEndcaps _endcap_msg;
  visualization_msgs::Marker _marker;

  std::string _world_frame;
  std::string _camera_frame;
  tf2_ros::Buffer _tf_buffer;
  tf2_ros::TransformListener _tf_listener;
  std::shared_ptr<interface::node_status_t> _status;
};

}  // namespace perception
