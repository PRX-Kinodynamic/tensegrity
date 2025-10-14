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
#include <perception/utils.hpp>

#include <interface/node_status.hpp>
#include <interface/TensegrityBars.h>

#include <interface/type_conversions.hpp>
#include <estimation/bar_utilities.hpp>

namespace perception
{
template <class Base>
class tensegrity_filter_nodelet_t : public Base
{
  using Derived = tensegrity_filter_nodelet_t<Base>;

public:
  tensegrity_filter_nodelet_t()
    : _idx(0)
    , _tf_listener(_tf_buffer)
    , _depth_received(false)
    , _tf_received(false)
    , _intrinsic_received(false)
    , _intrinsic(Eigen::Matrix4d::Identity())
    , _int_ext_inv(Eigen::Matrix4d::Identity())
  {
  }

private:
  virtual void onInit()
  {
    ros::NodeHandle& private_nh{ Base::getPrivateNodeHandle() };

    _status = std::make_shared<interface::node_status_t>(private_nh);
    _status->status(interface::NodeStatus::PREPARING);

    dynamic_reconfigure::Server<perception::ColorFilterConfig>::CallbackType f;

    double& depth_scale{ _depth_scale };
    bool& use_depth{ _use_depth };
    bool& publish_rgb{ _publish_rgb };
    std::string subscriber_topic;
    std::string publisher_topic, camera_info_topic;
    std::string tensegrity_pose_topic;
    std::string& camera_frame{ _camera_frame };
    std::string& world_frame{ _world_frame };
    std::string& encoding{ _encoding };

    std::vector<int>& kernel_sizes{ _kernel_sizes };
    std::string& operations{ _operations };
    std::string depth_topic{ "" };
    std::string filters_namespace;
    std::string red_markers_topic;
    std::string green_markers_topic;
    std::string blue_markers_topic;
    std::string red_endcap_topic, green_endcap_topic, blue_endcap_topic;
    // /camera/filters/black/high
    PARAM_SETUP(private_nh, subscriber_topic);
    PARAM_SETUP(private_nh, publisher_topic);
    // PARAM_SETUP(private_nh, encoding);
    PARAM_SETUP(private_nh, filters_namespace);
    PARAM_SETUP(private_nh, kernel_sizes);
    PARAM_SETUP(private_nh, operations);
    PARAM_SETUP(private_nh, tensegrity_pose_topic);
    PARAM_SETUP(private_nh, camera_info_topic);
    PARAM_SETUP(private_nh, red_markers_topic);
    PARAM_SETUP(private_nh, green_markers_topic);
    PARAM_SETUP(private_nh, blue_markers_topic);
    PARAM_SETUP(private_nh, camera_frame);
    PARAM_SETUP(private_nh, world_frame);
    PARAM_SETUP(private_nh, red_endcap_topic);
    PARAM_SETUP(private_nh, green_endcap_topic);
    PARAM_SETUP(private_nh, blue_endcap_topic);

    PARAM_SETUP_WITH_DEFAULT(private_nh, depth_topic, depth_topic);
    PARAM_SETUP_WITH_DEFAULT(private_nh, use_depth, use_depth);
    PARAM_SETUP_WITH_DEFAULT(private_nh, depth_scale, depth_scale);
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

    _red_endcap_publisher = private_nh.advertise<interface::TensegrityEndcaps>(red_endcap_topic, 1, true);
    _green_endcap_publisher = private_nh.advertise<interface::TensegrityEndcaps>(green_endcap_topic, 1, true);
    _blue_endcap_publisher = private_nh.advertise<interface::TensegrityEndcaps>(blue_endcap_topic, 1, true);

    _red_markers_publisher = private_nh.advertise<visualization_msgs::Marker>(red_markers_topic, 1, true);
    _green_markers_publisher = private_nh.advertise<visualization_msgs::Marker>(green_markers_topic, 1, true);
    _blue_markers_publisher = private_nh.advertise<visualization_msgs::Marker>(blue_markers_topic, 1, true);
    _frame_publisher = private_nh.advertise<sensor_msgs::Image>(publisher_topic, 1, true);

    _image_subscriber = private_nh.subscribe(subscriber_topic, 1, &Derived::image_callback, this);
    _camera_info_subscriber = private_nh.subscribe(camera_info_topic, 1, &Derived::camera_info_callback, this);

    if (depth_topic != "")
    {
      _depth_subscriber = private_nh.subscribe(depth_topic, 1, &Derived::depth_callback, this);
    }

    _tensegrity_pose_subscriber =
        private_nh.subscribe(tensegrity_pose_topic, 1, &Derived::tensegrity_pose_callback, this);

    init_marker(red_marker, { 1.0, 0.0, 0.0, 1.0 });
    init_marker(green_marker, { 0.0, 1.0, 0.0, 1.0 });
    init_marker(blue_marker, { 0.0, 0.0, 1.0, 1.0 });

    _status->status(interface::NodeStatus::READY);
    _status->status(interface::NodeStatus::RUNNING);
  }

  void init_marker(visualization_msgs::Marker& marker, std::vector<double> rgba)
  {
    marker.action = visualization_msgs::Marker::ADD;
    marker.type = visualization_msgs::Marker::POINTS;

    marker.color.r = rgba[0];
    marker.color.g = rgba[1];
    marker.color.b = rgba[2];
    marker.color.a = rgba[3];

    marker.scale.x = 0.05;  // is point width,
    marker.scale.y = 0.05;  // is point height
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
    // const gtsam::Key key_Xr{ estimation::rod_symbol(estimation::RodColors::RED, _idx) };
    // const gtsam::Key key_Xg{ estimation::rod_symbol(estimation::RodColors::GREEN, _idx) };
    // const gtsam::Key key_Xb{ estimation::rod_symbol(estimation::RodColors::BLUE, _idx) };

    // graph.emplace_shared<gtsam::BetweenFactor>(key_Xr, key_Xg)
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

    erode_or_dilate(0, _frame_black, _frame_black, _operations, _elements);
    erode_or_dilate(0, _frame_red, _frame_red, _operations, _elements);
    erode_or_dilate(0, _frame_green, _frame_green, _operations, _elements);
    erode_or_dilate(0, _frame_blue, _frame_blue, _operations, _elements);
    _mask = _frame_black + _frame_red + _frame_green + _frame_blue;

    _frame_out.release();
    cv::copyTo(frame->image, _frame_out, _mask);

    _msg = cv_bridge::CvImage(message->header, "mono8", _mask).toImageMsg();

    _msg->header.stamp = ros::Time::now();
    _msg->header.seq = message->header.seq;

    _frame_publisher.publish(_msg);
    _rgb_received = true;
    compute_endcaps();
  }

  void compute_endcaps()
  {
    query_tf();
    if (_depth_received and _rgb_received and _tf_received and _intrinsic_received and _use_depth)
    {
      find_endcaps(_frame_red, red_endcaps);
      find_endcaps(_frame_green, green_endcaps);
      find_endcaps(_frame_blue, blue_endcaps);

      update_marker(red_marker, red_endcaps);
      update_marker(green_marker, green_endcaps);
      update_marker(blue_marker, blue_endcaps);

      _red_endcap_publisher.publish(red_endcaps);
      _green_endcap_publisher.publish(green_endcaps);
      _blue_endcap_publisher.publish(blue_endcaps);

      _red_markers_publisher.publish(red_marker);
      _green_markers_publisher.publish(green_marker);
      _blue_markers_publisher.publish(blue_marker);
      _rgb_received = false;
      _depth_received = false;
    }
  }

  void depth_callback(const sensor_msgs::ImageConstPtr message)
  {
    cv_bridge::CvImageConstPtr frame{ cv_bridge::toCvShare(message) };
    frame->image.copyTo(_depth);
    _depth_received = true;
    compute_endcaps();
    // _depth = _depth / _depth_scale;
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

  void find_endcaps(const cv::Mat& img, interface::TensegrityEndcaps& endcap_msg)
  {
    endcap_msg.endcaps.clear();
    endcap_msg.covDim.clear();
    endcap_msg.covariances.clear();

    std::vector<cv::Rect> bounding_recs;
    std::vector<cv::Moments> moments;
    find_contours(img, moments, bounding_recs);

    cv::Scalar mean, stddev;
    for (size_t k = 0; k < bounding_recs.size(); k++)
    {
      const cv::Moments& m{ moments[k] };
      const cv::Rect& rec{ bounding_recs[k] };

      const double x_img{ m.m10 / m.m00 };
      const double y_img{ m.m01 / m.m00 };
      cv::Mat roi{ _depth(rec) };
      cv::Mat mask{ roi != 0 };

      cv::meanStdDev(roi, mean, stddev, mask);
      const double z{ mean[0] / _depth_scale };
      const double z_sigma{ stddev[0] / _depth_scale };
      const Eigen::Vector4d uv(x_img * z, y_img * z, z, 1);
      const Eigen::Vector4d pw{ _int_ext_inv * uv };

      // DEBUG_VARS(x_img, y_img, z);
      // DEBUG_VARS(uv.transpose());
      // DEBUG_VARS(_int_ext_inv);
      // DEBUG_VARS(pw.transpose());
      endcap_msg.endcaps.push_back(create_point(pw));
    }
  }
  void update_marker(visualization_msgs::Marker& marker, interface::TensegrityEndcaps& endcaps)
  {
    tensegrity::utils::init_header(marker.header, "world");

    marker.points.clear();
    marker.points.insert(marker.points.end(), endcaps.endcaps.begin(), endcaps.endcaps.end());
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

      quat.w() = tf.transform.rotation.w;
      quat.x() = tf.transform.rotation.x;
      quat.y() = tf.transform.rotation.y;
      quat.z() = tf.transform.rotation.z;

      t[0] = tf.transform.translation.x;
      t[1] = tf.transform.translation.y;
      t[2] = tf.transform.translation.z;

      R = quat.toRotationMatrix().transpose();

      _int_ext_inv.block<3, 3>(0, 0) = R;
      _int_ext_inv.col(3).head(3) = -R * t;

      DEBUG_VARS(_intrinsic);
      DEBUG_VARS(_int_ext_inv);
      _int_ext_inv = _int_ext_inv * _intrinsic.inverse();
      DEBUG_VARS(_int_ext_inv);
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
    _intrinsic_received = true;
    _tf_received = false;
  }
  void find_contours(const cv::Mat& img, std::vector<cv::Moments>& moments, std::vector<cv::Rect>& bounding_recs)
  {
    std::vector<std::vector<cv::Point>> contours_out;
    cv::findContours(img, contours_out, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    moments.clear();
    bounding_recs.clear();

    std::sort(contours_out.begin(), contours_out.end(),
              [](cv::InputArray a, cv::InputArray b) { return cv::contourArea(a) > cv::contourArea(b); });

    const std::size_t max_countours{ contours_out.size() };
    for (size_t k = 0; k < max_countours; k++)
    {
      if (cv::contourArea(contours_out[k]) < 1.0)
        continue;
      // DEBUG_VARS(cv::contourArea(contours_out[k]))
      moments.push_back(cv::moments(contours_out[k], false));
      bounding_recs.push_back(cv::boundingRect(contours_out[k]));
    }
  }

  interface::TensegrityEndcaps red_endcaps, green_endcaps, blue_endcaps;

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
  cv::Mat _depth;

  sensor_msgs::ImagePtr _msg, _msg_rgb;

  ros::Subscriber _image_subscriber, _tensegrity_pose_subscriber, _depth_subscriber, _camera_info_subscriber;
  ros::Publisher _frame_publisher, _frame_rgb_publisher;
  ros::Publisher _red_markers_publisher, _green_markers_publisher, _blue_markers_publisher;
  ros::Publisher _red_endcap_publisher, _green_endcap_publisher, _blue_endcap_publisher;

  visualization_msgs::Marker red_marker, green_marker, blue_marker;

  bool _use_depth, _depth_received, _rgb_received;
  double _depth_scale;

  gtsam::Pose3 _btw_red_green, _btw_red_blue, _btw_green_blue;

  bool _tf_received, _intrinsic_received;

  Eigen::Matrix4d _intrinsic;
  Eigen::Matrix4d _int_ext_inv;
  Eigen::Matrix4d _extrinsic;

  std::string _world_frame;
  std::string _camera_frame;
  tf2_ros::Buffer _tf_buffer;
  tf2_ros::TransformListener _tf_listener;
  std::shared_ptr<interface::node_status_t> _status;
};

}  // namespace perception
