#pragma once

// Ros
#include <ros/ros.h>

#include <interface/NodeStatus.h>
#include <tensegrity_utils/assert.hpp>
#include <tf2_ros/transform_listener.h>
#include <sensor_msgs/CameraInfo.h>
#include <tensegrity_utils/dbg_utils.hpp>
namespace interface
{

// The "Ros" way is to have a publisher for camera intrinsics and use TF for extrinsics.
// The purpose of this class is to combine these subscribers into a single class and use a gtsam::Camera
// to store and use them
class ros_camera_interface_t
{
  using This = ros_camera_interface_t;
  using Camera = gtsam::PinholeCamera<gtsam::Cal3_S2>;
  using CameraPtr = std::shared_ptr<Camera>;

  std::string _camera_frame;
  std::string _world_frame;

  ros::Subscriber _camera_info_subscriber;

  tf2_ros::Buffer _tf_buffer;
  tf2_ros::TransformListener _tf_listener;

  gtsam::Cal3_S2 _camera_calibration;
  bool _intrinsic_received;
  CameraPtr _camera;
  gtsam::Pose3 _extrinsic;

public:
  ros_camera_interface_t(ros::NodeHandle& nh) : _tf_listener(_tf_buffer)
  {
    std::string camera_info_topic;
    std::string& camera_frame{ _camera_frame };
    std::string& world_frame{ _world_frame };

    PARAM_SETUP(nh, camera_frame);
    PARAM_SETUP(nh, world_frame);
    PARAM_SETUP(nh, camera_info_topic);

    _camera_info_subscriber = nh.subscribe(camera_info_topic, 1, &This::camera_info_callback, this);
  }

  bool valid()
  {
    return _camera != nullptr;
  }

  CameraPtr camera()
  {
    return _camera;
  }

private:
  bool query_tf()
  {
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

      _extrinsic = gtsam::Pose3(gtsam::Rot3(R), -R * t);
      // _extrinsic.block<3, 3>(0, 0) = R;
      // _extrinsic.col(3).head(3) = -R * t;

      return true;
    }
    catch (tf2::TransformException& ex)
    {
    }
    return false;
  }

  void camera_info_callback(const sensor_msgs::CameraInfoConstPtr msg)
  {
    if (not _intrinsic_received)
    {
      const double fx{ msg->K[0] };
      const double cx{ msg->K[2] };
      const double fy{ msg->K[4] };
      const double cy{ msg->K[5] };
      _camera_calibration = gtsam::Cal3_S2(fx, fy, 0.0, cx, cy);
      _intrinsic_received = true;
      while (not query_tf())
      {
        // const std::string msg{ "Camera TF [" + _camera_frame + "] not received. Waiting... " };
        PRINT_MSG_ONCE("Camera TF not received. Waiting... ");
        ros::Duration(1.0).sleep();
      }
      _camera = std::make_shared<Camera>(_extrinsic, _camera_calibration);
    }
  }
};
}  // namespace interface