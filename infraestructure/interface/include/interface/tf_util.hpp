#pragma once

// Ros
#include <ros/ros.h>

#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>

#include <tensegrity_utils/assert.hpp>

namespace interface
{

// The "Ros" way is to have a publisher for camera intrinsics and use TF for extrinsics.
// The purpose of this class is to combine these subscribers into a single class and use a gtsam::Camera
// to store and use them
class ros_tf_utils_t
{
  std::string _other_frame;
  std::string _world_frame;

  tf2_ros::Buffer _tf_buffer;
  tf2_ros::TransformListener _tf_listener;

public:
  ros_tf_utils_t(ros::NodeHandle& nh) : _tf_listener(_tf_buffer)
  {
    std::string& other_frame{ _other_frame };
    std::string& world_frame{ _world_frame };

    PARAM_SETUP(nh, other_frame);
    PARAM_SETUP(nh, world_frame);
  }

  std::string other_frame() const
  {
    return _other_frame;
  }
  std::string world_frame() const
  {
    return _world_frame;
  }

  bool query(gtsam::Pose3& transform)
  {
    try
    {
      Eigen::Vector3d t;
      Eigen::Matrix3d R;
      Eigen::Quaterniond quat;
      const geometry_msgs::TransformStamped tf{ _tf_buffer.lookupTransform(_other_frame, _world_frame, ros::Time(0)) };

      quat.w() = tf.transform.rotation.w;
      quat.x() = tf.transform.rotation.x;
      quat.y() = tf.transform.rotation.y;
      quat.z() = tf.transform.rotation.z;

      t[0] = tf.transform.translation.x;
      t[1] = tf.transform.translation.y;
      t[2] = tf.transform.translation.z;

      R = quat.toRotationMatrix().transpose();

      transform = gtsam::Pose3(gtsam::Rot3(R), -R * t);
      // _extrinsic.block<3, 3>(0, 0) = R;
      // _extrinsic.col(3).head(3) = -R * t;

      return true;
    }
    catch (tf2::TransformException& ex)
    {
    }
    return false;
  }
};
}  // namespace interface