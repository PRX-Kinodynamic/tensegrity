#pragma once
// #include <interface/copy.hpp>
#include <interface/eigen_interface.hpp>

#include <geometry_msgs/Transform.h>
#include <geometry_msgs/Pose.h>
#include <gtsam/config.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>

// Copy and other interface utils
namespace interface
{
// Out <- In;
inline void copy(geometry_msgs::Quaternion& quat_msg, const gtsam::Rot3& rot)
{
  const Eigen::Quaterniond quat{ rot.toQuaternion() };
  copy(quat_msg, quat);
}

inline void copy(gtsam::Rot3& rot, const geometry_msgs::Quaternion& quat_msg)
{
  const double& w{ quat_msg.w };
  const double& x{ quat_msg.x };
  const double& y{ quat_msg.y };
  const double& z{ quat_msg.z };
  rot = gtsam::Rot3(w, x, y, z);
  // const Eigen::Quaterniond quat{ rot.toQuaternion() };
  // copy(quat_msg, quat);
}

// Out <- In;
inline void copy(geometry_msgs::Transform& tf, const gtsam::Pose3& pose)
{
  copy(tf.translation, pose.translation());
  copy(tf.rotation, pose.rotation());
}

// Out <- In;
inline void copy(geometry_msgs::Pose& out, const gtsam::Pose3& in)
{
  copy(out.position, in.translation());
  copy(out.orientation, in.rotation());
}

// Out <- In;
// inline void copy(gtsam::Pose3& out, const geometry_msgs::Pose& in)
// {
//   copy(out.translation(), in.position);
//   copy(out.rotation(), in.orientation);
// }

}  // namespace interface