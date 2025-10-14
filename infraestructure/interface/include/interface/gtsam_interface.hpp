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

inline void copy(geometry_msgs::Point& pt_out, const Eigen::Vector3d& pt_in)
{
  pt_out.x = pt_in[0];
  pt_out.y = pt_in[1];
  pt_out.z = pt_in[2];
}

// Out <- In;
inline void copy(gtsam::Pose3& out, const geometry_msgs::Pose& in)
{
  Eigen::Vector3d t;
  gtsam::Rot3 R;
  copy(t, in.position);
  copy(R, in.orientation);
  out = gtsam::Pose3(R, t);
}

}  // namespace interface