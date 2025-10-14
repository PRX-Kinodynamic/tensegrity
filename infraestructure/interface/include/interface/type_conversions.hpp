#pragma once
#include <tensegrity_utils/type_conversions.hpp>
#include <factor_graphs/type_conversions.hpp>
#include <interface/TensegrityEndcaps.h>
#include <interface/TensegrityBars.h>

namespace tensegrity
{
namespace utils

{

template <
    typename StringType, typename MsgType,
    std::enable_if_t<std::is_same<StringType, std::string>::value && std::is_same<MsgType, geometry_msgs::Point>::value,
                     bool> = true>
inline StringType convert_to(const MsgType& from)
{
  std::stringstream strstr;
  strstr << convert_to<std::string>(from.x) << " ";
  strstr << convert_to<std::string>(from.y) << " ";
  strstr << convert_to<std::string>(from.z) << " ";
  return strstr.str();
}

template <typename StringType, typename MsgType,
          std::enable_if_t<std::is_same<StringType, std::string>::value &&
                               std::is_same<MsgType, geometry_msgs::Quaternion>::value,
                           bool> = true>
inline StringType convert_to(const MsgType& from)
{
  std::stringstream strstr;
  strstr << convert_to<std::string>(from.w) << " ";
  strstr << convert_to<std::string>(from.x) << " ";
  strstr << convert_to<std::string>(from.y) << " ";
  strstr << convert_to<std::string>(from.z) << " ";
  return strstr.str();
}

template <typename MsgType, typename RotType,
          std::enable_if_t<std::is_same<RotType, gtsam::Quaternion>::value &&
                               std::is_same<MsgType, geometry_msgs::Quaternion>::value,
                           bool> = true>
inline MsgType convert_to(const RotType& from)
{
  MsgType msg;
  msg.w = from.w();
  msg.x = from.x();
  msg.y = from.y();
  msg.z = from.z();
  return msg;
}

template <typename MsgType, typename RotType,
          std::enable_if_t<std::is_same<RotType, gtsam::Rot3>::value &&
                               std::is_same<MsgType, geometry_msgs::Quaternion>::value,
                           bool> = true>
inline MsgType convert_to(const RotType& from)
{
  return convert_to<MsgType>(from.toQuaternion());
}

template <
    typename MsgType, typename InType,
    std::enable_if_t<std::is_same<InType, Eigen::Vector3d>::value && std::is_same<MsgType, geometry_msgs::Point>::value,
                     bool> = true>
inline MsgType convert_to(const InType& from)
{
  geometry_msgs::Point pt;
  pt.x = from[0];
  pt.y = from[1];
  pt.z = from[2];
  return pt;
}

template <
    typename MsgType, typename InType,
    std::enable_if_t<std::is_same<InType, gtsam::Pose3>::value && std::is_same<MsgType, geometry_msgs::Pose>::value,
                     bool> = true>
inline MsgType convert_to(const InType& from)
{
  geometry_msgs::Pose pose;
  pose.position = convert_to<geometry_msgs::Point>(from.translation());
  pose.orientation = convert_to<geometry_msgs::Quaternion>(from.rotation());
  return pose;
}

template <
    typename StringType, typename MsgType,
    std::enable_if_t<std::is_same<StringType, std::string>::value && std::is_same<MsgType, geometry_msgs::Pose>::value,
                     bool> = true>
inline StringType convert_to(const MsgType& from)
{
  std::stringstream strstr;
  strstr << convert_to<std::string>(from.position) << " ";
  strstr << convert_to<std::string>(from.orientation) << " ";
  return strstr.str();
}

template <
    typename Point, typename MsgType,
    std::enable_if_t<std::is_same<Point, Eigen::Vector3d>::value && std::is_same<MsgType, geometry_msgs::Point>::value,
                     bool> = true>
inline Eigen::Vector3d convert_to(const MsgType& from)
{
  const double x{ from.x };
  const double y{ from.y };
  const double z{ from.z };

  return { x, y, z };
}

template <typename Rotation, typename MsgType,
          std::enable_if_t<std::is_same<Rotation, gtsam::Rot3>::value &&
                               std::is_same<MsgType, geometry_msgs::Quaternion>::value,
                           bool> = true>
inline Rotation convert_to(const MsgType& from)
{
  const double w{ from.w };
  const double x{ from.x };
  const double y{ from.y };
  const double z{ from.z };

  return gtsam::Rot3(w, x, y, z);
}

template <typename Pose, typename MsgType,
          std::enable_if_t<std::is_same<Pose, gtsam::Pose3>::value && std::is_same<MsgType, geometry_msgs::Pose>::value,
                           bool> = true>
inline gtsam::Pose3 convert_to(const MsgType& from)
{
  const Eigen::Vector3d p{ convert_to<Eigen::Vector3d>(from.position) };
  const gtsam::Rot3 R{ convert_to<gtsam::Rot3>(from.orientation) };
  return gtsam::Pose3(R, p);
}

}  // namespace utils
}  // namespace tensegrity