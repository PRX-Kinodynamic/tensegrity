#pragma once
#include <tensegrity_utils/type_conversions.hpp>
#include <gtsam/geometry/Pose3.h>

namespace tensegrity
{
namespace utils

{

template <
    typename StringType, typename GtsamType,
    std::enable_if_t<std::is_same<StringType, std::string>::value && std::is_same<GtsamType, gtsam::Quaternion>::value,
                     bool> = true>
inline StringType convert_to(const GtsamType& from)
{
  std::stringstream strstr;
  strstr << convert_to<std::string>(from.w()) << " ";
  strstr << convert_to<std::string>(from.x()) << " ";
  strstr << convert_to<std::string>(from.y()) << " ";
  strstr << convert_to<std::string>(from.z()) << " ";
  return strstr.str();
}

template <typename GtsamTypeTo, typename GtsamTypeFrom,
          std::enable_if_t<std::is_same<GtsamTypeTo, gtsam::Quaternion>::value &&
                               std::is_same<GtsamTypeFrom, gtsam::Rot3>::value,
                           bool> = true>
inline GtsamTypeTo convert_to(const GtsamTypeFrom& from)
{
  return from.toQuaternion();
}

template <typename StringType, typename GtsamType,
          std::enable_if_t<std::is_same<StringType, std::string>::value && std::is_same<GtsamType, gtsam::Rot3>::value,
                           bool> = true>
inline StringType convert_to(const GtsamType& from)
{
  return convert_to<std::string>(convert_to<gtsam::Quaternion>(from));
}

template <typename StringType, typename GtsamType,
          std::enable_if_t<std::is_same<StringType, std::string>::value && std::is_same<GtsamType, gtsam::Pose3>::value,
                           bool> = true>
inline StringType convert_to(const GtsamType& from)
{
  std::stringstream strstr;
  strstr << convert_to<std::string>(from.rotation()) << " ";
  strstr << convert_to<std::string>(from.translation().x()) << " ";
  strstr << convert_to<std::string>(from.translation().y()) << " ";
  strstr << convert_to<std::string>(from.translation().z()) << " ";
  return strstr.str();
}

}  // namespace utils
}  // namespace tensegrity