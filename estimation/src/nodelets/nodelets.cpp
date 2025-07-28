#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.hpp>

#include <ml4kp_bridge/defs.h>

#include <estimation/tensegrity_estimation.hpp>
#include <estimation/pose_from_poly.hpp>

namespace estimation
{

using TensegrityEstimation = tensegrity_estimation_t<nodelet::Nodelet>;
using PosesFromPoly = pose_from_poly_t<nodelet::Nodelet>;

}  // namespace estimation
PLUGINLIB_EXPORT_CLASS(estimation::TensegrityEstimation, nodelet::Nodelet);
PLUGINLIB_EXPORT_CLASS(estimation::PosesFromPoly, nodelet::Nodelet);
