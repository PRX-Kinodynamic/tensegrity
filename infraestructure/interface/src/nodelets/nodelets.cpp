#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.hpp>

#include <interface/tensegrity_bars_to_markers.hpp>
#include <interface/tensegrity_trajectory_to_markers.hpp>
#include <interface/tensegrity_bars_to_file.hpp>
#include <interface/bars_array_to_markers.hpp>
namespace interface
{

using TensegrityBarsToMarkers = tensegrity_bars_to_markers_t<nodelet::Nodelet>;
using TensegrityTrajectoryToMarkers = tensegrity_trajectory_to_markers_t<nodelet::Nodelet>;
using TensegrityBarsToFile = tensegrity_bars_to_file_t<nodelet::Nodelet>;
using TensegrityBarsArrayToMarkers = tensegrity_array_to_markers_t<nodelet::Nodelet>;

}  // namespace interface
PLUGINLIB_EXPORT_CLASS(interface::TensegrityBarsToMarkers, nodelet::Nodelet);
PLUGINLIB_EXPORT_CLASS(interface::TensegrityTrajectoryToMarkers, nodelet::Nodelet);
PLUGINLIB_EXPORT_CLASS(interface::TensegrityBarsToFile, nodelet::Nodelet);
PLUGINLIB_EXPORT_CLASS(interface::TensegrityBarsArrayToMarkers, nodelet::Nodelet);
