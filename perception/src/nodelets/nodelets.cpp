#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.hpp>

#include <perception/color_filter.hpp>
#include <perception/dilation_nodelet.hpp>
#include <perception/mask_nodelet.hpp>
#include <perception/endcap_position_detector_nodelet.hpp>

namespace perception
{

using ColorFilter = color_filter_t<nodelet::Nodelet>;
using DilateImage = dilation_nodelet_t<nodelet::Nodelet>;
using MaskImage = mask_nodelet_t<nodelet::Nodelet>;
using EndcapPosition = endcap_position_detector_nodelet_t<nodelet::Nodelet>;

}  // namespace perception

PLUGINLIB_EXPORT_CLASS(perception::ColorFilter, nodelet::Nodelet);
PLUGINLIB_EXPORT_CLASS(perception::DilateImage, nodelet::Nodelet);
PLUGINLIB_EXPORT_CLASS(perception::MaskImage, nodelet::Nodelet);
PLUGINLIB_EXPORT_CLASS(perception::EndcapPosition, nodelet::Nodelet);
