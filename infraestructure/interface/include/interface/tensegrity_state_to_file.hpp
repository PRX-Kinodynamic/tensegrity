#include <set>
#include <thread>
#include <fstream>
#include <iostream>

// Ros
#include <ros/ros.h>
#include <ros/package.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>

#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseArray.h>

#include <tensegrity_utils/type_conversions.hpp>
#include <tensegrity_utils/std_utils.hpp>
#include <tensegrity_utils/dbg_utils.hpp>
#include <tensegrity_utils/rosparams_utils.hpp>

#include <factor_graphs/defs.hpp>

#include <interface/TensegrityBars.h>
#include <interface/node_status.hpp>

#include <resource_retriever/retriever.h>

#include <visualization_msgs/MarkerArray.h>

namespace interface
{
template <typename Base>
class tensegrity_bars_to_file_t : public Base
{
  using Derived = tensegrity_bars_to_file_t<Base>;

public:
  tensegrity_bars_to_file_t()
  {
  }

  ~tensegrity_bars_to_file_t(){};

  virtual void onInit()
  {
    ros::NodeHandle& private_nh{ Base::getPrivateNodeHandle() };

    std::string tensegrity_bars_topicname;
    std::string filename;
    std::string node_id;
    double alpha{ 1.0 };

    PARAM_SETUP(private_nh, node_id);
    PARAM_SETUP(private_nh, filename);
    PARAM_SETUP(private_nh, tensegrity_bars_topicname);

    _file.open(filename, std::ios::out);
    _status = std::make_shared<node_status_t>(private_nh, node_id);
    _status->status(interface::NodeStatus::PREPARING);

    _tensegrity_bars_subscriber = private_nh.subscribe(tensegrity_bars_topicname, 1, &Derived::callback, this);

    _status->status(interface::NodeStatus::READY);
    _status->status(interface::NodeStatus::RUNNING);
  }

private:
  void callback(const interface::TensegrityBarsConstPtr msg)
  {
    // _red_marker.header = msg->header;
    // _green_marker.header = msg->header;
    // _blue_marker.header = msg->header;

    // _red_marker.pose = msg->bar_red;
    // _green_marker.pose = msg->bar_green;
    // _blue_marker.pose = msg->bar_blue;

    _file << convert_to<std::string>(msg->header.stamp) << " ";
    // _file << convert_to<std::string>(msg->bar_red) << " ";
    // _file << convert_to<std::string>(msg->bar_green) << " ";
    // _file << convert_to<std::string>(msg->bar_blue) << "\n";
  }

  ros::Subscriber _tensegrity_bars_subscriber;

  std::shared_ptr<node_status_t> _status;

  std::ofstream _file;
};
}  // namespace interface