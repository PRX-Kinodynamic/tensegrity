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
#include <interface/type_conversions.hpp>
#include <interface/type_conversions.hpp>

#include <resource_retriever/retriever.h>

#include <visualization_msgs/MarkerArray.h>

namespace interface
{
template <typename Base>
class tensegrity_bars_to_file_t : public Base
{
  using Derived = tensegrity_bars_to_file_t<Base>;

  using Point = Eigen::Vector3d;
  using Endcaps = std::pair<Point, Point>;

public:
  tensegrity_bars_to_file_t() : _offset({ 0.0, 0.0, 0.325 / 2.0 })
  {
  }

  ~tensegrity_bars_to_file_t()
  {
    _status->status(interface::NodeStatus::FINISH);
    _poses_file.close();
    _endcaps_file.close();
  }

  virtual void onInit()
  {
    ros::NodeHandle& private_nh{ Base::getPrivateNodeHandle() };

    std::string tensegrity_bars_topicname;
    std::string poses_filename, endcap_filename;
    std::string node_id;
    double alpha{ 1.0 };

    PARAM_SETUP(private_nh, node_id);
    PARAM_SETUP(private_nh, poses_filename);
    PARAM_SETUP(private_nh, endcap_filename);
    PARAM_SETUP(private_nh, tensegrity_bars_topicname);

    // const std::string timestamp{ tensegrity::utils::timestamp() };
    // poses_filename += "_" + timestamp + ".txt";
    // endcap_filename += "_" + timestamp + ".txt";

    _poses_file.open(poses_filename, std::ios::out);
    _endcaps_file.open(endcap_filename, std::ios::out);

    _status = std::make_shared<node_status_t>(private_nh, node_id);
    _status->status(interface::NodeStatus::PREPARING);

    _tensegrity_bars_subscriber = private_nh.subscribe(tensegrity_bars_topicname, 1, &Derived::callback, this);

    _poses_file << "# Time idx PositionRed QuatRed PositionGreen QuatGreen PositionBlue QuatBlue\n";
    _endcaps_file << "# Time idx EndcapARed EndcapBRed EndcapAGreen EndcapBGreen EndcapABlue EndcapBBlue \n";
    _status->status(interface::NodeStatus::READY);
    _status->status(interface::NodeStatus::RUNNING);
  }

private:
  Endcaps compute_endcaps(const geometry_msgs::Pose& pose_msg)
  {
    using tensegrity::utils::convert_to;
    const gtsam::Pose3 pose{ convert_to<gtsam::Pose3>(pose_msg) };
    // Eigen::Vector3d pt
    // copy(pose, pose_msg);
    const Point pA{ pose * _offset };
    const Point pB{ pose * -_offset };
    return { pA, pB };
  }

  void callback(const interface::TensegrityBarsConstPtr msg)
  {
    using tensegrity::utils::convert_to;
    const std::size_t idx{ msg->header.seq };
    const std::string timestamp{ tensegrity::utils::convert_to<std::string>(msg->header.stamp) };
    _poses_file << timestamp << " " << idx << " ";
    _poses_file << tensegrity::utils::convert_to<std::string>(msg->bar_red) << " ";
    _poses_file << tensegrity::utils::convert_to<std::string>(msg->bar_green) << " ";
    _poses_file << tensegrity::utils::convert_to<std::string>(msg->bar_blue) << " ";
    _poses_file << "\n";

    const Endcaps red_endcaps{ compute_endcaps(msg->bar_red) };
    const Endcaps green_endcaps{ compute_endcaps(msg->bar_green) };
    const Endcaps blue_endcaps{ compute_endcaps(msg->bar_blue) };

    _endcaps_file << timestamp << " " << idx << " ";
    _endcaps_file << convert_to<std::string>(red_endcaps.first) << " ";
    _endcaps_file << convert_to<std::string>(red_endcaps.second) << " ";
    _endcaps_file << convert_to<std::string>(green_endcaps.first) << " ";
    _endcaps_file << convert_to<std::string>(green_endcaps.second) << " ";
    _endcaps_file << convert_to<std::string>(blue_endcaps.first) << " ";
    _endcaps_file << convert_to<std::string>(blue_endcaps.second) << " ";
    _endcaps_file << "\n";
  }

  Eigen::Vector3d _offset;

  ros::Subscriber _tensegrity_bars_subscriber;

  std::shared_ptr<node_status_t> _status;

  std::ofstream _poses_file;
  std::ofstream _endcaps_file;
};
}  // namespace interface