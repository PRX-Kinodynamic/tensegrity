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
// #include <interface/type_conversions.hpp>
#include <interface/TensegrityEndcaps.h>
// #include <resource_retriever/retriever.h>

#include <visualization_msgs/MarkerArray.h>

namespace interface
{
template <typename Base>
class tensegrity_endcaps_to_file_t : public Base
{
  using Derived = tensegrity_endcaps_to_file_t<Base>;

  using Point = Eigen::Vector3d;

public:
  tensegrity_endcaps_to_file_t()
  {
  }

  ~tensegrity_endcaps_to_file_t()
  {
    _status->status(interface::NodeStatus::FINISH);
    _endcaps_file.close();
  }

  virtual void onInit()
  {
    ros::NodeHandle& private_nh{ Base::getPrivateNodeHandle() };

    std::string tensegrity_endcaps_topicname;
    std::string poses_filename, endcap_filename;
    std::string node_id;
    double alpha{ 1.0 };

    PARAM_SETUP(private_nh, node_id);
    PARAM_SETUP(private_nh, endcap_filename);
    PARAM_SETUP(private_nh, tensegrity_endcaps_topicname);

    // const std::string timestamp{ tensegrity::utils::timestamp() };
    // poses_filename += "_" + timestamp + ".txt";
    // endcap_filename += "_" + timestamp + ".txt";

    _endcaps_file.open(endcap_filename, std::ios::out);

    _status = std::make_shared<node_status_t>(private_nh, node_id);
    _status->status(interface::NodeStatus::PREPARING);

    _tensegrity_endcaps_subscriber = private_nh.subscribe(tensegrity_endcaps_topicname, 1, &Derived::callback, this);

    _endcaps_file << "# Time idx id X Y Z \n";
    _status->status(interface::NodeStatus::READY);
    _status->status(interface::NodeStatus::RUNNING);
  }

private:
  void callback(const interface::TensegrityEndcapsConstPtr msg)
  {
    using tensegrity::utils::convert_to;
    const std::size_t idx{ msg->header.seq };
    const std::string timestamp{ tensegrity::utils::convert_to<std::string>(msg->header.stamp) };

    // const Endcaps red_endcaps{ compute_endcaps(msg->bar_red) };
    // const Endcaps green_endcaps{ compute_endcaps(msg->bar_green) };
    // const Endcaps blue_endcaps{ compute_endcaps(msg->bar_blue) };

    TENSEGRITY_ASSERT(msg->endcaps.size() == msg->ids.size(),
                      "[tensegrity_endcaps_to_file_t] Endcaps size does not match ids");
    for (int i = 0; i < msg->endcaps.size(); ++i)
    {
      _endcaps_file << timestamp << " " << idx << " ";
      _endcaps_file << convert_to<std::string>(static_cast<int>(msg->ids[i])) << " ";
      _endcaps_file << convert_to<std::string>(msg->endcaps[i].x) << " ";
      _endcaps_file << convert_to<std::string>(msg->endcaps[i].y) << " ";
      _endcaps_file << convert_to<std::string>(msg->endcaps[i].z) << " ";
      _endcaps_file << "\n";
    }
  }

  ros::Subscriber _tensegrity_endcaps_subscriber;

  std::shared_ptr<node_status_t> _status;

  std::ofstream _endcaps_file;
};
}  // namespace interface
