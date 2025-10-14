#include <set>
#include <thread>

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

#include <interface/TensegrityBarsArray.h>
#include <interface/node_status.hpp>
// #include <interface/type_conversions.hpp>
#include <interface/defs.hpp>
#include <gtsam/geometry/Pose3.h>

#include <resource_retriever/retriever.h>

#include <visualization_msgs/MarkerArray.h>

namespace interface
{
template <typename Base>
class tensegrity_array_to_markers_t : public Base
{
  using Derived = tensegrity_array_to_markers_t<Base>;

public:
  tensegrity_array_to_markers_t() : offset_p({ 0, 0, 0.325 / 2.0 }), offset_m({ 0, 0, -0.325 / 2.0 })
  {
  }

  ~tensegrity_array_to_markers_t() {};

  virtual void onInit()
  {
    ros::NodeHandle& private_nh{ Base::getPrivateNodeHandle() };

    std::string tensegrity_trajectory_topicname;
    // std::string tensegrity_markers_topicname;
    // std::string path_topic_name;
    // std::string endcap_topic_name;
    std::string node_id;
    bool& use_mesh{ _use_mesh };
    PARAM_SETUP(private_nh, node_id);
    PARAM_SETUP(private_nh, use_mesh);
    PARAM_SETUP(private_nh, tensegrity_trajectory_topicname);
    // PARAM_SETUP(private_nh, tensegrity_markers_topicname);

    _status = std::make_shared<node_status_t>(private_nh, node_id);
    _status->status(interface::NodeStatus::PREPARING);

    const std::string marker_topicname{ tensegrity_trajectory_topicname + "/marker" };
    const std::string marker_array_topicname{ tensegrity_trajectory_topicname + "/marker_array" };
    if (use_mesh)
    {
      _marker_array_publisher = private_nh.advertise<visualization_msgs::MarkerArray>(marker_array_topicname, 1, true);
    }
    else
    {
      _marker_publisher = private_nh.advertise<visualization_msgs::Marker>(marker_topicname, 1, true);
    }

    _tensegrity_bars_subscriber = private_nh.subscribe(tensegrity_trajectory_topicname, 1, &Derived::callback, this);

    tensegrity::utils::init_header(_marker.header, "world");

    // const std::string path{ ros::package::getPath("interface") };

    if (_use_mesh)
    {
      const std::string bar_url{
        "https://raw.githubusercontent.com/PRX-Kinodynamic/tensegrity/719a15d19519c6aa86ea06a61fedd7598fdde87c/"
        "infraestructure/interface/models/bars/struct_with_socks.glb"
      };
      _marker.type = visualization_msgs::Marker::MESH_RESOURCE;
      _marker.action = visualization_msgs::Marker::ADD;
      _marker.scale.x = 1.0;
      _marker.mesh_resource = bar_url;
    }
    else
    {
      _marker.type = visualization_msgs::Marker::LINE_LIST;
      _marker.action = visualization_msgs::Marker::ADD;
      _marker.scale.x = 0.02;
    }

    _status->status(interface::NodeStatus::READY);
    _status->status(interface::NodeStatus::RUNNING);
  }

private:
  void callback(const interface::TensegrityBarsArrayConstPtr msg)
  {
    _markers.markers.clear();
    _marker.points.clear();
    _marker.colors.clear();
    _marker.header = msg->header;

    gtsam::Pose3 pose;
    for (int i = 0; i < msg->bars.size(); ++i)
    {
      if (_use_mesh)
      {
        _markers.markers.push_back(_marker);
        _markers.markers.back().header.stamp = msg->header.stamp;
        _markers.markers.back().id = i;
        _markers.markers.back().pose = msg->bars[i];
        _markers.markers.back().color = msg->colors[i];
      }
      else
      {
        interface::copy(pose, msg->bars[i]);
        const Eigen::Vector3d ptp{ pose.transformFrom(offset_p) };
        const Eigen::Vector3d ptm{ pose.transformFrom(offset_m) };
        _marker.points.emplace_back();
        interface::copy(_marker.points.back(), ptp);
        _marker.points.emplace_back();
        interface::copy(_marker.points.back(), ptm);
        _marker.colors.emplace_back(msg->colors[i]);
        _marker.colors.emplace_back(msg->colors[i]);
      }
      // i++;
    }

    if (_use_mesh)
    {
      _marker_array_publisher.publish(_markers);
    }
    else
    {
      _marker_publisher.publish(_marker);
    }
  }

  bool _use_mesh;

  visualization_msgs::Marker _marker;
  visualization_msgs::MarkerArray _markers;
  ros::Publisher _marker_publisher, _marker_array_publisher;

  ros::Subscriber _tensegrity_bars_subscriber;

  std::shared_ptr<node_status_t> _status;
  Eigen::Vector3d offset_p, offset_m;
};
}  // namespace interface