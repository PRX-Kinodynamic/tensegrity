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

#include <interface/TensegrityTrajectory.h>
#include <interface/node_status.hpp>

//#include <resource_retriever/retriever.h>

#include <visualization_msgs/MarkerArray.h>

namespace interface
{
template <typename Base>
class tensegrity_trajectory_to_markers_t : public Base
{
  using Derived = tensegrity_trajectory_to_markers_t<Base>;

public:
  tensegrity_trajectory_to_markers_t()
  {
  }

  ~tensegrity_trajectory_to_markers_t() {};

  virtual void onInit()
  {
    ros::NodeHandle& private_nh{ Base::getPrivateNodeHandle() };

    std::string tensegrity_trajectory_topicname;
    // std::string tensegrity_markers_topicname;
    // std::string path_topic_name;
    // std::string endcap_topic_name;
    std::string node_id;
    double alpha{ 1.0 };

    PARAM_SETUP(private_nh, node_id);
    PARAM_SETUP(private_nh, tensegrity_trajectory_topicname);
    // PARAM_SETUP(private_nh, tensegrity_markers_topicname);
    PARAM_SETUP_WITH_DEFAULT(private_nh, alpha, alpha);

    _status = std::make_shared<node_status_t>(private_nh, node_id);
    _status->status(interface::NodeStatus::PREPARING);

    _tensegrity_bars_subscriber = private_nh.subscribe(tensegrity_trajectory_topicname, 1, &Derived::callback, this);

    const std::string red_marker_topicname{ tensegrity_trajectory_topicname + "/red_array" };
    const std::string green_marker_topicname{ tensegrity_trajectory_topicname + "/green_array" };
    const std::string blue_marker_topicname{ tensegrity_trajectory_topicname + "/blue_array" };

    _red_marker_publisher = private_nh.advertise<visualization_msgs::MarkerArray>(red_marker_topicname, 1, true);
    _green_marker_publisher = private_nh.advertise<visualization_msgs::MarkerArray>(green_marker_topicname, 1, true);
    _blue_marker_publisher = private_nh.advertise<visualization_msgs::MarkerArray>(blue_marker_topicname, 1, true);

    tensegrity::utils::init_header(_red_mark.header, "world");
    tensegrity::utils::init_header(_green_mark.header, "world");
    tensegrity::utils::init_header(_blue_mark.header, "world");

    _red_mark.type = visualization_msgs::Marker::MESH_RESOURCE;
    _green_mark.type = visualization_msgs::Marker::MESH_RESOURCE;
    _blue_mark.type = visualization_msgs::Marker::MESH_RESOURCE;

    const std::string path{ ros::package::getPath("interface") };

    _red_mark.action = visualization_msgs::Marker::ADD;
    _green_mark.action = visualization_msgs::Marker::ADD;
    _blue_mark.action = visualization_msgs::Marker::ADD;

    _red_mark.scale.x = 1.0;
    _red_mark.scale.y = 1.0;
    _red_mark.scale.z = 1.0;

    _green_mark.scale = _red_mark.scale;
    _blue_mark.scale = _red_mark.scale;

    _red_mark.color.r = 1.0;
    _red_mark.color.g = 0.0;
    _red_mark.color.b = 0.0;
    _red_mark.color.a = alpha;

    _green_mark.color.r = 0.0;
    _green_mark.color.g = 1.0;
    _green_mark.color.b = 0.0;
    _green_mark.color.a = alpha;

    _blue_mark.color.r = 0.0;
    _blue_mark.color.g = 0.0;
    _blue_mark.color.b = 1.0;
    _blue_mark.color.a = alpha;

    const std::string bar_url{
      "https://raw.githubusercontent.com/PRX-Kinodynamic/tensegrity/719a15d19519c6aa86ea06a61fedd7598fdde87c/"
      "infraestructure/interface/models/bars/struct_with_socks.glb"
    };
    _red_mark.mesh_resource = bar_url;
    _green_mark.mesh_resource = bar_url;
    _blue_mark.mesh_resource = bar_url;
    _status->status(interface::NodeStatus::READY);
    _status->status(interface::NodeStatus::RUNNING);
  }

private:
  void callback(const interface::TensegrityTrajectoryConstPtr msg)
  {
    _red_markers.markers.clear();
    _green_markers.markers.clear();
    _blue_markers.markers.clear();

    _red_mark.header = msg->header;
    _green_mark.header = msg->header;
    _blue_mark.header = msg->header;

    int i{ 0 };
    for (auto state : msg->trajectory)
    {
      _red_markers.markers.push_back(_red_mark);
      _green_markers.markers.push_back(_green_mark);
      _blue_markers.markers.push_back(_blue_mark);

      _red_markers.markers.back().pose = state.bar_red;
      _green_markers.markers.back().pose = state.bar_green;
      _blue_markers.markers.back().pose = state.bar_blue;

      _red_markers.markers.back().header.stamp = msg->header.stamp;
      _red_markers.markers.back().id = i;
      i++;

      // DEBUG_VARS(_red_markers.markers.back());
    }

    // _markers.markers.clear();
    // _markers.markers.push_back(_red_marker);
    // _markers.markers.push_back(_green_marker);
    // _markers.markers.push_back(_blue_marker);
    // DEBUG_VARS(_red_markers.markers.size());
    // DEBUG_VARS(_green_marker.pose);
    // DEBUG_VARS(_blue_marker.pose);
    // _markers_publisher.publish(_markers);

    _red_marker_publisher.publish(_red_markers);
    _green_marker_publisher.publish(_green_markers);
    _blue_marker_publisher.publish(_blue_markers);
  }

  visualization_msgs::Marker _red_mark, _green_mark, _blue_mark;
  visualization_msgs::MarkerArray _red_markers, _green_markers, _blue_markers;
  ros::Publisher _red_marker_publisher, _green_marker_publisher, _blue_marker_publisher;

  ros::Subscriber _tensegrity_bars_subscriber;

  std::shared_ptr<node_status_t> _status;
};
}  // namespace interface
