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

#include <interface/TensegrityBars.h>
#include <interface/node_status.hpp>

#include <resource_retriever/retriever.h>

#include <visualization_msgs/MarkerArray.h>

namespace interface
{
template <typename Base>
class tensegrity_bars_to_markers_t : public Base
{
  using Derived = tensegrity_bars_to_markers_t<Base>;

public:
  tensegrity_bars_to_markers_t()
  {
  }

  ~tensegrity_bars_to_markers_t(){};

  virtual void onInit()
  {
    ros::NodeHandle& private_nh{ Base::getPrivateNodeHandle() };

    std::string tensegrity_bars_topicname;
    std::string tensegrity_markers_topicname;
    std::string path_topic_name;
    std::string endcap_topic_name;
    std::string node_id;
    double alpha{ 1.0 };

    PARAM_SETUP(private_nh, node_id);
    PARAM_SETUP(private_nh, tensegrity_bars_topicname);
    PARAM_SETUP(private_nh, tensegrity_markers_topicname);
    PARAM_SETUP_WITH_DEFAULT(private_nh, alpha, alpha);

    _status = std::make_shared<node_status_t>(private_nh, node_id);
    _status->status(interface::NodeStatus::PREPARING);

    _tensegrity_bars_subscriber = private_nh.subscribe(tensegrity_bars_topicname, 1, &Derived::callback, this);

    const std::string red_marker_topicname{ tensegrity_markers_topicname + "/red" };
    const std::string green_marker_topicname{ tensegrity_markers_topicname + "/green" };
    const std::string blue_marker_topicname{ tensegrity_markers_topicname + "/blue" };

    _red_marker_publisher = private_nh.advertise<visualization_msgs::Marker>(red_marker_topicname, 1, true);
    _green_marker_publisher = private_nh.advertise<visualization_msgs::Marker>(green_marker_topicname, 1, true);
    _blue_marker_publisher = private_nh.advertise<visualization_msgs::Marker>(blue_marker_topicname, 1, true);

    tensegrity::utils::init_header(_red_marker.header, "world");
    tensegrity::utils::init_header(_green_marker.header, "world");
    tensegrity::utils::init_header(_blue_marker.header, "world");

    _red_marker.type = visualization_msgs::Marker::MESH_RESOURCE;
    _green_marker.type = visualization_msgs::Marker::MESH_RESOURCE;
    _blue_marker.type = visualization_msgs::Marker::MESH_RESOURCE;

    const std::string path{ ros::package::getPath("interface") };

    _red_marker.action = visualization_msgs::Marker::ADD;
    _green_marker.action = visualization_msgs::Marker::ADD;
    _blue_marker.action = visualization_msgs::Marker::ADD;

    _red_marker.scale.x = 1.0;
    _red_marker.scale.y = 1.0;
    _red_marker.scale.z = 1.0;

    _green_marker.scale = _red_marker.scale;
    _blue_marker.scale = _red_marker.scale;

    _red_marker.color.r = 1.0;
    _red_marker.color.g = 0.0;
    _red_marker.color.b = 0.0;
    _red_marker.color.a = alpha;

    _green_marker.color.r = 0.0;
    _green_marker.color.g = 1.0;
    _green_marker.color.b = 0.0;
    _green_marker.color.a = alpha;

    _blue_marker.color.r = 0.0;
    _blue_marker.color.g = 0.0;
    _blue_marker.color.b = 1.0;
    _blue_marker.color.a = alpha;

    const std::string bar_url{
      "https://raw.githubusercontent.com/PRX-Kinodynamic/tensegrity/719a15d19519c6aa86ea06a61fedd7598fdde87c/"
      "infraestructure/interface/models/bars/struct_with_socks.glb"
    };
    _red_marker.mesh_resource = bar_url;
    _green_marker.mesh_resource = bar_url;
    _blue_marker.mesh_resource = bar_url;
    _status->status(interface::NodeStatus::READY);
    _status->status(interface::NodeStatus::RUNNING);
  }

private:
  void callback(const interface::TensegrityBarsConstPtr msg)
  {
    _red_marker.header = msg->header;
    _green_marker.header = msg->header;
    _blue_marker.header = msg->header;

    _red_marker.pose = msg->bar_red;
    _green_marker.pose = msg->bar_green;
    _blue_marker.pose = msg->bar_blue;

    // _markers.markers.clear();
    // _markers.markers.push_back(_red_marker);
    // _markers.markers.push_back(_green_marker);
    // _markers.markers.push_back(_blue_marker);
    // DEBUG_VARS(_red_marker.pose);
    // DEBUG_VARS(_green_marker.pose);
    // DEBUG_VARS(_blue_marker.pose);
    // _markers_publisher.publish(_markers);
    _red_marker_publisher.publish(_red_marker);
    _green_marker_publisher.publish(_green_marker);
    _blue_marker_publisher.publish(_blue_marker);
  }

  visualization_msgs::Marker _red_marker, _green_marker, _blue_marker;
  ros::Publisher _red_marker_publisher, _green_marker_publisher, _blue_marker_publisher;

  ros::Subscriber _tensegrity_bars_subscriber;

  std::shared_ptr<node_status_t> _status;
};
}  // namespace interface