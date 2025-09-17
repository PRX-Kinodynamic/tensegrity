#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <Eigen/Dense>
#include <Eigen/Core>

#include <ros/console.h>
#include <ros/assert.h>
#include <tensegrity_utils/assert.hpp>
#include <tensegrity_utils/dbg_utils.hpp>
#include <tensegrity_utils/rosparams_utils.hpp>
#include <tensegrity_utils/std_utils.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <visualization_msgs/Marker.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
// #include <message_filters/subscriber.h>
// #include <message_filters/time_synchronizer.h>
#include <opencv2/core/fast_math.hpp>

#include <interface/TensegrityEndcaps.h>
#include <interface/node_status.hpp>

namespace perception
{
template <class Base>
class endcap_to_marker_t : public Base
{
  using Derived = endcap_to_marker_t<Base>;

public:
  endcap_to_marker_t() : _endcap_id(-1), _max_marker_points(10)
  {
  }

private:
  virtual void onInit()
  {
    ros::NodeHandle& private_nh{ Base::getPrivateNodeHandle() };

    std::string node_id{ "" };
    std::string endcap_topic{ "" };
    std::string marker_topic{ "" };

    int& max_endcaps_to_visualize{ _max_marker_points };
    double point_scale{ 0.03 };
    std::vector<double> rgba = { 0.0, 0.0, 0.0, 1.0 };

    PARAM_SETUP(private_nh, node_id);
    PARAM_SETUP(private_nh, endcap_topic);
    PARAM_SETUP_WITH_DEFAULT(private_nh, rgba, rgba);
    PARAM_SETUP_WITH_DEFAULT(private_nh, point_scale, point_scale);
    PARAM_SETUP_WITH_DEFAULT(private_nh, marker_topic, marker_topic);
    PARAM_SETUP_WITH_DEFAULT(private_nh, max_endcaps_to_visualize, max_endcaps_to_visualize);

    _status = std::make_shared<interface::node_status_t>(private_nh, node_id);
    _status->status(interface::NodeStatus::PREPARING);

    if (marker_topic == "")
    {
      marker_topic = endcap_topic + "/marker";
    }

    tensegrity::utils::init_header(_marker.header, "world");
    _marker.action = visualization_msgs::Marker::ADD;
    _marker.type = visualization_msgs::Marker::POINTS;

    _marker.color.r = rgba[0];
    _marker.color.g = rgba[1];
    _marker.color.b = rgba[2];
    _marker.color.a = rgba[3];

    _marker.scale.x = 0.03;  // is point width,
    _marker.scale.y = 0.03;  // is point height

    _endcap_topic = private_nh.subscribe(endcap_topic, 1, &Derived::callback, this);
    _marker_publisher = private_nh.advertise<visualization_msgs::Marker>(marker_topic, 1, true);

    _status->status(interface::NodeStatus::READY);
    _status->status(interface::NodeStatus::RUNNING);
  }

  // void update_marker()
  // {
  // }

  void callback(const interface::TensegrityEndcapsConstPtr msg)
  {
    _marker.header = msg->header;
    if (_marker.points.size() >= _max_marker_points)
    {
      int to_eliminate = _marker.points.size() - _max_marker_points;
      _marker.points.erase(_marker.points.begin(), _marker.points.begin() + to_eliminate);
    }
    if (_max_marker_points > 0)
    {
      _marker.points.insert(_marker.points.end(), msg->endcaps.begin(), msg->endcaps.end());
      _marker_publisher.publish(_marker);
    }
  }

  int _max_marker_points;

  ros::Subscriber _endcap_topic;
  ros::Publisher _marker_publisher;

  int _endcap_id;
  // interface::TensegrityEndcaps _endcap_msg;
  visualization_msgs::Marker _marker;

  std::shared_ptr<interface::node_status_t> _status;
};

}  // namespace perception
