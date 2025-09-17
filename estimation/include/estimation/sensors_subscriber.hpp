#pragma once

#include <tuple>
#include <interface/TensegrityStamped.h>

namespace estimation
{
class sensors_callback_t
{
  using Translation = Eigen::Vector3d;

public:
  using Cables = Eigen::Vector<double, 9>;
  using Motors = Eigen::Vector<double, 6>;
  using OptCables = std::optional<Cables>;
  using Meassurements = std::tuple<Cables, Motors, std::size_t>;
  using OptMeassurements = std::optional<Meassurements>;
  sensors_callback_t(ros::NodeHandle& nh)
  {
    std::string sensors_topicname;
    PARAM_SETUP(nh, sensors_topicname);
    PARAM_SETUP(nh, cable_unit_conversion);
    _subscriber = nh.subscribe(sensors_topicname, 10, &sensors_callback_t::callback, this);
  }

  OptMeassurements get_last_meassurements()
  {
    if (msgs_queue.size() > 0)
    {
      // Meassurements m;
      const interface::TensegrityStamped zi{ msgs_queue.back() };
      msgs_queue.clear();
      const Cables cables{ get_cables(zi) };
      // _idx = zi.header.seq;
      return std::make_tuple(cables, Motors::Zero(), zi.header.seq);
    }
    return {};
  }

private:
  void callback(const interface::TensegrityStampedConstPtr msg)
  {
    // DEBUG_VARS(msg->header.seq);
    msgs_queue.push_back(*msg);
  }

  Cables get_cables(const interface::TensegrityStamped& msg)
  {
    Cables cables;
    for (auto sensor : msg.sensors)
    {
      // DEBUG_VARS(sensor)
      cables[sensor.id] = sensor.length * cable_unit_conversion;
    }
    return cables;
  }

private:
  double cable_unit_conversion;
  std::deque<interface::TensegrityStamped> msgs_queue;
  ros::Subscriber _subscriber;
};

}  // namespace estimation