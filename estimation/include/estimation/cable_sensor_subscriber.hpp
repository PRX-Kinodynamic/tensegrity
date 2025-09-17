#pragma once

#include <interface/TensegrityLengthSensor.h>

namespace estimation
{
struct cables_callback_t
{
  using Translation = Eigen::Vector3d;

  cables_callback_t(ros::NodeHandle& nh, std::string topicname)
  {
    PARAM_SETUP(nh, cable_unit_conversion);
    _subscriber = nh.subscribe(topicname, 10, &cables_callback_t::callback, this);
  }

  void callback(const interface::TensegrityLengthSensorConstPtr msg)
  {
    meassurements.push_back(*msg);
  }

  std::vector<double> get_last_observation()
  {
    std::vector<double> m;
    if (meassurements.size() == 0)
      return m;
    const interface::TensegrityLengthSensor sensors{ meassurements.back() };

    const auto zi = sensors.length;

    for (int i = 0; i < 9; ++i)
    {
      m.push_back(zi[i] * cable_unit_conversion);
    }

    meassurements.clear();
    return m;
  }

  double cable_unit_conversion;
  std::deque<interface::TensegrityLengthSensor> meassurements;
  ros::Subscriber _subscriber;
};

}  // namespace estimation