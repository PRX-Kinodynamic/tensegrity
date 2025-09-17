#pragma once
#include <interface/defs.hpp>

struct rod_callback_t
{
  using Translation = Eigen::Vector3d;
  using OptTranslation = std::optional<Translation>;
  using RodObservation = std::pair<OptTranslation, OptTranslation>;

  rod_callback_t(ros::NodeHandle& nh, std::string topicname)
  {
    _red_subscriber = nh.subscribe(topicname, 10, &rod_callback_t::endcap_callback, this);
  }

  void endcap_callback(const interface::TensegrityEndcapsConstPtr msg)
  {
    observations.push_back(*msg);
    remove_invalid_observations(observations.back());
    frame = msg->header.frame_id;
  }

  // Observations may contain NaNs => remove those
  static void remove_invalid_observations(interface::TensegrityEndcaps& observations)
  {
    Translation z;
    // for (auto zin : observations.endcaps)
    std::size_t i{ 0 };
    // bool nan_found{ false };
    while (i < observations.endcaps.size())
    {
      auto zin = observations.endcaps[i];
      interface::copy(z, zin);
      if (std::isnan(z.template maxCoeff<Eigen::PropagateNaN>()))
      {
        observations.endcaps.erase(observations.endcaps.begin() + i);
      }
      else
      {
        i++;
      }
    }
  }

  RodObservation get_last_observation()
  {
    if (observations.size() > 0)
    {
      const RodObservation z{ get_observations(observations.back()) };
      observations.clear();
      return z;
    }
    return { {}, {} };  // Pair of empty elements.
  }

  static RodObservation get_observations(interface::TensegrityEndcaps& obs)
  {
    Translation zA, zB;

    if (obs.endcaps.size() >= 2)
    {
      interface::copy(zA, obs.endcaps[0]);
      interface::copy(zB, obs.endcaps[1]);
      return { zA, zB };
    }
    if (obs.endcaps.size() == 1)
    {
      interface::copy(zA, obs.endcaps[0]);
      return { zA, {} };
    }

    return { {}, {} };  // Pair of empty elements.
  }

  std::string frame;

  std::deque<interface::TensegrityEndcaps> observations;
  ros::Subscriber _red_subscriber;
};
