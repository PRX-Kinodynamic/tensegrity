#include <iostream>
#include <cstdlib>
#include <tuple>
#include <ros/ros.h>

#include <algorithm>
#include <optional>
#include <deque>

#include <std_msgs/Bool.h>
#include <geometry_msgs/PoseArray.h>
#include <visualization_msgs/Marker.h>

#include <factor_graphs/defs.hpp>

#include <tensegrity_utils/type_conversions.hpp>
#include <tensegrity_utils/std_utils.hpp>
#include <tensegrity_utils/dbg_utils.hpp>
#include <tensegrity_utils/rosparams_utils.hpp>

#include <interface/TensegrityEndcaps.h>
#include <interface/TensegrityBars.h>
#include <interface/node_status.hpp>

#include <estimation/bar_utilities.hpp>
#include <estimation/endcap_subscriber.hpp>
#include <estimation/tensegrity_cap_obs_factor.hpp>
#include <estimation/endcap_observation_factor.hpp>
#include <estimation/SE3_observation_factor.hpp>
#include <estimation/cable_sensor_subscriber.hpp>

#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/basis/FitBasis.h>
#include <gtsam/basis/Chebyshev2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

using tensegrity::utils::convert_to;
using SF = factor_graphs::symbol_factory_t;
using Translation = Eigen::Vector3d;

using Rotation = gtsam::Rot3;
using SE3 = gtsam::Pose3;
using Velocity = Eigen::Vector<double, 6>;

// using SE3ObsFactor = estimation::endcap_observation_t<SE3>;
using SE3ObsFactor = estimation::SE3_observation_factor_t<SE3>;
using NewSE3ObsFactor = estimation::SE3_symmetric_observation_factor_t;
// using ScrewAxis = prx::fg::screw_axis_t;
using Integrator = factor_graphs::lie_integration_factor_t<SE3, Velocity>;
// using Integrator = prx::fg::lie_integration_factor_t<SE3, ScrewAxis>;
// using ScrewSmothing = prx::fg::screw_smoothing_factor_t;
using Graph = gtsam::NonlinearFactorGraph;
using Values = gtsam::Values;

using OptSE3 = std::optional<SE3>;
using OptTranslation = std::optional<Translation>;
using RodObservation = std::pair<OptTranslation, OptTranslation>;
// using ChebManifoldSE3 = manifold_evaluation_t<gtsam::Chebyshev2, SE3>;

// const Translation offset({ 0, 0, 0.325 / 2.0 });
const Translation offset({ 0, 0, 0.325 / 2.0 });
const Rotation Roffset(0.0, 0.0, 1.0, 0.0);

void add_two_observations(const estimation::RodColors color, const int idx, const Translation zA, const Translation zB,
                          Graph& graph, Values& values)
{
  using BarEstimationFactor = estimation::bar_two_observations_factor_t;

  // using BarRotationFactor = rotation_symmetric_factor_t;
  gtsam::noiseModel::Base::shared_ptr z_noise{ gtsam::noiseModel::Isotropic::Sigma(3, 1e-1) };
  const gtsam::Key key_se3{ rod_symbol(color, idx) };
  const gtsam::Key key_rot_offset{ rotation_symbol(color, idx) };

  // values.insert(key_se3, SE3());
  // values.insert(key_rot_offset, Rotation());
  graph.emplace_shared<BarEstimationFactor>(key_se3, key_rot_offset, zA, offset, Roffset,
                                            BarEstimationFactor::ObservationIdx::First, z_noise);
  graph.emplace_shared<BarEstimationFactor>(key_se3, key_rot_offset, zB, offset, Roffset,
                                            BarEstimationFactor::ObservationIdx::Second, z_noise);
}

// void add_one_observation(const RodColors color, const int idx, const Translation z, Graph& graph, Values& values)
// {
//   using BarEstimationFactor = estimation::bar_two_observations_factor_t;

//   gtsam::noiseModel::Base::shared_ptr z_noise{ gtsam::noiseModel::Isotropic::Sigma(3, 1) };

//   const gtsam::Key key_se3{ rod_symbol(color, idx) };
//   const gtsam::Key key_rot_offset{ rotation_symbol(color, idx) };
//   graph.emplace_shared<BarEstimationFactor>(key_se3, key_rot_offset, zA, offset, Roffset,
//                                             BarEstimationFactor::ObservationIdx::First, z_noise);
// }

RodObservation get_observations(rod_callback_t& rod)
{
  Translation zA, zB;
  if (rod.observations.size() == 0)
    return { {}, {} };

  interface::TensegrityEndcaps obs{ rod.observations.back() };  //  Get the last one (more recent one) always
  rod.observations.clear();

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

// Add to values only if it is not there yet
template <typename ValueType>
void add_to_values(Values& values, const gtsam::Key& key, const ValueType v)
{
  if (not values.exists(key))
  {
    values.insert(key, v);
  }
}

void add_cable_meassurements(estimation::cables_callback_t& cables_callback, const int idx, Graph& graph,
                             const double cable_unit_conversion)
{
  using CablesFactor = estimation::cable_length_observations_factor_t;
  if (cables_callback.meassurements.size() == 0)
    return;

  gtsam::noiseModel::Base::shared_ptr z_noise{ gtsam::noiseModel::Isotropic::Sigma(1, 1) };
  const interface::TensegrityLengthSensor sensors{ cables_callback.meassurements.back() };
  const auto zi = sensors.length;

  cables_callback.meassurements.clear();

  const gtsam::Key key_Xred{ rod_symbol(estimation::RodColors::RED, idx) };
  const gtsam::Key key_Xgreen{ rod_symbol(estimation::RodColors::GREEN, idx) };
  const gtsam::Key key_Xblue{ rod_symbol(estimation::RodColors::BLUE, idx) };
  const gtsam::Key key_Rred{ rotation_symbol(estimation::RodColors::RED, idx) };
  const gtsam::Key key_Rgreen{ rotation_symbol(estimation::RodColors::GREEN, idx) };
  const gtsam::Key key_Rblue{ rotation_symbol(estimation::RodColors::BLUE, idx) };

  // DEBUG_VARS(zi);
  graph.emplace_shared<CablesFactor>(key_Xblue, key_Rblue, key_Xgreen, key_Rgreen,
                                     zi[0] * cable_unit_conversion,  // no-lint
                                     offset, Roffset, CablesFactor::CableDistanceId::BB, z_noise);
  graph.emplace_shared<CablesFactor>(key_Xgreen, key_Rgreen, key_Xred, key_Rred,
                                     zi[1] * cable_unit_conversion,  // no-lint
                                     offset, Roffset, CablesFactor::CableDistanceId::BB, z_noise);
  graph.emplace_shared<CablesFactor>(key_Xred, key_Rred, key_Xblue, key_Rblue,
                                     zi[2] * cable_unit_conversion,  // no-lint
                                     offset, Roffset, CablesFactor::CableDistanceId::BB, z_noise);

  graph.emplace_shared<CablesFactor>(key_Xgreen, key_Rgreen, key_Xred, key_Rred,
                                     zi[3] * cable_unit_conversion,  // no-lint
                                     offset, Roffset, CablesFactor::CableDistanceId::AA, z_noise);
  graph.emplace_shared<CablesFactor>(key_Xred, key_Rred, key_Xblue, key_Rblue,
                                     zi[4] * cable_unit_conversion,  // no-lint
                                     offset, Roffset, CablesFactor::CableDistanceId::AA, z_noise);
  graph.emplace_shared<CablesFactor>(key_Xblue, key_Rblue, key_Xgreen, key_Rgreen,
                                     zi[5] * cable_unit_conversion,  // no-lint
                                     offset, Roffset, CablesFactor::CableDistanceId::AA, z_noise);

  graph.emplace_shared<CablesFactor>(key_Xgreen, key_Rgreen, key_Xblue, key_Rblue,
                                     zi[6] * cable_unit_conversion,  // no-lint
                                     offset, Roffset, CablesFactor::CableDistanceId::AB, z_noise);
  graph.emplace_shared<CablesFactor>(key_Xred, key_Rred, key_Xgreen, key_Rgreen,
                                     zi[7] * cable_unit_conversion,  // no-lint
                                     offset, Roffset, CablesFactor::CableDistanceId::AB, z_noise);
  graph.emplace_shared<CablesFactor>(key_Xblue, key_Rblue, key_Xred, key_Rred,
                                     zi[8] * cable_unit_conversion,  // no-lint
                                     offset, Roffset, CablesFactor::CableDistanceId::AB, z_noise);
}

int main(int argc, char* argv[])
{
  using Callback = const boost::function<void(const interface::TensegrityEndcapsConstPtr)>;

  ros::init(argc, argv, "EKFEstimation");
  ros::NodeHandle nh("~");

  std::string red_endcaps_topic, blue_endcaps_topic, green_endcaps_topic;
  std::string tensegrity_pose_topic;
  std::string cables_topic;

  bool use_cable_sensors{ true };

  double cable_unit_conversion{ 1.0 };

  PARAM_SETUP(nh, red_endcaps_topic);
  PARAM_SETUP(nh, blue_endcaps_topic);
  PARAM_SETUP(nh, green_endcaps_topic);
  PARAM_SETUP(nh, tensegrity_pose_topic);
  PARAM_SETUP(nh, cables_topic);
  PARAM_SETUP(nh, cable_unit_conversion);
  PARAM_SETUP_WITH_DEFAULT(nh, use_cable_sensors, use_cable_sensors)

  interface::node_status_t node_status(nh, "/nodes/ekf/");

  ros::Publisher tensegrity_bars_publisher{ nh.advertise<interface::TensegrityBars>(tensegrity_pose_topic, 1, true) };

  rod_callback_t red_rod(nh, red_endcaps_topic);
  rod_callback_t blue_rod(nh, blue_endcaps_topic);
  rod_callback_t green_rod(nh, green_endcaps_topic);
  estimation::cables_callback_t cables_callback(nh, cables_topic);

  PRINT_MSG("Starting to listen")

  gtsam::LevenbergMarquardtParams lm_params;
  // lm_params.setVerbosityLM("SILENT");
  lm_params.setVerbosityLM("SUMMARY");
  lm_params.setMaxIterations(20);

  int idx{ 0 };

  ros::Rate rate(30);

  SE3 red_bar{ factor_graphs::random<SE3>() };
  SE3 green_bar{ factor_graphs::random<SE3>() };
  SE3 blue_bar{ factor_graphs::random<SE3>() };

  factor_graphs::levenberg_marquardt_t lm_helper(nh, "/nodes/state_estimation/fg", lm_params);

  node_status.status(interface::NodeStatus::READY);
  bool first_red{ false };
  bool first_green{ false };
  bool first_blue{ false };
  while (ros::ok())
  {
    if (node_status.status() == interface::NodeStatus::RUNNING)
    {
      Graph graph;
      Values values;
      int total_constraints{ 0 };
      bool run_fg{ false };

      const RodObservation red_observations{ get_observations(red_rod) };
      const RodObservation green_observations{ get_observations(green_rod) };
      const RodObservation blue_observations{ get_observations(blue_rod) };

      if (red_observations.first and red_observations.second)  // Two observations
      {
        const Translation& zA{ red_observations.first.value() };
        const Translation& zB{ red_observations.second.value() };
        add_two_observations(estimation::RodColors::RED, idx, zA, zB, graph, values);
        total_constraints += 6;
        // run_fg = true;
        add_to_values(values, rod_symbol(estimation::RodColors::RED, idx), red_bar);
        add_to_values(values, rotation_symbol(estimation::RodColors::RED, idx), Rotation());
        first_red = true;
      }
      // else if (use_cable_sensors and (red_observations.first or red_observations.second))  // One observation only
      // {
      //   const bool first_valid{ red_observations.first };
      //   const bool second_valid{ red_observations.second };
      //   // PRINT_MSG("RED")
      //   // DEBUG_VARS(first_valid, second_valid);
      //   // There is one value, either in first or second ==> The Zero is never returned but needed for completeness
      //   const Translation& z{ red_observations.first.value_or(red_observations.second.value_or(Translation::Zero()))
      //   }; add_one_observation(RodColors::RED, idx, z, graph, values); total_constraints += 3; add_to_values(values,
      //   rod_symbol(RodColors::RED, idx), factor_graphs::random<SE3>());
      // }

      if (green_observations.first and green_observations.second)
      {
        const Translation& zA{ green_observations.first.value() };
        const Translation& zB{ green_observations.second.value() };
        add_two_observations(estimation::RodColors::GREEN, idx, zA, zB, graph, values);
        total_constraints += 6;
        // run_fg = true;

        add_to_values(values, rod_symbol(estimation::RodColors::GREEN, idx), green_bar);
        add_to_values(values, rotation_symbol(estimation::RodColors::GREEN, idx), Rotation());
        first_green = true;
      }

      if (blue_observations.first and blue_observations.second)
      {
        const Translation& zA{ blue_observations.first.value() };
        const Translation& zB{ blue_observations.second.value() };
        add_two_observations(estimation::RodColors::BLUE, idx, zA, zB, graph, values);
        total_constraints += 6;
        // run_fg = true;

        add_to_values(values, rod_symbol(estimation::RodColors::BLUE, idx), blue_bar);
        add_to_values(values, rotation_symbol(estimation::RodColors::BLUE, idx), Rotation());
        first_blue = true;
      }

      if (not use_cable_sensors)
      {
        run_fg = true;
        // add_to_values(values, rod_symbol(estimation::RodColors::RED, idx), red_bar);
        // add_to_values(values, rod_symbol(estimation::RodColors::GREEN, idx), green_bar);
        // add_to_values(values, rod_symbol(estimation::RodColors::BLUE, idx), blue_bar);
      }
      // else if (use_cable_sensors and total_constraints >= 6)
      else if (use_cable_sensors and first_red and first_green and first_blue)
      {
        DEBUG_VARS(total_constraints, use_cable_sensors);
        add_cable_meassurements(cables_callback, idx, graph, cable_unit_conversion);
        run_fg = true;
        // add_to_values(values, rod_symbol(estimation::RodColors::RED, idx), red_bar);
        // add_to_values(values, rod_symbol(estimation::RodColors::GREEN, idx), green_bar);
        // add_to_values(values, rod_symbol(estimation::RodColors::BLUE, idx), blue_bar);
        // add_to_values(values, rotation_symbol(estimation::RodColors::RED, idx), Rotation());
        // add_to_values(values, rotation_symbol(estimation::RodColors::GREEN, idx), Rotation());
        // add_to_values(values, rotation_symbol(estimation::RodColors::BLUE, idx), Rotation());

        const gtsam::noiseModel::Base::shared_ptr rot_noise{ gtsam::noiseModel::Isotropic::Sigma(3, 1e-5) };

        const gtsam::Key key_rot_red{ rotation_symbol(estimation::RodColors::RED, idx) };
        const gtsam::Key key_rot_green{ rotation_symbol(estimation::RodColors::GREEN, idx) };
        const gtsam::Key key_rot_blue{ rotation_symbol(estimation::RodColors::BLUE, idx) };

        if (not values.exists(key_rot_red))
        {
          graph.addPrior(key_rot_red, Rotation(), rot_noise);
          add_to_values(values, key_rot_red, Rotation());
        }
        if (not values.exists(key_rot_green))
        {
          graph.addPrior(key_rot_green, Rotation(), rot_noise);
          add_to_values(values, key_rot_green, Rotation());
        }
        if (not values.exists(key_rot_blue))
        {
          graph.addPrior(key_rot_blue, Rotation(), rot_noise);
          add_to_values(values, key_rot_blue, Rotation());
        }
      }

      if (run_fg)
      {
        gtsam::noiseModel::Base::shared_ptr prior_noise{ gtsam::noiseModel::Isotropic::Sigma(6, 1e-1) };

        if (first_red)
        {
          graph.addPrior(rod_symbol(estimation::RodColors::RED, idx), red_bar, prior_noise);
          add_to_values(values, rod_symbol(estimation::RodColors::RED, idx), red_bar);
        }
        if (first_green)
        {
          graph.addPrior(rod_symbol(estimation::RodColors::GREEN, idx), green_bar, prior_noise);
          add_to_values(values, rod_symbol(estimation::RodColors::GREEN, idx), green_bar);
        }
        if (first_blue)
        {
          graph.addPrior(rod_symbol(estimation::RodColors::BLUE, idx), blue_bar, prior_noise);
          add_to_values(values, rod_symbol(estimation::RodColors::BLUE, idx), blue_bar);
        }

        // DEBUG_VARS(first_red, first_green, first_blue);
        // values.print("values", SF::formatter);
        const Values result{ lm_helper.optimize(graph, values, true) };

        update_pose(red_bar, estimation::RodColors::RED, idx, result);
        update_pose(green_bar, estimation::RodColors::GREEN, idx, result);
        update_pose(blue_bar, estimation::RodColors::BLUE, idx, result);

        DEBUG_VARS(red_bar);

        estimation::publish_tensegrity_msg(red_bar, green_bar, blue_bar, tensegrity_bars_publisher, red_rod.frame);

        idx++;
      }
    }
    else if (node_status.status() == interface::NodeStatus::STOPPED)
    {
      break;
    }
    ros::spinOnce();
    rate.sleep();
  }
  node_status.status(interface::NodeStatus::STOPPED);
  ros::spinOnce();
  rate.sleep();

  return 0;
}
