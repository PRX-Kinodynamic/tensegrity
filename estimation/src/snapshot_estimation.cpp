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

#include <estimation/endcap_subscriber.hpp>
#include <estimation/tensegrity_cap_obs_factor.hpp>
#include <estimation/endcap_observation_factor.hpp>
#include <estimation/SE3_observation_factor.hpp>
// #include <estimation/cable_sensor_subscriber.hpp>
#include <estimation/bar_utilities.hpp>
#include <estimation/sensors_subscriber.hpp>

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

using estimation::RodColors;
using OptSE3 = std::optional<SE3>;
using OptTranslation = std::optional<Translation>;
using RodObservation = std::pair<OptTranslation, OptTranslation>;
// using ChebManifoldSE3 = manifold_evaluation_t<gtsam::Chebyshev2, SE3>;
using RotationFixIdentity = estimation::rotation_fix_identity_t;
using RotationOffsetFactor = estimation::endcap_rotation_offset_factor_t;

// const Translation offset({ 0, 0, 0.325 / 2.0 });
const Translation offset({ 0, 0, 0.325 / 2.0 });
const Rotation Roffset(0.0, 0.0, 1.0, 0.0);

bool update_fg(const RodObservation& observations, const estimation::RodColors color,
               gtsam::NonlinearFactorGraph& graph, gtsam::Values& values,
               estimation::observation_update_t& observation_update)
{
  if (observations.first and observations.second)  // Two observations
  {
    observation_update.idx = 0;
    observation_update.j = 0;
    observation_update.color = color;
    observation_update.zA = observations.first.value();
    observation_update.zB = observations.second.value();
    observation_update.pose = factor_graphs::random<SE3>();
    add_two_observations(observation_update, graph, values);
    return true;
  }
  return false;
}

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "SnapshotEstimation");
  ros::NodeHandle nh("~");

  std::string red_endcaps_topic, blue_endcaps_topic, green_endcaps_topic;
  std::string tensegrity_pose_topic;
  std::string poses_filename{ "" };
  bool autostart{ false };

  PARAM_SETUP(nh, red_endcaps_topic);
  PARAM_SETUP(nh, blue_endcaps_topic);
  PARAM_SETUP(nh, green_endcaps_topic);
  PARAM_SETUP(nh, tensegrity_pose_topic);
  PARAM_SETUP_WITH_DEFAULT(nh, poses_filename, poses_filename);
  PARAM_SETUP_WITH_DEFAULT(nh, autostart, autostart);

  interface::node_status_t node_status(nh, "/nodes/state_estimation/", false);
  const bool save_poses{ poses_filename != "" };
  std::ofstream poses_file;
  if (save_poses)
    poses_file.open(poses_filename, std::ios::out);

  ros::Publisher tensegrity_bars_publisher{ nh.advertise<interface::TensegrityBars>(tensegrity_pose_topic, 1, true) };

  rod_callback_t red_rod(nh, red_endcaps_topic);
  rod_callback_t blue_rod(nh, blue_endcaps_topic);
  rod_callback_t green_rod(nh, green_endcaps_topic);

  PRINT_MSG("Starting to listen")

  gtsam::LevenbergMarquardtParams lm_params;
  lm_params.setVerbosityLM("SILENT");
  // lm_params.setVerbosityLM("SUMMARY");
  lm_params.setMaxIterations(10);

  int idx{ 0 };
  ros::Rate rate(30);  // sleep for 1 sec

  SE3 red_bar{ factor_graphs::random<SE3>() };
  SE3 green_bar{ factor_graphs::random<SE3>() };
  SE3 blue_bar{ factor_graphs::random<SE3>() };

  factor_graphs::levenberg_marquardt_t lm_helper(nh, "/nodes/state_estimation/fg", lm_params);

  estimation::observation_update_t observation_update;
  observation_update.offset = offset;
  observation_update.Roffset = Roffset;
  // estimation::ColorMapping color_map{ estimation::create_cable_map(cable_map_filename) };

  node_status.status(interface::NodeStatus::READY);
  std::string red_str, green_str, blue_str;
  const std::string invalid_str{ "NaN NaN NaN NaN NaN NaN NaN " };
  if (autostart)
    node_status.status(interface::NodeStatus::RUNNING);
  while (ros::ok())
  {
    if (node_status.status() == interface::NodeStatus::RUNNING)
    {
      Graph graph;
      Values values;
      int total_constraints{ 0 };
      bool run_fg{ false };

      const RodObservation red_observations{ red_rod.get_last_observation() };
      const RodObservation green_observations{ green_rod.get_last_observation() };
      const RodObservation blue_observations{ blue_rod.get_last_observation() };

      const bool valid_red{ update_fg(red_observations, estimation::RodColors::RED, graph, values,
                                      observation_update) };
      const bool valid_green{ update_fg(green_observations, estimation::RodColors::GREEN, graph, values,
                                        observation_update) };
      const bool valid_blue{ update_fg(blue_observations, estimation::RodColors::BLUE, graph, values,
                                       observation_update) };

      if (valid_red or valid_green or valid_blue)
      {
        // graph.print("graph", SF::formatter);
        // values.print("values", SF::formatter);
        Values result{ lm_helper.optimize(graph, values, true) };

        if (valid_red)
        {
          update_pose(red_bar, estimation::RodColors::RED, 0, result);
          red_str = tensegrity::utils::convert_to<std::string>(red_bar);
        }
        else
        {
          red_str = invalid_str;
        }
        if (valid_green)
        {
          update_pose(green_bar, estimation::RodColors::GREEN, 0, result);
          green_str = tensegrity::utils::convert_to<std::string>(green_bar);
        }
        else
        {
          green_str = invalid_str;
        }
        if (valid_blue)
        {
          update_pose(blue_bar, estimation::RodColors::BLUE, 0, result);
          blue_str = tensegrity::utils::convert_to<std::string>(blue_bar);
        }
        else
        {
          blue_str = invalid_str;
        }
        idx = red_rod.seq;

        poses_file << idx << " ";
        poses_file << red_rod.ti << " ";
        poses_file << red_str << " ";
        poses_file << green_str << " ";
        poses_file << blue_str << " ";
        poses_file << "\n";

        estimation::publish_tensegrity_msg(red_bar, green_bar, blue_bar, tensegrity_bars_publisher, red_rod.frame, idx);
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
