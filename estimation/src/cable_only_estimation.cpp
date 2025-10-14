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

#include <estimation/bar_utilities.hpp>
#include <estimation/endcap_subscriber.hpp>
#include <estimation/tensegrity_cap_obs_factor.hpp>
#include <estimation/endcap_observation_factor.hpp>
#include <estimation/SE3_observation_factor.hpp>
// #include <estimation/cable_sensor_subscriber.hpp>
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

using OptSE3 = std::optional<SE3>;
using OptTranslation = std::optional<Translation>;
using RodObservation = std::pair<OptTranslation, OptTranslation>;
// using ChebManifoldSE3 = manifold_evaluation_t<gtsam::Chebyshev2, SE3>;

// const Translation offset({ 0, 0, 0.325 / 2.0 });
const Translation offset({ 0, 0, 0.325 / 2.0 });
const Rotation Roffset(0.0, 0.0, 1.0, 0.0);

int main(int argc, char* argv[])
{
  using Callback = const boost::function<void(const interface::TensegrityEndcapsConstPtr)>;

  ros::init(argc, argv, "StateEstimation");
  ros::NodeHandle nh("~");

  // std::string cables_topic;
  std::string tensegrity_pose_topic;
  std::string cable_map_filename;
  // PARAM_SETUP(nh, cables_topic);
  PARAM_SETUP(nh, tensegrity_pose_topic);
  PARAM_SETUP(nh, cable_map_filename);
  // PARAM_SETUP(nh, red_poly_topic);
  // PARAM_SETUP(nh, red_observations);
  // PARAM_SETUP(nh, stop_topic);
  // PARAM_SETUP_WITH_DEFAULT(nh, max_time, max_time)
  // PARAM_SETUP_WITH_DEFAULT(nh, N, N)
  // rosrun estimation state_estimation _red_endcaps_topic:=/tensegrity/endcap/red/positions
  // _blue_endcaps_topic:=/tensegrity/endcap/blue/positions _green_endcaps_topic:=/tensegrity/endcap/green/positions
  // _tensegrity_pose_topic:=/tensegrity/poses

  ros::Publisher tensegrity_bars_publisher{ nh.advertise<interface::TensegrityBars>(tensegrity_pose_topic, 1, true) };
  // estimation::cables_callback_t cables_callback(nh, cables_topic);
  estimation::sensors_callback_t sensors_callback(nh);
  // rod_callback_t red_rod(nh, red_endcaps_topic);
  // rod_callback_t blue_rod(nh, blue_endcaps_topic);
  // rod_callback_t green_rod(nh, green_endcaps_topic);

  // stop_t stop(nh, stop_topic);

  PRINT_MSG("Starting to listen")

  // int idx{ 0 };
  // int green_idx{ 0 };
  // int blue_idx{ 0 };
  ros::Rate rate(30);  // sleep for 1 sec

  SE3 red_bar{ factor_graphs::random<SE3>() };
  SE3 green_bar{ factor_graphs::random<SE3>() };
  SE3 blue_bar{ factor_graphs::random<SE3>() };

  using CablesFactor = estimation::cable_length_observations_factor_t;
  gtsam::noiseModel::Base::shared_ptr z_noise{ gtsam::noiseModel::Isotropic::Sigma(1, 1) };
  gtsam::noiseModel::Base::shared_ptr prior_Xnoise{ gtsam::noiseModel::Isotropic::Sigma(6, 1e-5) };

  estimation::ColorMapping cable_map{ estimation::create_cable_map(cable_map_filename) };

  gtsam::LevenbergMarquardtParams lm_params;

  lm_params.setVerbosityLM("SILENT");
  // lm_params.setVerbosityLM("SUMMARY");
  lm_params.setMaxIterations(10);
  // const gtsam::Values result{ estimation::compute_initialization(initial_estimate_filename, offset, Roffset) };
  factor_graphs::levenberg_marquardt_t lm_helper(nh, "/nodes/state_estimation/fg", lm_params);

  estimation::observation_update_t observation_update;
  observation_update.offset = offset;
  observation_update.Roffset = Roffset;

  estimation::sensors_callback_t::OptMeassurements meassurements;
  estimation::sensors_callback_t::Cables cables;

  while (ros::ok())
  {
    Graph graph;
    Values values;

    meassurements = sensors_callback.get_last_meassurements();
    // const bool valid{ sensors_callback.get_data(cables) };
    // const std::vector<double> meassurements{ cables_callback.get_last_observation() };
    // DEBUG_VARS(meassurements.has_value())
    if (meassurements.has_value())
    {
      // PRINT_MSG("Got new meassurements")
      const std::size_t idx{ std::get<2>(*meassurements) };
      cables = std::get<0>(*meassurements);

      const gtsam::Key key_Xred{ estimation::rod_symbol(estimation::RodColors::RED, idx) };
      const gtsam::Key key_Xgreen{ estimation::rod_symbol(estimation::RodColors::GREEN, idx) };
      const gtsam::Key key_Xblue{ estimation::rod_symbol(estimation::RodColors::BLUE, idx) };
      // const gtsam::Key key_Rred{ estimation::rotation_symbol(estimation::RodColors::RED, idx) };
      // const gtsam::Key key_Rgreen{ estimation::rotation_symbol(estimation::RodColors::GREEN, idx) };
      // const gtsam::Key key_Rblue{ estimation::rotation_symbol(estimation::RodColors::BLUE, idx) };

      values.insert(key_Xred, red_bar);
      values.insert(key_Xgreen, green_bar);
      values.insert(key_Xblue, blue_bar);
      // values.insert(key_Rred, Rotation());
      // values.insert(key_Rgreen, Rotation());
      // values.insert(key_Rblue, Rotation());

      estimation::add_cable_meassurements(observation_update, cables, idx, graph, values, cable_map);

      graph.addPrior<SE3>(key_Xred, SE3(gtsam::Rot3(0.707, 0.707, 0.0, 0.0), Eigen::Vector3d::Zero()), prior_Xnoise);
      // graph.addPrior<Rotation>(key_Rred, Rotation());

      const Values result{ lm_helper.optimize(graph, values, true) };

      // graph.printErrors(result, "Errors: ", SF::formatter);
      estimation::update_pose(red_bar, estimation::RodColors::RED, idx, result);
      estimation::update_pose(green_bar, estimation::RodColors::GREEN, idx, result);
      estimation::update_pose(blue_bar, estimation::RodColors::BLUE, idx, result);

      // DEBUG_VARS(red_bar);
      // DEBUG_VARS(green_bar);

      estimation::publish_tensegrity_msg(red_bar, green_bar, blue_bar, tensegrity_bars_publisher, "world", idx);
      // break;
    }

    ros::spinOnce();
    rate.sleep();
  }

  return 0;
}
