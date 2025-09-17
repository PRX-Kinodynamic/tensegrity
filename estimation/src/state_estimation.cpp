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

// gtsam::Key rotation_symbol(const RodColors rod, const int t, const int ab)
// {
//   return SF::create_hashed_symbol("R^{", color_str(rod), "}_{", t, "_", ab, "}");
// }

gtsam::Key rodvel_symbol(const int rod, const double t)
{
  return SF::create_hashed_symbol("\\dot{X}^{", rod, "}_{", t, "}");
}

inline void print(const gtsam::Pose3& p, std::string msg)
{
  auto tr = p.translation().transpose();
  auto q = p.rotation().toQuaternion();
  DEBUG_VARS(msg, tr, q);
}

void add_two_observations(const RodColors color, const int idx, const Translation zA, const Translation zB,
                          Graph& graph, Values& values)
{
  using BarEstimationFactor = estimation::bar_two_observations_factor_t;

  using EndcapObservationFactor = estimation::endcap_observation_factor_t;
  using RotationOffsetFactor = estimation::endcap_rotation_offset_factor_t;
  // using BarRotationFactor = rotation_symmetric_factor_t;
  gtsam::noiseModel::Base::shared_ptr z_noise{ gtsam::noiseModel::Isotropic::Sigma(3, 1e-1) };
  gtsam::noiseModel::Base::shared_ptr rot_prior_nm{ gtsam::noiseModel::Isotropic::Sigma(3, 1e-4) };
  const gtsam::Key key_se3{ estimation::rod_symbol(color, idx) };
  const gtsam::Key key_rotA_offset{ estimation::rotation_symbol(color, idx, 0) };
  const gtsam::Key key_rotB_offset{ estimation::rotation_symbol(color, idx, 1) };

  graph.emplace_shared<EndcapObservationFactor>(key_se3, key_rotA_offset, zA, offset, z_noise);
  graph.emplace_shared<EndcapObservationFactor>(key_se3, key_rotB_offset, zB, offset, z_noise);

  graph.emplace_shared<RotationOffsetFactor>(key_rotA_offset, key_rotB_offset, Roffset, rot_prior_nm);
  graph.emplace_shared<RotationOffsetFactor>(key_rotB_offset, key_rotA_offset, Roffset, rot_prior_nm);

  // graph.emplace_shared<gtsam::BetweenFactor<gtsam::Rot3>>(key_rotA_offset, key_rotB_offset, Roffset, rot_prior_nm);
  // graph.emplace_shared<gtsam::BetweenFactor<gtsam::Rot3>>(key_rotB_offset, key_rotA_offset, Roffset, rot_prior_nm);

  graph.emplace_shared<RotationFixIdentity>(key_rotA_offset, key_rotB_offset, Roffset, rot_prior_nm);
  // values.insert(key_se3, SE3());
  // values.insert(key_rot_offset, Rotation());
  // graph.emplace_shared<BarEstimationFactor>(key_se3, key_rot_offset, zA, offset, Roffset,
  //                                           BarEstimationFactor::ObservationIdx::First, z_noise);
  // graph.emplace_shared<BarEstimationFactor>(key_se3, key_rot_offset, zB, offset, Roffset,
  //                                           BarEstimationFactor::ObservationIdx::Second, z_noise);
  // graph.addPrior(key_rot_offset, Rotation(), rot_prior_nm);
}

void add_one_observation(const RodColors color, const int idx, const Translation z, Graph& graph, Values& values)
{
  using BarEstimationFactor = estimation::bar_two_observations_factor_t;

  using EndcapObservationFactor = estimation::endcap_observation_factor_t;

  const gtsam::Key key_se3{ rod_symbol(color, idx) };
  const gtsam::Key key_rotA_offset{ estimation::rotation_symbol(color, idx, 0) };
  const gtsam::Key key_rotB_offset{ estimation::rotation_symbol(color, idx, 1) };
  gtsam::noiseModel::Base::shared_ptr z_noise{ gtsam::noiseModel::Isotropic::Sigma(3, 1e-1) };
  gtsam::noiseModel::Base::shared_ptr rot_prior_nm{ gtsam::noiseModel::Isotropic::Sigma(3, 1e-4) };

  graph.emplace_shared<EndcapObservationFactor>(key_se3, key_rotA_offset, z, offset, z_noise);

  graph.emplace_shared<RotationOffsetFactor>(key_rotA_offset, key_rotB_offset, Roffset, rot_prior_nm);
  graph.emplace_shared<RotationOffsetFactor>(key_rotB_offset, key_rotA_offset, Roffset, rot_prior_nm);

  graph.emplace_shared<RotationFixIdentity>(key_rotA_offset, key_rotB_offset, Roffset, rot_prior_nm);
  // graph.addPrior(key_rotB_offset, Rotation(), rot_prior_nm);
}

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
    // DEBUG_VARS(zA.transpose(), zB.transpose());
    const double normAB{ (zA - zB).norm() };
    // DEBUG_VARS(obs.endcaps.size(), normAB);
    return { zA, zB };
  }
  if (obs.endcaps.size() == 1)
  {
    interface::copy(zA, obs.endcaps[0]);
    return { zA, {} };
  }

  return { {}, {} };  // Pair of empty elements.
}

void add_cable_meassurements(estimation::sensors_callback_t& sensors_callback, const int idx, Graph& graph,
                             Values& values, estimation::ColorMapping& cable_map)
{
  // using CablesFactor = estimation::cable_length_observations_factor_t;
  using CablesFactor = estimation::cable_length_no_rotation_factor_t;
  // if (cables_callback.meassurements.size() == 0)
  //   return;

  gtsam::noiseModel::Base::shared_ptr z_noise{ gtsam::noiseModel::Isotropic::Sigma(1, 1) };
  gtsam::noiseModel::Base::shared_ptr rot_prior_nm{ gtsam::noiseModel::Isotropic::Sigma(3, 1e-4) };
  // const interface::TensegrityLengthSensor sensors{ cables_callback.meassurements.back() };
  // const auto zi = sensors.length;
  // cables_callback.meassurements.clear();
  // const std::vector<double> zi{ cables_callback.get_last_observation() };
  estimation::sensors_callback_t::Cables cables;
  estimation::sensors_callback_t::OptMeassurements meassurements;
  meassurements = sensors_callback.get_last_meassurements();
  // const bool valid{ sensors_callback.get_data(cables) };

  if (not meassurements)
  {
    return;
  }
  cables = std::get<0>(*meassurements);

  for (int i = 0; i < 9; ++i)
  {
    const gtsam::Key key_Xi{ rod_symbol(cable_map[i].first, idx) };
    const gtsam::Key key_Xj{ rod_symbol(cable_map[i].second, idx) };
    const gtsam::Key key_Ri{ rotation_symbol(cable_map[i].first, idx, 2 + i) };
    const gtsam::Key key_Rj{ rotation_symbol(cable_map[i].second, idx, 2 + i) };

    // PRINT_KEYS(key_Xi, key_Xj);
    values.insert(key_Ri, Rotation());
    values.insert(key_Rj, Rotation());
    graph.emplace_shared<CablesFactor>(key_Xi, key_Xj, key_Ri, cables[i], offset, z_noise);
    // graph.emplace_shared<RotationFixIdentity>(key_Ri, key_Rj, rot_prior_nm);

    graph.emplace_shared<RotationOffsetFactor>(key_Ri, key_Rj, Roffset, rot_prior_nm);
    graph.emplace_shared<RotationOffsetFactor>(key_Rj, key_Ri, Roffset, rot_prior_nm);

    graph.emplace_shared<RotationFixIdentity>(key_Ri, key_Rj, Roffset, rot_prior_nm);
  }
  // graph.emplace_shared<CablesFactor>(key_Xblue, key_Rblue, key_Xgreen, key_Rgreen, zi[0],  // no-lint
  //                                    offset, Roffset, CablesFactor::CableDistanceId::BB, z_noise);
  // graph.emplace_shared<CablesFactor>(key_Xgreen, key_Rgreen, key_Xred, key_Rred, zi[1],  // no-lint
  //                                    offset, Roffset, CablesFactor::CableDistanceId::BB, z_noise);
  // graph.emplace_shared<CablesFactor>(key_Xred, key_Rred, key_Xblue, key_Rblue, zi[2],  // no-lint
  //                                    offset, Roffset, CablesFactor::CableDistanceId::BB, z_noise);

  // graph.emplace_shared<CablesFactor>(key_Xgreen, key_Rgreen, key_Xred, key_Rred, zi[3],  // no-lint
  //                                    offset, Roffset, CablesFactor::CableDistanceId::AA, z_noise);
  // graph.emplace_shared<CablesFactor>(key_Xred, key_Rred, key_Xblue, key_Rblue, zi[4],  // no-lint
  //                                    offset, Roffset, CablesFactor::CableDistanceId::AA, z_noise);
  // graph.emplace_shared<CablesFactor>(key_Xblue, key_Rblue, key_Xgreen, key_Rgreen, zi[5],  // no-lint
  //                                    offset, Roffset, CablesFactor::CableDistanceId::AA, z_noise);

  // graph.emplace_shared<CablesFactor>(key_Xgreen, key_Rgreen, key_Xblue, key_Rblue, zi[6],  // no-lint
  //                                    offset, Roffset, CablesFactor::CableDistanceId::AB, z_noise);
  // graph.emplace_shared<CablesFactor>(key_Xred, key_Rred, key_Xgreen, key_Rgreen, zi[7],  // no-lint
  //                                    offset, Roffset, CablesFactor::CableDistanceId::AB, z_noise);
  // graph.emplace_shared<CablesFactor>(key_Xblue, key_Rblue, key_Xred, key_Rred, zi[8],  // no-lint
  //                                    offset, Roffset, CablesFactor::CableDistanceId::AB, z_noise);

  // graph.addPrior<SE3>(key_Xred, SE3(), prior_Xnoise);
  // graph.addPrior<Rotation>(key_Rred, Rotation());

  // gtsam::LevenbergMarquardtOptimizer optimizer(graph, values, lm_params);
  // const Values result{ optimizer.optimize() };

  // update_pose(red_bar, RodColors::RED, red_idx, result);
  // update_pose(green_bar, RodColors::GREEN, green_idx, result);
  // update_pose(blue_bar, RodColors::BLUE, blue_idx, result);

  // DEBUG_VARS(red_bar);
  // DEBUG_VARS(green_bar);

  // publish_tensegrity_msg(red_bar, green_bar, blue_bar, tensegrity_bars_publisher);
}

void publish_tensegrity_msg(const SE3& red_pose, const SE3& green_pose, const SE3& blue_pose, ros::Publisher& publisher,
                            const std::string frame)
{
  interface::TensegrityBars bars;
  tensegrity::utils::update_header(bars.header);
  bars.header.frame_id = frame;
  // const SE3 red_pose{ get_pose_from_poly(ti, RodColors::RED, polys, values) };
  interface::copy(bars.bar_red, red_pose);
  interface::copy(bars.bar_green, green_pose);
  interface::copy(bars.bar_blue, blue_pose);

  publisher.publish(bars);
}

int main(int argc, char* argv[])
{
  using Callback = const boost::function<void(const interface::TensegrityEndcapsConstPtr)>;

  ros::init(argc, argv, "StateEstimation");
  ros::NodeHandle nh("~");

  std::string red_endcaps_topic, blue_endcaps_topic, green_endcaps_topic;
  // std::string stop_topic;
  // std::string red_observations;
  std::string tensegrity_pose_topic;
  // std::string cables_topic;
  std::string cable_map_filename;
  // int N{ 30 * 3 };
  // double max_time{ -1.0 };
  bool use_cable_sensors{ true };

  PARAM_SETUP(nh, red_endcaps_topic);
  PARAM_SETUP(nh, blue_endcaps_topic);
  PARAM_SETUP(nh, green_endcaps_topic);
  PARAM_SETUP(nh, tensegrity_pose_topic);
  // PARAM_SETUP(nh, cables_topic);
  PARAM_SETUP(nh, cable_map_filename);
  // PARAM_SETUP(nh, frame);
  PARAM_SETUP_WITH_DEFAULT(nh, use_cable_sensors, use_cable_sensors)
  // PARAM_SETUP(nh, red_observations);
  // PARAM_SETUP(nh, stop_topic);
  // PARAM_SETUP_WITH_DEFAULT(nh, max_time, max_time)
  // PARAM_SETUP_WITH_DEFAULT(nh, N, N)

  // rosrun estimation state_estimation _red_endcaps_topic:=/tensegrity/endcap/red/positions
  // _blue_endcaps_topic:=/tensegrity/endcap/blue/positions _green_endcaps_topic:=/tensegrity/endcap/green/positions
  // _tensegrity_pose_topic:=/tensegrity/poses
  interface::node_status_t node_status(nh, "/nodes/state_estimation/");

  // ros::Publisher red_poly_publisher{ nh.advertise<geometry_msgs::PoseArray>(red_poly_topic, 1, true) };
  // ros::Publisher red_observations_publisher{ nh.advertise<visualization_msgs::Marker>(red_observations, 1, true) };
  ros::Publisher tensegrity_bars_publisher{ nh.advertise<interface::TensegrityBars>(tensegrity_pose_topic, 1, true) };

  rod_callback_t red_rod(nh, red_endcaps_topic);
  rod_callback_t blue_rod(nh, blue_endcaps_topic);
  rod_callback_t green_rod(nh, green_endcaps_topic);
  // estimation::cables_callback_t cables_callback(nh, cables_topic);
  estimation::sensors_callback_t sensors_callback(nh);
  estimation::ColorMapping cable_map{ estimation::create_cable_map(cable_map_filename) };
  // stop_t stop(nh, stop_topic);

  PRINT_MSG("Starting to listen")

  gtsam::LevenbergMarquardtParams lm_params;
  lm_params.setVerbosityLM("SILENT");
  // lm_params.setVerbosityLM("SUMMARY");
  lm_params.setMaxIterations(10);

  int idx{ 0 };
  // int green_idx{ 0 };
  // int blue_idx{ 0 };
  ros::Rate rate(30);  // sleep for 1 sec

  // SE3 red_bar{ Rotation(), Translation::Random() };
  SE3 red_bar{ factor_graphs::random<SE3>() };
  SE3 green_bar{ factor_graphs::random<SE3>() };
  SE3 blue_bar{ factor_graphs::random<SE3>() };

  factor_graphs::levenberg_marquardt_t lm_helper(nh, "/nodes/state_estimation/fg", lm_params);

  estimation::sensors_callback_t::OptMeassurements meassurements;
  estimation::sensors_callback_t::Cables cables;
  estimation::observation_update_t observation_update;
  observation_update.offset = offset;
  observation_update.Roffset = Roffset;
  // estimation::ColorMapping color_map{ estimation::create_cable_map(cable_map_filename) };

  // DEBUG_VARS(node_status)
  node_status.status(interface::NodeStatus::READY);
  // DEBUG_VARS(node_status)
  while (ros::ok())
  {
    if (node_status.status() == interface::NodeStatus::RUNNING)
    {
      PRINT_MSG_ONCE("Running");
      Graph graph;
      Values values;
      int total_constraints{ 0 };
      bool run_fg{ false };

      const RodObservation red_observations{ get_observations(red_rod) };
      const RodObservation green_observations{ get_observations(green_rod) };
      const RodObservation blue_observations{ get_observations(blue_rod) };

      if (red_observations.first and red_observations.second)  // Two observations
      {
        observation_update.idx = idx;
        observation_update.color = RodColors::RED;
        observation_update.zA = red_observations.first.value();
        observation_update.zB = red_observations.second.value();
        estimation::add_two_observations(observation_update, graph, values);

        total_constraints += 6;
        run_fg = true;
        estimation::add_to_values(values, rod_symbol(RodColors::RED, idx), red_bar);
        estimation::add_to_values(values, rotation_symbol(RodColors::RED, idx, 0), Rotation());
        estimation::add_to_values(values, rotation_symbol(RodColors::RED, idx, 1), Roffset.inverse());
        PRINT_MSG("Two Red observations")
      }
      else if (use_cable_sensors and red_observations.first or red_observations.second)  // One observation only
      {
        const bool first_valid{ red_observations.first };
        const bool second_valid{ red_observations.second };
        PRINT_MSG("RED")
        DEBUG_VARS(first_valid, second_valid);
        // There is one value, either in first or second ==> The Zero is never returned but needed for completeness
        const Translation& z{ red_observations.first.value_or(red_observations.second.value_or(Translation::Zero())) };
        add_one_observation(RodColors::RED, idx, z, graph, values);
        total_constraints += 3;
        estimation::add_to_values(values, rod_symbol(RodColors::RED, idx), red_bar);
        estimation::add_to_values(values, rotation_symbol(RodColors::RED, idx, 0), Rotation());
        estimation::add_to_values(values, rotation_symbol(RodColors::RED, idx, 1), Roffset.inverse());
        // const gtsam::noiseModel::Base::shared_ptr rot_noise{ gtsam::noiseModel::Isotropic::Sigma(1, 1e-5) };
        // graph.addPrior(rotation_symbol(RodColors::RED, idx), Rotation(), rot_noise);
        // add_to_values(values, rotation_symbol(RodColors::RED, idx), Rotation());
      }

      if (green_observations.first and green_observations.second)
      {
        observation_update.idx = idx;
        observation_update.color = RodColors::GREEN;
        observation_update.zA = green_observations.first.value();
        observation_update.zB = green_observations.second.value();
        estimation::add_two_observations(observation_update, graph, values);

        total_constraints += 6;
        run_fg = true;

        estimation::add_to_values(values, rod_symbol(RodColors::GREEN, idx), green_bar);
        estimation::add_to_values(values, rotation_symbol(RodColors::GREEN, idx, 0), Rotation());
        estimation::add_to_values(values, rotation_symbol(RodColors::GREEN, idx, 1), Roffset.inverse());
      }
      else if (use_cable_sensors and green_observations.first or green_observations.second)  // One observation only
      {
        const bool first_valid{ green_observations.first };
        const bool second_valid{ green_observations.second };
        PRINT_MSG("GREEN")
        // DEBUG_VARS(first_valid, second_valid);
        // There is one value, either in first or second
        const Translation& z{ green_observations.first.value_or(
            green_observations.second.value_or(Translation::Zero())) };
        add_one_observation(RodColors::GREEN, idx, z, graph, values);
        total_constraints += 3;
        estimation::add_to_values(values, rod_symbol(RodColors::GREEN, idx), green_bar);
        estimation::add_to_values(values, rotation_symbol(RodColors::GREEN, idx, 0), Rotation());
        estimation::add_to_values(values, rotation_symbol(RodColors::GREEN, idx, 1), Roffset.inverse());
        // const gtsam::noiseModel::Base::shared_ptr rot_noise{ gtsam::noiseModel::Isotropic::Sigma(1, 1e-5) };
        // graph.addPrior(rotation_symbol(RodColors::GREEN, idx), Rotation(), rot_noise);
        // add_to_values(values, rotation_symbol(RodColors::GREEN, idx), Rotation());
      }

      if (blue_observations.first and blue_observations.second)
      {
        observation_update.idx = idx;
        observation_update.color = RodColors::BLUE;
        observation_update.zA = blue_observations.first.value();
        observation_update.zB = blue_observations.second.value();
        estimation::add_two_observations(observation_update, graph, values);

        total_constraints += 6;
        run_fg = true;

        estimation::add_to_values(values, rod_symbol(RodColors::BLUE, idx), blue_bar);
        estimation::add_to_values(values, rotation_symbol(RodColors::BLUE, idx, 0), Rotation());
        estimation::add_to_values(values, rotation_symbol(RodColors::BLUE, idx, 1), Roffset.inverse());
      }
      else if (use_cable_sensors and blue_observations.first or blue_observations.second)  // One observation only
      {
        const bool first_valid{ blue_observations.first };
        const bool second_valid{ blue_observations.second };
        PRINT_MSG("BLUE ")
        DEBUG_VARS(first_valid, second_valid);
        // There is one value, either in first or second
        const Translation& z{ blue_observations.first.value_or(
            blue_observations.second.value_or(Translation::Zero())) };
        add_one_observation(RodColors::BLUE, idx, z, graph, values);
        total_constraints += 3;
        estimation::add_to_values(values, rod_symbol(RodColors::BLUE, idx), blue_bar);
        estimation::add_to_values(values, rotation_symbol(RodColors::BLUE, idx, 0), Rotation());
        estimation::add_to_values(values, rotation_symbol(RodColors::BLUE, idx, 1), Roffset.inverse());
      }

      if (use_cable_sensors and total_constraints >= 12)
      {
        meassurements = sensors_callback.get_last_meassurements();
        if (meassurements.has_value())
        {
          // PRINT_MSG("Got new meassurements")
          // const std::size_t idx{ std::get<2>(*meassurements) };
          cables = std::get<0>(*meassurements);
          estimation::add_cable_meassurements(observation_update, cables, idx, graph, values, cable_map);

          estimation::add_to_values(values, rod_symbol(RodColors::RED, idx), red_bar);
          estimation::add_to_values(values, rod_symbol(RodColors::GREEN, idx), red_bar);
          estimation::add_to_values(values, rod_symbol(RodColors::BLUE, idx), red_bar);
          run_fg = true;
        }
      }

      // graph.print("Graph", SF::formatter);
      if (run_fg)
      {
        // DEBUG_VARS(total_constraints, use_cable_sensors);

        // gtsam::LevenbergMarquardtOptimizer optimizer(graph, values, lm_params);
        // const Values result{ optimizer.optimize() };
        // values.print("Values", SF::formatter);
        // graph.printErrors(values, "Errors", SF::formatter);
        Values result{ lm_helper.optimize(graph, values, true) };

        // result.print("Values", SF::formatter);
        // graph.printErrors(result, "Errors", SF::formatter);
        // const Values result{ calculate_estimate_safe(graph, values, lm_params) };

        // const double fg_error{ graph.error(result) };
        // if (fg_error > 0.5 or std::isinf(fg_error) or std::isnan(fg_error))
        //   DEBUG_VARS(fg_error);
        update_pose(red_bar, RodColors::RED, idx, result);
        update_pose(green_bar, RodColors::GREEN, idx, result);
        update_pose(blue_bar, RodColors::BLUE, idx, result);

        publish_tensegrity_msg(red_bar, green_bar, blue_bar, tensegrity_bars_publisher, red_rod.frame);

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
