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

enum RodColors
{
  RED = 0,
  GREEN,
  BLUE
};

std::string color_str(const RodColors rod)
{
  std::string color;
  switch (rod)
  {
    case RED:
      color = "RED";
      break;
    case GREEN:
      color = "GREEN";
      break;
    case BLUE:
      color = "BLUE";
      break;
    default:
      color = "NA";
  }
  return color;
}
gtsam::Key rod_symbol(const RodColors rod, const int t)
{
  return SF::create_hashed_symbol("X^{", color_str(rod), "}_{", t, "}");
}

gtsam::Key rotation_symbol(const RodColors rod, const int t)
{
  return SF::create_hashed_symbol("R^{", color_str(rod), "}_{", t, "}");
}

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

  // using BarRotationFactor = rotation_symmetric_factor_t;
  gtsam::noiseModel::Base::shared_ptr z_noise{ gtsam::noiseModel::Isotropic::Sigma(3, 1e-2) };
  const gtsam::Key key_se3{ rod_symbol(color, idx) };
  const gtsam::Key key_rot_offset{ rotation_symbol(color, idx) };

  // values.insert(key_se3, SE3());
  // values.insert(key_rot_offset, Rotation());
  graph.emplace_shared<BarEstimationFactor>(key_se3, key_rot_offset, zA, offset, Roffset,
                                            BarEstimationFactor::ObservationIdx::First, z_noise);
  graph.emplace_shared<BarEstimationFactor>(key_se3, key_rot_offset, zB, offset, Roffset,
                                            BarEstimationFactor::ObservationIdx::Second, z_noise);
}

void add_one_observation(const RodColors color, const int idx, const Translation z, Graph& graph, Values& values)
{
  using BarSingleFactor = estimation::SE3_observation_factor_t<SE3>;
  const gtsam::Key key_se3{ rod_symbol(color, idx) };
  gtsam::noiseModel::Base::shared_ptr z_noise{ gtsam::noiseModel::Isotropic::Sigma(3, 1) };
  graph.emplace_shared<BarSingleFactor>(key_se3, offset, z, z_noise);
}

void update_pose(SE3& pose, const RodColors color, const int idx, const Values& values)
{
  const gtsam::Key key_se3{ rod_symbol(color, idx) };
  const gtsam::Key key_rot_offset{ rotation_symbol(color, idx) };

  // only update pose if it has been estimated
  if (values.exists(key_se3) and values.exists(key_rot_offset))
  {
    const SE3 xt{ values.at<SE3>(key_se3) };
    const Rotation Rt{ values.at<Rotation>(key_rot_offset) };
    const Translation pt_zero{ Translation::Zero() };
    const SE3 xr{ SE3(Rt, pt_zero) };
    pose = xt * xr;
    // pose = values.at<SE3>(key_se3);
  }
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
    DEBUG_VARS(zA.transpose(), zB.transpose());
    const double normAB{ (zA - zB).norm() };
    DEBUG_VARS(obs.endcaps.size(), normAB);
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

void add_cable_meassurements(estimation::cables_callback_t& cables_callback, const int idx, Graph& graph)
{
  using CablesFactor = estimation::cable_length_observations_factor_t;
  if (cables_callback.meassurements.size() == 0)
    return;

  gtsam::noiseModel::Base::shared_ptr z_noise{ gtsam::noiseModel::Isotropic::Sigma(1, 1) };
  const interface::TensegrityLengthSensor sensors{ cables_callback.meassurements.back() };
  const auto zi = sensors.length;

  cables_callback.meassurements.clear();

  const gtsam::Key key_Xred{ rod_symbol(RodColors::RED, idx) };
  const gtsam::Key key_Xgreen{ rod_symbol(RodColors::GREEN, idx) };
  const gtsam::Key key_Xblue{ rod_symbol(RodColors::BLUE, idx) };
  const gtsam::Key key_Rred{ rotation_symbol(RodColors::RED, idx) };
  const gtsam::Key key_Rgreen{ rotation_symbol(RodColors::GREEN, idx) };
  const gtsam::Key key_Rblue{ rotation_symbol(RodColors::BLUE, idx) };

  // add_to_values(values, );

  // values.insert(key_Xred, red_bar);
  // values.insert(key_Xgreen, green_bar);
  // values.insert(key_Xblue, blue_bar);
  // values.insert(key_Rred, Rotation());
  // values.insert(key_Rgreen, Rotation());
  // values.insert(key_Rblue, Rotation());

  graph.emplace_shared<CablesFactor>(key_Xblue, key_Rblue, key_Xgreen, key_Rgreen, zi[0],  // no-lint
                                     offset, Roffset, CablesFactor::CableDistanceId::BB, z_noise);
  graph.emplace_shared<CablesFactor>(key_Xgreen, key_Rgreen, key_Xred, key_Rred, zi[1],  // no-lint
                                     offset, Roffset, CablesFactor::CableDistanceId::BB, z_noise);
  graph.emplace_shared<CablesFactor>(key_Xred, key_Rred, key_Xblue, key_Rblue, zi[2],  // no-lint
                                     offset, Roffset, CablesFactor::CableDistanceId::BB, z_noise);

  graph.emplace_shared<CablesFactor>(key_Xgreen, key_Rgreen, key_Xred, key_Rred, zi[3],  // no-lint
                                     offset, Roffset, CablesFactor::CableDistanceId::AA, z_noise);
  graph.emplace_shared<CablesFactor>(key_Xred, key_Rred, key_Xblue, key_Rblue, zi[4],  // no-lint
                                     offset, Roffset, CablesFactor::CableDistanceId::AA, z_noise);
  graph.emplace_shared<CablesFactor>(key_Xblue, key_Rblue, key_Xgreen, key_Rgreen, zi[5],  // no-lint
                                     offset, Roffset, CablesFactor::CableDistanceId::AA, z_noise);

  graph.emplace_shared<CablesFactor>(key_Xgreen, key_Rgreen, key_Xblue, key_Rblue, zi[6],  // no-lint
                                     offset, Roffset, CablesFactor::CableDistanceId::AB, z_noise);
  graph.emplace_shared<CablesFactor>(key_Xred, key_Rred, key_Xgreen, key_Rgreen, zi[7],  // no-lint
                                     offset, Roffset, CablesFactor::CableDistanceId::AB, z_noise);
  graph.emplace_shared<CablesFactor>(key_Xblue, key_Rblue, key_Xred, key_Rred, zi[8],  // no-lint
                                     offset, Roffset, CablesFactor::CableDistanceId::AB, z_noise);

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
  std::string cables_topic;
  // int N{ 30 * 3 };
  // double max_time{ -1.0 };
  bool use_cable_sensors{ true };

  PARAM_SETUP(nh, red_endcaps_topic);
  PARAM_SETUP(nh, blue_endcaps_topic);
  PARAM_SETUP(nh, green_endcaps_topic);
  PARAM_SETUP(nh, tensegrity_pose_topic);
  PARAM_SETUP(nh, cables_topic);
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
  estimation::cables_callback_t cables_callback(nh, cables_topic);

  // stop_t stop(nh, stop_topic);

  PRINT_MSG("Starting to listen")

  gtsam::LevenbergMarquardtParams lm_params;
  // lm_params.setVerbosityLM("SILENT");
  lm_params.setVerbosityLM("SUMMARY");
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

  // DEBUG_VARS(node_status)
  node_status.status(interface::NodeStatus::READY);
  // DEBUG_VARS(node_status)
  while (ros::ok())
  {
  }

  return 0;
}
