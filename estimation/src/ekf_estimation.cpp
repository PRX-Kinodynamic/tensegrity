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
#include <estimation/ekf.hpp>

#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/basis/FitBasis.h>
#include <gtsam/basis/Chebyshev2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

using tensegrity::utils::convert_to;
using SF = factor_graphs::symbol_factory_t;
using Translation = Eigen::Vector3d;

// using ChebManifoldSE3 = manifold_evaluation_t<gtsam::Chebyshev2, SE3>;

// const Translation offset({ 0, 0, 0.325 / 2.0 });

// void add_two_observations(const estimation::RodColors color, const int idx, const Translation zA, const Translation
// zB,
//                           Graph& graph, Values& values)
// {
//   using BarEstimationFactor = estimation::bar_two_observations_factor_t;

//   // using BarRotationFactor = rotation_symmetric_factor_t;
//   gtsam::noiseModel::Base::shared_ptr z_noise{ gtsam::noiseModel::Isotropic::Sigma(3, 1e-1) };
//   const gtsam::Key key_se3{ rod_symbol(color, idx) };
//   const gtsam::Key key_rot_offset{ rotation_symbol(color, idx) };

//   // values.insert(key_se3, SE3());
//   // values.insert(key_rot_offset, Rotation());
//   graph.emplace_shared<BarEstimationFactor>(key_se3, key_rot_offset, zA, offset, Roffset,
//                                             BarEstimationFactor::ObservationIdx::First, z_noise);
//   graph.emplace_shared<BarEstimationFactor>(key_se3, key_rot_offset, zB, offset, Roffset,
//                                             BarEstimationFactor::ObservationIdx::Second, z_noise);
// }

struct ekf_node_t
{
  using This = ekf_node_t;
  using EKF = estimation::extended_kalman_filter_t;

  using Rotation = gtsam::Rot3;
  using SE3 = gtsam::Pose3;
  using Velocity = Eigen::Vector<double, 6>;

  using SE3ObsFactor = estimation::SE3_observation_factor_t<SE3>;
  using NewSE3ObsFactor = estimation::SE3_symmetric_observation_factor_t;
  using Integrator = factor_graphs::lie_integration_factor_t<SE3, Velocity>;
  using Graph = gtsam::NonlinearFactorGraph;
  using Values = gtsam::Values;

  using OptSE3 = std::optional<SE3>;
  using OptTranslation = std::optional<Translation>;
  using RodObservation = std::pair<OptTranslation, OptTranslation>;

  const Translation offset;
  const Rotation Roffset;

  std::string red_endcaps_topic, blue_endcaps_topic, green_endcaps_topic;
  std::string tensegrity_pose_topic;
  std::string cables_topic, initial_estimate_filename;
  double cable_unit_conversion{ 1.0 };

  bool use_cable_sensors{ true };

  ros::Publisher tensegrity_bars_publisher;

  gtsam::LevenbergMarquardtParams lm_params;
  std::shared_ptr<factor_graphs::levenberg_marquardt_t> lm_helper;
  std::shared_ptr<interface::node_status_t> node_status;

  std::shared_ptr<rod_callback_t> red_rod;
  std::shared_ptr<rod_callback_t> blue_rod;
  std::shared_ptr<rod_callback_t> green_rod;

  std::shared_ptr<estimation::cables_callback_t> cables_callback;

  SE3 red_bar{ factor_graphs::random<SE3>() };
  SE3 green_bar{ factor_graphs::random<SE3>() };
  SE3 blue_bar{ factor_graphs::random<SE3>() };
  Velocity red_vel;
  Velocity green_vel;
  Velocity blue_vel;

  ros::Timer _ekf_timer;

  std::shared_ptr<EKF> ekf;

  double accum_dt;

  ekf_node_t(ros::NodeHandle& nh)
    : offset({ 0, 0, 0.325 / 2.0 })
    , Roffset(0.0, 0.0, 1.0, 0.0)
    , node_status(interface::node_status_t::create(nh, "/nodes/ekf/"))
    , red_bar(factor_graphs::random<SE3>())
    , green_bar(factor_graphs::random<SE3>())
    , blue_bar(factor_graphs::random<SE3>())
    , red_vel(Velocity::Zero())
    , green_vel(Velocity::Zero())
    , blue_vel(Velocity::Zero())
    , accum_dt(0)
  {
    double frequency;
    // lm_params.setVerbosityLM("SILENT");
    lm_params.setVerbosityLM("SUMMARY");
    lm_params.setMaxIterations(20);
    lm_helper = std::make_shared<factor_graphs::levenberg_marquardt_t>(nh, "/nodes/state_estimation/fg", lm_params);

    PARAM_SETUP(nh, red_endcaps_topic);
    PARAM_SETUP(nh, blue_endcaps_topic);
    PARAM_SETUP(nh, green_endcaps_topic);
    PARAM_SETUP(nh, tensegrity_pose_topic);
    PARAM_SETUP(nh, cables_topic);
    PARAM_SETUP(nh, cable_unit_conversion);
    PARAM_SETUP(nh, frequency);
    PARAM_SETUP(nh, initial_estimate_filename);
    PARAM_SETUP_WITH_DEFAULT(nh, use_cable_sensors, use_cable_sensors)

    red_rod = std::make_shared<rod_callback_t>(nh, red_endcaps_topic);
    blue_rod = std::make_shared<rod_callback_t>(nh, blue_endcaps_topic);
    green_rod = std::make_shared<rod_callback_t>(nh, green_endcaps_topic);

    cables_callback = std::make_shared<estimation::cables_callback_t>(nh, cables_topic);

    tensegrity_bars_publisher = nh.advertise<interface::TensegrityBars>(tensegrity_pose_topic, 1, true);
    // ros::Publisher tensegrity_bars_publisher{ nh.advertise<interface::TensegrityBars>(tensegrity_pose_topic, 1, true)
    // };

    const ros::Duration ekf_timer(1.0 / frequency);
    _ekf_timer = nh.createTimer(ekf_timer, &This::update, this);
    const gtsam::Values result{ estimation::compute_initialization(initial_estimate_filename, offset, Roffset) };

    // result.print("result", SF::formatter);

    gtsam::noiseModel::Base::shared_ptr z_noise{ gtsam::noiseModel::Isotropic::Sigma(6, 1e0) };

    ekf = std::make_shared<EKF>(nh, use_cable_sensors);

    red_bar = result.at<SE3>(estimation::rod_symbol(estimation::RodColors::RED, 0));
    green_bar = result.at<SE3>(estimation::rod_symbol(estimation::RodColors::GREEN, 0));
    blue_bar = result.at<SE3>(estimation::rod_symbol(estimation::RodColors::BLUE, 0));

    DEBUG_VARS(red_bar)
    DEBUG_VARS(green_bar)
    DEBUG_VARS(blue_bar)

    ekf->set_prior(estimation::RodColors::RED, red_bar, z_noise);
    ekf->set_prior(estimation::RodColors::GREEN, green_bar, z_noise);
    ekf->set_prior(estimation::RodColors::BLUE, blue_bar, z_noise);

    // Assuming this is the correct frame...
    red_rod->frame = "world";
    estimation::publish_tensegrity_msg(red_bar, green_bar, blue_bar, tensegrity_bars_publisher, "world");
    node_status->status(interface::NodeStatus::READY);
  }

  void update(const ros::TimerEvent& event)
  {
    if (node_status->status() != interface::NodeStatus::RUNNING)
      return;
    // DEBUG_VARS(event.current_real, event.last_real);
    const ros::Duration duration_real{ event.current_real - event.last_real };
    const double dt{ duration_real.toSec() };
    accum_dt += dt;

    const RodObservation red_observations{ red_rod->get_last_observation() };
    const RodObservation green_observations{ green_rod->get_last_observation() };
    const RodObservation blue_observations{ blue_rod->get_last_observation() };

    const std::vector<double> cables{ cables_callback->get_last_observation() };

    int total_observations{ 0 };
    total_observations += red_observations.first ? 1 : 0;
    total_observations += red_observations.second ? 1 : 0;
    total_observations += green_observations.first ? 1 : 0;
    total_observations += green_observations.second ? 1 : 0;
    total_observations += blue_observations.first ? 1 : 0;
    total_observations += blue_observations.second ? 1 : 0;

    // DEBUG_VARS(dt, red_vel.transpose());
    if (total_observations >= 1)
    {
      DEBUG_VARS(accum_dt, red_vel.transpose());
      // ekf->predict(red_vel, green_vel, blue_vel, accum_dt);
      ekf->update(red_observations, green_observations, blue_observations, cables);

      const OptSE3 red_bar_next{ ekf->get_estimation(estimation::RodColors::RED) };
      const OptSE3 green_bar_next{ ekf->get_estimation(estimation::RodColors::GREEN) };
      const OptSE3 blue_bar_next{ ekf->get_estimation(estimation::RodColors::BLUE) };

      if (red_bar_next)
      {
        const SE3 red_diff{ (*red_bar_next).between(red_bar) };
        red_vel = SE3::Logmap(red_diff) / accum_dt;
        red_bar = *red_bar_next;
      }
      if (green_bar_next)
      {
        const SE3 green_diff{ (*green_bar_next).between(green_bar) };
        green_vel = SE3::Logmap(green_diff) / accum_dt;
        green_bar = *green_bar_next;
      }
      if (blue_bar_next)
      {
        const SE3 blue_diff{ (*blue_bar_next).between(blue_bar) };
        blue_vel = SE3::Logmap(blue_diff) / accum_dt;
        blue_bar = *blue_bar_next;
      }
      accum_dt = 0;
    }

    DEBUG_VARS(blue_bar);
    estimation::publish_tensegrity_msg(red_bar, green_bar, blue_bar, tensegrity_bars_publisher, red_rod->frame);
  }
};

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "EKFEstimation");
  ros::NodeHandle nh("~");

  // PRINT_MSG("Starting to listen")
  ekf_node_t node(nh);
  ros::spin();

  return 0;
}
