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
#include <interface/TensegrityTrajectory.h>
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

using LieIntegrator = factor_graphs::lie_integration_factor_t<SE3, Velocity>;
// using ChebManifoldSE3 = manifold_evaluation_t<gtsam::Chebyshev2, SE3>;

// const Translation offset({ 0, 0, 0.325 / 2.0 });

struct traj_estimation_t
{
  using This = traj_estimation_t;

  ros::Timer _graph_timer, _publisher_timer, _traj_pub_timer;
  ros::Subscriber _red_subscriber, _green_subscriber, _blue_subscriber;
  ros::Publisher _tensegrity_bars_publisher, _tensegrity_traj_publisher;

  Graph graph;
  Values values;

  std::array<SE3, 3> _bars;
  const std::array<estimation::RodColors, 1> _colors;

  int _idx;
  // int rod_idx;
  // std::array<int, 3> rotation_idx;
  double graph_resolution;
  double optimization_frequency;
  double publisher_frequency;

  ros::Time prev_dt;

  gtsam::LevenbergMarquardtParams _lm_params;
  std::shared_ptr<factor_graphs::levenberg_marquardt_t> _lm_helper;
  std::shared_ptr<interface::node_status_t> node_status;

  std::shared_ptr<estimation::cables_callback_t> cables_callback;

  std::shared_ptr<rod_callback_t> red_rod;
  std::shared_ptr<rod_callback_t> blue_rod;
  std::shared_ptr<rod_callback_t> green_rod;

  estimation::observation_update_t _observation_update;

  Translation _offset;
  Rotation _Roffset;

  ros::Time prev_timepoint;

  traj_estimation_t(ros::NodeHandle& nh)
    : _idx(0)
    , _offset({ 0, 0, 0.325 / 2.0 })
    , _Roffset(0.0, 0.0, 1.0, 0.0)
    , node_status(interface::node_status_t::create(nh, "/nodes/traj_estimation/"))
    , prev_dt(0)
    , _bars({ factor_graphs::random<SE3>(), factor_graphs::random<SE3>(), factor_graphs::random<SE3>() })
    , _colors({ estimation::RodColors::RED })
  // , _colors({ estimation::RodColors::RED, estimation::RodColors::GREEN, estimation::RodColors::BLUE })
  {
    // interface::node_status_t node_status(nh, "/nodes/traj_estimation/");
    _observation_update.offset = _offset;
    _observation_update.Roffset = _Roffset;

    // lm_params.setVerbosityLM("SILENT");
    _lm_params.setVerbosityLM("SUMMARY");
    _lm_params.setMaxIterations(10);
    _lm_helper = std::make_shared<factor_graphs::levenberg_marquardt_t>(nh, "/nodes/state_estimation/fg", _lm_params);

    std::string red_endcaps_topic, blue_endcaps_topic, green_endcaps_topic;
    std::string tensegrity_pose_topic, tensegrity_traj_topic;
    std::string cables_topic;
    std::string initial_estimate_filename;
    // double resolution;
    bool use_cable_sensors{ true };
    double trajectory_pub_frequency;

    PARAM_SETUP(nh, red_endcaps_topic);
    PARAM_SETUP(nh, blue_endcaps_topic);
    PARAM_SETUP(nh, green_endcaps_topic);
    PARAM_SETUP(nh, graph_resolution);        // in seconds
    PARAM_SETUP(nh, optimization_frequency);  // How fast to run the optimizer
    PARAM_SETUP(nh, publisher_frequency)
    PARAM_SETUP(nh, trajectory_pub_frequency)

    PARAM_SETUP(nh, tensegrity_pose_topic);
    PARAM_SETUP(nh, tensegrity_traj_topic)
    PARAM_SETUP(nh, initial_estimate_filename)

    red_rod = std::make_shared<rod_callback_t>(nh, red_endcaps_topic);
    blue_rod = std::make_shared<rod_callback_t>(nh, blue_endcaps_topic);
    green_rod = std::make_shared<rod_callback_t>(nh, green_endcaps_topic);

    cables_callback = std::make_shared<estimation::cables_callback_t>(nh, cables_topic);

    // const gtsam::Key key_Xred0{ rod_symbol(estimation::RodColors::RED, rod_idx) };
    // values.insert(key_Xred0, factor_graphs::random<SE3>());

    const ros::Duration graph_timer(1.0 / optimization_frequency);
    const ros::Duration publisher_timer(1.0 / publisher_frequency);
    const ros::Duration traj_pub_timer(1.0 / trajectory_pub_frequency);
    // ros::Subscriber sub = n.subscribe<sensor_msgs::Image> ("image_topic", 10,
    //                                boost::bind(processImagecallback, _1, argc, argv) );
    // _red_subscriber = nh.subscribe<interface::TensegrityEndcaps>(
    //     red_endcaps_topic, 1, boost::bind(boost::mem_fn(&This::endcap_callback), this, _1,
    //     estimation::RodColors::RED));
    // &This::endcap_callback, this);
    // _green_subscriber = nh.subscribe(blue_endcaps_topic, 1, &This::endcap_callback, this);
    // _blue_subscriber = nh.subscribe(green_endcaps_topic, 1, &This::endcap_callback, this);

    _tensegrity_bars_publisher = nh.advertise<interface::TensegrityBars>(tensegrity_pose_topic, 1, true);
    _tensegrity_traj_publisher = nh.advertise<interface::TensegrityTrajectory>(tensegrity_traj_topic, 1, true);

    _traj_pub_timer = nh.createTimer(traj_pub_timer, &This::publish_trajectory, this);
    _graph_timer = nh.createTimer(graph_timer, &This::update, this);
    _publisher_timer = nh.createTimer(publisher_timer, &This::publish_pose, this);

    prev_timepoint = ros::Time::now();

    const gtsam::Values result{ estimation::compute_initialization(initial_estimate_filename, _offset, _Roffset) };

    const gtsam::Key key_red_0{ estimation::rod_symbol(estimation::RodColors::RED, _idx, 0) };
    const gtsam::Key key_green_0{ estimation::rod_symbol(estimation::RodColors::GREEN, _idx, 0) };
    const gtsam::Key key_blue_0{ estimation::rod_symbol(estimation::RodColors::BLUE, _idx, 0) };

    values.insert(key_red_0, result.at<SE3>(key_red_0));
    values.insert(key_green_0, result.at<SE3>(key_green_0));
    values.insert(key_blue_0, result.at<SE3>(key_blue_0));

    graph.addPrior(key_red_0, result.at<SE3>(key_red_0));
    graph.addPrior(key_green_0, result.at<SE3>(key_green_0));
    graph.addPrior(key_blue_0, result.at<SE3>(key_blue_0));

    add_new_step();

    node_status->status(interface::NodeStatus::READY);
  }

  ~traj_estimation_t()
  {
    // node_status->status(interface::NodeStatus::STOPPED);
    // ros::Duration(1.0).sleep();
  }

  bool add_endcap_observations(const estimation::RodColors color, const RodObservation& observations, const double dt)
  {
    using BarVelObservationFactor = estimation::bar_vel_observation_factor_t;
    using RotationOffsetFactor = estimation::endcap_rotation_offset_factor_t;
    using RotationFixIdentity = estimation::rotation_fix_identity_t;
    gtsam::noiseModel::Base::shared_ptr z_noise{ gtsam::noiseModel::Isotropic::Sigma(3, 1e0) };
    gtsam::noiseModel::Base::shared_ptr rot_prior_nm{ gtsam::noiseModel::Isotropic::Sigma(3, 1e-4) };

    const gtsam::Key key_se3{ estimation::rod_symbol(color, _idx, 0) };
    const gtsam::Key key_rotA_offset{ estimation::rotation_symbol(color, _idx, 0) };
    const gtsam::Key key_rotB_offset{ estimation::rotation_symbol(color, _idx, 1) };
    const gtsam::Key key_xdot{ estimation::rodvel_symbol(color, _idx) };

    if (observations.first and observations.second)  // Two observations
    {
      const Translation zA{ *(observations.first) };
      const Translation zB{ *(observations.second) };

      graph.emplace_shared<BarVelObservationFactor>(key_se3, key_rotA_offset, key_xdot, zA, _offset, dt, z_noise);
      graph.emplace_shared<BarVelObservationFactor>(key_se3, key_rotB_offset, key_xdot, zB, _offset, dt, z_noise);

      graph.emplace_shared<RotationOffsetFactor>(key_rotA_offset, key_rotB_offset, _Roffset, rot_prior_nm);
      graph.emplace_shared<RotationOffsetFactor>(key_rotB_offset, key_rotA_offset, _Roffset, rot_prior_nm);

      graph.emplace_shared<RotationFixIdentity>(key_rotA_offset, key_rotB_offset, _Roffset, rot_prior_nm);

      // _observation_update.color = color;
      // _observation_update.zA = *(observations.first);
      // _observation_update.zB = *(observations.second);
      // _observation_update.idx = _idx;
      // _observation_update.j++;
      // _observation_update.pose = values.at<SE3>(estimation::rod_symbol(color, _idx, 0));
      const SE3 xcurr{ values.at<SE3>(estimation::rod_symbol(color, _idx, 0)) };

      // estimation::add_two_observations(_observation_update, graph, values);

      estimation::add_to_values(values, key_se3, xcurr);
      estimation::add_to_values(values, key_rotA_offset, Rotation());
      estimation::add_to_values(values, key_rotB_offset, _Roffset.inverse());
      return true;
    }
    return false;
  }

  void add_new_step()
  {
    gtsam::noiseModel::Base::shared_ptr xdot_nm{ gtsam::noiseModel::Isotropic::Sigma(6, 1e-1) };

    for (auto color : _colors)
    {
      const gtsam::Key key_0{ estimation::rod_symbol(color, _idx, 0) };
      const gtsam::Key key_1{ estimation::rod_symbol(color, _idx + 1, 0) };
      const gtsam::Key key_xdot{ estimation::rodvel_symbol(color, _idx) };

      graph.emplace_shared<LieIntegrator>(key_1, key_0, key_xdot, xdot_nm, graph_resolution);

      const gtsam::Key key_xdot_prev{ estimation::rodvel_symbol(color, _idx - 1) };
      const SE3 x0{ values.at<SE3>(key_0) };
      if (values.exists(key_xdot_prev))
      {
        const Velocity xdot{ values.at<Velocity>(key_xdot_prev) };

        const SE3 x1_pred{ LieIntegrator::predict(x0, xdot, graph_resolution) };
        estimation::add_to_values(values, key_xdot, xdot);
        estimation::add_to_values(values, key_1, x1_pred);
      }
      else
      {
        // If there is no prev xdot (at the beginnning), just replicate the next one
        const Velocity zero{ Velocity::Zero() };
        estimation::add_to_values(values, key_xdot, zero);
        estimation::add_to_values(values, key_1, x0);
      }
    }
  }

  void update(const ros::TimerEvent& event)
  {
    if (node_status->status() != interface::NodeStatus::RUNNING)
    {
      prev_timepoint = ros::Time::now();
      return;
    }

    const ros::Duration duration_real{ event.current_real - prev_timepoint };

    const double dt{ duration_real.toSec() };
    if (dt >= graph_resolution)
    {
      _idx++;
      add_new_step();
      prev_timepoint = ros::Time::now();
    }
    // DEBUG_VARS(_idx, dt, graph_resolution);

    const RodObservation red_observations{ red_rod->get_last_observation() };
    const RodObservation green_observations{ green_rod->get_last_observation() };
    const RodObservation blue_observations{ blue_rod->get_last_observation() };

    const std::vector<double> cables{ cables_callback->get_last_observation() };

    bool z_added{ false };
    z_added |= add_endcap_observations(estimation::RodColors::RED, red_observations, dt);

    if (z_added)
    {
      values = _lm_helper->optimize(graph, values, true);
      // graph.printErrors(values, "Errors", SF::formatter);
    }
  }

  void publish_pose(const ros::TimerEvent& event)
  {
    if (node_status->status() != interface::NodeStatus::RUNNING)
    {
      return;
    }

    for (auto color : _colors)
    {
      const gtsam::Key key_0{ estimation::rod_symbol(color, _idx, 0) };
      const gtsam::Key key_1{ estimation::rod_symbol(color, _idx + 1, 0) };
      const SE3 x0{ values.at<SE3>(key_0) };
      const SE3 x1{ values.at<SE3>(key_1) };
      const ros::Duration tdiff{ ros::Time::now() - prev_timepoint };
      const double dt{ tdiff.toSec() / graph_resolution };

      const SE3 x01{ x0.interpolateRt(x1, dt) };
      _bars[color] = x01;

      PRINT_KEYS(key_0, key_1);
      DEBUG_VARS(_idx, dt)
      // DEBUG_VARS(x0)
      // DEBUG_VARS(x1)
      // DEBUG_VARS(x01)
    }
    estimation::publish_tensegrity_msg(_bars[estimation::RodColors::RED], _bars[estimation::RodColors::GREEN],
                                       _bars[estimation::RodColors::BLUE], _tensegrity_bars_publisher, red_rod->frame);
  }

  void publish_trajectory(const ros::TimerEvent& event)
  {
    interface::TensegrityTrajectory traj;
    const int i_curr{ _idx };
    const gtsam::Values curr_values{ values };  // make a copy to avoid race conditions

    traj.header.stamp = ros::Time::now();
    traj.header.frame_id = red_rod->frame;

    std::array<SE3, 3> bars_aux;

    for (int i = 0; i < i_curr; ++i)
    {
      for (auto color : _colors)
      {
        const gtsam::Key key_x{ estimation::rod_symbol(color, i, 0) };
        const SE3 pose{ values.at<SE3>(key_x) };
        bars_aux[color] = pose;
      }
      traj.trajectory.emplace_back();
      interface::copy(traj.trajectory.back().bar_red, bars_aux[estimation::RodColors::RED]);
      interface::copy(traj.trajectory.back().bar_green, bars_aux[estimation::RodColors::GREEN]);
      interface::copy(traj.trajectory.back().bar_blue, bars_aux[estimation::RodColors::BLUE]);

      // const SE3 green_pose{};
      // const SE3 blue_pose{};

      // interface::copy(bars.bar_green, green_pose);
      // interface::copy(bars.bar_blue, blue_pose);
    }

    _tensegrity_traj_publisher.publish(traj);
  }
};

int main(int argc, char* argv[])
{
  using Callback = const boost::function<void(const interface::TensegrityEndcapsConstPtr)>;

  ros::init(argc, argv, "TrajectoryEstimation");
  ros::NodeHandle nh("~");

  traj_estimation_t estimator(nh);
  ros::spin();

  return 0;
}
