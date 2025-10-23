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
#include <interface/TensegrityBarsArray.h>
#include <interface/TensegrityTrajectory.h>
#include <interface/node_status.hpp>

#include <estimation/bar_utilities.hpp>
#include <estimation/endcap_subscriber.hpp>
#include <estimation/tensegrity_cap_obs_factor.hpp>
#include <estimation/endcap_observation_factor.hpp>
#include <estimation/SE3_observation_factor.hpp>
#include <estimation/cable_sensor_subscriber.hpp>
#include <estimation/sensors_subscriber.hpp>

#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/basis/FitBasis.h>
#include <gtsam/basis/Chebyshev2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Marginals.h>

using tensegrity::utils::convert_to;
using SF = factor_graphs::symbol_factory_t;
using Translation = Eigen::Vector3d;

using Rotation = gtsam::Rot3;
using SE3 = gtsam::Pose3;
using Velocity = Eigen::Vector<double, 6>;
using Color = Eigen::Vector<double, 3>;

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
using SF = factor_graphs::symbol_factory_t;

const Color black(Color::Zero());
const Color blue(0.0, 0.0, 1.0);
const Color green(0.0, 1.0, 0.0);
const Color red(1.0, 0.0, 0.0);
// const Translation offset({ 0, 0, 0.325 / 2.0 });

struct traj_estimation_t
{
  using This = traj_estimation_t;

  ros::Timer _graph_timer, _publisher_timer, _traj_pub_timer;
  ros::Subscriber _endcaps_subscriber;
  ros::Publisher _tensegrity_traj_publisher;
  ros::Publisher _tensegrity_multi_publisher, _tensegrity_bars_publisher;
  std::array<ros::Publisher, 6> _endcaps_pubs;
  Graph _graph;
  Values _values;

  // std::array<SE3, 3> _bars;
  const std::array<estimation::RodColors, 6> _rod_colors;
  const std::array<Eigen::Vector3d, 6> _colors;

  int _idx;
  // int rod_idx;
  // std::array<int, 3> rotation_idx;
  double graph_resolution;
  double optimization_frequency;
  double publisher_frequency;

  ros::Time prev_dt;

  gtsam::LevenbergMarquardtParams _lm_params;
  std::shared_ptr<factor_graphs::levenberg_marquardt_t> _lm_helper;
  std::shared_ptr<interface::node_status_t> _node_status;

  std::shared_ptr<estimation::cables_callback_t> cables_callback;

  std::shared_ptr<rod_callback_t> red_rod;
  std::shared_ptr<rod_callback_t> blue_rod;
  std::shared_ptr<rod_callback_t> green_rod;

  estimation::observation_update_t _observation_update;

  Translation _offset;
  Rotation _Roffset;

  ros::Time _prev_timepoint;
  ros::Timer _timer;
  bool visualize;
  std::vector<interface::TensegrityEndcaps> _endcaps_q;
  int window_size;
  double window_dt;
  double _traj_ti;
  int _state_idx;
  int _rot_idx;
  std::vector<std::string> initial_endcaps_params;
  std::array<gtsam::Pose3, 3> _poses;
  std::array<Eigen::Vector3d, 6> _endcaps;
  std::array<bool, 6> _endcaps_valid;
  std::array<gtsam::JacobianFactor::shared_ptr, 6> _endcap_priors;

  estimation::tensegrity_graph_inputs_t _tg_input;
  std::shared_ptr<estimation::sensors_callback_t> _sensors_callback;
  // std::shared_ptr<estimation::cables_callback_t> _cables_callback;

  traj_estimation_t(ros::NodeHandle& nh)
    : _idx(0)
    , _offset({ 0, 0, 0.325 / 2.0 })
    , _Roffset(0.0, 0.0, 1.0, 0.0)
    , visualize(true)
    , prev_dt(0)
    , _poses({ factor_graphs::random<gtsam::Pose3>(), factor_graphs::random<gtsam::Pose3>(),
               factor_graphs::random<gtsam::Pose3>() })
    , _colors({ red, red, green, green, blue, blue })
    , _rod_colors({ estimation::RodColors::RED, estimation::RodColors::RED, estimation::RodColors::GREEN,
                    estimation::RodColors::GREEN, estimation::RodColors::BLUE, estimation::RodColors::BLUE })
    , window_size(10)
    , _traj_ti(0.0)
    , _prev_timepoint(0)
    , _rot_idx(0)
    , _state_idx(0)
  // , _colors({ estimation::RodColors::RED, estimation::RodColors::GREEN, estimation::RodColors::BLUE })
  {
    // interface::node_status_t node_status(nh, "/nodes/traj_estimation/");
    _observation_update.offset = _offset;
    _observation_update.Roffset = _Roffset;

    _tg_input.offset = _offset;
    _tg_input.Roffset = _Roffset;
    // lm_params.setVerbosityLM("SILENT");
    _lm_params.setVerbosityLM("SUMMARY");
    _lm_params.setMaxIterations(10);
    _lm_helper = std::make_shared<factor_graphs::levenberg_marquardt_t>(nh, "/nodes/state_estimation/fg", _lm_params);

    std::string red_endcaps_topic, blue_endcaps_topic, green_endcaps_topic;
    std::string tensegrity_pose_topic, tensegrity_traj_topic;
    std::string cables_topic, cable_map_filename;
    std::string initial_estimate_filename;
    std::string tensegrity_endcaps_topic, tensegrity_multi_topic;
    // double resolution;
    bool use_cable_sensors{ true };
    double trajectory_pub_frequency;
    double frequency{ 10 };

    PARAM_SETUP(nh, visualize);
    PARAM_SETUP(nh, tensegrity_pose_topic)
    PARAM_SETUP(nh, cable_map_filename)
    PARAM_SETUP(nh, tensegrity_endcaps_topic);
    PARAM_SETUP(nh, initial_endcaps_params);
    PARAM_SETUP_WITH_DEFAULT(nh, frequency, frequency);
    PARAM_SETUP_WITH_DEFAULT(nh, window_dt, window_dt);
    PARAM_SETUP_WITH_DEFAULT(nh, window_size, window_size);
    // PARAM_SETUP(nh, blue_endcaps_topic);
    // PARAM_SETUP(nh, green_endcaps_topic);
    // PARAM_SETUP(nh, graph_resolution);        // in seconds
    // PARAM_SETUP(nh, optimization_frequency);  // How fast to run the optimizer
    // PARAM_SETUP(nh, publisher_frequency)
    // PARAM_SETUP(nh, trajectory_pub_frequency)

    _node_status = interface::node_status_t::create(nh, false);

    _endcaps_subscriber = nh.subscribe(tensegrity_endcaps_topic, 1, &This::endcaps_callback, this);
    _sensors_callback = std::make_shared<estimation::sensors_callback_t>(nh);

    const ros::Duration timer(1.0 / frequency);
    _timer = nh.createTimer(timer, &This::timer_callback, this);
    // PARAM_SETUP(nh, tensegrity_pose_topic);
    // PARAM_SETUP(nh, tensegrity_traj_topic)
    // PARAM_SETUP(nh, initial_estimate_filename)
    _tensegrity_bars_publisher = nh.advertise<interface::TensegrityBars>(tensegrity_pose_topic, 1, true);
    // _tensegrity_multi_publisher = nh.advertise<interface::TensegrityBarsArray>(tensegrity_multi_topic, 1, true);

    // _tg_input.cable_map = {};
    _tg_input.cable_noise = gtsam::noiseModel::Isotropic::Sigma(1, 1.6e-2);
    _tg_input.cable_map = estimation::create_cable_map_fix_endcaps(cable_map_filename);

    // red_rod = std::make_shared<rod_callback_t>(nh, red_endcaps_topic);
    // blue_rod = std::make_shared<rod_callback_t>(nh, blue_endcaps_topic);
    // green_rod = std::make_shared<rod_callback_t>(nh, green_endcaps_topic);

    // cables_callback = std::make_shared<estimation::cables_callback_t>(nh, cables_topic);

    // // const gtsam::Key key_Xred0{ rod_symbol(estimation::RodColors::RED, rod_idx) };
    // // values.insert(key_Xred0, factor_graphs::random<SE3>());

    // const ros::Duration graph_timer(1.0 / optimization_frequency);
    // const ros::Duration publisher_timer(1.0 / publisher_frequency);
    // const ros::Duration traj_pub_timer(1.0 / trajectory_pub_frequency);
    // // ros::Subscriber sub = n.subscribe<sensor_msgs::Image> ("image_topic", 10,
    // //                                boost::bind(processImagecallback, _1, argc, argv) );
    // // _red_subscriber = nh.subscribe<interface::TensegrityEndcaps>(
    // //     red_endcaps_topic, 1, boost::bind(boost::mem_fn(&This::endcap_callback), this, _1,
    // //     estimation::RodColors::RED));
    // // &This::endcap_callback, this);
    // // _green_subscriber = nh.subscribe(blue_endcaps_topic, 1, &This::endcap_callback, this);
    // // _blue_subscriber = nh.subscribe(green_endcaps_topic, 1, &This::endcap_callback, this);

    // _tensegrity_bars_publisher = nh.advertise<interface::TensegrityBars>(tensegrity_pose_topic, 1, true);

    for (int i = 0; i < 6; ++i)
    {
      const std::string topicname{ "/tensegrity/endcaps/traj/" + std::to_string(i) };
      _endcaps_pubs[i] = nh.advertise<visualization_msgs::Marker>(topicname, 1, true);
    }
    // _traj_pub_timer = nh.createTimer(traj_pub_timer, &This::publish_trajectory, this);
    // _graph_timer = nh.createTimer(graph_timer, &This::update, this);
    // _publisher_timer = nh.createTimer(publisher_timer, &This::publish_pose, this);

    // prev_timepoint = ros::Time::now();

    // const gtsam::Values result{ estimation::compute_initialization(initial_estimate_filename, _offset, _Roffset) };

    // const gtsam::Key key_red_0{ estimation::rod_symbol(estimation::RodColors::RED, _idx, 0) };
    // const gtsam::Key key_green_0{ estimation::rod_symbol(estimation::RodColors::GREEN, _idx, 0) };
    // const gtsam::Key key_blue_0{ estimation::rod_symbol(estimation::RodColors::BLUE, _idx, 0) };

    // values.insert(key_red_0, result.at<SE3>(key_red_0));
    // values.insert(key_green_0, result.at<SE3>(key_green_0));
    // values.insert(key_blue_0, result.at<SE3>(key_blue_0));

    // graph.addPrior(key_red_0, result.at<SE3>(key_red_0));
    // graph.addPrior(key_green_0, result.at<SE3>(key_green_0));
    // graph.addPrior(key_blue_0, result.at<SE3>(key_blue_0));

    // add_new_step();

    // _node_status->status(interface::NodeStatus::READY);
  }

  ~traj_estimation_t()
  {
    // node_status->status(interface::NodeStatus::STOPPED);
    // ros::Duration(1.0).sleep();
  }
  void init_endcaps()
  {
    bool all_endcaps_received{ true };

    for (int i = 0; i < initial_endcaps_params.size(); ++i)
    {
      // Assuming pose = (quat, pos)
      std::vector<double> params_in;
      all_endcaps_received &= tensegrity::utils::param_check_then_get(initial_endcaps_params[i], params_in);
      // DEBUG_VARS(i, all_endcaps_received)
      if (all_endcaps_received)
      {
        _endcaps[i] = Eigen::Vector3d(params_in[0], params_in[1], params_in[2]);
        _endcaps_valid[i] = true;
      }
    }

    if (all_endcaps_received)
    {
      gtsam::noiseModel::Isotropic::shared_ptr z0_noise{ gtsam::noiseModel::Isotropic::Sigma(3, 1e-3) };
      for (int i = 0; i < 6; ++i)
      {
        gtsam::Ordering key_ordering;
        gtsam::Key key{ estimation::endcap_symbol(i / 2, 0, i) };

        key_ordering += key;

        gtsam::Values values;
        gtsam::GaussianFactorGraph linearFactorGraph;

        values.insert(key, _endcaps[i]);
        // linearFactorGraph.push_back(prior);
        const gtsam::PriorFactor<Eigen::Vector3d> init_prior(key, _endcaps[i], z0_noise);
        linearFactorGraph.push_back(init_prior.linearize(values));
        const gtsam::GaussianConditional::shared_ptr marginal{
          linearFactorGraph.marginalMultifrontalBayesNet(key_ordering)->front()
        };
        const gtsam::VectorValues result{ marginal->solve(gtsam::VectorValues()) };

        _endcap_priors[i] = boost::make_shared<gtsam::JacobianFactor>(
            marginal->keys().front(), marginal->getA(marginal->begin()),
            marginal->getb() - marginal->getA(marginal->begin()) * result[key], marginal->get_model());
      }

      publish_estimation();
      _node_status->status(interface::NodeStatus::RUNNING);
    }
  }

  void publish_estimation()
  {
    if (visualize)
    {
      for (int i = 0; i < 6; ++i)
      {
        if (_endcaps_valid[i])
          point_to_marker(_endcaps[i], _colors[i], i, _endcaps_pubs[i]);
        else
          point_to_marker(_endcaps[i], _colors[i] * 0.2, i, _endcaps_pubs[i]);
        // remove_marker(i, _endcaps_pubs[i]);
      }
      estimation::publish_tensegrity_msg(_poses[0], _poses[1], _poses[2], _tensegrity_bars_publisher, "world", 0);
    }
  }

  gtsam::Pose3 estimate_single_bar(const Eigen::Vector3d& e0, const Eigen::Vector3d& e1)
  {
    gtsam::Values values;
    gtsam::NonlinearFactorGraph graph;
    estimation::observation_update_t observation_update;
    observation_update.offset = _offset;
    observation_update.Roffset = _Roffset;
    observation_update.pose = gtsam::Pose3(gtsam::Rot3(), (e0 + e1) / 2.0);

    observation_update.zA = e0;
    observation_update.zB = e1;
    observation_update.color = estimation::RodColors::RED;
    estimation::add_two_observations(observation_update, graph, values);
    const gtsam::Key key{ observation_update.key_se3 };
    const gtsam::Values result{ _lm_helper->optimize(graph, values, true) };
    return result.at<gtsam::Pose3>(key);
  }

  void remove_marker(const int id, ros::Publisher& pub)
  {
    visualization_msgs::Marker marker;
    tensegrity::utils::init_header(marker.header, "world");
    marker.action = visualization_msgs::Marker::DELETE;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.ns = "TrajEstimationPoint";
    marker.id = id;

    pub.publish(marker);
  }

  void point_to_marker(const Eigen::Vector3d& pt, const Eigen::Vector3d& pt_color, const int id, ros::Publisher& pub)
  {
    visualization_msgs::Marker marker;
    tensegrity::utils::init_header(marker.header, "world");
    marker.action = visualization_msgs::Marker::ADD;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.ns = "TrajEstimationPoint";
    marker.id = id;

    marker.scale.x = 0.04;
    marker.scale.y = 0.04;
    marker.scale.z = 0.04;

    marker.pose.orientation.w = 1.0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;

    marker.pose.position.x = pt[0];
    marker.pose.position.y = pt[1];
    marker.pose.position.z = pt[2];
    marker.color.r = pt_color[0];
    marker.color.g = pt_color[1];
    marker.color.b = pt_color[2];
    marker.color.a = 0.8;

    pub.publish(marker);
  }

  void endcaps_callback(const interface::TensegrityEndcapsConstPtr msg)
  {
    _endcaps_q.emplace_back(*msg);
    process_endcaps(*msg);
    // select_endcaps(*msg);
    // build_all_graphs(*);
  }

  void process_endcaps(const interface::TensegrityEndcaps& endcaps_new)
  {
    Eigen::Vector3d e_new;
    std::array<std::vector<std::optional<Eigen::Vector3d>>, 6> prop_endcaps;
    for (int i = 0; i < endcaps_new.endcaps.size(); ++i)
    {
      int id{ endcaps_new.ids[i] };
      interface::copy(e_new, endcaps_new.endcaps[i]);
      const double d0{ (e_new - _endcaps[2 * id]).norm() };
      const double d1{ (e_new - _endcaps[2 * id + 1]).norm() };
      // DEBUG_VARS(id, 2 * id, 2 * id + 1);
      if (d0 < d1)
      {
        prop_endcaps[2 * id].push_back(e_new);
      }
      else
      {
        prop_endcaps[2 * id + 1].push_back(e_new);
      }
      // prop_endcaps[2 * id].push_back(e_new);
      // prop_endcaps[2 * id + 1].push_back(e_new);
    }
    for (int i = 0; i < 6; ++i)
    {
      _tg_input.noise_models[i] = gtsam::noiseModel::Isotropic::Sigma(3, 1e-2);
      if (prop_endcaps[i].size() == 0)
      {
        DEBUG_VARS(i, "OPT")
        prop_endcaps[i].push_back(std::nullopt);
      }
    }
    DEBUG_VARS(prop_endcaps[0].size());
    DEBUG_VARS(prop_endcaps[1].size());
    DEBUG_VARS(prop_endcaps[2].size());
    DEBUG_VARS(prop_endcaps[3].size());
    DEBUG_VARS(prop_endcaps[4].size());
    DEBUG_VARS(prop_endcaps[5].size());

    std::vector<estimation::tensegrity_graph_output_t> outputs{ build_all_graphs(prop_endcaps) };
    DEBUG_VARS(outputs.size());

    gtsam::Values values;
    gtsam::NonlinearFactorGraph graph;

    for (auto& out : outputs)
    {
      values.insert(out.values);
      graph.push_back(out.graph);
    }
    const gtsam::Values res{ _lm_helper->optimize(graph, values, true) };
    // graph.printErrors(res, "Res ", SF::formatter);
    const double init_error{ graph.error(values) };
    const double tot_error{ graph.error(res) };
    double prev_error{ 10000 };
    int idx{ 0 };
    int min_idx{ 0 };
    for (auto& out : outputs)
    {
      const double error{ out.graph.error(res) };
      if (error < prev_error)
      {
        prev_error = error;
        min_idx = idx;
        DEBUG_VARS(min_idx, error);
      }
      idx++;
    }

    if (std::isinf(init_error) or std::isinf(tot_error) or tot_error > 1e20)
    {
      graph.printErrors(res, "Min Graph ", SF::formatter);
    }
    PRINT_KEY_CONTAINER(outputs[min_idx].keys_pose);
    _poses[0] = res.at<gtsam::Pose3>(outputs[min_idx].keys_pose[0]);
    _poses[1] = res.at<gtsam::Pose3>(outputs[min_idx].keys_pose[1]);
    _poses[2] = res.at<gtsam::Pose3>(outputs[min_idx].keys_pose[2]);

    _endcaps[0] = _poses[0] * (res.at<gtsam::Rot3>(outputs[min_idx].keys_rotA[0]) * _offset);
    _endcaps[1] = _poses[0] * (res.at<gtsam::Rot3>(outputs[min_idx].keys_rotB[0]) * _offset);
    _endcaps[2] = _poses[1] * (res.at<gtsam::Rot3>(outputs[min_idx].keys_rotA[1]) * _offset);
    _endcaps[3] = _poses[1] * (res.at<gtsam::Rot3>(outputs[min_idx].keys_rotB[1]) * _offset);
    _endcaps[4] = _poses[2] * (res.at<gtsam::Rot3>(outputs[min_idx].keys_rotA[2]) * _offset);
    _endcaps[5] = _poses[2] * (res.at<gtsam::Rot3>(outputs[min_idx].keys_rotB[2]) * _offset);

    publish_estimation();
  }

  void select_endcaps(const interface::TensegrityEndcaps& endcaps_new)
  {
    gtsam::noiseModel::Isotropic::shared_ptr z0_noise{ gtsam::noiseModel::Isotropic::Sigma(3, 1e-2) };
    Eigen::Vector3d e_new;
    std::array<std::vector<Eigen::Vector3d>, 3> endcaps_prop;
    // interface::TensegrityEndcaps endcaps_other;
    std::array<std::vector<Eigen::Vector3d>, 6> rejected;
    std::array<std::set<int>, 3> accepted;
    for (int i = 0; i < endcaps_new.endcaps.size(); ++i)
    {
      int id{ endcaps_new.ids[i] };
      interface::copy(e_new, endcaps_new.endcaps[i]);
      endcaps_prop[id].push_back(e_new);
    }

    for (int i = 0; i < 6; ++i)
    {
      _endcaps_valid[i] = false;
      const Eigen::Vector3d e_prev{ _endcaps[i] };
      const gtsam::Key key{ estimation::endcap_symbol(i / 2, 0, i) };
      gtsam::Ordering key_ordering;

      key_ordering += key;

      gtsam::Values values;

      values.insert(key, _endcaps[i]);
      double prev_error{ 1000 };
      gtsam::JacobianFactor::shared_ptr next_prior{ nullptr };
      for (int j = 0; j < endcaps_prop[i / 2].size(); ++j)
      {
        e_new = endcaps_prop[i / 2][j];

        gtsam::GaussianFactorGraph linearFactorGraph;
        linearFactorGraph.push_back(_endcap_priors[i]);
        const gtsam::PriorFactor<Eigen::Vector3d> init_prior(key, e_new, z0_noise);
        linearFactorGraph.push_back(init_prior.linearize(values));
        const gtsam::GaussianConditional::shared_ptr marginal{
          linearFactorGraph.marginalMultifrontalBayesNet(key_ordering)->front()
        };
        const gtsam::VectorValues result{ marginal->solve(gtsam::VectorValues()) };
        const double error{ linearFactorGraph.error(result) };

        // DEBUG_VARS(i, error, e_prev.transpose(), e_new.transpose());
        if (error < 10.597 and error < prev_error)
        {
          prev_error = error;
          _endcaps[i] = e_new;
          _endcaps_valid[i] = true;
          accepted[i / 2].insert(j);

          next_prior = boost::make_shared<gtsam::JacobianFactor>(
              marginal->keys().front(), marginal->getA(marginal->begin()),
              marginal->getb() - marginal->getA(marginal->begin()) * result[key], marginal->get_model());
        }
        // else
        // {
        //   // endcaps_other.endcaps.push_back(endcaps_new.endcaps[j]);
        //   rejected[i].emplace_back(e_new);
        // }
      }

      if (next_prior)
      {
        _endcap_priors[i] = next_prior;
      }
    }

    for (int i = 0; i < 6; ++i)
    {
      if (_endcaps_valid[i])
        continue;
      for (int j = 0; j < endcaps_prop[i / 2].size(); ++j)
      {
        if (accepted[i / 2].count(j))
          continue;

        rejected[i].emplace_back(endcaps_prop[i / 2][j]);
      }
      // DEBUG_VARS(rejected[i])
    }
    recover_endcaps(rejected);

    // publish_estimation();
  }

  void recover_endcaps(std::array<std::vector<Eigen::Vector3d>, 6>& prop_endcaps)
  {
    _tg_input.reset();
    for (int i = 0; i < 3; ++i)
    {
      _tg_input.init_poses[i] = _poses[i];
    }
    for (int i = 0; i < 6; ++i)
    {
      // const int bar_id{ i / 2 };
      // DEBUG_VARS(_endcaps_valid[i], _endcaps_valid[i + 1])
      if (_endcaps_valid[i])
      {
        _tg_input.endcaps[i] = _endcaps[i];
        _tg_input.noise_models[i] = gtsam::noiseModel::Isotropic::Sigma(3, 1e-3);
      }
    }

    for (int i = 0; i < 6; ++i)
    {
      if (_endcaps_valid[i])
        continue;
      for (int j = 0; j < prop_endcaps[i].size(); ++j)
      {
      }
    }
  }

  std::vector<estimation::tensegrity_graph_output_t>
  build_all_graphs(std::array<std::vector<std::optional<Eigen::Vector3d>>, 6>& prop_endcaps)
  {
    std::vector<estimation::tensegrity_graph_output_t> outputs;
    _tg_input.add_cable_meassurements = true;
    _tg_input.cables = std::get<0>(_sensors_callback->back());
    for (int i = 0; i < prop_endcaps[0].size(); ++i)
    {
      _tg_input.endcaps[0] = prop_endcaps[0][i];
      for (int ii = 0; ii < prop_endcaps[1].size(); ++ii)
      {
        _tg_input.endcaps[1] = prop_endcaps[1][ii];
        for (int iii = 0; iii < prop_endcaps[2].size(); ++iii)
        {
          _tg_input.endcaps[2] = prop_endcaps[2][iii];
          for (int iv = 0; iv < prop_endcaps[3].size(); ++iv)
          {
            _tg_input.endcaps[3] = prop_endcaps[3][iv];
            for (int v = 0; v < prop_endcaps[4].size(); ++v)
            {
              _tg_input.endcaps[4] = prop_endcaps[4][v];
              for (int vi = 0; vi < prop_endcaps[5].size(); ++vi)
              {
                _tg_input.endcaps[5] = prop_endcaps[5][vi];
                _tg_input.idx_major++;
                outputs.push_back(estimation::create_tensegrity_graph(_tg_input));
              }
            }
          }
        }
      }
    }
    return outputs;
    //   if (endcap_id == 6)
    //     return;
    // if (_endcaps_valid[endcap_id])
    //   build_graph_recur(endcap_id + 1, prop_idx, prop_endcaps);

    // _tg_input.endcaps[endcap_id] = prop_endcaps[endcap_id][prop_idx];
    // _tg_input.noise_models[endcap_id] = gtsam::noiseModel::Isotropic::Sigma(3, 1e-2);
    // build_graph_recur(endcap_id + 1, prop_endcaps);
  }

  void recover_endcaps_v0(std::array<std::vector<Eigen::Vector3d>, 6>& prop_endcaps)
  {
    using EndcapObservationFactor = estimation::endcap_observation_factor_t;
    using RotationOffsetFactor = estimation::endcap_rotation_offset_factor_t;
    using RotationFixIdentity = estimation::rotation_fix_identity_t;

    gtsam::Values values;
    gtsam::NonlinearFactorGraph graph;
    // estimation::observation_update_t observation_update;

    // validated endcap, sub[mm] belief
    const gtsam::noiseModel::Base::shared_ptr zvalid_noise{ gtsam::noiseModel::Isotropic::Sigma(3, 1e-3) };
    const gtsam::noiseModel::Base::shared_ptr zprop_noise{ gtsam::noiseModel::Isotropic::Sigma(3, 1e-2) };
    const gtsam::noiseModel::Base::shared_ptr rot_prior_nm{ gtsam::noiseModel::Isotropic::Sigma(3, 1e-4) };

    std::array<std::vector<EndcapObservationFactor>, 3> test_factors;

    for (int i = 0; i < 6; i += 2)
    {
      const int bar_id{ i / 2 };
      DEBUG_VARS(_endcaps_valid[i], _endcaps_valid[i + 1])
      if (_endcaps_valid[i] and _endcaps_valid[i + 1])
      {
        continue;
        // const gtsam::Key key_se3{ estimation::rod_symbol(_rod_colors[i], 0) };
        // const gtsam::Key key_rotA_offset{ estimation::rotation_symbol(_rod_colors[i], 0, 0) };
        // const gtsam::Key key_rotB_offset{ estimation::rotation_symbol(_rod_colors[i], 0, 1) };

        // estimation::add_to_values(values, key_se3, _poses[bar_id]);
        // estimation::add_to_values(values, key_rotA_offset, gtsam::Rot3());
        // estimation::add_to_values(values, key_rotB_offset, _Roffset.inverse());

        // graph.emplace_shared<EndcapObservationFactor>(key_se3, key_rotA_offset, _endcaps[i], _offset, zvalid_noise);
        // graph.emplace_shared<EndcapObservationFactor>(key_se3, key_rotB_offset, _endcaps[i + 1], _offset,
        // zvalid_noise);

        // graph.emplace_shared<RotationOffsetFactor>(key_rotA_offset, key_rotB_offset, _Roffset, rot_prior_nm);
        // graph.emplace_shared<RotationOffsetFactor>(key_rotB_offset, key_rotA_offset, _Roffset, rot_prior_nm);

        // graph.emplace_shared<RotationFixIdentity>(key_rotA_offset, key_rotB_offset, _Roffset, rot_prior_nm);
      }
      else if (_endcaps_valid[i] or _endcaps_valid[i + 1])
      {
        int valid_idx{ _endcaps_valid[i] ? i : i + 1 };
        int invalid_idx{ _endcaps_valid[i] ? i + 1 : i };

        for (int j = 0; j < prop_endcaps[invalid_idx].size(); ++j)
        {
          const Eigen::Vector3d& valid_e{ _endcaps[valid_idx] };
          const Eigen::Vector3d& prop_e{ prop_endcaps[invalid_idx][j] };

          const gtsam::Key key_se3{ estimation::rod_symbol(_rod_colors[i], j) };
          const gtsam::Key key_rotA_offset{ estimation::rotation_symbol(_rod_colors[i], j, 0) };
          const gtsam::Key key_rotB_offset{ estimation::rotation_symbol(_rod_colors[i], j, 1) };

          estimation::add_to_values(values, key_se3, _poses[bar_id]);
          estimation::add_to_values(values, key_rotA_offset, gtsam::Rot3());
          estimation::add_to_values(values, key_rotB_offset, _Roffset.inverse());

          graph.emplace_shared<EndcapObservationFactor>(key_se3, key_rotA_offset, valid_e, _offset, zvalid_noise);
          graph.emplace_shared<EndcapObservationFactor>(key_se3, key_rotB_offset, prop_e, _offset, zprop_noise);
          test_factors[bar_id].emplace_back(key_se3, key_rotB_offset, prop_e, _offset, zprop_noise);

          graph.emplace_shared<RotationOffsetFactor>(key_rotA_offset, key_rotB_offset, _Roffset, rot_prior_nm);
          graph.emplace_shared<RotationOffsetFactor>(key_rotB_offset, key_rotA_offset, _Roffset, rot_prior_nm);

          graph.emplace_shared<RotationFixIdentity>(key_rotA_offset, key_rotB_offset, _Roffset, rot_prior_nm);
        }
      }
    }
    const gtsam::Values result{ _lm_helper->optimize(graph, values, true) };
    graph.printErrors(result, "Res ", SF::formatter);

    for (int i = 0; i < 3; ++i)
    {
      int invalid_idx{ _endcaps_valid[i * 2] ? i * 2 + 1 : i * 2 };

      bool validated{ false };
      gtsam::Key key_se3;
      gtsam::Key key_rot;
      double prev_error{ std::numeric_limits<double>::max() };
      for (auto factor : test_factors[i])
      {
        const double error{ factor.error(result) };
        if (error < prev_error)
        {
          prev_error = error;
          key_se3 = factor.key<1>();
          key_rot = factor.key<2>();
          validated = true;
          DEBUG_VARS(error)
          PRINT_KEYS(key_se3, key_rot)
        }
      }

      if (validated)
      {
        const gtsam::Pose3 pose{ result.at<gtsam::Pose3>(key_se3) };
        const gtsam::Rot3 rot{ result.at<gtsam::Rot3>(key_rot) };
        _endcaps[invalid_idx] = pose * (rot * _offset);
        _endcaps_valid[invalid_idx] = true;
      }
    }

    // observation_update.zA = e0;
    // observation_update.zB = e1;
    // observation_update.color = _rod_colors[i];
    // estimation::add_two_observations(observation_update, graph, values);
    // const gtsam::Key key{ observation_update.key_se3 };

    // return result.at<gtsam::Pose3>(key);
  }

  // gtsam::NonlinearFactorGraph create_tensegrity_graph(std::vector<>)
  // {
  //   gtsam::NonlinearFactorGraph graph;
  //   std::map<gtsam::Key, gtsam::Key> map;
  //   const gtsam::noiseModel::Base::shared_ptr rot_prior_nm{ gtsam::noiseModel::Isotropic::Sigma(3, 1e-4) };
  //   for (int i = 0; i < 6; i += 2)
  //   {
  //     const gtsam::Key key_se3{ estimation::rod_symbol(_rod_colors[i], 0) };
  //     const gtsam::Key key_rotA_offset{ estimation::rotation_symbol(_rod_colors[i], 0, 0) };
  //     const gtsam::Key key_rotB_offset{ estimation::rotation_symbol(_rod_colors[i], 0, 1) };

  //     // map[key_se3] = key_se3;
  //     // map[key_rotA_offset] = key_rotA_offset;
  //     // map[key_rotB_offset] = key_rotB_offset;

  //     // estimation::add_to_values(values, key_se3, _poses[bar_id]);
  //     // estimation::add_to_values(values, key_rotA_offset, gtsam::Rot3());
  //     // estimation::add_to_values(values, key_rotB_offset, _Roffset.inverse());

  //     graph.emplace_shared<EndcapObservationFactor>(key_se3, key_rotA_offset, valid_e, _offset, zvalid_noise);
  //     graph.emplace_shared<EndcapObservationFactor>(key_se3, key_rotB_offset, prop_e, _offset, zprop_noise);
  //     // test_factors[bar_id].emplace_back(key_se3, key_rotB_offset, prop_e, _offset, zprop_noise);

  //     graph.emplace_shared<RotationOffsetFactor>(key_rotA_offset, key_rotB_offset, _Roffset, rot_prior_nm);
  //     graph.emplace_shared<RotationOffsetFactor>(key_rotB_offset, key_rotA_offset, _Roffset, rot_prior_nm);

  //     graph.emplace_shared<RotationFixIdentity>(key_rotA_offset, key_rotB_offset, _Roffset, rot_prior_nm);
  //   }
  // }

  void timer_callback(const ros::TimerEvent& event)
  {
    if (_node_status->status() == interface::NodeStatus::PREPARING)
    {
      init_endcaps();
    }
    else if (_node_status->status() == interface::NodeStatus::RUNNING)
    {
      // publish_estimation();

      // run_fg();
      // if (visualize)
      // {
      // for (int i = 0; i < 6; ++i)
      // {
      //   if (_valid_estimate[i])
      //     point_to_marker(_last_estimate[i], colors[i / 2], estimation_pub[i]);
      // }
      // }
    }
  }

  // TODO: "cluster" with the previous estimate (endcaps) and retain the must likely ones
  // then, compute the bars given those.
  void compute_all_bars(const interface::TensegrityEndcaps& endcaps)
  {
    // for (int j = 0; j < endcaps.size(); ++j, ++_rot_idx)
    // {
    //   if (_prev_timepoint.isZero())
    //   {
    //     _prev_timepoint = _endcaps_q[i].header.stamp;
    //   }

    //   interface::interface::copy(ei, _endcaps_q[i].endcaps[j]);
    //   const int id{ _endcaps_q[i].ids[j] };
    //   const double score{ _endcaps_q[i].scores[j] };

    //   DEBUG_VARS(id, score, ei.transpose());
    //   const gtsam::Key key_rot_offset{ estimation::rotation_symbol(id, _state_idx, _rot_idx) };
    //   const gtsam::Key key_x(estimation::rod_symbol(id, _state_idx));
    //   const gtsam::Key key_xdot(estimation::rodvel_symbol(id, _state_idx));
    //   // const gtsam::Key key_rotB_offset{ estimation::rotation_symbol(id, _state_idx, 1) };
    //   const double dt{ (_prev_timepoint - _endcaps_q[i].header.stamp).toSec() };
    //   rot_keys[id].push_back(key_rot_offset);

    //   _values.insert_or_assign(key_x, gtsam::Pose3());
    //   _values.insert_or_assign(key_xdot, vel_zero);
    //   _values.insert_or_assign(key_rot_offset, gtsam::Rot3());
    //   DiagonalNM::shared_ptr diag_nm{ DiagonalNM::Sigmas(Eigen::Vector3d::Ones() / score) };
    //   graph_to_add.emplace_shared<BarVelObservationFactor>(key_x, key_rot_offset, key_xdot, ei, _offset, dt,
    //   diag_nm); added_to_graph = true;
    // }
  }

  void run_fg()
  {
    using GaussianNM = gtsam::noiseModel::Gaussian;
    using DiagonalNM = gtsam::noiseModel::Diagonal;
    using BarVelObservationFactor = estimation::bar_vel_observation_factor_t;
    using RotationOffsetFactor = estimation::endcap_rotation_offset_factor_t;
    using RotationFixIdentity = estimation::rotation_fix_identity_t;

    const Eigen::Vector<double, 6> vel_zero{ Eigen::Vector<double, 6>::Zero() };
    gtsam::NonlinearFactorGraph graph_to_add;
    bool added_to_graph{ false };
    for (int i = 0; i < _endcaps_q.size(); ++i)
    {
      Eigen::Vector3d ei;
      // double _traj_ti;
      // int _state_idx;
      std::map<int, std::vector<gtsam::Key>> rot_keys;

      for (int j = 0; j < _endcaps_q[i].endcaps.size(); ++j, ++_rot_idx)
      {
        DEBUG_PRINT
        if (_prev_timepoint.isZero())
        {
          _prev_timepoint = _endcaps_q[i].header.stamp;
        }

        interface::copy(ei, _endcaps_q[i].endcaps[j]);
        const int id{ _endcaps_q[i].ids[j] };
        const double score{ _endcaps_q[i].scores[j] };

        DEBUG_VARS(id, score, ei.transpose());
        const gtsam::Key key_rot_offset{ estimation::rotation_symbol(id, _state_idx, _rot_idx) };
        const gtsam::Key key_x(estimation::rod_symbol(id, _state_idx));
        const gtsam::Key key_xdot(estimation::rodvel_symbol(id, _state_idx));
        // const gtsam::Key key_rotB_offset{ estimation::rotation_symbol(id, _state_idx, 1) };
        const double dt{ (_prev_timepoint - _endcaps_q[i].header.stamp).toSec() };
        rot_keys[id].push_back(key_rot_offset);

        _values.insert_or_assign(key_x, gtsam::Pose3());
        _values.insert_or_assign(key_xdot, vel_zero);
        _values.insert_or_assign(key_rot_offset, gtsam::Rot3());
        DiagonalNM::shared_ptr diag_nm{ DiagonalNM::Sigmas(Eigen::Vector3d::Ones() / score) };
        graph_to_add.emplace_shared<BarVelObservationFactor>(key_x, key_rot_offset, key_xdot, ei, _offset, dt, diag_nm);
        added_to_graph = true;
      }
      add_rotation_factors(graph_to_add, rot_keys);
      _state_idx++;
    }
    _endcaps_q.clear();

    if (added_to_graph)
    {
      gtsam::FactorIndices added_indices{ _graph.add_factors(graph_to_add) };
      _values = _lm_helper->optimize(_graph, _values, true);
      _graph.printErrors(_values, "Errors ", SF::formatter);

      gtsam::Pose3 pose_red{ _values.at<gtsam::Pose3>(estimation::rod_symbol(0, _state_idx - 1)) };
      gtsam::Pose3 pose_green{ _values.at<gtsam::Pose3>(estimation::rod_symbol(1, _state_idx - 1)) };
      gtsam::Pose3 pose_blue{ _values.at<gtsam::Pose3>(estimation::rod_symbol(2, _state_idx - 1)) };
      estimation::publish_tensegrity_msg(pose_red, pose_green, pose_blue, _tensegrity_bars_publisher, "world", 0);
    }
  }

  void add_rotation_factors(gtsam::NonlinearFactorGraph& graph, std::map<int, std::vector<gtsam::Key>>& map)
  {
    using RotationOffsetFactor = estimation::endcap_rotation_offset_factor_t;
    using RotationFixIdentity = estimation::rotation_fix_identity_t;
    gtsam::noiseModel::Base::shared_ptr rot_prior_nm{ gtsam::noiseModel::Isotropic::Sigma(3, 1e-3) };
    for (auto pair : map)
    {
      // DEBUG_VARS(pair.first);
      PRINT_KEY_CONTAINER(pair.second);
      for (int i = 0; i < pair.second.size(); ++i)
      {
        const gtsam::Key key_rotA_offset{ pair.second[i] };
        for (int j = i + 1; j < pair.second.size(); ++j)
        {
          const gtsam::Key key_rotB_offset{ pair.second[j] };
          graph.emplace_shared<RotationOffsetFactor>(key_rotA_offset, key_rotB_offset, _Roffset, rot_prior_nm);
          graph.emplace_shared<RotationOffsetFactor>(key_rotB_offset, key_rotA_offset, _Roffset, rot_prior_nm);

          graph.emplace_shared<RotationFixIdentity>(key_rotA_offset, key_rotB_offset, _Roffset, rot_prior_nm);
        }
      }
    }
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
