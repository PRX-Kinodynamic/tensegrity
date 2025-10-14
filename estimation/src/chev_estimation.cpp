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
#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <factor_graphs/defs.hpp>
#include <nav_msgs/Path.h>

#include <tensegrity_utils/type_conversions.hpp>
#include <tensegrity_utils/std_utils.hpp>
#include <tensegrity_utils/dbg_utils.hpp>
#include <tensegrity_utils/rosparams_utils.hpp>

#include <interface/type_conversions.hpp>
#include <interface/TensegrityEndcaps.h>
#include <interface/TensegrityBars.h>
#include <interface/TensegrityTrajectory.h>
#include <interface/node_status.hpp>
#include <interface/tf_util.hpp>

#include <estimation/bar_utilities.hpp>
#include <estimation/endcap_subscriber.hpp>
#include <estimation/tensegrity_cap_obs_factor.hpp>
#include <estimation/endcap_observation_factor.hpp>
#include <estimation/SE3_observation_factor.hpp>
#include <estimation/cable_sensor_subscriber.hpp>
#include <estimation/sensors_subscriber.hpp>
#include <estimation/cheb_utils.hpp>
#include <estimation/cheb_factors.hpp>

#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/basis/FitBasis.h>
#include <gtsam/basis/Chebyshev2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/basis/Chebyshev.h>

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

// using Camera = gtsam::PinholeCamera<gtsam::Cal3_S2>;
// using CameraPtr = std::shared_ptr<Camera>;

using LieIntegrator = factor_graphs::lie_integration_factor_t<SE3, Velocity>;
// using ChebManifoldSE3 = manifold_evaluation_t<gtsam::Chebyshev2, SE3>;

class cable_length_cheb_bar_factor_t
  : public gtsam::NoiseModelFactorN<gtsam::ParameterMatrix<6>, gtsam::ParameterMatrix<6>>
{
public:
  // using SE3 = gtsam::Pose3;
  using ParameterMatrix = gtsam::ParameterMatrix<6>;
  using Translation = Eigen::Vector<double, 3>;
  using Rotation = gtsam::Rot3;
  using SkewMatrix = Eigen::Matrix<double, 3, 3>;
  using Base = gtsam::NoiseModelFactorN<gtsam::ParameterMatrix<6>, gtsam::ParameterMatrix<6>>;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;
  using Jacobian = Eigen::Matrix<double, 3, 6>;
  using Meassurement = Eigen::Vector<double, 1>;
  using Error = Eigen::Vector<double, 1>;

  // using BarFactor = bar_two_observations_factor_t;
  // using BarFactor = endcap_observation_factor_t;
  // using Vector = Eigen::Vector<double, Dim>;

  cable_length_cheb_bar_factor_t(const gtsam::Key key_pmi, const gtsam::Key key_pmj,  // no-lint
                                 const Translation offsetA, const Translation offsetB, const double zij, const int N,
                                 const double ti, const double a, const double b, const NoiseModel& cost_model)
    : Base(cost_model, key_pmi, key_pmj), _zij(zij), _offsetA(offsetA), _offsetB(offsetB), _func(N, ti, a, b)
  {
  }

  static Meassurement predict(const ParameterMatrix& pmi, const ParameterMatrix& pmj,                       // no-lint
                              const Translation& offsetA, const Translation& offsetB,                       // no-lint
                              const gtsam::Chebyshev2Basis::ManifoldEvaluationFunctor<gtsam::Pose3>& func,  // no-lint
                              gtsam::OptionalJacobian<-1, -1> Hpmi = boost::none,
                              gtsam::OptionalJacobian<-1, -1> Hpmj = boost::none)
  {
    Eigen::MatrixXd xi_H_pmi, xj_H_pmj;
    const bool compute_deriv{ Hpmi or Hpmj };

    const gtsam::Pose3 xi{ func(pmi, compute_deriv ? &xi_H_pmi : nullptr) };
    const gtsam::Pose3 xj{ func(pmj, compute_deriv ? &xj_H_pmj : nullptr) };

    // xi.print("xi");
    Eigen::Matrix<double, 3, 6> zA_H_xi, zB_H_xj;
    const Translation zA{ xi.transformFrom(offsetA, compute_deriv ? &zA_H_xi : nullptr) };
    const Translation zB{ xj.transformFrom(offsetB, compute_deriv ? &zB_H_xj : nullptr) };

    const Eigen::Matrix<double, 3, 3> diff_H_zA{ Eigen::Matrix<double, 3, 3>::Identity() };
    const Eigen::Matrix<double, 3, 3> diff_H_zB{ -Eigen::Matrix<double, 3, 3>::Identity() };
    Eigen::Matrix<double, 1, 3> d_H_diff;
    const Translation diff{ zA - zB };
    const double distance{ gtsam::norm3(diff, compute_deriv ? &d_H_diff : nullptr) };
    // DEBUG_VARS(zA.transpose(), zB.transpose(), distance);

    if (Hpmi)
    {
      *Hpmi = d_H_diff * diff_H_zA * zA_H_xi * xi_H_pmi;
    }
    if (Hpmj)
    {
      *Hpmj = d_H_diff * diff_H_zB * zB_H_xj * xj_H_pmj;
    }
    return Meassurement(distance);
  }

  virtual Eigen::VectorXd evaluateError(const ParameterMatrix& pmi, const ParameterMatrix& pmj,  // no-lint
                                        boost::optional<Eigen::MatrixXd&> Hpmi = boost::none,
                                        boost::optional<Eigen::MatrixXd&> Hpmj = boost::none) const override
  {
    const Meassurement prediction{ predict(pmi, pmj, _offsetA, _offsetB, _func, Hpmi, Hpmj) };

    const Error err{ prediction - _zij };
    return err;
  }

  void print(const std::string& s, const gtsam::KeyFormatter& keyFormatter) const override
  {
    const std::string key_xi{ keyFormatter(this->template key<1>()) };
    const std::string key_xj{ keyFormatter(this->template key<2>()) };

    // const std::string plant_name{ _prx_system->get_pathname() };
    std::cout << s << " ";
    std::cout << "[ " << key_xi << " " << key_xj << " ]\n";
    std::cout << "\t zij: " << _zij[0] << "\n";

    if (this->noiseModel_)
      this->noiseModel_->print("  noise model: ");
    else
      std::cout << "no noise model" << std::endl;
    std::cout << "\n";
  }

private:
  const gtsam::Chebyshev2Basis::ManifoldEvaluationFunctor<gtsam::Pose3> _func;

  const Translation _offsetA;
  const Translation _offsetB;
  const Meassurement _zij;
};

struct chev_estimation_t
{
  using This = chev_estimation_t;
  using ManifoldEvaluationFactor = gtsam::ManifoldEvaluationFactor<gtsam::Chebyshev2Basis, gtsam::Pose3>;
  ros::Timer _finish_timer;
  ros::Subscriber _bars_subscriber;
  // ros::Publisher _tensegrity_bars_publisher, _tensegrity_traj_publisher;
  std::array<ros::Publisher, 6> _markers_publishers;
  std::array<ros::Publisher, 3> _bars_markers_publishers;

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
  std::shared_ptr<interface::node_status_t> node_status, publisher_node;

  std::shared_ptr<estimation::cables_callback_t> cables_callback;

  // std::shared_ptr<rod_callback_t> red_rod;
  // std::shared_ptr<rod_callback_t> blue_rod;
  // std::shared_ptr<rod_callback_t> green_rod;

  estimation::observation_update_t _observation_update;

  Translation _offset;
  Rotation _Roffset;

  ros::Time prev_timepoint;
  std::map<double, std::array<gtsam::Pose3, 3>> input_bars;
  // std::vector<std::array<gtsam::Pose3, 3>> vector_bars;
  std::vector<std::array<Eigen::Vector3d, 6>> endcaps;
  std::vector<ros::Time> timestamps;
  ros::Time t0, tT;

  int _N;
  double dt;
  double a, b;
  double maxT;
  bool fg_not_run;
  gtsam::ParameterMatrix<6> _red_cheb_matrix, _green_cheb_matrix, _blue_cheb_matrix;
  ros::Publisher _tensegrity_bars_publisher;
  std::array<ros::Publisher, 3> _path_publisher;
  // double _a, _b, _N;
  std::array<visualization_msgs::Marker, 6> _markers;
  std::array<visualization_msgs::Marker, 3> _bars_markers;

  gtsam::Values result;
  std::ofstream _json_file;
  std::shared_ptr<estimation::sensors_callback_t> sensors_callback;
  std::vector<gtsam::ParameterMatrix<6>> _chev_mats;

  estimation::ColorMappingNum cable_map;
  std::shared_ptr<interface::ros_tf_utils_t> _tf_utils;

  chev_estimation_t(ros::NodeHandle& nh)
    : _idx(0)
    , _offset({ 0, 0, 0.36 / 2.0 })
    , _Roffset(0.0, 0.0, 1.0, 0.0)
    , node_status(interface::node_status_t::create(nh, "/nodes/chev_estimation/"))
    , prev_dt(0)
    , _bars({ factor_graphs::random<SE3>(), factor_graphs::random<SE3>(), factor_graphs::random<SE3>() })
    , _colors({ estimation::RodColors::RED })
    , t0(0)
    , fg_not_run(true)
    , _red_cheb_matrix(5)
    , _green_cheb_matrix(5)
    , _blue_cheb_matrix(5)
  // , _colors({ estimation::RodColors::RED, estimation::RodColors::GREEN, estimation::RodColors::BLUE })
  {
    // lm_params.setVerbosityLM("SILENT");
    _lm_params.setVerbosityLM("SUMMARY");
    _lm_params.setMaxIterations(10);
    _lm_helper = std::make_shared<factor_graphs::levenberg_marquardt_t>(nh, "/nodes/state_estimation/fg", _lm_params);

    std::string tensegrity_bars_topicname;
    std::string publisher_node_id, cable_map_filename;  // subscribe to this, when it finished, run the estimation
    // std::string tensegrity_pose_topic, tensegrity_traj_topic;
    std::string json_output_filename, camera_info_topic;

    // double resolution;
    bool use_cable_sensors{ true };
    double trajectory_pub_frequency;
    int& N{ _N };

    PARAM_SETUP(nh, tensegrity_bars_topicname);
    PARAM_SETUP(nh, publisher_node_id);
    PARAM_SETUP(nh, json_output_filename);
    PARAM_SETUP(nh, N);
    PARAM_SETUP(nh, cable_map_filename);

    sensors_callback = std::make_shared<estimation::sensors_callback_t>(nh);
    _tf_utils = std::make_shared<interface::ros_tf_utils_t>(nh);
    cable_map = estimation::create_cable_map_fix_endcaps(cable_map_filename);

    _json_file.open(json_output_filename, std::ios::out);

    publisher_node = interface::node_status_t::create(nh, publisher_node_id, true);
    _tensegrity_bars_publisher = nh.advertise<interface::TensegrityBars>("/tensegrity/cheb/output", 1, true);

    const ros::Duration finish_timer(1.0 / 10.0);
    _bars_subscriber = nh.subscribe(tensegrity_bars_topicname, 1, &This::bar_callback, this);
    _finish_timer = nh.createTimer(finish_timer, &This::finish_timer_callback, this);

    const std::vector<Eigen::Vector3d> colors{ { 1.0, 0.0, 0.0 }, { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 },
                                               { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 1.0 }, { 0.0, 0.0, 1.0 } };
    for (int i = 0; i < 6; ++i)
    {
      tensegrity::utils::init_header(_markers[i].header, "real_sense");
      _markers[i].action = visualization_msgs::Marker::ADD;
      _markers[i].type = visualization_msgs::Marker::POINTS;
      _markers[i].scale.x = 0.005;  // is point width,
      _markers[i].scale.y = 0.01;   // is point height
      _markers[i].color.r = colors[i][0];
      _markers[i].color.g = colors[i][1];
      _markers[i].color.b = colors[i][2];
      _markers[i].color.a = 1.0;

      const std::string topicname{ "/tensegrity/endcaps/input/e" + std::to_string(i) };
      _markers_publishers[i] = nh.advertise<visualization_msgs::Marker>(topicname, 1, true);
    }
    for (int i = 0; i < 3; ++i)
    {
      tensegrity::utils::init_header(_bars_markers[i].header, _tf_utils->world_frame());
      _bars_markers[i].action = visualization_msgs::Marker::ADD;
      _bars_markers[i].type = visualization_msgs::Marker::LINE_STRIP;
      _bars_markers[i].scale.x = 0.005;  // is point width,
      _bars_markers[i].scale.y = 0.005;  // is point height
      _bars_markers[i].color.r = colors[i * 2][0];
      _bars_markers[i].color.g = colors[i * 2][1];
      _bars_markers[i].color.b = colors[i * 2][2];
      _bars_markers[i].color.a = 1.0;

      const std::string topicname{ "/tensegrity/bars/chev/e" + std::to_string(i) };
      _bars_markers_publishers[i] = nh.advertise<visualization_msgs::Marker>(topicname, 1, true);
      _path_publisher[i] = nh.advertise<nav_msgs::Path>("/tensegrity/cheb/path/b" + std::to_string(i), 1, true);
    }

    node_status->status(interface::NodeStatus::READY);
  }

  ~chev_estimation_t()
  {
  }

  gtsam::Key rotation_key(const int t, const int i)
  {
    return factor_graphs::symbol_factory_t::create_hashed_symbol("R^{", t, "}_{", i, "}");
  }

  void run_fg()
  {
    using EndcapObservationFactor = estimation::endcap_observation_factor_t;
    using RotationOffsetFactor = estimation::endcap_rotation_offset_factor_t;
    using RotationFixIdentity = estimation::rotation_fix_identity_t;
    using EndcapsChebFactor = estimation::endcaps_cheb_factor_t<gtsam::Chebyshev2Basis>;
    std::array<gtsam::Key, 3> kendcaps;

    kendcaps[0] = gtsam::Symbol('R', 0);
    kendcaps[1] = gtsam::Symbol('G', 0);
    kendcaps[2] = gtsam::Symbol('B', 0);

    gtsam::Pose3 transform;
    const bool tf_received{ _tf_utils->query(transform) };
    TENSEGRITY_ASSERT(tf_received, "Transform not available!");
    DEBUG_PRINT
    gtsam::Values values;
    gtsam::NonlinearFactorGraph graph;
    // auto model = gtsam::noiseModel::Diagonal::Sigmas(Eigen::Vector<double, 6>(1, 1, 0, 1, 1, 1));
    // _N = std::floor(maxT) * 2;
    // _N = 15;
    // _N = input_bars.size() * 2;
    for (int i = 0; i < 3; ++i)
    {
      // _chev_mats.push_back();
      values.insert(kendcaps[i], gtsam::ParameterMatrix<6>(_N));
    }

    DEBUG_VARS(_N);
    int rot_tot{ 0 };
    gtsam::noiseModel::Base::shared_ptr rot_prior_nm{ gtsam::noiseModel::Isotropic::Sigma(3, 1e-4) };
    gtsam::noiseModel::Base::shared_ptr z_noise{ gtsam::noiseModel::Isotropic::Sigma(3, 1e0) };
    gtsam::noiseModel::Base::shared_ptr cable_noise{ gtsam::noiseModel::Isotropic::Sigma(1, 1e0) };
    // gtsam::noiseModel::Base::shared_ptr cable_noise{ gtsam::noiseModel::Isotropic::Sigma(1, 1.6e-2) };

    t0 = std::min(t0, std::get<3>(sensors_callback->at(0)));
    tT = std::max(tT, std::get<3>(sensors_callback->back()));

    maxT = (tT - t0).toSec();

    a = t0.toSec();
    b = tT.toSec();
    for (int i = 0; i < endcaps.size(); ++i)
    {
      const double ti{ timestamps[i].toSec() };
      // const double ti{ (timestamps[i] - t0).toSec() };
      for (int j = 0; j < 3; ++j)
      {
        const Translation zA{ transform.transformFrom(endcaps[i][j * 2]) };
        const Translation zB{ transform.transformFrom(endcaps[i][j * 2 + 1]) };
        // DEBUG_VARS(zA.transpose(), zB.transpose())
        graph.emplace_shared<EndcapsChebFactor>(kendcaps[j], _offset, zA, _N, ti, a, b, z_noise);
        graph.emplace_shared<EndcapsChebFactor>(kendcaps[j], _Roffset * _offset, zB, _N, ti, a, b, z_noise);
        // endcaps_cheb_factor_t(const gtsam::Key key_pm, const Translation offset, const Translation& zi, const int N,
        //                        const double ti, const double a, const double b, const NoiseModel& cost_model)
      }
    }
    result = _lm_helper->optimize(graph, values, true);
    // for (int i = 0; i < 3; ++i)
    // {
    //   _chev_mats.push_back(result.at<gtsam::ParameterMatrix<6>>(kendcaps[i]));
    //   // DEBUG_VARS(i, _chev_mats[i]);
    // }

    // std::array<Translation, 6> offsets{ _offset, -_offset, _offset, -_offset, _offset, -_offset };
    // for (int i = 0; i < sensors_callback->size(); ++i)
    // {
    //   const estimation::sensors_callback_t::Meassurements m{ sensors_callback->at(i) };
    //   const estimation::sensors_callback_t::Cables& cables{ std::get<0>(m) };
    //   const ros::Time& tros{ std::get<3>(sensors_callback->at(i)) };
    //   // const double ti{ (tros - t0).toSec() };
    //   const double ti{ tros.toSec() };
    //   int j = 0;
    //   // DEBUG_VARS(ti, cables.transpose());

    //   for (; j < 3; ++j)
    //   {
    //     const int ci{ cable_map[j].first };
    //     const int cj{ cable_map[j].second };
    //     const gtsam::Key key_Xi{ kendcaps[ci / 2] };
    //     const gtsam::Key key_Xj{ kendcaps[cj / 2] };
    //     const Translation zA{ offsets[ci] };
    //     const Translation zB{ offsets[cj] };
    //     const double zcable{ cables[j] };
    //     // DEBUG_VARS(key_Xi, key_Xj)
    //     const gtsam::Chebyshev2Basis::ManifoldEvaluationFunctor<gtsam::Pose3> func(_N, ti, a, b);

    //     const gtsam::Pose3 xi{ func(_chev_mats[ci / 2]) };
    //     const gtsam::Pose3 xj{ func(_chev_mats[cj / 2]) };
    //     // xi.print("xi");
    //     // xj.print("xj");
    //     double predicted{ cable_length_cheb_bar_factor_t::predict(_chev_mats[ci / 2], _chev_mats[cj / 2], zA, zB,
    //                                                               func)[0] };
    //     const std::string tistr{ tensegrity::utils::convert_to<std::string>(ti) };
    //     // DEBUG_VARS(ci, cj, ci / 2, cj / 2)
    //     // DEBUG_VARS(tistr, zcable, predicted)
    //     graph.emplace_shared<cable_length_cheb_bar_factor_t>(key_Xi, key_Xj, zA, zB, zcable, _N, ti, a, b,
    //     cable_noise);
    //     // cable_length_cheb_bar_factor_t(const gtsam::Key key_pmi, const gtsam::Key key_pmj,  // no-lint
    //     //                                const Translation offsetA, const Translation offsetB, const double zij,
    //     //                                const int N, const double ti, const double a, const double b,
    //     //                                const NoiseModel& cost_model)
    //   }

    // for (; j < 6; ++j)
    // {
    //   const gtsam::Key key_Xi{ kendcaps[cable_map[j].first / 2] };
    //   const gtsam::Key key_Xj{ kendcaps[cable_map[j].second / 2] };
    //   const Translation zA{ offsets[cable_map[j].first] };
    //   const Translation zB{ offsets[cable_map[j].second] };

    //   graph.emplace_shared<cable_length_cheb_bar_factor_t>(key_Xi, key_Xj, zA, zB, cables[j], _N, ti, a, b,
    //                                                        cable_noise);
    // }

    // for (; j < 9; ++j)
    // {
    //   const gtsam::Key key_Xi{ kendcaps[cable_map[j].first / 2] };
    //   const gtsam::Key key_Xj{ kendcaps[cable_map[j].second / 2] };
    //   const Translation zA{ offsets[cable_map[j].first] };
    //   const Translation zB{ offsets[cable_map[j].second] };

    //   graph.emplace_shared<cable_length_cheb_bar_factor_t>(key_Xi, key_Xj, zA, zB, cables[j], _N, ti, a, b,
    //                                                        cable_noise);
    // }
    // }
    // DEBUG_PRINT
    // result = _lm_helper->optimize(graph, result, true);

    _chev_mats.clear();
    for (int i = 0; i < 3; ++i)
    {
      _chev_mats.push_back(result.at<gtsam::ParameterMatrix<6>>(kendcaps[i]));
      // DEBUG_VARS(i, _chev_mats[i]);
    }
    // _red_cheb_matrix = result.at<gtsam::ParameterMatrix<6>>(kred);
    // _green_cheb_matrix = result.at<gtsam::ParameterMatrix<6>>(kgreen);
    // _blue_cheb_matrix = result.at<gtsam::ParameterMatrix<6>>(kblue);
  }

  void finish_timer_callback(const ros::TimerEvent& event)
  {
    // PRINT_MSG("Checking status")
    // DEBUG_VARS(*publisher_node)
    if (publisher_node->status() == interface::NodeStatus::FINISH and fg_not_run)
    {
      PRINT_MSG("Running FG!")
      run_fg();
      fg_not_run = false;
      dt = 0;
    }
    if (not fg_not_run)
    {
      std::array<nav_msgs::Path, 3> paths;
      for (double ti = a; ti < b; ti += 0.1)
      {
        const gtsam::Chebyshev2Basis::ManifoldEvaluationFunctor<SE3> f(_N, ti, a, b);
        for (int i = 0; i < 3; ++i)
        {
          const gtsam::Pose3 xi{ f(_chev_mats[i]) };
          const Translation pti{ xi.translation() };
          _bars_markers[i].points.emplace_back();
          _bars_markers[i].points.back().x = pti[0];
          _bars_markers[i].points.back().y = pti[1];
          _bars_markers[i].points.back().z = pti[2];
          paths[i].poses.emplace_back();
          paths[i].poses.back().header.frame_id = _tf_utils->world_frame();
          paths[i].poses.back().pose = tensegrity::utils::convert_to<geometry_msgs::Pose>(xi);
          // if (i == 0)
          // {
          //   xi.print("xi");
          //   const Eigen::Matrix<double, 1, 3> zA{ xi.transformFrom(_offset).transpose() };
          //   const Eigen::Matrix<double, 1, 3> zB{ xi.transformFrom(_Roffset * _offset).transpose() };
          //   DEBUG_VARS(zA, zB)
          // }
        }
      }

      for (int i = 0; i < 6; ++i)
      {
        _markers_publishers[i].publish(_markers[i]);
      }
      for (int i = 0; i < 3; ++i)
      {
        _bars_markers[i].header.frame_id = _tf_utils->world_frame();
        _bars_markers_publishers[i].publish(_bars_markers[i]);

        paths[i].header.frame_id = _tf_utils->world_frame();
        _path_publisher[i].publish(paths[i]);
      }

      const std::string cheb0_str{ estimation::cheb_to_json_array(_chev_mats[0]) };
      const std::string cheb1_str{ estimation::cheb_to_json_array(_chev_mats[1]) };
      const std::string cheb2_str{ estimation::cheb_to_json_array(_chev_mats[2]) };
      // if (dt < endcaps.size())
      // {
      //   const int idx{ static_cast<int>(dt) };
      //   gtsam::Key kred{ gtsam::Symbol('R', idx) };
      //   gtsam::Key kgreen{ gtsam::Symbol('G', idx) };
      //   gtsam::Key kblue{ gtsam::Symbol('B', idx) };
      //   dt++;

      //   const gtsam::Pose3 red_pose{ result.at<gtsam::Pose3>(kred) };
      //   const gtsam::Pose3 green_pose{ result.at<gtsam::Pose3>(kgreen) };
      //   const gtsam::Pose3 blue_pose{ result.at<gtsam::Pose3>(kblue) };

      //   const std::string red_str{ tensegrity::utils::convert_to<std::string>(red_pose) };
      //   const std::string green_str{ tensegrity::utils::convert_to<std::string>(green_pose) };
      //   const std::string blue_str{ tensegrity::utils::convert_to<std::string>(blue_pose) };

      const std::string t0str{ tensegrity::utils::convert_to<std::string>(t0.toSec()) };
      const std::string tTstr{ tensegrity::utils::convert_to<std::string>(tT.toSec()) };
      DEBUG_VARS(t0str, tTstr)

      _json_file << "{\n";
      _json_file << "\"N\": " << _N << ",\n";
      _json_file << "\"a\": " << t0str << ",\n";
      _json_file << "\"b\": " << tTstr << ",\n";
      _json_file << "\"offset\": [" << _offset[0] << ", " << _offset[1] << ", " << _offset[2] << "],\n";
      _json_file << "\"0\": " << cheb0_str << ",\n";
      _json_file << "\"1\": " << cheb1_str << ",\n";
      _json_file << "\"2\": " << cheb2_str << "\n";
      _json_file << "}";
      //   _poses_file << idx << " ";
      //   _poses_file << timestamps[idx] << " ";
      //   _poses_file << red_str << " ";
      //   _poses_file << green_str << " ";
      //   _poses_file << blue_str << " ";
      //   _poses_file << "\n";
      //   estimation::publish_tensegrity_msg(red_pose, green_pose, blue_pose, _tensegrity_bars_publisher,
      //   "real_sense");
      // }
      // else
      // {
      _json_file.close();
      ros::shutdown();
      // }
    }
  }

  void bar_callback(const interface::TensegrityEndcapsConstPtr msg)
  {
    if (t0.isZero())
    {
      t0 = msg->header.stamp;
    }
    endcaps.emplace_back();
    timestamps.push_back(msg->header.stamp);
    tT = msg->header.stamp;
    for (int i = 0; i < 6; ++i)
    {
      const geometry_msgs::Point& endcap{ msg->endcaps[i] };
      endcaps.back()[i] = Eigen::Vector3d(endcap.x, endcap.y, endcap.z);
      _markers[i].points.push_back(endcap);
      _markers[i].header = msg->header;
    }
  }
};

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "ChevEstimation");
  ros::NodeHandle nh("~");

  chev_estimation_t estimator(nh);
  ros::spin();

  return 0;
}
