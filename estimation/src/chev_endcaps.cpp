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
#include <estimation/cheb_factors.hpp>
#include <estimation/cheb_utils.hpp>

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

using LieIntegrator = factor_graphs::lie_integration_factor_t<SE3, Velocity>;
// using ChebManifoldSE3 = manifold_evaluation_t<gtsam::Chebyshev2, SE3>;
// using ChebClass = gtsam::Chebyshev2Basis;
using ChebClass = gtsam::Chebyshev2;
using VectorEvaluationFactor = gtsam::VectorEvaluationFactor<ChebClass, 3>;
// using VectorEvaluationFactor = gtsam::VectorEvaluationFactor<gtsam::Chebyshev2Basis, 3>;

class cable_length_cheb_factor_t : public gtsam::NoiseModelFactorN<gtsam::ParameterMatrix<3>, gtsam::ParameterMatrix<3>>
{
public:
  // using SE3 = gtsam::Pose3;
  using ParameterMatrix = gtsam::ParameterMatrix<3>;
  using Translation = Eigen::Vector<double, 3>;
  using Rotation = gtsam::Rot3;
  using SkewMatrix = Eigen::Matrix<double, 3, 3>;
  using Base = gtsam::NoiseModelFactorN<gtsam::ParameterMatrix<3>, gtsam::ParameterMatrix<3>>;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;
  using Jacobian = Eigen::Matrix<double, 3, 6>;
  using Meassurement = Eigen::Vector<double, 1>;
  using Error = Eigen::Vector<double, 1>;

  // using BarFactor = bar_two_observations_factor_t;
  // using BarFactor = endcap_observation_factor_t;
  // using Vector = Eigen::Vector<double, Dim>;

  cable_length_cheb_factor_t(const gtsam::Key key_pmi, const gtsam::Key key_pmj, const double zij, const int N,
                             const double ti, const double a, const double b, const NoiseModel& cost_model)
    : Base(cost_model, key_pmi, key_pmj), _zij(zij), _func(N, ti, a, b)
  {
  }

  static Meassurement predict(const ParameterMatrix& pmi, const ParameterMatrix& pmj,  // no-lint
                              ChebClass::VectorEvaluationFunctor<3> func,              // no-lint
                              gtsam::OptionalJacobian<-1, -1> Hpmi = boost::none,
                              gtsam::OptionalJacobian<-1, -1> Hpmj = boost::none)
  {
    Eigen::MatrixXd ei_H_pmi, ei_H_pmj;
    const bool compute_deriv{ Hpmi or Hpmj };

    const Translation ei{ func(pmi, compute_deriv ? &ei_H_pmi : nullptr) };
    const Translation ej{ func(pmj, compute_deriv ? &ei_H_pmj : nullptr) };

    const Eigen::Matrix<double, 3, 3> diff_H_ei{ Eigen::Matrix<double, 3, 3>::Identity() };
    const Eigen::Matrix<double, 3, 3> diff_H_ej{ -Eigen::Matrix<double, 3, 3>::Identity() };
    Eigen::Matrix<double, 1, 3> d_H_diff;
    const Translation diff{ ei - ej };
    const double distance{ gtsam::norm3(diff, compute_deriv ? &d_H_diff : nullptr) };
    // DEBUG_VARS(ei.transpose(), ej.transpose(), distance);

    if (Hpmi)
    {
      *Hpmi = d_H_diff * diff_H_ei * ei_H_pmi;
    }
    if (Hpmj)
    {
      *Hpmj = d_H_diff * diff_H_ej * ei_H_pmj;
    }
    return Meassurement(distance);
  }

  virtual Eigen::VectorXd evaluateError(const ParameterMatrix& pmi, const ParameterMatrix& pmj,  // no-lint
                                        boost::optional<Eigen::MatrixXd&> Hpmi = boost::none,
                                        boost::optional<Eigen::MatrixXd&> Hpmj = boost::none) const override
  {
    const Meassurement prediction{ predict(pmi, pmj, _func, Hpmi, Hpmj) };

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
  const ChebClass::VectorEvaluationFunctor<3> _func;

  const Meassurement _zij;
};

struct chev_estimation_t
{
  using This = chev_estimation_t;
  // using ManifoldEvaluationFactor = gtsam::ManifoldEvaluationFactor<gtsam::Chebyshev2, gtsam::Pose3>;

  ros::Timer _finish_timer;
  ros::Subscriber _bars_subscriber;
  std::array<ros::Publisher, 6> _markers_publishers;
  std::array<ros::Publisher, 6> _endcap_markers_publishers;
  std::array<ros::Publisher, 3> _bars_markers_publishers;
  // ros::Publisher _tensegrity_bars_publisher, _tensegrity_traj_publisher;

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
  std::vector<std::array<gtsam::Pose3, 3>> vector_bars;
  std::vector<std::array<Eigen::Vector3d, 6>> endcaps;
  std::vector<ros::Time> timestamps;
  ros::Time t0;

  int _N, _Nbars;
  double a, b;
  double dt;
  double maxT;
  bool fg_not_run;
  // gtsam::ParameterMatrix<3> _red_cheb_matrix, _green_cheb_matrix, _blue_cheb_matrix;
  std::vector<gtsam::ParameterMatrix<3>> _chev_mats;
  std::vector<gtsam::ParameterMatrix<6>> _chev_bars_mats;
  ros::Publisher _tensegrity_bars_publisher;
  // double _a, _b, _N;
  // gtsam::ParameterMatrix<DIM> _poly_matrix;
  gtsam::Values result;
  std::ofstream _poses_file;
  std::array<visualization_msgs::Marker, 6> _markers;
  std::array<visualization_msgs::Marker, 6> _endcap_markers;
  std::array<visualization_msgs::Marker, 3> _bars_markers;
  std::shared_ptr<estimation::sensors_callback_t> sensors_callback;
  estimation::ColorMappingNum cable_map;
  std::shared_ptr<interface::ros_tf_utils_t> _tf_utils;
  double _Nbars_ratio;
  bool use_cable_sensors;

  chev_estimation_t(ros::NodeHandle& nh)
    : _idx(0)
    , _offset({ 0, 0, 0.36 / 2.0 })
    , _Roffset(0.0, 0.0, 1.0, 0.0)
    , node_status(interface::node_status_t::create(nh))
    , prev_dt(0)
    , _bars({ factor_graphs::random<SE3>(), factor_graphs::random<SE3>(), factor_graphs::random<SE3>() })
    , _colors({ estimation::RodColors::RED })
    , t0(0)
    , fg_not_run(true)
    , use_cable_sensors(true)
  // ,_chev_mats()
  // , _red_cheb_matrix(5)
  // , _green_cheb_matrix(5)
  // , _blue_cheb_matrix(5)
  // , _colors({ estimation::RodColors::RED, estimation::RodColors::GREEN, estimation::RodColors::BLUE })
  {
    // lm_params.setVerbosityLM("SILENT");
    _lm_params.setVerbosityLM("SUMMARY");
    _lm_params.setMaxIterations(100);
    _lm_helper = std::make_shared<factor_graphs::levenberg_marquardt_t>(nh, "/nodes/state_estimation/fg", _lm_params);

    std::string tensegrity_bars_topicname;
    std::string publisher_node_id;  // subscribe to this, when it finished, run the estimation
    // std::string tensegrity_pose_topic, tensegrity_traj_topic;
    std::string cheb_json_filename, cable_map_filename;
    // std::string cables_topic;
    // std::string initial_estimate_filename;
    // double resolution;
    double trajectory_pub_frequency;
    double& Nbars_ratio{ _Nbars_ratio };

    // int& N{ _N };
    PARAM_SETUP(nh, tensegrity_bars_topicname);
    PARAM_SETUP(nh, publisher_node_id);
    PARAM_SETUP(nh, cheb_json_filename);
    PARAM_SETUP(nh, cable_map_filename);
    PARAM_SETUP(nh, use_cable_sensors);
    PARAM_SETUP(nh, Nbars_ratio);

    sensors_callback = std::make_shared<estimation::sensors_callback_t>(nh);
    cable_map = estimation::create_cable_map_fix_endcaps(cable_map_filename);
    _tf_utils = std::make_shared<interface::ros_tf_utils_t>(nh);

    DEBUG_VARS(cheb_json_filename);
    _poses_file.open(cheb_json_filename, std::ios::out);

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
      _markers[i].type = visualization_msgs::Marker::LINE_STRIP;
      _markers[i].scale.x = 0.005;  // is point width,
      _markers[i].scale.y = 0.005;  // is point height
      _markers[i].color.r = colors[i][0];
      _markers[i].color.g = colors[i][1];
      _markers[i].color.b = colors[i][2];
      _markers[i].color.a = 1.0;

      tensegrity::utils::init_header(_endcap_markers[i].header, "real_sense");
      _endcap_markers[i].action = visualization_msgs::Marker::ADD;
      _endcap_markers[i].type = visualization_msgs::Marker::POINTS;
      _endcap_markers[i].scale.x = 0.005;  // is point width,
      _endcap_markers[i].scale.y = 0.005;  // is point height
      _endcap_markers[i].color.r = colors[i][0];
      _endcap_markers[i].color.g = colors[i][1];
      _endcap_markers[i].color.b = colors[i][2];
      _endcap_markers[i].color.a = 1.0;

      const std::string topicname{ "/tensegrity/endcaps/chev/e" + std::to_string(i) };
      const std::string endcap_topicname{ "/tensegrity/endcaps/input/e" + std::to_string(i) };
      _markers_publishers[i] = nh.advertise<visualization_msgs::Marker>(topicname, 1, true);
      _endcap_markers_publishers[i] = nh.advertise<visualization_msgs::Marker>(endcap_topicname, 1, true);
    }
    for (int i = 0; i < 3; ++i)
    {
      tensegrity::utils::init_header(_bars_markers[i].header, "real_sense");
      _bars_markers[i].action = visualization_msgs::Marker::ADD;
      _bars_markers[i].type = visualization_msgs::Marker::LINE_STRIP;
      _bars_markers[i].scale.x = 0.003;  // is point width,
      _bars_markers[i].scale.y = 0.003;  // is point height
      _bars_markers[i].color.r = colors[i * 2][0];
      _bars_markers[i].color.g = colors[i * 2][1];
      _bars_markers[i].color.b = colors[i * 2][2];
      _bars_markers[i].color.a = 1.0;

      const std::string topicname{ "/tensegrity/bars/chev/e" + std::to_string(i) };
      _bars_markers_publishers[i] = nh.advertise<visualization_msgs::Marker>(topicname, 1, true);
    }
    node_status->status(interface::NodeStatus::READY);
  }

  ~chev_estimation_t()
  {
  }

  static gtsam::Key endcap_key(const estimation::RodColors rod, const int t, const int i = 0)
  {
    return factor_graphs::symbol_factory_t::create_hashed_symbol(estimation::color_str(rod), "^{", t, "}_{", i, "}");
  }
  static gtsam::Key bar_key(const estimation::RodColors rod, const int t, const int i = 0)
  {
    return factor_graphs::symbol_factory_t::create_hashed_symbol("B", estimation::color_str(rod), "^{", t, "}_{", i,
                                                                 "}");
  }

  static std::vector<gtsam::ParameterMatrix<3>> estimate_endcap_trajectory(
      const int Ncurr, const std::vector<std::array<Eigen::Vector3d, 6>>& endcaps_curr,
      const std::vector<ros::Time>& timestamps_curr, const std::shared_ptr<estimation::sensors_callback_t> sensors,
      const estimation::ColorMappingNum& cable_map, std::shared_ptr<factor_graphs::levenberg_marquardt_t> lm_helper,
      const double a, const double b, const bool use_cable_sensors)
  {
    using EndcapObservationFactor = estimation::endcap_observation_factor_t;
    using RotationOffsetFactor = estimation::endcap_rotation_offset_factor_t;
    using RotationFixIdentity = estimation::rotation_fix_identity_t;
    std::array<gtsam::Key, 6> kendcaps;

    kendcaps[0] = endcap_key(estimation::RodColors::RED, 0);
    kendcaps[1] = endcap_key(estimation::RodColors::RED, 1);
    kendcaps[2] = endcap_key(estimation::RodColors::GREEN, 0);
    kendcaps[3] = endcap_key(estimation::RodColors::GREEN, 1);
    kendcaps[4] = endcap_key(estimation::RodColors::BLUE, 0);
    kendcaps[5] = endcap_key(estimation::RodColors::BLUE, 1);

    gtsam::Values values;
    gtsam::NonlinearFactorGraph graph;

    auto model = gtsam::noiseModel::Isotropic::Sigma(3, 1e0);
    for (int i = 0; i < 6; ++i)
    {
      values.insert(kendcaps[i], gtsam::ParameterMatrix<3>(Ncurr));
    }

    gtsam::noiseModel::Base::shared_ptr cable_noise{ gtsam::noiseModel::Isotropic::Sigma(1, 1e0) };

    // const ros::Time t0{ std::min(timestamps_curr[0], std::get<3>(sensors->at(0))) };
    // const ros::Time tT{ std::max(timestamps_curr.back(), std::get<3>(sensors->back())) };

    // const double a{ t0.toSec() };
    // const double b{ tT.toSec() };
    DEBUG_VARS(a, b, Ncurr)
    for (int i = 0; i < endcaps_curr.size(); ++i)
    {
      const double ti{ timestamps_curr[i].toSec() };
      for (int j = 0; j < 6; ++j)
      {
        // const Translation zA{ transform.transformFrom(endcaps_curr[i][j]) };
        const Translation zA{ endcaps_curr[i][j] };
        if (not std::isnan(zA.template maxCoeff<Eigen::PropagateNaN>()))
        {
          graph.emplace_shared<VectorEvaluationFactor>(kendcaps[j], zA, model, Ncurr, ti, a, b);
        }
      }
    }

    if (use_cable_sensors)
    {
      values = lm_helper->optimize(graph, values, true);
      for (int i = 0; i < sensors->size(); ++i)
      {
        const estimation::sensors_callback_t::Meassurements m{ sensors->at(i) };
        const estimation::sensors_callback_t::Cables& cables{ std::get<0>(m) };
        const ros::Time& tros{ std::get<3>(sensors->at(i)) };
        const double ti{ tros.toSec() };
        int j = 0;
        // DEBUG_VARS(cables.transpose())
        for (; j < 3; ++j)
        {
          const gtsam::Key key_Xi{ kendcaps[cable_map[j].first] };
          const gtsam::Key key_Xj{ kendcaps[cable_map[j].second] };

          graph.emplace_shared<cable_length_cheb_factor_t>(key_Xi, key_Xj, cables[j], Ncurr, ti, a, b, cable_noise);
        }
        for (; j < 6; ++j)
        {
          const gtsam::Key key_Xi{ kendcaps[cable_map[j].first] };
          const gtsam::Key key_Xj{ kendcaps[cable_map[j].second] };

          graph.emplace_shared<cable_length_cheb_factor_t>(key_Xi, key_Xj, cables[j], Ncurr, ti, a, b, cable_noise);
        }
        for (; j < 9; ++j)
        {
          const gtsam::Key key_Xi{ kendcaps[cable_map[j].first] };
          const gtsam::Key key_Xj{ kendcaps[cable_map[j].second] };

          graph.emplace_shared<cable_length_cheb_factor_t>(key_Xi, key_Xj, cables[j], Ncurr, ti, a, b, cable_noise);
        }
      }
    }
    const gtsam::Values result{ lm_helper->optimize(graph, values, true) };

    std::vector<gtsam::ParameterMatrix<3>> chev_mats;

    for (int i = 0; i < 6; ++i)
    {
      chev_mats.push_back(result.at<gtsam::ParameterMatrix<3>>(kendcaps[i]));
    }
    return chev_mats;
  }

  void run_fg()
  {
    // using EndcapObservationFactor = estimation::endcap_observation_factor_t;
    // using RotationOffsetFactor = estimation::endcap_rotation_offset_factor_t;
    // using RotationFixIdentity = estimation::rotation_fix_identity_t;
    // std::array<gtsam::Key, 6> kendcaps;

    // kendcaps[0] = endcap_key(estimation::RodColors::RED, 0);
    // kendcaps[1] = endcap_key(estimation::RodColors::RED, 1);
    // kendcaps[2] = endcap_key(estimation::RodColors::GREEN, 0);
    // kendcaps[3] = endcap_key(estimation::RodColors::GREEN, 1);
    // kendcaps[4] = endcap_key(estimation::RodColors::BLUE, 0);
    // kendcaps[5] = endcap_key(estimation::RodColors::BLUE, 1);

    // gtsam::Values values;
    // gtsam::NonlinearFactorGraph graph;

    // gtsam::Pose3 transform;
    // const bool tf_received{ _tf_utils->query(transform) };
    // TENSEGRITY_ASSERT(tf_received, "Transform not available!");

    // auto model = gtsam::noiseModel::Isotropic::Sigma(3, 1e0);
    // // auto model = gtsam::noiseModel::Diagonal::Sigmas(Eigen::Vector<double, 6>(1, 1, 0, 1, 1, 1));
    // // _N = endcaps.size() / 2;
    // // _N = 100;
    // // _N = 25;
    // for (int i = 0; i < 6; ++i)
    // {
    //   // _chev_mats.push_back();
    //   values.insert(kendcaps[i], gtsam::ParameterMatrix<3>(_N));
    // }
    // // _red_cheb_matrix = gtsam::ParameterMatrix<3>(_N);
    // // _green_cheb_matrix = gtsam::ParameterMatrix<3>(_N);
    // // _blue_cheb_matrix = gtsam::ParameterMatrix<3>(_N);

    // // values.insert(kred, _red_cheb_matrix);
    // // values.insert(kgreen, _green_cheb_matrix);
    // // values.insert(kblue, _blue_cheb_matrix);
    // gtsam::noiseModel::Base::shared_ptr cable_noise{ gtsam::noiseModel::Isotropic::Sigma(1, 1e0) };

    // const ros::Time t0{ std::min(timestamps[0], std::get<3>(sensors_callback->at(0))) };
    // const ros::Time tT{ std::max(timestamps.back(), std::get<3>(sensors_callback->back())) };

    // a = t0.toSec();
    // b = tT.toSec();
    // DEBUG_VARS(a, b, _N)
    // // maxT = (tT - t0).toSec();
    // for (int i = 0; i < endcaps.size(); ++i)
    // {
    //   const double ti{ timestamps[i].toSec() };
    //   // const double ti{ (timestamps[i] - t0).toSec() };
    //   for (int j = 0; j < 6; ++j)
    //   {
    //     // transform
    //     const Translation zA{ transform.transformFrom(endcaps[i][j]) };
    //     graph.emplace_shared<VectorEvaluationFactor>(kendcaps[j], zA, model, _N, ti, a, b);
    //   }
    // }
    // values = _lm_helper->optimize(graph, values, true);

    // // for (int j = 0; j < 9; ++j)
    // // {
    // //   const gtsam::Key key_Xi{ kendcaps[cable_map[j].first] };
    // //   const gtsam::Key key_Xj{ kendcaps[cable_map[j].second] };
    // //   // DEBUG_VARS(cable_map[j].first, cable_map[j].second);
    // //   // PRINT_KEYS(key_Xi, key_Xj);
    // // }

    // for (int i = 0; i < sensors_callback->size(); ++i)
    // {
    //   const estimation::sensors_callback_t::Meassurements m{ sensors_callback->at(i) };
    //   const estimation::sensors_callback_t::Cables& cables{ std::get<0>(m) };
    //   const ros::Time& tros{ std::get<3>(sensors_callback->at(i)) };
    //   const double ti{ tros.toSec() };
    //   // const double ti{ (tros - t0).toSec() };
    //   int j = 0;
    //   // DEBUG_VARS(ti, cables.transpose());
    //   for (; j < 3; ++j)
    //   {
    //     const gtsam::Key key_Xi{ kendcaps[cable_map[j].first] };
    //     const gtsam::Key key_Xj{ kendcaps[cable_map[j].second] };

    //     graph.emplace_shared<cable_length_cheb_factor_t>(key_Xi, key_Xj, cables[j], _N, ti, a, b, cable_noise);
    //   }
    //   for (; j < 6; ++j)
    //   {
    //     const gtsam::Key key_Xi{ kendcaps[cable_map[j].first] };
    //     const gtsam::Key key_Xj{ kendcaps[cable_map[j].second] };

    //     graph.emplace_shared<cable_length_cheb_factor_t>(key_Xi, key_Xj, cables[j], _N, ti, a, b, cable_noise);
    //   }
    //   for (; j < 9; ++j)
    //   {
    //     const gtsam::Key key_Xi{ kendcaps[cable_map[j].first] };
    //     const gtsam::Key key_Xj{ kendcaps[cable_map[j].second] };

    //     graph.emplace_shared<cable_length_cheb_factor_t>(key_Xi, key_Xj, cables[j], _N, ti, a, b, cable_noise);
    //   }
    // }

    // // values.insert(gtsam::Symbol('R', vector_bars.size() - 1), vector_bars.back()[0]);
    // // values.insert(gtsam::Symbol('G', vector_bars.size() - 1), vector_bars.back()[1]);
    // // values.insert(gtsam::Symbol('B', vector_bars.size() - 1), vector_bars.back()[2]);

    // result = _lm_helper->optimize(graph, values, true);
    // for (int i = 0; i < 6; ++i)
    // {
    //   _chev_mats.push_back(result.at<gtsam::ParameterMatrix<3>>(kendcaps[i]));
    // }
    gtsam::Pose3 transform;
    const bool tf_received{ _tf_utils->query(transform) };
    TENSEGRITY_ASSERT(tf_received, "Transform not available!");
    TENSEGRITY_ASSERT(timestamps.size() > 0, "timestamps size is zero!");
    TENSEGRITY_ASSERT(sensors_callback->size() > 0, "sensors size is zero!");

    const ros::Time t0{ std::min(timestamps[0], std::get<3>(sensors_callback->at(0))) };
    const ros::Time tT{ std::max(timestamps.back(), std::get<3>(sensors_callback->back())) };

    a = t0.toSec();
    b = tT.toSec();

    std::vector<std::array<Eigen::Vector3d, 6>> endcaps_train, endcaps_test;
    std::vector<ros::Time> timestamps_train, timestamps_test;

    // const Translation zA{ transform.transformFrom(endcaps_curr[i][j]) };

    endcaps_train.push_back(endcaps[0]);
    timestamps_train.push_back(timestamps[0]);
    for (int i = 1; i < endcaps.size() - 1; ++i)
    {
      if (factor_graphs::random_uniform(0.0, 1.0) < 0.7)
      {
        endcaps_train.push_back(endcaps[i]);
        timestamps_train.push_back(timestamps[i]);
      }
      else
      {
        endcaps_test.push_back(endcaps[i]);
        timestamps_test.push_back(timestamps[i]);
      }
    }
    endcaps_train.push_back(endcaps.back());
    timestamps_train.push_back(timestamps.back());

    transform_endcaps(endcaps_test, transform);
    transform_endcaps(endcaps_train, transform);

    // int Nlow{ endcaps_test.size() * 0.1 };   // Start with N <- 10% of test
    // int Nhigh{ endcaps_test.size() * 0.5 };  // High set to 0.5 since its highly unlikely that 50% of data is close
    // to chev points
    int Ni{ static_cast<int>(endcaps_test.size() * 0.1) };
    // auto binary_search = [&](int Ni, int Nlow, int Nhigh) {
    //   _chev_mats = estimate_endcap_trajectory(Ni, endcaps_train, timestamps_train, sensors_callback, cable_map,
    //                                           _lm_helper, a, b);

    //   const double error{ evaluate_estimation(endcaps_test, timestamps_test, _chev_mats, Ni, a, b) };
    // };
    double curr_err{ 0.0 };
    double prev_err{ std::numeric_limits<double>::max() };
    // for (int Ni = 10; Ni < 40; ++Ni)
    std::vector<gtsam::ParameterMatrix<3>> curr_cheb_mats;

    while (true)
    {
      std::vector<gtsam::ParameterMatrix<3>> curr_cheb_mats{ estimate_endcap_trajectory(
          Ni, endcaps_train, timestamps_train, sensors_callback, cable_map, _lm_helper, a, b, use_cable_sensors) };

      curr_err = evaluate_estimation(endcaps_test, timestamps_test, curr_cheb_mats, Ni, a, b);
      if (prev_err > curr_err)
      {
        prev_err = curr_err;
        _chev_mats.swap(curr_cheb_mats);
        _N = Ni;
        Ni++;
      }
      else
      {
        break;
      }
    }
    cheb_endcaps_to_bars();
  }

  static void transform_endcaps(std::vector<std::array<Eigen::Vector3d, 6>>& endcaps_data, const gtsam::Pose3& tf)
  {
    for (auto& vt : endcaps_data)
    {
      for (int i = 0; i < 6; ++i)
      {
        if (not std::isnan(vt[i].template maxCoeff<Eigen::PropagateNaN>()))
        {
          vt[i] = tf.transformFrom(vt[i]);
        }
      }
    }
  }

  static double evaluate_estimation(const std::vector<std::array<Eigen::Vector3d, 6>>& endcaps_test,
                                    const std::vector<ros::Time>& timestamps_test,
                                    const std::vector<gtsam::ParameterMatrix<3>>& cheb_mats, const int Ntest,
                                    const double a, const double b)
  {
    double error{ 0 };
    for (int i = 0; i < timestamps_test.size(); ++i)
    {
      const double& ti{ timestamps_test[i].toSec() };
      const ChebClass::VectorEvaluationFunctor<3> func(Ntest, ti, a, b);
      for (int j = 0; j < 6; ++j)
      {
        const Eigen::Vector3d z{ endcaps_test[i][j] };

        if (not std::isnan(z.template maxCoeff<Eigen::PropagateNaN>()))
        {
          const Eigen::Vector3d e{ func(cheb_mats[j]) };

          error += (z - e).norm();
        }
      }
    }
    return error;
  }

  void cheb_endcaps_to_bars()
  {
    // gtsam::Chebyshev2
    using EndcapsChebFactor = estimation::endcaps_cheb_factor_t<gtsam::Chebyshev2Basis>;
    // using EndcapsChebFactor = estimation::endcaps_cheb_factor_t<ChebClass>;
    // int N{ 40 };
    // _Nbars = 40;

    std::array<gtsam::Key, 3> kbars;

    kbars[0] = bar_key(estimation::RodColors::RED, 0);
    kbars[1] = bar_key(estimation::RodColors::GREEN, 0);
    kbars[2] = bar_key(estimation::RodColors::BLUE, 0);
    gtsam::noiseModel::Base::shared_ptr z_noise{ gtsam::noiseModel::Isotropic::Sigma(3, 1e0) };
    int Nstep{ static_cast<int>(endcaps.size() * 0.2) };
    _Nbars = 0;
    DEBUG_VARS(Nstep, endcaps.size())
    bool succeded{ false };
    while (not succeded)
    {
      _Nbars += Nstep;
      TENSEGRITY_ASSERT(_Nbars < endcaps.size(), "Max coeffs reached!")
      DEBUG_VARS(_Nbars)
      gtsam::Values values;
      gtsam::NonlinearFactorGraph graph;
      Eigen::VectorXd pts{ gtsam::Chebyshev2::Points(_Nbars, a, b) };

      for (int i = 0; i < 3; ++i)
      {
        values.insert(kbars[i], gtsam::ParameterMatrix<6>(_Nbars));
      }
      for (auto ti : pts)
      {
        const ChebClass::VectorEvaluationFunctor<3> func(_N, ti, a, b);
        // const std::string tistr{ tensegrity::utils::convert_to<std::string>(ti) };
        // const std::string astr{ tensegrity::utils::convert_to<std::string>(a) };
        // const std::string bstr{ tensegrity::utils::convert_to<std::string>(b) };
        // DEBUG_VARS(tistr, N, astr, bstr);
        for (int j = 0; j < 3; ++j)
        {
          const Eigen::Vector3d zA{ func(_chev_mats[j * 2]) };
          const Eigen::Vector3d zB{ func(_chev_mats[j * 2 + 1]) };
          // DEBUG_VARS(zA.transpose(), zB.transpose());
          graph.emplace_shared<EndcapsChebFactor>(kbars[j], _offset, zA, _Nbars, ti, a, b, z_noise);
          graph.emplace_shared<EndcapsChebFactor>(kbars[j], -_offset, zB, _Nbars, ti, a, b, z_noise);
        }
      }

      const gtsam::Values res{ _lm_helper->optimize(graph, values, true) };
      std::vector<gtsam::ParameterMatrix<6>> curr_cheb_mats;
      for (int i = 0; i < 3; ++i)
      {
        // _chev_bars_mats.push_back(res.at<gtsam::ParameterMatrix<6>>(kbars[i]));
        curr_cheb_mats.push_back(res.at<gtsam::ParameterMatrix<6>>(kbars[i]));
        // gtsam::ParameterMatrix<6> cheb0{ res.at<gtsam::ParameterMatrix<6>>(kbars[0]) };
        // DEBUG_VARS(cheb0.transpose());
      }

      // curr_cheb_mats.push_back(_chev_bars_mats[0]);
      succeded = compute_envelope(curr_cheb_mats, _Nbars);
    }
  }

  bool compute_envelope(std::vector<gtsam::ParameterMatrix<6>>& chebs, const int Nb)
  {
    Eigen::VectorXd envelope(Eigen::VectorXd::Zero(Nb));

    for (int i = 0; i < Nb; ++i)
    {
      double env{ -1 };
      for (int j = i; j < Nb; ++j)
      {
        for (auto& ch : chebs)
        {
          const Eigen::MatrixXd mat{ ch.matrix() };
          for (int k = 0; k < 6; ++k)
          {
            env = std::max(std::fabs(mat(k, j)), env);
          }
        }
      }
      envelope[i] = env;
    }
    TENSEGRITY_ASSERT(envelope[0] > 0.0, "Envelope(0) is Zero!")
    envelope = envelope / envelope[0];

    const double log_tol{ std::log(1e-4) };
    int Nnew{ Nb };
    bool success{ false };
    // Find plateau
    for (int i = 0; i < Nb; ++i)
    {
      const double& env{ envelope[i] };
      const double env_j2{ std::round(1.25 * i + 5) };
      const double r{ 3.0 * (1.0 - std::log(env) / log_tol) };
      const double coeff{ env / env_j2 };
      if (coeff > r)
      {
        DEBUG_VARS(i, env, env_j2, coeff, r);
        Nnew = i;
        success = true;
        break;
      }
    }
    //
    // // Block of size (p,q), starting at (i,j)  matrix.block(i,j,p,q);
    if (success)
    {
      for (int i = 0; i < chebs.size(); ++i)
      {
        _chev_bars_mats.emplace_back(chebs[i].matrix().block(0, 0, 6, Nnew));
      }
      _Nbars = Nnew;
    }
    else
    {
      PRINT_MSG("Plateau not reached")
      DEBUG_VARS(_Nbars)
      // DEBUG_VARS(chebs[0])
    }
    return success;
    // DEBUG_PRINT

    // for (int i = 0; i < Nb; ++i)
    // {
    //   std::vector<double> c;
    //   for (int j = 0; j < 6; ++j)
    //   {
    //     c.push_back(chebs[0].matrix()(j, i));
    //   }
    //   c.push_back(envelope[i]);
    //   DEBUG_VARS(c);
    // }
    // DEBUG_VARS(chebs[0]);
    // DEBUG_VARS(envelope);
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
      for (double ti = a; ti < b; ti += 0.1)
      {
        const ChebClass::VectorEvaluationFunctor<3> f(_N, ti, a, b);
        const gtsam::Chebyshev2Basis::ManifoldEvaluationFunctor<SE3> f_se3(_Nbars, ti, a, b);
        // const Eigen::Vector3d red_endcap{ f(_red_cheb_matrix) };
        // const Eigen::Vector3d green_endcap{ f(_green_cheb_matrix) };
        // const Eigen::Vector3d blue_endcap{ f(_blue_cheb_matrix) };

        for (int i = 0; i < 6; ++i)
        {
          const Eigen::Vector3d curr_endcap{ f(_chev_mats[i]) };
          // DEBUG_VARS(_N, ti, maxT, curr_endcap.transpose());
          _markers[i].points.emplace_back();
          _markers[i].points.back().x = curr_endcap[0];
          _markers[i].points.back().y = curr_endcap[1];
          _markers[i].points.back().z = curr_endcap[2];
        }

        for (int i = 0; i < 3; ++i)
        {
          const gtsam::Pose3 curr_xi{ f_se3(_chev_bars_mats[i]) };

          const Translation pti{ curr_xi.translation() };
          _bars_markers[i].points.emplace_back();
          _bars_markers[i].points.back().x = pti[0];
          _bars_markers[i].points.back().y = pti[1];
          _bars_markers[i].points.back().z = pti[2];
        }

        // _poses_file << "\n";
        // estimation::publish_tensegrity_msg(red_pose, green_pose, blue_pose, _tensegrity_bars_publisher,
        // "real_sense");
      }
      // _markers_publishers[0].publish(_markers[0]);

      for (int i = 0; i < 6; ++i)
      {
        _markers[i].header.frame_id = _tf_utils->world_frame();
        _endcap_markers[i].header.frame_id = _tf_utils->other_frame();
        // _endcap_markers_publishers[i].header.frame_id = _tf_utils->world_frame();

        _endcap_markers_publishers[i].publish(_endcap_markers[i]);
        _markers_publishers[i].publish(_markers[i]);
      }

      for (int i = 0; i < 3; ++i)
      {
        _bars_markers[i].header.frame_id = _tf_utils->world_frame();

        _bars_markers_publishers[i].publish(_bars_markers[i]);
      }

      const std::string cheb0_str{ estimation::cheb_to_json_array(_chev_bars_mats[0]) };
      const std::string cheb1_str{ estimation::cheb_to_json_array(_chev_bars_mats[1]) };
      const std::string cheb2_str{ estimation::cheb_to_json_array(_chev_bars_mats[2]) };

      const std::string astr{ tensegrity::utils::convert_to<std::string>(a) };
      const std::string bstr{ tensegrity::utils::convert_to<std::string>(b) };

      // DEBUG_VARS(cheb0_str)
      _poses_file << "{\n";
      _poses_file << "\"N\": " << _Nbars << ",\n";
      _poses_file << "\"a\": " << astr << ",\n";
      _poses_file << "\"b\": " << bstr << ",\n";
      _poses_file << "\"offset\": [" << _offset[0] << ", " << _offset[1] << ", " << _offset[2] << "],\n";
      _poses_file << "\"0\": " << cheb0_str << ",\n";
      _poses_file << "\"1\": " << cheb1_str << ",\n";
      _poses_file << "\"2\": " << cheb2_str << "\n";
      _poses_file << "}";
      _poses_file.close();
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
    timestamps.emplace_back(msg->header.stamp);
    for (int i = 0; i < 6; ++i)
    {
      const geometry_msgs::Point& endcap{ msg->endcaps[i] };
      endcaps.back()[i] = Eigen::Vector3d(endcap.x, endcap.y, endcap.z);
      _endcap_markers[i].points.push_back(endcap);
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
