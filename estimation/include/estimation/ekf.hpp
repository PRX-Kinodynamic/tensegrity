#pragma once

#include <factor_graphs/defs.hpp>

#include <interface/TensegrityLengthSensor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/linear/GaussianConditional.h>
#include <gtsam/linear/GaussianBayesNet.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/nonlinear/Marginals.h>

#include <estimation/bar_utilities.hpp>
namespace estimation
{
class lie_motion_model_factor_t : public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Pose3>
{
  using SE3 = gtsam::Pose3;
  using Rotation = gtsam::Rot3;
  using Base = gtsam::NoiseModelFactorN<SE3, SE3>;

  static constexpr Eigen::Index DimX{ gtsam::traits<SE3>::dimension };
  using Xdot = Eigen::Vector<double, DimX>;

  using LieIntegrator = factor_graphs::lie_integration_factor_t<SE3, Xdot>;

  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;

  using DerivativeX = Eigen::Matrix<double, DimX, DimX>;
  using OptDeriv = boost::optional<Eigen::MatrixXd&>;
  template <typename T>
  using OptionalMatrix = boost::optional<Eigen::MatrixXd&>;

  lie_motion_model_factor_t() = delete;
  lie_motion_model_factor_t(const lie_motion_model_factor_t& other) = delete;

public:
  lie_motion_model_factor_t(const gtsam::Key key_xt1, const gtsam::Key key_xt0, const NoiseModel& cost_model,
                            const Xdot xdot, const double dt)
    : Base(cost_model, key_xt1, key_xt0), _xdot(xdot), _dt(dt)
  {
  }

  ~lie_motion_model_factor_t() override
  {
  }

  virtual Eigen::VectorXd evaluateError(const SE3& x1, const SE3& x0,  // no-lint
                                        OptDeriv H1 = boost::none, OptDeriv H0 = boost::none) const override
  {
    // const bool compute_deriv{ H1 or H0 or HR };
    // Eigen::MatrixXd err_H_x1;
    // Eigen::MatrixXd err_H_x0;

    // Eigen::MatrixXd x1p_H_x1;
    // Eigen::MatrixXd x1p_H_rinv;
    // Eigen::MatrixXd rinv_H_rot;

    const Eigen::VectorXd error{ LieIntegrator::error(x1, x0, _xdot, _dt, H1, H0) };

    // if (H1)
    // {
    //   *H1 = err_H_x1p * x1p_H_x1;
    // }
    // if (H0)
    // {
    //   *H0 = err_H_x0;
    // }

    return error;
  }

private:
  const Xdot _xdot;
  const double _dt;
};

class extended_kalman_filter_t
{
  using SE3 = gtsam::Pose3;
  using Velocity = Eigen::Vector<double, 6>;

  using Rotation = gtsam::Rot3;
  using Translation = Eigen::Vector3d;

  using OptSE3 = std::optional<SE3>;
  using OptTranslation = std::optional<Translation>;
  using RodObservation = std::pair<OptTranslation, OptTranslation>;

  using Values = gtsam::Values;
  using GaussianFG = gtsam::GaussianFactorGraph;
  using GaussianCondPtr = gtsam::GaussianConditional::shared_ptr;
  using GaussianBayesNetPtr = gtsam::GaussianBayesNet::shared_ptr;

  using JacobianFactor = gtsam::JacobianFactor;
  using JacobianFactorPtr = gtsam::JacobianFactor::shared_ptr;
  using GaussianNoiseModel = gtsam::noiseModel::Gaussian::shared_ptr;

  using BarTwoEstimationFactor = estimation::bar_two_observations_factor_t;

  using LieIntegratorFactor = factor_graphs::lie_integration_factor_t<SE3, Velocity>;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;

  using SF = factor_graphs::symbol_factory_t;

  using BarEstimationFactor = estimation::bar_two_observations_factor_t;
  using EndcapObservationFactor = estimation::endcap_observation_factor_t;
  using RotationOffsetFactor = estimation::endcap_rotation_offset_factor_t;
  using RotationFixIdentity = estimation::rotation_fix_identity_t;

public:
  /// @name Standard Constructors
  /// @{

  // ExtendedKalmanFilter(Key key_initial, T x_initial, NoiseModel P_initial)
  extended_kalman_filter_t(ros::NodeHandle& nh, bool use_cables)
    : _idx(0)
    , k_Xcurrent({ rod_symbol(RodColors::RED, 0), rod_symbol(RodColors::GREEN, 0), rod_symbol(RodColors::BLUE, 0) })
    , k_Rcurrent({ rotation_symbol(RodColors::RED, 0), rotation_symbol(RodColors::GREEN, 0),
                   rotation_symbol(RodColors::BLUE, 0) })
    , _bars({ factor_graphs::random<SE3>(), factor_graphs::random<SE3>(), factor_graphs::random<SE3>() })
    , _rotations({ Rotation(), Rotation(), Rotation() })
    , _valid({ false, false, false })
    , _offset({ 0, 0, 0.325 / 2.0 })
    , _Roffset(0.0, 0.0, 1.0, 0.0)
    , _xdot_noise(gtsam::noiseModel::Isotropic::Sigma(6, 1e0))
    , _use_cables(use_cables)
  {
    _observation_update.offset = _offset;
    _observation_update.Roffset = _Roffset;

    // lm_params.setVerbosityLM("SILENT");
    _lm_params.setVerbosityLM("SUMMARY");
    _lm_params.setMaxIterations(10);

    std::string cable_map_filename;

    PARAM_SETUP(nh, cable_map_filename);

    _cable_map = estimation::create_cable_map(cable_map_filename);

    _lm_helper = std::make_shared<factor_graphs::levenberg_marquardt_t>(nh, "/nodes/state_estimation/fg", _lm_params);

    // const int n{ 6 };
    // // const gtsam::noiseModel initial_nm{ P_initial != nullptr ? P_initial : gtsam::noiseModel::Unit::Create(6) };
    // const auto initial_nm = gtsam::noiseModel::Unit::Create(6);
    // const Eigen::Vector<double, 6> intial_b{ Eigen::Vector<double, 6>::Zero() };
    // const Eigen::MatrixXd initial_R{ initial_nm->R() };
    // _priors[RodColors::RED] =
    //     JacobianFactorPtr(new JacobianFactor(k_Xcurrent[RodColors::RED], initial_R, intial_b, initial_nm));
    // _priors[RodColors::GREEN] =
    //     JacobianFactorPtr(new JacobianFactor(k_Xcurrent[RodColors::GREEN], initial_R, intial_b, initial_nm));
    // _priors[RodColors::BLUE] =
    //     JacobianFactorPtr(new JacobianFactor(k_Xcurrent[RodColors::BLUE], initial_R, intial_b, initial_nm));

    // const Eigen::Vector3d sigams_R({ 0, 0, 1 });
    // _rot_noise_model = gtsam::noiseModel::Diagonal::Sigmas(sigams_R);
  }

  void set_prior(estimation::RodColors color, const SE3 prior, NoiseModel& noise_model)
  {
    _bars[color] = prior;
    _priors[color] = noise_model;
    _valid[color] = true;
  }

  void predict(const Velocity red_vel, const Velocity green_vel, const Velocity blue_vel, const double dt)
  {
    // Create a Gaussian Factor Graph
    gtsam::Values values;
    gtsam::NonlinearFactorGraph graph;
    gtsam::KeyVector keys;

    predict_bar(RodColors::RED, red_vel, dt, graph, values);
    predict_bar(RodColors::GREEN, green_vel, dt, graph, values);
    predict_bar(RodColors::BLUE, blue_vel, dt, graph, values);

    if (graph.size() > 0)
    {
      PRINT_MSG("Solving Predict")
      // PRINT_KEYS(k_Xcurrent[RodColors::RED], _priors[RodColors::RED]->front());
      solve(graph, values);
      // _idx++;
    }
  }

  void add_endcap_observations(const RodColors color, const RodObservation& observations,
                               gtsam::NonlinearFactorGraph& graph, Values& values)
  {
    gtsam::noiseModel::Base::shared_ptr z_noise{ gtsam::noiseModel::Isotropic::Sigma(3, 1e0) };
    gtsam::noiseModel::Base::shared_ptr rot_prior_nm{ gtsam::noiseModel::Isotropic::Sigma(3, 1e-4) };
    const gtsam::Key key_se3{ k_Xcurrent[color] };
    const gtsam::Key key_rotA_offset{ estimation::rotation_symbol(color, _idx, 0) };
    const gtsam::Key key_rotB_offset{ estimation::rotation_symbol(color, _idx, 1) };

    if (_valid[color])
    {
      graph.addPrior(key_se3, _bars[color]);
      // graph.addPrior(key_se3, _bars[color], _priors[color]);
      estimation::add_to_values(values, key_se3, _bars[color]);
    }

    if (observations.first and observations.second)  // Two observations
    {
      PRINT_MSG("Using TWO observations");

      _observation_update.color = color;
      _observation_update.zA = *(observations.first);
      _observation_update.zB = *(observations.second);
      // const Translation zA{ *(observations.first) };
      // const Translation zB{ *(observations.second) };

      // graph.addPrior(key_se3, _bars[color], _priors[color]);

      estimation::add_two_observations(_observation_update, graph, values);

      estimation::add_to_values(values, key_se3, _bars[color]);
      estimation::add_to_values(values, key_rotA_offset, Rotation());
      estimation::add_to_values(values, key_rotB_offset, _Roffset.inverse());

      // graph.emplace_shared<EndcapObservationFactor>(key_se3, key_rotA_offset, zA, _offset, z_noise);
      // graph.emplace_shared<EndcapObservationFactor>(key_se3, key_rotB_offset, zB, _offset, z_noise);

      // graph.emplace_shared<RotationOffsetFactor>(key_rotA_offset, key_rotB_offset, _Roffset, rot_prior_nm);
      // graph.emplace_shared<RotationOffsetFactor>(key_rotB_offset, key_rotA_offset, _Roffset, rot_prior_nm);

      // graph.emplace_shared<RotationFixIdentity>(key_rotA_offset, key_rotB_offset, _Roffset, rot_prior_nm);

      // values.insert(key_se3, _bars[color]);
      // values.insert(key_rotA_offset, Rotation());
      // values.insert(key_rotB_offset, _Roffset.inverse());
      _valid[color] = true;
    }
    else if (observations.first or observations.second)  // One observations
    {
      if (_valid[color])
      {
        PRINT_MSG("Using ONE observation");

        // const Translation z{ observations.first ? *(observations.first) : *(observations.second) };

        // graph.addPrior(key_se3, _bars[color]);
        _observation_update.color = color;
        _observation_update.zA = observations.first ? *(observations.first) : *(observations.second);
        estimation::add_one_observation(_observation_update, graph, values);

        // graph.emplace_shared<EndcapObservationFactor>(key_se3, key_rotA_offset, z, _offset, z_noise);
        //
        // graph.emplace_shared<RotationOffsetFactor>(key_rotA_offset, key_rotB_offset, _Roffset, rot_prior_nm);
        // graph.emplace_shared<RotationOffsetFactor>(key_rotB_offset, key_rotA_offset, _Roffset, rot_prior_nm);
        //
        // graph.emplace_shared<RotationFixIdentity>(key_rotA_offset, key_rotB_offset, _Roffset, rot_prior_nm);

        estimation::add_to_values(values, key_se3, _bars[color]);
        estimation::add_to_values(values, key_rotA_offset, Rotation());
        estimation::add_to_values(values, key_rotB_offset, _Roffset.inverse());

        // values.insert(key_se3, _bars[color]);
        // values.insert(key_rotA_offset, Rotation());
        // values.insert(key_rotB_offset, _Roffset.inverse());
      }
    }
  }

  void add_cable_meassurements(const std::vector<double>& zi, gtsam::NonlinearFactorGraph& graph, Values& values)
  {
    // using CablesFactor = estimation::cable_length_observations_factor_t;
    using CablesFactor = estimation::cable_length_no_rotation_factor_t;

    if (not _use_cables)
      return;

    if (zi.size() != 9)
      return;

    if (not _valid[RodColors::RED] or not _valid[RodColors::GREEN] or not _valid[RodColors::BLUE])
      return;

    gtsam::noiseModel::Base::shared_ptr z_noise{ gtsam::noiseModel::Isotropic::Sigma(1, 1.6e-3) };
    gtsam::noiseModel::Base::shared_ptr rot_prior_nm{ gtsam::noiseModel::Isotropic::Sigma(3, 1e-5) };

    for (int i = 0; i < 9; ++i)
    {
      const gtsam::Key key_Xi{ rod_symbol(_cable_map[i].first, _idx) };
      const gtsam::Key key_Xj{ rod_symbol(_cable_map[i].second, _idx) };
      const gtsam::Key key_Ri{ rotation_symbol(_cable_map[i].first, _idx, 2 + i) };
      const gtsam::Key key_Rj{ rotation_symbol(_cable_map[i].second, _idx, 2 + i) };

      // PRINT_KEYS(key_Xi, key_Xj);
      // values.insert(key_Ri, Rotation());
      // values.insert(key_Rj, Rotation());
      estimation::add_to_values(values, key_Ri, Rotation());
      estimation::add_to_values(values, key_Rj, Rotation());
      estimation::add_to_values(values, k_Xcurrent[_cable_map[i].first], _bars[_cable_map[i].first]);
      estimation::add_to_values(values, k_Xcurrent[_cable_map[i].second], _bars[_cable_map[i].second]);

      graph.emplace_shared<CablesFactor>(key_Xi, key_Xj, key_Ri, zi[i], _offset, z_noise);

      graph.emplace_shared<RotationOffsetFactor>(key_Ri, key_Rj, _Roffset, rot_prior_nm);
      graph.emplace_shared<RotationOffsetFactor>(key_Rj, key_Ri, _Roffset, rot_prior_nm);

      graph.emplace_shared<RotationFixIdentity>(key_Ri, key_Rj, _Roffset, rot_prior_nm);
    }
  }

  /**
   * Calculate posterior density P(x_) ~ L(z|x) P(x)
   * The likelihood L(z|x) should be given as a unary factor on x
   */
  void update(const RodObservation& z_red, const RodObservation& z_green, const RodObservation& z_blue,
              const std::vector<double>& cable_array)
  {
    // const auto keys = measurementFactor.keys();

    // Create a Gaussian Factor Graph
    // GaussianFG linearFactorGraph;
    gtsam::NonlinearFactorGraph graph;

    // Linearize measurement factor and add it to the Kalman Filter graph
    Values values;

    // gtsam::KeyVector keys;

    k_Xcurrent[RodColors::RED] = rod_symbol(RodColors::RED, _idx);
    k_Xcurrent[RodColors::GREEN] = rod_symbol(RodColors::GREEN, _idx);
    k_Xcurrent[RodColors::BLUE] = rod_symbol(RodColors::BLUE, _idx);

    _observation_update.idx = _idx;

    add_endcap_observations(RodColors::RED, z_red, graph, values);
    add_endcap_observations(RodColors::GREEN, z_green, graph, values);
    add_endcap_observations(RodColors::BLUE, z_blue, graph, values);
    add_cable_meassurements(cable_array, graph, values);

    if (graph.size())
    {
      PRINT_MSG("Solving Update")
      solve(graph, values);
    }
  }

  OptSE3 get_estimation(const RodColors color)
  {
    if (_valid[color])
    {
      return _bars[color];
    }
    return {};
  }

private:
  void predict_bar(const RodColors& color, const Velocity& vel, const double dt, gtsam::NonlinearFactorGraph& graph,
                   Values& values)
  {
    if (not _valid[color])
    {
      return;
    }
    const gtsam::Key k_Xnext{ rod_symbol(color, _idx + 1) };
    // const gtsam::Key k_Xnext{ rod_symbol(color, _idx + 1) };

    // GaussianFG linearFactorGraph;

    // Add in previous posterior as prior on the first state
    // graph.addPrior(k_Xcurrent[color], _bars[color], _priors[color]);
    graph.addPrior(k_Xcurrent[color], _bars[color]);

    values.insert(k_Xcurrent[color], _bars[color]);

    values.insert(k_Xnext, _bars[color]);

    graph.emplace_shared<lie_motion_model_factor_t>(k_Xnext, k_Xcurrent[color], _xdot_noise, vel, dt);
    // lie_motion_model_factor_t red_motion(k_Xnext, k_Xcurrent[color], k_Rcurrent[color], _xdot_noise, vel, dt);

    // gtsam::SharedDiagonal nm{ gtsam::noiseModel::Unit::Create(3) };
    // const Eigen::MatrixXd initial_R{ nm->R() };
    // const Eigen::Vector3d intial_b{ Eigen::Vector3d::Zero() };
    // JacobianFactorPtr Rprior{ JacobianFactorPtr(new JacobianFactor(k_Rcurrent[color], initial_R, intial_b, nm)) };
    // linearFactorGraph.push_back(Rprior);

    // const Eigen::MatrixXd initial_R{ _rot_noise_model->R() };
    // const Eigen::Vector3d intial_b{ Eigen::Vector3d::Zero() };
    // JacobianFactorPtr Rzprior{ JacobianFactorPtr(
    //     new JacobianFactor(k_Rcurrent[RodColors::RED], initial_R, intial_b, _rot_noise_model)) };
    // linearFactorGraph.push_back(Rzprior);

    // linearFactorGraph.push_back(red_motion.linearize(linearizationPoint));

    // keys.push_back(k_Rcurrent[color]);
    // keys.push_back(k_Xnext);

    k_Xcurrent[color] = k_Xnext;
  }

  void solve(gtsam::NonlinearFactorGraph& graph, Values& values)
  {
    // if (not _valid[RodColors::RED])
    //   return;
    try
    {
      const Values result{ _lm_helper->optimize(graph, values, true) };

      // graph.printErrors(result, "Errors", SF::formatter);
      update_pose(RodColors::RED, result);
      update_pose(RodColors::GREEN, result);
      update_pose(RodColors::BLUE, result);

      update_prior(RodColors::RED, graph, result);
      update_prior(RodColors::GREEN, graph, result);
      update_prior(RodColors::BLUE, graph, result);
    }
    catch (gtsam::ValuesKeyDoesNotExist e)
    {
      PRINT_MSG("[EKF] Optimization failed!")
      PRINT_MSG(e.what());
      PRINT_MSG("[gtsam::ValuesKeyDoesNotExist] Variable: " + SF::formatter(e.key()) + "\n");
      exit(-1);
    }
    catch (gtsam::IndeterminantLinearSystemException e)
    {
      PRINT_MSG("[EKF] Optimization failed!")
      PRINT_MSG(e.what());
      PRINT_MSG("[gtsam::IndeterminantLinearSystemException] Variable: " + SF::formatter(e.nearbyVariable()) + "\n");

      // factor_graphs::indeterminant_linear_system_helper(&graph);
      exit(-1);
    }
    // update_prior(_green_prior, marginal_green, result_green, key_green);
    // update_prior(_blue_prior, marginal_blue, result_blue, key_blue);
    // _red_prior = boost::make_shared<JacobianFactor>(                                             // no-lint
    //     marginal_red->keys().front(),                                                            // no-lint
    //     marginal_red->getA(marginal_red->begin()),                                               // no-lint
    //     marginal_red->getb() - marginal_red->getA(marginal_red->begin()) * result_red[key_red],  // no-lint
    //     marginal_red->get_model());
  }

  void update_pose(const RodColors color, const Values& result)
  {
    if (result.exists(k_Xcurrent[color]))
    {
      _bars[color] = result.at<SE3>(k_Xcurrent[color]);

      // const gtsam::Rot3 rot_current{ linearizationPoint.at<gtsam::Rot3>(k_Rcurrent[color]) };
      // _rotations[color] = gtsam::traits<gtsam::Rot3>::Retract(rot_current, result[k_Rcurrent[color]]);

      // const SE3 x_current{ linearizationPoint.at<SE3>(k_Xcurrent[color]) };
      // _bars[color] = gtsam::traits<SE3>::Retract(x_current, result[k_Xcurrent[color]]);
    }
  }

  void update_prior(const RodColors color, const gtsam::NonlinearFactorGraph& graph, const gtsam::Values& result)
  {
    // marginal->print("marginal", SF::formatter);
    if (result.exists(k_Xcurrent[color]))
    {
      gtsam::Marginals marginals(graph, result);
      auto info_mat = marginals.marginalInformation(k_Xcurrent[color]);
      DEBUG_VARS(info_mat);
      _priors[color] = gtsam::noiseModel::Gaussian::Information(info_mat);
      // const gtsam::Key pk{ rod_symbol(color, _idx) };
      // // PriorFactor(Key key, const VALUE& prior, const SharedNoiseModel& model = nullptr) :
      // gtsam::Values v({ { pk, gtsam::genericValue(_bars[color]) } });
      // gtsam::PriorFactor<SE3> prior(pk, _bars[color]);

      // auto gaussian_factor = prior.linearize(v);
      // _priors[color] = boost::make_shared<JacobianFactor>(*gaussian_factor);

      // // GaussianCondPtr prior_marginal;
      // for (auto marginal : *bayes_net)
      // {
      //   if (marginal->keys().size() == 1 and marginal->keys()[0] == k_Xcurrent[color])
      //   {
      // PRINT_KEYS(k_Xcurrent[color]);
      // // marginal->print("marginal ", SF::formatter);
      // // prior_marginal = marginal;
      // _priors[color] = boost::make_shared<JacobianFactor>(                                   // no-lint
      //     rod_symbol(color, _idx),                                                           // no-lint
      //     marginal->getA(marginal->begin()),                                                 // no-lint
      //     marginal->getb() - marginal->getA(marginal->begin()) * result[k_Xcurrent[color]],  // no-lint
      //     marginal->get_model());
      // _priors[color]->print("prior", SF::formatter);
      // return;
      //   }
      // }
    }
  }

  estimation::observation_update_t _observation_update;

  int _idx;
  std::array<gtsam::Key, 3> k_Xcurrent;
  std::array<gtsam::Key, 3> k_Rcurrent;

  std::array<bool, 3> _valid;
  // gtsam::Key k_Xred;
  // gtsam::Key k_Xgreen;
  // gtsam::Key k_Xblue;

  // gtsam::Key k_Rred;
  // gtsam::Key k_Rgreen;
  // gtsam::Key k_Rblue;

  std::array<SE3, 3> _bars;
  std::array<gtsam::Rot3, 3> _rotations;
  // SE3 _green_bar;
  // SE3 _blue_bar;

  // T x_;                                     // linearization point
  // std::array<JacobianFactorPtr, 3> _priors;  // Gaussian density on x_
  std::array<NoiseModel, 3> _priors;  // Gaussian density on x_
  // JacobianFactorPtr _green_prior;           // Gaussian density on x_
  // JacobianFactorPtr _blue_prior;            // Gaussian density on x_
  JacobianFactorPtr _Rprior;  // Prior on Rz;

  gtsam::noiseModel::Base::shared_ptr _xdot_noise;

  const Translation _offset;
  const Rotation _Roffset;

  gtsam::SharedDiagonal _rot_noise_model;

  bool _use_cables;

  estimation::ColorMapping _cable_map;

  gtsam::LevenbergMarquardtParams _lm_params;
  std::shared_ptr<factor_graphs::levenberg_marquardt_t> _lm_helper;
};

}  // namespace estimation