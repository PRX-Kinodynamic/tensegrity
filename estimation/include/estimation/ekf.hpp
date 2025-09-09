#pragma once

#include <factor_graphs/defs.hpp>

#include <interface/TensegrityLengthSensor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/linear/GaussianConditional.h>
#include <gtsam/linear/GaussianBayesNet.h>
#include <gtsam/linear/GaussianFactorGraph.h>

#include <estimation/bar_utilities.hpp>

namespace estimation
{
class lie_motion_model_factor_t : public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Pose3, gtsam::Rot3>
{
  using SE3 = gtsam::Pose3;
  using Rotation = gtsam::Rot3;
  using Base = gtsam::NoiseModelFactorN<SE3, SE3, Rotation>;

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
  lie_motion_model_factor_t(const gtsam::Key key_xt1, const gtsam::Key key_xt0, const gtsam::Key key_rot,
                            const NoiseModel& cost_model, const Xdot xdot, const double dt)
    : Base(cost_model, key_xt1, key_xt0, key_rot), _xdot(xdot), _dt(dt)
  {
  }

  ~lie_motion_model_factor_t() override
  {
  }

  static SE3 rotate(const SE3& x0, const Rotation& rot, OptDeriv H0 = boost::none, OptDeriv HR = boost::none)
  {
    const bool compute_deriv{ H0 or HR };
    Eigen::Matrix<double, 6, 6> x0p_H_x0;
    Eigen::Matrix<double, 6, 6> x0p_H_xR;
    Eigen::Matrix<double, 6, 3> xR_H_rot;

    const Eigen::Vector3d zero{ Eigen::Vector3d::Zero() };
    const SE3 xR{ SE3::Create(rot, zero, xR_H_rot) };
    const SE3 x0p{ gtsam::traits<SE3>::Compose(x0, xR,  // no-lint
                                               compute_deriv ? &x0p_H_x0 : nullptr,
                                               compute_deriv ? &x0p_H_xR : nullptr) };

    if (H0)
    {
      *H0 = x0p_H_x0;
    }
    if (HR)
    {
      *HR = x0p_H_xR * xR_H_rot;
    }
    return x0p;
  }

  static SE3 predict(const SE3& x0, const Rotation& rot, const Xdot& vel, const double dt)
  {
    const SE3 x0p{ rotate(x0, rot) };

    return LieIntegrator::predict(x0p, vel, dt);
  }

  virtual Eigen::VectorXd evaluateError(const SE3& x1, const SE3& x0, const Rotation& rot,  // no-lint
                                        OptDeriv H1 = boost::none, OptDeriv H0 = boost::none,
                                        OptDeriv HR = boost::none) const override
  {
    const bool compute_deriv{ H1 or H0 or HR };
    Eigen::MatrixXd err_H_x1p;
    Eigen::MatrixXd err_H_x0;

    Eigen::MatrixXd x1p_H_x1;
    Eigen::MatrixXd x1p_H_rinv;
    Eigen::MatrixXd rinv_H_rot;

    const Rotation rot_inv{ rot.inverse(rinv_H_rot) };
    const SE3 x1p{ rotate(x1, rot_inv, x1p_H_x1, x1p_H_rinv) };

    const Eigen::VectorXd error{ LieIntegrator::error(x1p, x0, _xdot, _dt,  // no-lint
                                                      err_H_x1p, err_H_x0) };

    // DEBUG_VARS(x0);
    // DEBUG_VARS(x1);
    // DEBUG_VARS(rot);
    // DEBUG_VARS(x1p);
    // DEBUG_VARS(_dt, _xdot.transpose());
    // DEBUG_VARS(error.transpose());
    if (H1)
    {
      *H1 = err_H_x1p * x1p_H_x1;
    }
    if (H0)
    {
      *H0 = err_H_x0;
    }
    if (HR)
    {
      *HR = err_H_x1p * x1p_H_rinv * rinv_H_rot;
    }

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

  using SF = factor_graphs::symbol_factory_t;

public:
  /// @name Standard Constructors
  /// @{

  // ExtendedKalmanFilter(Key key_initial, T x_initial, NoiseModel P_initial)
  extended_kalman_filter_t(const GaussianNoiseModel P_initial, bool use_cables)
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
    const int n{ 6 };
    // const gtsam::noiseModel initial_nm{ P_initial != nullptr ? P_initial : gtsam::noiseModel::Unit::Create(6) };
    const auto initial_nm = gtsam::noiseModel::Unit::Create(6);
    const Eigen::Vector<double, 6> intial_b{ Eigen::Vector<double, 6>::Zero() };
    const Eigen::MatrixXd initial_R{ initial_nm->R() };
    _priors[RodColors::RED] =
        JacobianFactorPtr(new JacobianFactor(k_Xcurrent[RodColors::RED], initial_R, intial_b, initial_nm));
    _priors[RodColors::GREEN] =
        JacobianFactorPtr(new JacobianFactor(k_Xcurrent[RodColors::GREEN], initial_R, intial_b, initial_nm));
    _priors[RodColors::BLUE] =
        JacobianFactorPtr(new JacobianFactor(k_Xcurrent[RodColors::BLUE], initial_R, intial_b, initial_nm));

    const Eigen::Vector3d sigams_R({ 0, 0, 1 });
    _rot_noise_model = gtsam::noiseModel::Diagonal::Sigmas(sigams_R);
  }

  /**
   * Calculate predictive density
   *     \f$ P(x_) ~ \int  P(x_min) P(x_min, x_)\f$
   * The motion model should be given as a factor with key1 for \f$x_min\f$ and key2 for \f$x_\f$
   */
  // void predict(const NoiseModelFactor& motionFactor)
  // const gtsam::Key k_Xred_next{ rod_symbol(RodColors::RED, _idx) };
  // const gtsam::Key k_Xgreen_next{ rod_symbol(RodColors::GREEN, _idx) };
  // const gtsam::Key k_Xblue_next{ rod_symbol(RodColors::BLUE, _idx) };
  void predict(const Velocity red_vel, const Velocity green_vel, const Velocity blue_vel, const double dt)
  {
    // Create a Gaussian Factor Graph
    Values linearizationPoint;
    GaussianFG linearFactorGraph;
    gtsam::KeyVector keys;

    if (_valid[RodColors::RED])
    {
      predict_bar(RodColors::RED, red_vel, dt, linearFactorGraph, linearizationPoint, keys);
    }
    if (_valid[RodColors::GREEN])
    {
      predict_bar(RodColors::GREEN, green_vel, dt, linearFactorGraph, linearizationPoint, keys);
    }
    if (_valid[RodColors::BLUE])
    {
      predict_bar(RodColors::BLUE, blue_vel, dt, linearFactorGraph, linearizationPoint, keys);
    }
    // // Add in previous posterior as prior on the first state
    // linearFactorGraph.push_back(_red_prior);
    // // linearFactorGraph.push_back(_green_prior);
    // // linearFactorGraph.push_back(_blue_prior);

    // // Linearize motion model and add it to the Kalman Filter graph
    // linearizationPoint.insert(k_Xred, _red_bar);
    // // linearizationPoint.insert(k_Xgreen, _green_bar);
    // // linearizationPoint.insert(k_Xblue, _blue_bar);

    // linearizationPoint.insert(k_Xred_next, _red_bar);
    // // linearizationPoint.insert(k_Xgreen_next, _green_bar);
    // // linearizationPoint.insert(k_Xblue_next, _blue_bar);

    // gtsam::noiseModel::Base::shared_ptr xd_noise{ gtsam::noiseModel::Isotropic::Sigma(3, 1e0) };

    // lie_motion_model_factor_t red_motion(k_Xred_next, k_Xred, xd_noise, red_vel, dt);
    // // lie_motion_model_factor_t green_motion(k_Xgreen_next, k_Xgreen, xd_noise, green_vel, dt);
    // // lie_motion_model_factor_t blue_motion(k_Xblue_next, k_Xblue, xd_noise, blue_vel, dt);

    // // lie_motion_model_factor_t(const gtsam::Key key_xt1, const gtsam::Key key_xt0, const NoiseModel& cost_model,
    // //                           const Xdot xdot, const double dt)

    // linearFactorGraph.push_back(red_motion.linearize(linearizationPoint));
    // // linearFactorGraph.push_back(green_motion.linearize(linearizationPoint));
    // // linearFactorGraph.push_back(blue_motion.linearize(linearizationPoint));

    // Solve the factor graph and update the current state estimate
    // and the posterior for the next iteration.
    if (linearFactorGraph.size() > 0)
    {
      PRINT_MSG("Solving Predict")
      // PRINT_KEYS(k_Xcurrent[RodColors::RED], _priors[RodColors::RED]->front());
      solve(linearFactorGraph, linearizationPoint, keys);
      // _idx++;
    }
    // k_Xred = k_Xred_next;
    // k_Xgreen = k_Xgreen_next;
    // k_Xblue = k_Xblue_next;
  }

  void add_endcap_observations(const RodColors color, const RodObservation& observations, GaussianFG& linearFactorGraph,
                               Values& linearizationPoint, gtsam::KeyVector& keys)
  {
    using BarEstimationFactor = estimation::bar_two_observations_factor_t;
    using EndcapObservationFactor = estimation::endcap_observation_factor_t;
    using RotationOffsetFactor = estimation::endcap_rotation_offset_factor_t;
    using RotationFixIdentity = estimation::rotation_fix_identity_t;

    gtsam::noiseModel::Base::shared_ptr z_noise{ gtsam::noiseModel::Isotropic::Sigma(3, 1e0) };
    gtsam::noiseModel::Base::shared_ptr rot_prior_nm{ gtsam::noiseModel::Isotropic::Sigma(3, 1e-4) };
    const gtsam::Key key_se3{ k_Xcurrent[color] };
    const gtsam::Key key_rotA_offset{ estimation::rotation_symbol(color, _idx, 0) };
    const gtsam::Key key_rotB_offset{ estimation::rotation_symbol(color, _idx, 1) };

    if (observations.first and observations.second)  // Two observations
    {
      PRINT_MSG("Using TWO observations");

      const Translation zA{ *(observations.first) };
      const Translation zB{ *(observations.second) };

      // const std::string color_{ color_str(color) };
      // DEBUG_VARS(color_, zA.transpose(), zB.transpose())

      linearizationPoint.insert(key_se3, _bars[color]);
      linearizationPoint.insert(key_rotA_offset, Rotation());
      linearizationPoint.insert(key_rotB_offset, _Roffset.inverse());

      const Eigen::MatrixXd initial_R{ _rot_noise_model->R() };
      const Eigen::Vector3d intial_b{ Eigen::Vector3d::Zero() };
      const JacobianFactorPtr Rprior{ JacobianFactorPtr(
          new JacobianFactor(key_rotA_offset, initial_R, intial_b, _rot_noise_model)) };

      EndcapObservationFactor eof0(key_se3, key_rotA_offset, zA, _offset, z_noise);
      EndcapObservationFactor eof1(key_se3, key_rotB_offset, zB, _offset, z_noise);

      RotationOffsetFactor rof0(key_rotA_offset, key_rotB_offset, _Roffset, rot_prior_nm);
      RotationOffsetFactor rof1(key_rotB_offset, key_rotA_offset, _Roffset, rot_prior_nm);

      RotationFixIdentity rfi(key_rotA_offset, key_rotB_offset, _Roffset, rot_prior_nm);

      // BarTwoEstimationFactor z_factor_A(k_Xcurrent[color], k_Rcurrent[color], zA, _offset, _Roffset,
      //                                   BarTwoEstimationFactor::ObservationIdx::First, z_noise);
      // BarTwoEstimationFactor z_factor_B(k_Xcurrent[color], k_Rcurrent[color], zB, _offset, _Roffset,
      //                                   BarTwoEstimationFactor::ObservationIdx::Second, z_noise);

      linearFactorGraph.push_back(_priors[color]);
      linearFactorGraph.push_back(Rprior);
      linearFactorGraph.push_back(eof0.linearize(linearizationPoint));
      linearFactorGraph.push_back(eof1.linearize(linearizationPoint));
      linearFactorGraph.push_back(rof0.linearize(linearizationPoint));
      linearFactorGraph.push_back(rof1.linearize(linearizationPoint));
      linearFactorGraph.push_back(rfi.linearize(linearizationPoint));

      keys.push_back(key_se3);
      keys.push_back(key_rotA_offset);
      keys.push_back(key_rotB_offset);

      _valid[color] = true;
    }
    // else if (observations.first or observations.second)  // One observations
    // {
    //   if (_valid[color])
    //   {
    //     PRINT_MSG("Using ONE observation");

    //     const Translation z{ observations.first ? *(observations.first) : *(observations.second) };

    //     linearizationPoint.insert(k_Xcurrent[color], _bars[color]);
    //     linearizationPoint.insert(k_Rcurrent[color], Rotation());

    //     BarTwoEstimationFactor red_factor_A(k_Xcurrent[color], k_Rcurrent[color], z, _offset, _Roffset,
    //                                         BarTwoEstimationFactor::ObservationIdx::First, z_noise);

    //     const Eigen::MatrixXd initial_R{ _rot_noise_model->R() };
    //     const Eigen::Vector3d intial_b{ Eigen::Vector3d::Zero() };
    //     _Rprior = JacobianFactorPtr(new JacobianFactor(k_Rcurrent[color], initial_R, intial_b, _rot_noise_model));

    //     linearFactorGraph.push_back(_priors[color]);
    //     linearFactorGraph.push_back(_Rprior);
    //     linearFactorGraph.push_back(red_factor_A.linearize(linearizationPoint));

    //     keys.push_back(k_Rcurrent[color]);
    //     keys.push_back(k_Xcurrent[color]);
    //   }
    // }
  }

  void add_cable_meassurements(const std::vector<double>& zi, GaussianFG& linearFactorGraph, Values& linearizationPoint,
                               gtsam::KeyVector& keys)
  {
    using CablesFactor = estimation::cable_length_observations_factor_t;

    if (not _use_cables)
      return;

    if (zi.size() != 9)
      return;

    if (not _valid[RodColors::RED] or not _valid[RodColors::GREEN] or not _valid[RodColors::BLUE])
      return;

    if (not linearizationPoint.exists(k_Xcurrent[RodColors::RED]))
    {
      linearizationPoint.insert(k_Xcurrent[RodColors::RED], _bars[RodColors::RED]);
      linearizationPoint.insert(k_Rcurrent[RodColors::RED], Rotation());
      keys.push_back(k_Rcurrent[RodColors::RED]);
      keys.push_back(k_Xcurrent[RodColors::RED]);
      const Eigen::MatrixXd initial_R{ _rot_noise_model->R() };
      const Eigen::Vector3d intial_b{ Eigen::Vector3d::Zero() };
      const JacobianFactorPtr Rprior{ JacobianFactorPtr(
          new JacobianFactor(k_Rcurrent[RodColors::RED], initial_R, intial_b, _rot_noise_model)) };

      linearFactorGraph.push_back(_priors[RodColors::RED]);
      linearFactorGraph.push_back(Rprior);
    }
    if (not linearizationPoint.exists(k_Xcurrent[RodColors::GREEN]))
    {
      linearizationPoint.insert(k_Xcurrent[RodColors::GREEN], _bars[RodColors::GREEN]);
      linearizationPoint.insert(k_Rcurrent[RodColors::GREEN], Rotation());
      keys.push_back(k_Rcurrent[RodColors::GREEN]);
      keys.push_back(k_Xcurrent[RodColors::GREEN]);
      const Eigen::MatrixXd initial_R{ _rot_noise_model->R() };
      const Eigen::Vector3d intial_b{ Eigen::Vector3d::Zero() };
      const JacobianFactorPtr Rprior{ JacobianFactorPtr(
          new JacobianFactor(k_Rcurrent[RodColors::GREEN], initial_R, intial_b, _rot_noise_model)) };

      linearFactorGraph.push_back(_priors[RodColors::GREEN]);
      linearFactorGraph.push_back(Rprior);
    }
    if (not linearizationPoint.exists(k_Xcurrent[RodColors::BLUE]))
    {
      linearizationPoint.insert(k_Xcurrent[RodColors::BLUE], _bars[RodColors::BLUE]);
      linearizationPoint.insert(k_Rcurrent[RodColors::BLUE], Rotation());
      keys.push_back(k_Rcurrent[RodColors::BLUE]);
      keys.push_back(k_Xcurrent[RodColors::BLUE]);
      const Eigen::MatrixXd initial_R{ _rot_noise_model->R() };
      const Eigen::Vector3d intial_b{ Eigen::Vector3d::Zero() };
      const JacobianFactorPtr Rprior{ JacobianFactorPtr(
          new JacobianFactor(k_Rcurrent[RodColors::BLUE], initial_R, intial_b, _rot_noise_model)) };

      linearFactorGraph.push_back(_priors[RodColors::BLUE]);
      linearFactorGraph.push_back(Rprior);
    }

    gtsam::noiseModel::Base::shared_ptr z_noise{ gtsam::noiseModel::Isotropic::Sigma(1, 1) };

    const gtsam::Key key_Xred{ k_Xcurrent[RodColors::RED] };
    const gtsam::Key key_Xgreen{ k_Xcurrent[RodColors::GREEN] };
    const gtsam::Key key_Xblue{ k_Xcurrent[RodColors::BLUE] };
    const gtsam::Key key_Rred{ k_Rcurrent[RodColors::RED] };
    const gtsam::Key key_Rgreen{ k_Rcurrent[RodColors::GREEN] };
    const gtsam::Key key_Rblue{ k_Rcurrent[RodColors::BLUE] };

    CablesFactor c0(key_Xblue, key_Rblue, key_Xgreen, key_Rgreen, zi[0],  // no-lint
                    _offset, _Roffset, CablesFactor::CableDistanceId::BB, z_noise);
    CablesFactor c1(key_Xgreen, key_Rgreen, key_Xred, key_Rred, zi[1],  // no-lint
                    _offset, _Roffset, CablesFactor::CableDistanceId::BB, z_noise);
    CablesFactor c2(key_Xred, key_Rred, key_Xblue, key_Rblue, zi[2],  // no-lint
                    _offset, _Roffset, CablesFactor::CableDistanceId::BB, z_noise);

    CablesFactor c3(key_Xgreen, key_Rgreen, key_Xred, key_Rred, zi[3],  // no-lint
                    _offset, _Roffset, CablesFactor::CableDistanceId::AA, z_noise);
    CablesFactor c4(key_Xred, key_Rred, key_Xblue, key_Rblue, zi[4],  // no-lint
                    _offset, _Roffset, CablesFactor::CableDistanceId::AA, z_noise);
    CablesFactor c5(key_Xblue, key_Rblue, key_Xgreen, key_Rgreen, zi[5],  // no-lint
                    _offset, _Roffset, CablesFactor::CableDistanceId::AA, z_noise);

    CablesFactor c6(key_Xgreen, key_Rgreen, key_Xblue, key_Rblue, zi[6],  // no-lint
                    _offset, _Roffset, CablesFactor::CableDistanceId::AB, z_noise);
    CablesFactor c7(key_Xred, key_Rred, key_Xgreen, key_Rgreen, zi[7],  // no-lint
                    _offset, _Roffset, CablesFactor::CableDistanceId::AB, z_noise);
    CablesFactor c8(key_Xblue, key_Rblue, key_Xred, key_Rred, zi[8],  // no-lint
                    _offset, _Roffset, CablesFactor::CableDistanceId::AB, z_noise);

    linearFactorGraph.push_back(c0.linearize(linearizationPoint));
    linearFactorGraph.push_back(c1.linearize(linearizationPoint));
    linearFactorGraph.push_back(c2.linearize(linearizationPoint));
    linearFactorGraph.push_back(c3.linearize(linearizationPoint));
    linearFactorGraph.push_back(c4.linearize(linearizationPoint));
    linearFactorGraph.push_back(c5.linearize(linearizationPoint));
    linearFactorGraph.push_back(c6.linearize(linearizationPoint));
    linearFactorGraph.push_back(c7.linearize(linearizationPoint));
    linearFactorGraph.push_back(c8.linearize(linearizationPoint));
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
    GaussianFG linearFactorGraph;

    // Linearize measurement factor and add it to the Kalman Filter graph
    Values linearizationPoint;

    gtsam::KeyVector keys;

    k_Xcurrent[RodColors::RED] = rod_symbol(RodColors::RED, _idx);
    k_Xcurrent[RodColors::GREEN] = rod_symbol(RodColors::GREEN, _idx);
    k_Xcurrent[RodColors::BLUE] = rod_symbol(RodColors::BLUE, _idx);

    add_endcap_observations(RodColors::RED, z_red, linearFactorGraph, linearizationPoint, keys);
    add_endcap_observations(RodColors::GREEN, z_green, linearFactorGraph, linearizationPoint, keys);
    add_endcap_observations(RodColors::BLUE, z_blue, linearFactorGraph, linearizationPoint, keys);
    add_cable_meassurements(cable_array, linearFactorGraph, linearizationPoint, keys);

    if (linearFactorGraph.size())
    {
      PRINT_MSG("Solving Update")
      // PRINT_KEYS(k_Xcurrent[RodColors::RED], _priors[RodColors::RED]->front());
      solve(linearFactorGraph, linearizationPoint, keys);
      // _idx++;
    }

    // return x_;
  }

  OptSE3 get_estimation(const RodColors color)
  {
    if (_valid[color])
    {
      // const SE3 xt{ values.at<SE3>(key_se3) };
      // const Rotation Rt{ values.at<Rotation>(key_rot_offset) };
      // const Translation pt_zero{ Translation::Zero() };
      // const SE3 xr{ SE3(_rotations[RodColors::RED], pt_zero) };
      // pose = xt * xr;
      return _bars[color];
    }
    return {};
  }

private:
  void predict_bar(const RodColors& color, const Velocity& vel, const double dt, GaussianFG& linearFactorGraph,
                   Values& linearizationPoint, gtsam::KeyVector& keys)
  {
    const gtsam::Key k_Xnext{ rod_symbol(color, _idx + 1) };
    // const gtsam::Key k_Xnext{ rod_symbol(color, _idx + 1) };

    // GaussianFG linearFactorGraph;

    // Add in previous posterior as prior on the first state
    linearFactorGraph.push_back(_priors[color]);

    // Linearize motion model and add it to the Kalman Filter graph

    linearizationPoint.insert(k_Xcurrent[color], _bars[color]);

    linearizationPoint.insert(k_Xnext, lie_motion_model_factor_t::predict(_bars[color], _rotations[color], vel, dt));

    linearizationPoint.insert(k_Rcurrent[color], _rotations[color]);

    lie_motion_model_factor_t red_motion(k_Xnext, k_Xcurrent[color], k_Rcurrent[color], _xdot_noise, vel, dt);

    gtsam::SharedDiagonal nm{ gtsam::noiseModel::Unit::Create(3) };
    const Eigen::MatrixXd initial_R{ nm->R() };
    const Eigen::Vector3d intial_b{ Eigen::Vector3d::Zero() };
    JacobianFactorPtr Rprior{ JacobianFactorPtr(new JacobianFactor(k_Rcurrent[color], initial_R, intial_b, nm)) };
    linearFactorGraph.push_back(Rprior);

    // const Eigen::MatrixXd initial_R{ _rot_noise_model->R() };
    // const Eigen::Vector3d intial_b{ Eigen::Vector3d::Zero() };
    // JacobianFactorPtr Rzprior{ JacobianFactorPtr(
    //     new JacobianFactor(k_Rcurrent[RodColors::RED], initial_R, intial_b, _rot_noise_model)) };
    // linearFactorGraph.push_back(Rzprior);

    linearFactorGraph.push_back(red_motion.linearize(linearizationPoint));

    keys.push_back(k_Rcurrent[color]);
    keys.push_back(k_Xnext);

    k_Xcurrent[color] = k_Xnext;
  }

  void solve(const GaussianFG& linear_factor_graph, const Values& linearizationPoint, gtsam::KeyVector& keys)
  {
    if (not _valid[RodColors::RED])
      return;
    // Compute the marginal on the last key
    // Solve the linear factor graph, converting it into a linear Bayes Network
    // P(x0,x1) = P(x0|x1)*P(x1)

    try
    {
      // gtsam::Ordering red_ordering{ k_Xcurrent[RodColors::RED] };
      // gtsam::Ordering rot_ordering{ k_Rcurrent[RodColors::RED] };
      // gtsam::Ordering red_ordering{ k_Rcurrent[RodColors::RED], k_Xcurrent[RodColors::RED] };
      gtsam::Ordering red_ordering;
      red_ordering += keys;

      // red_ordering.print("Ordering ", SF::formatter);

      // gtsam::KeyVector green_keys({ k_Rcurrent[RodColors::GREEN], k_Xcurrent[RodColors::GREEN] });
      // gtsam::Ordering red_ordering{ k_Xcurrent[RodColors::RED], k_Rcurrent[RodColors::RED] };
      // PRINT_KEYS(k_Rcurrent[RodColors::RED], k_Xcurrent[RodColors::RED]);
      // PRINT_KEYS(_priors[RodColors::RED]->front());
      // linear_factor_graph.print("linear graph", SF::formatter);
      // gtsam::Ordering green_ordering{ key_green };
      // gtsam::Ordering blue_ordering{ key_blue };
      // lastKeyAsOrdering += lastKey;

      const auto bayes_net = linear_factor_graph.marginalMultifrontalBayesNet(red_ordering);
      // bayes_net->print("bayes_net", SF::formatter);
      // const GaussianCondPtr marginal_red{ linear_factor_graph.marginalMultifrontalBayesNet(red_ordering)->front() };

      // const GaussianCondPtr marginal_red{ linear_factor_graph.marginalMultifrontalBayesNet(red_ordering) };
      // const GaussianCondPtr marginal_green{ linear_factor_graph.marginalMultifrontalBayesNet(green_ordering)->front()
      // }; const GaussianCondPtr marginal_blue{
      // linear_factor_graph.marginalMultifrontalBayesNet(blue_ordering)->front()

      const gtsam::VectorValues result{ bayes_net->optimize() };
      // result.print("result", SF::formatter);
      // DEBUG_VARS(bayes_net->error(result));
      // linear_factor_graph.printErrors(result, "linearFG", SF::formatter);
      // bayes_net->print("bayes_net", SF::formatter);

      // const gtsam::VectorValues result_rot{ bayes_net->back()->solve(gtsam::VectorValues()) };

      update_pose(RodColors::RED, linearizationPoint, result);
      update_pose(RodColors::GREEN, linearizationPoint, result);
      update_pose(RodColors::BLUE, linearizationPoint, result);

      // DEBUG_VARS(_rotations[RodColors::GREEN].toQuaternion());
      // DEBUG_VARS(_bars[RodColors::GREEN]);
      // const gtsam::Rot3 rot_current{ linearizationPoint.at<gtsam::Rot3>(k_Rcurrent[RodColors::RED]) };
      // _rotations[RodColors::RED] = gtsam::traits<gtsam::Rot3>::Retract(rot_current,
      // result[k_Rcurrent[RodColors::RED]]);

      // const SE3 x_current{ linearizationPoint.at<SE3>(k_Xcurrent[RodColors::RED]) };
      // _bars[RodColors::RED] = gtsam::traits<SE3>::Retract(x_current, result[k_Xcurrent[RodColors::RED]]);

      // exit(0);
      // Extract the current estimate of x1,P1
      // bayes_net->print("GaussianConditional", SF::formatter);
      // auto marginal_red = bayes_net->front();
      // const gtsam::VectorValues result_red{ marginal_red->solve(gtsam::VectorValues()) };
      // const VectorValues result_green{ marginal_green->solve(VectorValues()) };
      // const VectorValues result_blue{ marginal_blue->solve(VectorValues()) };

      // const SE3 current_red{ linearizationPoint.at<SE3>(k_Xcurrent[RodColors::RED]) };
      // const SE3& current_green = linearizationPoint.at<SE3>(key_green);
      // const SE3& current_blue = linearizationPoint.at<SE3>(key_blue);

      // update_value(RodColors::RED, linearizationPoint, result_red);
      // DEBUG_PRINT

      // _bars[RodColors::RED] = gtsam::traits<SE3>::Retract(current_red, result_red[k_Xcurrent[RodColors::RED]]);
      // _bars[RodColors::RED] = ;
      // _green_bar = gtsam::traits<SE3>::Retract(current_green, result_green[key_green]);
      // _blue_bar = gtsam::traits<SE3>::Retract(current_blue, result_blue[key_blue]);

      // Create a Jacobian Factor from the root node of the produced Bayes Net.
      // This will act as a prior for the next iteration.
      // The linearization point of this prior must be moved to the new estimate of x,
      // and the key/index needs to be reset to 0, the first key in the next iteration.
      // DEBUG_VARS(marginal_red->nrFrontals(), marginal_red->nrParents());
      // DEBUG_VARS(marginal_green->nrFrontals(), marginal_green->nrParents());
      // DEBUG_VARS(marginal_blue->nrFrontals(), marginal_blue->nrParents());
      // assert(marginal->nrFrontals() == 1);
      // assert(marginal->nrParents() == 0);

      // update_prior(_priors[RodColors::RED], marginal_red, result_red, k_Xcurrent[RodColors::RED]);

      update_prior(RodColors::RED, bayes_net, result);
      update_prior(RodColors::GREEN, bayes_net, result);
      update_prior(RodColors::BLUE, bayes_net, result);
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

      factor_graphs::indeterminant_linear_system_helper(&linear_factor_graph);
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

  void update_pose(const RodColors color, const Values& linearizationPoint, const gtsam::VectorValues& result)
  {
    if (result.exists(k_Rcurrent[color]) and result.exists(k_Xcurrent[color]))
    {
      const gtsam::Rot3 rot_current{ linearizationPoint.at<gtsam::Rot3>(k_Rcurrent[color]) };
      _rotations[color] = gtsam::traits<gtsam::Rot3>::Retract(rot_current, result[k_Rcurrent[color]]);

      const SE3 x_current{ linearizationPoint.at<SE3>(k_Xcurrent[color]) };
      _bars[color] = gtsam::traits<SE3>::Retract(x_current, result[k_Xcurrent[color]]);
    }
  }

  void update_prior(const RodColors color, const GaussianBayesNetPtr bayes_net, const gtsam::VectorValues& result)
  {
    // marginal->print("marginal", SF::formatter);
    if (result.exists(k_Xcurrent[color]))
    {
      const gtsam::Key pk{ rod_symbol(color, _idx) };
      // PriorFactor(Key key, const VALUE& prior, const SharedNoiseModel& model = nullptr) :
      gtsam::Values v({ { pk, gtsam::genericValue(_bars[color]) } });
      gtsam::PriorFactor<SE3> prior(pk, _bars[color]);

      auto gaussian_factor = prior.linearize(v);
      _priors[color] = boost::make_shared<JacobianFactor>(*gaussian_factor);

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
  std::array<JacobianFactorPtr, 3> _priors;  // Gaussian density on x_
  // JacobianFactorPtr _green_prior;           // Gaussian density on x_
  // JacobianFactorPtr _blue_prior;            // Gaussian density on x_
  JacobianFactorPtr _Rprior;  // Prior on Rz;

  gtsam::noiseModel::Base::shared_ptr _xdot_noise;

  const Translation _offset;
  const Rotation _Roffset;

  gtsam::SharedDiagonal _rot_noise_model;

  bool _use_cables;
};

}  // namespace estimation