#pragma once
#include <array>
#include <numeric>
#include <functional>

#include <gtsam/config.h>
#include <gtsam/base/Testable.h>
#include <gtsam/nonlinear/Expression.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/slam/BetweenFactor.h>

#include <factor_graphs/symbols_factory.hpp>
#include <factor_graphs/lie_integrator.hpp>
#include <tensegrity_utils/rosparams_utils.hpp>
#include <tensegrity_utils/dbg_utils.hpp>

namespace estimation
{
template <typename SE3>
class tensegrity_cap_observation_fix_t : public gtsam::NoiseModelFactorN<SE3>
{
  // using SE3 = prx::fg::se3_t;
  using Velocity = Eigen::Vector<double, 3>;
  using Base = gtsam::NoiseModelFactorN<SE3>;
  using Derived = tensegrity_cap_observation_fix_t;

  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;

  using OptDeriv = boost::optional<Eigen::MatrixXd&>;
  using LieIntegrator = factor_graphs::lie_integrator_t<SE3, Velocity>;

  using Translation = Eigen::Vector<double, 3>;
  using SF = factor_graphs::symbol_factory_t;

  static constexpr Eigen::Index DimX{ gtsam::traits<SE3>::dimension };
  static constexpr Eigen::Index DimZ{ gtsam::traits<Translation>::dimension };
  static constexpr Eigen::Index DimXdot{ gtsam::traits<Velocity>::dimension };

  tensegrity_cap_observation_fix_t() = delete;

public:
  tensegrity_cap_observation_fix_t(const tensegrity_cap_observation_fix_t& other) = delete;

  tensegrity_cap_observation_fix_t(const gtsam::Key key_x, const double dt,        // no-lint
                                   const Translation offset, const Translation z,  // no-lint
                                   const NoiseModel& cost_model, const std::string label = "TensegrityCapObservation")
    : Base(cost_model, key_x), _dt(dt), _z(z), _offset(offset), _label(label)
  {
  }

  ~tensegrity_cap_observation_fix_t() override
  {
  }

  static Translation predict(const SE3& x, const Translation& offset, const double& dt,
                             gtsam::OptionalJacobian<DimZ, DimX> Hx = boost::none)
  {
    Eigen::Matrix<double, DimZ, DimX> p_H_x;

    const Translation cap_prediction{ x.transformTo(offset, Hx) };

    return cap_prediction;
  }

  virtual bool active(const gtsam::Values& values) const override
  {
    const bool activated{ _dt > 0 };

    return activated;
  }

  // Error is: z (-) q_^{predicted}_1; where q_^{predicted}_1 = q0 (+) qdot dt, for a fix (known) dt
  virtual Eigen::VectorXd evaluateError(const SE3& x, OptDeriv Hx = boost::none) const override
  {
    const Translation pt_expected{ predict(x, _offset, _dt, Hx) };
    const Translation error{ pt_expected - _z };

    return error;
  }

  void to_stream(std::ostream& os, const gtsam::Values& values) const
  {
    const char sp{ ' ' };

    const gtsam::Key kx{ this->template key<1>() };

    const SE3 x{ values.at<SE3>(kx) };

    os << _label << sp;
    os << SF::formatter(kx) << " " << x << sp;
    os << "Z: " << _z << " dt:" << _dt << sp;
    os << "\n";
  }

private:
  const double _dt;
  const Translation _z;
  const Translation _offset;
  const std::string _label;
};

template <typename SE3>
class tensegrity_cap_observation_t : public gtsam::NoiseModelFactorN<SE3, Eigen::Vector<double, 3>>
{
  using Velocity = Eigen::Vector<double, 3>;
  using Base = gtsam::NoiseModelFactorN<SE3, Velocity>;
  using Derived = tensegrity_cap_observation_t;

  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;

  using OptDeriv = boost::optional<Eigen::MatrixXd&>;
  using LieIntegrator = factor_graphs::lie_integrator_t<SE3, Velocity>;

  using Translation = Eigen::Vector<double, 3>;
  using SF = factor_graphs::symbol_factory_t;

  static constexpr Eigen::Index DimX{ gtsam::traits<SE3>::dimension };
  static constexpr Eigen::Index DimZ{ gtsam::traits<Translation>::dimension };
  static constexpr Eigen::Index DimXdot{ gtsam::traits<Velocity>::dimension };

  tensegrity_cap_observation_t() = delete;

public:
  tensegrity_cap_observation_t(const tensegrity_cap_observation_t& other) = delete;

  tensegrity_cap_observation_t(const gtsam::Key key_x, const gtsam::Key key_xdot, const double dt,  // no-lint
                               const Translation offset, const Translation z,                       // no-lint
                               const NoiseModel& cost_model, const std::string label = "TensegrityCapObservation")
    : Base(cost_model, key_x, key_xdot), _dt(dt), _z(z), _offset(offset), _label(label)
  {
  }

  ~tensegrity_cap_observation_t() override
  {
  }

  static Translation predict(const SE3& x, const Velocity& xdot, const Translation& offset, const double& dt,
                             gtsam::OptionalJacobian<DimZ, DimX> Hx = boost::none,
                             gtsam::OptionalJacobian<DimZ, DimXdot> Hxdot = boost::none)
  {
    Eigen::Matrix<double, DimZ, DimX> p_H_xdt;
    Eigen::Matrix<double, DimX, DimX> xdt_H_x, xdt_H_xdot;
    const Eigen::Vector<double, 6> vel{ 0, 0, 0, xdot[0], xdot[1], xdot[2] };
    const SE3 x_dt{ LieIntegrator::integrate(x, vel, dt, Hx ? &xdt_H_x : nullptr, Hxdot ? &xdt_H_xdot : nullptr) };

    const Translation cap_prediction{ x_dt.transformTo(offset, p_H_xdt) };
    if (Hx)
    {
      *Hx = p_H_xdt * xdt_H_x;
    }
    if (Hxdot)
    {
      *Hxdot = p_H_xdt * xdt_H_xdot;
    }
    return cap_prediction;
  }

  virtual bool active(const gtsam::Values& values) const override
  {
    const bool activated{ _dt > 0 };

    return activated;
  }

  // Error is: z (-) q_^{predicted}_1; where q_^{predicted}_1 = q0 (+) qdot dt, for a fix (known) dt
  virtual Eigen::VectorXd evaluateError(const SE3& x, const Velocity& xdot,  // no-lint
                                        OptDeriv Hx = boost::none, OptDeriv Hxdot = boost::none) const override
  {
    const Translation pt_expected{ predict(x, xdot, _offset, _dt, Hx, Hxdot) };
    const Translation error{ pt_expected - _z };

    return error;
  }

  void to_stream(std::ostream& os, const gtsam::Values& values) const
  {
    const char sp{ ' ' };

    const gtsam::Key kx{ this->template key<1>() };
    const gtsam::Key kxdot{ this->template key<2>() };

    const SE3 x{ values.at<SE3>(kx) };
    const Velocity xdot{ values.at<Velocity>(kxdot) };

    os << _label << sp;
    os << SF::formatter(kx) << " " << x << sp;
    os << SF::formatter(kxdot) << " " << xdot << sp;
    os << "Z: " << _z << " dt:" << _dt << sp;
    os << "\n";
  }

private:
  const double _dt;
  const Translation _z;
  const Translation _offset;
  const std::string _label;
};

class SE3_single_observation_factor_t : public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Rot3>
{
public:
  using SE3 = gtsam::Pose3;
  using Translation = Eigen::Vector<double, 3>;
  using Rotation = gtsam::Rot3;
  using SkewMatrix = Eigen::Matrix<double, 3, 3>;
  using Base = gtsam::NoiseModelFactor1<SE3, gtsam::Rot3>;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;
  using Jacobian = Eigen::Matrix<double, 3, 6>;

  // using Vector = Eigen::Vector<double, Dim>;

  SE3_single_observation_factor_t(const gtsam::Key key_se3, const gtsam::Key key_offset, const Translation zi,
                                  const Translation offset, const NoiseModel& cost_model)
    : Base(cost_model, key_se3, key_offset), _zi(zi), _offset(offset)
  {
  }

  static Translation prediction(const SE3& x, const Rotation& Roffset, const Translation& offset,  // no-lint
                                boost::optional<Eigen::MatrixXd&> Hx = boost::none,
                                boost::optional<Eigen::MatrixXd&> HR = boost::none)
  {
    Eigen::MatrixXd rOff_H_off;
    Eigen::MatrixXd pred_H_x;
    Eigen::MatrixXd pred_H_rOff;

    const Translation rot_offset{ Roffset.rotate(offset, rOff_H_off) };
    const Translation pred{ x.transformFrom(rot_offset, pred_H_x, pred_H_rOff) };

    if (Hx)
    {
      *Hx = pred_H_x;
    }
    if (HR)
    {
      *HR = pred_H_rOff * rOff_H_off;
    }
    return pred;
  }

  static Eigen::VectorXd compute_error(const SE3& x, const Rotation& Roffset, const Translation& zi,
                                       const Translation& offset,  // no-lint
                                       boost::optional<Eigen::MatrixXd&> Hx = boost::none,
                                       boost::optional<Eigen::MatrixXd&> HR = boost::none)
  {
    // const Translation  x.transformFrom(_zA, Hself = boost::none,
    //                             OptionalJacobian<3, 3> Hpoint = boost::none) const;

    // pRZ.transformFrom(p0i.transformFrom(zB)) - p0i.transformFrom(zA)

    Eigen::MatrixXd pred_H_x;
    Eigen::MatrixXd pred_H_Roff;
    // Eigen::MatrixXd pB_H_xinv;
    // Eigen::MatrixXd pbRot_H_pB;

    const Translation pred{ prediction(x, Roffset, offset, pred_H_x, pred_H_Roff) };  // no-lint
    const Translation err{ pred - zi };

    if (Hx)
    {
      const Eigen::Matrix3d err_H_pred{ Eigen::Matrix3d::Identity() };
      *Hx = err_H_pred * pred_H_x;
    }
    if (HR)
    {
      const Eigen::Matrix3d err_H_pred{ Eigen::Matrix3d::Identity() };
      // const Eigen::Matrix3d err_H_zi{ -Eigen::Matrix3d::Identity() };
      *HR = err_H_pred * pred_H_Roff;
    }
    return err;
  }

  virtual Eigen::VectorXd evaluateError(const SE3& x, const Rotation& Roffset,
                                        boost::optional<Eigen::MatrixXd&> Hx = boost::none,
                                        boost::optional<Eigen::MatrixXd&> HR = boost::none) const override
  {
    const Eigen::VectorXd error{ compute_error(x, Roffset, _zi, _offset, Hx, HR) };

    return error;
  }

private:
  Translation _offset;
  Translation _zi;
};

class rotation_symmetric_factor_t : public gtsam::NoiseModelFactor1<gtsam::Rot3, gtsam::Rot3>
{
public:
  using SE3 = gtsam::Pose3;
  using Translation = Eigen::Vector<double, 3>;
  using Rotation = gtsam::Rot3;
  using SkewMatrix = Eigen::Matrix<double, 3, 3>;
  using Base = gtsam::NoiseModelFactorN<Rotation, Rotation>;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;
  using Jacobian = Eigen::Matrix<double, 3, 6>;

  // using Vector = Eigen::Vector<double, Dim>;

  template <typename RotationType>
  rotation_symmetric_factor_t(const gtsam::Key key_Ri, const gtsam::Key key_Rk, const RotationType offset,
                              const NoiseModel& cost_model)
    : Base(cost_model, key_Ri, key_Rk), _offset(offset)
  {
  }

  static Eigen::VectorXd compute_error(const Rotation& Ri, const Rotation& Rk, const Rotation& Roff,
                                       boost::optional<Eigen::MatrixXd&> HRi = boost::none,
                                       boost::optional<Eigen::MatrixXd&> HRk = boost::none)
  {
    Eigen::MatrixXd Rkoff_H_Rk;
    Eigen::MatrixXd RkoffInv_H_Rkoff;
    Eigen::MatrixXd RikInv_H_Rkoff;
    Eigen::MatrixXd RikInv_H_Ri;
    Eigen::MatrixXd err_H_RikInv;

    const Rotation Rkoff{ gtsam::traits<Rotation>::Compose(Rk, Roff, Rkoff_H_Rk) };
    const Rotation Rkoff_inv{ Rkoff.inverse(RkoffInv_H_Rkoff) };
    const Rotation RikInv{ gtsam::traits<Rotation>::Compose(Rkoff_inv, Ri, RikInv_H_Rkoff, RikInv_H_Ri) };
    const Eigen::Vector3d err{ Rotation::Logmap(RikInv, err_H_RikInv) };

    if (HRi)
    {
      *HRi = err_H_RikInv * RikInv_H_Ri;
    }
    if (HRk)
    {
      // const Eigen::Matrix3d err_H_zi{ -Eigen::Matrix3d::Identity() };
      *HRk = err_H_RikInv * RikInv_H_Rkoff * RkoffInv_H_Rkoff * Rkoff_H_Rk;
    }
    return err;
  }

  virtual Eigen::VectorXd evaluateError(const Rotation& Ri, const Rotation& Rk,
                                        boost::optional<Eigen::MatrixXd&> HRi = boost::none,
                                        boost::optional<Eigen::MatrixXd&> HRk = boost::none) const override
  {
    const Eigen::VectorXd error{ compute_error(Ri, Rk, _offset, HRi, HRk) };

    return error;
  }

private:
  Rotation _offset;
};

// Bar observation factor, which considers that two observations are availble and therefore only one rotation key is
// needed, considering that there is a symmetry (rotation offset) between observations
class bar_two_observations_factor_t : public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Rot3>
{
public:
  using SE3 = gtsam::Pose3;
  using Translation = Eigen::Vector<double, 3>;
  using Rotation = gtsam::Rot3;
  using SkewMatrix = Eigen::Matrix<double, 3, 3>;
  using Base = gtsam::NoiseModelFactor1<SE3, gtsam::Rot3>;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;
  using Jacobian = Eigen::Matrix<double, 3, 6>;

  enum ObservationIdx
  {
    First = 0,
    Second
  };
  // using Vector = Eigen::Vector<double, Dim>;

  template <typename RotationType>
  bar_two_observations_factor_t(const gtsam::Key key_se3, const gtsam::Key key_offset, const Translation zi,
                                const Translation offset, const RotationType Roffset,
                                const ObservationIdx observation_id, const NoiseModel& cost_model)
    : Base(cost_model, key_se3, key_offset), _zi(zi), _offset(offset), _observation_id(observation_id), _Roff(Roffset)
  {
  }

  static Translation predict(const SE3& x, const Rotation& Rx, const Rotation& Roff, const Translation& offset,
                             const ObservationIdx& observation_id,  // no-lint
                             gtsam::OptionalJacobian<3, 6> Hx = boost::none,
                             gtsam::OptionalJacobian<3, 3> HR = boost::none)
  {
    const bool compute_deriv{ Hx or HR };

    Eigen::Matrix<double, 3, 6> pred_H_xp;
    Eigen::Matrix<double, 6, 6> xdt_H_xp;
    // Eigen::Matrix<double, 6, 6> xdt_H_xdot;
    Eigen::Matrix<double, 6, 6> xp_H_x;
    Eigen::Matrix<double, 6, 6> xp_H_xrot;
    Eigen::Matrix<double, 6, 3> xrot_H_r;
    Eigen::Matrix3d pred_H_Off;
    Eigen::Matrix3d r_H_Rx{ Eigen::Matrix3d::Identity() };

    Rotation rotation{ Rx };
    if (observation_id == ObservationIdx::First)
    {
      rotation = gtsam::traits<Rotation>::Compose(Rx, Roff, compute_deriv ? &r_H_Rx : nullptr);
    }
    const Translation zero{ Translation::Zero() };
    const gtsam::Pose3 xrot{ gtsam::Pose3::Create(rotation, zero, compute_deriv ? &xrot_H_r : nullptr) };

    const gtsam::Pose3 xp{ gtsam::traits<gtsam::Pose3>::Compose(x, xrot, compute_deriv ? &xp_H_x : nullptr,
                                                                compute_deriv ? &xp_H_xrot : nullptr) };

    const Translation pred{ xp.transformFrom(offset, compute_deriv ? &pred_H_xp : nullptr) };
    if (Hx)
    {
      *Hx = pred_H_xp * xp_H_x;
    }
    if (HR)
    {
      *HR = pred_H_xp * xp_H_xrot * xrot_H_r * r_H_Rx;
    }
    return pred;
  }

  static Eigen::VectorXd compute_error(const SE3& x, const Rotation& Rx, const Rotation& Roff, const Translation& zi,
                                       const Translation& offset, const ObservationIdx& observation_id,  // no-lint
                                       boost::optional<Eigen::MatrixXd&> Hx = boost::none,
                                       boost::optional<Eigen::MatrixXd&> HR = boost::none)
  {
    Eigen::MatrixXd pred_H_x;
    Eigen::MatrixXd pred_H_Roff;
    // Eigen::MatrixXd pB_H_xinv;
    // Eigen::MatrixXd pbRot_H_pB;

    const Translation pred{ predict(x, Rx, Roff, offset, observation_id, pred_H_x, pred_H_Roff) };  // no-lint
    const Translation err{ pred - zi };

    if (Hx)
    {
      const Eigen::Matrix3d err_H_pred{ Eigen::Matrix3d::Identity() };
      *Hx = err_H_pred * pred_H_x;
    }
    if (HR)
    {
      const Eigen::Matrix3d err_H_pred{ Eigen::Matrix3d::Identity() };
      // const Eigen::Matrix3d err_H_zi{ -Eigen::Matrix3d::Identity() };
      *HR = err_H_pred * pred_H_Roff;
    }
    return err;
  }

  virtual Eigen::VectorXd evaluateError(const SE3& x, const Rotation& Rx,
                                        boost::optional<Eigen::MatrixXd&> Hx = boost::none,
                                        boost::optional<Eigen::MatrixXd&> HR = boost::none) const override
  {
    // compute_error(const SE3& x, const Rotation& Rx, const Rotation& Roff, const Translation& zi,
    //                                    const Translation& offset,
    const Eigen::VectorXd error{ compute_error(x, Rx, _Roff, _zi, _offset, _observation_id, Hx, HR) };

    return error;
  }

  void print(const std::string& s, const gtsam::KeyFormatter& keyFormatter) const override
  {
    const std::string key_x{ keyFormatter(this->template key<1>()) };
    const std::string key_R{ keyFormatter(this->template key<2>()) };

    // const std::string plant_name{ _prx_system->get_pathname() };
    std::cout << s << " ";
    std::cout << "[ " << key_x << " " << key_R << "]\n";
    std::cout << "\t offset: " << _offset.transpose() << "\n";
    std::cout << "\t zi: " << _zi.transpose() << "\n";
    std::cout << "\t Roff:\n" << _Roff << "\n";

    if (_observation_id == ObservationIdx::First)
    {
      std::cout << "\t observation_id: First\n";
    }
    else
    {
      std::cout << "\t observation_id: Second\n";
    }

    if (this->noiseModel_)
      this->noiseModel_->print("  noise model: ");
    else
      std::cout << "no noise model" << std::endl;
    std::cout << "\n";
  }

private:
  const Translation _offset;
  const Translation _zi;
  const Rotation _Roff;
  const ObservationIdx _observation_id;
};

class cable_length_observations_factor_t
  : public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Rot3, gtsam::Pose3, gtsam::Rot3>
{
public:
  using SE3 = gtsam::Pose3;
  using Translation = Eigen::Vector<double, 3>;
  using Rotation = gtsam::Rot3;
  using SkewMatrix = Eigen::Matrix<double, 3, 3>;
  using Base = gtsam::NoiseModelFactor1<SE3, Rotation, SE3, Rotation>;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;
  using Jacobian = Eigen::Matrix<double, 3, 6>;
  using Meassurement = Eigen::Vector<double, 1>;
  using Error = Eigen::Vector<double, 1>;

  using BarFactor = bar_two_observations_factor_t;

  //
  enum CableDistanceId
  {
    AA = 0,
    AB,
    BB
  };
  // using Vector = Eigen::Vector<double, Dim>;

  template <typename RotationType>
  cable_length_observations_factor_t(const gtsam::Key key_Xi, const gtsam::Key key_Ri,  // no-lint
                                     const gtsam::Key key_Xj, const gtsam::Key key_Rj,  // no-lint
                                     const double zij, const Translation offset, const RotationType Roffset,
                                     const CableDistanceId cable_distance_id, const NoiseModel& cost_model)
    : Base(cost_model, key_Xi, key_Ri, key_Xj, key_Rj)
    , _zij(zij)
    , _offset(offset)
    , _Roff(Roffset)
    , _cable_distance_id(cable_distance_id)
  {
  }

  static Meassurement predict(const SE3& xi, const Rotation& Ri, const SE3& xj, const Rotation& Rj,  // no-lint
                              const Translation& offset, const Rotation& Roff,
                              const CableDistanceId& distance_id,  // no-lint
                              gtsam::OptionalJacobian<1, 6> Hxi = boost::none,
                              gtsam::OptionalJacobian<1, 3> HRi = boost::none,
                              gtsam::OptionalJacobian<1, 6> Hxj = boost::none,
                              gtsam::OptionalJacobian<1, 3> HRj = boost::none)
  {
    BarFactor::ObservationIdx endcapId_i, endcapId_j;
    if (distance_id == CableDistanceId::AA)
    {
      endcapId_i = BarFactor::ObservationIdx::First;
      endcapId_j = BarFactor::ObservationIdx::First;
    }
    else if (distance_id == CableDistanceId::AB)
    {
      endcapId_i = BarFactor::ObservationIdx::First;
      endcapId_j = BarFactor::ObservationIdx::Second;
    }
    else if (distance_id == CableDistanceId::BB)
    {
      endcapId_i = BarFactor::ObservationIdx::Second;
      endcapId_j = BarFactor::ObservationIdx::Second;
    }
    else
    {
      TENSEGRITY_THROW("observation_id not supported: ");
    }

    Eigen::Matrix<double, 3, 6> ei_H_xi, ej_H_xj;
    Eigen::Matrix<double, 3, 3> ei_H_Ri, ej_H_Rj;

    const bool compute_deriv{ Hxi or HRi or Hxj or HRj };

    // Compute endcaps of Xi and Xj, depending on endcapId_i and endcapId_j
    const Translation ei{ BarFactor::predict(xi, Ri, Roff, offset, endcapId_i,    // no-lint
                                             compute_deriv ? &ei_H_xi : nullptr,  // no-lint
                                             compute_deriv ? &ei_H_Ri : nullptr) };

    const Translation ej{ BarFactor::predict(xj, Rj, Roff, offset, endcapId_j,    // no-lint
                                             compute_deriv ? &ej_H_xj : nullptr,  // no-lint
                                             compute_deriv ? &ej_H_Rj : nullptr) };

    const Translation diff{ ei - ej };
    const double distance{ diff.norm() };
    const Eigen::Matrix3d diff_H_ei{ Eigen::Matrix3d::Identity() };
    const Eigen::Matrix3d diff_H_ej{ -Eigen::Matrix3d::Identity() };
    const Eigen::Matrix<double, 1, 3> dist_H_diff{ diff / distance };  // (1 / ||diff||) \dot diff

    // if (distance < 0.01)
    // {
    //   DEBUG_VARS(xi);
    //   DEBUG_VARS(xj);
    //   DEBUG_VARS(ei.transpose());
    //   DEBUG_VARS(ej.transpose());
    //   DEBUG_VARS(diff.transpose());
    //   DEBUG_VARS(distance);
    // }

    if (Hxi)
    {
      *Hxi = dist_H_diff * diff_H_ei * ei_H_xi;
    }
    if (HRi)
    {
      *HRi = dist_H_diff * diff_H_ei * ei_H_Ri;
    }
    if (Hxj)
    {
      *Hxj = dist_H_diff * diff_H_ej * ej_H_xj;
    }
    if (HRj)
    {
      *HRj = dist_H_diff * diff_H_ej * ej_H_Rj;
    }
    return Meassurement(distance);
  }

  static Error compute_error(const SE3& xi, const Rotation& Ri, const SE3& xj, const Rotation& Rj, const double& zij,
                             const Translation& offset, const Rotation& Roff,
                             const CableDistanceId& cable_distance_id,  // no-lint
                             boost::optional<Eigen::MatrixXd&> Hxi = boost::none,
                             boost::optional<Eigen::MatrixXd&> HRi = boost::none,
                             boost::optional<Eigen::MatrixXd&> Hxj = boost::none,
                             boost::optional<Eigen::MatrixXd&> HRj = boost::none)
  {
    // const bool compute_deriv{ Hxi or HRi or Hxj or HRj };
    // Eigen::Matrix<double, 1, 6> pred_H_xi, pred_H_xj;
    // Eigen::Matrix<double, 1, 3> pred_H_Ri, pred_H_Rj;

    const Meassurement prediction{ predict(xi, Ri, xj, Rj, offset, Roff, cable_distance_id,  // no-lint
                                           Hxi, HRi, Hxj, HRj) };

    // compute_deriv ? &pred_H_xi : nullptr,             // no-lint
    // compute_deriv ? &pred_H_Ri : nullptr,             // no-lint
    // compute_deriv ? &pred_H_xj : nullptr,             // no-lint
    // compute_deriv ? &pred_H_Rj : nullptr) };
    const Error err{ prediction - Meassurement(zij) };
    // DEBUG_VARS(zij, prediction);
    // *Hxi = err_H_pred * pred_H_xi;
    return err;
  }

  virtual Eigen::VectorXd evaluateError(const SE3& xi, const Rotation& Ri, const SE3& xj, const Rotation& Rj,
                                        boost::optional<Eigen::MatrixXd&> Hxi = boost::none,
                                        boost::optional<Eigen::MatrixXd&> HRi = boost::none,
                                        boost::optional<Eigen::MatrixXd&> Hxj = boost::none,
                                        boost::optional<Eigen::MatrixXd&> HRj = boost::none) const override
  {
    const Error error{ compute_error(xi, Ri, xj, Rj, _zij, _offset, _Roff, _cable_distance_id, Hxi, HRi, Hxj, HRj) };
    return error;
  }

private:
  const double _zij;
  const Translation _offset;
  const Rotation _Roff;
  const CableDistanceId _cable_distance_id;
};

// Bar observation factor, which considers that two observations are availble and therefore only one rotation key is
// needed, considering that there is a symmetry (rotation offset) between observations
class endcap_observation_factor_t : public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Rot3>
{
public:
  using SE3 = gtsam::Pose3;
  using Translation = Eigen::Vector<double, 3>;
  using Rotation = gtsam::Rot3;
  using SkewMatrix = Eigen::Matrix<double, 3, 3>;
  using Base = gtsam::NoiseModelFactor1<SE3, gtsam::Rot3>;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;
  using Jacobian = Eigen::Matrix<double, 3, 6>;

  // using Vector = Eigen::Vector<double, Dim>;

  endcap_observation_factor_t(const gtsam::Key key_se3, const gtsam::Key key_offset, const Translation zi,
                              const Translation offset, const NoiseModel& cost_model)
    : Base(cost_model, key_se3, key_offset), _zi(zi), _offset(offset)
  {
  }

  static Translation predict(const SE3& x, const Rotation& Rx, const Translation& offset,
                             gtsam::OptionalJacobian<3, 6> Hx = boost::none,
                             gtsam::OptionalJacobian<3, 3> HR = boost::none)
  {
    const bool compute_deriv{ Hx or HR };

    Eigen::Matrix<double, 3, 6> pred_H_xp;
    // Eigen::Matrix<double, 6, 6> xdt_H_xp;
    // Eigen::Matrix<double, 6, 6> xdt_H_xdot;
    Eigen::Matrix<double, 6, 6> xp_H_x;
    Eigen::Matrix<double, 6, 6> xp_H_xrot;
    Eigen::Matrix<double, 6, 3> xrot_H_Rx;
    // Eigen::Matrix3d pred_H_Off;

    const Translation zero{ Translation::Zero() };
    const gtsam::Pose3 xrot{ gtsam::Pose3::Create(Rx, zero, compute_deriv ? &xrot_H_Rx : nullptr) };

    const gtsam::Pose3 xp{ gtsam::traits<gtsam::Pose3>::Compose(x, xrot, compute_deriv ? &xp_H_x : nullptr,
                                                                compute_deriv ? &xp_H_xrot : nullptr) };

    const Translation pred{ xp.transformFrom(offset, compute_deriv ? &pred_H_xp : nullptr) };
    if (Hx)
    {
      *Hx = pred_H_xp * xp_H_x;
    }
    if (HR)
    {
      *HR = pred_H_xp * xp_H_xrot * xrot_H_Rx;
    }
    return pred;
  }

  static Eigen::VectorXd compute_error(const SE3& x, const Rotation& Rx, const Translation& zi,
                                       const Translation& offset,  // no-lint
                                       boost::optional<Eigen::MatrixXd&> Hx = boost::none,
                                       boost::optional<Eigen::MatrixXd&> HR = boost::none)
  {
    Eigen::MatrixXd pred_H_x;
    Eigen::MatrixXd pred_H_Roff;
    // Eigen::MatrixXd pB_H_xinv;
    // Eigen::MatrixXd pbRot_H_pB;

    const Translation pred{ endcap_observation_factor_t::predict(x, Rx, offset, pred_H_x, pred_H_Roff) };  // no-lint
    const Translation err{ pred - zi };

    if (Hx)
    {
      const Eigen::Matrix3d err_H_pred{ Eigen::Matrix3d::Identity() };
      *Hx = err_H_pred * pred_H_x;
    }
    if (HR)
    {
      const Eigen::Matrix3d err_H_pred{ Eigen::Matrix3d::Identity() };
      *HR = err_H_pred * pred_H_Roff;
    }
    return err;
  }

  virtual Eigen::VectorXd evaluateError(const SE3& x, const Rotation& Rx,
                                        boost::optional<Eigen::MatrixXd&> Hx = boost::none,
                                        boost::optional<Eigen::MatrixXd&> HR = boost::none) const override
  {
    const Eigen::VectorXd error{ compute_error(x, Rx, _zi, _offset, Hx, HR) };

    return error;
  }

  void print(const std::string& s, const gtsam::KeyFormatter& keyFormatter) const override
  {
    const std::string key_x{ keyFormatter(this->template key<1>()) };
    const std::string key_R{ keyFormatter(this->template key<2>()) };

    // const std::string plant_name{ _prx_system->get_pathname() };
    std::cout << s << " ";
    std::cout << "[ " << key_x << " " << key_R << "]\n";
    std::cout << "\t offset: " << _offset.transpose() << "\n";
    std::cout << "\t zi: " << _zi.transpose() << "\n";

    if (this->noiseModel_)
      this->noiseModel_->print("  noise model: ");
    else
      std::cout << "no noise model" << std::endl;
    std::cout << "\n";
  }

private:
  const Translation _offset;
  const Translation _zi;
};

class endcap_rotation_offset_factor_t : public gtsam::NoiseModelFactorN<gtsam::Rot3, gtsam::Rot3>
{
public:
  using SE3 = gtsam::Pose3;
  using Rotation = gtsam::Rot3;
  using SkewMatrix = Eigen::Matrix<double, 3, 3>;
  using Base = gtsam::NoiseModelFactor1<Rotation, Rotation>;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;
  using Jacobian = Eigen::Matrix<double, 3, 6>;

  // using Vector = Eigen::Vector<double, Dim>;

  template <typename RotationType>
  endcap_rotation_offset_factor_t(const gtsam::Key key_rotA, const gtsam::Key key_rotB, const RotationType Roff,
                                  const NoiseModel& cost_model)
    : Base(cost_model, key_rotA, key_rotB), _Roff(Roff)
  {
  }

  static Eigen::VectorXd compute_error(const Rotation& Ra, const Rotation& Rb, const Rotation& Roff,  // no-lint
                                       gtsam::OptionalJacobian<3, 3> HRa = boost::none,
                                       gtsam::OptionalJacobian<3, 3> HRb = boost::none)
  {
    const bool compute_deriv{ HRa or HRb };

    Eigen::Matrix<double, 3, 3> Rab_H_Ra;
    Eigen::Matrix<double, 3, 3> Rab_H_Rb;
    Eigen::Matrix<double, 3, 3> Rba_H_Ra;
    Eigen::Matrix<double, 3, 3> Rba_H_Rb;
    Eigen::Matrix<double, 3, 3> err_H_Rab;
    Eigen::Matrix<double, 3, 3> err_H_Rba;
    // Eigen::Matrix3d pred_H_Off;

    // const Rotation RoffT{ Roff.inverse() };
    // const Rotation RbT{ Rb.inverse(RbT_H_Rb) };

    const Rotation Rab{ gtsam::traits<gtsam::Rot3>::Compose(Ra, Rb,  // no-lint
                                                            compute_deriv ? &Rab_H_Ra : nullptr,
                                                            compute_deriv ? &Rab_H_Rb : nullptr) };

    // const Rotation Rba{ gtsam::traits<gtsam::Rot3>::Compose(Rb, Ra,  // no-lint
    //                                                         compute_deriv ? &Rba_H_Rb : nullptr,
    //                                                         compute_deriv ? &Rba_H_Ra : nullptr) };

    // const Rotation between{ Rab.between(Roff,                                // no-lint
    //                                     compute_deriv ? &b_H_Rab : nullptr,  // no-lint
    //                                     compute_deriv ? &b_H_Roff : nullptr) };
    // const Eigen::VectorXd error{ Rotation::Logmap(between, compute_deriv ? &err_H_b : nullptr) };
    const Eigen::VectorXd error{ gtsam::traits<gtsam::Rot3>::Local(Rab, Roff, err_H_Rab) };
    if (HRa)
    {
      *HRa = err_H_Rab * Rab_H_Ra;
    }
    if (HRb)
    {
      *HRb = err_H_Rab * Rab_H_Rb;
    }
    return error;
  }

  virtual Eigen::VectorXd evaluateError(const Rotation& Ra, const Rotation& Rb,
                                        boost::optional<Eigen::MatrixXd&> HRa = boost::none,
                                        boost::optional<Eigen::MatrixXd&> HRb = boost::none) const override
  {
    const Eigen::VectorXd error{ compute_error(Ra, Rb, _Roff, HRa, HRb) };

    return error;
  }

private:
  const Rotation _Roff;
};

class rotation_fix_identity_t : public gtsam::NoiseModelFactorN<gtsam::Rot3, gtsam::Rot3>
{
public:
  using SE3 = gtsam::Pose3;
  using Rotation = gtsam::Rot3;
  using SkewMatrix = Eigen::Matrix<double, 3, 3>;
  using Base = gtsam::NoiseModelFactor1<Rotation, Rotation>;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;
  using Jacobian = Eigen::Matrix<double, 3, 6>;

  // using Vector = Eigen::Vector<double, Dim>;

  template <typename RotationType>
  rotation_fix_identity_t(const gtsam::Key key_rotA, const gtsam::Key key_rotB, const RotationType Roff,
                          const NoiseModel& cost_model)
    : Base(cost_model, key_rotA, key_rotB), _Roff(Roff)
  {
  }

  static Eigen::VectorXd compute_error(const Rotation& Ra, const Rotation& Rb, const Rotation& Roff,  // no-lint
                                       gtsam::OptionalJacobian<3, 3> HRa = boost::none,
                                       gtsam::OptionalJacobian<3, 3> HRb = boost::none)
  {
    const bool compute_deriv{ HRa or HRb };

    Eigen::Matrix<double, 3, 3> Rboff_H_Rb;
    Eigen::Matrix<double, 3, 3> Rboff_H_Roff;
    // Eigen::Matrix<double, 3, 3> RbT_H_Rb;
    // Eigen::Matrix<double, 3, 3> RbTa_H_RbT;
    // Eigen::Matrix<double, 3, 3> RbTa_H_Ra;
    Eigen::Matrix<double, 3, 3> err_H_Ra;
    Eigen::Matrix<double, 3, 3> err_H_Rboff;
    // Eigen::Matrix<double, 3, 3> err_H_Rab;
    // Eigen::Matrix<double, 3, 3> err_H_Rba;

    // const gtsam::Rot3 Rab{ gtsam::traits<gtsam::Rot3>::Compose(RbT, Ra,  // no-lint
    //                                                            compute_deriv ? &RbTa_H_RbT : nullptr,
    //                                                            compute_deriv ? &RbTa_H_Ra : nullptr) };

    // const gtsam::Rot3 RbT{ Rb.inverse(RbT_H_Rb) };                        // no-lint
    // const gtsam::Rot3 RbTa{ gtsam::traits<gtsam::Rot3>::Compose(RbT, Ra,  // no-lint
    //                                                             compute_deriv ? &RbTa_H_RbT : nullptr,
    //                                                             compute_deriv ? &RbTa_H_Ra : nullptr) };
    // const gtsam::Rot3 Rab{ gtsam::traits<gtsam::Rot3>::Compose(Ra, Rb,  // no-lint
    //                                                            compute_deriv ? &Rab_H_Ra : nullptr,
    //                                                            compute_deriv ? &Rab_H_Rb : nullptr) };
    const gtsam::Rot3 Rboff{ gtsam::traits<gtsam::Rot3>::Compose(Rb, Roff,  // no-lint
                                                                 compute_deriv ? &Rboff_H_Rb : nullptr,
                                                                 compute_deriv ? &Rboff_H_Roff : nullptr) };

    const Eigen::VectorXd error{ gtsam::traits<gtsam::Rot3>::Local(Ra, Rboff, err_H_Ra, err_H_Rboff) };
    // const Eigen::VectorXd error{ gtsam::traits<gtsam::Rot3>::Local(Rab, Rba, err_H_Rab, err_H_Rba) };
    if (HRa)
    {
      // err_H_Rab = Matrix::Identity(3,3);
      *HRa = err_H_Ra;
    }
    if (HRb)
    {
      *HRb = err_H_Rboff * Rboff_H_Rb;
    }
    return error;
  }

  virtual Eigen::VectorXd evaluateError(const Rotation& Ra, const Rotation& Rb,
                                        boost::optional<Eigen::MatrixXd&> HRa = boost::none,
                                        boost::optional<Eigen::MatrixXd&> HRb = boost::none) const override
  {
    const Eigen::VectorXd error{ compute_error(Ra, Rb, _Roff, HRa, HRb) };

    return error;
  }

private:
  const Rotation _Roff;
};

class cable_length_no_rotation_factor_t : public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Pose3, gtsam::Rot3>
{
public:
  using SE3 = gtsam::Pose3;
  using Translation = Eigen::Vector<double, 3>;
  using Rotation = gtsam::Rot3;
  using SkewMatrix = Eigen::Matrix<double, 3, 3>;
  using Base = gtsam::NoiseModelFactor1<SE3, SE3, Rotation>;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;
  using Jacobian = Eigen::Matrix<double, 3, 6>;
  using Meassurement = Eigen::Vector<double, 1>;
  using Error = Eigen::Vector<double, 1>;

  // using BarFactor = bar_two_observations_factor_t;
  using BarFactor = endcap_observation_factor_t;
  // using Vector = Eigen::Vector<double, Dim>;

  cable_length_no_rotation_factor_t(const gtsam::Key key_Xi, const gtsam::Key key_Xj, const gtsam::Key key_Ri,
                                    const double zij, const Translation offset, const NoiseModel& cost_model)
    : Base(cost_model, key_Xi, key_Xj, key_Ri), _zij(zij), _offset(offset)
  {
  }

  static Meassurement predict(const SE3& xi, const SE3& xj, const Rotation& Ri, const Translation& offset,
                              gtsam::OptionalJacobian<1, 6> Hxi = boost::none,
                              gtsam::OptionalJacobian<1, 6> Hxj = boost::none,
                              gtsam::OptionalJacobian<1, 3> HRi = boost::none)
  {
    Eigen::Matrix<double, 3, 6> ei_H_xi, ej_H_xj;
    Eigen::Matrix<double, 3, 3> ei_H_Ri;

    const bool compute_deriv{ Hxi or Hxj or HRi };

    // Compute endcaps of Xi and Xj, depending on endcapId_i and endcapId_j
    const Translation ei{ BarFactor::predict(xi, Ri, offset,                      // no-lint
                                             compute_deriv ? &ei_H_xi : nullptr,  // no-lint
                                             compute_deriv ? &ei_H_Ri : nullptr) };

    const Rotation Rj{ Rotation() };  // Identity
    const Translation ej{ BarFactor::predict(xj, Rj, offset, compute_deriv ? &ej_H_xj : nullptr) };

    const Translation diff{ ei - ej };
    const double distance{ diff.norm() };
    const Eigen::Matrix3d diff_H_ei{ Eigen::Matrix3d::Identity() };
    const Eigen::Matrix3d diff_H_ej{ -Eigen::Matrix3d::Identity() };
    const Eigen::Matrix<double, 1, 3> dist_H_diff{ diff / distance };  // (1 / ||diff||) \dot diff

    // DEBUG_VARS(ei.transpose(), ej.transpose());
    if (Hxi)
    {
      *Hxi = dist_H_diff * diff_H_ei * ei_H_xi;
    }
    if (Hxj)
    {
      *Hxj = dist_H_diff * diff_H_ej * ej_H_xj;
    }
    if (HRi)
    {
      *HRi = dist_H_diff * diff_H_ei * ei_H_Ri;
    }
    return Meassurement(distance);
  }

  static Error compute_error(const SE3& xi, const SE3& xj, const Rotation& Ri, const double& zij,
                             const Translation& offset,  // no-lint
                             boost::optional<Eigen::MatrixXd&> Hxi = boost::none,
                             boost::optional<Eigen::MatrixXd&> Hxj = boost::none,
                             boost::optional<Eigen::MatrixXd&> HRi = boost::none)
  {
    const Meassurement prediction{ predict(xi, xj, Ri, offset, Hxi, Hxj, HRi) };

    // No derivatives for meassurement zi necessary.
    const Error err{ prediction - Meassurement(zij) };
    // DEBUG_VARS(err, prediction, zij)
    return err;
  }

  virtual Eigen::VectorXd evaluateError(const SE3& xi, const SE3& xj, const Rotation& Ri,
                                        boost::optional<Eigen::MatrixXd&> Hxi = boost::none,
                                        boost::optional<Eigen::MatrixXd&> Hxj = boost::none,
                                        boost::optional<Eigen::MatrixXd&> HRi = boost::none) const override
  {
    const Error error{ compute_error(xi, xj, Ri, _zij, _offset, Hxi, Hxj, HRi) };
    return error;
  }

  void print(const std::string& s, const gtsam::KeyFormatter& keyFormatter) const override
  {
    const std::string key_xi{ keyFormatter(this->template key<1>()) };
    const std::string key_xj{ keyFormatter(this->template key<2>()) };
    const std::string key_Ri{ keyFormatter(this->template key<3>()) };

    // const std::string plant_name{ _prx_system->get_pathname() };
    std::cout << s << " ";
    std::cout << "[ " << key_xi << " " << key_xj << " " << key_Ri << "]\n";
    std::cout << "\t offset: " << _offset.transpose() << "\n";
    std::cout << "\t zij: " << _zij << "\n";

    if (this->noiseModel_)
      this->noiseModel_->print("  noise model: ");
    else
      std::cout << "no noise model" << std::endl;
    std::cout << "\n";
  }

private:
  const double _zij;
  const Translation _offset;
};

class bar_vel_observation_factor_t
  : public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Rot3, Eigen::Vector<double, 6>>
{
public:
  using SE3 = gtsam::Pose3;
  using Translation = Eigen::Vector<double, 3>;
  using Rotation = gtsam::Rot3;
  using Velocity = Eigen::Vector<double, 6>;
  using SkewMatrix = Eigen::Matrix<double, 3, 3>;
  using Base = gtsam::NoiseModelFactorN<SE3, gtsam::Rot3, Velocity>;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;
  using Jacobian = Eigen::Matrix<double, 3, 6>;

  using LieIntegrator = factor_graphs::lie_integrator_t<SE3, Velocity>;

  bar_vel_observation_factor_t(const gtsam::Key key_se3, const gtsam::Key key_offset, const gtsam::Key key_vel,
                               const Translation zi, const Translation offset, const double dt,
                               const NoiseModel& cost_model)
    : Base(cost_model, key_se3, key_offset, key_vel), _zi(zi), _offset(offset), _dt(dt)
  {
  }

  static Translation predict(const SE3& x, const Rotation& Rx, const Velocity& xdot, const Translation& offset,
                             const double dt,  // no-lint
                             gtsam::OptionalJacobian<3, 6> Hx = boost::none,
                             gtsam::OptionalJacobian<3, 3> HR = boost::none,
                             gtsam::OptionalJacobian<3, 6> Hxdot = boost::none)
  {
    const bool compute_deriv{ Hx or HR or Hxdot };

    // Eigen::Matrix<double, 3, 3> rOff_H_r;
    Eigen::Matrix<double, 6, 6> x0dt_H_x;
    Eigen::Matrix<double, 6, 6> x0dt_H_xdot;
    Eigen::Matrix<double, 3, 6> pred_H_x0dt;
    Eigen::Matrix<double, 3, 3> pred_H_Rx;

    const SE3 x0dt{ LieIntegrator::integrate(x, xdot, dt, x0dt_H_x, x0dt_H_xdot) };

    const Translation pred{ endcap_observation_factor_t::predict(x0dt, Rx, offset,                        // no-lint
                                                                 compute_deriv ? &pred_H_x0dt : nullptr,  // no-lint
                                                                 compute_deriv ? &pred_H_Rx : nullptr) };

    if (Hx)
    {
      *Hx = pred_H_x0dt * x0dt_H_x;
    }
    if (HR)
    {
      *HR = pred_H_Rx;
    }
    if (Hxdot)
    {
      *Hxdot = pred_H_x0dt * x0dt_H_xdot;
    }
    return pred;
  }

  static Eigen::VectorXd compute_error(const SE3& x, const Rotation& Rx, const Velocity& xdot, const Translation& zi,
                                       const Translation& offset, const double& dt,  // no-lint
                                       boost::optional<Eigen::MatrixXd&> Hx = boost::none,
                                       boost::optional<Eigen::MatrixXd&> HR = boost::none,
                                       boost::optional<Eigen::MatrixXd&> Hxdot = boost::none)
  {
    const Translation pred{ predict(x, Rx, xdot, offset, dt, Hx, HR, Hxdot) };  // no-lint
    const Translation err{ pred - zi };

    return err;
  }

  virtual Eigen::VectorXd evaluateError(const SE3& x, const Rotation& Rx, const Velocity& xdot,
                                        boost::optional<Eigen::MatrixXd&> Hx = boost::none,
                                        boost::optional<Eigen::MatrixXd&> HR = boost::none,
                                        boost::optional<Eigen::MatrixXd&> Hxdot = boost::none) const override
  {
    const Eigen::VectorXd error{ compute_error(x, Rx, xdot, _zi, _offset, _dt, Hx, HR, Hxdot) };

    return error;
  }

  void print(const std::string& s, const gtsam::KeyFormatter& keyFormatter) const override
  {
    const std::string key_x{ keyFormatter(this->template key<1>()) };
    const std::string key_R{ keyFormatter(this->template key<2>()) };
    const std::string key_xdot{ keyFormatter(this->template key<3>()) };

    // const std::string plant_name{ _prx_system->get_pathname() };
    std::cout << s << " ";
    std::cout << "[ " << key_x << " " << key_R << "]\n";
    std::cout << "\t offset: " << _offset.transpose() << "\n";
    std::cout << "\t zi: " << _zi.transpose() << "\n";

    if (this->noiseModel_)
      this->noiseModel_->print("  noise model: ");
    else
      std::cout << "no noise model" << std::endl;
    std::cout << "\n";
  }

private:
  const Translation _offset;
  const Translation _zi;
  const double _dt;
};

class cable_length_factor_t : public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Pose3, gtsam::Rot3, gtsam::Rot3>
{
public:
  using SE3 = gtsam::Pose3;
  using Translation = Eigen::Vector<double, 3>;
  using Rotation = gtsam::Rot3;
  using SkewMatrix = Eigen::Matrix<double, 3, 3>;
  using Base = gtsam::NoiseModelFactor1<SE3, SE3, Rotation, Rotation>;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;
  using Jacobian = Eigen::Matrix<double, 3, 6>;
  using Meassurement = Eigen::Vector<double, 1>;
  using Error = Eigen::Vector<double, 1>;

  // using BarFactor = bar_two_observations_factor_t;
  using BarFactor = endcap_observation_factor_t;
  // using Vector = Eigen::Vector<double, Dim>;

  cable_length_factor_t(const gtsam::Key key_Xi, const gtsam::Key key_Xj, const gtsam::Key key_Ri,
                        const gtsam::Key key_Rj, const double zij, const Translation offset,
                        const NoiseModel& cost_model)
    : Base(cost_model, key_Xi, key_Xj, key_Ri, key_Rj), _zij(zij), _offset(offset)
  {
  }

  static Meassurement predict(const SE3& xi, const SE3& xj, const Rotation& Ri, const Rotation& Rj,
                              const Translation& offset,  // no-lint
                              gtsam::OptionalJacobian<1, 6> Hxi = boost::none,
                              gtsam::OptionalJacobian<1, 6> Hxj = boost::none,
                              gtsam::OptionalJacobian<1, 3> HRi = boost::none,
                              gtsam::OptionalJacobian<1, 3> HRj = boost::none)
  {
    Eigen::Matrix<double, 3, 6> ei_H_xi, ej_H_xj;
    Eigen::Matrix<double, 3, 3> ei_H_Ri, ej_H_Rj;

    const bool compute_deriv{ Hxi or Hxj or HRi or HRj };

    // Compute endcaps of Xi and Xj, depending on endcapId_i and endcapId_j
    const Translation ei{ BarFactor::predict(xi, Ri, offset,                      // no-lint
                                             compute_deriv ? &ei_H_xi : nullptr,  // no-lint
                                             compute_deriv ? &ei_H_Ri : nullptr) };

    // const Rotation Rj{ Rotation() };  // Identity
    const Translation ej{ BarFactor::predict(xj, Rj, offset,                      // no-lint
                                             compute_deriv ? &ej_H_xj : nullptr,  // no-lint
                                             compute_deriv ? &ej_H_Rj : nullptr) };

    const Translation diff{ ei - ej };
    const double distance{ diff.norm() };
    const Eigen::Matrix3d diff_H_ei{ Eigen::Matrix3d::Identity() };
    const Eigen::Matrix3d diff_H_ej{ -Eigen::Matrix3d::Identity() };
    const Eigen::Matrix<double, 1, 3> dist_H_diff{ diff / distance };  // (1 / ||diff||) \dot diff

    // DEBUG_VARS(ei.transpose(), ej.transpose());
    if (Hxi)
    {
      *Hxi = dist_H_diff * diff_H_ei * ei_H_xi;
    }
    if (Hxj)
    {
      *Hxj = dist_H_diff * diff_H_ej * ej_H_xj;
    }
    if (HRi)
    {
      *HRi = dist_H_diff * diff_H_ei * ei_H_Ri;
    }
    if (HRj)
    {
      *HRj = dist_H_diff * diff_H_ej * ej_H_Rj;
    }
    return Meassurement(distance);
  }

  static Error compute_error(const SE3& xi, const SE3& xj, const Rotation& Ri, const Rotation& Rj, const double& zij,
                             const Translation& offset,  // no-lint
                             boost::optional<Eigen::MatrixXd&> Hxi = boost::none,
                             boost::optional<Eigen::MatrixXd&> Hxj = boost::none,
                             boost::optional<Eigen::MatrixXd&> HRi = boost::none,
                             boost::optional<Eigen::MatrixXd&> HRj = boost::none)
  {
    const Meassurement prediction{ predict(xi, xj, Ri, Rj, offset, Hxi, Hxj, HRi, HRj) };

    // No derivatives for meassurement zi necessary.
    const Error err{ prediction - Meassurement(zij) };
    // DEBUG_VARS(err, prediction, zij)
    return err;
  }

  virtual Eigen::VectorXd evaluateError(const SE3& xi, const SE3& xj, const Rotation& Ri, const Rotation& Rj,
                                        boost::optional<Eigen::MatrixXd&> Hxi = boost::none,
                                        boost::optional<Eigen::MatrixXd&> Hxj = boost::none,
                                        boost::optional<Eigen::MatrixXd&> HRi = boost::none,
                                        boost::optional<Eigen::MatrixXd&> HRj = boost::none) const override
  {
    const Error error{ compute_error(xi, xj, Ri, Rj, _zij, _offset, Hxi, Hxj, HRi, HRj) };
    return error;
  }

  void print(const std::string& s, const gtsam::KeyFormatter& keyFormatter) const override
  {
    const std::string key_xi{ keyFormatter(this->template key<1>()) };
    const std::string key_xj{ keyFormatter(this->template key<2>()) };
    const std::string key_Ri{ keyFormatter(this->template key<3>()) };
    const std::string key_Rj{ keyFormatter(this->template key<4>()) };

    // const std::string plant_name{ _prx_system->get_pathname() };
    std::cout << s << " ";
    std::cout << "[ " << key_xi << " " << key_xj << " " << key_Ri << " " << key_Rj << "]\n";
    std::cout << "\t offset: " << _offset.transpose() << "\n";
    std::cout << "\t zij: " << _zij << "\n";

    if (this->noiseModel_)
      this->noiseModel_->print("  noise model: ");
    else
      std::cout << "no noise model" << std::endl;
    std::cout << "\n";
  }

private:
  const double _zij;
  const Translation _offset;
};

class tensegrity_chirality_factor_t : public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>
{
public:
  using SE3 = gtsam::Pose3;
  using Translation = Eigen::Vector<double, 3>;
  using Rotation = gtsam::Rot3;
  using SkewMatrix = Eigen::Matrix<double, 3, 3>;
  using Base = gtsam::NoiseModelFactor1<SE3, SE3, SE3>;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;
  using Jacobian = Eigen::Matrix<double, 3, 6>;
  using Meassurement = Eigen::Vector<double, 1>;
  using Error = Eigen::Vector<double, 1>;

  // using BarFactor = bar_two_observations_factor_t;
  using BarFactor = endcap_observation_factor_t;
  // using Vector = Eigen::Vector<double, Dim>;

  template <typename RotationType>
  tensegrity_chirality_factor_t(const gtsam::Key key_Xi, const gtsam::Key key_Xj, const gtsam::Key key_Xk,
                                const Translation offset, const RotationType Roff, const NoiseModel& cost_model)
    : Base(cost_model, key_Xi, key_Xj, key_Xk), _offset(offset), _Roff(Roff)
  {
  }

  static Error predict(const SE3& xi, const SE3& xj, const SE3& xk,      // no-lint
                       const Translation& offset, const Rotation& Roff,  // no-lint
                       gtsam::OptionalJacobian<1, 6> Hxi = boost::none,  // no-lint
                       gtsam::OptionalJacobian<1, 6> Hxj = boost::none,  // no-lint
                       gtsam::OptionalJacobian<1, 6> Hxk = boost::none)
  {
    Eigen::Matrix<double, 3, 6> piA_H_xi, pjA_H_xj;
    Eigen::Matrix<double, 3, 6> pjB_H_xj, pkB_H_xk;
    Eigen::Matrix<double, 1, 3> dir_H_pijA, dir_H_pjkB;

    const bool compute_deriv{ Hxi or Hxj or Hxk };

    // Compute endcaps of Xi and Xj, depending on endcapId_i and endcapId_j
    const Translation Pi_A{ BarFactor::predict(xi, Rotation(), offset,  // no-lint
                                               compute_deriv ? &piA_H_xi : nullptr) };

    const Translation Pj_A{ BarFactor::predict(xj, Rotation(), offset,  // no-lint
                                               compute_deriv ? &pjA_H_xj : nullptr) };

    const Translation Pj_B{ BarFactor::predict(xj, Roff, offset,  // no-lint
                                               compute_deriv ? &pjB_H_xj : nullptr) };

    const Translation Pk_B{ BarFactor::predict(xk, Roff, offset,  // no-lint
                                               compute_deriv ? &pkB_H_xk : nullptr) };

    const Translation Pij_A{ Pi_A - Pj_A };
    const Translation Pjk_B{ Pj_B - Pk_B };

    const Eigen::Matrix3d PijA_H_piA{ Eigen::Matrix3d::Identity() };
    const Eigen::Matrix3d PjkB_H_pjB{ Eigen::Matrix3d::Identity() };
    const Eigen::Matrix3d PijA_H_pjA{ -Eigen::Matrix3d::Identity() };
    const Eigen::Matrix3d PjkB_H_pkB{ -Eigen::Matrix3d::Identity() };

    const double dir{ gtsam::dot(Pij_A, Pjk_B, dir_H_pijA, dir_H_pjkB) };

    if (Hxi)
    {
      *Hxi = dir_H_pijA * PijA_H_piA * piA_H_xi;
    }
    if (Hxj)
    {
      *Hxj = dir_H_pijA * PijA_H_pjA * pjA_H_xj + dir_H_pjkB * PjkB_H_pjB * pjB_H_xj;
    }
    if (Hxk)
    {
      *Hxk = dir_H_pjkB * PjkB_H_pkB * pkB_H_xk;
    }

    return Error(dir);
  }

  virtual Eigen::VectorXd evaluateError(const SE3& xi, const SE3& xj, const SE3& xk,
                                        boost::optional<Eigen::MatrixXd&> Hxi = boost::none,
                                        boost::optional<Eigen::MatrixXd&> Hxj = boost::none,
                                        boost::optional<Eigen::MatrixXd&> Hxk = boost::none) const override
  {
    Error dir{ predict(xi, xj, xk, _offset, _Roff, Hxi, Hxj, Hxk) };
    if (dir[0] > 0)
    {
      dir[0] = 0;
    }
    return dir;
  }

  void print(const std::string& s, const gtsam::KeyFormatter& keyFormatter) const override
  {
    const std::string key_xi{ keyFormatter(this->template key<1>()) };
    const std::string key_xj{ keyFormatter(this->template key<2>()) };
    const std::string key_xk{ keyFormatter(this->template key<3>()) };

    // const std::string plant_name{ _prx_system->get_pathname() };
    std::cout << s << " ";
    std::cout << "[ " << key_xi << " " << key_xj << " " << key_xk << "]\n";
    std::cout << "\t offset: " << _offset.transpose() << "\n";
    // std::cout << "\t Roff: " << _Roff << "\n";

    // if (this->noiseModel_)
    //   this->noiseModel_->print("  noise model: ");
    // else
    //   std::cout << "no noise model" << std::endl;
    std::cout << "\n";
  }

private:
  const Rotation _Roff;
  const Translation _offset;
};

class tensegrity_triangle_factor_t : public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>
{
public:
  using SE3 = gtsam::Pose3;
  using Translation = Eigen::Vector<double, 3>;
  using Rotation = gtsam::Rot3;
  using SkewMatrix = Eigen::Matrix<double, 3, 3>;
  using Base = gtsam::NoiseModelFactor1<SE3, SE3, SE3>;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;
  using Jacobian = Eigen::Matrix<double, 3, 6>;
  using Meassurement = Eigen::Vector<double, 1>;
  using Error = Eigen::VectorXd;

  // using BarFactor = bar_two_observations_factor_t;
  using BarFactor = endcap_observation_factor_t;
  // using Vector = Eigen::Vector<double, Dim>;
  using Triangle = std::tuple<gtsam::Point3, Translation, Translation>;
  enum TriangleFactorType
  {
    ForceTriangle,
    ScaleEquivalent,
    ParallelTriangles
  };

  template <typename RotationType>
  tensegrity_triangle_factor_t(const gtsam::Key key_Xi, const gtsam::Key key_Xj, const gtsam::Key key_Xk,
                               const Translation offset, const RotationType Roff, const TriangleFactorType factor_type,
                               const NoiseModel& cost_model)
    : Base(cost_model, key_Xi, key_Xj, key_Xk), _offset(offset), _Roff(Roff), _factor_type(factor_type)
  {
  }

  static Triangle get_triangle(const SE3& xi, const SE3& xj, const SE3& xk,          // no-lint
                               const Translation& offset, const Rotation& Roff,      // no-lint
                               gtsam::OptionalJacobian<3, 6> p_H_xi = boost::none,   // no-lint
                               gtsam::OptionalJacobian<3, 6> v0_H_xi = boost::none,  // no-lint
                               gtsam::OptionalJacobian<3, 6> v1_H_xi = boost::none,  // no-lint
                               gtsam::OptionalJacobian<3, 6> v0_H_xj = boost::none,  // no-lint
                               gtsam::OptionalJacobian<3, 6> v1_H_xj = boost::none,  // no-lint
                               gtsam::OptionalJacobian<3, 6> v0_H_xk = boost::none,  // no-lint
                               gtsam::OptionalJacobian<3, 6> v1_H_xk = boost::none)
  {
    Eigen::Matrix<double, 3, 6> pi_H_xi, pj_H_xj, pk_H_xk;
    const bool compute_deriv{ p_H_xi or v0_H_xi or v1_H_xi or v0_H_xj or v1_H_xj or v0_H_xk or v1_H_xk };

    const Translation pt_i{ BarFactor::predict(xi, Roff, offset,  // no-lint
                                               compute_deriv ? &pi_H_xi : nullptr) };
    const Translation pt_j{ BarFactor::predict(xj, Roff, offset,  // no-lint
                                               compute_deriv ? &pj_H_xj : nullptr) };
    const Translation pt_k{ BarFactor::predict(xk, Roff, offset,  // no-lint
                                               compute_deriv ? &pk_H_xk : nullptr) };

    const Translation v0{ pt_j - pt_i };
    const Translation v1{ pt_k - pt_i };

    if (compute_deriv)
    {
      const Eigen::Matrix3d v0_H_pj{ Eigen::Matrix3d::Identity() };
      const Eigen::Matrix3d v0_H_pi{ -Eigen::Matrix3d::Identity() };
      const Eigen::Matrix3d v1_H_pk{ Eigen::Matrix3d::Identity() };
      const Eigen::Matrix3d v1_H_pi{ -Eigen::Matrix3d::Identity() };
      if (p_H_xi)
      {
        *p_H_xi = pi_H_xi;
      }
      if (v0_H_xi)
      {
        *v0_H_xi = v0_H_pi * pi_H_xi;
      }
      if (v1_H_xi)
      {
        *v1_H_xi = v1_H_pi * pi_H_xi;
      }
      if (v0_H_xj)
      {
        *v0_H_xj = v0_H_pj * pj_H_xj;
      }
      if (v1_H_xj)
      {
        *v1_H_xj = Eigen::Matrix<double, 3, 6>::Zero();
      }
      if (v0_H_xk)
      {
        *v0_H_xk = Eigen::Matrix<double, 3, 6>::Zero();
      }
      if (v1_H_xk)
      {
        *v1_H_xk = v1_H_pk * pk_H_xk;
      }
    }

    return { pt_i, v0, v1 };
  }

  // A triangle is: (P,v0,v1) s.t. v0 not parallel to v1.
  static Error force_triangle(const SE3& xi, const SE3& xj, const SE3& xk,      // no-lint
                              const Translation& offset, const Rotation& Roff,  // no-lint
                              gtsam::OptionalJacobian<1, 6> Hxi = boost::none,  // no-lint
                              gtsam::OptionalJacobian<1, 6> Hxj = boost::none,  // no-lint
                              gtsam::OptionalJacobian<1, 6> Hxk = boost::none)
  {
    const bool compute_deriv{ Hxi or Hxj or Hxk };
    Eigen::Matrix<double, 3, 6> v0_H_xi, v1_H_xi, v0_H_xj, v1_H_xj, v0_H_xk, v1_H_xk;
    const Triangle triangle{ get_triangle(xi, xj, xk, offset, Roff,  // no-lint
                                          boost::none, v0_H_xi, v1_H_xi, v0_H_xj, v1_H_xj, v0_H_xk, v1_H_xk) };

    const Translation p{ std::get<0>(triangle) };
    const Translation v0{ std::get<1>(triangle) };
    const Translation v1{ std::get<2>(triangle) };

    Eigen::Matrix<double, 2, 3> u0_H_v0, u1_H_v1;
    Eigen::Matrix<double, 1, 2> u01dot_H_u0, u01dot_H_u1;
    const gtsam::Unit3 u0{ gtsam::Unit3::FromPoint3(v0, compute_deriv ? &u0_H_v0 : nullptr) };
    const gtsam::Unit3 u1{ gtsam::Unit3::FromPoint3(v1, compute_deriv ? &u1_H_v1 : nullptr) };
    const double u01dot{ u0.dot(u1, u01dot_H_u0, u01dot_H_u1) };

    // DEBUG_VARS(p.transpose());
    // DEBUG_VARS(v0.transpose());
    // DEBUG_VARS(v1.transpose());
    // // DEBUG_VARS(u0, u1);
    // DEBUG_VARS(u01dot);
    if (Hxi)
    {
      *Hxi = u01dot_H_u0 * u0_H_v0 * v0_H_xi + u01dot_H_u1 * u1_H_v1 * v1_H_xi;
    }
    if (Hxj)
    {
      *Hxj = u01dot_H_u0 * u0_H_v0 * v0_H_xj + u01dot_H_u1 * u1_H_v1 * v1_H_xj;
    }
    if (Hxk)
    {
      *Hxk = u01dot_H_u0 * u0_H_v0 * v0_H_xk + u01dot_H_u1 * u1_H_v1 * v1_H_xk;
    }
    if (u01dot > 0.98 or u01dot < -0.98)  // 0.02 ~= 1 degree. dotprod 1 -> parallel
    {
      return Eigen::Vector<double, 1>(u01dot);
    }
    return Eigen::Vector<double, 1>(0.0);
  }

  static Error scale_equivalent(const SE3& xi, const SE3& xj, const SE3& xk,      // no-lint
                                const Translation& offset, const Rotation& Roff,  // no-lint
                                gtsam::OptionalJacobian<4, 6> Hxi = boost::none,  // no-lint
                                gtsam::OptionalJacobian<4, 6> Hxj = boost::none,  // no-lint
                                gtsam::OptionalJacobian<4, 6> Hxk = boost::none)
  {
    const bool compute_deriv{ Hxi or Hxj or Hxk };
    Eigen::Matrix<double, 3, 6> v0A_H_xi, v1A_H_xi, v0A_H_xj, v1A_H_xj, v0A_H_xk, v1A_H_xk;
    Eigen::Matrix<double, 3, 6> v0B_H_xi, v1B_H_xi, v0B_H_xj, v1B_H_xj, v0B_H_xk, v1B_H_xk;
    const Triangle triangleA{ get_triangle(xi, xj, xk, offset, Rotation(),  // no-lint
                                           boost::none,                     // no-lint
                                           compute_deriv ? &v0A_H_xi : nullptr, compute_deriv ? &v1A_H_xi : nullptr,
                                           compute_deriv ? &v0A_H_xj : nullptr, compute_deriv ? &v1A_H_xj : nullptr,
                                           compute_deriv ? &v0A_H_xk : nullptr, compute_deriv ? &v1A_H_xk : nullptr) };
    const Triangle triangleB{ get_triangle(xj, xk, xi, offset, Roff,  // no-lint
                                           boost::none,               // no-lint
                                           compute_deriv ? &v0B_H_xj : nullptr, compute_deriv ? &v1B_H_xj : nullptr,
                                           compute_deriv ? &v0B_H_xk : nullptr, compute_deriv ? &v1B_H_xk : nullptr,
                                           compute_deriv ? &v0B_H_xi : nullptr, compute_deriv ? &v1B_H_xi : nullptr) };

    const Translation v0A{ std::get<1>(triangleA) };
    const Translation v1A{ std::get<2>(triangleA) };

    const Translation v0B{ std::get<1>(triangleB) };
    const Translation v1B{ std::get<2>(triangleB) };

    Eigen::Matrix<double, 2, 3> u0A_H_v0A, u1A_H_v1A;
    Eigen::Matrix<double, 2, 3> u0B_H_v0B, u1B_H_v1B;

    const gtsam::Unit3 u0A{ gtsam::Unit3::FromPoint3(v0A, u0A_H_v0A) };
    const gtsam::Unit3 u1A{ gtsam::Unit3::FromPoint3(v1A, u1A_H_v1A) };

    const gtsam::Unit3 u0B{ gtsam::Unit3::FromPoint3(v0B, u0B_H_v0B) };
    const gtsam::Unit3 u1B{ gtsam::Unit3::FromPoint3(v1B, u1B_H_v1B) };

    Eigen::Matrix<double, 2, 2> errA_H_u0A, errA_H_u1A;
    Eigen::Matrix<double, 2, 2> errB_H_u0B, errB_H_u1B;
    const Eigen::Vector<double, 2> errorA{ u0A.errorVector(u1A, errA_H_u0A, errA_H_u1A) };
    const Eigen::Vector<double, 2> errorB{ u0B.errorVector(u1B, errB_H_u0B, errB_H_u1B) };

    const Eigen::Vector<double, 4> err{ { errorA[0], errorA[1], errorB[0], errorB[1] } };
    Eigen::Matrix<double, 4, 2> err_H_errA{ Eigen::Matrix<double, 4, 2>::Zero() };
    Eigen::Matrix<double, 4, 2> err_H_errB{ Eigen::Matrix<double, 4, 2>::Zero() };
    err_H_errA(0, 0) = 1.0;
    err_H_errA(1, 1) = 1.0;
    err_H_errB(2, 0) = 1.0;
    err_H_errB(3, 1) = 1.0;

    if (Hxi)
    {
      // TODO: simplify
      *Hxi = err_H_errA * errA_H_u0A * u0A_H_v0A * v0A_H_xi    // no-lint
             + err_H_errA * errA_H_u1A * u1A_H_v1A * v1A_H_xi  // no-lint
             + err_H_errB * errB_H_u0B * u0B_H_v0B * v0B_H_xi  // no-lint
             + err_H_errB * errB_H_u1B * u1B_H_v1B * v1B_H_xi;
    }
    if (Hxj)
    {
      *Hxj = err_H_errA * errA_H_u0A * u0A_H_v0A * v0A_H_xj    // no-lint
             + err_H_errA * errA_H_u1A * u1A_H_v1A * v1A_H_xj  // no-lint
             + err_H_errB * errB_H_u0B * u0B_H_v0B * v0B_H_xj  // no-lint
             + err_H_errB * errB_H_u1B * u1B_H_v1B * v1B_H_xj;
    }
    if (Hxk)
    {
      *Hxk = err_H_errA * errA_H_u0A * u0A_H_v0A * v0A_H_xk    // no-lint
             + err_H_errA * errA_H_u1A * u1A_H_v1A * v1A_H_xk  // no-lint
             + err_H_errB * errB_H_u0B * u0B_H_v0B * v0B_H_xk  // no-lint
             + err_H_errB * errB_H_u1B * u1B_H_v1B * v1B_H_xk;
    }

    return err;
  }

  static Error coplanarity(const SE3& xi, const SE3& xj, const SE3& xk,      // no-lint
                           const Translation& offset, const Rotation& Roff,  // no-lint
                           gtsam::OptionalJacobian<1, 6> Hxi = boost::none,  // no-lint
                           gtsam::OptionalJacobian<1, 6> Hxj = boost::none,  // no-lint
                           gtsam::OptionalJacobian<1, 6> Hxk = boost::none)
  {
    Eigen::Matrix<double, 3, 6> p1_H_xi, p2_H_xj, p3_H_xk, p4_H_xi;
    const bool compute_deriv{ Hxi or Hxj or Hxk };
    const Translation pt_1{ BarFactor::predict(xi, Roff, offset,  // no-lint
                                               compute_deriv ? &p1_H_xi : nullptr) };
    const Translation pt_2{ BarFactor::predict(xj, Roff, offset,  // no-lint
                                               compute_deriv ? &p2_H_xj : nullptr) };
    const Translation pt_3{ BarFactor::predict(xk, Roff, offset,  // no-lint
                                               compute_deriv ? &p3_H_xk : nullptr) };
    const Translation pt_4{ xi.translation(p4_H_xi) };

    const Eigen::Matrix<double, 3, 3> p21_H_pt2{ Eigen::Matrix3d::Identity() };
    const Eigen::Matrix<double, 3, 3> p41_H_pt4{ Eigen::Matrix3d::Identity() };
    const Eigen::Matrix<double, 3, 3> p31_H_pt3{ Eigen::Matrix3d::Identity() };
    const Eigen::Matrix<double, 3, 3> p21_H_pt1{ -Eigen::Matrix3d::Identity() };
    const Eigen::Matrix<double, 3, 3> p41_H_pt1{ -Eigen::Matrix3d::Identity() };
    const Eigen::Matrix<double, 3, 3> p31_H_pt1{ -Eigen::Matrix3d::Identity() };

    const Translation pt21{ pt_2 - pt_1 };
    const Translation pt41{ pt_4 - pt_1 };
    const Translation pt31{ pt_3 - pt_1 };

    Eigen::Matrix<double, 3, 3> cross_H_pt21, cross_H_pt41;
    const Translation cross{ gtsam::cross(pt21, pt41, cross_H_pt21, cross_H_pt41) };

    Eigen::Matrix<double, 1, 3> dot_H_cross, dot_H_p31;
    const double dot{ gtsam::dot(cross, pt31, dot_H_cross, dot_H_p31) };

    Eigen::Matrix<double, 1, 1> err_H_dot{ 2.0 * dot / 0.02 };
    const double err{ (dot * dot / 0.02) - 1 };

    if (dot < 1e-3)
    {
      err_H_dot(0, 0) = 1;  //??
    }
    // DEBUG_VARS(pt_1.transpose());
    // DEBUG_VARS(pt_2.transpose());
    // DEBUG_VARS(pt_3.transpose());
    // DEBUG_VARS(pt_4.transpose());
    // DEBUG_VARS(dot, err);
    if (Hxi)
    {
      *Hxi = err_H_dot * dot_H_cross * cross_H_pt21 * p21_H_pt1 * p1_H_xi    // no-lint
             + err_H_dot * dot_H_cross * cross_H_pt41 * p41_H_pt1 * p1_H_xi  // no-lint
             + err_H_dot * dot_H_p31 * p31_H_pt1 * p1_H_xi                   // no-lint
             + err_H_dot * dot_H_cross * cross_H_pt41 * p41_H_pt4 * p4_H_xi;
    }
    if (Hxj)
    {
      *Hxj = err_H_dot * dot_H_cross * cross_H_pt21 * p21_H_pt2 * p2_H_xj;  // no-lint
    }
    if (Hxk)
    {
      *Hxk = err_H_dot * dot_H_p31 * p31_H_pt3 * p3_H_xk;  // no-lint
    }

    return Eigen::Vector<double, 1>(std::min(err, 0.0));
  }

  static Error parallel_triangles(const SE3& xi, const SE3& xj, const SE3& xk,      // no-lint
                                  const Translation& offset, const Rotation& Roff,  // no-lint
                                  gtsam::OptionalJacobian<3, 6> Hxi = boost::none,  // no-lint
                                  gtsam::OptionalJacobian<3, 6> Hxj = boost::none,  // no-lint
                                  gtsam::OptionalJacobian<3, 6> Hxk = boost::none)
  {
    const bool compute_deriv{ Hxi or Hxj or Hxk };
    Eigen::Matrix<double, 3, 6> v0A_H_xi, v1A_H_xi, v0A_H_xj, v1A_H_xj, v0A_H_xk, v1A_H_xk;
    Eigen::Matrix<double, 3, 6> v0B_H_xi, v1B_H_xi, v0B_H_xj, v1B_H_xj, v0B_H_xk, v1B_H_xk;
    const Triangle triangleA{ get_triangle(xi, xj, xk, offset, Rotation(),  // no-lint
                                           boost::none,                     // no-lint
                                           compute_deriv ? &v0A_H_xi : nullptr, compute_deriv ? &v1A_H_xi : nullptr,
                                           compute_deriv ? &v0A_H_xj : nullptr, compute_deriv ? &v1A_H_xj : nullptr,
                                           compute_deriv ? &v0A_H_xk : nullptr, compute_deriv ? &v1A_H_xk : nullptr) };
    const Triangle triangleB{ get_triangle(xj, xk, xi, offset, Roff,  // no-lint
                                           boost::none,               // no-lint
                                           compute_deriv ? &v0B_H_xj : nullptr, compute_deriv ? &v1B_H_xj : nullptr,
                                           compute_deriv ? &v0B_H_xk : nullptr, compute_deriv ? &v1B_H_xk : nullptr,
                                           compute_deriv ? &v0B_H_xi : nullptr, compute_deriv ? &v1B_H_xi : nullptr) };

    const Translation pA{ std::get<0>(triangleA) };
    const Translation v0A{ std::get<1>(triangleA) };
    const Translation v1A{ std::get<2>(triangleA) };

    const Translation pB{ std::get<0>(triangleB) };
    const Translation v0B{ std::get<1>(triangleB) };
    const Translation v1B{ std::get<2>(triangleB) };

    Eigen::Matrix<double, 3, 3> cA_H_v0A, cA_H_v1A;
    Eigen::Matrix<double, 3, 3> cB_H_v0B, cB_H_v1B;
    const Translation crossA{ gtsam::cross(v0A, v1A, cA_H_v0A, cA_H_v1A) };
    const Translation crossB{ gtsam::cross(v0B, v1B, cB_H_v0B, cB_H_v1B) };

    Eigen::Matrix<double, 3, 3> cAB_H_cA, cAB_H_cB;
    const Translation crossAB{ gtsam::cross(crossA, crossB, cAB_H_cA, cAB_H_cB) };

    // DEBUG_VARS(pA.transpose());
    // DEBUG_VARS(v0A.transpose());
    // DEBUG_VARS(v1A.transpose());
    // DEBUG_VARS(crossA.transpose());

    // DEBUG_VARS(pB.transpose());
    // DEBUG_VARS(v0B.transpose());
    // DEBUG_VARS(v1B.transpose());
    // DEBUG_VARS(crossB.transpose());

    // DEBUG_VARS(crossAB.transpose());
    if (Hxi)
    {
      *Hxi = cAB_H_cA * cA_H_v0A * v0A_H_xi     // no-lint
             + cAB_H_cA * cA_H_v1A * v1A_H_xi   // no-lint
             + cAB_H_cB * cB_H_v0B * v0B_H_xi   // no-lint
             + cAB_H_cB * cB_H_v1B * v1B_H_xi;  // no-lint
    }
    if (Hxj)
    {
      *Hxj = cAB_H_cA * cA_H_v0A * v0A_H_xj     // no-lint
             + cAB_H_cA * cA_H_v1A * v1A_H_xj   // no-lint
             + cAB_H_cB * cB_H_v0B * v0B_H_xj   // no-lint
             + cAB_H_cB * cB_H_v1B * v1B_H_xj;  // no-lint
    }
    if (Hxk)
    {
      *Hxk = cAB_H_cA * cA_H_v0A * v0A_H_xk     // no-lint
             + cAB_H_cA * cA_H_v1A * v1A_H_xk   // no-lint
             + cAB_H_cB * cB_H_v0B * v0B_H_xk   // no-lint
             + cAB_H_cB * cB_H_v1B * v1B_H_xk;  // no-lint
    }

    return crossAB;
  }

  static Error aligned_triangle(const SE3& xi, const SE3& xj, const SE3& xk,      // no-lint
                                const Translation& offset, const Rotation& Roff,  // no-lint
                                gtsam::OptionalJacobian<3, 6> Hxi = boost::none,  // no-lint
                                gtsam::OptionalJacobian<3, 6> Hxj = boost::none,  // no-lint
                                gtsam::OptionalJacobian<3, 6> Hxk = boost::none)
  {
    const bool compute_deriv{ Hxi or Hxj or Hxk };
    Eigen::Matrix<double, 3, 6> v0A_H_xi, v1A_H_xi, v0A_H_xj, v1A_H_xj, v0A_H_xk, v1A_H_xk;
    const Triangle triangleA{ get_triangle(xi, xj, xk, offset, Roff,  // no-lint
                                           boost::none,               // no-lint
                                           compute_deriv ? &v0A_H_xi : nullptr, compute_deriv ? &v1A_H_xi : nullptr,
                                           compute_deriv ? &v0A_H_xj : nullptr, compute_deriv ? &v1A_H_xj : nullptr,
                                           compute_deriv ? &v0A_H_xk : nullptr, compute_deriv ? &v1A_H_xk : nullptr) };

    const Translation pA{ std::get<0>(triangleA) };
    const Translation v0A{ std::get<1>(triangleA) };
    const Translation v1A{ std::get<2>(triangleA) };

    Eigen::Matrix<double, 3, 3> u0A_H_v0A, u1A_H_v1A;

    const Translation u0A{ gtsam::normalize(v0A, u0A_H_v0A) };
    const Translation u1A{ gtsam::normalize(v1A, u1A_H_v1A) };

    const Translation uoffset{ gtsam::normalize(offset) };

    Eigen::Matrix<double, 3, 3> cA_H_u0A, cA_H_u1A;
    const Translation crossA{ gtsam::cross(u0A, u1A, cA_H_u0A, cA_H_u1A) };

    Eigen::Matrix<double, 3, 3> cAB_H_cA;
    const Translation crossAB{ gtsam::cross(crossA, uoffset, cAB_H_cA) };

    // DEBUG_VARS(pA.transpose());
    // DEBUG_VARS(v0A.transpose());
    // DEBUG_VARS(v1A.transpose());
    // DEBUG_VARS(u0A.transpose());
    // DEBUG_VARS(u1A.transpose());
    // DEBUG_VARS(uoffset.transpose());
    // DEBUG_VARS(crossA.transpose());

    // DEBUG_VARS(crossAB.transpose());
    if (Hxi)
    {
      *Hxi = cAB_H_cA * cA_H_u0A * u0A_H_v0A * v0A_H_xi     // no-lint
             + cAB_H_cA * cA_H_u1A * u1A_H_v1A * v1A_H_xi;  // no-lint
    }
    if (Hxj)
    {
      *Hxj = cAB_H_cA * cA_H_u0A * u0A_H_v0A * v0A_H_xj     // no-lint
             + cAB_H_cA * cA_H_u1A * u1A_H_v1A * v1A_H_xj;  // no-lint
    }
    if (Hxk)
    {
      *Hxk = cAB_H_cA * cA_H_u0A * u0A_H_v0A * v0A_H_xk     // no-lint
             + cAB_H_cA * cA_H_u1A * u1A_H_v1A * v1A_H_xk;  // no-lint
    }

    return crossAB;
  }
  virtual Eigen::VectorXd evaluateError(const SE3& xi, const SE3& xj, const SE3& xk,
                                        boost::optional<Eigen::MatrixXd&> Hxi = boost::none,
                                        boost::optional<Eigen::MatrixXd&> Hxj = boost::none,
                                        boost::optional<Eigen::MatrixXd&> Hxk = boost::none) const override
  {
    if (_factor_type == TriangleFactorType::ForceTriangle)
    {
      return force_triangle(xi, xj, xk, _offset, _Roff, Hxi, Hxj, Hxk);
    }
    else if (_factor_type == TriangleFactorType::ScaleEquivalent)
    {
      return scale_equivalent(xi, xj, xk, _offset, _Roff, Hxi, Hxj, Hxk);
    }
    else if (_factor_type == TriangleFactorType::ParallelTriangles)
    {
      return aligned_triangle(xi, xj, xk, _offset, _Roff, Hxi, Hxj, Hxk);
    }
    else
    {
      TENSEGRITY_THROW("Unknown option for TriangleFactor");
    }
  }

  void print(const std::string& s, const gtsam::KeyFormatter& keyFormatter) const override
  {
    const std::string key_xi{ keyFormatter(this->template key<1>()) };
    const std::string key_xj{ keyFormatter(this->template key<2>()) };
    const std::string key_xk{ keyFormatter(this->template key<3>()) };

    // const std::string plant_name{ _prx_system->get_pathname() };
    std::cout << s << " ";
    std::cout << "[ " << key_xi << " " << key_xj << " " << key_xk << "]\n";
    std::cout << "\t offset: " << _offset.transpose() << "\n";
    if (_factor_type == TriangleFactorType::ForceTriangle)
    {
      std::cout << "\t type: " << "ForceTriangle" << "\n";
    }
    else if (_factor_type == TriangleFactorType::ScaleEquivalent)
    {
      std::cout << "\t type: " << "ScaleEquivalent" << "\n";
    }
    else if (_factor_type == TriangleFactorType::ParallelTriangles)
    {
      std::cout << "\t type: " << "ParallelTriangles" << "\n";
    }
    // std::cout << "\t Roff: " << _Roff << "\n";

    // if (this->noiseModel_)
    //   this->noiseModel_->print("  noise model: ");
    // else
    //   std::cout << "no noise model" << std::endl;
    std::cout << "\n";
  }

private:
  const Rotation _Roff;
  const Translation _offset;
  const TriangleFactorType _factor_type;
};

class tensegrity_triangles_aligned_factor_t
  : public gtsam::NoiseModelFactorN<gtsam::Pose3, gtsam::Rot3, gtsam::Pose3, gtsam::Rot3, gtsam::Pose3, gtsam::Rot3>
{
public:
  using SE3 = gtsam::Pose3;
  using Translation = Eigen::Vector<double, 3>;
  using Rotation = gtsam::Rot3;
  using SkewMatrix = Eigen::Matrix<double, 3, 3>;
  using Base = gtsam::NoiseModelFactor1<SE3, Rotation, SE3, Rotation, SE3, Rotation>;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;
  using Jacobian = Eigen::Matrix<double, 3, 6>;
  using Meassurement = Eigen::Vector<double, 1>;
  using Error = Eigen::VectorXd;

  // using BarFactor = bar_two_observations_factor_t;
  using BarFactor = endcap_observation_factor_t;
  // using Vector = Eigen::Vector<double, Dim>;
  using Triangle = std::tuple<gtsam::Point3, Translation, Translation>;

  template <typename RotationType>
  tensegrity_triangles_aligned_factor_t(const gtsam::Key key_Xi, const gtsam::Key key_Ri,  // no-lint
                                        const gtsam::Key key_Xj, const gtsam::Key key_Rj,  // no-lint
                                        const gtsam::Key key_Xk, const gtsam::Key key_Rk, const Translation offset,
                                        const RotationType Roff, const NoiseModel& cost_model)
    : Base(cost_model, key_Xi, key_Ri, key_Xj, key_Rj, key_Xk, key_Rk), _offset(offset), _Roff(Roff)
  {
  }

  static Triangle get_triangle(const SE3& xi, const SE3& xj, const SE3& xk,                 // no-lint
                               const Rotation& Ri, const Rotation& Rj, const Rotation& Rk,  // no-lint
                               const Translation& offset, const Rotation& Roff,             // no-lint
                               gtsam::OptionalJacobian<3, 6> p_H_xi = boost::none,          // no-lint
                               gtsam::OptionalJacobian<3, 3> p_H_Ri = boost::none,          // no-lint
                               gtsam::OptionalJacobian<3, 6> v0_H_xi = boost::none,         // no-lint
                               gtsam::OptionalJacobian<3, 6> v1_H_xi = boost::none,         // no-lint
                               gtsam::OptionalJacobian<3, 3> v0_H_Ri = boost::none,         // no-lint
                               gtsam::OptionalJacobian<3, 3> v1_H_Ri = boost::none,         // no-lint
                               gtsam::OptionalJacobian<3, 6> v0_H_xj = boost::none,         // no-lint
                               gtsam::OptionalJacobian<3, 6> v1_H_xj = boost::none,         // no-lint
                               gtsam::OptionalJacobian<3, 3> v0_H_Rj = boost::none,         // no-lint
                               gtsam::OptionalJacobian<3, 3> v1_H_Rj = boost::none,         // no-lint
                               gtsam::OptionalJacobian<3, 6> v0_H_xk = boost::none,         // no-lint
                               gtsam::OptionalJacobian<3, 6> v1_H_xk = boost::none,         // no-lint
                               gtsam::OptionalJacobian<3, 3> v0_H_Rk = boost::none,         // no-lint
                               gtsam::OptionalJacobian<3, 3> v1_H_Rk = boost::none)         // no-lint

  {
    Eigen::Matrix<double, 3, 6> pi_H_xi, pj_H_xj, pk_H_xk;
    Eigen::Matrix<double, 3, 3> pi_H_Ri, pj_H_Rj, pk_H_Rk;
    const bool compute_deriv{ p_H_xi or v0_H_xi or v1_H_xi or v0_H_xj or v1_H_xj or v0_H_xk or v1_H_xk };

    const Translation pt_i{ BarFactor::predict(xi, Ri, offset,                      // no-lint
                                               compute_deriv ? &pi_H_xi : nullptr,  // no-lint
                                               compute_deriv ? &pi_H_Ri : nullptr) };
    const Translation pt_j{ BarFactor::predict(xj, Rj, offset,  // no-lint
                                               compute_deriv ? &pj_H_xj : nullptr,
                                               compute_deriv ? &pj_H_Rj : nullptr) };
    const Translation pt_k{ BarFactor::predict(xk, Rk, offset,  // no-lint
                                               compute_deriv ? &pk_H_xk : nullptr,
                                               compute_deriv ? &pk_H_Rk : nullptr) };

    const Translation v0{ pt_j - pt_i };
    const Translation v1{ pt_k - pt_i };

    if (compute_deriv)
    {
      const Eigen::Matrix3d v0_H_pj{ Eigen::Matrix3d::Identity() };
      const Eigen::Matrix3d v0_H_pi{ -Eigen::Matrix3d::Identity() };
      const Eigen::Matrix3d v0_H_pk{ Eigen::Matrix3d::Zero() };
      const Eigen::Matrix3d v1_H_pi{ -Eigen::Matrix3d::Identity() };
      const Eigen::Matrix3d v1_H_pj{ Eigen::Matrix3d::Zero() };
      const Eigen::Matrix3d v1_H_pk{ Eigen::Matrix3d::Identity() };
      if (p_H_xi)
      {
        *p_H_xi = pi_H_xi;
      }
      if (p_H_Ri)
      {
        *p_H_Ri = pi_H_Ri;
      }
      if (v0_H_xi)
      {
        *v0_H_xi = v0_H_pi * pi_H_xi;
      }
      if (v1_H_xi)
      {
        *v1_H_xi = v1_H_pi * pi_H_xi;
      }
      if (v0_H_Ri)
      {
        *v0_H_Ri = v0_H_pi * pi_H_Ri;
      }
      if (v1_H_Ri)
      {
        *v1_H_Ri = v1_H_pi * pi_H_Ri;
      }
      if (v0_H_xj)
      {
        *v0_H_xj = v0_H_pj * pj_H_xj;
      }
      if (v1_H_xj)
      {
        *v1_H_xj = v1_H_pj * pj_H_xj;
      }
      if (v0_H_Rj)
      {
        *v0_H_Rj = v0_H_pj * pj_H_Rj;
      }
      if (v1_H_Rj)
      {
        *v1_H_Rj = v1_H_pj * pj_H_Rj;
      }
      if (v0_H_xk)
      {
        *v0_H_xk = v0_H_pk * pk_H_xk;
      }
      if (v1_H_xk)
      {
        *v1_H_xk = v1_H_pk * pk_H_xk;
      }
      if (v0_H_Rk)
      {
        *v0_H_Rk = v0_H_pk * pk_H_Rk;
      }
      if (v1_H_Rk)
      {
        *v1_H_Rk = v1_H_pk * pk_H_Rk;
      }
    }

    return { pt_i, v0, v1 };
  }

  static Error aligned_triangle(const SE3& xi, const SE3& xj, const SE3& xk,                 // no-lint
                                const Rotation& Ri, const Rotation& Rj, const Rotation& Rk,  // no-lint
                                const Translation& offset, const Rotation& Roff,             // no-lint
                                gtsam::OptionalJacobian<3, 6> Hxi = boost::none,             // no-lint
                                gtsam::OptionalJacobian<3, 6> Hxj = boost::none,             // no-lint
                                gtsam::OptionalJacobian<3, 6> Hxk = boost::none,             // no-lint
                                gtsam::OptionalJacobian<3, 3> HRi = boost::none,             // no-lint
                                gtsam::OptionalJacobian<3, 3> HRj = boost::none,             // no-lint
                                gtsam::OptionalJacobian<3, 3> HRk = boost::none)
  {
    const bool compute_deriv{ Hxi or Hxj or Hxk };
    Eigen::Matrix<double, 3, 6> pAi_H_xi, v0A_H_xi, v1A_H_xi, v0A_H_xj, v1A_H_xj, v0A_H_xk, v1A_H_xk;
    Eigen::Matrix<double, 3, 3> pAi_H_Ri, v0A_H_Ri, v1A_H_Ri, v0A_H_Rj, v1A_H_Rj, v0A_H_Rk, v1A_H_Rk;
    const Triangle triangleA{ get_triangle(xi, xj, xk,                // no-lint
                                           Ri, Rj, Rk, offset, Roff,  // no-lint
                                           pAi_H_xi, pAi_H_Ri,        // no-lint
                                           compute_deriv ? &v0A_H_xi : nullptr, compute_deriv ? &v1A_H_xi : nullptr,
                                           compute_deriv ? &v0A_H_Ri : nullptr, compute_deriv ? &v1A_H_Ri : nullptr,
                                           compute_deriv ? &v0A_H_xj : nullptr, compute_deriv ? &v1A_H_xj : nullptr,
                                           compute_deriv ? &v0A_H_Rj : nullptr, compute_deriv ? &v1A_H_Rj : nullptr,
                                           compute_deriv ? &v0A_H_xk : nullptr, compute_deriv ? &v1A_H_xk : nullptr,
                                           compute_deriv ? &v0A_H_Rk : nullptr, compute_deriv ? &v1A_H_Rk : nullptr) };
    Eigen::Matrix<double, 3, 6> pBk_H_xk;
    Eigen::Matrix<double, 3, 3> pBk_H_Rk;
    const Translation pB_k{ BarFactor::predict(xk, Rk * Roff, offset,  // no-lint
                                               compute_deriv ? &pBk_H_xk : nullptr,
                                               compute_deriv ? &pBk_H_Rk : nullptr) };

    const Translation pA_i{ std::get<0>(triangleA) };
    const Translation v0A{ std::get<1>(triangleA) };
    const Translation v1A{ std::get<2>(triangleA) };

    const Eigen::Matrix<double, 3, 3> vik_H_pAi{ Eigen::Matrix<double, 3, 3>::Identity() };
    const Eigen::Matrix<double, 3, 3> vik_H_pBk{ -Eigen::Matrix<double, 3, 3>::Identity() };
    const Translation vik{ pA_i - pB_k };

    Eigen::Matrix<double, 3, 3> u0A_H_v0A, u1A_H_v1A;

    const Translation u0A{ gtsam::normalize(v0A, u0A_H_v0A) };
    const Translation u1A{ gtsam::normalize(v1A, u1A_H_v1A) };

    Eigen::Matrix<double, 3, 3> uik_H_vik;
    const Translation uik{ gtsam::normalize(vik, uik_H_vik) };

    Eigen::Matrix<double, 3, 3> cA_H_u0A, cA_H_u1A;
    const Translation crossA{ gtsam::cross(u0A, u1A, cA_H_u0A, cA_H_u1A) };

    Eigen::Matrix<double, 3, 3> cAB_H_cA, cAB_H_uik;
    const Translation crossAB{ gtsam::cross(crossA, uik, cAB_H_cA, cAB_H_uik) };

    // DEBUG_VARS(pA_i.transpose());
    // DEBUG_VARS(v0A.transpose());
    // DEBUG_VARS(v1A.transpose());
    // DEBUG_VARS(u0A.transpose());
    // DEBUG_VARS(u1A.transpose());
    // DEBUG_VARS(vik.transpose());
    // DEBUG_VARS(uik.transpose());
    // DEBUG_VARS(crossA.transpose());

    // DEBUG_VARS(crossAB.transpose());
    // Ri.print("Ri");
    // Rk.print("Rk");
    if (Hxi)
    {
      *Hxi = cAB_H_uik * uik_H_vik * vik_H_pAi * pAi_H_xi   // no-lint
             + cAB_H_cA * cA_H_u0A * u0A_H_v0A * v0A_H_xi   // no-lint
             + cAB_H_cA * cA_H_u1A * u1A_H_v1A * v1A_H_xi;  // no-lint
    }
    if (HRi)
    {
      *HRi = cAB_H_uik * uik_H_vik * vik_H_pAi * pAi_H_Ri   // no-lint
             + cAB_H_cA * cA_H_u0A * u0A_H_v0A * v0A_H_Ri   // no-lint
             + cAB_H_cA * cA_H_u1A * u1A_H_v1A * v1A_H_Ri;  // no-lint
    }
    if (Hxj)
    {
      *Hxj = cAB_H_cA * cA_H_u0A * u0A_H_v0A * v0A_H_xj     // no-lint
             + cAB_H_cA * cA_H_u1A * u1A_H_v1A * v1A_H_xj;  // no-lint
    }
    if (HRj)
    {
      *HRj = cAB_H_cA * cA_H_u0A * u0A_H_v0A * v0A_H_Rj     // no-lint
             + cAB_H_cA * cA_H_u1A * u1A_H_v1A * v1A_H_Rj;  // no-lint
    }
    if (Hxk)
    {
      *Hxk = cAB_H_uik * uik_H_vik * vik_H_pBk * pBk_H_xk   // no-lint
             + cAB_H_cA * cA_H_u0A * u0A_H_v0A * v0A_H_xk   // no-lint
             + cAB_H_cA * cA_H_u1A * u1A_H_v1A * v1A_H_xk;  // no-lint
    }
    if (HRk)
    {
      *HRk = cAB_H_uik * uik_H_vik * vik_H_pBk * pBk_H_Rk   // no-lint
             + cAB_H_cA * cA_H_u0A * u0A_H_v0A * v0A_H_Rk   // no-lint
             + cAB_H_cA * cA_H_u1A * u1A_H_v1A * v1A_H_Rk;  // no-lint
    }

    return crossAB;
  }
  virtual Eigen::VectorXd evaluateError(const SE3& xi, const Rotation& Ri,  // no-lint
                                        const SE3& xj, const Rotation& Rj,  // no-lint
                                        const SE3& xk, const Rotation& Rk,  // no-lint
                                        boost::optional<Eigen::MatrixXd&> Hxi = boost::none,
                                        boost::optional<Eigen::MatrixXd&> HRi = boost::none,
                                        boost::optional<Eigen::MatrixXd&> Hxj = boost::none,
                                        boost::optional<Eigen::MatrixXd&> HRj = boost::none,
                                        boost::optional<Eigen::MatrixXd&> Hxk = boost::none,
                                        boost::optional<Eigen::MatrixXd&> HRk = boost::none) const override
  {
    return aligned_triangle(xi, xj, xk, Ri, Rj, Rk, _offset, _Roff, Hxi, Hxj, Hxk, HRi, HRj, HRk);
  }

  void print(const std::string& s, const gtsam::KeyFormatter& keyFormatter) const override
  {
    const std::string key_xi{ keyFormatter(this->template key<1>()) };
    const std::string key_Ri{ keyFormatter(this->template key<2>()) };
    const std::string key_xj{ keyFormatter(this->template key<3>()) };
    const std::string key_Rj{ keyFormatter(this->template key<4>()) };
    const std::string key_xk{ keyFormatter(this->template key<5>()) };
    const std::string key_Rk{ keyFormatter(this->template key<6>()) };
    // const std::string plant_name{ _prx_system->get_pathname() };
    std::cout << s << "Aligned triangles:";
    std::cout << "[ " << key_xi << " " << key_xj << " " << key_xk << " " << key_Ri << " " << key_Rj << " " << key_Rk
              << "]\n";
    std::cout << "\t offset: " << _offset.transpose() << "\n";

    // std::cout << "\t type: " << "ParallelTriangles" << "\n";

    // std::cout << "\t Roff: " << _Roff << "\n";

    // if (this->noiseModel_)
    //   this->noiseModel_->print("  noise model: ");
    // else
    //   std::cout << "no noise model" << std::endl;
    std::cout << "\n";
  }

private:
  const Rotation _Roff;
  const Translation _offset;
};
}  // namespace estimation
