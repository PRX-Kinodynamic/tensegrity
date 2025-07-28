#pragma once
#include <array>
#include <numeric>
#include <functional>

#include <gtsam/config.h>
#include <gtsam/base/Testable.h>
#include <gtsam/nonlinear/Expression.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include "prx/factor_graphs/utilities/symbols_factory.hpp"
#include "prx/factor_graphs/lie_groups/lie_integrator.hpp"
#include <utils/rosparams_utils.hpp>
#include <utils/dbg_utils.hpp>

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
  using LieIntegrator = prx::fg::lie_integrator_t<SE3, Velocity>;

  using Translation = Eigen::Vector<double, 3>;
  using SF = prx::fg::symbol_factory_t;

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
    // Eigen::Matrix<double, DimX, DimX> xdt_H_x, xdt_H_xdot;
    // const Eigen::Vector<double, 6> vel{ 0, 0, 0, xdot[0], xdot[1], xdot[2] };
    // const SE3 x_dt{ LieIntegrator::integrate(x, vel, dt, Hx ? &xdt_H_x : nullptr, Hxdot ? &xdt_H_xdot : nullptr) };
    // const Translation cap_prediction{ x_dt.action(offset, p_H_xdt) };
    const Translation cap_prediction{ x.transformTo(offset, Hx) };
    // if (Hx)
    // {
    //   *Hx = p_H_x;
    // }
    // if (Hxdot)
    // {
    //   *Hxdot = p_H_xdt * xdt_H_xdot;
    // }
    return cap_prediction;
  }

  virtual bool active(const gtsam::Values& values) const override
  {
    const bool activated{ _dt > 0 };
    // DEBUG_VARS(activated);
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
    const char sp{ prx::constants::separating_value };

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
  // using SE3 = prx::fg::se3_t;
  using Velocity = Eigen::Vector<double, 3>;
  using Base = gtsam::NoiseModelFactorN<SE3, Velocity>;
  using Derived = tensegrity_cap_observation_t;

  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;

  using OptDeriv = boost::optional<Eigen::MatrixXd&>;
  using LieIntegrator = prx::fg::lie_integrator_t<SE3, Velocity>;

  using Translation = Eigen::Vector<double, 3>;
  using SF = prx::fg::symbol_factory_t;

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
    // const Translation cap_prediction{ x_dt.action(offset, p_H_xdt) };
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
    // DEBUG_VARS(activated);
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
    const char sp{ prx::constants::separating_value };

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
}  // namespace estimation