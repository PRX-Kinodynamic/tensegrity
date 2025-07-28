#pragma once
#include <array>
#include <numeric>
#include <gtsam/config.h>
#include <gtsam/base/Testable.h>
#include <gtsam/nonlinear/Expression.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <prx/factor_graphs/lie_groups/se3.hpp>
#include <prx/factor_graphs/lie_groups/lie_integrator.hpp>

namespace estimation
{
// Endcap is observed P=(x,y,z) at time t+epsilon. The endcap's bar pose is q_t and velocity \dot{q}_t at time t.
// The prediction first "moves" q_t to q_{t+epsilon} using \dot{q} and then finds the position of the endcap given the
// offset.
class endcap_dynamic_observation_t : public gtsam::NoiseModelFactor1<prx::fg::se3_t, Eigen::Vector<double, 6>>
{
  using Base = gtsam::NoiseModelFactor1<prx::fg::se3_t, Eigen::Vector<double, 6>>;

public:
  using SE3 = prx::fg::se3_t;
  using Velocity = Eigen::Vector<double, 6>;
  using Translation = Eigen::Vector<double, 3>;
  using Observation = Eigen::Vector<double, 3>;
  using Rotation = Eigen::Matrix<double, 3, 3>;
  using SkewMatrix = Eigen::Matrix<double, 3, 3>;
  using Jacobian36 = Eigen::Matrix<double, 3, 6>;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;
  using LieIntegrator = prx::fg::lie_integrator_t<SE3, Velocity>;

  // using Vector = Eigen::Vector<double, Dim>;
  static constexpr Eigen::Index DimX{ 6 };
  static constexpr Eigen::Index DimXdot{ 6 };
  static constexpr Eigen::Index DimZ{ 3 };

  endcap_dynamic_observation_t(const gtsam::Key kse3, const gtsam::Key kvel, const Translation offset,
                               const Observation z, const double dt, const NoiseModel& cost_model)
    : Base(cost_model, kse3, kvel), _offset(offset), _z(z), _dt(dt)
  {
  }

  virtual bool active(const gtsam::Values& values) const override
  {
    const bool activated{ _dt > 0 };
    return activated;
  }

  static Translation predict(const SE3& x, const Velocity& xdot, const double& dt, const Translation& offset,
                             gtsam::OptionalJacobian<DimZ, DimX> Hx = boost::none,
                             gtsam::OptionalJacobian<DimZ, DimXdot> Hxdot = boost::none)
  {
    Eigen::Matrix<double, DimX, DimX> xt_H_x;      // Deriv error wrt between
    Eigen::Matrix<double, DimX, DimXdot> xt_H_xd;  // Deriv error wrt between
    const SE3 xt{ LieIntegrator::integrate(x, xdot, dt, xt_H_x, xt_H_xd) };
    const Translation endcap{ xt * offset };

    if (Hx or Hxdot)
    {
      Jacobian36 pr_H_xt{ Jacobian36::Zero() };  // Deriv error wrt between
      const Rotation R{ xt.rotation_matrix() };
      const SkewMatrix sk{ gtsam::skewSymmetric(-offset[0], -offset[1], -offset[2]) };
      pr_H_xt.leftCols<3>() = R * sk;
      pr_H_xt.rightCols<3>() = R;
      if (Hx)
      {
        *Hx = pr_H_xt * xt_H_x;
      }
      if (Hxdot)
      {
        *Hxdot = pr_H_xt * xt_H_xd;
      }
    }
    return endcap;
  }

  virtual Eigen::VectorXd evaluateError(const SE3& x, const Velocity& xdot,
                                        boost::optional<Eigen::MatrixXd&> Hx = boost::none,
                                        boost::optional<Eigen::MatrixXd&> Hxdot = boost::none) const override
  {
    const Translation pt_expected{ predict(x, xdot, _dt, _offset, Hx, Hxdot) };
    const Translation error{ pt_expected - _z };

    return error;
  }

private:
  Translation _offset;
  Translation _z;
  double _dt;
};

}  // namespace estimation