#pragma once

#include <Eigen/Dense>
#include <Eigen/Core>
#include <gtsam/base/OptionalJacobian.h>
// #include "prx/factor_graphs/lie_groups/lie_operators.hpp"
// #include "prx/factor_graphs/factors/noise_model_factors.hpp"
// #include "prx/utilities/math/first_order_derivative.hpp"

namespace factor_graphs
{
//
// Numerical integrator that does the equivalent of:
// x_{t+1} <- x_t + \dot{x} dt
// But using lie-groups operators, this is
// x_{t+1} = f(x_t, \dot{x}_t)  = x_t * expmap(\dot{x}_t * dt)
// For a given prediction \hat{x}_{t+1}, the error is then:
// f(x_t, \dot{x}_t) - \hat{x}_{t+1}
template <typename X, typename Xdot, typename Dt = double>
class lie_integrator_t
{
  using Model = lie_integrator_t<X, Xdot, Dt>;
  static constexpr Eigen::Index DimX{ gtsam::traits<X>::dimension };
  static constexpr Eigen::Index DimXdot{ gtsam::traits<Xdot>::dimension };

  using MatXX = Eigen::Matrix<double, DimX, DimX>;
  using MatXXdot = Eigen::Matrix<double, DimX, DimXdot>;
  using MatXdotXdot = Eigen::Matrix<double, DimXdot, DimXdot>;
  using MatXdotDt = Eigen::Matrix<double, DimXdot, 1>;

public:
  lie_integrator_t() = default;

  static X integrate(const X& x, const Xdot& xdot, const Dt& dt,  // no-lint
                     gtsam::OptionalJacobian<DimX, DimX> Hx = boost::none,
                     gtsam::OptionalJacobian<DimX, DimXdot> Hxdot = boost::none,
                     gtsam::OptionalJacobian<DimX, 1> Hdt = boost::none)
  {
    Eigen::Matrix<double, DimXdot, DimXdot> tau_H_qdot;
    Eigen::Matrix<double, DimXdot, 1> tau_H_dt;

    Eigen::Matrix<double, DimX, DimXdot> exp_H_tau;

    Eigen::Matrix<double, DimX, DimX> q1_H_exp;
    Eigen::Matrix<double, DimX, DimX> q1_H_q0;

    const Xdot xdot_dt{ xdot * dt };
    const X exmap_xdot_dt{ X::Expmap(xdot_dt, (Hxdot or Hdt) ? &exp_H_tau : nullptr) };
    const X xj{ gtsam::traits<X>::Compose(x, exmap_xdot_dt, Hx, (Hxdot or Hdt) ? &q1_H_exp : nullptr) };

    if (Hxdot)
    {
      tau_H_qdot = dt * Eigen::Matrix<double, DimXdot, DimXdot>::Identity();
      *Hxdot = q1_H_exp * exp_H_tau * tau_H_qdot;
    }

    if (Hdt)
    {
      tau_H_dt = xdot;
      *Hdt = q1_H_exp * exp_H_tau * tau_H_dt;
    }

    return xj;
  }
};

template <typename X, typename Xdot, typename... Types>
class lie_integration_factor_t : public gtsam::NoiseModelFactorN<X, X, Xdot, Types...>
{
  using Base = gtsam::NoiseModelFactor3<X, X, Xdot, Types...>;
  using LieIntegrator = lie_integrator_t<X, Xdot, Types...>;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;
  static constexpr Eigen::Index DimX{ gtsam::traits<X>::dimension };
  static constexpr Eigen::Index DimXdot{ gtsam::traits<Xdot>::dimension };
  static constexpr std::size_t NumTypes{ sizeof...(Types) };

  using DerivativeX = Eigen::Matrix<double, DimX, DimX>;
  using OptDeriv = boost::optional<Eigen::MatrixXd&>;
  template <typename T>
  using OptionalMatrix = boost::optional<Eigen::MatrixXd&>;

  using MatXX = Eigen::Matrix<double, DimX, DimX>;
  using MatXXdot = Eigen::Matrix<double, DimX, DimXdot>;
  using MatXdotXdot = Eigen::Matrix<double, DimXdot, DimXdot>;
  using MatXdotDt = Eigen::Matrix<double, DimXdot, 1>;

  lie_integration_factor_t() = delete;
  lie_integration_factor_t(const lie_integration_factor_t& other) = delete;

public:
  template <std::size_t Num = NumTypes, typename std::enable_if_t<(0 == Num), bool> = true>
  lie_integration_factor_t(const gtsam::Key key_xt1, const gtsam::Key key_xt0, const gtsam::Key key_xdot,
                           const NoiseModel& cost_model, const double h, const std::string label = "LieOdeIntegration")
    : Base(cost_model, key_xt1, key_xt0, key_xdot), _h(h), _label(label)
  {
  }

  template <std::size_t Num = NumTypes, typename std::enable_if_t<(1 == Num), bool> = true>
  lie_integration_factor_t(const gtsam::Key key_xt1, const gtsam::Key key_xt0, const gtsam::Key key_xdot,
                           const gtsam::Key key_dt, const NoiseModel& cost_model,
                           const std::string label = "LieOdeIntegration")
    : Base(cost_model, key_xt1, key_xt0, key_xdot, key_dt), _h(0.0), _label(label)
  {
  }

  ~lie_integration_factor_t() override
  {
  }

  template <typename Dt>
  static X predict(const X& x, const Xdot& xdot, const Dt dt,                   // no-lint
                   gtsam::OptionalJacobian<DimX, DimX> Hx = boost::none,        // no-lint
                   gtsam::OptionalJacobian<DimX, DimXdot> Hxdot = boost::none,  // no-lint
                   gtsam::OptionalJacobian<DimX, 1> Hdt = boost::none)
  {
    return LieIntegrator::integrate(x, xdot, dt, Hx, Hxdot, Hdt);
  }

  // x1_predicted <- x0 + xdot dt
  // Error is: x1_predicted - x1_observed
  template <typename Dt>
  static Eigen::VectorXd error(const X& x1, const X& x0, const Xdot& xdot, const Dt& dt,
                               boost::optional<Eigen::MatrixXd&> Hx1 = boost::none,
                               boost::optional<Eigen::MatrixXd&> Hx0 = boost::none,
                               boost::optional<Eigen::MatrixXd&> Hxdot = boost::none,
                               boost::optional<Eigen::MatrixXd&> Hdt = boost::none)
  {
    Eigen::Matrix<double, DimX, DimX> err_H_b;       // Deriv error wrt between
    Eigen::Matrix<double, DimX, DimX> b_H_q1;        // Deriv between wrt x1
    Eigen::Matrix<double, DimX, DimX> b_H_qp;        // Deriv between wrt predicted
    Eigen::Matrix<double, DimX, DimX> qp_H_q0;       // Deriv predicted wrt x0
    Eigen::Matrix<double, DimX, DimXdot> qp_H_qdot;  // Deriv predicted wrt xdot
    Eigen::Matrix<double, DimX, 1> qp_H_qdt;         // Deriv predicted wrt dt

    const X prediction{ predict(x0, xdot, dt,                  // no-lint
                                Hx0 ? &qp_H_q0 : nullptr,      // no-lint
                                Hxdot ? &qp_H_qdot : nullptr,  // no-lint
                                Hdt ? &qp_H_qdt : nullptr) };
    // X1_p (-) x1 => Eq. 26 from "A micro Lie theory [...]" https://arxiv.org/pdf/1812.01537.pdf
    const X between{ x1.between(prediction,                                 // no-lint
                                (Hx0 or Hxdot or Hdt) ? &b_H_q1 : nullptr,  // no-lint
                                (Hx0 or Hxdot or Hdt) ? &b_H_qp : nullptr) };
    const Eigen::VectorXd error{ X::Logmap(between, (Hx0 or Hxdot or Hdt) ? &err_H_b : nullptr) };

    if (Hx1)
    {
      *Hx1 = err_H_b * b_H_q1;
    }
    if (Hx0)
    {
      *Hx0 = err_H_b * b_H_qp * qp_H_q0;
    }
    if (Hxdot)
    {
      *Hxdot = err_H_b * b_H_qp * qp_H_qdot;
    }
    if (Hdt)
    {
      *Hdt = err_H_b * b_H_qp * qp_H_qdt;
    }

    return error;
  }

  virtual Eigen::VectorXd evaluateError(const X& x1, const X& x0, const Xdot& xdot, const Types&... xd,  // no-lint
                                        OptDeriv H1 = boost::none, OptDeriv H0 = boost::none,
                                        OptDeriv Hdot = boost::none, OptionalMatrix<Types>... H) const override
  {
    if constexpr (0 == NumTypes)
    {
      return error(x1, x0, xdot, _h, H1, H0, Hdot);
    }
    else
    {
      return error(x1, x0, xdot, xd..., H1, H0, Hdot, H...);
    }
  }

private:
  const double _h;
  const std::string _label;
};

}  // namespace factor_graphs
