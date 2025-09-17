#pragma once
#include <array>
#include <numeric>
#include <functional>

#include <gtsam/config.h>
#include <gtsam/base/Testable.h>
#include <gtsam/nonlinear/Expression.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <factor_graphs/symbols_factory.hpp>

namespace factor_graphs
{
template <typename X, typename Xdot, typename... Types>
class euler_integration_factor_t : public gtsam::NoiseModelFactorN<X, X, Xdot, Types...>
{
  using Base = gtsam::NoiseModelFactorN<X, X, Xdot, Types...>;
  using Derived = euler_integration_factor_t<X, Xdot, Types...>;

  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;

  static constexpr std::size_t NumTypes{ sizeof...(Types) };

  using OptDeriv = boost::optional<Eigen::MatrixXd&>;
  template <typename T>
  using OptionalMatrix = boost::optional<Eigen::MatrixXd&>;

  static constexpr Eigen::Index DimX{ gtsam::traits<X>::dimension };
  static constexpr Eigen::Index DimXdot{ gtsam::traits<Xdot>::dimension };

  euler_integration_factor_t() = delete;

public:
  euler_integration_factor_t(const euler_integration_factor_t& other) = delete;

  template <std::size_t Num = NumTypes, typename std::enable_if_t<(0 == Num), bool> = true>
  euler_integration_factor_t(const gtsam::Key key_xt1, const gtsam::Key key_xt0, const gtsam::Key key_xdot,
                             const NoiseModel& cost_model, const double h, const std::string label = "EulerIntegration")
    : Base(cost_model, key_xt1, key_xt0, key_xdot), _h(h), _label(label)
  {
  }

  template <std::size_t Num = NumTypes, typename std::enable_if_t<(1 == Num), bool> = true>
  euler_integration_factor_t(const gtsam::Key key_xt1, const gtsam::Key key_xt0, const gtsam::Key key_xdot,
                             const gtsam::Key key_dt, const NoiseModel& cost_model,
                             const std::string label = "EulerIntegration")
    : Base(cost_model, key_xt1, key_xt0, key_xdot, key_dt), _h(0.0), _label(label)
  {
  }

  ~euler_integration_factor_t() override
  {
  }

  template <typename Dt>
  static X integrate(const X& xi, const Xdot& xdot_i, const Dt& dt,  // no-lint
                     OptDeriv Hx = boost::none, OptDeriv Hxdot = boost::none, OptDeriv Hdt = boost::none)
  {
    return predict(xi, xdot_i, dt, Hx, Hxdot, Hdt);
  }

  template <typename Dt>
  static X predict(const X& x, const Xdot& xdot, const Dt& dt,  // no-lint
                   OptDeriv Hx = boost::none, OptDeriv Hxdot = boost::none, OptDeriv Hdt = boost::none)
  {
    // clang-format off
    if (Hx){ *Hx = Eigen::Matrix<double, DimX, DimX>::Identity(); }
    if (Hxdot){ *Hxdot = dt * Eigen::Matrix<double, DimXdot, DimXdot>::Identity(); }
    if (Hdt){ *Hdt = xdot; }
    // clang-format on
    return x + xdot * dt;
  }

  // x1_predicted <- x0 + xdot dt
  // Error is: x1_predicted - x1_observed
  template <typename Dt>
  static X error(const X& x1, const X& x0, const Xdot& xdot, const Dt& dt,  // no-lint
                 OptDeriv H1 = boost::none, OptDeriv H0 = boost::none, OptDeriv Hdot = boost::none,
                 OptDeriv Hdt = boost::none)
  {
    static constexpr Eigen::Index DimX{ gtsam::traits<X>::dimension };

    const X prediction{ predict(x0, xdot, dt, H0, Hdot, Hdt) };
    const X error{ prediction - x1 };

    // clang-format off
    if (H1) { *H1 = -1 * Eigen::Matrix<double, DimX, DimX>::Identity(); }
    // clang-format on

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

  template <typename Dt>
  void dt_to_stream(std::ostream& os, const gtsam::Values& values) const
  {
    const char sp{ ' ' };
    const gtsam::Key key_dt{ this->template key<4>() };
    const Dt dt{ values.at<Dt>(key_dt) };
    os << symbol_factory_t::formatter(key_dt) << sp << dt << sp;
  }

  void to_stream(std::ostream& os, const gtsam::Values& values) const
  {
    // const X& x1, const X& x0, const Xdot& xdot
    const X x1{ values.at<X>(this->template key<1>()) };
    const X x0{ values.at<X>(this->template key<2>()) };
    const Xdot xdot{ values.at<Xdot>(this->template key<3>()) };
    const char sp{ ' ' };
    os << _label << sp;
    os << symbol_factory_t::formatter(this->template key<1>()) << " " << x1.transpose() << sp;
    os << symbol_factory_t::formatter(this->template key<2>()) << " " << x0.transpose() << sp;
    os << symbol_factory_t::formatter(this->template key<3>()) << " " << xdot.transpose() << sp;
    if constexpr (0 == NumTypes)
    {
      os << "dt " << _h << sp;
    }
    else
    {
      dt_to_stream<Types...>(values);
    }
    os << "\n";
  }

private:
  const double _h;
  const std::string _label;
};

}  // namespace factor_graphs
