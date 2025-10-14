#pragma once
#include <array>
#include <numeric>
#include <gtsam/config.h>
#include <gtsam/base/Testable.h>
#include <gtsam/nonlinear/Expression.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace factor_graphs
{
template <typename Cmp>
struct DoubleCmp
{
  using Jacobian = Eigen::Matrix<double, 1, 1>;
  using OptionalMatrix = boost::optional<Eigen::MatrixXd&>;

  constexpr bool operator()(const double& lhs, const double& rhs) const
  {
    return _cmp(lhs, rhs);
  }

  Eigen::VectorXd error(const double& lhs, const double& rhs, OptionalMatrix H0 = boost::none) const
  {
    const Eigen::VectorXd error{ Eigen::Vector<double, 1>(lhs - rhs) };

    if (H0)
    {
      *H0 = Jacobian::Identity();
    }
    return error;
  }

private:
  Cmp _cmp;
};

template <typename Vector>
struct VectorLessThanCmp
{
  static constexpr Eigen::Index Dim{ Vector::RowsAtCompileTime };
  using Jacobian = Eigen::DiagonalMatrix<double, Dim, Dim>;
  using OptionalMatrix = boost::optional<Eigen::MatrixXd&>;

  constexpr bool operator()(const Vector& lhs, const Vector& rhs) const
  {
    // const Vector comp{  };
    return (lhs.array() < rhs.array()).any();
  }

  Eigen::VectorXd error(const Vector& lhs, const Vector& rhs, OptionalMatrix H0 = boost::none) const
  {
    // const Vector error{ (lhs.array() < rhs.array()).matrix().template cast<double>() };

    const Eigen::ArrayXd lesser_array{ (lhs.array() < rhs.array()).template cast<double>() };
    const Vector error{ (lhs.array() * lesser_array).matrix().template cast<double>() };

    // const Vector error{ error_array.matrix().template cast<double>() };

    if (H0)
    {
      *H0 = -1.0 * Jacobian(error);
    }
    return error;
  }
};

template <typename Vector>
struct VectorGreaterThanCmp
{
  static constexpr Eigen::Index Dim{ Vector::RowsAtCompileTime };
  using Jacobian = Eigen::DiagonalMatrix<double, Dim, Dim>;
  using OptionalMatrix = boost::optional<Eigen::MatrixXd&>;

  constexpr bool operator()(const Vector& lhs, const Vector& rhs) const
  {
    // const Vector comp{  };
    return (lhs.array() > rhs.array()).any();
  }

  Eigen::VectorXd error(const Vector& lhs, const Vector& rhs, OptionalMatrix H0 = boost::none) const
  {
    const Eigen::ArrayXd greater_array{ (lhs.array() > rhs.array()).template cast<double>() };
    const Vector error{ (lhs.array() * greater_array).matrix().template cast<double>() };

    // const Vector error{ error_array.matrix().template cast<double>() };

    if (H0)
    {
      *H0 = -1.0 * Jacobian(error);
    }
    return error;
  }
};

// ComparisonFn: Returns true when the constrain is NOT satisfied.
//               Activate when: (value *FN* constraint) is true. eg: *value* < (constraint=0) (using std::less)
template <typename Type, typename ComparisonFn>
class constraint_factor_t : public gtsam::NoiseModelFactor1<Type>
{
  using Base = gtsam::NoiseModelFactor1<Type>;
  using Derived = constraint_factor_t;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;

  static constexpr Eigen::Index Dim{ gtsam::traits<Type>::dimension };

  using OptionalMatrix = boost::optional<Eigen::MatrixXd&>;
  using ErrorVector = Eigen::Vector<double, Dim>;
  using Jacobian = Eigen::Matrix<double, Dim, Dim>;

public:
  constraint_factor_t(const gtsam::Key& key, const Type& constrain, const NoiseModel& cost_model = nullptr)
    : Base(cost_model, key), _constrain(constrain), _comparison()
  {
  }

  virtual bool active(const gtsam::Values& values) const override
  {
    const Type val{ values.at<Type>(this->template key<1>()) };
    const bool activated{ _comparison(val, _constrain) };
    return activated;
  }

  template <typename Floating, std::enable_if_t<std::is_floating_point<Floating>::value, bool> = true>
  Eigen::VectorXd compute_error(const Type& x0, OptionalMatrix H0 = boost::none) const
  {
    const Type err{ _constrain - x0 };
    const ErrorVector error{ ErrorVector(err) };
    if (H0)
    {
      *H0 = -err * Jacobian::Identity();
    }
    return error;
  }

  template <typename Floating, std::enable_if_t<not std::is_floating_point<Floating>::value, bool> = true>
  Eigen::VectorXd compute_error(const Type& x0, OptionalMatrix H0 = boost::none) const
  {
    return _comparison.error(x0, _constrain, H0);
  }

  // Error and jacobian is in the same dim as Type. Could be parametrized somehow
  virtual Eigen::VectorXd evaluateError(const Type& x0, OptionalMatrix H0 = boost::none) const override
  {
    return compute_error<Type>(x0, H0);
  }

private:
  Type _constrain;
  ComparisonFn _comparison;
};
}  // namespace factor_graphs
