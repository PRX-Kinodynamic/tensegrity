#pragma once
// #define EIGEN_MAX_ALIGN_BYTES = 64
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/basis/FitBasis.h>
#include <gtsam/basis/Chebyshev2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/basis/Chebyshev.h>

namespace estimation
{
template <typename ChebType>
class endcaps_cheb_factor_t : public gtsam::NoiseModelFactorN<gtsam::ParameterMatrix<6>>
{
public:
  using SE3 = gtsam::Pose3;
  using ParameterMatrix = gtsam::ParameterMatrix<6>;
  using Translation = Eigen::Vector<double, 3>;
  using Rotation = gtsam::Rot3;
  using Base = gtsam::NoiseModelFactorN<gtsam::ParameterMatrix<6>>;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;
  using Error = Eigen::Vector<double, 1>;
  using ManifoldFunctor = typename ChebType::template ManifoldEvaluationFunctor<SE3>;
  // using BarFactor = bar_two_observations_factor_t;
  // using BarFactor = endcap_observation_factor_t;
  // using Vector = Eigen::Vector<double, Dim>;

  endcaps_cheb_factor_t(const gtsam::Key key_pm, const Translation offset, const Translation& zi, const int N,
                        const double ti, const double a, const double b, const NoiseModel& cost_model)
    : Base(cost_model, key_pm), _zi(zi), _func(N, ti, a, b), _offset(offset)
  {
  }

  static Translation predict(const ParameterMatrix& pm, const Translation& offset,  // no-lint
                             const ManifoldFunctor& func,                           // no-lint
                             gtsam::OptionalJacobian<-1, -1> Hpm = boost::none)
  {
    Eigen::MatrixXd x_H_pm;
    Eigen::Matrix<double, 3, 6> pred_H_x;

    const SE3 x{ func(pm, Hpm ? &x_H_pm : nullptr) };

    const Translation pred{ x.transformFrom(offset, pred_H_x) };

    if (Hpm)
    {
      *Hpm = pred_H_x * x_H_pm;
    }

    return pred;
  }
  virtual bool sendable() const override
  {
    return false;
  }
  virtual Eigen::VectorXd evaluateError(const ParameterMatrix& pm,  // no-lint
                                        boost::optional<Eigen::MatrixXd&> Hpm = boost::none) const override
  {
    const Translation prediction{ predict(pm, _offset, _func, Hpm) };

    const Translation err{ prediction - _zi };

    return err;
  }

  void print(const std::string& s, const gtsam::KeyFormatter& keyFormatter) const override
  {
    const std::string key_xi{ keyFormatter(this->template key<1>()) };

    // const std::string plant_name{ _prx_system->get_pathname() };
    std::cout << s << " ";
    std::cout << "[ " << key_xi << " ]\n";
    std::cout << "\t zij: " << _zi.transpose() << "\n";

    if (this->noiseModel_)
      this->noiseModel_->print("  noise model: ");
    else
      std::cout << "no noise model" << std::endl;
    std::cout << "\n";
  }

private:
  const ManifoldFunctor _func;

  const Translation _zi;
  const Translation _offset;
};
}  // namespace estimation