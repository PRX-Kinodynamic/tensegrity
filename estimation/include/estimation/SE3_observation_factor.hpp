#pragma once
#include <array>
#include <numeric>
#include <gtsam/config.h>
#include <gtsam/base/Testable.h>
#include <gtsam/nonlinear/Expression.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <factor_graphs/defs.hpp>

// #include "prx/factor_graphs/lie_groups/se3.hpp"
// #include "prx/factor_graphs/utilities/symbols_factory.hpp"

namespace estimation
{
template <typename SE3>
class SE3_observation_factor_t : public gtsam::NoiseModelFactor1<SE3>
{
public:
  // using SE3 = prx::fg::se3_t;
  using Translation = Eigen::Vector<double, 3>;
  using Observation = Eigen::Vector<double, 3>;
  using Rotation = Eigen::Matrix<double, 3, 3>;
  using SkewMatrix = Eigen::Matrix<double, 3, 3>;
  using Base = gtsam::NoiseModelFactor1<SE3>;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;
  using Jacobian = Eigen::Matrix<double, 3, 6>;

  // using Vector = Eigen::Vector<double, Dim>;

  SE3_observation_factor_t(const gtsam::Key key, const Translation offset, const Observation z,
                           const NoiseModel& cost_model)
    : Base(cost_model, key)
    , _offset(offset)
    , _z(z)
    , _Hzero(Jacobian::Zero())
    , _skew_offset(gtsam::skewSymmetric(-offset[0], -offset[1], -offset[2]))
  {
  }

  static Translation predict(const SE3& x, const Translation& offset)
  {
    return x * offset;
  }

  virtual Eigen::VectorXd evaluateError(const SE3& x, boost::optional<Eigen::MatrixXd&> H0 = boost::none) const override
  {
    const Translation pt_expected{ predict(x, _offset) };
    const Translation error{ pt_expected - _z };

    if (H0)
    {
      const Rotation R{ x.rotation().matrix() };
      *H0 = _Hzero;
      H0->leftCols<3>() = R * _skew_offset;
      H0->rightCols<3>() = R;
    }
    return error;
  }

  void print(const std::string& s,
             const gtsam::KeyFormatter& keyFormatter = factor_graphs::symbol_factory_t::formatter) const override
  {
    const std::string key_x{ keyFormatter(this->template key<1>()) };

    std::cout << "SE3 Observation Factor: [" << key_x << "]\n";
    std::cout << "\t Offset: " << _offset.transpose() << "\n";
    std::cout << "\t Observation: " << _z.transpose() << "\n";
  }

private:
  Translation _offset;
  Translation _z;
  const Jacobian _Hzero;
  const SkewMatrix _skew_offset;
};

// template <typename SE3>
class SE3_symmetric_observation_factor_t : public gtsam::NoiseModelFactor1<gtsam::Pose3>
{
public:
  using SE3 = gtsam::Pose3;
  using Translation = Eigen::Vector<double, 3>;
  using Rotation = gtsam::Rot3;
  using SkewMatrix = Eigen::Matrix<double, 3, 3>;
  using Base = gtsam::NoiseModelFactor1<SE3>;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;
  using Jacobian = Eigen::Matrix<double, 3, 6>;

  // using Vector = Eigen::Vector<double, Dim>;

  template <typename RotationType>
  SE3_symmetric_observation_factor_t(const gtsam::Key key, const RotationType offset, const Translation zA,
                                     const Translation zB, const NoiseModel& cost_model)
    : Base(cost_model, key), _offset_inv(SE3(Rotation(offset), Translation::Zero()).inverse()), _zA(zA), _zB(zB)
  {
  }

  static Eigen::VectorXd compute_error(const SE3& x, const Translation& zA, const Translation& zB,
                                       const SE3& offset_inv, boost::optional<Eigen::MatrixXd&> H0 = boost::none)
  {
    // const Translation  x.transformFrom(_zA, Hself = boost::none,
    //                             OptionalJacobian<3, 3> Hpoint = boost::none) const;

    // pRZ.transformFrom(p0i.transformFrom(zB)) - p0i.transformFrom(zA)

    Eigen::MatrixXd xinv_H_x;
    Eigen::MatrixXd pA_H_xinv;
    Eigen::MatrixXd xRinv_H_xinv;
    Eigen::MatrixXd Rx_H_x;
    Eigen::MatrixXd Rxinv_H_Rx;
    Eigen::MatrixXd pB_H_xRinv;
    // Eigen::MatrixXd pBrot_H_pB;
    const SE3 xinv{ x.inverse(xinv_H_x) };
    const SE3 xRinv{ gtsam::traits<SE3>::Compose(xinv, offset_inv, xRinv_H_xinv) };

    const Translation pA{ xinv.transformFrom(zA, pA_H_xinv, boost::none) };
    const Translation pB{ xRinv.transformFrom(zB, pB_H_xRinv, boost::none) };
    const Translation error{ pA - pB };

    // const Translation pB{ xinv.transformFrom(zB, pB_H_xinv, pB_H_zB) };
    // const Translation pBrot{ offset.rotate(pB, boost::none, pBrot_H_pB) };
    // const Translation error{ pA - pBrot };

    DEBUG_VARS(x);
    DEBUG_VARS(offset_inv);
    DEBUG_VARS(zA.transpose(), zB.transpose());
    DEBUG_VARS(pA.transpose(), pB.transpose());
    // DEBUG_VARS(pBrot.transpose());
    DEBUG_VARS(error.transpose());

    if (H0)
    {
      const Eigen::Matrix3d err_H_pA{ Eigen::Matrix3d::Identity() };
      const Eigen::Matrix3d err_H_pB{ -Eigen::Matrix3d::Identity() };
      *H0 = err_H_pA * pA_H_xinv * xinv_H_x +  // no-lint
            err_H_pB * pB_H_xRinv * xRinv_H_xinv * xinv_H_x;

      // err_H_pA = I
      // *H0 = /* err_H_pA */ pA_H_xinv * xinv_H_x +  // no-lint
      //       err_H_pB * pBrot_H_pB * pB_H_xinv * xinv_H_x;
    }
    return error;
  }

  virtual Eigen::VectorXd evaluateError(const SE3& x, boost::optional<Eigen::MatrixXd&> H0 = boost::none) const override
  {
    const Eigen::VectorXd error{ compute_error(x, _zA, _zB, _offset_inv, H0) };

    return error;
  }

private:
  SE3 _offset_inv;
  Translation _zA;
  Translation _zB;
  // const Jacobian _Hzero;
  // const SkewMatrix _skew_offset;
};

}  // namespace estimation
