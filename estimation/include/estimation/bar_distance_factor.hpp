#pragma once
#include <array>
#include <numeric>
#include <gtsam/config.h>
#include <gtsam/base/Testable.h>
#include <gtsam/nonlinear/Expression.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <factor_graphs/defs.hpp>

namespace estimation
{
// Endcap is observed P=(x,y,z) at time t+epsilon. The endcap's bar pose is q_t and velocity \dot{q}_t at time t.
// The prediction first "moves" q_t to q_{t+epsilon} using \dot{q} and then finds the position of the endcap given the
// offset.
class bar_distance_factor_t : public gtsam::NoiseModelFactor1<gtsam::Pose3, gtsam::Pose3>
{
public:
  using SE3 = gtsam::Pose3;
  using Base = gtsam::NoiseModelFactor1<SE3, SE3>;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;
  using Translation = Eigen::Vector3d;

  bar_distance_factor_t(const gtsam::Key kbar_i, const gtsam::Key kbar_j, const double min_dist,
                        const NoiseModel& cost_model)
    : Base(cost_model, kbar_i, kbar_j), _min_dist(min_dist)
  {
  }

  static double predict(const SE3& xi, const SE3& xj,  // no-lint
                        gtsam::OptionalJacobian<1, 6> Hxi = boost::none,
                        gtsam::OptionalJacobian<1, 6> Hxj = boost::none)
  {
    Eigen::Matrix<double, 3, 6> pi_H_xi, pi_H_xj;
    Eigen::Matrix<double, 1, 3> d_H_pi, d_H_pj;
    const Translation pti{ xi.translation(pi_H_xi) };
    const Translation ptj{ xj.translation(pi_H_xj) };
    const double dist{ gtsam::distance3(pti, ptj, d_H_pi, d_H_pj) };

    DEBUG_VARS(pti.transpose());
    DEBUG_VARS(ptj.transpose());
    if (Hxi)
    {
      *Hxi = d_H_pi * pi_H_xi;
    }
    if (Hxj)
    {
      *Hxj = d_H_pj * pi_H_xj;
    }

    return dist;
  }

  virtual Eigen::VectorXd evaluateError(const SE3& xi, const SE3& xj,
                                        boost::optional<Eigen::MatrixXd&> Hxi = boost::none,
                                        boost::optional<Eigen::MatrixXd&> Hxj = boost::none) const override
  {
    const double dist{ predict(xi, xj, Hxi, Hxj) };
    const Eigen::Vector<double, 1> error(dist - _min_dist);
    DEBUG_VARS(dist, _min_dist, error[0]);

    // return Eigen::Vector<double, 1>(0.0);
    return error;
  }

private:
  double _min_dist;
};

}  // namespace estimation