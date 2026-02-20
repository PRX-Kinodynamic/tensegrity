#include <iostream>
#include <cstdlib>
#include <tuple>
#include <ros/ros.h>

#include <algorithm>
#include <optional>
#include <deque>

#include <std_msgs/Bool.h>
#include <geometry_msgs/PoseArray.h>
#include <visualization_msgs/Marker.h>

#include <factor_graphs/defs.hpp>

#include <tensegrity_utils/type_conversions.hpp>
#include <tensegrity_utils/std_utils.hpp>
#include <tensegrity_utils/dbg_utils.hpp>
#include <tensegrity_utils/rosparams_utils.hpp>


#include <interface/defs.hpp>

#include <estimation/tensegrity_cap_obs_factor.hpp>
#include <estimation/endcap_observation_factor.hpp>
#include <estimation/SE3_observation_factor.hpp>

#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/basis/FitBasis.h>
#include <gtsam/basis/Chebyshev2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

using tensegrity::utils::convert_to;
using SF = factor_graphs::symbol_factory_t;
using Translation = Eigen::Vector3d;

using Rotation = gtsam::Rot3;
using SE3 = gtsam::Pose3;
using Velocity = Eigen::Vector<double, 6>;

// using SE3ObsFactor = estimation::endcap_observation_t<SE3>;
using SE3ObsFactor = estimation::SE3_observation_factor_t<SE3>;
using NewSE3ObsFactor = estimation::SE3_symmetric_observation_factor_t;
// using ScrewAxis = prx::fg::screw_axis_t;
using Integrator = factor_graphs::lie_integration_factor_t<SE3, Velocity>;
// using Integrator = prx::fg::lie_integration_factor_t<SE3, ScrewAxis>;
// using ScrewSmothing = prx::fg::screw_smoothing_factor_t;
using Graph = gtsam::NonlinearFactorGraph;
using Values = gtsam::Values;

using OptTranslation = std::optional<Translation>;
using RodObersvation = std::pair<OptTranslation, OptTranslation>;
// using ChebManifoldSE3 = manifold_evaluation_t<gtsam::Chebyshev2, SE3>;

const Translation offset({ 0, 0, 0.325 / 2.0 });
const Rotation Roffset(0.0, 0.0, 0.0, 1.0);

gtsam::Key rod_symbol(const int rod, const double t)
{
  return SF::create_hashed_symbol("X^{", rod, "}_{", t, "}");
}

gtsam::Key rodvel_symbol(const int rod, const double t)
{
  return SF::create_hashed_symbol("\\dot{X}^{", rod, "}_{", t, "}");
}

enum RodColors
{
  RED = 0,
  GREEN,
  BLUE
};

struct polynomials_t
{
  using PolyParams = std::tuple<int, double, double, double, gtsam::Key>;  // {N, a, b, tx, key}
  // using PolyParams = std::tuple<int, double, double>;  // {N, a, b}
  template <typename RodObservations>
  polynomials_t(int N_, double dt_, const RodObservations& red, const RodObservations& green,
                const RodObservations& blue)
    : N(N_), t0(get_t0(red, green, blue)), tF(get_tF(red, green, blue)), dt(get_max_duration(dt_))
  {
  }

  template <typename RodObservations>
  ros::Time get_t0(const RodObservations& red, const RodObservations& green, const RodObservations& blue)
  {
    const ros::Time t0R{ red.observations[0].header.stamp };
    const ros::Time t0G{ green.observations[0].header.stamp };
    const ros::Time t0B{ blue.observations[0].header.stamp };

    return std::min(t0R, std::min(t0G, t0B));
  }

  template <typename RodObservations>
  ros::Time get_tF(const RodObservations& red, const RodObservations& green, const RodObservations& blue)
  {
    const ros::Time tFR{ red.observations.back().header.stamp };
    const ros::Time tFG{ green.observations.back().header.stamp };
    const ros::Time tFB{ blue.observations.back().header.stamp };

    return std::max(tFR, std::max(tFG, tFB));
  }

  ros::Duration get_max_duration(const double dt_in)
  {
    ros::Duration max_dur{ tF - t0 };
    if (dt_in < max_dur.toSec())
    {
      max_dur = ros::Duration(dt_in);
    }
    return max_dur;
  }

  static gtsam::Key poly_symbol(const RodColors rod, const int poly)
  {
    return SF::create_hashed_symbol("p^{", rod, "}_{", poly, "}");
  }

  PolyParams prev_params(const ros::Time ti, const RodColors rod) const
  {
    // const double tip{ ti - dt).toSec() };
    // TENSEGRITY_ASSERT(tip > 0, "[polynomials_t] Trying to access invalid previous polynomial ");
    return params(ti - dt, rod);
  }

  PolyParams params(const ros::Time ti, const RodColors rod) const
  {
    const double tz{ (ti - t0).toSec() };
    const double dt_sec{ dt.toSec() };
    const int segment{ static_cast<int>(tz / dt_sec) };
    // const double{ std::fmod(ti, segment) };
    // const double ti_p{ ti - dt * segment };

    const double a{ segment * dt_sec };
    const double b{ a + dt_sec };

    return { N, a, b, tz, poly_symbol(rod, segment) };
  }

  std::vector<PolyParams> all_params(const ros::Time tF, const RodColors rod) const
  {
    std::vector<PolyParams> ps;
    // const double dt_sec{ dt.toSec() };
    for (ros::Time ti{ t0 }; ti < tF; ti += dt)
    {
      ps.push_back(params(ti, rod));
    }
    return ps;
  }

  template <typename StampedMsg>
  double time_delta(const StampedMsg& msg) const
  {
    return (msg.header.stamp - t0).toSec();
  }

  const int N;
  const ros::Time t0;
  const ros::Time tF;
  const ros::Duration dt;
  // std::vector<PolyParams> _params;
  // void std::vector<PolyParams> params;
};

template <class Basis, typename Type>
class manifold_evaluation_t
  : public gtsam::NoiseModelFactorN<gtsam::ParameterMatrix<gtsam::traits<Type>::dimension>, Type>
{
  static constexpr Eigen::Index Dim{ gtsam::traits<Type>::dimension };

  using Base = gtsam::NoiseModelFactorN<gtsam::ParameterMatrix<Dim>, Type>;
  using Derived = manifold_evaluation_t<Basis, Type>;
  using OptDeriv = boost::optional<Eigen::MatrixXd&>;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;

  using JacobianXX = Eigen::Matrix<double, Dim, Dim>;
  using JacobianXP = Eigen::Matrix<double, Dim, -1>;

public:
  manifold_evaluation_t(const gtsam::Key keyPolyParams, const gtsam::Key keyX, const NoiseModel& cost_model,
                        const size_t N, double x, double a, double b)
    : Base(cost_model, keyPolyParams, keyX), _evaluation_function(N, x, a, b)
  {
  }

  virtual Eigen::VectorXd evaluateError(const gtsam::ParameterMatrix<Dim>& P, const Type& x,  // no-lint
                                        OptDeriv Hp = boost::none, OptDeriv Hx = boost::none) const override
  {
    const bool compute_derivs{ (Hx or Hp) };
    // BASIS::template ManifoldEvaluationFunctor<T>(N, x);
    const Type predicted{ _evaluation_function(P, compute_derivs ? &dxp_H_P : nullptr) };

    // X1_p (-) x1 => Eq. 26 from "A micro Lie theory [...]" https://arxiv.org/pdf/1812.01537.pdf
    const Type between{ x.between(predicted,                          // no-lint
                                  compute_derivs ? &b_H_x : nullptr,  // no-lint
                                  compute_derivs ? &b_H_xp : nullptr) };
    const Eigen::VectorXd error{ Type::Logmap(between, compute_derivs ? &err_H_b : nullptr) };

    if (Hp)
    {
      *Hp = err_H_b * b_H_xp * dxp_H_P;
    }
    if (Hx)
    {
      *Hx = err_H_b * b_H_x;
    }

    return error;
  }

private:
  const typename Basis::template ManifoldEvaluationFunctor<Type> _evaluation_function;

  mutable JacobianXX err_H_b;  // Deriv error wrt between

  mutable JacobianXX b_H_x;   // Deriv between wrt x
  mutable JacobianXX b_H_xp;  // Deriv between wrt x_{predicted}

  mutable Eigen::MatrixXd dxp_H_P;  // Deriv \dot{predicted} wrt Params
};

// Constraining: Pa(END) == Pb(START), where Pa goes before Pb
template <class Basis, typename Type>
class polynomials_connect_factor
  : public gtsam::NoiseModelFactorN<gtsam::ParameterMatrix<gtsam::traits<Type>::dimension>,
                                    gtsam::ParameterMatrix<gtsam::traits<Type>::dimension>>
{
  static constexpr Eigen::Index Dim{ gtsam::traits<Type>::dimension };

  using Base =
      gtsam::NoiseModelFactorN<gtsam::ParameterMatrix<Dim>, gtsam::ParameterMatrix<gtsam::traits<Type>::dimension>>;
  using Derived = polynomials_connect_factor<Basis, Type>;
  using OptDeriv = boost::optional<Eigen::MatrixXd&>;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;

  using JacobianXX = Eigen::Matrix<double, Dim, Dim>;
  using JacobianXP = Eigen::Matrix<double, Dim, -1>;

public:
  polynomials_connect_factor(const gtsam::Key keyPolyA, const gtsam::Key keyPolyB, const NoiseModel& cost_model,
                             const size_t N, double polyA_a, double polyA_b, double polyB_a, double polyB_b)
    : Base(cost_model, keyPolyA, keyPolyB)
    , _evaluation_function_Pa(N, polyA_b, polyA_a, polyA_b)
    , _evaluation_function_Pb(N, polyB_a, polyB_a, polyB_b)
  {
  }

  virtual Eigen::VectorXd evaluateError(const gtsam::ParameterMatrix<Dim>& Pa, const gtsam::ParameterMatrix<Dim>& Pb,
                                        OptDeriv Hpa = boost::none, OptDeriv Hpb = boost::none) const override
  {
    const bool compute_derivs{ (Hpa or Hpb) };
    // BASIS::template ManifoldEvaluationFunctor<T>(N, x);
    const Type predictedA{ _evaluation_function_Pa(Pa, compute_derivs ? &predA_H_Pa : nullptr) };
    const Type predictedB{ _evaluation_function_Pb(Pb, compute_derivs ? &predB_H_Pb : nullptr) };

    // X1_p (-) x1 => Eq. 26 from "A micro Lie theory [...]" https://arxiv.org/pdf/1812.01537.pdf
    const Type between{ predictedA.between(predictedB,                             // no-lint
                                           compute_derivs ? &b_H_predA : nullptr,  // no-lint
                                           compute_derivs ? &b_H_predB : nullptr) };
    const Eigen::VectorXd error{ Type::Logmap(between, compute_derivs ? &err_H_b : nullptr) };

    if (Hpa)
    {
      *Hpa = err_H_b * b_H_predA * predA_H_Pa;
    }
    if (Hpb)
    {
      *Hpb = err_H_b * b_H_predB * predB_H_Pb;
    }
    // DEBUG_VARS(predictedA)
    // DEBUG_VARS(predictedB)
    // DEBUG_VARS(error.transpose());
    return error;
  }

private:
  const typename Basis::template ManifoldEvaluationFunctor<Type> _evaluation_function_Pa;
  const typename Basis::template ManifoldEvaluationFunctor<Type> _evaluation_function_Pb;

  mutable JacobianXX err_H_b;  // Deriv error wrt between

  mutable Eigen::MatrixXd predA_H_Pa;  // Deriv between wrt x
  mutable Eigen::MatrixXd predB_H_Pb;  // Deriv between wrt x_{predicted}

  mutable JacobianXX b_H_predA;
  mutable JacobianXX b_H_predB;
};

struct rod_callback_t
{
  rod_callback_t(ros::NodeHandle& nh, std::string topicname)
  {
    _red_subscriber = nh.subscribe(topicname, 10, &rod_callback_t::endcap_callback, this);
  }

  void endcap_callback(const interface::TensegrityEndcapsConstPtr msg)
  {
    // if (msg->endcaps.size() > 0)
    // {
    observations.push_back(*msg);
    // }
  }

  std::deque<interface::TensegrityEndcaps> observations;
  ros::Subscriber _red_subscriber;
};

struct stop_t
{
  stop_t(ros::NodeHandle& nh, std::string topicname) : stop(false)
  {
    _stop_subscriber = nh.subscribe(topicname, 10, &stop_t::callback, this);
  }

  void callback(const std_msgs::BoolConstPtr msg)
  {
    stop = msg->data;
  }
  bool stop;
  ros::Subscriber _stop_subscriber;
};

SE3 init_rod_pose(interface::TensegrityEndcaps& observations)
{
  Translation zA, zB;
  interface::copy(zA, observations.endcaps[0]);
  if (observations.endcaps.size() > 1)  // both encaps were observed
  {
    interface::copy(zB, observations.endcaps[1]);
  }
  else if (observations.endcaps.size() == 1)  // only one was observed
  {
    zB = zA + offset;  // todo: add randomness to the guess?
  }

  const Translation int_translation{ (zA + zB) / 2.0 };
  DEBUG_VARS(zA.transpose(), zB.transpose());
  DEBUG_VARS(int_translation.transpose());
  return SE3(Rotation(), int_translation);
}

bool add_observation(const interface::TensegrityEndcaps& observations, const SE3& prev, const gtsam::Key& rodK,
                     Graph& graph)
{
  if (observations.endcaps.size() == 0)
  {
    PRINT_MSG("No observations!")
    return false;
  }
  Translation zA, zB;

  const Translation pA{ prev * offset };
  const Translation pB{ prev * -offset };

  // DEBUG_VARS(observations.endcaps)
  interface::copy(zA, observations.endcaps[0]);

  const double zApA_dist{ (pA - zA).norm() };
  const double zApB_dist{ (pB - zA).norm() };

  // LOG_VARS(observations.endcaps.size())
  // LOG_VARS(prev)
  // LOG_VARS(pA.transpose(), pB.transpose());
  // LOG_VARS(zA.transpose());
  // LOG_VARS(zApA_dist, zApB_dist);

  // DEBUG_VARS(zA.transpose())

  if (observations.endcaps.size() == 2)
  {
    // SE3_symmetric_observation_factor_t(const gtsam::Key key, const Rotation offset, const Translation zA,
    // const Translation zB, const NoiseModel& cost_model)
    interface::copy(zB, observations.endcaps[1]);

    graph.emplace_shared<NewSE3ObsFactor>(rodK, Roffset, zA, zB, nullptr);
  }

  if (observations.endcaps.size() > 1)  // both encaps were observed
  {
    interface::copy(zB, observations.endcaps[1]);
    const double zBpA_dist{ (pA - zB).norm() };
    const double zBpB_dist{ (pB - zB).norm() };

    LOG_VARS(zB.transpose());
    // LOG_VARS(zBpA_dist, zBpB_dist);

    if (zApA_dist < zApB_dist)
    {
      graph.emplace_shared<SE3ObsFactor>(rodK, offset, zA, nullptr);
      graph.emplace_shared<SE3ObsFactor>(rodK, -offset, zB, nullptr);
    }
    else
    {
      graph.emplace_shared<SE3ObsFactor>(rodK, -offset, zA, nullptr);
      graph.emplace_shared<SE3ObsFactor>(rodK, offset, zB, nullptr);
    }
  }
  else if (observations.endcaps.size() == 1)  // only one was observed
  {
    if (zApA_dist < zApB_dist)
    {
      graph.emplace_shared<SE3ObsFactor>(rodK, offset, zA, nullptr);
    }
    else
    {
      graph.emplace_shared<SE3ObsFactor>(rodK, -offset, zA, nullptr);
    }
  }
  return true;
}
inline void print(const gtsam::Pose3& p, std::string msg)
{
  auto tr = p.translation().transpose();
  auto q = p.rotation().toQuaternion();
  DEBUG_VARS(msg, tr, q);
}

template <typename ParamMat>
geometry_msgs::PoseArray compute_path_from_poly(const ParamMat& mat, const int N, const double& a, const double& b,
                                                const double& step, const double& tlimit)
{
  geometry_msgs::PoseArray pose_arr_msg;
  for (double ti = a; ti < tlimit; ti += step)
  {
    const gtsam::Chebyshev2::ManifoldEvaluationFunctor<SE3> f(N, ti, a, b);
    const SE3 se3{ f(mat) };

    // DEBUG_VARS(a, b, tlimit, ti, step);
    // print(se3, "[RosMsg]");

    pose_arr_msg.poses.emplace_back();
    interface::copy(pose_arr_msg.poses.back(), se3);
  }
  // DEBUG_VARS(pose_arr_msg.poses.size());

  return pose_arr_msg;
  // _path_publisher.publish(pose_arr_msg);
}

// Observations may contain NaNs => remove those
void remove_invalid_observations(interface::TensegrityEndcaps& observations)
{
  Translation z;
  // for (auto zin : observations.endcaps)
  std::size_t i{ 0 };
  // bool nan_found{ false };
  while (i < observations.endcaps.size())
  {
    auto zin = observations.endcaps[i];
    interface::copy(z, zin);
    // if (z.array().isNaN())
    if (std::isnan(z.template maxCoeff<Eigen::PropagateNaN>()))
    {
      observations.endcaps.erase(observations.endcaps.begin() + i);
      // nan_found = true;
    }
    else
    {
      i++;
    }
  }

  // if (nan_found)
  // {
  //   // DEBUG_VARS(observations);
  //   LOG_VARS(observations)
  // }
}

bool add_to_graph(const interface::TensegrityEndcaps& obs, const gtsam::Key& key_rod, const SE3& prev_pose,
                  const polynomials_t& polys, const RodColors color, Graph& graph, Values& values)
{
  using ChebManifoldSE3 = manifold_evaluation_t<gtsam::Chebyshev2, SE3>;
  using PolysStartEndConstraint = polynomials_connect_factor<gtsam::Chebyshev2, SE3>;
  // const gtsam::Key key_red{ rod_symbol(red_id, i) };
  const gtsam::noiseModel::Base::shared_ptr se3_noise{ gtsam::noiseModel::Isotropic::Sigma(6, 1) };

  // { red_rod.observations[0] };
  // tz = obs.header.stamp.toSec() - polys.t0;

  const bool z_added{ add_observation(obs, prev_pose, key_rod, graph) };
  // bool add_observation(const interface::TensegrityEndcaps& observations, SE3& prev, const gtsam::Key& rodK, Graph&
  // graph)

  if (z_added)
  {
    const ros::Time& tz{ obs.header.stamp };

    auto [N, a, b, t_ab, key_poly] = polys.params(tz, color);
    graph.emplace_shared<ChebManifoldSE3>(key_poly, key_rod, se3_noise, N, t_ab, a, b);
    values.insert(key_rod, prev_pose);
    if (not values.exists(key_poly))
    {
      // PRINT_KEYS(key_poly)
      auto [prev_N, prev_a, prev_b, t_prev, prev_key_poly] = polys.prev_params(tz, color);
      if (values.exists(prev_key_poly))  // Avoid adding constraint to first poly
      {
        // PRINT_KEYS(prev_key_poly, key_poly)
        // DEBUG_VARS(N, prev_a, prev_b, a, b);
        graph.emplace_shared<PolysStartEndConstraint>(prev_key_poly, key_poly, se3_noise, N, prev_a, prev_b, a, b);
      }
      values.insert(key_poly, gtsam::ParameterMatrix<6>(N));
    }
  }
  return z_added;
}

SE3 get_pose_from_poly(const ros::Time ti, const RodColors& color, const polynomials_t& polys, Values& values)
{
  // interface::TensegrityBars bars;
  auto [N, a, b, t_ab, key_poly] = polys.params(ti, color);

  gtsam::ParameterMatrix<6> params_poly{ values.at<gtsam::ParameterMatrix<6>>(key_poly) };
  const gtsam::Chebyshev2::ManifoldEvaluationFunctor<SE3> chev_fun(N, t_ab, a, b);
  return chev_fun(params_poly);
}

// void update_pose(SE3& prev_pose, const double& tz, const polynomials_t& polys, const RodColors& color, Values&
// values)
// {
//   // TODO: Go over every poly, not only the current one
//   // auto [N, a, b, tzp, key_poly] = polys.params(tz, color);

//   // gtsam::ParameterMatrix<6> params_poly{ values.at<gtsam::ParameterMatrix<6>>(key_poly) };
//   // const gtsam::Chebyshev2::ManifoldEvaluationFunctor<SE3> chev_fun(N, tzp, a, b);
//   // prev_pose = chev_fun(params_poly);
//   // PRINT_KEYS(key_poly);
//   prev_pose = get_pose_from_poly(tz, color, polys, values);

//   // print(prev_pose, "New pose");

//   // print(pose_red, "[Current]");
//   // const geometry_msgs::PoseArray poly_msg{ compute_path_from_poly(params_poly, N, a, b, step, tzp) };
//   // publisher.publish(poly_msg);
// }

void publish_polynomial_traj(const polynomials_t& polys, const RodColors& color, const ros::Time& tF, const double step,
                             ros::Publisher& publisher, Values& values)
{
  // DEBUG_VARS(tF, step);
  auto all_params = polys.all_params(tF, color);
  geometry_msgs::PoseArray poly_msg;
  tensegrity::utils::init_header(poly_msg.header, "world");
  for (auto [N, a, b, t_ab, key_poly] : all_params)
  {
    const gtsam::ParameterMatrix<6> params_poly{ values.at<gtsam::ParameterMatrix<6>>(key_poly) };

    // PRINT_KEYS(key_poly);
    const double tmax{ std::min(b, t_ab) };
    // LOG_VARS(SF::formatter(key_poly), N, a, b, tzp, step, tmax);
    // LOG_VARS(params_poly);
    // DEBUG_VARS(N, a, b, tmax, step);
    // LOG_VARS(N, a, b, tz);
    // LOG_VARS(params_poly);
    const geometry_msgs::PoseArray msg{ compute_path_from_poly(params_poly, N, a, b, step, tmax) };
    poly_msg.poses.insert(poly_msg.poses.end(), msg.poses.begin(), msg.poses.end());
    // DEBUG_VARS(msg.poses.size(), poly_msg.poses.size());
  }
  publisher.publish(poly_msg);
}

void publish_tensegrity_msg(const ros::Time& ti, const polynomials_t& polys, Values& values, ros::Publisher& publisher)
{
  interface::TensegrityBars bars;
  bars.header.stamp = ti;
  const SE3 red_pose{ get_pose_from_poly(ti, RodColors::RED, polys, values) };
  interface::copy(bars.bar_red, red_pose);

  bars.bar_green.position.z = 10;  // get it far to avoid it being in the img
  bars.bar_blue.position.z = 10;   // get it far to avoid it being in the img

  bars.bar_green.orientation.w = 1;  // get it far to avoid it being in the img
  bars.bar_blue.orientation.w = 1;   // get it far to avoid it being in the img
  publisher.publish(bars);
}

int main(int argc, char* argv[])
{
  using Callback = const boost::function<void(const interface::TensegrityEndcapsConstPtr)>;

  ros::init(argc, argv, "FullTrajEstimation");
  ros::NodeHandle nh("~");

  std::string red_endcaps_topic, blue_endcaps_topic, green_endcaps_topic;
  std::string red_poly_topic, blue_poly_topic, green_poly_topic;
  std::string stop_topic;
  std::string red_observations;
  std::string tensegrity_pose_topic;

  int N{ 30 * 3 };
  double max_time{ -1.0 };

  PARAM_SETUP(nh, red_endcaps_topic);
  PARAM_SETUP(nh, blue_endcaps_topic);
  PARAM_SETUP(nh, green_endcaps_topic);
  PARAM_SETUP(nh, red_poly_topic);
  PARAM_SETUP(nh, red_observations);
  PARAM_SETUP(nh, tensegrity_pose_topic);
  PARAM_SETUP(nh, stop_topic);
  PARAM_SETUP_WITH_DEFAULT(nh, max_time, max_time)
  PARAM_SETUP_WITH_DEFAULT(nh, N, N)

  // rosrun estimation full_traj_estimation _red_endcaps_topic:=/tensegrity/endcap/red/positions
  // _blue_endcaps_topic:=/tensegrity/endcap/blue/positions _green_endcaps_topic:=/tensegrity/endcap/green/positions
  // _red_poly_topic:=/tensegrity/poly/red _red_observations:=/tensegrity/observations/red
  // _tensegrity_pose_topic:=/tensegrity/poses _stop_topic:=/estimation/run_fg _max_time:=1.0 _N:=3

  // const std::string filename_red{ params["red_file"].as<>() };

  ros::Publisher red_poly_publisher{ nh.advertise<geometry_msgs::PoseArray>(red_poly_topic, 1, true) };
  ros::Publisher red_observations_publisher{ nh.advertise<visualization_msgs::Marker>(red_observations, 1, true) };
  ros::Publisher tensegrity_bars_publisher{ nh.advertise<interface::TensegrityBars>(tensegrity_pose_topic, 1, true) };

  Graph graph;
  Values initial_values;

  rod_callback_t red_rod(nh, red_endcaps_topic);
  rod_callback_t blue_rod(nh, blue_endcaps_topic);
  rod_callback_t green_rod(nh, green_endcaps_topic);

  stop_t stop(nh, stop_topic);

  PRINT_MSG("Starting to listen")

  ros::Rate rate(30);  // sleep for 1 sec
  while (ros::ok())
  {
    if (stop.stop)
      break;
    ros::spinOnce();
    rate.sleep();
  }

  // double t0, tF;
  // find_time_range(t0, tF, red_rod, blue_rod, green_rod);

  // if (max_time < 0)
  // {
  //   max_time = tF - t0;
  // }
  polynomials_t polys(N, max_time, red_rod, blue_rod, green_rod);

  const std::size_t red_tot_obs{ red_rod.observations.size() };
  const std::size_t green_tot_obs{ green_rod.observations.size() };
  const std::size_t blue_tot_obs{ blue_rod.observations.size() };
  const std::size_t max_obs{ std::max(red_tot_obs, std::max(green_tot_obs, blue_tot_obs)) };
  const double avg_z{ (red_tot_obs + green_tot_obs + blue_tot_obs) / 3.0 };
  // const double dt_avg{ (tF - t0) / avg_z };

  // DEBUG_VARS(avg_z, dt_avg)
  // std::vector<interface::TensegrityEndcaps> red_observations;

  // auto red_callback = boost::bind(&endcap_callback, _1, &red_observations);

  // ros::Subscriber red_subscriber{ nh.subscribe<interface::TensegrityEndcaps>(red_endcaps_topic, 10, red_callback)
  // }; _green_subscriber = private_nh.subscribe(green_endcaps_topic, 10, &Derived::endcap_callback, this);
  // _blue_subscriber = private_nh.subscribe(blue_endcaps_topic, 10, &Derived::endcap_callback, this);

  gtsam::noiseModel::Base::shared_ptr se3_noise{ gtsam::noiseModel::Isotropic::Sigma(6, 1) };

  // rod_t rod_red;
  // const double noise_sigma{ 1.0 };

  const int red_id{ 0 };
  const int green_id{ 1 };
  const int blue_id{ 2 };

  // interface::TensegrityEndcaps& red_first_obs{ red_rod.observations[0] };

  visualization_msgs::Marker marker_red_obs;
  tensegrity::utils::init_header(marker_red_obs.header, "world");
  marker_red_obs.action = visualization_msgs::Marker::ADD;
  marker_red_obs.type = visualization_msgs::Marker::POINTS;

  marker_red_obs.color.r = 1.0;
  marker_red_obs.color.g = 0.0;
  marker_red_obs.color.b = 0.0;
  marker_red_obs.color.a = 0.5;

  marker_red_obs.scale.x = 0.03;  // is point width,
  marker_red_obs.scale.y = 0.03;  // is point height

  SE3 pose_red{ init_rod_pose(red_rod.observations[0]) };

  gtsam::LevenbergMarquardtParams lm_params;
  lm_params.setVerbosityLM("SUMMARY");
  lm_params.setMaxIterations(10);
  // for (double ti{ a }; ti < b; ti += dt_avg)
  // initial_values.insert(key_red, pose_red);
  ros::Time red_ti;
  bool red_z_added;

  print(pose_red, "[Initial]");
  int dummy{ 0 };

  // const double a{ 0 };
  // const double b{ tF - t0 };
  // gtsam::Key key_Pred{ poly_symbol(red_id, 0) };   // Polynomial on Red bar
  // gtsam::Key key_Pgreen{ poly_symbol(green_id) };  // Polynomial on Green bar
  // gtsam::Key key_Pblue{ poly_symbol(blue_id) };    // Polynomial on Blue bar

  // auto [N, a, b, tzp, key_poly] = polys.params(tz, color);
  // auto [std::ignore, std::ignore, std::ignore, std::ignore, key_poly_red_0] = polys.params(t0, RodColors::Red);
  // initial_values.insert(key_poly_red_0, gtsam::ParameterMatrix<6>(N));
  for (std::size_t i{ 0 }; i < max_obs; ++i)
  {
    DEBUG_VARS(i, max_obs);

    if (i < red_tot_obs)
    {
      // if ()
      const gtsam::Key key_red{ rod_symbol(red_id, i) };

      interface::TensegrityEndcaps& red_endcap{ red_rod.observations[0] };
      remove_invalid_observations(red_endcap);
      red_ti = red_endcap.header.stamp;
      red_z_added = add_to_graph(red_endcap, key_red, pose_red, polys, RodColors::RED, graph, initial_values);

      marker_red_obs.points.insert(marker_red_obs.points.end(), red_endcap.endcaps.begin(), red_endcap.endcaps.end());
      marker_red_obs.colors = std::vector<std_msgs::ColorRGBA>(marker_red_obs.points.size(), marker_red_obs.color);

      red_observations_publisher.publish(marker_red_obs);
    }
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial_values, lm_params);
    initial_values = optimizer.optimize();

    if (i < red_tot_obs)
    {
      if (red_z_added)
      {
        // update_pose(pose_red, red_ti, polys, RodColors::RED, initial_values);
        pose_red = get_pose_from_poly(red_ti, RodColors::RED, polys, initial_values);

        publish_polynomial_traj(polys, RodColors::RED, red_ti, 0.01, red_poly_publisher, initial_values);
        publish_tensegrity_msg(red_ti, polys, initial_values, tensegrity_bars_publisher);
      }
      red_rod.observations.pop_front();
    }

    if (dummy == 0)
    {
      PRINT_MSG("Waiting...");
      std::cin >> dummy;
    }
    else
    {
      dummy--;
    }
  };

  PRINT_MSG("Finished!");
  while (ros::ok())
  {
    ros::spinOnce();
    rate.sleep();
  }

  return 0;
}
