#include <set>
#include <thread>

// Ros
#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseArray.h>
#include <tf2_ros/transform_broadcaster.h>

// mj-ros
#include <utils/rosparams_utils.hpp>
#include <ml4kp_bridge/defs.h>
#include <ml4kp_bridge/fg_utils.hpp>
#include <estimation/TrajectoryEstimation.h>
#include <estimation/StateEstimation.h>
#include <estimation/fg_trajectory_estimation.hpp>
#include <analytical/fg_ltv_sde.hpp>
#include <utils/std_utils.hpp>
#include <utils/dbg_utils.hpp>
#include <interface/TensegrityEndcaps.h>
#include <estimation/ty_endcap_dyn_z_factor.hpp>

// ML4KP
#include <prx/factor_graphs/utilities/symbols_factory.hpp>
#include <prx/factor_graphs/lie_groups/lie_integrator.hpp>
#include <prx/factor_graphs/factors/euler_integration_factor.hpp>
#include <prx/factor_graphs/factors/se3_observation.hpp>
#include <prx/factor_graphs/utilities/values_utilities.hpp>
#include <prx/factor_graphs/utilities/dbg_utills.hpp>

// Gtsam
#include <gtsam/basis/FitBasis.h>
#include <gtsam/basis/Chebyshev2.h>

namespace estimation
{
template <typename Base>
class pose_from_poly_t : public Base
{
  using Derived = pose_from_poly_t<Base>;

  static constexpr std::size_t DIM{ 6 };

  using Values = gtsam::Values;
  using FactorGraph = gtsam::NonlinearFactorGraph;

  using SF = prx::fg::symbol_factory_t;

  using SE3 = prx::fg::se3_t;
  using PolyMatrix = Eigen::Matrix<double, DIM, -1>;

public:
  pose_from_poly_t() : _t0(ros::Time::ZERO), _poly_matrix(6), _step(0.1)
  {
  }

  ~pose_from_poly_t() {};

  virtual void onInit()
  {
    ros::NodeHandle& private_nh{ Base::getPrivateNodeHandle() };

    std::string se3_topic_name;
    std::string path_topic_name;
    std::string endcap_topic_name;
    PARAM_SETUP(private_nh, se3_topic_name);
    PARAM_SETUP(private_nh, path_topic_name);
    PARAM_SETUP(private_nh, endcap_topic_name);

    double& step_size{ _step };
    double& polynoimal_degree{ _N };
    double& polynoimal_lower_bound{ _a };
    double& polynoimal_upper_bound{ _b };

    std::vector<double> matrix_values;

    PARAM_SETUP(private_nh, polynoimal_degree);
    PARAM_SETUP(private_nh, polynoimal_lower_bound);
    PARAM_SETUP(private_nh, polynoimal_upper_bound);
    PARAM_SETUP(private_nh, matrix_values);
    PARAM_SETUP_WITH_DEFAULT(private_nh, step_size, step_size);

    DEBUG_VARS(matrix_values.size(), matrix_values.size() / DIM);
    PolyMatrix in_matrix{ PolyMatrix::Zero(DIM, polynoimal_degree) };
    ml4kp_bridge::copy(in_matrix, matrix_values);
    _poly_matrix = gtsam::ParameterMatrix<DIM>(in_matrix);
    // DEBUG_VARS(_poly_matrix);

    _endcap_subscriber = private_nh.subscribe(endcap_topic_name, 1, &Derived::pose_callback, this);

    _se3_publisher = private_nh.advertise<geometry_msgs::Pose>(se3_topic_name, 1, true);
    _path_publisher = private_nh.advertise<geometry_msgs::PoseArray>(path_topic_name, 1, true);

    compute_path_from_poly();
  }

private:
  void compute_path_from_poly()
  {
    _pose_arr_msg.poses.clear();
    for (double ti = _a; ti < _b; ti += _step)
    {
      const gtsam::Chebyshev2::ManifoldEvaluationFunctor<SE3> f(_N, ti, _a, _b);
      const SE3 se3{ f(_poly_matrix) };

      _pose_arr_msg.poses.emplace_back();
      ml4kp_bridge::copy(_pose_arr_msg.poses.back(), se3);
    }
    DEBUG_VARS(_pose_arr_msg.poses.size());

    _path_publisher.publish(_pose_arr_msg);
  }

  void pose_callback(const interface::TensegrityEndcapsConstPtr msg)
  {
    const ros::Time ti{ msg->header.stamp };

    if (_t0.is_zero())
    {
      _t0 = ti;
    }

    const double dt{ _a + (ti - _t0).toSec() };

    if (dt < _b)
    {
      const gtsam::Chebyshev2::ManifoldEvaluationFunctor<SE3> f(_N, dt, _a, _b);
      const SE3 se3{ f(_poly_matrix) };

      ml4kp_bridge::copy(_pose_msg, se3);
      _se3_publisher.publish(_pose_msg);
    }
  }

  ros::Time _t0;
  double _a, _b, _N;
  double _step;
  gtsam::ParameterMatrix<DIM> _poly_matrix;

  geometry_msgs::Pose _pose_msg;
  geometry_msgs::PoseArray _pose_arr_msg;

  ros::Publisher _se3_publisher;
  ros::Publisher _path_publisher;

  ros::Subscriber _endcap_subscriber;
};
}  // namespace estimation