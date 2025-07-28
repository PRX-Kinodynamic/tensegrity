#include <set>
#include <thread>

// Ros
#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_ros/transform_broadcaster.h>

// mj-ros
#include <tensegrity_utils/rosparams_utils.hpp>
// #include <ml4kp_bridge/defs.h>
// #include <estimation/TrajectoryEstimation.h>
// #include <estimation/StateEstimation.h>
// #include <estimation/fg_trajectory_estimation.hpp>
// #include <analytical/fg_ltv_sde.hpp>
#include <tensegrity_utils/std_utils.hpp>
#include <tensegrity_utils/dbg_utils.hpp>
#include <interface/TensegrityEndcaps.h>
// #include <estimation/ty_endcap_dyn_z_factor.hpp>

// GTSAM
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/ISAM2Params.h>

// ML4KP
// #include <prx/factor_graphs/utilities/symbols_factory.hpp>
// #include <prx/factor_graphs/lie_groups/lie_integrator.hpp>
// #include <prx/factor_graphs/factors/euler_integration_factor.hpp>
// #include <prx/factor_graphs/factors/se3_observation.hpp>
// #include <prx/factor_graphs/utilities/values_utilities.hpp>
// #include <prx/factor_graphs/utilities/dbg_utills.hpp>

namespace estimation
{
template <typename Base>
class tensegrity_estimation_t : public Base
{
  using Derived = tensegrity_estimation_t<Base>;

  using Values = gtsam::Values;
  using FactorGraph = gtsam::NonlinearFactorGraph;

  // using SF = prx::fg::symbol_factory_t;

  using SE3 = gtsam::Pose3;
  using Velocity = Eigen::Vector<double, 6>;
  using Acceleration = Eigen::Vector<double, 6>;
  using LieIntegratorFactor = prx::fg::lie_integration_factor_t<SE3, Velocity>;
  using EulerIntegratorFactor = prx::fg::euler_integration_factor_t<Velocity, Acceleration>;

public:
  tensegrity_estimation_t()
    : _isam_params(gtsam::ISAM2GaussNewtonParams(), 0.1, 10, true, true, gtsam::ISAM2Params::CHOLESKY, true,
                   prx::fg::symbol_factory_t::formatter, true)
    , _isam(_isam_params)
    , _p_offset(0.0, 0.0, 3.25 / 2.0)
    , _m_offset(0.0, 0.0, -3.25 / 2.0)
    , _ti(0)
    , _initialized(false)
    // , _endcap_ids({ 1, 2, 4 })
    , _endcap_ids()
    , _world_frame("world")
    , _qdot(Velocity::Zero())
    , _qddot(Acceleration::Zero())
  {
  }

  ~tensegrity_estimation_t() {};

  virtual void onInit()
  {
    ros::NodeHandle& private_nh{ Base::getPrivateNodeHandle() };

    std::string red_endcaps_topic;
    std::string green_endcaps_topic;
    std::string blue_endcaps_topic;
    std::string& world_frame{ _world_frame };

    // std::vector<int>& endcap_ids{ _endcap_ids };
    // std::string& world_frame{ _world_frame };
    // std::string& robot_frame{ _robot_frame };
    // std::string& output_dir{ _output_dir };

    double estimation_frequency{ 10 };

    PARAM_SETUP(private_nh, red_endcaps_topic);
    PARAM_SETUP(private_nh, green_endcaps_topic);
    PARAM_SETUP(private_nh, blue_endcaps_topic);
    // PARAM_SETUP_WITH_DEFAULT(private_nh, endcap_ids, endcap_ids);
    PARAM_SETUP_WITH_DEFAULT(private_nh, world_frame, world_frame);
    PARAM_SETUP_WITH_DEFAULT(private_nh, estimation_frequency, estimation_frequency);
    // PARAM_SETUP(private_nh, world_frame);
    // PARAM_SETUP(private_nh, robot_frame);
    // PARAM_SETUP(private_nh, output_dir)

    _red_subscriber = private_nh.subscribe(red_endcaps_topic, 10, &Derived::endcap_callback, this);
    _green_subscriber = private_nh.subscribe(green_endcaps_topic, 10, &Derived::endcap_callback, this);
    _blue_subscriber = private_nh.subscribe(blue_endcaps_topic, 10, &Derived::endcap_callback, this);

    const ros::Duration estimation_timer(1.0 / estimation_frequency);
    _estimation_timer = private_nh.createTimer(estimation_timer, &Derived::estimation_timer_callback, this);

    _prev_t = ros::Time(0);
    _ti = 0;
    _tf.header.seq = 0;
    _tf.header.frame_id = _world_frame;
  }

  static inline gtsam::Key keyBar(const int& id, const int& ti)
  {
    return SF::create_hashed_symbol("q^{", id, "}_{", ti, "}");
  }

  static inline gtsam::Key keyVel(const int& id, const int& ti)
  {
    return SF::create_hashed_symbol("\\dot{q}^{", id, "}_{", ti, "}");
  }

  inline double get_dt(const std_msgs::Header& header)
  {
    if (not _initialized)
    {
      _prev_t = header.stamp;
      _initialized = true;
    }

    return (header.stamp - _prev_t).toSec();
  }

  void get_or_init_SE3(SE3& se3, Velocity& vel, const gtsam::Key& kbar, const gtsam::Key& kvel,
                       const std::vector<geometry_msgs::Point>& endcaps)
  {
    if (_values.exists(kbar))
    {
      prx::fg::get_value_safe(_values, se3, kbar);
      prx::fg::get_value_safe(_values, vel, kvel);
    }
    else
    {
      const Eigen::Vector3d r1{ Eigen::Vector3d(endcaps[0].x, endcaps[0].y, endcaps[0].z) };
      Eigen::Vector3d r2;
      if (endcaps.size() >= 2)
      {
        r2 = Eigen::Vector3d(endcaps[1].x, endcaps[1].y, endcaps[1].z);
      }
      else
      {
        r2 = r1 + _p_offset / 2.0;  // Just pick a side and init with that.
      }
      const Eigen::Quaterniond init_quat{ Eigen::Quaterniond::FromTwoVectors(_m_offset.normalized(), r2 - r1) };

      vel = Velocity::Zero();
      se3 = SE3(init_quat, (r1 + r2) / 2.0);

      _values.insert(kbar, se3);
      _values.insert(kvel, vel);
      _next_values.insert(kbar, se3);
      _next_values.insert(kvel, vel);

      // _values.insert(kvel, zero);
      // _values.insert(kvel1, zero);
    }
  }

  void endcap_callback(const interface::TensegrityEndcapsConstPtr msg)
  {
    using ObservationFactor = estimation::endcap_dynamic_observation_t;

    const int id{ msg->barId };
    const gtsam::Key kbar{ keyBar(id, _ti) };
    const gtsam::Key kvel{ keyVel(id, _ti) };
    ObservationFactor::NoiseModel noise{ gtsam::noiseModel::Isotropic::Sigma(3, 1e0) };

    _endcap_ids.insert(id);
    DEBUG_VARS(id, _ti);
    PRINT_KEY(kbar);
    PRINT_KEY(kvel);
    const double dt{ get_dt(msg->header) };
    for (int i = 0; i < msg->endcaps.size(); ++i)
    {
      const geometry_msgs::Point& endcap{ msg->endcaps[i] };
      const Eigen::Vector<double, 3> z{ endcap.x, endcap.y, endcap.z };

      get_or_init_SE3(_q, _qdot, kbar, kvel, msg->endcaps);
      // get_or_init_SE3(_qdot, kvel, msg->endcaps);

      // prx::fg::get_value_safe(_values, _q, kbar);
      // calculate_estimate_safe(_q, kbar);
      // calculate_estimate_safe(_qdot, kvel);

      const double d0{ (ObservationFactor::predict(_q, _qdot, dt, _p_offset) - z).norm() };
      const double d1{ (ObservationFactor::predict(_q, _qdot, dt, _m_offset) - z).norm() };

      const Eigen::Vector<double, 3>& offset{ (d0 < d1) ? _p_offset : _m_offset };
      _next_graph.emplace_shared<ObservationFactor>(kbar, kvel, offset, z, dt, noise);
    }
    DEBUG_PRINT
  }

  template <typename Estimate>
  void calculate_estimate_safe(Estimate& variable, const gtsam::Key& key)
  {
    try
    {
      variable = _isam.calculateEstimate<Estimate>(key);
    }
    catch (gtsam::IndeterminantLinearSystemException e)
    {
      const std::string msg{ "[EXCEPTION] Var:" + SF::formatter(e.nearbyVariable()) + "\n" };
      std::cout << msg << std::string(e.what()) << std::endl;
    }
  }

  void create_integration_factor()
  {
    const ros::Time now{ ros::Time::now() };
    const double dt{ (now - _prev_t).toSec() };

    DEBUG_PRINT
    for (auto id : _endcap_ids)
    {
      const gtsam::Key kbar0{ keyBar(id, _ti) };
      // const gtsam::Key kvel0{ keyVel(id, _ti) };

      const gtsam::Key kbar1{ keyBar(id, _ti + 1) };
      const gtsam::Key kvel1{ keyVel(id, _ti + 1) };

      DEBUG_PRINT
      if (_values.exists(kbar0))
      {
        prx::fg::get_value_safe(_values, _q, kbar0);
        // prx::fg::get_value_safe(_values, _qdot, kvel1);

        DEBUG_PRINT
        const SE3 q_next{ LieIntegratorFactor::predict(_q, _qdot, dt) };
        const Velocity qdot_next{ EulerIntegratorFactor::predict(_qdot, _qddot, dt) };

        _next_values.insert(kbar1, q_next);
        _next_values.insert(kvel1, qdot_next);

        _next_graph.emplace_shared<LieIntegratorFactor>(kbar1, kbar0, kvel1, nullptr, dt);
      }
    }

    _prev_t = now;
  }

  void estimation_timer_callback(const ros::TimerEvent& event)
  {
    if (not _initialized)
      return;

    try
    {
      DEBUG_PRINT
      create_integration_factor();

      SF::symbols_to_file("/Users/Gary/pracsys/catkin_ws/factor_graph_symbols.txt");
      _next_graph.print("Graph", SF::formatter);
      _next_values.print("Vals", SF::formatter);

      DEBUG_PRINT
      _isam.getLinearizationPoint().print("Isam Vals", SF::formatter);
      _isam.getFactorsUnsafe().print("Isam graph", SF::formatter);

      DEBUG_VARS(_ti);
      // gtsam::NonlinearFactorGraph graph;
      // _isam2_result = _isam.update(graph, _next_values);
      _isam2_result = _isam.update(_next_graph, _next_values);
      DEBUG_PRINT
      _inserted_factors[_ti].insert(_inserted_factors[_ti].end(),
                                    _isam2_result.newFactorsIndices.begin(),  // no-lint
                                    _isam2_result.newFactorsIndices.end());
      DEBUG_PRINT
      _isam2_result.newFactorsIndices.clear();
    }
    catch (gtsam::IndeterminantLinearSystemException e)
    {
      const std::string msg{ "[EXCEPTION] Var:" + SF::formatter(e.nearbyVariable()) + "\n" };
      prx::fg::indeterminant_linear_system_helper(_next_graph, _isam.getLinearizationPoint());
      // failure_to_file(msg + e.what());
      std::cout << msg << std::string(e.what()) << std::endl;
      exit(-1);
    }
    catch (const std::out_of_range& oor)
    {
      std::cerr << "Out of Range error: " << oor.what() << '\n';
    }

    _values.insert_or_assign(_next_values);
    _next_values.clear();
    _next_graph.erase(_next_graph.begin(), _next_graph.end());

    DEBUG_PRINT
    update_estimates();
    _ti++;
  }

  void update_estimates()
  {
    utils::update_header(_tf.header);

    for (auto id : _endcap_ids)
    {
      const gtsam::Key kbar{ keyBar(id, _ti) };
      const gtsam::Key kvel{ keyVel(id, _ti) };
      const gtsam::Key kvel1{ keyVel(id, _ti + 1) };

      DEBUG_VARS(id, _ti);
      PRINT_KEY(kbar);
      PRINT_KEY(kvel);
      calculate_estimate_safe(_q, kbar);
      DEBUG_PRINT
      calculate_estimate_safe(_qdot, kvel);
      DEBUG_PRINT

      _values.insert_or_assign(kbar, _q);
      _values.insert_or_assign(kvel, _qdot);
      _values.insert_or_assign(kvel1, _qdot);
      DEBUG_PRINT

      ml4kp_bridge::copy(_tf.transform, _q);

      _tf.child_frame_id = "bar_" + prx::utilities::convert_to<std::string>(id);

      _tf_broadcaster.sendTransform(_tf);
    }
  }

private:
  gtsam::ISAM2Params _isam_params;
  gtsam::ISAM2 _isam;
  gtsam::ISAM2Result _isam2_result;

  gtsam::Values _values;  // Using this to store estimates. Not fed into isam

  gtsam::Values _next_values;
  gtsam::NonlinearFactorGraph _next_graph;

  // TF variables
  std::string _world_frame;
  std::string _robot_frame;
  geometry_msgs::TransformStamped _tf;
  tf2_ros::TransformBroadcaster _tf_broadcaster;

  std::size_t _ti;
  ros::Time _prev_t;

  bool _initialized;

  const Eigen::Vector<double, 3> _p_offset;
  const Eigen::Vector<double, 3> _m_offset;

  ros::Timer _estimation_timer;
  ros::Subscriber _red_subscriber, _green_subscriber, _blue_subscriber;

  std::unordered_map<std::size_t, gtsam::FactorIndices> _inserted_factors;

  std::set<int> _endcap_ids;

  SE3 _q;          // auxiliary
  Velocity _qdot;  // auxiliary
  Acceleration _qddot;
  // std::vector<SE3> _qs;
  // std::vector<Velocity> _qdots;
};
}  // namespace estimation