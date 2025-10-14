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

#include <interface/type_conversions.hpp>
#include <interface/TensegrityEndcaps.h>
#include <interface/TensegrityBars.h>
#include <interface/TensegrityTrajectory.h>
#include <interface/node_status.hpp>

#include <estimation/bar_utilities.hpp>
#include <estimation/endcap_subscriber.hpp>
#include <estimation/tensegrity_cap_obs_factor.hpp>
#include <estimation/endcap_observation_factor.hpp>
#include <estimation/SE3_observation_factor.hpp>
#include <estimation/cable_sensor_subscriber.hpp>

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

using OptSE3 = std::optional<SE3>;
using OptTranslation = std::optional<Translation>;
using RodObservation = std::pair<OptTranslation, OptTranslation>;

using LieIntegrator = factor_graphs::lie_integration_factor_t<SE3, Velocity>;
// using ChebManifoldSE3 = manifold_evaluation_t<gtsam::Chebyshev2, SE3>;

struct chev_estimation_t
{
  using This = chev_estimation_t;
  using ManifoldEvaluationFactor = gtsam::ManifoldEvaluationFactor<gtsam::Chebyshev2, gtsam::Pose3>;
  ros::Timer _finish_timer;
  ros::Subscriber _bars_subscriber;
  // ros::Publisher _tensegrity_bars_publisher, _tensegrity_traj_publisher;

  Graph graph;
  Values values;

  std::array<SE3, 3> _bars;
  const std::array<estimation::RodColors, 1> _colors;

  int _idx;
  // int rod_idx;
  // std::array<int, 3> rotation_idx;
  double graph_resolution;
  double optimization_frequency;
  double publisher_frequency;

  ros::Time prev_dt;

  gtsam::LevenbergMarquardtParams _lm_params;
  std::shared_ptr<factor_graphs::levenberg_marquardt_t> _lm_helper;
  std::shared_ptr<interface::node_status_t> node_status, publisher_node;

  std::shared_ptr<estimation::cables_callback_t> cables_callback;

  // std::shared_ptr<rod_callback_t> red_rod;
  // std::shared_ptr<rod_callback_t> blue_rod;
  // std::shared_ptr<rod_callback_t> green_rod;

  estimation::observation_update_t _observation_update;

  Translation _offset;
  Rotation _Roffset;

  ros::Time prev_timepoint;
  std::map<double, std::array<gtsam::Pose3, 3>> input_bars;
  std::vector<std::array<gtsam::Pose3, 3>> vector_bars;
  std::vector<std::array<Eigen::Vector3d, 6>> endcaps;
  std::vector<ros::Time> timestamps;
  ros::Time t0;

  double _N, dt;
  double maxT;
  bool fg_not_run;
  gtsam::ParameterMatrix<6> _red_cheb_matrix, _green_cheb_matrix, _blue_cheb_matrix;
  ros::Publisher _tensegrity_bars_publisher;
  // double _a, _b, _N;
  // gtsam::ParameterMatrix<DIM> _poly_matrix;
  gtsam::Values result;
  std::ofstream _poses_file;

  chev_estimation_t(ros::NodeHandle& nh)
    : _idx(0)
    , _offset({ 0, 0, 0.36 / 2.0 })
    , _Roffset(0.0, 0.0, 1.0, 0.0)
    , node_status(interface::node_status_t::create(nh, "/nodes/chev_estimation/"))
    , prev_dt(0)
    , _bars({ factor_graphs::random<SE3>(), factor_graphs::random<SE3>(), factor_graphs::random<SE3>() })
    , _colors({ estimation::RodColors::RED })
    , t0(0)
    , fg_not_run(true)
    , _red_cheb_matrix(5)
    , _green_cheb_matrix(5)
    , _blue_cheb_matrix(5)
  // , _colors({ estimation::RodColors::RED, estimation::RodColors::GREEN, estimation::RodColors::BLUE })
  {
    // lm_params.setVerbosityLM("SILENT");
    _lm_params.setVerbosityLM("SUMMARY");
    _lm_params.setMaxIterations(10);
    _lm_helper = std::make_shared<factor_graphs::levenberg_marquardt_t>(nh, "/nodes/state_estimation/fg", _lm_params);

    std::string tensegrity_bars_topicname;
    std::string publisher_node_id;  // subscribe to this, when it finished, run the estimation
    // std::string tensegrity_pose_topic, tensegrity_traj_topic;
    std::string poses_filename;
    // std::string cables_topic;
    // std::string initial_estimate_filename;
    // double resolution;
    bool use_cable_sensors{ true };
    double trajectory_pub_frequency;

    PARAM_SETUP(nh, tensegrity_bars_topicname);
    PARAM_SETUP(nh, publisher_node_id);
    PARAM_SETUP(nh, poses_filename);

    _poses_file.open(poses_filename, std::ios::out);

    publisher_node = interface::node_status_t::create(nh, publisher_node_id, true);
    _tensegrity_bars_publisher = nh.advertise<interface::TensegrityBars>("/tensegrity/cheb/output", 1, true);

    const ros::Duration finish_timer(1.0 / 10.0);
    _bars_subscriber = nh.subscribe(tensegrity_bars_topicname, 1, &This::bar_callback, this);
    _finish_timer = nh.createTimer(finish_timer, &This::finish_timer_callback, this);

    node_status->status(interface::NodeStatus::READY);
  }

  ~chev_estimation_t()
  {
  }

  gtsam::Key rotation_key(const int t, const int i)
  {
    return factor_graphs::symbol_factory_t::create_hashed_symbol("R^{", t, "}_{", i, "}");
  }

  void run_fg()
  {
    using EndcapObservationFactor = estimation::endcap_observation_factor_t;
    using RotationOffsetFactor = estimation::endcap_rotation_offset_factor_t;
    using RotationFixIdentity = estimation::rotation_fix_identity_t;
    // gtsam::Key kred{ gtsam::Symbol('R', 0) };
    // gtsam::Key kgreen{ gtsam::Symbol('G', 0) };
    // gtsam::Key kblue{ gtsam::Symbol('B', 0) };

    gtsam::Values values;
    gtsam::NonlinearFactorGraph graph;

    auto model = gtsam::noiseModel::Isotropic::Sigma(6, 1);
    // auto model = gtsam::noiseModel::Diagonal::Sigmas(Eigen::Vector<double, 6>(1, 1, 0, 1, 1, 1));
    // _N = std::floor(maxT) * 2;
    _N = 5;
    // _N = input_bars.size() * 2;
    _red_cheb_matrix = gtsam::ParameterMatrix<6>(_N);
    _green_cheb_matrix = gtsam::ParameterMatrix<6>(_N);
    _blue_cheb_matrix = gtsam::ParameterMatrix<6>(_N);

    // values.insert<gtsam::ParameterMatrix<6>>(kred, _red_cheb_matrix);
    // values.insert<gtsam::ParameterMatrix<6>>(kgreen, _green_cheb_matrix);
    // values.insert<gtsam::ParameterMatrix<6>>(kblue, _blue_cheb_matrix);
    // for (auto bars : input_bars)
    // {
    //   const double t{ bars.first };
    //   DEBUG_VARS(_N, t, maxT);
    //   graph.emplace_shared<ManifoldEvaluationFactor>(kred, bars.second[0], model, _N, t, 0, maxT);
    //   graph.emplace_shared<ManifoldEvaluationFactor>(kgreen, bars.second[1], model, _N, t, 0, maxT);
    //   graph.emplace_shared<ManifoldEvaluationFactor>(kblue, bars.second[2], model, _N, t, 0, maxT);
    // }
    int rot_tot{ 0 };
    gtsam::noiseModel::Base::shared_ptr rot_prior_nm{ gtsam::noiseModel::Isotropic::Sigma(3, 1e-4) };
    gtsam::noiseModel::Base::shared_ptr z_noise{ gtsam::noiseModel::Isotropic::Sigma(3, 1e0) };
    std::array<gtsam::Key, 3> kbar;
    for (int i = 0; i < endcaps.size(); ++i)
    {
      kbar[0] = gtsam::Symbol('R', i);
      kbar[1] = gtsam::Symbol('G', i);
      kbar[2] = gtsam::Symbol('B', i);
      // gtsam::Key kred1{ gtsam::Symbol('R', i + 1) };
      // gtsam::Key kgreen1{ gtsam::Symbol('G', i + 1) };
      // gtsam::Key kblue1{ gtsam::Symbol('B', i + 1) };

      // gtsam::Pose3 red_meassurement{ gtsam::traits<gtsam::Pose3>::Between(vector_bars[i][0], vector_bars[i + 1][0])
      // }; gtsam::Pose3 green_meassurement{ gtsam::traits<gtsam::Pose3>::Between(vector_bars[i][1], vector_bars[i +
      // 1][1]) }; gtsam::Pose3 blue_meassurement{ gtsam::traits<gtsam::Pose3>::Between(vector_bars[i][2], vector_bars[i
      // + 1][2]) };

      // values.insert(kred0, vector_bars[i][0]);
      // values.insert(kgreen0, vector_bars[i][1]);
      // values.insert(kblue0, vector_bars[i][2]);

      gtsam::Pose3 init_red(gtsam::Rot3(), (endcaps[i][0] + endcaps[i][1]) / 2.0);
      gtsam::Pose3 init_green(gtsam::Rot3(), (endcaps[i][2] + endcaps[i][3]) / 2.0);
      gtsam::Pose3 init_blue(gtsam::Rot3(), (endcaps[i][4] + endcaps[i][5]) / 2.0);

      values.insert(kbar[0], init_red);
      values.insert(kbar[1], init_green);
      values.insert(kbar[2], init_blue);

      // graph.emplace_shared<estimation::endcap_observations_t>(kbar[0], _offset, endcaps[i][0]);
      // graph.emplace_shared<estimation::endcap_observations_t>(kbar[0], -_offset, endcaps[i][1]);
      // graph.emplace_shared<estimation::endcap_observations_t>(kbar[1], _offset, endcaps[i][2]);
      // graph.emplace_shared<estimation::endcap_observations_t>(kbar[1], -_offset, endcaps[i][3]);
      // graph.emplace_shared<estimation::endcap_observations_t>(kbar[2], _offset, endcaps[i][4]);
      // graph.emplace_shared<estimation::endcap_observations_t>(kbar[2], -_offset, endcaps[i][5]);

      for (int j = 0; j < 6; j += 2)
      {
        const gtsam::Key key_rotA_offset{ rotation_key(i, j) };
        const gtsam::Key key_rotB_offset{ rotation_key(i, j + 1) };

        const Eigen::Vector3d zA{ endcaps[i][j] };
        const Eigen::Vector3d zB{ endcaps[i][j + 1] };

        // DEBUG_VARS(j, kbar[j / 3]);
        graph.emplace_shared<EndcapObservationFactor>(kbar[j / 2], key_rotA_offset, zA, _offset, z_noise);
        graph.emplace_shared<EndcapObservationFactor>(kbar[j / 2], key_rotB_offset, zB, _offset, z_noise);

        graph.emplace_shared<RotationOffsetFactor>(key_rotA_offset, key_rotB_offset, _Roffset, rot_prior_nm);
        graph.emplace_shared<RotationOffsetFactor>(key_rotB_offset, key_rotA_offset, _Roffset, rot_prior_nm);

        graph.emplace_shared<RotationFixIdentity>(key_rotA_offset, key_rotB_offset, _Roffset, rot_prior_nm);

        rot_tot += 2;
        values.insert(key_rotA_offset, gtsam::Rot3());
        values.insert(key_rotB_offset, _Roffset);
      }
    }

    // values.insert(gtsam::Symbol('R', vector_bars.size() - 1), vector_bars.back()[0]);
    // values.insert(gtsam::Symbol('G', vector_bars.size() - 1), vector_bars.back()[1]);
    // values.insert(gtsam::Symbol('B', vector_bars.size() - 1), vector_bars.back()[2]);

    result = _lm_helper->optimize(graph, values, true);
    // _red_cheb_matrix = result.at<gtsam::ParameterMatrix<6>>(kred);
    // _green_cheb_matrix = result.at<gtsam::ParameterMatrix<6>>(kgreen);
    // _blue_cheb_matrix = result.at<gtsam::ParameterMatrix<6>>(kblue);
  }

  void finish_timer_callback(const ros::TimerEvent& event)
  {
    // PRINT_MSG("Checking status")
    // DEBUG_VARS(*publisher_node)
    if (publisher_node->status() == interface::NodeStatus::FINISH and fg_not_run)
    {
      PRINT_MSG("Running FG!")
      run_fg();
      fg_not_run = false;
      dt = 0;
    }
    if (not fg_not_run)
    {
      // const gtsam::Chebyshev2::ManifoldEvaluationFunctor<SE3> f(_N, dt, 0, maxT);
      // const gtsam::Pose3 red_pose{ f(_red_cheb_matrix) };
      // const gtsam::Pose3 green_pose{ f(_green_cheb_matrix) };
      // const gtsam::Pose3 blue_pose{ f(_blue_cheb_matrix) };
      // dt += 0.1;
      // if (dt > maxT)
      // {
      //   PRINT_MSG("Finished!")
      //   ros::shutdown();
      // }

      if (dt < endcaps.size())
      {
        const int idx{ static_cast<int>(dt) };
        gtsam::Key kred{ gtsam::Symbol('R', idx) };
        gtsam::Key kgreen{ gtsam::Symbol('G', idx) };
        gtsam::Key kblue{ gtsam::Symbol('B', idx) };
        dt++;

        const gtsam::Pose3 red_pose{ result.at<gtsam::Pose3>(kred) };
        const gtsam::Pose3 green_pose{ result.at<gtsam::Pose3>(kgreen) };
        const gtsam::Pose3 blue_pose{ result.at<gtsam::Pose3>(kblue) };

        const std::string red_str{ tensegrity::utils::convert_to<std::string>(red_pose) };
        const std::string green_str{ tensegrity::utils::convert_to<std::string>(green_pose) };
        const std::string blue_str{ tensegrity::utils::convert_to<std::string>(blue_pose) };

        _poses_file << idx << " ";
        _poses_file << timestamps[idx] << " ";
        _poses_file << red_str << " ";
        _poses_file << green_str << " ";
        _poses_file << blue_str << " ";
        _poses_file << "\n";
        estimation::publish_tensegrity_msg(red_pose, green_pose, blue_pose, _tensegrity_bars_publisher, "real_sense");
      }
      else
      {
        _poses_file.close();
        ros::shutdown();
      }
    }
  }

  void bar_callback(const interface::TensegrityEndcapsConstPtr msg)
  {
    if (t0.isZero())
    {
      t0 = msg->header.stamp;
    }
    endcaps.emplace_back();
    timestamps.emplace_back(msg->header.stamp);
    for (int i = 0; i < 6; ++i)
    {
      const geometry_msgs::Point& endcap{ msg->endcaps[i] };
      // const gtsam::Pose3 red_pose{ tensegrity::utils::convert_to<gtsam::Pose3>(red) };
      endcaps.back()[i] = Eigen::Vector3d(endcap.x, endcap.y, endcap.z);
      // const geometry_msgs::Pose& green{ msg->bar_green };
      // const geometry_msgs::Pose& blue{ msg->bar_blue };
      // const gtsam::Pose3 green_pose{ tensegrity::utils::convert_to<gtsam::Pose3>(green) };
      // const gtsam::Pose3 blue_pose{ tensegrity::utils::convert_to<gtsam::Pose3>(blue) };
    }

    // const double ti{ (msg->header.stamp - t0).toSec() };
    // input_bars[ti] = { red_pose, green_pose, blue_pose };
    // vector_bars.push_back({ red_pose, green_pose, blue_pose });
    // maxT = ti;
    // t0 = msg->header.stamp;
    // DEBUG_VARS(ti, red_pose.translation().transpose())
    // DEBUG_VARS(ti, green_pose.translation().transpose())
    // DEBUG_VARS(ti, blue_pose.translation().transpose())
  }
};

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "ChevEstimation");
  ros::NodeHandle nh("~");

  chev_estimation_t estimator(nh);
  ros::spin();

  return 0;
}
