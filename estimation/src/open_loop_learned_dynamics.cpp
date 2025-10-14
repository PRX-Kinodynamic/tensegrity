#include <iostream>
#include <cstdlib>
#include <tuple>
#include <ros/ros.h>

#include <algorithm>
#include <optional>
#include <deque>

#include <std_msgs/Bool.h>
#include <std_msgs/Float64MultiArray.h>
#include <geometry_msgs/PoseArray.h>
#include <visualization_msgs/Marker.h>

#include <factor_graphs/defs.hpp>

#include <tensegrity_utils/type_conversions.hpp>
#include <tensegrity_utils/std_utils.hpp>
#include <tensegrity_utils/dbg_utils.hpp>
#include <tensegrity_utils/rosparams_utils.hpp>

#include <interface/TensegrityEndcaps.h>
#include <interface/TensegrityBars.h>
#include <interface/torch_nn_interface.hpp>

#include <estimation/bar_utilities.hpp>
#include <estimation/endcap_subscriber.hpp>
#include <estimation/tensegrity_cap_obs_factor.hpp>
#include <estimation/endcap_observation_factor.hpp>
#include <estimation/SE3_observation_factor.hpp>
#include <estimation/cable_sensor_subscriber.hpp>
#include <estimation/sensors_subscriber.hpp>

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
// using ChebManifoldSE3 = manifold_evaluation_t<gtsam::Chebyshev2, SE3>;

// const Translation offset({ 0, 0, 0.325 / 2.0 });
const Translation offset({ 0, 0, 0.325 / 2.0 });
const Rotation Roffset(0.0, 0.0, 1.0, 0.0);

struct nn_node_t
{
  ros::Subscriber initial_subscriber;
  Eigen::Vector<double, 39> start_state;  // (3p + 4q + 6v)* 3 = 39
  Eigen::Vector<double, 6> rest_lengths;
  Eigen::Vector<double, 6> motor_speeds;
  bool received;
  nn_node_t(ros::NodeHandle& nh) : received(false)
  {
    initial_subscriber = nh.subscribe("/simulator/initial_state", 10, &nn_node_t::initial_callback, this);
  }

  void initial_callback(std_msgs::Float64MultiArrayConstPtr msg)
  {
    int i = 0;
    for (; i < 39; ++i)
    {
      start_state[i] = msg->data[i];
    }
    for (int j = 0; j < 6; ++i, ++j)
    {
      rest_lengths[j] = msg->data[i];
    }
    for (int j = 0; j < 6; ++i, ++j)
    {
      motor_speeds[j] = msg->data[i];
    }
    DEBUG_VARS(start_state.transpose());
    DEBUG_VARS(rest_lengths.transpose());
    DEBUG_VARS(motor_speeds.transpose());
    received = true;
  }
};

int main(int argc, char* argv[])
{
  using Callback = const boost::function<void(const interface::TensegrityEndcapsConstPtr)>;

  using TorchModule = torch::jit::script::Module;
  using TorchModulePtr = std::shared_ptr<TorchModule>;

  using LinAcc = Eigen::Vector<double, 9>;
  using AngAcc = Eigen::Vector<double, 9>;
  using ContactLinAcc = Eigen::Vector<double, 9>;
  using ContactAngAcc = Eigen::Vector<double, 9>;
  using Toi = Eigen::Vector<double, 3>;
  using NextRestLens = Eigen::Vector<double, 6>;
  using NextMotorSpeeds = Eigen::Vector<double, 6>;
  using NextState = Eigen::Vector<double, 39>;
  using NnOutput =
      std::tuple<LinAcc, AngAcc, ContactLinAcc, ContactAngAcc, Toi, NextRestLens, NextMotorSpeeds, NextState>;

  using NnInputState = Eigen::Vector<double, 39>;
  using NnInputDt = Eigen::Vector<double, 1>;
  using NnInputControl = Eigen::Vector<double, 6>;
  using NnInputCables = Eigen::Vector<double, 6>;
  using NnInputMotorSpeeds = Eigen::Vector<double, 6>;
  using TorchInterface = interface::torch_nn_interface_t<TorchModule, NnOutput, NnInputState, NnInputControl, NnInputDt,
                                                         NnInputCables, NnInputMotorSpeeds>;

  ros::init(argc, argv, "StateEstimation");
  ros::NodeHandle nh("~");

  std::string tensegrity_pose_topic;
  std::string cable_map_filename;
  std::string torch_filename, publisher_node_id;

  PARAM_SETUP(nh, tensegrity_pose_topic);
  // PARAM_SETUP(nh, cable_map_filename);

  PARAM_SETUP(nh, torch_filename);
  PARAM_SETUP(nh, publisher_node_id);
  // PARAM_SETUP(nh, red_poly_topic);
  // PARAM_SETUP(nh, red_observations);
  // PARAM_SETUP(nh, stop_topic);
  // PARAM_SETUP_WITH_DEFAULT(nh, max_time, max_time)
  TorchInterface nn;
  nn.load_model(torch_filename);

  // rosrun estimation state_estimation _red_endcaps_topic:=/tensegrity/endcap/red/positions
  // _blue_endcaps_topic:=/tensegrity/endcap/blue/positions _green_endcaps_topic:=/tensegrity/endcap/green/positions
  // _tensegrity_pose_topic:=/tensegrity/poses
  interface::node_status_t node_status(nh);
  interface::node_status_t publisher_node(nh, publisher_node_id, true);

  ros::Publisher tensegrity_bars_publisher{ nh.advertise<interface::TensegrityBars>(tensegrity_pose_topic, 1, true) };
  ros::Publisher sim_initializer_publisher{ nh.advertise<std_msgs::Float64MultiArray>("/simulator/input", 1, true) };
  // ros::Publisher tensegrity_bars_publisher{ nh.advertise<interface::TensegrityBars>(tensegrity_pose_topic, 1, true)
  // };

  nn_node_t node(nh);
  estimation::sensors_callback_t sensors_callback(nh);
  // rod_callback_t red_rod(nh, red_endcaps_topic);
  // rod_callback_t blue_rod(nh, blue_endcaps_topic);
  // rod_callback_t green_rod(nh, green_endcaps_topic);

  // stop_t stop(nh, stop_topic);

  PRINT_MSG("Starting to listen")

  int idx{ 0 };
  // int green_idx{ 0 };
  // int blue_idx{ 0 };
  ros::Rate rate(30);  // sleep for 1 sec

  Translation initial{ { 0, 0, 0 } };
  SE3 red_bar{ Rotation(), initial };
  SE3 green_bar{ Rotation(), Translation(1, 0, 0) };
  SE3 blue_bar{ Rotation(), Translation(0, 1, 0) };

  using CablesFactor = estimation::cable_length_observations_factor_t;
  gtsam::noiseModel::Base::shared_ptr z_noise{ gtsam::noiseModel::Isotropic::Sigma(1, 1) };
  gtsam::noiseModel::Base::shared_ptr prior_Xnoise{ gtsam::noiseModel::Isotropic::Sigma(6, 1e-5) };

  // estimation::ColorMapping cable_map{ estimation::create_cable_map(cable_map_filename) };

  gtsam::LevenbergMarquardtParams lm_params;
  lm_params.setVerbosityLM("SUMMARY");
  lm_params.setMaxIterations(10);
  // const gtsam::Values result{ estimation::compute_initialization(initial_estimate_filename, offset, Roffset) };
  factor_graphs::levenberg_marquardt_t lm_helper(nh, "/nodes/state_estimation/fg", lm_params);

  estimation::observation_update_t observation_update;
  observation_update.offset = offset;
  observation_update.Roffset = Roffset;

  estimation::sensors_callback_t::OptMeassurements meassurements;
  std_msgs::Float64MultiArray initial_states;
  // initial_states.data = { -291.5031433105469,  999.2154541015625,  -183.045654296875,   // no-lint
  //                         10.175775527954102,  1161.7762451171875, -268.1787109375,     // no-lint
  //                         -161.33399963378906, 1007.384521484375,  -39.80809783935547,  // no-lint
  //                         -74.64183044433594,  1066.3685302734375, -382.4410095214844,  // no-lint
  //                         -250.14210510253906, 1162.1937255859375, -129.098876953125,   // no-lint
  //                         39.64032745361328,   1001.6146850585938, -261.7075500488281 };
  // for (int i = 0; i < initial_states.data.size(); ++i)
  // {
  //   initial_states.data[i] = initial_states.data[i] / 100;
  // }
  initial_states.data = { -0.2101508799049645,  -1.55697052059487,    0.21160320349436024,  // no-lint
                          -0.06533024640353524, 1.1963748911289007,   1.9324417601114203,   // no-lint
                          1.4718990931605866,   -0.49621601247996416, 0.25248600442390284,  // no-lint
                          -1.5546579028501473,  0.44611096299887554,  0.9713435516060003,
                          0.7682605278680437,   -1.0886189921601743,  1.7491155422475466,
                          -0.4100205918699853,  1.4993196711072316,   0.17500000000000002 };
  sim_initializer_publisher.publish(initial_states);

  // Eigen::Vector<double, 1> dt(0.01);
  NnOutput output;
  NnInputState state, next_state;
  NnInputDt dt(0.01);
  NnInputControl ctrl;
  NnInputCables rest_lengths;
  NnInputMotorSpeeds motor_speeds;

  using LieIntegrator = factor_graphs::lie_integrator_t<gtsam::Pose3, Eigen::Vector<double, 6>>;
  bool first{ true };
  ros::Time ti;
  node_status.status(interface::NodeStatus::READY);
  while (ros::ok())
  {
    // Graph graph;
    if (node_status.status() == interface::NodeStatus::READY and node.received)
    {
      state = node.start_state;
      rest_lengths = node.rest_lengths;
      motor_speeds = node.motor_speeds;
      first = false;
    }
    // meassurements = sensors_callback.get_last_meassurements();
    // if (meassurements and node_status.status() == interface::NodeStatus::RUNNING)
    if (node.received and publisher_node.status() == interface::NodeStatus::FINISH)
    {
      node_status.status(interface::NodeStatus::RUNNING);
      DEBUG_PRINT
      for (int i = 0; i < sensors_callback.size(); ++i)
      {
        const estimation::sensors_callback_t::Meassurements m{ sensors_callback.at(i) };
        ctrl = std::get<1>(m);
        const ros::Time tn{ std::get<3>(m) };
        if (i == 0)
        {
          ti = tn;
        }
        const double t0_sec{ ti.toSec() };
        const double tT_sec{ tn.toSec() };
        const std::string t0str{ tensegrity::utils::convert_to<std::string>(t0_sec) };
        const std::string tTstr{ tensegrity::utils::convert_to<std::string>(tT_sec) };

        DEBUG_VARS(i, sensors_callback.size())
        DEBUG_VARS(t0str, tTstr, ctrl.transpose())
        for (double t = t0_sec; t < tT_sec; t += dt[0])
        {
          output = nn(state, ctrl, dt, rest_lengths, motor_speeds);
          // return lin_acc, ang_acc, contact_lin_acc, contact_ang_acc, toi, next_rest_lens, next_motor_speeds,
          // next_state
          //   estimation::publish_tensegrity_msg(red_bar, green_bar, blue_bar, tensegrity_bars_publisher, "world");
          rest_lengths = std::get<5>(output);
          motor_speeds = std::get<6>(output);
          state = std::get<7>(output);
        }
        // lin_acc (9, ) 0-9
        // ang_acc (9, ) 9-17
        // contact_lin_acc (9, ) 18-
        // contact_ang_acc (9, ) 27
        // toi (3, ) 36
        // next_rest_lens (6, ) 39
        // next_motor_speeds (6, ) 45
        // next_state (39, ) 51-90
        // Eigen::Vector<double, 9> lin_acc{ output.segment<9>(0) };  // 0,1,2,3,4,5,6,7,8
        // Eigen::Vector<double, 9> ang_acc{ output.segment<9>(9) };  // 9,10,11,12,13,14,15,16,17
        // rest_lengths = output.segment<6>(39);
        // motor_speeds = output.segment<6>(45);
        // next_state = output.segment<39>(51);
        // Eigen::Vector<double, 9> ang_acc{ output.segment<9>(18) };  // 9,10,11,12,13,14,15,16,17

        // v_{t+1/2} = v_t + dt_0 * a_t
        // x_{t+1/2} = x_t + dt_0 * v_{t+1/2}
        // v_{t+1} = v_{t+1/2} + dt_1 * a_{t+1/2}
        // x_{t+1} = x_t + dt_1 * v_{t+1}
        // where dt_0 + dt_1 = dt
        Eigen::Vector3d q0{ state.segment<3>(0) };  // 0,1,2
        Eigen::Vector4d rot0(state.segment<4>(3));  // 3,4,5,6
        // Eigen::Vector3d q0dot{ state.segment<3>(7) };   // 7,8,9
        // Eigen::Vector3d r0dot{ state.segment<3>(10) };  // 10,11,12

        Eigen::Vector3d q1{ state.segment<3>(13) };  // 13,14,15
        Eigen::Vector4d rot1(state.segment<4>(16));  // 16,17,18,19
        // Eigen::Vector3d q1dot{ state.segment<3>(20) };  // 20,21,22
        // Eigen::Vector3d r1dot{ state.segment<3>(23) };  // 23,24,25

        Eigen::Vector3d q2{ state.segment<3>(26) };  // 26,27,28
        Eigen::Vector4d rot2(state.segment<4>(29));  // 29,30,31,32
        // Eigen::Vector3d q2dot{ state.segment<3>(33) };  // 33,34,35
        // Eigen::Vector3d r2dot{ state.segment<3>(36) };  // 36,37,38

        // Eigen::Vector<double, 6> qr0dot, qr1dot, qr2dot;

        gtsam::Pose3 x0(gtsam::Rot3(rot0[0], rot0[1], rot0[2], rot0[3]), q0);
        gtsam::Pose3 x1(gtsam::Rot3(rot1[0], rot1[1], rot1[2], rot1[3]), q1);
        gtsam::Pose3 x2(gtsam::Rot3(rot2[0], rot2[1], rot2[2], rot2[3]), q2);

        DEBUG_VARS(q0.transpose());
        DEBUG_VARS(q1.transpose());
        DEBUG_VARS(q2.transpose());
        // q0dot = q0dot + lin_acc.segment<3>(0) * dt[0];
        // q1dot = q1dot + lin_acc.segment<3>(3) * dt[0];
        // q2dot = q2dot + lin_acc.segment<3>(6) * dt[0];

        // r0dot = r0dot + ang_acc.segment<3>(0) * dt[0];
        // r1dot = r1dot + ang_acc.segment<3>(3) * dt[0];
        // r2dot = r2dot + ang_acc.segment<3>(6) * dt[0];

        // qr0dot << r0dot, q0dot;
        // qr1dot << r1dot, q1dot;
        // qr2dot << r2dot, q2dot;
        // gtsam::Pose3 next0(rot0, q0);
        // gtsam::Pose3 next1(rot1, q1);
        // gtsam::Pose3 next2(rot2, q2);
        // Eigen::Quaterniond q0_next{ next0.rotation().toQuaternion() };
        // Eigen::Quaterniond q1_next{ next1.rotation().toQuaternion() };
        // Eigen::Quaterniond q2_next{ next2.rotation().toQuaternion() };

        // state.segment<3>(0) = next0.translation();
        // state[3] = q0_next.w();
        // state[4] = q0_next.x();
        // state[5] = q0_next.y();
        // state[6] = q0_next.z();
        // state.segment<3>(7) = q0dot;
        // state.segment<3>(10) = r0dot;

        // state.segment<3>(13) = next1.translation();
        // state[16] = q1_next.w();
        // state[17] = q1_next.x();
        // state[18] = q1_next.y();
        // state[19] = q1_next.z();
        // state.segment<3>(20) = q1dot;
        // state.segment<3>(23) = r1dot;

        // state.segment<3>(26) = next2.translation();
        // state[29] = q2_next.w();
        // state[30] = q2_next.x();
        // state[31] = q2_next.y();
        // state[32] = q2_next.z();
        // state.segment<3>(33) = q2dot;
        // state.segment<3>(36) = r2dot;

        estimation::publish_tensegrity_msg(x0, x1, x2, tensegrity_bars_publisher, "world");

        ti = tn;
      }
      break;
    }

    ros::spinOnce();
    rate.sleep();
  }

  return 0;
}
