#include <iostream>
#include <array>
#include <numeric>
#include <algorithm>
#include <tuple>
#include <tensegrity_utils/rosparams_utils.hpp>
#include <tensegrity_utils/dbg_utils.hpp>
#include <tensegrity_utils/gnn.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>

#include <dynamic_reconfigure/server.h>
#include <perception/ColorFilterConfig.h>
#include <perception/utils.hpp>

#include <interface/node_status.hpp>
#include <interface/TensegrityBars.h>
#include <interface/type_conversions.hpp>

#include <interface/type_conversions.hpp>
#include <estimation/bar_utilities.hpp>

#include <gtsam/config.h>
#include <gtsam/base/Testable.h>
#include <gtsam/nonlinear/Expression.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/slam/BetweenFactor.h>

#include <factor_graphs/defs.hpp>
#include <factor_graphs/constraint_factor.hpp>

#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/features2d.hpp>

#include <visualization_msgs/Marker.h>

using Rotation = gtsam::Rot3;
using Translation = Eigen::Vector3d;
using Color = Eigen::Vector<double, 3>;
using TranslationColor = Eigen::Vector<double, 6>;
using Camera = gtsam::PinholeCamera<gtsam::Cal3_S2>;
// using Camera = gtsam::PinholeCamera<gtsam::Cal3_S2>;
// using GnnPt = Eigen::Vector4d;
using Metric = utils::eucledian_metric_t<Translation>;
using Gnn = utils::graph_nearest_neighbors_t<Translation, Metric>;
using NodePtr = Gnn::NodePtr;

using PositiveDepthFactor = factor_graphs::constraint_factor_t<double, std::less<double>>;

const Translation offset({ 0, 0, 0.325 / 2.0 });
const Rotation Roffset(0.0, 0.0, 1.0, 0.0);
const Color black(Color::Zero());
const Color blue(1.0, 0.0, 0.0);
const Color green(0.0, 1.0, 0.0);
const Color red(0.0, 0.0, 1.0);

struct tensegrity_3d_icp_t
{
  using This = tensegrity_3d_icp_t;
  int max_iterations;
  // gtsam::Pose3 red_pose, green_pose, blue_pose;
  std::array<gtsam::Pose3, 3> poses;

  cv::Scalar low_red;
  cv::Scalar high_red;
  cv::Scalar low_green;
  cv::Scalar high_green;
  cv::Scalar low_blue;
  cv::Scalar high_blue;
  cv::Scalar low_black;
  cv::Scalar high_black;

  int max_pts;
  double depth_scale;
  std::string image_topic, depth_topic;
  cv_bridge::CvImagePtr depth_frame;

  ros::Subscriber image_subscriber, depth_subscriber, tensegrity_bars_subscriber;
  cv::Mat img_hsv, img_depth;
  cv::Mat _frame_red0, _frame_green, _frame_blue, _frame_black;

  std::vector<cv::Mat> elements;
  gtsam::PinholeCamera<gtsam::Cal3_S2> camera;

  bool use_between_factor;
  bool bars_poses_received, rgb_received, depth_received;
  gtsam::LevenbergMarquardtParams lm_params;
  std::shared_ptr<factor_graphs::levenberg_marquardt_t> lm_helper;
  std::shared_ptr<interface::node_status_t> _node_status;

  std::vector<std::string> initial_poses_params;
  ros::Publisher tensegrity_bars_publisher;
  std::string type;
  std::string initial_filename;

  tensegrity_3d_icp_t(ros::NodeHandle& nh)
    : max_iterations(10)
    , low_red(160, 153, 57)
    , high_red(179, 255, 150)
    , low_green(70, 100, 30)
    , high_green(95, 255, 95)
    , low_blue(94, 62, 45)
    , high_blue(151, 255, 255)
    , low_black(0, 0, 0)
    , high_black(179, 255, 60)
    , bars_poses_received(false)
    , rgb_received(false)
    , depth_received(false)
    , initial_filename("")
  {
    std::string tensegrity_pose_topic;

    lm_params.setVerbosityLM("SILENT");
    // lm_params.setVerbosityLM("SUMMARY");
    lm_params.setMaxIterations(10);
    lm_helper = std::make_shared<factor_graphs::levenberg_marquardt_t>(nh, "/nodes/icp/fg", lm_params);

    _node_status = interface::node_status_t::create(nh, false);

    // PARAM_SETUP(nh, image_topic);
    // PARAM_SETUP(nh, depth_topic);
    // PARAM_SETUP(nh, max_pts);
    // PARAM_SETUP(nh, depth_scale);

    PARAM_SETUP(nh, tensegrity_pose_topic);
    // PARAM_SETUP(nh, initial_file);
    // PARAM_SETUP(nh, use_between_factor);
    PARAM_SETUP(nh, type);
    PARAM_SETUP(nh, initial_poses_params);
    PARAM_SETUP_WITH_DEFAULT(nh, initial_filename, initial_filename);

    tensegrity_bars_publisher = nh.advertise<interface::TensegrityBars>(tensegrity_pose_topic, 1, true);

    // image_subscriber = nh.subscribe(image_topic, 1, &This::image_callback, this);
    // depth_subscriber = nh.subscribe(depth_topic, 1, &This::depth_callback, this);

    // Eigen::Matrix4d extrinsic;
    // extrinsic << 1.0, 0.0, -0.01, 0.743, 0.0, -1.0, -0.007, 0.082, -0.01, 0.007, -1.0, 1.441, 0.0, 0.0, 0.0, 1.0;
    // gtsam::Cal3_S2 camera_calibration(901.67626953125, 901.7260131835938, 0.0, 640.0, 360.60618591308594);
    // camera = Camera(gtsam::Pose3(extrinsic), camera_calibration);
    _node_status->status(interface::NodeStatus::RUNNING);
  }

  void initialize_params()
  {
    if (type == "file")
    {
      const bool res{ init_from_file(initial_filename) };
      TENSEGRITY_ASSERT(res, "File initialization failed!");
      send_params();
      _node_status->status(interface::NodeStatus::FINISH);
    }
    else
    {
      PRINT_MSG("Unknown type initialization");
      ros::shutdown();
    }
  }

  void send_params()
  {
    for (int i = 0; i < initial_poses_params.size(); ++i)
    {
      std::vector<double> params_out;
      gtsam::Quaternion quat{ poses[i].rotation().toQuaternion() };
      Eigen::Vector3d t{ poses[i].translation() };
      params_out.push_back(quat.w());
      params_out.push_back(quat.x());
      params_out.push_back(quat.y());
      params_out.push_back(quat.z());
      params_out.push_back(t[0]);
      params_out.push_back(t[1]);
      params_out.push_back(t[2]);
      ros::param::set(initial_poses_params[i], params_out);
    }
    estimation::publish_tensegrity_msg(poses[0], poses[1], poses[2], tensegrity_bars_publisher, "world", 0);
  }

  bool init_from_file(const std::string filename)
  {
    if (not std::filesystem::exists(filename))
    {
      return false;
    }
    tensegrity::utils::csv_reader_t reader(filename);

    // Assuming (0,1) -> red; (2,3) -> green; (4,5) -> blue
    std::array<Eigen::Vector3d, 6> estimates;
    while (reader.has_next_line())
    {
      auto line = reader.next_line();
      if (line.size() >= 5)
      {
        if (line[0] == "#")
          continue;

        int idx{ tensegrity::utils::convert_to<int>(line[0]) };
        double x{ tensegrity::utils::convert_to<double>(line[2]) };
        double y{ tensegrity::utils::convert_to<double>(line[3]) };
        double z{ tensegrity::utils::convert_to<double>(line[4]) };
        estimates[idx] = Eigen::Vector3d(x, y, z);
      }
    }

    gtsam::Values values;
    gtsam::NonlinearFactorGraph graph;
    estimation::observation_update_t observation_update;
    observation_update.offset = offset;
    observation_update.Roffset = Roffset;
    observation_update.pose = gtsam::Pose3();

    observation_update.zA = estimates[0];
    observation_update.zB = estimates[1];
    observation_update.color = estimation::RodColors::RED;
    estimation::add_two_observations(observation_update, graph, values);
    const gtsam::Key red_key{ observation_update.key_se3 };

    observation_update.zA = estimates[2];
    observation_update.zB = estimates[3];
    observation_update.color = estimation::RodColors::GREEN;
    estimation::add_two_observations(observation_update, graph, values);
    const gtsam::Key green_key{ observation_update.key_se3 };

    observation_update.zA = estimates[4];
    observation_update.zB = estimates[5];
    observation_update.color = estimation::RodColors::BLUE;
    estimation::add_two_observations(observation_update, graph, values);
    const gtsam::Key blue_key{ observation_update.key_se3 };

    const gtsam::Values result{ lm_helper->optimize(graph, values, true) };

    poses[0] = result.at<gtsam::Pose3>(red_key);
    poses[1] = result.at<gtsam::Pose3>(green_key);
    poses[2] = result.at<gtsam::Pose3>(blue_key);

    return true;
  }

  // void image_callback(const sensor_msgs::ImageConstPtr message)
  // {
  //   cv_bridge::CvImageConstPtr frame{ cv_bridge::toCvShare(message) };
  //   cv::cvtColor(frame->image, img_hsv, cv::COLOR_BGR2HSV);
  //   rgb_received = true;
  // }
  // void depth_callback(const sensor_msgs::ImageConstPtr message)
  // {
  //   cv_bridge::CvImageConstPtr frame{ cv_bridge::toCvShare(message) };
  //   frame->image.copyTo(img_depth);
  //   depth_received = true;
  // }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "img_diff_test");
  ros::NodeHandle nh("~");

  tensegrity_3d_icp_t node(nh);
  while (ros::ok())
  {
    if (node._node_status->status() == interface::NodeStatus::RUNNING)
    {
      node.initialize_params();
    }
    else if (node._node_status->status() == interface::NodeStatus::FINISH)
    {
      break;
    }

    ros::spinOnce();
  }
  ros::spin();
}
