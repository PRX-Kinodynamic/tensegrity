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
#include <gtsam/geometry/Similarity2.h>
#include <gtsam/nonlinear/Marginals.h>

#include <factor_graphs/defs.hpp>
#include <factor_graphs/constraint_factor.hpp>
#include <estimation/bars_perception_factors.hpp>

#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <interface/TensegrityBarsArray.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <interface/ros_camera_interface.hpp>

#include <dynamic_reconfigure/server.h>
#include <perception/TensegrityInitializationConfig.h>

using Rotation = gtsam::Rot3;
using Pixel = Eigen::Vector2d;
using Translation = Eigen::Vector3d;
using Color = Eigen::Vector<double, 3>;
using TranslationColor = Eigen::Vector<double, 6>;
// using Camera = gtsam::PinholeCamera<gtsam::Cal3_S2>;
// using CameraPtr = std::shared_ptr<Camera>;

// struct Metric
// {
//   double operator()(const TranslationColor& a, const TranslationColor& b) const
//   {
//     const double err{ (a.head(2) - b.head(2)).norm() };
//     const double color_err{ (a.tail(3) - b.tail(3)).norm() < 0.1 ? 1.0 : 3.0 };
//     return err * color_err;
//   }
// };
using Metric = utils::eucledian_metric_t<Pixel>;
using Gnn = utils::graph_nearest_neighbors_t<Pixel, Metric>;
using NodePtr = Gnn::NodePtr;
using CvMatPtr = std::shared_ptr<cv::Mat>;
using ScaleBeliefs = std::vector<std::pair<double, double>>;
using EndcapsBeliefs = std::vector<std::pair<Pixel, Eigen::Matrix2d>>;
using PositiveDepthFactor = factor_graphs::constraint_factor_t<double, std::less<double>>;

const Color black(Color::Zero());
const Color blue(0.0, 0.0, 1.0);
const Color green(0.0, 1.0, 0.0);
const Color red(1.0, 0.0, 0.0);
const Pixel px_offset({ 0.325 / 2.0, 0.0 });

void publish_img(cv::Mat& img, ros::Publisher& pub, std::string encoding)
{
  auto msg = cv_bridge::CvImage(std_msgs::Header(), encoding, img).toImageMsg();
  pub.publish(msg);
}

cv::Mat transform_image(gtsam::Pose2& tf, ros::Publisher& pub, cv::Mat& img, cv::Mat& img_target)
{
  cv::Mat cv_tf, img_tf;
  Eigen::MatrixXd tf_mat{ tf.matrix().block(0, 0, 2, 3) };
  cv::eigen2cv(tf_mat, cv_tf);
  // DEBUG_VARS(tf_mat);
  // DEBUG_VARS(cv_tf);
  cv::warpAffine(img, img_tf, cv_tf, img.size());
  img_tf += img_target;
  publish_img(img_tf, pub, "mono8");
  return img_tf;
}

struct pointcloud_from_imgs_t
{
  using This = pointcloud_from_imgs_t;
  using DynReconfServer = dynamic_reconfigure::Server<perception::TensegrityInitializationConfig>;

  int max_iterations;
  std::vector<gtsam::Pose2> poses2;
  std::vector<double> scales, scales_cov;
  gtsam::Pose3 red_pose, green_pose, blue_pose;

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
  cv::Mat _frame_black;
  cv::Mat img_bars_estimation;
  std::array<cv::Mat, 3> _frame_colors;

  std::vector<cv::Mat> elements;
  // CameraPtr camera;

  ros::Publisher pub0_rgb_masks;
  ros::Publisher pub0_all_masks;
  ros::Publisher pub_red_marker;
  ros::Publisher pub_green_marker;
  ros::Publisher pub_blue_marker;
  ros::Publisher pub_img_red_marker;
  ros::Publisher pub_red_matches, points_marker_pub;
  ros::Publisher pub_green_matches, pub_weights, pub_elipses, pub_est_bars;
  ros::Publisher pub_blue_matches, pub_img_marker, pub_lines;
  ros::Publisher pub_sobel_x, pub_sobel_all;
  ros::Publisher clustered_black;
  std::array<ros::Publisher, 3> pts_covs_markers_pub;
  std::array<ros::Publisher, 6> estimation_pub;
  std::array<ros::Publisher, 3> pub_color_masks;  //, pub_green_masks, pub_blue_masks;

  cv::Mat img_rgb;
  bool use_between_factor;
  bool bars_poses_received, rgb_received, depth_received;
  gtsam::LevenbergMarquardtParams lm_params;
  std::shared_ptr<factor_graphs::levenberg_marquardt_t> lm_helper;

  int total_image_points;
  int total_sample_bars;
  const std::array<Color, 3> colors;
  const uint8_t red_idx, green_idx, blue_idx;
  const std::vector<double> xi_vals;
  bool visualize;
  std::vector<Eigen::Vector<double, 3>> points;
  std::vector<Eigen::Vector<double, 3>> point_colors;

  std::array<Eigen::Vector3d, 6> _last_estimate;
  std::array<bool, 6> _valid_estimate;

  ros::Timer _timer;

  int total_points;
  cv::Mat abs_grad_x, abs_grad_y;
  std::shared_ptr<interface::ros_camera_interface_t> _camera_interface;

  std::array<std::shared_ptr<DynReconfServer>, 6> _servers;
  std::array<bool, 6> _save_inits;
  CvMatPtr all_masks;

  std::shared_ptr<interface::node_status_t> _node_status;
  std::string initial_states_file;

  Translation _offset;
  Rotation _Roffset;

  pointcloud_from_imgs_t(ros::NodeHandle& nh)
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
    , colors({ red, green, blue })
    // , colors({ blue, green, red })
    , _offset(0, 0, 0.325 / 2.0)
    , _Roffset(0.0, 0.0, 1.0, 0.0)
    , red_idx(0)
    , green_idx(1)
    , blue_idx(2)
    , xi_vals({ 10.828, 13.816, 16.266, 18.467, 20.515, 22.458, 24.322, 26.124, 27.877, 29.588 })
    , visualize(false)
  {
    std::string tensegrity_pose_topic;

    lm_params.setVerbosityLM("SILENT");
    // lm_params.setVerbosityLM("SUMMARY");
    lm_params.setMaxIterations(10);
    lm_helper = std::make_shared<factor_graphs::levenberg_marquardt_t>(nh, "/nodes/icp/fg", lm_params);

    double frequency{ 15 };

    PARAM_SETUP(nh, image_topic);
    PARAM_SETUP(nh, depth_topic);
    PARAM_SETUP(nh, depth_scale);
    PARAM_SETUP(nh, visualize);
    PARAM_SETUP(nh, initial_states_file);
    PARAM_SETUP_WITH_DEFAULT(nh, frequency, frequency);

    _node_status = interface::node_status_t::create(nh, false);
    _camera_interface = std::make_shared<interface::ros_camera_interface_t>(nh);
    // PARAM_SETUP(nh, tensegrity_pose_topic);
    // PARAM_SETUP(nh, use_between_factor);
    // PARAM_SETUP(nh, total_sample_bars);
    // PARAM_SETUP(nh, total_image_points);
    // PARAM_SETUP(nh, depth_scale);

    // tensegrity_bars_subscriber = nh.subscribe(tensegrity_pose_topic, 1, &This::pose_callback, this);

    pub0_rgb_masks = nh.advertise<sensor_msgs::Image>("/img/masks/rgb", 1, true);
    pub0_all_masks = nh.advertise<sensor_msgs::Image>("/img/masks/all", 1, true);
    pub_red_marker = nh.advertise<visualization_msgs::Marker>("/bars/red", 1, true);
    pub_green_marker = nh.advertise<visualization_msgs::Marker>("/bars/green", 1, true);
    pub_blue_marker = nh.advertise<visualization_msgs::Marker>("/bars/blue", 1, true);
    pub_img_red_marker = nh.advertise<visualization_msgs::Marker>("/bars/red_img", 1, true);
    pub_red_matches = nh.advertise<visualization_msgs::Marker>("/bars/red/matches", 1, true);
    pub_green_matches = nh.advertise<visualization_msgs::Marker>("/bars/green/matches", 1, true);
    pub_blue_matches = nh.advertise<visualization_msgs::Marker>("/bars/blue/matches", 1, true);
    pub_img_marker = nh.advertise<visualization_msgs::Marker>("/img/pointcloud", 1, true);
    pub_lines = nh.advertise<sensor_msgs::Image>("/img/sobel/diff", 1, true);
    pub_sobel_x = nh.advertise<sensor_msgs::Image>("/img/sobel/x", 1, true);
    pub_sobel_all = nh.advertise<sensor_msgs::Image>("/img/sobel/all", 1, true);
    // pub_red_masks = nh.advertise<sensor_msgs::Image>("/img/mask/red/", 1, true);
    // pub_green_masks = nh.advertise<sensor_msgs::Image>("/img/mask/green/", 1, true);
    // pub_blue_masks = nh.advertise<sensor_msgs::Image>("/img/mask/blue/", 1, true);
    pub_weights = nh.advertise<sensor_msgs::Image>("/img/weights", 1, true);
    pub_elipses = nh.advertise<sensor_msgs::Image>("/img/ellipses", 1, true);
    pub_est_bars = nh.advertise<sensor_msgs::Image>("/img/bars/estimated", 1, true);
    points_marker_pub = nh.advertise<visualization_msgs::Marker>("/pointcloud", 1, true);
    clustered_black = nh.advertise<visualization_msgs::MarkerArray>("/bars/clustered", 1, true);
    // pts_covs_markers_red_pub = nh.advertise<visualization_msgs::MarkerArray>("/endcaps/estimated/red", 1, true);
    // points_marker_pub = nh.advertise<visualization_msgs::Marker>("/pointcloud", 1, true);
    for (int i = 0; i < 3; ++i)
    {
      pts_covs_markers_pub[i] =
          nh.advertise<visualization_msgs::MarkerArray>("/endcaps/clustered/" + std::to_string(i), 1, true);
    }

    for (int i = 0; i < 6; ++i)
    {
      estimation_pub[i] = nh.advertise<visualization_msgs::Marker>("/endcaps/estimated/" + std::to_string(i), 1, true);
    }

    image_subscriber = nh.subscribe(image_topic, 1, &This::image_callback, this);
    depth_subscriber = nh.subscribe(depth_topic, 1, &This::depth_callback, this);

    std::vector<int> kernel_sizes{ { 2, 5 } };
    for (int i = 0; i < kernel_sizes.size(); ++i)
    {
      const int erosion_size{ kernel_sizes[i] };
      elements.push_back(cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                   cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                                                   cv::Point(erosion_size, erosion_size)));
    }
    const ros::Duration timer(1.0 / frequency);
    _timer = nh.createTimer(timer, &This::timer_callback, this);

    const bool initialized{ init_from_file(initial_states_file) };

    if (initialized)
    {
      _node_status->status(interface::NodeStatus::READY);
    }
    else
    {
      PRINT_MSG("Running Manual initialization");
      for (int i = 0; i < 6; ++i)
      {
        ros::NodeHandle nh_i("/endcaps/init/" + std::to_string(i));

        _save_inits[i] = false;
        _servers[i] = std::make_shared<DynReconfServer>(nh_i);
        // f = boost::bind(&Derived::cfg_callback, this, _1, _2);
        _servers[i]->setCallback(boost::bind(&This::cfg_callback, this, _1, _2, i));
      }
    }
  }
  void cfg_callback(perception::TensegrityInitializationConfig& config, uint32_t level, int endcap)
  {
    if (_camera_interface->valid())
    {
      const Eigen::Vector2d px{ config.x, config.y };
      const Eigen::Vector3d pt{ _camera_interface->camera()->backproject(px, config.z) };
      _save_inits[endcap] = config.save;

      _last_estimate[endcap] = pt;
    }
    else
    {
      PRINT_MSG_ONCE("Camera info has not been received!")
    }
  }

  void timer_callback(const ros::TimerEvent& event)
  {
    if (_node_status->status() == interface::NodeStatus::PREPARING)
    {
      manual_initialization();
    }
    else if (_node_status->status() == interface::NodeStatus::RUNNING and rgb_received and depth_received)
    {
      // PRINT_MSG("Running icp");
      const auto start{ std::chrono::steady_clock::now() };
      run_icp();
      const auto finish{ std::chrono::steady_clock::now() };
      const std::chrono::duration<double> elapsed{ finish - start };
      const double iteration_dt{ elapsed.count() };
      DEBUG_VARS(iteration_dt);
      // DEBUG_VARS(iter_dt)
      rgb_received = false;
      depth_received = false;
    }
    if (visualize)
    {
      for (int i = 0; i < 6; ++i)
      {
        if (_valid_estimate[i])
          point_to_marker(_last_estimate[i], colors[i / 2], estimation_pub[i]);
      }
    }
    // ros::spinOnce();
  }

  void manual_initialization()
  {
    if (rgb_received and depth_received)
    {
      compute_masks();
      get_subimage(all_masks);

      points_to_marker(points, point_colors, points_marker_pub);

      bool save{ true };
      for (int i = 0; i < 6; ++i)
      {
        save &= _save_inits[i];
      }
      if (save)
      {
        _node_status->status(interface::NodeStatus::READY);
        init_to_file(initial_states_file);
      }
    }
  }

  void init_to_file(const std::string filename)
  {
    using namespace tensegrity::utils;
    std::ofstream out_file(filename, std::ios::out);

    std::array<std::string, 6> colors_str{ "red", "red", "green", "green", "blue", "blue" };
    for (int i = 0; i < 6; ++i)
    {
      out_file << convert_to<std::string>(i) << " " << colors_str[i] << " ";
      out_file << convert_to<std::string>(_last_estimate[i][0]) << " ";
      out_file << convert_to<std::string>(_last_estimate[i][1]) << " ";
      out_file << convert_to<std::string>(_last_estimate[i][2]) << "\n";
    }
    out_file.close();
  }

  bool init_from_file(const std::string filename)
  {
    if (not std::filesystem::exists(filename))
    {
      // const std::string msg = "File [" + filename + "] does not exists";
      // PRINT_MSG("File [" + filename + "] does not exists")
      // ros::shutdown();
      return false;
    }
    tensegrity::utils::csv_reader_t reader(filename);

    // Assuming (0,1) -> red; (2,3) -> green; (4,5) -> blue
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
        _last_estimate[idx] = Eigen::Vector3d(x, y, z);
      }
    }

    return true;
  }

  void image_callback(const sensor_msgs::ImageConstPtr message)
  {
    cv_bridge::CvImageConstPtr frame{ cv_bridge::toCvShare(message) };
    cv::cvtColor(frame->image, img_hsv, cv::COLOR_BGR2HSV);
    frame->image.copyTo(img_rgb);

    rgb_received = true;
  }
  void depth_callback(const sensor_msgs::ImageConstPtr message)
  {
    cv_bridge::CvImageConstPtr frame{ cv_bridge::toCvShare(message) };
    frame->image.copyTo(img_depth);
    // frame->image.copyTo(abs_grad_x);
    // CvMatPtr grad_x{ std::make_shared<cv::Mat>() };

    // cv::Mat grad_x, grad_y;
    // cv::Sobel(frame->image, grad_x, CV_16S, 1, 0);
    // cv::Sobel(frame->image, grad_y, CV_16S, 0, 1);
    // grad_x = grad_x + grad_y;
    // // // // converting back to CV_8U
    // cv::convertScaleAbs(grad_x, abs_grad_x);
    // cv::threshold(abs_grad_x, abs_grad_x, 127, 255, cv::THRESH_BINARY);
    // // // cv::convertScaleAbs(grad_y, abs_grad_y);
    // publish_img(abs_grad_x, pub_sobel_x, "mono8");
    // publish_img(abs_grad_y, pub_sobel_y, "mono8");

    depth_received = true;
  }

  void point_to_marker(const Eigen::Vector3d& pt, const Eigen::Vector3d& pt_color, ros::Publisher& pub)
  {
    visualization_msgs::Marker marker;
    tensegrity::utils::init_header(marker.header, "world");
    marker.action = visualization_msgs::Marker::ADD;
    marker.type = visualization_msgs::Marker::SPHERE;

    marker.scale.x = 0.04;
    marker.scale.y = 0.04;
    marker.scale.z = 0.04;

    marker.pose.orientation.w = 1.0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;

    marker.pose.position.x = pt[0];
    marker.pose.position.y = pt[1];
    marker.pose.position.z = pt[2];
    marker.color.r = pt_color[0];
    marker.color.g = pt_color[1];
    marker.color.b = pt_color[2];
    marker.color.a = 0.8;

    pub.publish(marker);
  }

  void points_to_marker(std::vector<Eigen::Vector3d>& pts, std::vector<Eigen::Vector3d>& pts_colors,
                        ros::Publisher& pub)
  {
    visualization_msgs::Marker marker;
    tensegrity::utils::init_header(marker.header, "world");
    marker.action = visualization_msgs::Marker::ADD;
    marker.type = visualization_msgs::Marker::POINTS;

    marker.scale.x = 0.01;  // is point width,
    marker.scale.y = 0.01;  // is point height
    // interface::copy(marker.pose, pose);
    // for (auto p : pc)
    for (int i = 0; i < points.size(); ++i)
    {
      marker.points.emplace_back();
      marker.points.back().x = pts[i][0];
      marker.points.back().y = pts[i][1];
      marker.points.back().z = pts[i][2];
      marker.colors.emplace_back();
      marker.colors.back().r = pts_colors[i][0];
      marker.colors.back().g = pts_colors[i][1];
      marker.colors.back().b = pts_colors[i][2];
      marker.colors.back().a = 1.0;
    }
    pub.publish(marker);
  }

  void points_with_cov_to_marker(const std::vector<Eigen::Vector3d>& pts, const Eigen::Vector3d pts_colors,
                                 const std::vector<Eigen::Matrix3d>& covs, ros::Publisher& pub)
  {
    visualization_msgs::Marker marker;
    visualization_msgs::MarkerArray all_markers;

    marker.action = visualization_msgs::Marker::DELETEALL;
    all_markers.markers.emplace_back(marker);
    pub.publish(all_markers);
    all_markers.markers.clear();

    tensegrity::utils::init_header(marker.header, "world");
    marker.action = visualization_msgs::Marker::ADD;
    marker.type = visualization_msgs::Marker::SPHERE;

    // interface::copy(marker.pose, pose);
    // for (auto p : pc)
    marker.pose.orientation.w = 1.0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.id = 0;

    for (int i = 0; i < pts.size(); ++i)
    {
      marker.scale.x = std::max(covs[i](0, 0), 0.01);  // If cov is too low, it will not be visible, so cap it to 1cm
      marker.scale.y = std::max(covs[i](1, 1), 0.01);  // If cov is too low, it will not be visible, so cap it to 1cm
      marker.scale.z = std::max(covs[i](2, 2), 0.01);  // If cov is too low, it will not be visible, so cap it to 1cm
      marker.pose.position.x = pts[i][0];
      marker.pose.position.y = pts[i][1];
      marker.pose.position.z = pts[i][2];
      marker.color.r = pts_colors[0];
      marker.color.g = pts_colors[1];
      marker.color.b = pts_colors[2];
      marker.color.a = 0.5;
      marker.id++;
      all_markers.markers.emplace_back(marker);
      // DEBUG_VARS(marker.scale.x, marker.scale.y, marker.scale.z);
    }
    DEBUG_VARS(all_markers.markers.size());
    pub.publish(all_markers);
  }

  template <typename Element, typename Covs>
  void cluster_fast(std::vector<Element>& vals_out, std::vector<Covs>& covs_out,  // no-lint
                    const std::vector<Element>& vals_in, const std::vector<Covs>& covs_in)
  {
    using GaussianNM = gtsam::noiseModel::Gaussian;
    // DEBUG_VARS(endcaps.size());
    if (vals_in.size() == 0)
      return;

    gtsam::Values values;
    // gtsam::NonlinearFactorGraph graph;

    double prev_error{ 0 };
    double adjusted_error{ 0 };
    std::vector<Element> rejected;
    std::vector<Covs> rejected_covs;
    gtsam::Key key{ gtsam::Symbol('X', 0) };
    values.insert(key, vals_in[0]);
    std::size_t clustered{ 0 };

    gtsam::JacobianFactor::shared_ptr prior;
    gtsam::Ordering key_ordering;
    key_ordering += key;

    int dummy;
    for (int i = 0; i < vals_in.size(); ++i)
    {
      // Linearize measurement factor and add it to the Kalman Filter graph
      // Values linearizationPoint;
      // values.insert(keys[0], x_);

      clustered++;
      // gtsam::NonlinearFactorGraph gDelta;

      const Element zi{ vals_in[i] };
      GaussianNM::shared_ptr z_noise{ GaussianNM::Information(covs_in[i].inverse()) };

      gtsam::GaussianFactorGraph linearFactorGraph;
      linearFactorGraph.push_back(prior);
      gtsam::PriorFactor<Element> curr_prior(key, zi, z_noise);
      linearFactorGraph.push_back(curr_prior.linearize(values));
      const gtsam::GaussianConditional::shared_ptr marginal{
        linearFactorGraph.marginalMultifrontalBayesNet(key_ordering)->front()
      };
      const gtsam::VectorValues result{ marginal->solve(gtsam::VectorValues()) };

      // sigma_inv = covs_in[i].second;

      // solve_prior(linearFactorGraph, values, key, prior);
      // gDelta.addPrior(key, zi, z_noise);
      // gtsam::FactorIndices indices{ graph.add_factors(gDelta) };
      // gtsam::Values result{ lm_helper->optimize(graph, values, true) };
      // const double error{ graph.error(result) };
      const double error{ linearFactorGraph.error(result) };
      // adjusted_error = error;
      const double derr{ error - prev_error };
      // DEBUG_VARS(zi.transpose(), error, prev_error, derr);
      if (derr > xi_vals[std::min(clustered, xi_vals.size())])
      {
        // for (auto idx : indices)
        // {
        //   graph.remove(idx);
        // }
        // result = lm_helper->optimize(graph, values, true);
        // adjusted_error = graph.error(result);
        rejected.push_back(vals_in[i]);
        rejected_covs.push_back(covs_in[i]);
        clustered--;
      }
      else
      {
        const Element& current{ values.at<Element>(key) };
        const Element x{ gtsam::traits<Element>::Retract(current, result[key]) };
        values.update(key, x);
        prior = boost::make_shared<gtsam::JacobianFactor>(
            marginal->keys().front(), marginal->getA(marginal->begin()),
            marginal->getb() - marginal->getA(marginal->begin()) * result[key], marginal->get_model());
        // const Eigen::Matrix<double, 1, 1> scale_sigma{ scale_beliefs[i].second };
        prev_error = error;
      }
      // std::cin >> dummy;
    }
    // const gtsam::Marginals marginals(graph, values);
    const Element res{ values.at<Element>(key) };
    const Covs res_cov{ prior->information().inverse() };
    // const Covs res_cov{ marginals.marginalCovariance(key) };

    vals_out.push_back(res);
    covs_out.push_back(res_cov);
    if (rejected.size() > 0)
    {
      // DEBUG_VARS(rejected.size());
      cluster_fast(vals_out, covs_out, rejected, rejected_covs);
    }
  }

  template <typename Element, typename Covs>
  void cluster(std::vector<Element>& vals_out, std::vector<Covs>& covs_out,  // no-lint
               const std::vector<Element>& vals_in, const std::vector<Covs>& covs_in)
  {
    PRINT_MSG("CLUSTER");
    using GaussianNM = gtsam::noiseModel::Gaussian;
    // DEBUG_VARS(endcaps.size());
    if (vals_in.size() == 0)
      return;

    gtsam::Values values;
    gtsam::NonlinearFactorGraph graph;

    double prev_error{ 0 };
    double adjusted_error{ 0 };
    std::vector<Element> rejected;
    std::vector<Covs> rejected_covs;
    gtsam::Key key{ gtsam::Symbol('X', 0) };
    values.insert(key, vals_in[0]);
    std::size_t clustered{ 0 };

    for (int i = 0; i < vals_in.size(); ++i)
    {
      clustered++;
      gtsam::NonlinearFactorGraph gDelta;

      const Element zi{ vals_in[i] };
      // sigma_inv = covs_in[i].second;
      GaussianNM::shared_ptr z_noise{ GaussianNM::Information(covs_in[i].inverse()) };

      gDelta.addPrior(key, zi, z_noise);
      gtsam::FactorIndices indices{ graph.add_factors(gDelta) };
      gtsam::Values result{ lm_helper->optimize(graph, values, true) };
      const double error{ graph.error(result) };
      adjusted_error = error;
      const double derr{ error - prev_error };
      // DEBUG_VARS(error, prev_error, derr);
      if (derr > xi_vals[std::min(clustered, xi_vals.size())])
      {
        for (auto idx : indices)
        {
          graph.remove(idx);
        }
        result = lm_helper->optimize(graph, values, true);
        adjusted_error = graph.error(result);
        rejected.push_back(vals_in[i]);
        rejected_covs.push_back(covs_in[i]);
        clustered--;
      }
      else
      {
        values = result;
        // const Eigen::Matrix<double, 1, 1> scale_sigma{ scale_beliefs[i].second };
      }

      prev_error = adjusted_error;
    }
    const gtsam::Marginals marginals(graph, values);
    const Element res{ values.at<Element>(key) };
    const Covs res_cov{ marginals.marginalCovariance(key) };

    vals_out.push_back(res);
    covs_out.push_back(res_cov);
    if (rejected.size() > 0)
    {
      cluster(vals_out, covs_out, rejected, rejected_covs);
    }
  }

  template <typename Element, typename Covs>
  void cluster_all(std::vector<Element>& vals_out, std::vector<Covs>& covs_out,  // no-lint
                   const std::vector<Element>& vals_in, const std::vector<Covs>& covs_in, int max_steps = -1)
  {
    std::vector<Element> v_in, v_out;
    std::vector<Covs> c_in, c_out;
    DEBUG_VARS(max_steps);
    std::size_t prev_size{ vals_in.size() };
    const auto start{ std::chrono::steady_clock::now() };
    cluster_fast(v_out, c_out, vals_in, covs_in);
    while (max_steps != 0 and prev_size > v_out.size())
    {
      v_in.clear();
      c_in.clear();
      v_out.swap(v_in);
      c_out.swap(c_in);
      // DEBUG_VARS(max_steps);
      // DEBUG_VARS(v_out.size(), v_in.size());
      prev_size = v_in.size();
      // const auto start{ std::chrono::steady_clock::now() };
      cluster_fast(v_out, c_out, v_in, c_in);
      // const auto finish{ std::chrono::steady_clock::now() };
      // const std::chrono::duration<double> elapsed_seconds{ finish - start };
      // DEBUG_VARS(elapsed_seconds.count());
      max_steps--;
    }
    const auto finish{ std::chrono::steady_clock::now() };
    const std::chrono::duration<double> elapsed_seconds{ finish - start };
    DEBUG_VARS(elapsed_seconds.count(), vals_in.size());
    vals_out.swap(v_out);
    covs_out.swap(c_out);
    DEBUG_VARS(max_steps);
  }

  void get_subimage(CvMatPtr& all_masks)
  {
    cv::Mat circle_mask{ cv::Mat::zeros(_frame_black.rows, _frame_black.cols, _frame_black.type()) };
    std::vector<std::vector<cv::Point>> contours_red, contours_green, contours_blue, all_contours;
    cv::findContours(*all_masks, all_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::sort(all_contours.begin(), all_contours.end(),
              [](cv::InputArray a, cv::InputArray b) { return cv::contourArea(a) > cv::contourArea(b); });
    int total_endcaps{ 0 };
    // const std::array<Color, 3> colors{ blue, green, red };
    std::array<int, 3> total_color{ 0, 0, 0 };

    cv::Moments mc(cv::moments(all_contours[0], false));
    const double xc{ mc.m10 / mc.m00 };
    const double yc{ mc.m01 / mc.m00 };
    const Pixel centroid{ xc, yc };

    const double area{ cv::contourArea(all_contours[0]) };
    const double diameter{ std::sqrt(4.0 * area / tensegrity::constants::pi) * 3.0 };

    const double cols{ static_cast<double>(_frame_colors[0].cols) };
    const double rows{ static_cast<double>(_frame_colors[0].rows) };
    const Pixel px_cp{ std::max(0.0, std::min(xc - diameter / 2, cols)),
                       std::max(0.0, std::min(yc - diameter / 2, rows)) };
    // const Pixel px_cm{ std::max(0.0, std::min(xc - diameter / 2, cols)),
    //                    std::max(0.0, std::min(yc - diameter / 2, rows)) };
    const double recp_width{ std::min(_frame_colors[0].cols - px_cp[0], diameter) };
    const double recp_height{ std::min(_frame_colors[0].rows - px_cp[1], diameter) };
    // const double recm_width{ std::min(_frame_colors[0].cols - px_cm[0], diameter) };
    // const double recm_height{ std::min(_frame_colors[0].rows - px_cm[1], diameter) };
    const cv::Rect rec(px_cp[0], px_cp[1], recp_width, recp_height);
    // const cv::Rect rec_m(px_cm[0], px_cm[1], recm_width, recm_height);

    // cv::Mat mred{ _frame_colors[red_idx](rec) };
    // cv::Mat mgreen{ _frame_colors[green_idx](rec) };
    // cv::Mat mblue{ _frame_colors[blue_idx](rec) };
    // cv::Mat mblack{ _frame_black(rec) };
    // cv::Mat mdepth{ img_depth(rec) };

    const int total_pts{ static_cast<int>(recp_width * recp_height) };

    points.clear();
    point_colors.clear();
    // points = Eigen::Matrix<double, 3, -1>::Zero(3, total_pts);
    // point_colors = Eigen::Matrix<double, 3, -1>::Zero(3, total_pts);

    Eigen::Vector2d pt;
    int idx{ 0 };
    for (int i = 0; i < recp_width; ++i)
    {
      for (int j = 0; j < recp_height; ++j)
      {
        pt = px_cp + Eigen::Vector2d(i, j);
        if (all_masks->at<uint8_t>(pt[1], pt[0]) > 0)
        {
          const double z{ img_depth.at<uint16_t>(pt[1], pt[0]) / depth_scale };

          // Point3 backproject(const Point2& p, double depth,

          points.push_back(_camera_interface->camera()->backproject(pt, z));
          // points.col(idx) = Eigen::Vector3d(pt[0] * z, pt[1] * z, z);
          point_colors.push_back(Eigen::Vector3d::Zero());
          if (_frame_colors[red_idx].at<uint8_t>(pt[1], pt[0]) > 0)
          {
            point_colors.back() += colors[red_idx];
          }
          if (_frame_colors[green_idx].at<uint8_t>(pt[1], pt[0]) > 0)
          {
            point_colors.back() += colors[green_idx];
          }
          if (_frame_colors[blue_idx].at<uint8_t>(pt[1], pt[0]) > 0)
          {
            point_colors.back() += colors[blue_idx];
          }
          // idx++;
        }
      }
    }

    // total_points = idx;
  }

  void process_colors()
  {
    std::array<std::vector<Eigen::Vector3d>, 3> pts_by_color;
    std::array<std::vector<Eigen::Matrix3d>, 3> covs_by_color;
    // std::vector<Eigen::Vector3d> pts_black;
    // std::vector<Eigen::Matrix3d> covs_black;
    for (int i = 0; i < points.size(); ++i)
    {
      const Eigen::Vector3d& ci{ point_colors[i] };
      // for (int j = 0; j < 3; ++j)
      for (auto j : { red_idx, green_idx, blue_idx })
      {
        // if (have_same_color(ci, colors[j]))
        if (ci[j] > 0)
        {
          pts_by_color[j].push_back(points[i]);
          covs_by_color[j].push_back(1e-3 * Eigen::Matrix3d::Identity());  // Isometric. In [mm]
        }
      }
      // if (ci.isZero())
      // {
      //   pts_black.push_back(points[i]);
      //   covs_black.push_back(1e-3 * Eigen::Matrix3d::Identity());  // Isometric. In [mm]
      // }
    }
    std::array<std::vector<Eigen::Vector3d>, 3> vals_out;
    std::array<std::vector<Eigen::Matrix3d>, 3> covs_out;
    // std::vector<Eigen::Vector3d> pts_black_clustered;
    // std::vector<Eigen::Matrix3d> covs_black_clustered;
    // const std::vector<Element>& vals_in, const std::vector<Covs>& covs_in
    cluster_all(vals_out[0], covs_out[0], pts_by_color[0], covs_by_color[0]);
    cluster_all(vals_out[1], covs_out[1], pts_by_color[1], covs_by_color[1]);
    cluster_all(vals_out[2], covs_out[2], pts_by_color[2], covs_by_color[2]);
    // cluster_fast(pts_black_clustered, covs_black_clustered, pts_black, covs_black);
    // cluster(vals_out2[0], covs_out2[0], vals_out[0], covs_out[0]);
    // for (int i = 0; i < vals_out[0].size(); ++i)
    // {
    //   DEBUG_VARS(vals_out[0][i].transpose());
    //   DEBUG_VARS(covs_out[0][i].diagonal().transpose());
    // }
    // DEBUG_VARS(vals_out[0].size(), pts_by_color[0].size())

    // remove_unfeasible_points(vals_out[0], covs_out[0]);
    if (visualize)
    {
      for (auto j : { red_idx, green_idx, blue_idx })
      {
        points_with_cov_to_marker(vals_out[j], colors[j], covs_out[j], pts_covs_markers_pub[j]);
      }
      // points_with_cov_to_marker(pts_black_clustered, black, covs_black_clustered, clustered_black);
      // points_with_cov_to_marker(vals_out[1], colors[1], covs_out[1], pts_covs_markers_pub[1]);
      // points_with_cov_to_marker(vals_out[2], colors[2], covs_out[2], pts_covs_markers_pub[2]);
    }
    update_estimate(vals_out, covs_out);
  }

  void update_estimate(const std::array<std::vector<Eigen::Vector3d>, 3>& pts,
                       const std::array<std::vector<Eigen::Matrix3d>, 3>& covs)
  {
    using GaussianNM = gtsam::noiseModel::Gaussian;
    gtsam::Key key{ gtsam::Symbol('X', 0) };
    gtsam::Ordering key_ordering;
    key_ordering += key;
    GaussianNM::shared_ptr z0_noise{ gtsam::noiseModel::Isotropic::Sigma(3, 1e-3) };
    std::array<double, 6> errors;
    std::array<Eigen::Vector3d, 6> proposed;
    for (int i = 0; i < 6; ++i)
    {
      gtsam::Values values;
      const Eigen::Vector3d z0{ _last_estimate[i] };
      values.insert(key, z0);
      int idx{ static_cast<int>(i / 2) };
      double prev_error{ std::numeric_limits<double>::max() };
      for (int j = 0; j < pts[idx].size(); ++j)
      {
        DEBUG_VARS(i, idx, j)
        // const Eigen::Matrix3d& cov{ covs[idx][j] };
        // GaussianNM::shared_ptr z1_noise{ GaussianNM::Information(cov.inverse()) };
        const Eigen::Vector3d& z1{ pts[idx][j] };
        const Eigen::Vector3d& diff{ z0 - z1 };
        // const Eigen::Matrix<double, 1, 3> diag{ cov.diagonal().transpose() };
        DEBUG_VARS(z0.transpose(), z1.transpose())
        const double error{ z0_noise->squaredMahalanobisDistance(diff) };
        // const double error{ diff.transpose() * c * diff };
        // gtsam::GaussianFactorGraph linearFactorGraph;
        // gtsam::PriorFactor<Eigen::Vector3d> prev_prior(key, z0, z0_noise);
        // gtsam::PriorFactor<Eigen::Vector3d> curr_prior(key, z1, z0_noise);
        // linearFactorGraph.push_back(prev_prior.linearize(values));
        // linearFactorGraph.push_back(curr_prior.linearize(values));
        // const gtsam::GaussianConditional::shared_ptr marginal{
        //   linearFactorGraph.marginalMultifrontalBayesNet(key_ordering)->front()
        // };
        // const gtsam::VectorValues result{ marginal->solve(gtsam::VectorValues()) };
        // const double error{ linearFactorGraph.error(result) };
        if (error < prev_error)
        {
          DEBUG_VARS(error, prev_error)
          proposed[i] = z1;
          errors[i] = error;
          prev_error = error;
        }
      }
    }

    // const double off_norm{ (2 * _offset).norm() };
    // for (int i = 0; i < 6; i += 2)
    // {
    //   const Eigen::Vector3d z0A{ _last_estimate[i] };
    //   const Eigen::Vector3d z0B{ _last_estimate[i + 1] };
    //   const Eigen::Vector3d z1A{ proposed[i] };
    //   const Eigen::Vector3d z1B{ proposed[i + 1] };

    //   const double errA{ std::fabs((z0B - z1A).norm() - off_norm) * 1000 };
    //   const double errB{ std::fabs((z0A - z1B).norm() - off_norm) * 1000 };

    //   errors[i] = errA;
    //   errors[i + 1] = errB;
    // }
    // DEBUG_VARS(errors)
    const double max_error{ 4900 };  // Accepting errors of up to 70 [mm] (70^2=4900) -> The bar's width is 35 [mm]
    for (int i = 0; i < 6; ++i)
    {
      // _valid_estimate[i] = false;
      // if (errors[i] < max_error)
      // {
      _valid_estimate[i] = true;
      _last_estimate[i] = proposed[i];
      // }
    }
  }

  void compute_masks()
  {
    cv::inRange(img_hsv, low_red, high_red, _frame_colors[red_idx]);
    cv::inRange(img_hsv, low_green, high_green, _frame_colors[green_idx]);
    cv::inRange(img_hsv, low_blue, high_blue, _frame_colors[blue_idx]);
    cv::inRange(img_hsv, low_black, high_black, _frame_black);

    cv::Mat color_masks{ _frame_colors[red_idx] + _frame_colors[green_idx] + _frame_colors[blue_idx] };
    all_masks = std::make_shared<cv::Mat>(_frame_black + color_masks);

    if (visualize)
    {
      publish_img(color_masks, pub0_rgb_masks, "mono8");
      publish_img(*all_masks, pub0_all_masks, "mono8");
    }
  }

  void run_icp()
  {
    bars_poses_received = false;
    rgb_received = false;
    depth_received = false;

    // perception::erode_or_dilate(0, *all_masks, *all_masks, "ED", elements);
    compute_masks();
    get_subimage(all_masks);

    points_to_marker(points, point_colors, points_marker_pub);

    process_colors();

    // cv::Mat grad_masks;
    // cv::Sobel(*all_masks, grad_masks, CV_16S, 1, 0);
    // cv::Mat abs_grad;
    // cv::convertScaleAbs(grad_masks, abs_grad);
    // cv::Mat diff = abs_grad - abs_grad_x;
    // publish_img(diff, pub_lines, "mono8");
    // publish_img(abs_grad, pub_sobel_all, "mono8");

    // pub_lines
    // publish_img(_frame_colors[red_idx], pub_red_masks, "mono8");
    // publish_img(_frame_colors[green_idx], pub_green_masks, "mono8");
    // publish_img(_frame_colors[blue_idx], pub_blue_masks, "mono8");
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "img_diff_test");
  ros::NodeHandle nh("~");

  pointcloud_from_imgs_t node(nh);
  // while (ros::ok())
  // {
  //   if (node.rgb_received and node.depth_received)
  //   {
  //     // PRINT_MSG("Running icp");
  //     auto start = ros::Time::now();
  //     node.run_icp();
  //     auto end = ros::Time::now();
  //     auto iter_dt = (end - start).toSec();
  //     DEBUG_VARS(iter_dt)
  //     node.rgb_received = false;
  //     node.depth_received = false;
  //   }
  //   ros::spinOnce();
  // }
  ros::spin();
}
