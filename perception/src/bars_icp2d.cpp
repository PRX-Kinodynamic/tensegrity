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

struct PointColor
{
  PointColor(const Translation p, const Color c) : pt(p), color(c)
  {
  }
  TranslationColor as_vector() const
  {
    TranslationColor v;
    v << pt, color;
    return v;
  }

  Translation pt;
  Color color;
};

class bar_2dtransform_factor : public gtsam::NoiseModelFactorN<gtsam::Pose3>
{
public:
  using SE3 = gtsam::Pose3;
  using Base = gtsam::NoiseModelFactorN<SE3>;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;
  using Pixel = Eigen::Vector2d;
  using Translation = Eigen::Vector3d;

  // Keys: kxi->Pose of the bar; ktf -> Pose between kxi and target(observed) point; kdepth-> depth of image
  bar_2dtransform_factor(const gtsam::Key kxi,                                          // no-lint
                         const Translation meassured_pt, const Translation bar_offset,  // no-lint
                         const NoiseModel& cost_model = nullptr)
    : Base(cost_model, kxi), _meassured_pt(meassured_pt), _bar_offset(bar_offset)
  {
  }

  virtual Eigen::VectorXd evaluateError(const SE3& xi,  // no-lint
                                        boost::optional<Eigen::MatrixXd&> Hxi = boost::none) const override
  {
    // Eigen::Matrix<double, 3, 1> ppx_H_depth;
    Eigen::Matrix<double, 3, 6> bpt_H_xi;
    Eigen::Matrix<double, 2, 3> bpx_H_bpt;
    // Eigen::Matrix<double, 3, 3> xp_H_bpt, ptf_H_ppx;
    // const Translation p_pixel{ _camera.backproject(_pixel, depth, boost::none, boost::none, ppx_H_depth) };
    // const Translation p_pixel{ Camera::BackprojectFromCamera(_pixel, depth, boost::none, ppx_H_depth) };
    // const Translation ptf_pixel{ _camera_tf.transformFrom(p_pixel, boost::none, ptf_H_ppx) };

    // const Translation xp{ tf.transformFrom(bar_pt, xp_H_tf, xp_H_bpt) };
    // const Translation error{ bar_pt - _meassured_pt };
    // const Eigen::Matrix3d err_H_bpt{ Eigen::Matrix3d::Identity() };
    // const Eigen::Matrix3d err_H_ppx{ -Eigen::Matrix3d::Identity() };

    try
    {
      const Translation bar_pt{ xi.transformFrom(_bar_offset, bpt_H_xi) };
      const Pixel bar_px{ _camera.project2(bar_pt, boost::none, bpx_H_bpt) };
      const Pixel error{ bar_px - _target_px };
      // const Eigen::Matrix3d err_H_bpx{ Eigen::Matrix3d::Identity() };
      if (Hxi)
      {
        *Hxi = bpx_H_bpt * bpt_H_xi;
      }
      return error;
    }
    catch (gtsam::CheiralityException& e)
    {
      if (Hxi)
        *Hxi = Eigen::Matrix<double, 2, 6>::Zero();
      return _camera.defaultErrorWhenTriangulatingBehindCamera();
    }
  }

private:
  const Pixel _target_px;
  const Translation _bar_offset;
  const Translation _meassured_pt;
  const Camera _camera;
};

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

void locations_to_keypoints(std::vector<cv::Point>& lcs, std::vector<cv::KeyPoint>& kps)
{
  kps.clear();
  for (auto l : lcs)
  {
    kps.emplace_back(l, 1.0);
  }
}

int imgs_to_pointcloud(std::vector<PointColor>& pc, cv::Mat& mask, cv::Mat& depth, std::size_t max_pts,
                       double depth_scale, const Camera& camera, const Color& color)
{
  int tot{ 0 };
  std::vector<cv::Point> pixels;
  cv::findNonZero(mask, pixels);

  // std::vector<std::vector<cv::Point>> _contours_out;
  // cv::findContours(mask, _contours_out, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  // DEBUG_VARS(stddev);
  // DEBUG_VARS(pixels.size());
  const std::size_t true_max_pts{ std::min(pixels.size(), max_pts) };
  for (int i = 0; i < true_max_pts; ++i)
  {
    const std::size_t idx{ factor_graphs::random_uniform<std::size_t>(0, pixels.size()) };
    // DEBUG_VARS(pixels.size(), idx);
    auto px = pixels[idx];
    // double dval = depth.at<uint16_t>(px.x, px.y);
    double dval = depth.at<uint16_t>(px.y, px.x);
    if (dval > 0)
    {
      const double z{ dval / depth_scale };

      const Eigen::Vector2d pixel{ px.x, px.y };
      // const Eigen::Vector2d pixel{ px.y, px.x };
      // const double x{ px.x * z };
      // const double y{ px.y * z };
      const Translation pt{ camera.backproject(pixel, z) };

      // DEBUG_VARS(dval, z, pt.transpose());
      pc.emplace_back(pt, color);
      tot++;
    }
    // if (tot >= max_pts)
    //   break;
  }
  return tot;
}
void compute_target_pointcloud(const std::vector<PointColor>& pc2D, std::vector<PointColor>& pc3D,
                               const std::vector<double> depth, const Camera& camera)
{
  // Eigen::Vector3d pt3d;
  // for (auto pixel : pc2D)
  for (int i = 0; i < pc2D.size(); ++i)
  {
    const Translation pixel{ pc2D[i].pt };
    const Color color{ pc2D[i].color };
    // const Translation pt{ cam_tf * Camera::BackprojectFromCamera(pixel.head(2), depth[i]) };
    const Translation pt{ camera.backproject(pixel.head(2), depth[i]) };

    // DEBUG_VARS(pt.transpose(), color.transpose());
    pc3D.emplace_back(pt, color);
  }
}
void endcap_offsets(std::vector<PointColor>& endcaps, int total_pts, Color color)
{
  const double endcap_rad{ 0.04 };

  for (int i = 0; i < total_pts; ++i)
  {
    const double rad{ factor_graphs::random_uniform<double>(0.0, endcap_rad) };
    const double theta{ factor_graphs::random_uniform<double>(0.0, 3.14159) };
    const double phi{ factor_graphs::random_uniform<double>(0.0, 2.0 * 3.14159) };

    const double z{ rad * std::cos(theta) };
    const double y{ rad * std::sin(theta) * std::sin(phi) };
    const double x{ rad * std::sin(theta) * std::cos(phi) };
    const double side{ z < 0 ? -1.0 : 1.0 };

    // const Rotation rot{ side > 0 ? Rotation() : Roffset };
    const Translation pt{ side * offset + Translation(x, y, z) };
    // const Translation ptp{ pose * pt };
    // DEBUG_VARS(side, x, y, z);
    // DEBUG_VARS(pt.transpose(), ptp.transpose());
    endcaps.emplace_back(pt, color);
    // DEBUG_VARS(endcaps.back());
  }
  // return endcaps;
}
void bar_offsets(std::vector<PointColor>& pts, int total_pts)
{
  const double max_x{ 0.04 };
  const double max_y{ 0.04 };
  const double max_z{ 0.31 / 2.0 };
  const double endcap_rad{ 0.04 };
  for (int i = 0; i < total_pts; ++i)
  {
    const double rad{ factor_graphs::random_uniform<double>(0.0, endcap_rad) };
    const double angle{ factor_graphs::random_uniform<double>(0.0, 2.0 * 3.14159) };

    const double x{ rad * std::cos(angle) };
    const double y{ rad * std::sin(angle) };
    // const double x{ factor_graphs::random_uniform<double>(-max_x, max_x) };
    // const double y{ factor_graphs::random_uniform<double>(-max_y, max_y) };
    const double z{ factor_graphs::random_uniform<double>(-max_z, max_z) };
    pts.emplace_back(Translation(x, y, z), black);
  }
}

void update_marker(visualization_msgs::Marker& marker, const std::vector<PointColor>& pc,
                   const gtsam::Pose3 pose = gtsam::Pose3(), bool clear = true)
{
  if (clear)
  {
    marker.points.clear();
    marker.colors.clear();
  }

  // tensegrity::utils::init_header(marker.header, "real_sense");
  tensegrity::utils::init_header(marker.header, "world");
  marker.action = visualization_msgs::Marker::ADD;
  marker.type = visualization_msgs::Marker::POINTS;

  marker.scale.x = 0.01;  // is point width,
  marker.scale.y = 0.01;  // is point height
  interface::copy(marker.pose, pose);
  for (auto p : pc)
  {
    // DEBUG_VARS(p.pt.transpose());
    marker.points.emplace_back();
    marker.points.back().x = p.pt[0];
    marker.points.back().y = p.pt[1];
    marker.points.back().z = p.pt[2];
    marker.colors.emplace_back();
    marker.colors.back().b = p.color[0];
    marker.colors.back().g = p.color[1];
    marker.colors.back().r = p.color[2];
    marker.colors.back().a = 1.0;
  }
}

void update_matches_marker(visualization_msgs::Marker& marker, const std::vector<PointColor>& matches)
{
  marker.points.clear();
  marker.colors.clear();

  // tensegrity::utils::init_header(marker.header, "real_sense");
  tensegrity::utils::init_header(marker.header, "world");
  marker.action = visualization_msgs::Marker::ADD;
  marker.type = visualization_msgs::Marker::LINE_LIST;

  marker.scale.x = 0.005;  // is point width,
  // marker.scale.y = 0.01;  // is point height
  // interface::copy(marker.pose, pose);
  for (int i = 0; i < matches.size(); i += 2)
  {
    // DEBUG_VARS(p.pt.transpose());
    marker.points.emplace_back();
    marker.points.back().x = matches[i].pt[0];
    marker.points.back().y = matches[i].pt[1];
    marker.points.back().z = matches[i].pt[2];
    marker.points.emplace_back();
    marker.points.back().x = matches[i + 1].pt[0];
    marker.points.back().y = matches[i + 1].pt[1];
    marker.points.back().z = matches[i + 1].pt[2];
    marker.colors.emplace_back();
    marker.colors.back().b = matches[i].color[0];
    marker.colors.back().g = matches[i].color[1];
    marker.colors.back().r = matches[i].color[2];
    marker.colors.back().a = 1.0;
    marker.colors.emplace_back();
    marker.colors.back().b = matches[i + 1].color[0];
    marker.colors.back().g = matches[i + 1].color[1];
    marker.colors.back().r = matches[i + 1].color[2];
    marker.colors.back().a = 1.0;
  }
}

void add_to_gnn(Gnn& gnn, std::vector<PointColor>& pointcloud)
{
  // DEBUG_VARS(pointcloud.size());
  for (int i = 0; i < pointcloud.size(); ++i)
  {
    PointColor pt{ pointcloud[i] };
    // const Translation xi{ pt.pt };
    // const Translation xb{ bar.transformFrom(xi) };
    // const Translation xtf{ tf.transformFrom(xb) };
    // DEBUG_VARS(xi.transpose(), xtf.transpose());
    // pointcloud[i].pt = xtf;
    // pt.pt = xtf;
    // gnn.emplace(pt.as_vector(), i);
    gnn.emplace(pt.pt, i);
    // locations_1[i].y = xtf[1];
  }
}

void pointcloud_to_image(cv::Mat& img, const PointColor& p, const Camera& camera)
{
  Translation pw{ p.pt };
  Color color{ p.color * 255 };
  // Eigen::Vector2d pixel{ camera.project2(pw) };
  auto [pixel, valid] = camera.projectSafe(pw);

  if (valid)
  {
    int u = pixel[0];
    int v = pixel[1];
    img.at<cv::Vec3b>(v, u) = cv::Vec3b(color[0], color[1], color[2]);
  }
}

void add_bar_factors(gtsam::NonlinearFactorGraph& graph, std::vector<PointColor>& bar_pc, gtsam::Pose3& bar_pose,
                     std::vector<PointColor>& target_pc, const gtsam::Key& kbar, Gnn& gnn,
                     std::vector<PointColor>& matches)
{
  NodePtr node;
  double distance;
  for (int i = 0; i < bar_pc.size(); ++i)
  {
    if (gnn.size() == 0)
      break;
    PointColor pc{ bar_pc[i] };
    const Translation offset_i{ pc.pt };
    const Translation x_off{ bar_pose * offset_i };
    // const Translation x_tf{ tf * x_off };
    pc.pt = x_off;
    gnn.single_query(pc.pt, distance, node);

    const PointColor target_pt{ target_pc[node->value_id()] };
    // graph.emplace_shared<bar_2dtransform_factor>(kbar, target_pt.pt, offset_i);

    matches.emplace_back(pc);
    matches.emplace_back(target_pt);
    // gnn.remove_node(node);
  }
}

int mask_to_2d_pointcloud(std::vector<PointColor>& pc, cv::Mat& img, const Color color, std::size_t max_pts)
{
  int tot{ 0 };
  std::vector<cv::Point> pixels;
  cv::findNonZero(img, pixels);
  const std::size_t true_max_pts{ std::min(pixels.size(), max_pts) };
  for (int i = 0; i < true_max_pts; ++i)
  {
    const std::size_t idx{ factor_graphs::random_uniform<std::size_t>(0, pixels.size()) };
    // DEBUG_VARS(pixels.size(), idx);
    auto px = pixels[idx];
    pc.emplace_back(Translation(px.x, px.y, 0.0), color);
    tot++;
    // if (tot >= max_pts)
    //   break;
  }
  return tot;
}

struct tensegrity_3d_icp_t
{
  using This = tensegrity_3d_icp_t;
  int max_iterations;
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
  cv::Mat _frame_red0, _frame_green, _frame_blue, _frame_black;

  std::vector<cv::Mat> elements;
  gtsam::PinholeCamera<gtsam::Cal3_S2> camera;

  ros::Publisher pub0_rgb_masks;
  ros::Publisher pub0_all_masks;
  ros::Publisher pub_red_marker;
  ros::Publisher pub_green_marker;
  ros::Publisher pub_blue_marker;
  ros::Publisher pub_img_red_marker;
  ros::Publisher pub_red_matches;
  ros::Publisher pub_green_matches;
  ros::Publisher pub_blue_matches, pub_img_marker;

  bool use_between_factor;
  bool bars_poses_received, rgb_received, depth_received;
  gtsam::LevenbergMarquardtParams lm_params;
  std::shared_ptr<factor_graphs::levenberg_marquardt_t> lm_helper;

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
  {
    std::string tensegrity_pose_topic;

    lm_params.setVerbosityLM("SILENT");
    // lm_params.setVerbosityLM("SUMMARY");
    lm_params.setMaxIterations(10);
    lm_helper = std::make_shared<factor_graphs::levenberg_marquardt_t>(nh, "/nodes/icp/fg", lm_params);

    PARAM_SETUP(nh, image_topic);
    PARAM_SETUP(nh, depth_topic);
    PARAM_SETUP(nh, max_pts);
    PARAM_SETUP(nh, depth_scale);

    PARAM_SETUP(nh, tensegrity_pose_topic);
    PARAM_SETUP(nh, use_between_factor);

    // tensegrity_bars_subscriber = nh.subscribe(tensegrity_pose_topic, 1, &This::pose_callback, this);

    pub0_rgb_masks = nh.advertise<sensor_msgs::Image>("/img0/rgb_masks", 1, true);
    pub0_all_masks = nh.advertise<sensor_msgs::Image>("/img0/all_masks", 1, true);
    pub_red_marker = nh.advertise<visualization_msgs::Marker>("/bars/red", 1, true);
    pub_green_marker = nh.advertise<visualization_msgs::Marker>("/bars/green", 1, true);
    pub_blue_marker = nh.advertise<visualization_msgs::Marker>("/bars/blue", 1, true);
    pub_img_red_marker = nh.advertise<visualization_msgs::Marker>("/bars/red_img", 1, true);
    pub_red_matches = nh.advertise<visualization_msgs::Marker>("/bars/red/matches", 1, true);
    pub_green_matches = nh.advertise<visualization_msgs::Marker>("/bars/green/matches", 1, true);
    pub_blue_matches = nh.advertise<visualization_msgs::Marker>("/bars/blue/matches", 1, true);
    pub_img_marker = nh.advertise<visualization_msgs::Marker>("/img0/pointcloud", 1, true);

    image_subscriber = nh.subscribe(image_topic, 1, &This::image_callback, this);
    // depth_subscriber = nh.subscribe(depth_topic, 1, &This::depth_callback, this);

    std::vector<int> kernel_sizes{ { 2, 5 } };
    for (int i = 0; i < kernel_sizes.size(); ++i)
    {
      const int erosion_size{ kernel_sizes[i] };
      elements.push_back(cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                   cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                                                   cv::Point(erosion_size, erosion_size)));
    }

    Eigen::Matrix4d extrinsic;
    // intrinsic << 901.67626953125, 0.0, 640.0, 0.0,        // no-lint
    //     0.0, 901.7260131835938, 360.60618591308594, 0.0,  // no-lint
    //     0.0, 0.0, 1.0, 0.0,                               // no-lint
    //     0.0, 0.0, 0.0, 1.0;
    extrinsic << 1.0, 0.0, -0.01, 0.743, 0.0, -1.0, -0.007, 0.082, -0.01, 0.007, -1.0, 1.441, 0.0, 0.0, 0.0, 1.0;
    gtsam::Cal3_S2 camera_calibration(901.67626953125, 901.7260131835938, 0.0, 640.0, 360.60618591308594);
    // Eigen::Matrix4d int_ext_inv{ extrinsic.inverse() * intrinsic.inverse() };
    // gtsam::Pose3 int_ext_inv{ extrinsic.inverse() * intrinsic.inverse() };
    camera = Camera(gtsam::Pose3(extrinsic), camera_calibration);
  }

  void pose_callback(const interface::TensegrityBarsConstPtr msg)
  {
    red_pose = tensegrity::utils::convert_to<gtsam::Pose3>(msg->bar_red);
    green_pose = tensegrity::utils::convert_to<gtsam::Pose3>(msg->bar_green);
    blue_pose = tensegrity::utils::convert_to<gtsam::Pose3>(msg->bar_blue);
    // bars_poses_received = true;
  }

  void image_callback(const sensor_msgs::ImageConstPtr message)
  {
    cv_bridge::CvImageConstPtr frame{ cv_bridge::toCvShare(message) };
    cv::cvtColor(frame->image, img_hsv, cv::COLOR_BGR2HSV);
    rgb_received = true;
  }

  void run_icp()
  {
    rgb_received = false;

    cv::inRange(img_hsv, low_red, high_red, _frame_red0);
    cv::inRange(img_hsv, low_green, high_green, _frame_green);
    cv::inRange(img_hsv, low_blue, high_blue, _frame_blue);
    cv::inRange(img_hsv, low_black, high_black, _frame_black);

    perception::erode_or_dilate(0, _frame_red0, _frame_red0, "ED", elements);
    perception::erode_or_dilate(0, _frame_green, _frame_green, "ED", elements);
    perception::erode_or_dilate(0, _frame_blue, _frame_blue, "ED", elements);

    cv::Mat color_masks{ _frame_red0 + _frame_green + _frame_blue };
    cv::Mat all_masks{ _frame_black + color_masks };

    publish_img(color_masks, pub0_rgb_masks, "mono8");
    publish_img(all_masks, pub0_all_masks, "mono8");

    // std::vector<PointColor> red_target_3dpc, blue_target_3dpc, green_target_3dpc, black_target_3dpc;
    // int red_pts{ imgs_to_pointcloud(red_target_3dpc, _frame_red0, img_depth, 2 * max_pts * 0.3, depth_scale, camera,
    //                                 red) };
    // int green_pts{ imgs_to_pointcloud(green_target_3dpc, _frame_green, img_depth, 2 * max_pts * 0.3, depth_scale,
    //                                   camera, green) };
    // int blue_pts{ imgs_to_pointcloud(blue_target_3dpc, _frame_blue, img_depth, 2 * max_pts * 0.3, depth_scale,
    // camera,
    //                                  blue) };
    // int black_pts{ imgs_to_pointcloud(black_target_3dpc, _frame_black, img_depth, 2 * max_pts * 0.1, depth_scale,
    //                                   camera, black) };
    // const int tot_red{ mask_to_2d_pointcloud(target_2dpc, _frame_red0, red, max_pts * 0.3) };
    // const int tot_black{ mask_to_2d_pointcloud(target_2dpc, _frame_black, black, max_pts * 0.7) };

    // gtsam::Pose3 red_pose, green_pose, blue_pose;
    // red_pose = gtsam::Pose3(Rotation(0.707, 0.707, 0, 0), Translation(0.5, -0.3, 0.0));
    // green_pose = gtsam::Pose3(Rotation(0.707, 0.707, 0, 0), Translation(0.5, -0.3, 0.0));
    // blue_pose = gtsam::Pose3(Rotation(0.707, 0.707, 0, 0), Translation(0.5, -0.3, 0.0));
    // std::vector<double> depths(target_2dpc.size(), 1.4);
    // compute_target_pointcloud(target_2dpc, target_3dpc, depths, camera);

    // std::vector<PointColor> red_endcap_sim_pts, green_endcap_sim_pts, blue_endcap_sim_pts;
    // std::vector<PointColor> red_bar_sim_pts, green_bar_sim_pts, blue_bar_sim_pts;
    // endcap_offsets(red_endcap_sim_pts, red_pts, red);
    // endcap_offsets(green_endcap_sim_pts, green_pts, green);
    // endcap_offsets(blue_endcap_sim_pts, blue_pts, blue);
    // bar_offsets(red_bar_sim_pts, black_pts * 0.5);
    // bar_offsets(green_bar_sim_pts, black_pts * 0.5);
    // bar_offsets(blue_bar_sim_pts, black_pts * 0.5);

    // visualization_msgs::Marker img_marker;
    // visualization_msgs::Marker red_marker;
    // visualization_msgs::Marker green_marker;
    // visualization_msgs::Marker blue_marker;
    // visualization_msgs::Marker red_matches_marker, green_matches_marker, blue_matches_marker;

    // update_marker(red_marker, red_bar_sim_pts, red_pose);
    // update_marker(green_marker, green_bar_sim_pts, green_pose);
    // update_marker(blue_marker, blue_bar_sim_pts, blue_pose);

    // update_marker(img_marker, red_target_3dpc, gtsam::Pose3(), true);
    // update_marker(img_marker, blue_target_3dpc, gtsam::Pose3(), false);
    // update_marker(img_marker, green_target_3dpc, gtsam::Pose3(), false);
    // update_marker(img_marker, black_target_3dpc, gtsam::Pose3(), false);

    // // pub_red_marker.publish(red_marker);
    // pub_img_marker.publish(img_marker);

    // double error{ 200000 };
    // double error_change{ 1000 };
    // double dummy;
    // int iterations{ 0 };

    // const gtsam::Pose3 red_green_btw{ gtsam::traits<gtsam::Pose3>::Between(red_pose, green_pose) };
    // const gtsam::Pose3 red_blue_btw{ gtsam::traits<gtsam::Pose3>::Between(red_pose, blue_pose) };
    // const gtsam::Pose3 green_blue_btw{ gtsam::traits<gtsam::Pose3>::Between(green_pose, blue_pose) };

    // gtsam::noiseModel::Base::shared_ptr btw_noise{ gtsam::noiseModel::Isotropic::Sigma(6, 1e-3) };

    // while (error > 1.0 and error_change > 1.0 and iterations < max_iterations)
    // {
    //   Gnn gnn{};
    //   gtsam::NonlinearFactorGraph graph;
    //   gtsam::Values values;
    //   // DEBUG_VARS(error);
    //   const gtsam::Key kred{ gtsam::Symbol('r', 0) };
    //   const gtsam::Key kgreen{ gtsam::Symbol('g', 0) };
    //   const gtsam::Key kblue{ gtsam::Symbol('b', 0) };
    //   values.insert(kred, red_pose);
    //   values.insert(kgreen, green_pose);
    //   values.insert(kblue, blue_pose);

    //   add_to_gnn(gnn, black_target_3dpc);
    //   add_to_gnn(gnn_red, red_target_3dpc);
    //   add_to_gnn(gnn_green, green_target_3dpc);
    //   add_to_gnn(gnn_blue, blue_target_3dpc);

    //   // NodePtr node;
    //   // double distance;

    //   std::vector<PointColor> red_matches, blue_matches, green_matches;
    //   add_bar_factors(graph, red_endcap_sim_pts, red_pose, red_target_3dpc, kred, gnn_red, red_matches);
    //   add_bar_factors(graph, green_endcap_sim_pts, green_pose, green_target_3dpc, kgreen, gnn_green, green_matches);
    //   add_bar_factors(graph, blue_endcap_sim_pts, blue_pose, blue_target_3dpc, kblue, gnn_blue, blue_matches);
    //   add_bar_factors(graph, red_bar_sim_pts, red_pose, black_target_3dpc, kred, gnn, red_matches);
    //   add_bar_factors(graph, green_bar_sim_pts, green_pose, black_target_3dpc, kgreen, gnn, green_matches);
    //   add_bar_factors(graph, blue_bar_sim_pts, blue_pose, black_target_3dpc, kblue, gnn, blue_matches);

    //   if (use_between_factor)
    //   {
    //     graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(kred, kgreen, red_green_btw, btw_noise);
    //     graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(kred, kblue, red_blue_btw, btw_noise);
    //     graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(kgreen, kblue, green_blue_btw, btw_noise);
    //   }

    //   const gtsam::Values result{ lm_helper->optimize(graph, values, true) };
    //   const gtsam::Pose3 new_red{ result.at<gtsam::Pose3>(kred) };
    //   const gtsam::Pose3 new_green{ result.at<gtsam::Pose3>(kgreen) };
    //   const gtsam::Pose3 new_blue{ result.at<gtsam::Pose3>(kblue) };

    //   const double prev_error{ error };
    //   error = graph.error(result);
    //   error_change = std::fabs(prev_error - error);

    //   update_marker(red_marker, red_endcap_sim_pts, red_pose, true);
    //   update_marker(red_marker, red_bar_sim_pts, red_pose, false);
    //   update_matches_marker(red_matches_marker, red_matches);

    //   update_marker(green_marker, green_endcap_sim_pts, green_pose);
    //   update_marker(green_marker, green_bar_sim_pts, green_pose, false);
    //   update_matches_marker(green_matches_marker, green_matches);

    //   update_marker(blue_marker, blue_endcap_sim_pts, blue_pose);
    //   update_marker(blue_marker, blue_bar_sim_pts, blue_pose, false);
    //   update_matches_marker(blue_matches_marker, blue_matches);

    //   pub_red_marker.publish(red_marker);
    //   pub_green_marker.publish(green_marker);
    //   pub_blue_marker.publish(blue_marker);
    //   pub_red_matches.publish(red_matches_marker);
    //   pub_green_matches.publish(green_matches_marker);
    //   pub_blue_matches.publish(blue_matches_marker);
    //   //   pub_img_red_marker.publish(red_img_marker);

    //   red_pose = new_red;
    //   green_pose = new_green;
    //   blue_pose = new_blue;
    //   iterations++;
    //   // std::cin >> dummy;
    // }
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "img_diff_test");
  ros::NodeHandle nh("~");

  tensegrity_3d_icp_t node(nh);
  while (ros::ok())
  {
    if (node.rgb_received)
    {
      // PRINT_MSG("Running icp");
      node.run_icp();
    }
    ros::spinOnce();
  }
}
