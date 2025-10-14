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

#include <interface/node_status.hpp>
#include <interface/TensegrityBars.h>

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
using Metric = utils::eucledian_metric_t<TranslationColor>;
using Gnn = utils::graph_nearest_neighbors_t<TranslationColor, Metric>;
using NodePtr = Gnn::NodePtr;

using PositiveDepthFactor = factor_graphs::constraint_factor_t<double, std::less<double>>;

const Translation offset({ 0, 0, 0.325 / 2.0 });
const Rotation Roffset(0.0, 0.0, 1.0, 0.0);
const Color black(Color::Zero());
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

class pixel_bar_transform_factor : public gtsam::NoiseModelFactorN<gtsam::Pose3, double>
{
public:
  using SE3 = gtsam::Pose3;
  using Base = gtsam::NoiseModelFactorN<SE3, double>;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;
  using Pixel = Eigen::Vector2d;
  using Translation = Eigen::Vector3d;

  // Keys: kxi->Pose of the bar; ktf -> Pose between kxi and target(observed) point; kdepth-> depth of image
  pixel_bar_transform_factor(const gtsam::Key kxi, const gtsam::Key kdepth,    // no-lint
                             const Pixel pixel, const Translation bar_offset,  // no-lint
                             const Camera camera, const NoiseModel& cost_model = nullptr)
    : Base(cost_model, kxi, kdepth), _pixel(pixel), _bar_offset(bar_offset), _camera(camera)
  {
  }

  virtual Eigen::VectorXd evaluateError(const SE3& xi, const double& depth,  // no-lint
                                        boost::optional<Eigen::MatrixXd&> Hxi = boost::none,
                                        boost::optional<Eigen::MatrixXd&> Hdepth = boost::none) const override
  {
    Eigen::Matrix<double, 3, 1> ppx_H_depth;
    Eigen::Matrix<double, 3, 6> bpt_H_xi, xp_H_tf;
    Eigen::Matrix<double, 3, 3> xp_H_bpt, ptf_H_ppx;
    const Translation p_pixel{ _camera.backproject(_pixel, depth, boost::none, boost::none, ppx_H_depth) };
    // const Translation p_pixel{ Camera::BackprojectFromCamera(_pixel, depth, boost::none, ppx_H_depth) };
    // const Translation ptf_pixel{ _camera_tf.transformFrom(p_pixel, boost::none, ptf_H_ppx) };

    const Translation bar_pt{ xi.transformFrom(_bar_offset, bpt_H_xi) };
    // const Translation xp{ tf.transformFrom(bar_pt, xp_H_tf, xp_H_bpt) };
    const Translation error{ bar_pt - p_pixel };
    const Eigen::Matrix3d err_H_bpt{ Eigen::Matrix3d::Identity() };
    const Eigen::Matrix3d err_H_ppx{ -Eigen::Matrix3d::Identity() };

    if (Hxi)
    {
      *Hxi = err_H_bpt * bpt_H_xi;
    }
    if (Hdepth)
    {
      *Hdepth = err_H_ppx * ppx_H_depth;
    }
    return error;
  }

private:
  const Pixel _pixel;
  const Translation _bar_offset;
  // const SE3 _camera_tf;
  Camera _camera;
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
  DEBUG_VARS(tf_mat);
  DEBUG_VARS(cv_tf);
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
    // const double angle{ factor_graphs::random_uniform<double>(0.0, 2.0 * 3.14159) };
    const double theta{ factor_graphs::random_uniform<double>(0.0, 3.14159) };
    const double phi{ factor_graphs::random_uniform<double>(0.0, 2.0 * 3.14159) };

    const double z{ rad * std::cos(theta) };
    const double y{ rad * std::sin(theta) * std::sin(phi) };
    const double x{ rad * std::sin(theta) * std::cos(phi) };
    const double side{ z < 0 ? -1.0 : 1.0 };
    // const double z{ rad * std::cos(angle) };
    // const double y{ rad * std::sin(angle) };
    // const double x{ factor_graphs::random_uniform<double>(-endcap_rad, endcap_rad) };
    // const double side{ z < 0 ? -1.0 : 1.0 };

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
                   const gtsam::Pose3 pose = gtsam::Pose3())
{
  marker.points.clear();
  marker.colors.clear();

  tensegrity::utils::init_header(marker.header, "world");
  marker.action = visualization_msgs::Marker::ADD;
  marker.type = visualization_msgs::Marker::POINTS;

  marker.scale.x = 0.03;  // is point width,
  marker.scale.y = 0.03;  // is point height
  interface::copy(marker.pose, pose);
  for (auto p : pc)
  {
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

// void add_to_gnn(Gnn& gnn, std::vector<PointColor>& pointcloud, const gtsam::Pose3& bar, const gtsam::Pose3& tf)
void add_to_gnn(Gnn& gnn, std::vector<PointColor>& pointcloud)
{
  DEBUG_VARS(pointcloud.size());
  for (int i = 0; i < pointcloud.size(); ++i)
  {
    PointColor pt{ pointcloud[i] };
    // const Translation xi{ pt.pt };
    // const Translation xb{ bar.transformFrom(xi) };
    // const Translation xtf{ tf.transformFrom(xb) };
    // DEBUG_VARS(xi.transpose(), xtf.transpose());
    // pointcloud[i].pt = xtf;
    // pt.pt = xtf;
    gnn.emplace(pt.as_vector(), i);
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

int main(int argc, char** argv)
{
  ros::init(argc, argv, "img_diff_test");
  ros::NodeHandle nh("~");

  std::string img0_path;
  // std::string img1_path;

  cv::Scalar low_red(160, 153, 57);
  cv::Scalar high_red(179, 255, 150);
  cv::Scalar low_black(0, 0, 0);
  cv::Scalar high_black(179, 255, 60);

  int max_pts;
  PARAM_SETUP(nh, img0_path);
  PARAM_SETUP(nh, max_pts);
  // PARAM_SETUP(nh, img1_path);

  ros::Publisher pub0{ nh.advertise<sensor_msgs::Image>("/img0", 1, true) };
  ros::Publisher pub0_red{ nh.advertise<sensor_msgs::Image>("/img0/red", 1, true) };
  ros::Publisher pub0_black{ nh.advertise<sensor_msgs::Image>("/img0/black", 1, true) };
  ros::Publisher pub1_keypoints{ nh.advertise<sensor_msgs::Image>("/img1/keypoints", 1, true) };
  ros::Publisher pub1_icp{ nh.advertise<sensor_msgs::Image>("/img1/icp", 1, true) };
  ros::Publisher pub_prediction{ nh.advertise<sensor_msgs::Image>("/img0/prediction", 1, true) };
  ros::Publisher pub_red_marker{ nh.advertise<visualization_msgs::Marker>("/bars/red", 1, true) };
  ros::Publisher pub_img_red_marker{ nh.advertise<visualization_msgs::Marker>("/bars/red_img", 1, true) };
  ros::Publisher pub_img_red_t0_marker{ nh.advertise<visualization_msgs::Marker>("/bars/red_img/original", 1, true) };

  cv::Mat img0{ cv::imread(img0_path) };
  // cv::Mat img1{ cv::imread(img1_path) };
  Eigen::Matrix4d intrinsic, extrinsic;

  intrinsic << 901.67626953125, 0.0, 640.0, 0.0,        // no-lint
      0.0, 901.7260131835938, 360.60618591308594, 0.0,  // no-lint
      0.0, 0.0, 1.0, 0.0,                               // no-lint
      0.0, 0.0, 0.0, 1.0;
  extrinsic << 1.0, 0.0, -0.01, 0.743, 0.0, -1.0, -0.007, 0.082, -0.01, 0.007, -1.0, 1.441, 0.0, 0.0, 0.0, 1.0;
  gtsam::Cal3_S2 camera_calibration(901.67626953125, 901.7260131835938, 0.0, 640.0, 360.60618591308594);
  // Eigen::Matrix4d int_ext_inv{ extrinsic.inverse() * intrinsic.inverse() };
  // gtsam::Pose3 int_ext_inv{ extrinsic.inverse() * intrinsic.inverse() };
  gtsam::PinholeCamera<gtsam::Cal3_S2> camera(gtsam::Pose3(extrinsic), camera_calibration);

  // gtsam::Pose3 int_ext_inv{ extrinsic * intrinsic.inverse() };
  // return CalibratedCamera(pose).backproject(point, depth);
  // int_ext_inv.print("int_ext_inv");
  cv::Mat img0_hsv, img1_hsv;
  cv::cvtColor(img0, img0_hsv, cv::COLOR_BGR2HSV);
  // cv::cvtColor(img1, img1_hsv, cv::COLOR_BGR2HSV);

  cv::Mat _frame_red0, _frame_black;
  cv::inRange(img0_hsv, low_red, high_red, _frame_red0);
  cv::inRange(img0_hsv, low_black, high_black, _frame_black);

  publish_img(img0, pub0, "bgr8");
  publish_img(_frame_red0, pub0_red, "mono8");
  publish_img(_frame_black, pub0_black, "mono8");

  std::vector<PointColor> target_2dpc;
  std::vector<PointColor> target_3dpc;
  const int tot_red{ mask_to_2d_pointcloud(target_2dpc, _frame_red0, red, max_pts * 0.3) };
  const int tot_black{ mask_to_2d_pointcloud(target_2dpc, _frame_black, black, max_pts * 0.7) };

  gtsam::Pose3 tf;
  gtsam::Pose3 red_pose(Roffset, Translation(0.5, -0.3, 0.0));
  std::vector<double> depths(target_2dpc.size(), 1.4);
  compute_target_pointcloud(target_2dpc, target_3dpc, depths, camera);

  std::vector<PointColor> red_bar_sim_pts;
  endcap_offsets(red_bar_sim_pts, max_pts * 0.3, red);
  bar_offsets(red_bar_sim_pts, max_pts * 0.7);

  visualization_msgs::Marker red_marker, red_img_marker;
  update_marker(red_marker, red_bar_sim_pts, red_pose);

  update_marker(red_img_marker, target_3dpc);

  pub_red_marker.publish(red_marker);
  pub_img_red_t0_marker.publish(red_img_marker);

  gtsam::LevenbergMarquardtParams lm_params;
  // lm_params.setVerbosityLM("SILENT");
  lm_params.setVerbosityLM("SUMMARY");
  lm_params.setMaxIterations(10);
  factor_graphs::levenberg_marquardt_t lm_helper(nh, "/nodes/icp/fg", lm_params);

  double error{ 100000 };
  double dummy;

  while (error > 1.0 and ros::ok())
  {
    Gnn gnn{};
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values values;
    DEBUG_VARS(error);
    tf.print("tf");
    const gtsam::Key kred{ gtsam::Symbol('r', 0) };
    // const gtsam::Key ktf{ gtsam::Symbol('t', 0) };
    values.insert(kred, red_pose);
    // values.insert(ktf, tf);

    add_to_gnn(gnn, target_3dpc);

    NodePtr node;
    double distance;
    std::vector<gtsam::Symbol> depth_keys;
    for (int i = 0; i < red_bar_sim_pts.size(); ++i)
    {
      // const gtsam::Key kdepth{ gtsam::Symbol('d', i) };
      PointColor pc{ red_bar_sim_pts[i] };
      const Translation offset_i{ pc.pt };
      const Translation x_off{ red_pose * offset_i };
      // const Translation x_tf{ tf * x_off };
      pc.pt = x_off;
      gnn.single_query(pc.as_vector(), distance, node);

      depth_keys.push_back(gtsam::Symbol('d', node->value_id()));
      values.insert(depth_keys.back(), depths[node->value_id()]);
      const PointColor pixel_pc{ target_2dpc[node->value_id()] };
      graph.emplace_shared<PositiveDepthFactor>(depth_keys.back(), 0.5);
      graph.emplace_shared<pixel_bar_transform_factor>(kred, depth_keys.back(),  // no-lint
                                                       pixel_pc.pt.head(2), offset_i, camera);
      // pixel_bar_transform_factor(const gtsam::Key kxi, const gtsam::Key ktf, const gtsam::Key kdepth,  // no-lint
      //                            const Pixel pixel, const Translation bar_offset,                      // no-lint
      //                            const SE3 camera_tf, const NoiseModel& cost_model = nullptr)
      gnn.remove_node(node);
    }

    // locations_to_keypoints(locations_1, keypoints_1);

    // cv::Mat img_matches;
    // cv::drawMatches(img0, keypoints_0, img1, keypoints_1, matches, img_matches);

    const gtsam::Values result{ lm_helper.optimize(graph, values, true) };
    const gtsam::Pose3 new_red{ result.at<gtsam::Pose3>(kred) };
    error = graph.error(result);

    new_red.print("new_red");

    for (int i = 0; i < depth_keys.size(); ++i)
    {
      int idx = depth_keys[i].index();
      depths[idx] = result.at<double>(depth_keys[i]);
      // DEBUG_VARS(depths[idx]);
    }
    std::vector<PointColor> new_target_3dpc;
    compute_target_pointcloud(target_2dpc, new_target_3dpc, depths, camera);
    update_marker(red_marker, red_bar_sim_pts, new_red);
    update_marker(red_img_marker, new_target_3dpc);
    cv::Mat img_pred{ cv::Mat::ones(img0.rows, img0.cols, img0.type()) };
    img_pred = ~img_pred;
    for (int i = 0; i < depth_keys.size(); ++i)
    {
      int idx = depth_keys[i].index();
      PointColor pt{ red_bar_sim_pts[i] };
      pt.pt = new_red * pt.pt;
      pointcloud_to_image(img_pred, pt, camera);
    }
    publish_img(img_pred, pub_prediction, "bgr8");

    pub_red_marker.publish(red_marker);
    pub_img_red_marker.publish(red_img_marker);

    red_pose = new_red;
    ros::spinOnce();
    // PRINT_MSG("cin");
    std::cin >> dummy;
  }
  PRINT_MSG("Ended");
  ros::spin();
}
