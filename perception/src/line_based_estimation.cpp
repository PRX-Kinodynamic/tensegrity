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

#include <interface/TensegrityBarsArray.h>
#include <visualization_msgs/Marker.h>

using Rotation = gtsam::Rot3;
using Pixel = Eigen::Vector2d;
using Translation = Eigen::Vector3d;
using Color = Eigen::Vector<double, 3>;
using TranslationColor = Eigen::Vector<double, 6>;
using Camera = gtsam::PinholeCamera<gtsam::Cal3_S2>;
using CameraPtr = std::shared_ptr<Camera>;

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

const Translation offset({ 0, 0, 0.325 / 2.0 });
const Rotation Roffset(0.0, 0.0, 1.0, 0.0);
const Color black(Color::Zero());
const Color blue(1.0, 0.0, 0.0);
const Color green(0.0, 1.0, 0.0);
const Color red(0.0, 0.0, 1.0);
const Pixel px_offset({ 0.325 / 2.0, 0.0 });

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
  static cv::Scalar color_to_cv(const Color color)
  {
    return cv::Scalar(color[0] * 255, color[1] * 255, color[2] * 255);
  }

  Translation pt;
  Color color;
};

struct bar2D_t
{
  double scale;
  gtsam::Pose2 pose;
  std::optional<Pixel> endcapA;
  std::optional<Pixel> endcapB;
  std::optional<Eigen::Matrix2d> sigmaA;
  std::optional<Eigen::Matrix2d> sigmaB;
  uint8_t color_id;

  bar2D_t(const gtsam::Pose2 pose_in, double scale_in, const uint8_t id_in)
    : pose(pose_in), scale(scale_in), color_id(id_in)
  {
  }
  template <typename EigenType>
  static bool isnan(EigenType& m)
  {
    return std::isnan(m.template maxCoeff<Eigen::PropagateNaN>());
  }
  void check()
  {
    if (endcapA and isnan(*endcapA))
      endcapA = {};
    if (endcapB and isnan(*endcapB))
      endcapB = {};
    if (sigmaA)
    {
      if (isnan(*sigmaA) or std::fabs((*sigmaA).determinant()) < 1e-2)
      {
        sigmaA = {};
      }
    }
    if (sigmaB)
    {
      if (isnan(*sigmaB) or std::fabs((*sigmaB).determinant()) < 1e-2)
      {
        sigmaB = {};
      }
    }
  }
  bool valid()
  {
    bool Avalid{ endcapA and sigmaA };
    bool Bvalid{ endcapB and sigmaB };
    return Avalid or Bvalid;
  }
};

class color_line_factor_t : public gtsam::NoiseModelFactorN<double, double>
{
public:
  using Base = gtsam::NoiseModelFactorN<double, double>;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;
  using Pixel = Eigen::Vector2d;
  using Translation = Eigen::Vector3d;

  // Y = a + Xb
  color_line_factor_t(const gtsam::Key ka, const gtsam::Key kb,  // no-lint
                      const Pixel pixel, const Color color,      // no-lint
                      const NoiseModel& cost_model = nullptr)
    : Base(cost_model, ka, kb), _pixel(pixel), _color_weight(color.norm() < 1e-2 ? 0.1 : 1.0)
  {
  }

  // virtual bool active(const gtsam::Values& values) const override
  // {
  //   if (color)
  //   {
  //     return _gnn_distance < (*_gnn_avg);
  //   }
  //   return true;
  // }

  virtual Eigen::VectorXd evaluateError(const double& a, const double& b,  // no-lint
                                        boost::optional<Eigen::MatrixXd&> Ha = boost::none,
                                        boost::optional<Eigen::MatrixXd&> Hb = boost::none) const override
  {
    const double& x{ _pixel[0] };
    const double& y{ _pixel[1] };
    const double error{ (a + x * b - y) * _color_weight };

    if (Ha)
    {
      *Ha = Eigen::Matrix<double, 1, 1>::Identity();
    }
    if (Hb)
    {
      *Hb = Eigen::Matrix<double, 1, 1>(x) * _color_weight;
    }
    return Eigen::Vector<double, 1>(error);
  }

private:
  const Pixel _pixel;
  const double _color_weight;
};

class line_sobel_factor_t : public gtsam::NoiseModelFactorN<Eigen::Vector2d>
{
public:
  using Base = gtsam::NoiseModelFactorN<Eigen::Vector2d>;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;
  using Pixel = Eigen::Vector2d;
  using Translation = Eigen::Vector3d;

  line_sobel_factor_t(const gtsam::Key kpx,                  // no-lint
                      const Pixel pixel, const int tot_pts,  // no-lint
                      std::shared_ptr<cv::Mat> mask_img,     // no-lint
                      std::shared_ptr<cv::Mat> sobel_x, std::shared_ptr<cv::Mat> sobel_y,
                      const NoiseModel& cost_model = nullptr)
    : Base(cost_model, kpx), _pixel(pixel), _mask_img(mask_img), _sobel_x(sobel_x), _sobel_y(sobel_y), _tot_pts(tot_pts)
  {
  }

  virtual Eigen::VectorXd evaluateError(const Pixel& px,  // no-lint
                                        boost::optional<Eigen::MatrixXd&> Hpx = boost::none) const override
  {
    Eigen::VectorXd error{ Eigen::VectorXd::Zero(_tot_pts) };
    Eigen::MatrixXd err_H_px{ Eigen::MatrixXd::Zero(_tot_pts, 2) };
    for (int i = 0; i < _tot_pts; ++i)
    {
      const double wi{ i / static_cast<double>(_tot_pts) };
      const Pixel vi{ (px - _pixel) * wi };  // vi_H_px = wi * I
      const Pixel px_i{ _pixel + vi };
      const double& x{ px_i[0] };
      const double& y{ px_i[1] };
      // DEBUG_VARS(wi, vi.transpose(), px_i.transpose());
      error[i] = _mask_img->at<uint8_t>(y, x) / 255 - 1.0;

      if (Hpx)
      {
        const double dx{ static_cast<double>(_sobel_x->at<uint16_t>(y, x)) / 65'535 };
        const double dy{ static_cast<double>(_sobel_y->at<uint16_t>(y, x)) / 65'535 };
        err_H_px(i, 0) = dx;
        err_H_px(i, 1) = dy;
        // DEBUG_VARS(i, x, y, error[i], wi, dx, dy);
      }
    }
    // DEBUG_VARS(error.transpose())
    if (Hpx)
    {
      // DEBUG_VARS(err_H_px)
      *Hpx = err_H_px;
    }
    // if (Hb)
    // {
    //   *Hb = Eigen::Matrix<double, 1, 1>(x) * _color_weight;
    // }
    return error;
  }

private:
  const Pixel _pixel;
  const int _tot_pts;
  std::shared_ptr<cv::Mat> _mask_img;
  std::shared_ptr<cv::Mat> _sobel_x;
  std::shared_ptr<cv::Mat> _sobel_y;
};

class point_sobel_factor_t : public gtsam::NoiseModelFactorN<Eigen::Vector2d>
{
public:
  using Base = gtsam::NoiseModelFactorN<Eigen::Vector2d>;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;
  using Pixel = Eigen::Vector2d;
  using Translation = Eigen::Vector3d;

  point_sobel_factor_t(const gtsam::Key kpx,               // no-lint
                       std::shared_ptr<cv::Mat> mask_img,  // no-lint
                       std::shared_ptr<cv::Mat> sobel_x, std::shared_ptr<cv::Mat> sobel_y,
                       const NoiseModel& cost_model = nullptr)
    : Base(cost_model, kpx), _mask_img(mask_img), _sobel_x(sobel_x), _sobel_y(sobel_y)
  {
  }

  virtual Eigen::VectorXd evaluateError(const Pixel& px,  // no-lint
                                        boost::optional<Eigen::MatrixXd&> Hpx = boost::none) const override
  {
    // Eigen::VectorXd error{ Eigen::VectorXd::Zero(_tot_pts) };
    const double& x{ px[0] };
    const double& y{ px[1] };
    // DEBUG_VARS(wi, vi.transpose(), px_i.transpose());
    const double error{ static_cast<double>(_mask_img->at<uint8_t>(y, x)) / 255.0 - 1.0 };

    if (Hpx)
    {
      Eigen::Matrix<double, 1, 2> err_H_px{ Eigen::Matrix<double, 1, 2>::Zero() };
      const double dx{ static_cast<double>(_sobel_x->at<uint16_t>(y, x)) / 65'535 };
      const double dy{ static_cast<double>(_sobel_y->at<uint16_t>(y, x)) / 65'535 };
      err_H_px(0, 0) = dx;
      err_H_px(0, 1) = dy;
      *Hpx = err_H_px;
      // DEBUG_VARS(x, y, error, dx, dy);
    }
    return Eigen::Vector<double, 1>(error);
  }

private:
  std::shared_ptr<cv::Mat> _mask_img;
  std::shared_ptr<cv::Mat> _sobel_x;
  std::shared_ptr<cv::Mat> _sobel_y;
};
class parallel_points_factor_t : public gtsam::NoiseModelFactorN<Eigen::Vector2d, Eigen::Vector2d, Eigen::Vector2d>
{
public:
  using Base = gtsam::NoiseModelFactorN<Eigen::Vector2d, Eigen::Vector2d, Eigen::Vector2d>;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;
  using Pixel = Eigen::Vector2d;
  using Translation = Eigen::Vector3d;

  parallel_points_factor_t(const gtsam::Key kp0, const gtsam::Key kp1, const gtsam::Key kp2,  // no-lint
                           const NoiseModel& cost_model = nullptr)
    : Base(cost_model, kp0, kp1, kp2)
  {
  }

  static double dot(const Pixel& p, const Pixel& q, gtsam::OptionalJacobian<1, 2> H1, gtsam::OptionalJacobian<1, 2> H2)
  {
    if (H1)
      *H1 << q[0], q[1];
    if (H2)
      *H2 << p[0], p[1];
    return p[0] * q[0] + p[1] * q[1];
  }

  virtual Eigen::VectorXd evaluateError(const Pixel& p0, const Pixel& p1, const Pixel& p2,  // no-lint
                                        boost::optional<Eigen::MatrixXd&> Hp0 = boost::none,
                                        boost::optional<Eigen::MatrixXd&> Hp1 = boost::none,
                                        boost::optional<Eigen::MatrixXd&> Hp2 = boost::none) const override
  {
    const Eigen::Matrix<double, 2, 2> v0_H_p0{ -Eigen::Matrix<double, 2, 2>::Identity() };
    const Eigen::Matrix<double, 2, 2> v0_H_p1{ Eigen::Matrix<double, 2, 2>::Identity() };
    const Eigen::Matrix<double, 2, 2> v1_H_p1{ -Eigen::Matrix<double, 2, 2>::Identity() };
    const Eigen::Matrix<double, 2, 2> v1_H_p2{ Eigen::Matrix<double, 2, 2>::Identity() };
    const Pixel v0{ p1 - p0 };
    const Pixel v1{ p2 - p1 };

    Eigen::Matrix<double, 1, 2> err_H_v0;
    Eigen::Matrix<double, 1, 2> err_H_v1;
    const double error{ 1.0 - dot(v0, v1, err_H_v0, err_H_v1) };
    if (Hp0)
    {
      // DEBUG_VARS(v0.transpose(), v1.transpose(), error);
      *Hp0 = -err_H_v0 * v0_H_p0;
    }
    if (Hp1)
    {
      *Hp1 = -err_H_v0 * v0_H_p1 - err_H_v1 * v1_H_p1;
    }
    if (Hp2)
    {
      *Hp2 = -err_H_v1 * v1_H_p2;
    }
    return Eigen::Vector<double, 1>(error);
  }

private:
};

class increase_dist_factor_t : public gtsam::NoiseModelFactorN<Eigen::Vector2d>
{
public:
  using Base = gtsam::NoiseModelFactorN<Eigen::Vector2d>;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;
  using Pixel = Eigen::Vector2d;
  using Translation = Eigen::Vector3d;

  increase_dist_factor_t(const gtsam::Key kpx,                      // no-lint
                         const Pixel pixel, const double max_dist,  // no-lint
                         const NoiseModel& cost_model = nullptr)
    : Base(cost_model, kpx), _pixel(pixel), _max_dist(max_dist)
  {
  }

  virtual Eigen::VectorXd evaluateError(const Pixel& px,  // no-lint
                                        boost::optional<Eigen::MatrixXd&> Hpx = boost::none) const override
  {
    Eigen::Matrix<double, 1, 2> dist_H_px;
    const double dist{ gtsam::distance2(px, _pixel, dist_H_px) };
    double error{ dist - _max_dist };
    if (Hpx)
    {
      Eigen::Matrix<double, 1, 1> err_H_dist{ -1.0 / (dist * dist) };
      *Hpx = err_H_dist * dist_H_px;
    }

    // DEBUG_VARS(dist, error);
    return Eigen::Vector<double, 1>(error);
  }

private:
  const Pixel _pixel;
  const double _max_dist;
};

class pose_scale_factor_t : public gtsam::NoiseModelFactorN<gtsam::Pose2, double>
{
public:
  using SE2 = gtsam::Pose2;
  using Base = gtsam::NoiseModelFactorN<gtsam::Pose2, double>;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;
  using Pixel = Eigen::Vector2d;
  using Translation = Eigen::Vector3d;

  pose_scale_factor_t(const gtsam::Key kx, const gtsam::Key ks,  // no-lint
                      const Pixel target, const Pixel offset,    // no-lint
                      const NoiseModel& cost_model = nullptr)
    : Base(cost_model, kx, ks), _target(target), _offset(offset)
  {
  }

  virtual Eigen::VectorXd evaluateError(const SE2& x, const double& scale,  // no-lint
                                        boost::optional<Eigen::MatrixXd&> Hx = boost::none,
                                        boost::optional<Eigen::MatrixXd&> Hs = boost::none) const override
  {
    Eigen::Matrix<double, 2, 3> px_H_x;
    Eigen::Matrix<double, 2, 2> px_H_offS;
    Eigen::Matrix<double, 2, 1> offS_H_scale{ _offset };

    const Pixel offset_scale{ _offset * scale };

    const Pixel px{ x.transformFrom(offset_scale, px_H_x, px_H_offS) };

    const Pixel error{ px - _target };

    // DEBUG_VARS( error.transpose());

    if (Hx)
    {
      *Hx = px_H_x;
    }
    if (Hs)
    {
      *Hs = px_H_offS * offS_H_scale;
    }
    return error;
  }

private:
  const Pixel _target;
  const Pixel _offset;
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

int imgs_to_pointcloud(std::vector<PointColor>& pc, cv::Mat& mask, std::size_t max_pts, const Color& color)
{
  int tot{ 0 };
  std::vector<cv::Point> pixels;
  cv::findNonZero(mask, pixels);

  const std::size_t true_max_pts{ std::min(pixels.size(), max_pts) };
  for (int i = 0; i < true_max_pts; ++i)
  {
    const std::size_t idx{ factor_graphs::random_uniform<std::size_t>(0, pixels.size()) };
    auto px = pixels[idx];

    const Translation pixel(px.x, px.y, 0.0);
    // const Translation pt{ camera.backproject(pixel, z) };

    pc.emplace_back(pixel, color);
    tot++;
  }
  return tot;
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

bool inside_bar(gtsam::Pose2& bar, const double scale, const Pixel& px)
{
  const double max_y{ 0.01 };
  const double max_z{ 0.31 / 2.0 };
  const Pixel offS{ bar.inverse() * px / scale };
  if (-max_z <= offS[0] and offS[0] <= max_z)
  {
    if (-max_y <= offS[1] and offS[1] <= max_y)
    {
      return true;
    }
  }
  return false;
}

Pixel sample_endpoint()
{
  const double max_y{ 0.01 };
  const double max_z{ 0.31 / 2.0 };
  const double y{ factor_graphs::random_uniform<double>(-max_y, max_y) };
  const double z{ factor_graphs::random_uniform<double>(-1.0, 1.0) < 0.0 ? -1.0 : 1.0 };
  return Pixel(max_z * z, y);
}

void bar_pointcloud_2d(std::vector<PointColor>& pts, int total_pts)
{
  const double max_y{ 0.01 };
  const double max_z{ 0.31 / 2.0 };
  for (int i = 0; i < total_pts; ++i)
  {
    const double y{ factor_graphs::random_uniform<double>(-max_y, max_y) };
    const double z{ factor_graphs::random_uniform<double>(-max_z, max_z) };
    pts.emplace_back(Translation(z, y, 0), black);
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

struct tensegrity_3d_icp_t
{
  using This = tensegrity_3d_icp_t;
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
  CameraPtr camera;

  ros::Publisher pub0_rgb_masks, pub_red_masks, pub_green_masks, pub_blue_masks;
  ros::Publisher pub0_all_masks;
  ros::Publisher pub_red_marker;
  ros::Publisher pub_green_marker;
  ros::Publisher pub_blue_marker;
  ros::Publisher pub_img_red_marker;
  ros::Publisher pub_red_matches, pub_3Dbars;
  ros::Publisher pub_green_matches, pub_weights, pub_elipses, pub_est_bars;
  ros::Publisher pub_blue_matches, pub_img_marker, pub_lines;
  ros::Publisher pub_sobel_x, pub_sobel_y;

  cv::Mat img_rgb;
  bool use_between_factor;
  bool bars_poses_received, rgb_received, depth_received;
  gtsam::LevenbergMarquardtParams lm_params;
  std::shared_ptr<factor_graphs::levenberg_marquardt_t> lm_helper;

  int total_image_points;
  int total_sample_bars;
  const std::array<Color, 3> colors;
  const uint8_t red_idx, green_idx, blue_idx;
  // std::array<std::vector<PointColor>, 3> bars_2d;
  std::vector<PointColor> bar_2d;
  const std::vector<double> xi_vals;
  bool visualize;

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
    , colors({ blue, green, red })
    , red_idx(2)
    , green_idx(1)
    , blue_idx(0)
    , xi_vals({ 10.828, 13.816, 16.266, 18.467, 20.515, 22.458, 24.322, 26.124, 27.877, 29.588 })
    , visualize(false)
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
    PARAM_SETUP(nh, total_sample_bars);
    PARAM_SETUP(nh, total_image_points);
    PARAM_SETUP(nh, visualize);
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
    pub_lines = nh.advertise<sensor_msgs::Image>("/img/lines", 1, true);
    pub_sobel_x = nh.advertise<sensor_msgs::Image>("/img/sobel/x", 1, true);
    pub_sobel_y = nh.advertise<sensor_msgs::Image>("/img/sobel/y", 1, true);
    pub_red_masks = nh.advertise<sensor_msgs::Image>("/img/mask/red/", 1, true);
    pub_green_masks = nh.advertise<sensor_msgs::Image>("/img/mask/green/", 1, true);
    pub_blue_masks = nh.advertise<sensor_msgs::Image>("/img/mask/blue/", 1, true);
    pub_weights = nh.advertise<sensor_msgs::Image>("/img/weights", 1, true);
    pub_elipses = nh.advertise<sensor_msgs::Image>("/img/ellipses", 1, true);
    pub_est_bars = nh.advertise<sensor_msgs::Image>("/img/bars/estimated", 1, true);
    pub_3Dbars = nh.advertise<visualization_msgs::Marker>("/bars/estimated", 1, true);

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

    Eigen::Matrix4d extrinsic;
    // intrinsic << 901.67626953125, 0.0, 640.0, 0.0,        // no-lint
    //     0.0, 901.7260131835938, 360.60618591308594, 0.0,  // no-lint
    //     0.0, 0.0, 1.0, 0.0,                               // no-lint
    //     0.0, 0.0, 0.0, 1.0;
    extrinsic << 1.0, 0.0, -0.01, 0.743,  // no-lint
        0.0, -1.0, -0.007, 0.082,         // no-lint
        -0.01, 0.007, -1.0, 1.441,        // no-lint
        0.0, 0.0, 0.0, 1.0;
    gtsam::Cal3_S2 camera_calibration(901.67626953125, 901.7260131835938, 0.0, 640.0, 360.60618591308594);
    // Eigen::Matrix4d int_ext_inv{ extrinsic.inverse() * intrinsic.inverse() };
    // gtsam::Pose3 int_ext_inv{ extrinsic.inverse() * intrinsic.inverse() };
    camera = std::make_shared<Camera>(gtsam::Pose3(extrinsic), camera_calibration);

    bar_pointcloud_2d(bar_2d, max_pts);
    // bar_pointcloud_2d(bars_2d[green_idx], max_pts);
    // bar_pointcloud_2d(bars_2d[blue_idx], max_pts);
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
    // CvMatPtr grad_x{ std::make_shared<cv::Mat>() };

    // cv::Mat grad_x, grad_y;
    // cv::Sobel(frame->image, grad_x, CV_16S, 1, 0);
    // // cv::Sobel(eroded, grad_y, CV_16S, 0, 1);

    // // converting back to CV_8U
    // cv::Mat abs_grad_x, abs_grad_y;
    // cv::convertScaleAbs(grad_x, abs_grad_x);
    // // cv::convertScaleAbs(grad_y, abs_grad_y);
    // publish_img(abs_grad_x, pub_sobel_x, "mono8");
    // publish_img(abs_grad_y, pub_sobel_y, "mono8");

    depth_received = true;
  }

  void add_bar_factors(Gnn& gnn, const uint8_t idx, gtsam::NonlinearFactorGraph& graph, gtsam::Values& values)
  {
    std::vector<PointColor>& bar_pc{ bar_2d };
    gtsam::Pose2& bar_pose{ poses2[idx] };
    double& scale{ scales[idx] };
    // std::vector<std::pair<Pixel, Pixel>>& matches
    NodePtr node;
    double distance;

    const gtsam::Key kx{ gtsam::Symbol('x', idx) };
    const gtsam::Key ks{ gtsam::Symbol('s', idx) };

    // int ten_per = max_pts * 0.1;
    // for (int i = 0; i < ten_per; ++i)
    // {
    //   const Pixel offset_i{ sample_endpoint() };
    //   const Pixel offset_scale_i{ offset_i * scale * 2 };
    //   const Pixel x_off{ bar_pose * offset_scale_i };
    //   gnn.single_query(x_off, distance, node);
    //   graph.emplace_shared<pose_scale_factor_t>(kx, ks, node->value(), offset_i);
    // }

    for (int i = 0; i < max_pts; ++i)
    {
      if (gnn.size() == 0)
        break;

      PointColor pc{ bar_pc[i] };
      const Pixel offset_i{ pc.pt.head(2) };
      const Pixel offset_scale_i{ offset_i * scale };
      const Pixel x_off{ bar_pose * offset_scale_i };
      // TranslationColor tc;
      // tc.head(2) = x_off;
      // tc.tail(3) = colors[idx] * 255;
      gnn.single_query(x_off, distance, node);
      // DEBUG_VARS(tc.transpose())
      // DEBUG_VARS(node->value().transpose())

      // matches.emplace_back(std::make_pair(x_off, node->value()));

      graph.emplace_shared<pose_scale_factor_t>(kx, ks, node->value(), offset_i);

      // gnn.remove_node(node);

      // DEBUG_VARS(red_px.transpose());
      // const cv::Point red_pt{ static_cast<int>(red_px[0]), static_cast<int>(red_px[1]) };
      // const cv::Point green_pt{ static_cast<int>(green_px[0]), static_cast<int>(green_px[1]) };
      // const cv::Point blue_pt{ static_cast<int>(blue_px[0]), static_cast<int>(blue_px[1]) };
      // cv::drawMarker(img_lines, red_pt, PointColor::color_to_cv(red), cv::MARKER_TILTED_CROSS, 2, 1);
      // cv::drawMarker(img_lines, green_pt, PointColor::color_to_cv(green), cv::MARKER_TILTED_CROSS, 2, 1);
      // cv::drawMarker(img_lines, blue_pt, PointColor::color_to_cv(blue), cv::MARKER_TILTED_CROSS, 2, 1);
    }
  }

  void apply_similarity_to_image(cv::Mat& img, const uint8_t idx)
  {
    std::vector<PointColor>& bar_pc{ bar_2d };
    gtsam::Pose2& bar_pose{ poses2[idx] };
    double& scale{ scales[idx] };
    // const Color& color{ colors[idx % 3] };
    const cv::Scalar color{ 255, 255, 255 };

    // std::vector<std::pair<Pixel, Pixel>>& matches
    for (auto pt : bar_pc)
    {
      const Pixel px{ pt.pt.head(2) * scale };
      const Pixel pxp{ bar_pose * px };

      // DEBUG_VARS(px.transpose(), pxp.transpose());

      const cv::Point cvpt{ static_cast<int>(pxp[0]), static_cast<int>(pxp[1]) };
      cv::drawMarker(img, cvpt, color, cv::MARKER_TILTED_CROSS, 2, 1);
    }
    // for (auto pair : matches)
    // {
    //   cv::Point p0(pair.first[0], pair.first[1]);
    //   cv::Point p1(pair.second[0], pair.second[1]);

    //   cv::line(img, p0, p1, cv::Scalar(217, 28, 201), 1);
    // }
  }

  void add_to_gnn(Gnn& gnn, const double diameter, const Pixel& centroid, cv::Mat& img, int cap_pts,
                  std::vector<uint8_t> bars_to_ignore)
  {
    std::vector<cv::Point> pixels;
    cv::findNonZero(img, pixels);
    const int max_target{ std::min(cap_pts, static_cast<int>(pixels.size())) };

    int i = gnn.size();
    int tot_iters{ 0 };

    while (gnn.size() < max_target and tot_iters < max_target * 2)
    {
      const std::size_t idx{ factor_graphs::random_uniform<std::size_t>(0, pixels.size()) };
      const cv::Point& ptcv{ pixels[idx] };
      const Pixel pt{ ptcv.x, ptcv.y };
      const double dist{ (centroid - pt).norm() };

      // for (auto bar_idx : bars_to_ignore)
      // {
      //   if (inside_bar(poses2[bar_idx], scales[bar_idx], pt))
      //     continue;
      // }

      if (dist < diameter)
      {
        pixels.erase(pixels.begin() + idx);
        // TranslationColor tc;
        // tc.head(2) = pt;
        // tc[3] = _frame_colors[blue_idx].at<uint8_t>(ptcv.y, ptcv.x);
        // tc[4] = _frame_colors[green_idx].at<uint8_t>(ptcv.y, ptcv.x);
        // tc[5] = _frame_colors[red_idx].at<uint8_t>(ptcv.y, ptcv.x);
        gnn.emplace(pt, i);
        i++;
      }
      tot_iters++;
    }
  }

  // Draw a bar as a thick line
  void add_bar(cv::Mat& img, const gtsam::Pose2& pose, const double scale, const Color color)
  {
    const Pixel off_p{ scale * px_offset };
    const Pixel off_m{ -scale * px_offset };

    const Pixel px_p{ pose.transformFrom(off_p) };
    const Pixel px_m{ pose.transformFrom(off_m) };
    cv::Point p0{ static_cast<int>(px_p[0]), static_cast<int>(px_p[1]) };
    cv::Point p1{ static_cast<int>(px_m[0]), static_cast<int>(px_m[1]) };
    cv::line(img, p0, p1, PointColor::color_to_cv(color), 5);
  }

  void add_ellipse(cv::Mat& img, const Pixel& mu, const Eigen::Matrix2d& sigma, const Color color)
  {
    if (std::isnan(mu.template maxCoeff<Eigen::PropagateNaN>()) and
        std::isnan(sigma.template maxCoeff<Eigen::PropagateNaN>()))
      return;

    // DEBUG_VARS(sigma);
    const cv::Point pt{ static_cast<int>(mu[0]), static_cast<int>(mu[1]) };
    const double a{ sigma(0, 0) };
    const double b{ sigma(0, 1) };
    const double c{ sigma(1, 1) };
    const double lambda1{ ((a + c) + std::sqrt(std::pow((a - c) / 2, 2) + 4 * b * b)) / 2.0 };
    const double lambda2{ ((a + c) - std::sqrt(std::pow((a - c) / 2, 2) + 4 * b * b)) / 2.0 };
    double angle{ 0 };
    if (std::fabs(b) < 1e-3)
    {
      if (a >= c)
      {
        angle = 0;
      }
      else
      {
        angle = tensegrity::constants::pi / 2.0;
      }
    }
    else
    {
      angle = std::atan2(lambda1 - a, b);
    }
    // DEBUG_VARS(a, b, c, lambda1, lambda2, angle);
    cv::ellipse(img, pt, cv::Size(lambda1, lambda2), angle, 0, 360, PointColor::color_to_cv(color));
  }

  void compute_endcaps(std::vector<bar2D_t>& estimated_bars, CvMatPtr& all_masks)
  {
    cv::Mat img_bars{ cv::Mat::zeros(img_rgb.rows, img_rgb.cols, img_rgb.type()) };
    cv::Mat ones{ cv::Mat::ones(img_rgb.rows, img_rgb.cols, _frame_black.type()) };
    ones = ones * poses2.size();
    cv::Mat channels[3];
    cv::split(img_bars, channels);

    std::vector<cv::Point> non_zeros;
    // bar2D_t bar_aux;
    Pixel muA, muB;
    Eigen::Matrix2d sigmaA, sigmaB;
    // std::vector<bar2D_t> bars;
    std::array<EndcapsBeliefs, 3> endcaps;
    std::array<ScaleBeliefs, 3> scale_beliefs;

    std::array<std::vector<double>, 3> bar_scales;
    // get all the endcaps where there is color, it could be more than the total endcaps (if overalping)
    // std::vector<std::vector<std::pair<Pixel, Pixel>>> means(3);
    // std::vector<std::vector<std::pair<Eigen::Matrix2d, Eigen::Matrix2d>>> stddevs(3);
    for (int i = 0; i < poses2.size(); ++i)
    {
      const double& s{ scales[i] };
      const gtsam::Pose2 x{ poses2[i] };

      // DEBUG_VARS(s);
      const double wh{ s * 0.04 };
      const double area_tot{ wh * wh };
      cv::Point rect_center(x.x(), x.y());
      cv::Size rect_size(s * 0.325 / 2.0, s * 0.04);
      const Pixel off_p{ s * px_offset };
      const Pixel off_m{ -s * px_offset };

      const Pixel px_p{ x.transformFrom(off_p) };
      const Pixel px_m{ x.transformFrom(off_m) };

      const double cols{ static_cast<double>(_frame_colors[0].cols) };
      const double rows{ static_cast<double>(_frame_colors[0].rows) };
      const Pixel px_cp{ std::max(0.0, std::min(px_p[0] - wh / 2, cols)),
                         std::max(0.0, std::min(px_p[1] - wh / 2, rows)) };
      const Pixel px_cm{ std::max(0.0, std::min(px_m[0] - wh / 2, cols)),
                         std::max(0.0, std::min(px_m[1] - wh / 2, rows)) };  // top left corner
      // const cv::RotatedRect rot_rect{ cv::RotatedRect(rect_center, rect_size, x.theta()) };

      const double recp_width{ std::min(_frame_colors[0].cols - px_cp[0], wh) };
      const double recp_height{ std::min(_frame_colors[0].rows - px_cp[1], wh) };
      const double recm_width{ std::min(_frame_colors[0].cols - px_cm[0], wh) };
      const double recm_height{ std::min(_frame_colors[0].rows - px_cm[1], wh) };
      const cv::Rect rec_p(px_cp[0], px_cp[1], recp_width, recp_height);
      const cv::Rect rec_m(px_cm[0], px_cm[1], recm_width, recm_height);
      // cv::RotatedRect rotatedRectangle(centerPoint, rectangleSize, rotationDegrees);

      // for (uint8_t i = 0; i < 3; ++i)
      for (uint8_t j = 0; j < 3; ++j)
      {
        cv::Moments mm_p{ cv::moments(_frame_colors[j](rec_p), true) };
        cv::Moments mm_m{ cv::moments(_frame_colors[j](rec_m), true) };

        // bar_aux.pose = x;
        // bar_aux.scale = s;
        muA = px_cp + Pixel(mm_p.m10 / mm_p.m00, mm_p.m01 / mm_p.m00);
        muB = px_cm + Pixel(mm_m.m10 / mm_m.m00, mm_m.m01 / mm_m.m00);
        sigmaA(0, 0) = mm_p.mu20 / mm_p.m00;
        sigmaA(0, 0) = mm_p.mu20 / mm_p.m00;
        sigmaA(0, 1) = mm_p.mu11 / mm_p.m00;
        sigmaA(1, 0) = mm_p.mu11 / mm_p.m00;
        sigmaA(1, 1) = mm_p.mu02 / mm_p.m00;
        sigmaB(0, 0) = mm_m.mu20 / mm_m.m00;
        sigmaB(0, 1) = mm_m.mu11 / mm_m.m00;
        sigmaB(1, 0) = mm_m.mu11 / mm_m.m00;
        sigmaB(1, 1) = mm_m.mu02 / mm_m.m00;

        const bool validA{ mm_p.m00 > 1 and mm_p.mu20 > 1 and mm_p.mu02 > 1 and
                           std::fabs(sigmaA.determinant()) > 1e-1 };
        const bool validB{ mm_m.m00 > 1 and mm_m.mu20 > 1 and mm_m.mu02 > 1 and
                           std::fabs(sigmaB.determinant()) > 1e-1 };

        if (validA and cv::countNonZero(_frame_colors[j](rec_p)) > 10)
        {
          endcaps[j].push_back(std::make_pair(muA, sigmaA));
          scale_beliefs[j].push_back(std::make_pair(s, scales_cov[i]));
          if (visualize)
            cv::add(channels[j](rec_p), ones(rec_p), channels[j](rec_p), _frame_colors[j](rec_p));
          // cv::rectangle(channels[j], rec_p, cv::Scalar(255, 255, 255), 1, cv::LINE_8);
        }
        else
        {
          // cv::findNonZero(_frame_colors[j](rec_m), non_zeros);
          if (validB and cv::countNonZero(_frame_colors[j](rec_m)) > 10)
          {
            endcaps[j].push_back(std::make_pair(px_p, 2 * sigmaB));
            scale_beliefs[j].push_back(std::make_pair(s, scales_cov[i]));
          }
        }

        if (validB and cv::countNonZero(_frame_colors[j](rec_m)) > 10)
        {
          // const double ratio{ (area_tot - mm_m.m00) / (area_tot) };
          endcaps[j].push_back(std::make_pair(muB, sigmaB));
          scale_beliefs[j].push_back(std::make_pair(s, scales_cov[i]));
          // DEBUG_VARS(ratio)
          if (visualize)
            cv::add(channels[j](rec_m), ones(rec_m), channels[j](rec_m), _frame_colors[j](rec_m));
          // cv::rectangle(channels[j], rec_m, cv::Scalar(255, 255, 255), 1, cv::LINE_8);
          // DEBUG_VARS(muB.transpose())
        }
        else
        {
          // cv::findNonZero(_frame_colors[j](rec_p), non_zeros);
          if (validA and cv::countNonZero(_frame_colors[j](rec_p)) > 10)
          {
            endcaps[j].push_back(std::make_pair(px_m, 2 * sigmaA));
            scale_beliefs[j].push_back(std::make_pair(s, scales_cov[i]));
          }
        }
      }
    }

    cv::Mat img_est_bars;
    cv::Mat img_elipses;
    if (visualize)
    {
      cv::normalize(channels[0], channels[0], 255.0, 0.0, cv::NORM_INF);
      cv::normalize(channels[1], channels[1], 255.0, 0.0, cv::NORM_INF);
      cv::normalize(channels[2], channels[2], 255.0, 0.0, cv::NORM_INF);
      cv::merge(channels, 3, img_bars);
      img_bars.copyTo(img_elipses);
      img_bars.copyTo(img_est_bars);

      for (auto& color_endcaps : endcaps)
      {
        for (auto& endcap : color_endcaps)
        {
          add_ellipse(img_bars, endcap.first, endcap.second, Color(120, 250, 255));
        }
      }
      publish_img(img_bars, pub_weights, "bgr8");
    }

    std::array<EndcapsBeliefs, 3> updated_endcaps;
    std::array<ScaleBeliefs, 3> updated_scales;
    for (int i = 0; i < 3; ++i)
    {
      cluster_endcaps(updated_endcaps[i], updated_scales[i], endcaps[i], scale_beliefs[i]);

      remove_outliers(updated_endcaps[i], updated_scales[i]);
    }

    if (visualize)
    {
      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < updated_endcaps[i].size(); ++j)
        {
          add_ellipse(img_elipses, updated_endcaps[i][j].first, updated_endcaps[i][j].second, Color(120, 250, 255));
        }
      }
      publish_img(img_elipses, pub_elipses, "bgr8");
    }

    for (int i = 0; i < 3; ++i)
    {
      bars_from_endcaps(estimated_bars, updated_endcaps[i], updated_scales[i], all_masks, i);
    }

    if (visualize)
    {
      // img_elipses.copyTo(img_est_bars);
      for (int i = 0; i < estimated_bars.size(); ++i)
      {
        add_bar(img_est_bars, estimated_bars[i].pose, estimated_bars[i].scale, colors[estimated_bars[i].color_id]);
      }
      publish_img(img_est_bars, pub_est_bars, "bgr8");
    }

    // if (visualize)
    // {
    //   publish_img(img_bars, pub_weights, "bgr8");
    // }
  }

  void remove_outliers(EndcapsBeliefs& endcaps, ScaleBeliefs& scale_beliefs)
  {
    if (scale_beliefs.size() == 0)
      return;

    std::size_t total = scale_beliefs.size();
    // DEBUG_VARS(mean, total);
    for (int i = total - 1; i >= 0; --i)
    {
      // const double diff{ scale_beliefs[i].first - mean };
      // const double dist{ diff * (1.0 / scale_beliefs[i].second) * diff };
      if (scale_beliefs[i].second > 100)  // Too hacky
      {
        // DEBUG_VARS(i, dist, xi_vals[0])
        endcaps.erase(endcaps.begin() + i);
        scale_beliefs.erase(scale_beliefs.begin() + i);
      }
    }
  }

  void cluster_endcaps(EndcapsBeliefs& out_endcaps, ScaleBeliefs& out_scales, EndcapsBeliefs& endcaps,
                       ScaleBeliefs& scale_beliefs)
  {
    using GaussianNM = gtsam::noiseModel::Gaussian;
    // DEBUG_VARS(endcaps.size());
    if (endcaps.size() == 0 or scale_beliefs.size() == 0)
      return;

    Eigen::Matrix2d sigma_inv;
    gtsam::Values values;
    gtsam::NonlinearFactorGraph graph;

    double prev_error{ 0 };
    double adjusted_error{ 0 };
    EndcapsBeliefs rejected;
    ScaleBeliefs rejected_scales;
    gtsam::Key key_endcap{ gtsam::Symbol('A', 0) };
    gtsam::Key key_scale{ gtsam::Symbol('s', 0) };
    values.insert(key_endcap, endcaps[0].first);
    values.insert(key_scale, scale_beliefs[0].first);
    std::size_t clustered{ 0 };
    for (int i = 0; i < endcaps.size(); ++i)
    {
      clustered++;
      gtsam::NonlinearFactorGraph gDelta;

      const Pixel zi{ endcaps[i].first };
      sigma_inv = endcaps[i].second;
      GaussianNM::shared_ptr z_noise{ GaussianNM::Information(sigma_inv.inverse()) };
      const gtsam::noiseModel::Base::shared_ptr scale_nm{ gtsam::noiseModel::Isotropic::Sigma(
          1, scale_beliefs[i].second) };

      gDelta.addPrior(key_endcap, zi, z_noise);
      gDelta.addPrior(key_scale, scale_beliefs[i].first, scale_nm);

      gtsam::FactorIndices indices{ graph.add_factors(gDelta) };
      gtsam::Values result{ lm_helper->optimize(graph, values, true) };
      const double error{ graph.error(result) };
      adjusted_error = error;
      if (error - prev_error > xi_vals[std::min(clustered, xi_vals.size())])
      {
        for (auto idx : indices)
        {
          graph.remove(idx);
        }
        // values.erase(key_endcap);
        result = lm_helper->optimize(graph, values, true);
        adjusted_error = graph.error(result);
        // values = result;
        rejected.push_back(endcaps[i]);
        rejected_scales.push_back(scale_beliefs[i]);
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
    const Pixel endcap_res{ values.at<Pixel>(key_endcap) };
    const double scale_res{ values.at<double>(key_scale) };
    const Eigen::Matrix2d res_cov{ marginals.marginalCovariance(key_endcap) };
    const Eigen::Matrix<double, 1, 1> scale_cov{ marginals.marginalCovariance(key_scale) };
    // const double scale{ values.at<double>(key_scale) };
    // const gtsam::Pose2 pose{ values.at<gtsam::Pose2>(key_se2) };
    // DEBUG_VARS(endcap_res.transpose(), scale_res);
    // DEBUG_VARS(res_cov);
    // DEBUG_VARS(scale_cov);

    out_endcaps.push_back({ endcap_res, res_cov });
    out_scales.push_back({ scale_res, scale_cov(0, 0) });
    if (rejected.size() > 0)
    {
      // EndcapsBeliefs res{ cluster_endcaps(rejected, rejected_scales) };
      // res.push_back({ endcap_res, res_cov });
      cluster_endcaps(out_endcaps, out_scales, rejected, rejected_scales);
      // return res;
    }
  }

  template <typename Element, typename Covs>
  void cluster(std::vector<Element>& vals_out, std::vector<Covs>& covs_out,  // no-lint
               const std::vector<Element>& vals_in, const std::vector<Covs>& covs_in)
  {
    using GaussianNM = gtsam::noiseModel::Gaussian;
    // DEBUG_VARS(endcaps.size());
    if (vals_in.size() == 0 or covs_in.size() == 0)
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
      if (error - prev_error > xi_vals[std::min(clustered, xi_vals.size())])
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

  void bars_from_endcaps(std::vector<bar2D_t>& out_bars, EndcapsBeliefs& endcaps, ScaleBeliefs& scale_beliefs,
                         CvMatPtr& masks, const uint8_t color_id)
  {
    using EndcapObservationFactor = perception::endcap_2Dobservation_scale_fix_factor_t;
    using RotationOffsetFactor = perception::endcap_2Drotation_offset_factor_t;
    using RotationFixIdentity = perception::rotation2D_fix_identity_t;
    using GaussianNM = gtsam::noiseModel::Gaussian;

    gtsam::Values values;
    gtsam::NonlinearFactorGraph graph;
    // const gtsam::Rot2 roff2D(tensegrity::constants::pi);
    // int tot_rots{ 0 };
    int tot_poses{ 0 };
    // std::vector<double> closest_distances;
    // std::vector<GnnMod::NodePtr> closest_nodes;
    gtsam::noiseModel::Base::shared_ptr rot_prior_nm{ gtsam::noiseModel::Isotropic::Sigma(1, 1e-4) };
    // std::vector<EndcapObservationFactor> test_factors;
    for (int i = 0; i < endcaps.size(); ++i)
    {
      const Pixel zA{ endcaps[i].first };
      for (int j = i + 1; j < endcaps.size(); ++j)
      {
        // DEBUG_VARS(i, j)
        const gtsam::Key key_se2{ gtsam::Symbol('x', tot_poses) };
        const gtsam::Key key_scale{ gtsam::Symbol('s', tot_poses) };
        const Pixel zB{ endcaps[j].first };
        const Pixel center{ (zA + zB) / 2.0 };
        const Pixel v0{ (zA - center).normalized() };
        const double dot{ v0.dot(Pixel(1.0, 0.0)) };
        values.insert(key_se2, gtsam::Pose2(gtsam::Rot2(std::acos(dot)), center));
        // values.insert(key_se2, gtsam::Pose2(gtsam::Rot2(std::acos(diff.)), (zA + zB) / 2.0));
        // values.insert(key_se2, gtsam::Pose2());
        values.insert(key_scale, scale_beliefs[j].first);
        graph.emplace_shared<pose_scale_factor_t>(key_se2, key_scale, zA, px_offset);
        graph.emplace_shared<pose_scale_factor_t>(key_se2, key_scale, zB, -px_offset);
        tot_poses++;
        // DEBUG_VARS(i, j, zA.transpose(), zB.transpose(), scale_beliefs[j].first);
        // graph.emplace_shared<EndcapObservationFactor>(key_se2, key_rotA_offset, zA, px_offset, scale, nullptr);
        // graph.emplace_shared<EndcapObservationFactor>(key_se2, key_rotB_offset, zB, px_offset, scale, nullptr);
        // graph.emplace_shared<RotationOffsetFactor>(key_rotA_offset, key_rotB_offset, roff2D, rot_prior_nm);
        // graph.emplace_shared<RotationOffsetFactor>(key_rotB_offset, key_rotA_offset, roff2D, rot_prior_nm);
        // graph.emplace_shared<RotationFixIdentity>(key_rotA_offset, key_rotB_offset, roff2D, rot_prior_nm);
      }
    }

    const gtsam::Values result{ lm_helper->optimize(graph, values, true) };
    const double error{ graph.error(result) };
    // result.print("result");
    // graph.printErrors(result, "Errors");
    // DEBUG_VARS(error, tot_poses);
    std::unordered_map<gtsam::Key, bool> found_ids;
    for (auto factor : graph)
    {
      const double error{ factor->error(result) };
      const gtsam::Key key_se2{ factor->keys()[0] };
      const gtsam::Key key_scale{ factor->keys()[1] };
      if (error < 1 and found_ids.count(key_se2) == 0)
      {
        const double scale{ result.at<double>(key_scale) };
        const gtsam::Pose2 pose{ result.at<gtsam::Pose2>(key_se2) };
        out_bars.emplace_back(pose, std::fabs(scale), color_id);
        // pose.print("result_pose ");
        // DEBUG_VARS(scale)
        if (not valid_bar(out_bars.back(), masks))
        {
          out_bars.pop_back();
        }
      }
      found_ids.insert({ key_se2, true });
      // else
      // {
      //   // factor->print("rejected");
      // }
    }
    // DEBUG_VARS(result_bars.size())
    // return result_bars;
  }

  bool valid_bar(bar2D_t& bars, CvMatPtr& masks)
  {
    // bars.pose.print("bar_pose");
    double step{ 1.0 / max_pts };
    int zeros{ 0 };
    const Pixel scaled_offset{ px_offset * bars.scale };
    // const Pixel
    for (int i = 0; i < max_pts / 2; ++i)
    {
      const double curr_step{ i * step };
      // DEBUG_VARS(curr_step, scaled_offset.transpose())
      const Pixel pxp{ bars.pose.transformFrom(Pixel(scaled_offset * curr_step)) };
      const Pixel pxm{ bars.pose.transformFrom(Pixel(-scaled_offset * curr_step)) };
      cv::Point ptp{ static_cast<int>(pxp[0]), static_cast<int>(pxp[1]) };
      cv::Point ptm{ static_cast<int>(pxm[0]), static_cast<int>(pxm[1]) };
      // DEBUG_VARS(pxp.transpose(), pxm.transpose());
      if (masks->at<uint8_t>(ptp) == 0)
      {
        zeros++;
      }
      if (masks->at<uint8_t>(ptm) == 0)
      {
        zeros++;
      }
    }
    // DEBUG_VARS(zeros, max_pts)
    return zeros < max_pts * 0.3;
  }
  bool above_ground_bar(const gtsam::Pose3& bar_pose)
  {
    double step{ 1.0 / max_pts };
    int below_zero{ 0 };
    // const Pixel
    for (int i = 0; i < max_pts / 2; ++i)
    {
      const double curr_step{ i * step };
      const Translation ptp{ bar_pose.transformFrom(Translation(offset * curr_step)) };
      const Translation ptm{ bar_pose.transformFrom(Translation(-offset * curr_step)) };
      if (ptp[2] < 0)
      {
        below_zero++;
      }
      if (ptm[2] < 0)
      {
        below_zero++;
      }
    }
    return below_zero < max_pts * 0.3;
  }

  std::vector<std::pair<gtsam::Pose2, double>> cluster_bars(std::vector<bar2D_t>& bars)
  {
    using EndcapObservationFactor = perception::endcap_2Dobservation_factor_t;
    using RotationOffsetFactor = perception::endcap_2Drotation_offset_factor_t;
    using RotationFixIdentity = perception::rotation2D_fix_identity_t;
    using GaussianNM = gtsam::noiseModel::Gaussian;

    Eigen::Matrix2d sigma_inv;
    gtsam::noiseModel::Base::shared_ptr rot_prior_nm{ gtsam::noiseModel::Isotropic::Sigma(1, 1e-4) };

    const gtsam::Rot2 roff2D(tensegrity::constants::pi);
    // GaussianNM unobse
    gtsam::Values values;
    gtsam::NonlinearFactorGraph graph;
    const gtsam::Key key_se2{ gtsam::Symbol('x', 0) };
    const gtsam::Key key_scale{ gtsam::Symbol('s', 0) };
    values.insert(key_se2, bars[0].pose);
    values.insert(key_scale, bars[0].scale);
    double prev_error{ 0 };
    double adjusted_error{ 0 };
    std::vector<bar2D_t> rejected;
    for (int i = 0; i < bars.size(); ++i)
    {
      gtsam::NonlinearFactorGraph gDelta;

      if (bars[i].color_id == 0)
      {
        const double& scale{ bars[i].scale };
        const gtsam::Pose2& pose{ bars[i].pose };
        // const gtsam::Key key_scale{ gtsam::Symbol('s', i) };
        const gtsam::Key key_rotA_offset{ gtsam::Symbol('r', i) };
        const gtsam::Key key_rotB_offset{ gtsam::Symbol('t', i) };
        // values.insert(key_scale, bars[i].scale);

        values.insert(key_rotA_offset, gtsam::Rot2());
        values.insert(key_rotB_offset, roff2D);

        if (bars[i].endcapA and bars[i].sigmaA)
        {
          const Pixel zA{ bars[i].endcapA.value() };
          sigma_inv = bars[i].sigmaA.value().inverse();
          GaussianNM::shared_ptr z_noise{ GaussianNM::Information(sigma_inv) };

          gDelta.emplace_shared<EndcapObservationFactor>(key_se2, key_scale, key_rotA_offset, zA, px_offset, z_noise);
          // DEBUG_VARS(i, zA.transpose());
        }
        if (bars[i].endcapB and bars[i].sigmaB)
        {
          const Pixel zB{ bars[i].endcapB.value() };
          GaussianNM::shared_ptr z_noise{ GaussianNM::Information(sigma_inv) };
          gDelta.emplace_shared<EndcapObservationFactor>(key_se2, key_scale, key_rotB_offset, zB, px_offset, z_noise);
          // DEBUG_VARS(i, zB.transpose());
        }

        gDelta.emplace_shared<RotationOffsetFactor>(key_rotA_offset, key_rotB_offset, roff2D, rot_prior_nm);
        gDelta.emplace_shared<RotationOffsetFactor>(key_rotB_offset, key_rotA_offset, roff2D, rot_prior_nm);

        gDelta.emplace_shared<RotationFixIdentity>(key_rotA_offset, key_rotB_offset, roff2D, rot_prior_nm);

        gtsam::FactorIndices indices{ graph.add_factors(gDelta) };
        gtsam::Values result{ lm_helper->optimize(graph, values, true) };
        // graph.printErrors(result, "Priors");
        const double error{ graph.error(result) };
        adjusted_error = error;
        if (error - prev_error > 5)  // chi^2 for df=3 and p=0.01 is 9.2~10, the FG returns 0.5 * D;
        {
          for (auto idx : indices)
          {
            graph.remove(idx);
          }
          // values.erase(key_scale);
          values.erase(key_rotA_offset);
          values.erase(key_rotB_offset);
          result = lm_helper->optimize(graph, values, true);
          adjusted_error = graph.error(result);
          rejected.push_back(bars[i]);
        }
        // DEBUG_VARS(i, error, adjusted_error);
        prev_error = adjusted_error;
      }
    }
    const double scale{ values.at<double>(key_scale) };
    const gtsam::Pose2 pose{ values.at<gtsam::Pose2>(key_se2) };
    if (rejected.size() > 1)
    {
      std::vector<std::pair<gtsam::Pose2, double>> res{ cluster_bars(rejected) };
      res.push_back({ pose, scale });
      return res;
    }
    else
    {
      return { { pose, scale } };
    }
  }

  void compute_lines(CvMatPtr& all_masks)
  {
    cv::Mat circle_mask{ cv::Mat::zeros(_frame_black.rows, _frame_black.cols, _frame_black.type()) };
    std::vector<std::vector<cv::Point>> contours_red, contours_green, contours_blue, all_contours;
    cv::findContours(*all_masks, all_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    // cv::findContours(_frame_red0, all_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    // cv::findContours(_frame_green, contours_green, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    // cv::findContours(_frame_blue, contours_blue, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::sort(all_contours.begin(), all_contours.end(),
              [](cv::InputArray a, cv::InputArray b) { return cv::contourArea(a) > cv::contourArea(b); });

    // CvMatPtr grad_x{ std::make_shared<cv::Mat>() };
    // CvMatPtr grad_y{ std::make_shared<cv::Mat>() };
    // // cv::Mat abs_grad_x, abs_grad_y;
    // cv::Sobel(*all_masks, *grad_x, CV_16S, 1, 0);
    // cv::Sobel(*all_masks, *grad_y, CV_16S, 0, 1);

    // converting back to CV_8U
    // cv::Mat abs_grad_x, abs_grad_y;
    // cv::convertScaleAbs(*grad_x, abs_grad_x);
    // cv::convertScaleAbs(*grad_y, abs_grad_y);
    // publish_img(abs_grad_x, pub_sobel_x, "mono8");
    // publish_img(abs_grad_y, pub_sobel_y, "mono8");

    // Eigen::Vector2d centroid{ Eigen::Vector2d::Zero() };
    int total_endcaps{ 0 };
    // const std::array<Color, 3> colors{ blue, green, red };
    std::array<int, 3> total_color{ 0, 0, 0 };

    // std::vector<std::pair<Eigen::Vector2d, Color>> endcaps;

    // cv::drawContours(img_lines, all_contours, 0, cv::Scalar(0, 0, 255), 1);
    cv::Moments mc(cv::moments(all_contours[0], false));
    const double xc{ mc.m10 / mc.m00 };
    const double yc{ mc.m01 / mc.m00 };
    const Pixel centroid{ xc, yc };

    const double area{ cv::contourArea(all_contours[0]) };
    const double diameter{ std::sqrt(4.0 * area / tensegrity::constants::pi) * 1.5 };
    // cv::circle(img_lines, cv::Point(xc, yc), diameter, PointColor::color_to_cv(red));
    cv::circle(circle_mask, cv::Point(xc, yc), diameter, cv::Scalar(255, 255, 255), -1);
    cv::bitwise_and(_frame_colors[red_idx], circle_mask, _frame_colors[red_idx]);
    cv::bitwise_and(_frame_colors[green_idx], circle_mask, _frame_colors[green_idx]);
    cv::bitwise_and(_frame_colors[blue_idx], circle_mask, _frame_colors[blue_idx]);

    // cv::Moments red_ms{ cv::moments(_frame_colors[red_idx], true) };
    // cv::Moments green_ms{ cv::moments(_frame_colors[green_idx], true) };
    // cv::Moments blue_ms{ cv::moments(_frame_colors[blue_idx], true) };
    // const Pixel red_centroid{ red_ms.m10 / red_ms.m00, red_ms.m01 / red_ms.m00 };
    // const Pixel green_centroid{ green_ms.m10 / green_ms.m00, green_ms.m01 / green_ms.m00 };
    // const Pixel blue_centroid{ blue_ms.m10 / blue_ms.m00, blue_ms.m01 / blue_ms.m00 };

    // std::vector<std::pair<double, uint8_t>> centroids;
    // centroids.push_back(std::make_pair((centroid - red_centroid).norm(), red_idx));
    // centroids.push_back(std::make_pair((centroid - green_centroid).norm(), green_idx));
    // centroids.push_back(std::make_pair((centroid - blue_centroid).norm(), blue_idx));
    // std::sort(centroids.begin(), centroids.end(),
    //           [](std::pair<double, uint8_t> a, std::pair<double, uint8_t> b) { return a.first < b.first; });

    // ForwardIt min_element(ForwardIt first, ForwardIt last);
    // cv::drawMarker(img_lines, cv::Point(red_centroid[0], red_centroid[1]), PointColor::color_to_cv(red),
    //                cv::MARKER_SQUARE, 10, 1);
    // cv::drawMarker(img_lines, cv::Point(green_centroid[0], green_centroid[1]), PointColor::color_to_cv(green),
    //                cv::MARKER_SQUARE, 10, 1);
    // cv::drawMarker(img_lines, cv::Point(blue_centroid[0], blue_centroid[1]), PointColor::color_to_cv(blue),
    //                cv::MARKER_SQUARE, 10, 1);
    const double _2pi{ 2.0 * tensegrity::constants::pi };
    const double max_eps{ diameter / 4.0 };
    poses2.clear();
    scales.clear();
    for (int i = 0; i < total_sample_bars; ++i)
    {
      // const double epsX{ factor_graphs::random_uniform<double>(-max_eps, max_eps) };
      // const double epsY{ factor_graphs::random_uniform<double>(-max_eps, max_eps) };
      // const double angle{ factor_graphs::random_uniform<double>(0, _2pi) };
      const double angle{ _2pi * i / total_sample_bars };

      poses2.push_back(gtsam::Pose2(xc, yc, angle));
      scales.push_back(diameter * 8);
      // poses2.back().print("pose");
    }
    // poses2[red_idx] = gtsam::Pose2(xc, yc, 0.0);
    // poses2[green_idx] = gtsam::Pose2(xc, yc, 2 * tensegrity::constants::pi / 3.0);
    // poses2[blue_idx] = gtsam::Pose2(xc, yc, 4 * tensegrity::constants::pi / 3.0);
    // scales[red_idx] = 750;
    // scales[green_idx] = 750;
    // scales[blue_idx] = 750;
    // red_similarity =
    // std::vector<double> errors(total_sample_bars, 1000);
    double error{ 1000 };
    double derr{ 1000 };
    // std::array<Gnn, 3> gnn_color;
    // add_to_gnn(gnn_red, diameter, centroid, _frame_red0);
    // add_to_gnn(gnn_green, diameter, centroid, _frame_green);
    // add_to_gnn(gnn_blue, diameter, centroid, _frame_blue);
    // std::vector<uint8_t> bars_to_ignore;

    int iter{ 0 };
    const int tot_iter{ 5 };
    Gnn gnn;
    add_to_gnn(gnn, diameter, centroid, *all_masks, total_image_points, {});

    while (error > 1.0 and derr > poses2.size() and iter < tot_iter)
    {
      // cv::Mat lines{ cv::Mat::zeros(img_lines.rows, img_lines.cols, img_lines.type()) };

      gtsam::Values values;
      gtsam::NonlinearFactorGraph graph;

      // for (auto pair : centroids)
      for (int i = 0; i < poses2.size(); ++i)
      {
        const gtsam::Key kx{ gtsam::Symbol('x', i) };
        const gtsam::Key ks{ gtsam::Symbol('s', i) };
        values.insert(kx, poses2[i]);
        values.insert(ks, scales[i]);
        // const uint8_t color_idx{ pair.second };
        // if (errors[i] > 1.0)
        // {
        add_bar_factors(gnn, i, graph, values);
        // }
      }
      const gtsam::Values result{ lm_helper->optimize(graph, values, true) };
      const gtsam::Marginals marginals(graph, result);

      const double prev_error{ error };
      error = graph.error(result);
      derr = std::fabs(prev_error - error);
      for (int i = 0; i < poses2.size(); ++i)
      {
        const gtsam::Key kx{ gtsam::Symbol('x', i) };
        const gtsam::Key ks{ gtsam::Symbol('s', i) };
        poses2[i] = result.at<gtsam::Pose2>(kx);
        scales[i] = result.at<double>(ks);

        scales_cov.push_back(marginals.marginalCovariance(ks)(0, 0));

        // apply_similarity_to_image(img_lines, i);
      }
      // return graph.error(result);

      // green_error = add_bar_factors(gnn, green_idx);
      // blue_error = add_bar_factors(gnn, blue_idx);

      // apply_similarity_to_image(img_lines, red_idx);
      // apply_similarity_to_image(img_lines, green_idx);
      // apply_similarity_to_image(img_lines, blue_idx);

      iter++;
      // DEBUG_VARS(error, derr, iter);
    }
    if (visualize)
    {
      cv::Mat img_lines;
      cv::copyTo(img_rgb, img_lines, *all_masks);
      for (int i = 0; i < poses2.size(); ++i)
      {
        apply_similarity_to_image(img_lines, i);
      }
      publish_img(img_lines, pub_lines, "bgr8");
    }
    // DEBUG_VARS(error, derr, iter);
    // apply_similarity_to_image(img_lines, color_idx);
    // publish_img(img_lines, pub_lines, "bgr8");
  }

  void point3D_to_image(cv::Mat& img, const PointColor& p)
  {
    Translation pw{ p.pt };
    Color color{ p.color * 255 };
    // Eigen::Vector2d pixel{ camera.project2(pw) };
    auto [pixel, valid] = camera->projectSafe(pw);

    if (valid)
    {
      int u = pixel[0];
      int v = pixel[1];
      img.at<cv::Vec3b>(v, u) = cv::Vec3b(color[0], color[1], color[2]);
    }
  }

  void bars_to_marker(std::array<std::vector<gtsam::Pose3>, 3>& bars)
  {
    visualization_msgs::Marker marker;
    marker.type = visualization_msgs::Marker::LINE_LIST;
    marker.action = visualization_msgs::Marker::ADD;
    marker.scale.x = 0.02;
    tensegrity::utils::init_header(marker.header, "world");

    for (int i = 0; i < bars.size(); ++i)
    {
      const Color& color{ colors[i] };
      for (int j = 0; j < bars[i].size(); ++j)
      {
        const gtsam::Pose3& pose{ bars[i][j] };
        // const double& scale{ std::get<2>(bars[i]) };

        const Translation ptp{ pose * (Translation(offset)) };
        const Translation ptm{ pose * Translation(-offset) };
        // const Translation ptp{ pose.translation() };
        // const Translation ptm{ pose.translation() };

        // DEBUG_VARS(ptp.transpose(), ptm.transpose())
        marker.points.emplace_back();
        interface::copy(marker.points.back(), ptp);
        marker.points.emplace_back();
        interface::copy(marker.points.back(), ptm);

        marker.colors.emplace_back();
        marker.colors.back().b = color[0];
        marker.colors.back().g = color[1];
        marker.colors.back().r = color[2];
        marker.colors.back().a = 1.0;
        marker.colors.emplace_back();
        marker.colors.back().b = color[0];
        marker.colors.back().g = color[1];
        marker.colors.back().r = color[2];
        marker.colors.back().a = 1.0;
      }
    }
    pub_3Dbars.publish(marker);
  }

  void bars2d_to_3d(std::vector<bar2D_t>& bars, CvMatPtr& all_masks)
  {
    using Bar2D3DFactor = perception::bars2D_to_3d_factor_t;
    using Pose3DepthFactor = perception::pose3_depth_factor_t;
    const gtsam::Pose3 camera_pose{ camera->pose() };
    const double initial_depth{ camera_pose.z() };

    gtsam::Values values;
    gtsam::NonlinearFactorGraph graph;
    gtsam::noiseModel::Base::shared_ptr depth_nm{ gtsam::noiseModel::Isotropic::Sigma(2, 1e0) };
    gtsam::noiseModel::Base::shared_ptr depth3_nm{ gtsam::noiseModel::Isotropic::Sigma(3, 1e-2) };
    const double step{ 1.0 / max_pts };

    int tot_bars{ 0 };
    for (auto& bar : bars)
    {
      // gtsam::Key key_se2{ gtsam::Symbol('x', tot_bars) };
      gtsam::Key key_se3{ gtsam::Symbol('y', tot_bars) };
      const Pixel pt2D{ bar.pose.t() };
      const double scale{ bar.scale };
      const Translation pt3dp{ camera->backproject(pt2D, initial_depth) };

      const gtsam::Rot3 init_rot{ gtsam::Rot3(0.707, 0, 0.707, 0) * gtsam::Rot3::RzRyRx(bar.pose.theta(), 0, 0) };
      // const gtsam::Rot3 init_rot{ gtsam::Rot3(0.707, 0.707, 0, 0) * gtsam::Rot3::RzRyRx(0, bar.pose.theta(), 0) };
      // const gtsam::Rot3 init_rot{ gtsam::Rot3::RzRyRx(0, 1.57, bar.pose.theta()) * gtsam::Rot3(0.707, 0.707, 0, 0) };
      // const gtsam::Pose3 init_pose3(gtsam::Rot3::RzRyRx(0, 0, 0), pt3dp);
      // const gtsam::Pose3 init_pose3(gtsam::Rot3::RzRyRx(0, 0, bar.pose.theta()) * gtsam::Rot3(0.707, -0.707, 0, 0),
      const gtsam::Pose3 init_pose3(init_rot, pt3dp);
      // DEBUG_VARS(pt2D.transpose(), pt3dp.transpose());
      // init_pose3.print("init_pose3");
      // poses.push_back({ init_pose3, colors[bar.color_id], bar.scale });

      values.insert(key_se3, init_pose3);
      // compute_error(init_pose3, bar.pose,  scale, px_offset, const Translation& offset3D,
      //                        camera);
      graph.emplace_shared<Bar2D3DFactor>(key_se3, bar.pose, px_offset, offset, camera, scale, depth_nm);
      graph.emplace_shared<Bar2D3DFactor>(key_se3, bar.pose, -px_offset, -offset, camera, scale, depth_nm);
      // values.insert_or_assign(key_se3, init_pose3);
      gtsam::NonlinearFactorGraph graph_black;
      const Pixel scaled_offset{ px_offset * scale };
      for (int i = -max_pts / 2; i < max_pts / 2; ++i)
      {
        //
        const double curr_step{ i * step };
        const Pixel offset2D(scaled_offset * curr_step);
        // const Pixel tf_2Doff{ bar.pose.transformFrom(offset2D) };

        const Translation offset3D{ offset * curr_step };
        const Pixel px{ bar.pose.transformFrom(offset2D) };
        // const Pixel pxm{ bar.pose.transformFrom(Pixel(-scaled_offset * curr_step)) };

        const double z_meas{ img_depth.at<uint16_t>(px[1], px[0]) / depth_scale };
        // if (all_masks->at<uint8_t>(px[1], px[0]) == 255 and z_meas > 0)
        if ((_frame_black.at<uint8_t>(px[1], px[0]) == 255 or
             _frame_colors[bar.color_id].at<uint8_t>(px[1], px[0]) == 255) and
            z_meas > 0)
        {
          // int id = bar.color_id;
          // key_assigned = true;
          // DEBUG_VARS(id, px.transpose(), offset3D.transpose(), z_meas);
          graph.emplace_shared<Pose3DepthFactor>(key_se3, px, offset3D, z_meas, camera, depth3_nm);
          // }
          // {
          // graph.emplace_shared<Pose3DepthFactor>(key_se3, px, offset3D, z_meas, camera, depth3_nm);
        }
      }
      tot_bars++;
    }

    const gtsam::Values result{ lm_helper->optimize(graph, values, true) };
    // graph.printErrors(result, "Errors ");
    // std::unordered_map<gtsam::Key, double> errors;
    // double total_errors{ 0.0 };
    // for (auto& factor : graph)
    // {
    //   const gtsam::Key key_se3{ factor->keys()[0] };
    //   const double error{ factor->error(result) };
    //   if (errors.count(key_se3) == 0)
    //     errors[key_se3] = error;
    //   else
    //     errors[key_se3] += error;
    //   total_errors += error;
    // }
    // total_errors = total_errors / errors.size();

    std::array<std::vector<gtsam::Pose3>, 3> poses;
    std::array<std::vector<Eigen::MatrixXd>, 3> poses_cov;
    // DEBUG_PRINT
    // const gtsam::Marginals marginals(graph, result);

    for (int j = 0; j < tot_bars; ++j)
    {
      gtsam::Key key_se3{ gtsam::Symbol('y', j) };
      const gtsam::Pose3 pose3{ result.at<gtsam::Pose3>(key_se3) };
      // DEBUG_VARS(j, errors[key_se3], total_errors);
      // if (errors[key_se3] < total_errors)
      // {
      if (above_ground_bar(pose3))
      {
        // const Eigen::MatrixXd res_cov{ marginals.marginalCovariance(key_se3) };
        const Eigen::MatrixXd res_cov{ Eigen::MatrixXd::Identity(6, 6) };
        poses[bars[j].color_id].push_back(pose3);
        poses_cov[bars[j].color_id].push_back(res_cov);
      }
      // }
    }
    std::array<std::vector<gtsam::Pose3>, 3> poses_clust;
    std::array<std::vector<Eigen::MatrixXd>, 3> poses_cov_clust;
    cluster(poses_clust[0], poses_cov_clust[0], poses[0], poses_cov[0]);
    bars_to_marker(poses);

    // update_marker(visualization_msgs::Marker & marker, const std::vector<PointColor>& pc,
    //               const gtsam::Pose3 pose = gtsam::Pose3(), bool clear = true)
    // bars2D_to_3d_factor_t(const gtsam::Key key_se2, const gtsam::Key key_se3, const Pixel offset2D,
    // const Translation offset3D, const CameraPtr camera, const NoiseModel& cost_model)
  }

  void run_icp()
  {
    bars_poses_received = false;
    rgb_received = false;
    depth_received = false;

    cv::inRange(img_hsv, low_red, high_red, _frame_colors[red_idx]);
    cv::inRange(img_hsv, low_green, high_green, _frame_colors[green_idx]);
    cv::inRange(img_hsv, low_blue, high_blue, _frame_colors[blue_idx]);
    cv::inRange(img_hsv, low_black, high_black, _frame_black);

    // perception::erode_or_dilate(0, _frame_red0, _frame_red0, "ED", elements);
    // perception::erode_or_dilate(0, _frame_green, _frame_green, "ED", elements);
    // perception::erode_or_dilate(0, _frame_blue, _frame_blue, "ED", elements);

    cv::Mat color_masks{ _frame_colors[red_idx] + _frame_colors[green_idx] + _frame_colors[blue_idx] };
    CvMatPtr all_masks = std::make_shared<cv::Mat>(_frame_black + color_masks);
    perception::erode_or_dilate(0, *all_masks, *all_masks, "ED", elements);

    if (visualize)
    {
      publish_img(color_masks, pub0_rgb_masks, "mono8");
      // publish_img(color_masks, pub0_rgb_masks, "mono8");
      publish_img(*all_masks, pub0_all_masks, "mono8");
    }

    // visualization_msgs::Marker img_marker;
    // visualization_msgs::Marker red_marker;
    // visualization_msgs::Marker green_marker;
    // visualization_msgs::Marker blue_marker;
    // visualization_msgs::Marker red_matches_marker, green_matches_marker, blue_matches_marker;

    // update_marker(red_marker, red_bar_sim_pts, red_pose);
    // update_marker(green_marker, green_bar_sim_pts, green_pose);
    // update_marker(blue_marker, blue_bar_sim_pts, blue_pose);

    // update_marker(img_marker, red_target_2dpc, gtsam::Pose3(), true);
    // update_marker(img_marker, blue_target_2dpc, gtsam::Pose3(), false);
    // update_marker(img_marker, green_target_2dpc, gtsam::Pose3(), false);
    // update_marker(img_marker, black_target_2dpc, gtsam::Pose3(), false);

    // img_rgb.copyTo(img_bars_estimation);

    // compute_line(img_rgb, _frame_red0, img_lines, all_masks, red_target_2dpc, black_target_2dpc, red);
    // compute_line(img_rgb, _frame_green, img_lines, all_masks, blue_target_2dpc, black_target_2dpc, green);
    // compute_lines(img_rgb, _frame_blue, img_lines, all_masks, green_target_2dpc, black_target_2dpc, blue);
    std::vector<bar2D_t> estimated_bars;
    compute_lines(all_masks);
    compute_endcaps(estimated_bars, all_masks);

    bars2d_to_3d(estimated_bars, all_masks);

    if (visualize)
    {
      publish_img(_frame_colors[red_idx], pub_red_masks, "mono8");
      publish_img(_frame_colors[green_idx], pub_green_masks, "mono8");
      publish_img(_frame_colors[blue_idx], pub_blue_masks, "mono8");
    }
    // gtsam::Values values;
    // gtsam::NonlinearFactorGraph graph;
    // const gtsam::Values result{ lm_helper->optimize(graph, values, true) };
    // const gtsam::Key ka{ gtsam::Symbol('a', 0) };
    // const gtsam::Key kb{ gtsam::Symbol('b', 0) };
    // const double a{ result.at<double>(ka) };
    // const double b{ result.at<double>(kb) };
    // cv::Point p0{ 0, static_cast<int>(a) };
    // cv::Point p1{ 1000, static_cast<int>(a + b * 1000) };

    // cv::Mat img_lines;
    // // cv::copyTo(img_rgb, img_lines);
    // img_rgb.copyTo(img_lines);
    // cv::line(img_lines, p0, p1, cv::Scalar(0.0, 0.0, 255.0), 5);
    // publish_img(img_lines, pub_lines, "bgr8");

    // pub_red_marker.publish(red_marker);
    // pub_img_marker.publish(img_marker);

    // double error{ 200000 };
    // double error_change{ 1000 };
    // double dummy;
    // int iterations{ 0 };

    // const gtsam::Pose3 red_green_btw{ gtsam::traits<gtsam::Pose3>::Between(red_pose, green_pose) };
    // const gtsam::Pose3 red_blue_btw{ gtsam::traits<gtsam::Pose3>::Between(red_pose, blue_pose) };
    // const gtsam::Pose3 green_blue_btw{ gtsam::traits<gtsam::Pose3>::Between(green_pose, blue_pose) };
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "img_diff_test");
  ros::NodeHandle nh("~");

  tensegrity_3d_icp_t node(nh);
  while (ros::ok())
  {
    if (node.rgb_received and node.depth_received)
    {
      // PRINT_MSG("Running icp");
      auto start = ros::Time::now();
      node.run_icp();
      auto end = ros::Time::now();
      auto iter_dt = (end - start).toSec();
      DEBUG_VARS(iter_dt)
      node.rgb_received = false;
      node.depth_received = false;
    }
    ros::spinOnce();
  }
}
