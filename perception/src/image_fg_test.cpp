#include <iostream>
#include <array>
#include <numeric>
#include <algorithm>
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
#include <factor_graphs/defs.hpp>

#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/features2d.hpp>

template <typename State>
struct metric_t
{
  double operator()(const State& a, const State& b) const
  {
    return (a - b).norm();
  }
};

class point_transform_factor_t : public gtsam::NoiseModelFactor1<gtsam::Pose2>
{
public:
  using SE2 = gtsam::Pose2;
  using Base = gtsam::NoiseModelFactor1<SE2>;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;
  using Translation = Eigen::Vector2d;

  point_transform_factor_t(const gtsam::Key ktf, const Translation p_i, const Translation p_j,
                           const NoiseModel& cost_model = nullptr)
    : Base(cost_model, ktf), _pi(p_i), _pj(p_j)
  {
  }

  static Translation predict(const SE2& tf, const Translation& p_i, const Translation& p_j,  // no-lint
                             gtsam::OptionalJacobian<2, 3> Htf = boost::none)
  {
    Eigen::Matrix<double, 2, 3> pjp_H_tf;
    // Eigen::Matrix<double, 3, 3> pjp_H_pj;
    const Translation pjp{ tf.transformFrom(p_j, pjp_H_tf) };

    const Translation diff{ p_i - pjp };
    // Eigen::Matrix<double, 3, 3> diff_H_pi{ Eigen::Matrix3d::Identity() };

    // DEBUG_VARS(p_i.transpose(), pjp.transpose());
    // DEBUG_VARS(diff.transpose());

    if (Htf)
    {
      const Eigen::Matrix<double, 2, 2> diff_H_pjp{ -Eigen::Matrix2d::Identity() };
      *Htf = diff_H_pjp * pjp_H_tf;
    }

    return diff;
  }

  virtual Eigen::VectorXd evaluateError(const SE2& tf,  // no-lint
                                        boost::optional<Eigen::MatrixXd&> Htf = boost::none) const override
  {
    const Translation error{ predict(tf, _pi, _pj, Htf) };
    return error;
  }

private:
  const Translation _pi, _pj;
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
    // KeyPoint (Point2f pt, float size, float angle=-1, float response=0, int octave=0, int class_id=-1)
  }
}

int main(int argc, char** argv)
{
  using State = Eigen::Vector2d;
  using Gnn = utils::graph_nearest_neighbors_t<State, metric_t<State>>;
  using NodePtr = Gnn::NodePtr;

  ros::init(argc, argv, "img_diff_test");
  ros::NodeHandle nh("~");

  std::string img0_path;
  std::string img1_path;

  cv::Scalar low_red(160, 153, 57);
  cv::Scalar high_red(179, 255, 150);
  cv::Scalar low_black(0, 0, 0);
  cv::Scalar high_black(179, 255, 60);

  PARAM_SETUP(nh, img0_path);
  PARAM_SETUP(nh, img1_path);

  ros::Publisher pub0{ nh.advertise<sensor_msgs::Image>("/img0", 1, true) };
  ros::Publisher pub0_keypoints{ nh.advertise<sensor_msgs::Image>("/img0/keypoints", 1, true) };
  ros::Publisher pub1{ nh.advertise<sensor_msgs::Image>("/img1", 1, true) };
  ros::Publisher pub1_keypoints{ nh.advertise<sensor_msgs::Image>("/img1/keypoints", 1, true) };
  ros::Publisher pub1_icp{ nh.advertise<sensor_msgs::Image>("/img1/icp", 1, true) };
  ros::Publisher pub_matches{ nh.advertise<sensor_msgs::Image>("/img1/matches", 1, true) };

  cv::Mat img0{ cv::imread(img0_path) };
  cv::Mat img1{ cv::imread(img1_path) };

  cv::Mat img0_hsv, img1_hsv;
  cv::cvtColor(img0, img0_hsv, cv::COLOR_BGR2HSV);
  cv::cvtColor(img1, img1_hsv, cv::COLOR_BGR2HSV);

  cv::Mat _frame_red0, _frame_red1;
  cv::inRange(img0_hsv, low_red, high_red, _frame_red0);
  cv::inRange(img1_hsv, low_red, high_red, _frame_red1);

  std::vector<cv::Point> locations_0, locations_1;  // output, locations of non-zero pixels
  std::vector<cv::KeyPoint> keypoints_0, keypoints_1;
  cv::findNonZero(_frame_red0, locations_0);
  cv::findNonZero(_frame_red1, locations_1);

  locations_0 = { locations_0[0],   locations_0[50],  locations_0[100], locations_0[150],
                  locations_0[200], locations_0[250], locations_0[300] };
  locations_1 = { locations_1[0],   locations_1[50],  locations_1[100], locations_1[150],
                  locations_1[200], locations_1[250], locations_1[300] };
  locations_to_keypoints(locations_0, keypoints_0);
  locations_to_keypoints(locations_1, keypoints_1);

  // access pixel coordinates
  // Point pnt = locations[i];
  // auto orb = cv::ORB::create(100);

  PRINT_MSG("keypoints computed");
  publish_img(img0, pub0, "bgr8");
  publish_img(img1, pub1, "bgr8");
  publish_img(_frame_red0, pub0_keypoints, "mono8");
  publish_img(_frame_red1, pub1_keypoints, "mono8");

  DEBUG_VARS(locations_0.size(), locations_1.size());
  gtsam::Pose2 tf;
  int max_nn = std::min(locations_0.size(), locations_1.size());
  // int max_nn = 10;

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
    const gtsam::Key ktf{ gtsam::Symbol('x', 0) };
    values.insert(ktf, tf);

    // for (auto pt : locations_1)
    for (int i = 0; i < max_nn; ++i)
    {
      auto pt = locations_1[i];
      const State xi{ pt.x, pt.y };
      const State xtf{ tf.transformFrom(xi) };

      DEBUG_VARS(xi.transpose(), xtf.transpose());
      gnn.emplace(xtf, i);
      locations_1[i].x = xtf[0];
      locations_1[i].y = xtf[1];
    }

    NodePtr node;
    double distance;
    std::vector<cv::DMatch> matches;
    for (int i = 0; i < max_nn; ++i)
    {
      auto cv_pt = locations_0[i];
      const State p_i{ cv_pt.x, cv_pt.y };
      gnn.single_query(p_i, distance, node);
      const State p_jp{ gnn.single_query(p_i) };
      // auto cv_pt1 = locations_1[node->get_index()];
      // const State pj{ cv_pt1.x, cv_pt1.y };
      matches.emplace_back(i, node->value_id(), distance);

      DEBUG_VARS(p_i.transpose(), p_jp.transpose());
      graph.emplace_shared<point_transform_factor_t>(ktf, p_i, p_jp);
      gnn.remove_node(node);
    }

    locations_to_keypoints(locations_1, keypoints_1);

    cv::Mat img_matches;
    cv::drawMatches(img0, keypoints_0, img1, keypoints_1, matches, img_matches);

    const gtsam::Values result{ lm_helper.optimize(graph, values, true) };
    tf = result.at<gtsam::Pose2>(ktf);
    error = graph.error(result);

    PRINT_MSG("drawing matches");

    PRINT_MSG("Done drawing matches");
    img1 = transform_image(tf, pub1_icp, img1, img0);

    publish_img(img_matches, pub_matches, "bgr8");
    ros::spinOnce();
    // PRINT_MSG("cin");
    std::cin >> dummy;
    // PRINT_MSG("cin");
  }
  PRINT_MSG("Ended");
  ros::spin();
}
