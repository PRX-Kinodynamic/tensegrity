#pragma once
#include <factor_graphs/defs.hpp>
#include <interface/defs.hpp>
#include <tensegrity_utils/csv_reader.hpp>
#include <estimation/tensegrity_cap_obs_factor.hpp>
#include <estimation/bar_distance_factor.hpp>

namespace estimation
{
enum RodColors
{
  RED = 0,
  GREEN,
  BLUE
};
using ColorMapping = std::vector<std::pair<RodColors, RodColors>>;
using ColorMappingNum = std::array<std::pair<int, int>, 9>;

std::string color_str(const RodColors rod)
{
  std::string color;
  switch (rod)
  {
    case RED:
      color = "RED";
      break;
    case GREEN:
      color = "GREEN";
      break;
    case BLUE:
      color = "BLUE";
      break;
    default:
      color = "NA";
  }
  return color;
}

RodColors color_from_int(const int id)
{
  RodColors color;
  switch (id)
  {
    case 0:
      color = RED;
      break;
    case 1:
      color = GREEN;
      break;
    case 2:
      color = BLUE;
      break;
    default:
      TENSEGRITY_THROW("Wrong number for initializing RodColors");
  }
  return color;
}

RodColors color_str(std::string color)
{
  std::transform(color.begin(), color.end(), color.begin(), [](unsigned char c) { return std::tolower(c); });
  if (color == "red")
  {
    return RodColors::RED;
  }
  if (color == "green")
  {
    return RodColors::GREEN;
  }
  if (color == "blue")
  {
    return RodColors::BLUE;
  }
  TENSEGRITY_THROW("Wrong color " << color);
  return RodColors::BLUE;
}

ColorMapping create_cable_map(const std::string filename)
{
  tensegrity::utils::csv_reader_t reader(filename);
  ColorMapping map;
  for (int i = 0; i < 9; ++i)
  {
    TENSEGRITY_ASSERT(reader.has_next_line(), "Not enough cable map data!");
    auto line = reader.next_line();
    const RodColors color_a{ estimation::color_str(line[2]) };
    const RodColors color_b{ estimation::color_str(line[3]) };

    map.push_back(std::make_pair(color_a, color_b));
  }
  return map;
}

ColorMappingNum create_cable_map_fix_endcaps(const std::string filename)
{
  tensegrity::utils::csv_reader_t reader(filename);
  ColorMappingNum map;
  for (int i = 0; i < 9; ++i)
  {
    TENSEGRITY_ASSERT(reader.has_next_line(), "Not enough cable map data!");
    auto line = reader.next_line();
    const int color_a{ tensegrity::utils::convert_to<int>(line[0]) };
    const int color_b{ tensegrity::utils::convert_to<int>(line[1]) };

    map[i] = { color_a, color_b };
    // map.push_back({ color_a, color_b });
  }
  return map;
}

std::map<RodColors, std::vector<Eigen::Vector3d>> read_initial_estimates(const std::string filename)
{
  tensegrity::utils::csv_reader_t reader(filename);
  std::map<RodColors, std::vector<Eigen::Vector3d>> estimates;
  for (int i = 0; i < 6; ++i)
  {
    TENSEGRITY_ASSERT(reader.has_next_line(), "Not enough initial estimates data!");
    auto line = reader.next_line();
    if (line[0] == "#")  // Skip comment line
    {
      i--;
      continue;
    }

    const RodColors color{ estimation::color_str(line[1]) };

    const double x{ tensegrity::utils::convert_to<double>(line[5]) };
    const double y{ tensegrity::utils::convert_to<double>(line[6]) };
    const double z{ tensegrity::utils::convert_to<double>(line[7]) };

    const Eigen::Vector3d pt({ x, y, z });
    estimates[color].push_back(pt);
  }
  return estimates;
}

gtsam::Key rod_symbol(const RodColors rod, const int t, const int i = 0)
{
  return factor_graphs::symbol_factory_t::create_hashed_symbol("X^{", color_str(rod), "}_{", t, ",", i, "}");
}
gtsam::Key rod_symbol(const int rod, const int t, const int i = 0)
{
  return rod_symbol(color_from_int(rod), t, i);
}

gtsam::Key rotation_symbol(const RodColors rod, const int t, const int i = 0)
{
  return factor_graphs::symbol_factory_t::create_hashed_symbol("R^{", color_str(rod), "}_{", t, ",", i, "}");
}
gtsam::Key rotation_symbol(const int rod, const int t, const int i = 0)
{
  return rotation_symbol(color_from_int(rod), t, i);
}

gtsam::Key rodvel_symbol(const RodColors rod, const int t)
{
  return factor_graphs::symbol_factory_t::create_hashed_symbol("\\dot{X}^{", color_str(rod), "}_{", t, "}");
}
gtsam::Key rodvel_symbol(const int rod, const int t)
{
  return rodvel_symbol(color_from_int(rod), t);
}

gtsam::Key endcap_symbol(const RodColors rod, const int t, const int i = 0)
{
  return factor_graphs::symbol_factory_t::create_hashed_symbol("Endcap^{", color_str(rod), "}_{", t, ",", i, "}");
}
gtsam::Key endcap_symbol(const int rod, const int t, const int i)
{
  return endcap_symbol(color_from_int(rod), t, i);
}

// Add to values only if it is not there yet
template <typename ValueType>
void add_to_values(gtsam::Values& values, const gtsam::Key& key, const ValueType v)
{
  if (not values.exists(key))
  {
    values.insert(key, v);
  }
}

void update_pose(gtsam::Pose3& pose, const RodColors color, const int idx, const gtsam::Values& values)
{
  using Translation = Eigen::Vector3d;
  // const gtsam::Key key_se3{ rod_symbol(color, idx) };
  // const gtsam::Key key_rot_offset{ rotation_symbol(color, idx) };

  // // only update pose if it has been estimated
  // if (values.exists(key_se3) and values.exists(key_rot_offset))
  // {
  //   const gtsam::Pose3 xt{ values.at<gtsam::Pose3>(key_se3) };
  //   const gtsam::Rot3 Rt{ values.at<gtsam::Rot3>(key_rot_offset) };
  //   const Translation pt_zero{ Translation::Zero() };
  //   const gtsam::Pose3 xr{ gtsam::Pose3(Rt, pt_zero) };
  //   pose = xt * xr;
  // }
  const gtsam::Key key_se3{ estimation::rod_symbol(color, idx) };
  const gtsam::Key key_rotA_offset{ estimation::rotation_symbol(color, idx, 0) };

  // only update pose if it has been estimated
  if (values.exists(key_se3))
  {
    pose = values.at<gtsam::Pose3>(key_se3);
    // gtsam::Rot3 R{ values.at<gtsam::Rot3>(key_rotA_offset) };
    // DEBUG_VARS(Rt);
    // DEBUG_VARS(R);
    // DEBUG_VARS(pose);
  }
}

void publish_tensegrity_msg(const gtsam::Pose3& red_pose, const gtsam::Pose3& green_pose, const gtsam::Pose3& blue_pose,
                            ros::Publisher& publisher, const std::string frame, int idx = 0)
{
  interface::TensegrityBars bars;
  tensegrity::utils::update_header(bars.header);
  bars.header.seq = idx;
  bars.header.frame_id = frame;
  // const SE3 red_pose{ get_pose_from_poly(ti, RodColors::RED, polys, values) };
  interface::copy(bars.bar_red, red_pose);
  interface::copy(bars.bar_green, green_pose);
  interface::copy(bars.bar_blue, blue_pose);

  publisher.publish(bars);
}

struct observation_update_t
{
  using SE3 = gtsam::Pose3;
  using OptSE3 = std::optional<SE3>;

  using Translation = Eigen::Vector3d;
  using OptTranslation = std::optional<Translation>;

  int idx = 0;
  int j = 0;
  RodColors color;
  OptTranslation zA;
  OptTranslation zB;
  Translation offset;
  gtsam::Rot3 Roffset;

  OptSE3 pose = {};
  gtsam::Key key_se3;
};

void add_two_observations(observation_update_t& observation_update, gtsam::NonlinearFactorGraph& graph,
                          gtsam::Values& values)
{
  using BarEstimationFactor = estimation::bar_two_observations_factor_t;

  using EndcapObservationFactor = estimation::endcap_observation_factor_t;
  using RotationOffsetFactor = estimation::endcap_rotation_offset_factor_t;
  using RotationFixIdentity = estimation::rotation_fix_identity_t;
  // using BarRotationFactor = rotation_symmetric_factor_t;
  gtsam::noiseModel::Base::shared_ptr z_noise{ gtsam::noiseModel::Isotropic::Sigma(3, 1e-1) };
  gtsam::noiseModel::Base::shared_ptr rot_prior_nm{ gtsam::noiseModel::Isotropic::Sigma(3, 1e-4) };

  const int& idx{ observation_update.idx };
  const int& j{ observation_update.j };
  const RodColors& color{ observation_update.color };
  const Eigen::Vector3d zA{ *(observation_update.zA) };
  const Eigen::Vector3d zB{ *(observation_update.zB) };
  const Eigen::Vector3d& offset{ observation_update.offset };
  const gtsam::Rot3& Roffset{ observation_update.Roffset };

  const gtsam::Key key_se3{ estimation::rod_symbol(color, idx, j) };
  const gtsam::Key key_rotA_offset{ estimation::rotation_symbol(color, idx, 0) };
  const gtsam::Key key_rotB_offset{ estimation::rotation_symbol(color, idx, 1) };
  observation_update.key_se3 = key_se3;

  graph.emplace_shared<EndcapObservationFactor>(key_se3, key_rotA_offset, zA, offset, z_noise);
  graph.emplace_shared<EndcapObservationFactor>(key_se3, key_rotB_offset, zB, offset, z_noise);

  graph.emplace_shared<RotationOffsetFactor>(key_rotA_offset, key_rotB_offset, Roffset, rot_prior_nm);
  graph.emplace_shared<RotationOffsetFactor>(key_rotB_offset, key_rotA_offset, Roffset, rot_prior_nm);

  graph.emplace_shared<RotationFixIdentity>(key_rotA_offset, key_rotB_offset, Roffset, rot_prior_nm);

  if (observation_update.pose)
  {
    estimation::add_to_values(values, key_se3, *(observation_update.pose));
    estimation::add_to_values(values, key_rotA_offset, gtsam::Rot3());
    estimation::add_to_values(values, key_rotB_offset, Roffset.inverse());
  }
}

void add_one_observation(observation_update_t& observation_update, gtsam::NonlinearFactorGraph& graph,
                         gtsam::Values& values)
{
  // using BarEstimationFactor = estimation::bar_two_observations_factor_t;
  using EndcapObservationFactor = estimation::endcap_observation_factor_t;
  using RotationOffsetFactor = estimation::endcap_rotation_offset_factor_t;
  using RotationFixIdentity = estimation::rotation_fix_identity_t;

  const int& idx{ observation_update.idx };
  const RodColors& color{ observation_update.color };
  const Eigen::Vector3d z{ *(observation_update.zA) };  // Only one observation, assume it is in zA
  const Eigen::Vector3d& offset{ observation_update.offset };
  const gtsam::Rot3& Roffset{ observation_update.Roffset };

  const gtsam::Key key_se3{ rod_symbol(color, idx) };
  const gtsam::Key key_rotA_offset{ estimation::rotation_symbol(color, idx, 0) };
  const gtsam::Key key_rotB_offset{ estimation::rotation_symbol(color, idx, 1) };
  gtsam::noiseModel::Base::shared_ptr z_noise{ gtsam::noiseModel::Isotropic::Sigma(3, 1e-1) };
  gtsam::noiseModel::Base::shared_ptr rot_prior_nm{ gtsam::noiseModel::Isotropic::Sigma(3, 1e-4) };

  graph.emplace_shared<EndcapObservationFactor>(key_se3, key_rotA_offset, z, offset, z_noise);

  graph.emplace_shared<RotationOffsetFactor>(key_rotA_offset, key_rotB_offset, Roffset, rot_prior_nm);
  graph.emplace_shared<RotationOffsetFactor>(key_rotB_offset, key_rotA_offset, Roffset, rot_prior_nm);

  graph.emplace_shared<RotationFixIdentity>(key_rotA_offset, key_rotB_offset, Roffset, rot_prior_nm);
  // graph.addPrior(key_rotB_offset, Rotation(), rot_prior_nm);
}

gtsam::Values compute_initialization(const std::string initial_estimate_filename, const Eigen::Vector3d& offset,
                                     const gtsam::Rot3& Roffset)
{
  std::map<estimation::RodColors, std::vector<Eigen::Vector3d>> estimates{ estimation::read_initial_estimates(
      initial_estimate_filename) };

  gtsam::NonlinearFactorGraph graph;
  gtsam::Values values;
  estimation::observation_update_t observation_update;
  observation_update.idx = 0;
  observation_update.offset = offset;
  observation_update.Roffset = Roffset;

  for (auto pair : estimates)
  {
    const estimation::RodColors color{ pair.first };
    // const RodObservation z0{ std::make_pair(pair.second[0], pair.second[1]) };
    observation_update.color = color;
    observation_update.zA = pair.second[0];
    observation_update.zB = pair.second[1];
    estimation::add_two_observations(observation_update, graph, values);

    estimation::add_to_values(values, rod_symbol(color, 0), factor_graphs::random<gtsam::Pose3>());
    estimation::add_to_values(values, rotation_symbol(color, 0, 0), gtsam::Rot3());
    estimation::add_to_values(values, rotation_symbol(color, 0, 1), Roffset.inverse());
  }
  gtsam::LevenbergMarquardtParams lm_params;
  std::shared_ptr<factor_graphs::levenberg_marquardt_t> lm_helper;
  lm_params.setVerbosityLM("SILENT");
  // lm_params.setVerbosityLM("SUMMARY");
  lm_params.setMaxIterations(20);
  gtsam::LevenbergMarquardtOptimizer optimizer(graph, values, lm_params);
  const gtsam::Values result{ optimizer.optimize() };
  return result;
}

void add_triangle_factors(observation_update_t& observation_update, gtsam::NonlinearFactorGraph& graph,
                          gtsam::Values& values)
{
  using Rotation = gtsam::Rot3;
  using TriangleAlignedFactor = tensegrity_triangles_aligned_factor_t;
  const int idx{ observation_update.idx };
  const Eigen::Vector3d& offset{ observation_update.offset };
  const Rotation& Roffset{ observation_update.Roffset };

  const gtsam::Key key_Xr{ rod_symbol(estimation::RodColors::RED, idx) };
  const gtsam::Key key_Xg{ rod_symbol(estimation::RodColors::GREEN, idx) };
  const gtsam::Key key_Xb{ rod_symbol(estimation::RodColors::BLUE, idx) };
  const gtsam::Key key_Rr{ rotation_symbol(estimation::RodColors::RED, idx, 0) };
  const gtsam::Key key_Rg{ rotation_symbol(estimation::RodColors::GREEN, idx, 0) };
  const gtsam::Key key_Rb{ rotation_symbol(estimation::RodColors::BLUE, idx, 0) };
  const gtsam::Key key_Rrp{ rotation_symbol(estimation::RodColors::RED, idx, 1) };
  const gtsam::Key key_Rgp{ rotation_symbol(estimation::RodColors::GREEN, idx, 1) };
  const gtsam::Key key_Rbp{ rotation_symbol(estimation::RodColors::BLUE, idx, 1) };

  gtsam::noiseModel::Base::shared_ptr parallel_triangles_nm{ gtsam::noiseModel::Isotropic::Sigma(3, 1e0) };

  graph.emplace_shared<TriangleAlignedFactor>(key_Xr, key_Rr, key_Xg, key_Rg, key_Xb, key_Rb, offset, Roffset,
                                              parallel_triangles_nm);
  graph.emplace_shared<TriangleAlignedFactor>(key_Xg, key_Rg, key_Xb, key_Rb, key_Xr, key_Rr, offset, Roffset,
                                              parallel_triangles_nm);
  graph.emplace_shared<TriangleAlignedFactor>(key_Xb, key_Rb, key_Xr, key_Rr, key_Xg, key_Rg, offset, Roffset,
                                              parallel_triangles_nm);
  //
  graph.emplace_shared<TriangleAlignedFactor>(key_Xb, key_Rbp, key_Xg, key_Rgp, key_Xr, key_Rrp, offset, Roffset,
                                              parallel_triangles_nm);
  graph.emplace_shared<TriangleAlignedFactor>(key_Xr, key_Rrp, key_Xb, key_Rbp, key_Xg, key_Rgp, offset, Roffset,
                                              parallel_triangles_nm);
  graph.emplace_shared<TriangleAlignedFactor>(key_Xg, key_Rgp, key_Xr, key_Rrp, key_Xb, key_Rbp, offset, Roffset,
                                              parallel_triangles_nm);

  add_to_values(values, key_Rr, gtsam::Rot3());
  add_to_values(values, key_Rg, gtsam::Rot3());
  add_to_values(values, key_Rb, gtsam::Rot3());
  add_to_values(values, key_Rrp, Roffset);
  add_to_values(values, key_Rgp, Roffset);
  add_to_values(values, key_Rbp, Roffset);
}

void add_cable_meassurements(observation_update_t& observation_update, const Eigen::Vector<double, 9>& zi,
                             const int idx, gtsam::NonlinearFactorGraph& graph, gtsam::Values& values,
                             estimation::ColorMapping& cable_map)
{
  using Rotation = gtsam::Rot3;
  // using CablesFactor = estimation::cable_length_observations_factor_t;
  using CablesFactor = estimation::cable_length_factor_t;
  using RotationOffsetFactor = estimation::endcap_rotation_offset_factor_t;
  using RotationFixIdentity = estimation::rotation_fix_identity_t;
  using ChiralityFactor = tensegrity_chirality_factor_t;
  using TriangleFactor = tensegrity_triangle_factor_t;
  using TriangleAlignedFactor = tensegrity_triangles_aligned_factor_t;
  using DistanceBarFactor = bar_distance_factor_t;

  const Eigen::Vector3d& offset{ observation_update.offset };
  const Rotation& Roffset{ observation_update.Roffset };

  gtsam::noiseModel::Base::shared_ptr z_noise{ gtsam::noiseModel::Isotropic::Sigma(1, 1.6e-2) };
  gtsam::noiseModel::Base::shared_ptr chirality_noise{ gtsam::noiseModel::Isotropic::Sigma(1, 1e-3) };
  gtsam::noiseModel::Base::shared_ptr rot_prior_nm{ gtsam::noiseModel::Isotropic::Sigma(3, 1e-4) };
  gtsam::noiseModel::Base::shared_ptr force_triangle_nm{ gtsam::noiseModel::Isotropic::Sigma(1, 1e-6) };
  gtsam::noiseModel::Base::shared_ptr coplanarity_nm{ gtsam::noiseModel::Isotropic::Sigma(1, 1e-1) };
  gtsam::noiseModel::Base::shared_ptr parallel_triangles_nm{ gtsam::noiseModel::Isotropic::Sigma(3, 1e0) };
  gtsam::noiseModel::Base::shared_ptr scale_equivalent_nm{ gtsam::noiseModel::Isotropic::Sigma(4, 1e-4) };
  gtsam::noiseModel::Base::shared_ptr distance_nm{ gtsam::noiseModel::Isotropic::Sigma(1, 1e-2) };
  // const interface::TensegrityLengthSensor sensors{ cables_callback.meassurements.back() };
  // const auto zi = sensors.length;
  // cables_callback.meassurements.clear();
  // const std::vector<double> zi{ cables_callback.get_last_observation() };
  // DEBUG_VARS(zi.transpose());
  int i = 0;
  for (; i < 3; ++i)
  {
    const gtsam::Key key_Xi{ rod_symbol(cable_map[i].first, idx) };
    const gtsam::Key key_Xj{ rod_symbol(cable_map[i].second, idx) };
    const gtsam::Key key_Ri{ rotation_symbol(cable_map[i].first, idx, 0) };
    const gtsam::Key key_Rj{ rotation_symbol(cable_map[i].second, idx, 0) };

    graph.emplace_shared<CablesFactor>(key_Xi, key_Xj, key_Ri, key_Rj, zi[i], offset, z_noise);
  }
  for (; i < 6; ++i)
  {
    const gtsam::Key key_Xi{ rod_symbol(cable_map[i].first, idx) };
    const gtsam::Key key_Xj{ rod_symbol(cable_map[i].second, idx) };
    const gtsam::Key key_Ri{ rotation_symbol(cable_map[i].first, idx, 1) };
    const gtsam::Key key_Rj{ rotation_symbol(cable_map[i].second, idx, 1) };

    graph.emplace_shared<CablesFactor>(key_Xi, key_Xj, key_Ri, key_Rj, zi[i], offset, z_noise);
  }
  for (; i < 9; ++i)
  {
    const gtsam::Key key_Xi{ rod_symbol(cable_map[i].first, idx) };
    const gtsam::Key key_Xj{ rod_symbol(cable_map[i].second, idx) };
    const gtsam::Key key_Ri{ rotation_symbol(cable_map[i].first, idx, 0) };
    const gtsam::Key key_Rj{ rotation_symbol(cable_map[i].second, idx, 1) };

    graph.emplace_shared<CablesFactor>(key_Xi, key_Xj, key_Ri, key_Rj, zi[i], offset, z_noise);
  }

  for (auto color : { RodColors::RED, RodColors::GREEN, RodColors::BLUE })
  {
    const gtsam::Key key_Ri{ rotation_symbol(color, idx, 0) };
    const gtsam::Key key_Rj{ rotation_symbol(color, idx, 1) };

    add_to_values(values, key_Ri, gtsam::Rot3());
    add_to_values(values, key_Rj, Roffset);
    graph.emplace_shared<RotationOffsetFactor>(key_Ri, key_Rj, Roffset, rot_prior_nm);
    graph.emplace_shared<RotationOffsetFactor>(key_Rj, key_Ri, Roffset, rot_prior_nm);

    graph.emplace_shared<RotationFixIdentity>(key_Ri, key_Rj, Roffset, rot_prior_nm);
  }

  // const gtsam::Key key_Xr{ rod_symbol(estimation::RodColors::RED, idx) };
  // const gtsam::Key key_Xg{ rod_symbol(estimation::RodColors::GREEN, idx) };
  // const gtsam::Key key_Xb{ rod_symbol(estimation::RodColors::BLUE, idx) };
  // const gtsam::Key key_Rr{ rotation_symbol(estimation::RodColors::RED, idx, 0) };
  // const gtsam::Key key_Rg{ rotation_symbol(estimation::RodColors::GREEN, idx, 0) };
  // const gtsam::Key key_Rb{ rotation_symbol(estimation::RodColors::BLUE, idx, 0) };
  // const gtsam::Key key_Rrp{ rotation_symbol(estimation::RodColors::RED, idx, 1) };
  // const gtsam::Key key_Rgp{ rotation_symbol(estimation::RodColors::GREEN, idx, 1) };
  // const gtsam::Key key_Rbp{ rotation_symbol(estimation::RodColors::BLUE, idx, 1) };

  // graph.emplace_shared<TriangleAlignedFactor>(key_Xr, key_Rr, key_Xg, key_Rg, key_Xb, key_Rb, offset, Roffset,
  //                                             parallel_triangles_nm);
  // graph.emplace_shared<TriangleAlignedFactor>(key_Xg, key_Rg, key_Xb, key_Rb, key_Xr, key_Rr, offset, Roffset,
  //                                             parallel_triangles_nm);
  // graph.emplace_shared<TriangleAlignedFactor>(key_Xb, key_Rb, key_Xr, key_Rr, key_Xg, key_Rg, offset, Roffset,
  //                                             parallel_triangles_nm);
  // //
  // graph.emplace_shared<TriangleAlignedFactor>(key_Xb, key_Rbp, key_Xg, key_Rgp, key_Xr, key_Rrp, offset, Roffset,
  //                                             parallel_triangles_nm);
  // graph.emplace_shared<TriangleAlignedFactor>(key_Xr, key_Rrp, key_Xb, key_Rbp, key_Xg, key_Rgp, offset, Roffset,
  //                                             parallel_triangles_nm);
  // graph.emplace_shared<TriangleAlignedFactor>(key_Xg, key_Rgp, key_Xr, key_Rrp, key_Xb, key_Rbp, offset, Roffset,
  //                                             parallel_triangles_nm);
  // graph.emplace_shared<DistanceBarFactor>(key_Xr, key_Xg, 0.03, distance_nm);
  // graph.emplace_shared<DistanceBarFactor>(key_Xg, key_Xb, 0.03, distance_nm);
  // graph.emplace_shared<DistanceBarFactor>(key_Xb, key_Xr, 0.03, distance_nm);
  // const TriangleFactor::TriangleFactorType force_triangle{ TriangleFactor::TriangleFactorType::ForceTriangle };
  // const TriangleFactor::TriangleFactorType coplanarity{ TriangleFactor::TriangleFactorType::Coplanarity };
  // const TriangleFactor::TriangleFactorType parallel_triangles{ TriangleFactor::TriangleFactorType::ParallelTriangles
  // }; graph.emplace_shared<TriangleFactor>(key_Xr, key_Xg, key_Xb, offset, Rotation(), parallel_triangles,
  //                                      parallel_triangles_nm);
  // graph.emplace_shared<TriangleFactor>(key_Xr, key_Xg, key_Xb, offset, Roffset, parallel_triangles,
  //                                      parallel_triangles_nm);

  // const TriangleFactor::TriangleFactorType scale_equivalent{ TriangleFactor::TriangleFactorType::ScaleEquivalent };
  // graph.emplace_shared<TriangleFactor>(key_Xr, key_Xg, key_Xb, offset, Roffset, scale_equivalent,
  // scale_equivalent_nm);

  // graph.emplace_shared<ChiralityFactor>(key_Xb, key_Xr, key_Xg, offset, Roffset, chirality_noise);
  // graph.emplace_shared<ChiralityFactor>(key_Xg, key_Xb, key_Xr, offset, Roffset, chirality_noise);
  // graph.emplace_shared<ChiralityFactor>(key_Xr, key_Xg, key_Xb, offset, Roffset, chirality_noise);
}

struct tensegrity_graph_inputs_t
{
  using SE3 = gtsam::Pose3;
  using OptSE3 = std::optional<SE3>;

  using Endcap = Eigen::Vector3d;
  using OptEndcap = std::optional<Endcap>;

  void reset()
  {
    endcaps = { std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt };
    init_poses = { std::nullopt, std::nullopt, std::nullopt };
    noise_models = { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
    cable_noise = nullptr;
  }

  int idx_major;
  bool add_cable_meassurements;
  // int idx_minor;

  Eigen::Vector3d offset;
  gtsam::Rot3 Roffset;

  std::array<OptEndcap, 6> endcaps;

  std::array<OptSE3, 3> init_poses;

  std::array<gtsam::noiseModel::Base::shared_ptr, 6> noise_models;

  // Cable vars
  Eigen::Vector<double, 9> cables;
  std::array<std::pair<int, int>, 9> cable_map;
  gtsam::noiseModel::Base::shared_ptr cable_noise;
};

struct tensegrity_graph_output_t
{
  std::array<gtsam::Key, 3> keys_pose;
  std::array<gtsam::Key, 3> keys_rotA;
  std::array<gtsam::Key, 3> keys_rotB;

  gtsam::Values values;
  gtsam::NonlinearFactorGraph graph;
};

tensegrity_graph_output_t create_tensegrity_graph(const tensegrity_graph_inputs_t& input)
{
  using CablesFactor = estimation::cable_length_factor_t;
  using EndcapObservationFactor = estimation::endcap_observation_factor_t;
  using RotationOffsetFactor = estimation::endcap_rotation_offset_factor_t;
  using RotationFixIdentity = estimation::rotation_fix_identity_t;

  tensegrity_graph_output_t output;

  gtsam::Values& values{ output.values };
  gtsam::NonlinearFactorGraph& graph{ output.graph };

  const Eigen::Vector3d& offset{ input.offset };
  const gtsam::Rot3& Roffset{ input.Roffset };

  const gtsam::noiseModel::Base::shared_ptr rot_prior_nm{ gtsam::noiseModel::Isotropic::Sigma(3, 1e-4) };
  for (int i = 0; i < 3; ++i)
  {
    output.keys_pose[i] = estimation::rod_symbol(i, input.idx_major);
    output.keys_rotA[i] = estimation::rotation_symbol(i, input.idx_major, 0);
    output.keys_rotB[i] = estimation::rotation_symbol(i, input.idx_major, 1);

    const gtsam::Pose3 init_pose{ input.init_poses[i] ? *(input.init_poses[i]) : gtsam::Pose3() };
    estimation::add_to_values(values, output.keys_pose[i], init_pose);
    estimation::add_to_values(values, output.keys_rotA[i], gtsam::Rot3());
    estimation::add_to_values(values, output.keys_rotB[i], Roffset.inverse());

    if (input.endcaps[2 * i])
    {
      const Eigen::Vector3d& ei{ *(input.endcaps[2 * i]) };
      const gtsam::noiseModel::Base::shared_ptr z_noise{ input.noise_models[2 * i] };
      graph.emplace_shared<EndcapObservationFactor>(output.keys_pose[i], output.keys_rotA[i], ei, offset, z_noise);
    }
    if (input.endcaps[2 * i + 1])
    {
      const Eigen::Vector3d& ei{ *(input.endcaps[2 * i + 1]) };
      const gtsam::noiseModel::Base::shared_ptr z_noise{ input.noise_models[2 * i + 1] };
      graph.emplace_shared<EndcapObservationFactor>(output.keys_pose[i], output.keys_rotB[i], ei, offset, z_noise);
    }

    graph.emplace_shared<RotationOffsetFactor>(output.keys_rotA[i], output.keys_rotB[i], Roffset, rot_prior_nm);
    graph.emplace_shared<RotationOffsetFactor>(output.keys_rotB[i], output.keys_rotA[i], Roffset, rot_prior_nm);

    graph.emplace_shared<RotationFixIdentity>(output.keys_rotA[i], output.keys_rotB[i], Roffset, rot_prior_nm);
  }

  if (input.add_cable_meassurements)
  {
    const Eigen::Vector<double, 9>& zi{ input.cables };

    const gtsam::noiseModel::Base::shared_ptr cable_noise{ input.cable_noise };
    int i = 0;
    for (; i < 3; ++i)
    {
      const gtsam::Key key_Xi{ rod_symbol(input.cable_map[i].first, input.idx_major) };
      const gtsam::Key key_Xj{ rod_symbol(input.cable_map[i].second, input.idx_major) };
      const gtsam::Key key_Ri{ rotation_symbol(input.cable_map[i].first, input.idx_major, 0) };
      const gtsam::Key key_Rj{ rotation_symbol(input.cable_map[i].second, input.idx_major, 0) };

      graph.emplace_shared<CablesFactor>(key_Xi, key_Xj, key_Ri, key_Rj, zi[i], offset, cable_noise);
    }
    for (; i < 6; ++i)
    {
      const gtsam::Key key_Xi{ rod_symbol(input.cable_map[i].first, input.idx_major) };
      const gtsam::Key key_Xj{ rod_symbol(input.cable_map[i].second, input.idx_major) };
      const gtsam::Key key_Ri{ rotation_symbol(input.cable_map[i].first, input.idx_major, 1) };
      const gtsam::Key key_Rj{ rotation_symbol(input.cable_map[i].second, input.idx_major, 1) };

      graph.emplace_shared<CablesFactor>(key_Xi, key_Xj, key_Ri, key_Rj, zi[i], offset, cable_noise);
    }
    for (; i < 9; ++i)
    {
      const gtsam::Key key_Xi{ rod_symbol(input.cable_map[i].first, input.idx_major) };
      const gtsam::Key key_Xj{ rod_symbol(input.cable_map[i].second, input.idx_major) };
      const gtsam::Key key_Ri{ rotation_symbol(input.cable_map[i].first, input.idx_major, 0) };
      const gtsam::Key key_Rj{ rotation_symbol(input.cable_map[i].second, input.idx_major, 1) };

      graph.emplace_shared<CablesFactor>(key_Xi, key_Xj, key_Ri, key_Rj, zi[i], offset, cable_noise);
    }

    // for (auto color : { RodColors::RED, RodColors::GREEN, RodColors::BLUE })
    // {
    //   const gtsam::Key key_Ri{ rotation_symbol(color, idx, 0) };
    //   const gtsam::Key key_Rj{ rotation_symbol(color, idx, 1) };

    //   add_to_values(values, key_Ri, gtsam::Rot3());
    //   add_to_values(values, key_Rj, Roffset);
    //   graph.emplace_shared<RotationOffsetFactor>(key_Ri, key_Rj, Roffset, rot_prior_nm);
    //   graph.emplace_shared<RotationOffsetFactor>(key_Rj, key_Ri, Roffset, rot_prior_nm);

    //   graph.emplace_shared<RotationFixIdentity>(key_Ri, key_Rj, Roffset, rot_prior_nm);
    // }
  }
  return output;
}

}  // namespace estimation