#pragma once

namespace estimation
{
enum RodColors
{
  RED = 0,
  GREEN,
  BLUE
};

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
gtsam::Key rod_symbol(const RodColors rod, const int t)
{
  return SF::create_hashed_symbol("X^{", color_str(rod), "}_{", t, "}");
}

gtsam::Key rotation_symbol(const RodColors rod, const int t)
{
  return SF::create_hashed_symbol("R^{", color_str(rod), "}_{", t, "}");
}

gtsam::Key rodvel_symbol(const int rod, const double t)
{
  return SF::create_hashed_symbol("\\dot{X}^{", rod, "}_{", t, "}");
}

void update_pose(SE3& pose, const RodColors color, const int idx, const Values& values)
{
  const gtsam::Key key_se3{ rod_symbol(color, idx) };
  const gtsam::Key key_rot_offset{ rotation_symbol(color, idx) };

  // only update pose if it has been estimated
  if (values.exists(key_se3))
  {
    const SE3 xt{ values.at<SE3>(key_se3) };
    const Rotation Rt{ values.at<Rotation>(key_rot_offset) };
    const Translation pt_zero{ Translation::Zero() };
    const SE3 xr{ SE3(Rt, pt_zero) };
    pose = xt * xr;
  }
}

void publish_tensegrity_msg(const SE3& red_pose, const SE3& green_pose, const SE3& blue_pose, ros::Publisher& publisher,
                            const std::string frame)
{
  interface::TensegrityBars bars;
  tensegrity::utils::update_header(bars.header);
  bars.header.frame_id = frame;
  // const SE3 red_pose{ get_pose_from_poly(ti, RodColors::RED, polys, values) };
  interface::copy(bars.bar_red, red_pose);
  interface::copy(bars.bar_green, green_pose);
  interface::copy(bars.bar_blue, blue_pose);

  publisher.publish(bars);
}
}  // namespace estimation