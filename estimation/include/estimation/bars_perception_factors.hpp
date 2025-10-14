#pragma once
#include <array>
#include <numeric>
#include <functional>

#include <gtsam/config.h>
#include <gtsam/base/Testable.h>
#include <gtsam/nonlinear/Expression.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/slam/BetweenFactor.h>

#include <factor_graphs/symbols_factory.hpp>
#include <factor_graphs/lie_integrator.hpp>
#include <tensegrity_utils/rosparams_utils.hpp>
#include <tensegrity_utils/dbg_utils.hpp>

namespace perception
{
class endcap_2Dobservation_factor_t : public gtsam::NoiseModelFactorN<gtsam::Pose2, double, gtsam::Rot2>
{
public:
  using SE2 = gtsam::Pose2;
  using Pixel = Eigen::Vector<double, 2>;
  using Rotation = gtsam::Rot2;
  using Base = gtsam::NoiseModelFactor1<SE2, double, gtsam::Rot2>;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;

  // using Vector = Eigen::Vector<double, Dim>;

  endcap_2Dobservation_factor_t(const gtsam::Key key_se2, const gtsam::Key key_scale, const gtsam::Key key_offset,
                                const Pixel zi, const Pixel offset, const NoiseModel& cost_model)
    : Base(cost_model, key_se2, key_scale, key_offset), _zi(zi), _offset(offset)
  {
  }

  static Pixel predict(const SE2& x, const double& scale, const Rotation& Rx, const Pixel& offset,
                       gtsam::OptionalJacobian<2, 3> Hx = boost::none,  // no-lint
                       gtsam::OptionalJacobian<2, 1> Hs = boost::none,  // no-lint
                       gtsam::OptionalJacobian<2, 1> HR = boost::none)
  {
    const bool compute_deriv{ Hx or HR };

    Eigen::Matrix<double, 2, 3> pred_H_xp;
    Eigen::Matrix<double, 3, 3> xp_H_x;
    Eigen::Matrix<double, 3, 3> xp_H_xrot;
    Eigen::Matrix<double, 3, 1> xrot_H_Rx;
    Eigen::Matrix<double, 2, 2> pred_H_sOff;

    xrot_H_Rx << 1, 0, 0;
    const Pixel zero{ Pixel::Zero() };
    const SE2 xrot{ SE2(Rx, zero) };

    const SE2 xp{ gtsam::traits<SE2>::Compose(x, xrot, compute_deriv ? &xp_H_x : nullptr,
                                              compute_deriv ? &xp_H_xrot : nullptr) };

    const Eigen::Matrix<double, 2, 1> sOff_H_s{ offset };
    const Pixel scaled_off{ offset * scale };
    const Pixel pred{ xp.transformFrom(scaled_off, compute_deriv ? &pred_H_xp : nullptr,
                                       compute_deriv ? &pred_H_sOff : nullptr) };
    if (Hx)
    {
      *Hx = pred_H_xp * xp_H_x;
    }
    if (HR)
    {
      *HR = pred_H_xp * xp_H_xrot * xrot_H_Rx;
    }
    if (Hs)
    {
      *Hs = pred_H_sOff * sOff_H_s;
    }
    return pred;
  }

  static Eigen::VectorXd compute_error(const SE2& x, const double& scale, const Rotation& Rx, const Pixel& zi,
                                       const Pixel& offset,  // no-lint
                                       boost::optional<Eigen::MatrixXd&> Hx = boost::none,
                                       boost::optional<Eigen::MatrixXd&> Hs = boost::none,
                                       boost::optional<Eigen::MatrixXd&> HR = boost::none)
  {
    const Pixel pred{ predict(x, scale, Rx, offset, Hx, Hs, HR) };  // no-lint
    const Pixel err{ pred - zi };

    return err;
  }

  virtual Eigen::VectorXd evaluateError(const SE2& x, const double& scale, const Rotation& Rx,
                                        boost::optional<Eigen::MatrixXd&> Hx = boost::none,
                                        boost::optional<Eigen::MatrixXd&> Hs = boost::none,
                                        boost::optional<Eigen::MatrixXd&> HR = boost::none) const override
  {
    const Eigen::VectorXd error{ compute_error(x, scale, Rx, _zi, _offset, Hx, Hs, HR) };

    return error;
  }

  void print(const std::string& s, const gtsam::KeyFormatter& keyFormatter) const override
  {
    const std::string key_x{ keyFormatter(this->template key<1>()) };
    const std::string key_s{ keyFormatter(this->template key<2>()) };
    const std::string key_R{ keyFormatter(this->template key<3>()) };

    std::cout << s << " ";
    std::cout << "[ " << key_x << " " << key_s << " " << key_R << "]\n";
    std::cout << "\t offset: " << _offset.transpose() << "\n";
    std::cout << "\t zi: " << _zi.transpose() << "\n";

    if (this->noiseModel_)
      this->noiseModel_->print("  noise model: ");
    else
      std::cout << "no noise model" << std::endl;
    std::cout << "\n";
  }

private:
  const Pixel _offset;
  const Pixel _zi;
};

class endcap_2Dobservation_scale_fix_factor_t : public gtsam::NoiseModelFactorN<gtsam::Pose2, gtsam::Rot2>
{
public:
  using SE2 = gtsam::Pose2;
  using Pixel = Eigen::Vector<double, 2>;
  using Rotation = gtsam::Rot2;
  using Base = gtsam::NoiseModelFactor1<SE2, gtsam::Rot2>;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;

  // using Vector = Eigen::Vector<double, Dim>;

  endcap_2Dobservation_scale_fix_factor_t(const gtsam::Key key_se2, const gtsam::Key key_offset, const Pixel zi,
                                          const Pixel offset, const double scale, const NoiseModel& cost_model)
    : Base(cost_model, key_se2, key_offset), _zi(zi), _offset(offset), _scale(scale)
  {
  }

  virtual Eigen::VectorXd evaluateError(const SE2& x, const Rotation& Rx,
                                        boost::optional<Eigen::MatrixXd&> Hx = boost::none,
                                        boost::optional<Eigen::MatrixXd&> HR = boost::none) const override
  {
    const Eigen::VectorXd error{ endcap_2Dobservation_factor_t::compute_error(x, _scale, Rx, _zi, _offset, Hx,
                                                                              boost::none, HR) };

    return error;
  }

  void print(const std::string& s, const gtsam::KeyFormatter& keyFormatter) const override
  {
    const std::string key_x{ keyFormatter(this->template key<1>()) };
    const std::string key_R{ keyFormatter(this->template key<2>()) };

    std::cout << s << " ";
    std::cout << "[ " << key_x << " " << key_R << "]\n";
    std::cout << "\t offset: " << _offset.transpose() << "\n";
    std::cout << "\t zi: " << _zi.transpose() << "\n";

    if (this->noiseModel_)
      this->noiseModel_->print("  noise model: ");
    else
      std::cout << "no noise model" << std::endl;
    std::cout << "\n";
  }

  inline double scale() const
  {
    return _scale;
  }

private:
  const Pixel _offset;
  const Pixel _zi;
  const double _scale;
};

class endcap_2Drotation_offset_factor_t : public gtsam::NoiseModelFactorN<gtsam::Rot2, gtsam::Rot2>
{
public:
  using Rotation = gtsam::Rot2;
  using Base = gtsam::NoiseModelFactor1<Rotation, Rotation>;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;

  template <typename RotationType>
  endcap_2Drotation_offset_factor_t(const gtsam::Key key_rotA, const gtsam::Key key_rotB, const RotationType Roff,
                                    const NoiseModel& cost_model)
    : Base(cost_model, key_rotA, key_rotB), _Roff(Roff)
  {
  }

  static Eigen::VectorXd compute_error(const Rotation& Ra, const Rotation& Rb, const Rotation& Roff,  // no-lint
                                       gtsam::OptionalJacobian<1, 1> HRa = boost::none,
                                       gtsam::OptionalJacobian<1, 1> HRb = boost::none)
  {
    const bool compute_deriv{ HRa or HRb };

    Eigen::Matrix<double, 1, 1> Rab_H_Ra;
    Eigen::Matrix<double, 1, 1> Rab_H_Rb;
    Eigen::Matrix<double, 1, 1> Rba_H_Ra;
    Eigen::Matrix<double, 1, 1> Rba_H_Rb;
    Eigen::Matrix<double, 1, 1> err_H_Rab;
    Eigen::Matrix<double, 1, 1> err_H_Rba;

    const Rotation Rab{ gtsam::traits<gtsam::Rot2>::Compose(Ra, Rb,  // no-lint
                                                            compute_deriv ? &Rab_H_Ra : nullptr,
                                                            compute_deriv ? &Rab_H_Rb : nullptr) };

    const Eigen::VectorXd error{ gtsam::traits<gtsam::Rot2>::Local(Rab, Roff, err_H_Rab) };
    if (HRa)
    {
      *HRa = err_H_Rab * Rab_H_Ra;
    }
    if (HRb)
    {
      *HRb = err_H_Rab * Rab_H_Rb;
    }
    return error;
  }

  virtual Eigen::VectorXd evaluateError(const Rotation& Ra, const Rotation& Rb,
                                        boost::optional<Eigen::MatrixXd&> HRa = boost::none,
                                        boost::optional<Eigen::MatrixXd&> HRb = boost::none) const override
  {
    const Eigen::VectorXd error{ compute_error(Ra, Rb, _Roff, HRa, HRb) };

    return error;
  }

private:
  const Rotation _Roff;
};

class rotation2D_fix_identity_t : public gtsam::NoiseModelFactorN<gtsam::Rot2, gtsam::Rot2>
{
public:
  using Rotation = gtsam::Rot2;
  using Base = gtsam::NoiseModelFactor1<Rotation, Rotation>;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;

  template <typename RotationType>
  rotation2D_fix_identity_t(const gtsam::Key key_rotA, const gtsam::Key key_rotB, const RotationType Roff,
                            const NoiseModel& cost_model)
    : Base(cost_model, key_rotA, key_rotB), _Roff(Roff)
  {
  }

  static Eigen::VectorXd compute_error(const Rotation& Ra, const Rotation& Rb, const Rotation& Roff,  // no-lint
                                       gtsam::OptionalJacobian<1, 1> HRa = boost::none,
                                       gtsam::OptionalJacobian<1, 1> HRb = boost::none)
  {
    const bool compute_deriv{ HRa or HRb };

    Eigen::Matrix<double, 1, 1> Rboff_H_Rb;
    Eigen::Matrix<double, 1, 1> Rboff_H_Roff;
    Eigen::Matrix<double, 1, 1> err_H_Ra;
    Eigen::Matrix<double, 1, 1> err_H_Rboff;

    const gtsam::Rot2 Rboff{ gtsam::traits<gtsam::Rot2>::Compose(Rb, Roff,  // no-lint
                                                                 compute_deriv ? &Rboff_H_Rb : nullptr,
                                                                 compute_deriv ? &Rboff_H_Roff : nullptr) };

    const Eigen::VectorXd error{ gtsam::traits<gtsam::Rot2>::Local(Ra, Rboff, err_H_Ra, err_H_Rboff) };
    if (HRa)
    {
      // err_H_Rab = Matrix::Identity(3,3);
      *HRa = err_H_Ra;
    }
    if (HRb)
    {
      *HRb = err_H_Rboff * Rboff_H_Rb;
    }
    return error;
  }

  virtual Eigen::VectorXd evaluateError(const Rotation& Ra, const Rotation& Rb,
                                        boost::optional<Eigen::MatrixXd&> HRa = boost::none,
                                        boost::optional<Eigen::MatrixXd&> HRb = boost::none) const override
  {
    const Eigen::VectorXd error{ compute_error(Ra, Rb, _Roff, HRa, HRb) };

    return error;
  }

private:
  const Rotation _Roff;
};

// class bars2D_to_3d_factor_t : public gtsam::NoiseModelFactorN<gtsam::Pose2, gtsam::Pose3>
// {
// public:
//   using SE2 = gtsam::Pose2;
//   using SE3 = gtsam::Pose3;
//   using Pixel = Eigen::Vector<double, 2>;
//   using Translation = Eigen::Vector<double, 3>;
//   using Rotation = gtsam::Rot2;
//   using Base = gtsam::NoiseModelFactor1<SE2, SE3>;
//   using NoiseModel = gtsam::noiseModel::Base::shared_ptr;
//   using Camera = gtsam::PinholeCamera<gtsam::Cal3_S2>;
//   using CameraPtr = std::shared_ptr<Camera>;

//   // using Vector = Eigen::Vector<double, Dim>;

//   bars2D_to_3d_factor_t(const gtsam::Key key_se2, const gtsam::Key key_se3, const Pixel offset2D,
//                         const Translation offset3D, const CameraPtr camera, const double& scale,
//                         const NoiseModel& cost_model)
//     : Base(cost_model, key_se2, key_se3), _offset2D(offset2D), _offset3D(offset3D), _camera(camera), _scale(scale)
//   {
//   }

//   static Pixel compute_error(const SE2& x, const SE3& y,  // no-lint
//                              const double& scale, const Pixel& offset2D, const Translation& offset3D,
//                              const CameraPtr& camera,                         // no-lint
//                              gtsam::OptionalJacobian<2, 3> Hx = boost::none,  // no-lint
//                              gtsam::OptionalJacobian<2, 6> Hy = boost::none)
//   {
//     const bool compute_deriv{ Hx or Hy };

//     // Eigen::Matrix<double, 2, 3> s2Doff_H_s{ offset2D };
//     const Pixel scaled_2Doff{ scale * offset2D };

//     Eigen::Matrix<double, 2, 3> tf2D_H_x;
//     // Eigen::Matrix<double, 2, 2> tf2D_H_sOff;
//     const Pixel tf_2Doff{ x.transformFrom(scaled_2Doff, tf2D_H_x) };

//     Eigen::Matrix<double, 3, 6> tf3D_H_y;
//     const Translation tf_3Doff{ y.transformFrom(offset3D, tf3D_H_y) };

//     Eigen::Matrix<double, 2, 3> ptP_H_tf3D;
//     const Pixel pt_proj{ camera->project2(tf_3Doff, boost::none, ptP_H_tf3D) };

//     Eigen::Matrix<double, 2, 2> err_H_ptP{ Eigen::Matrix<double, 2, 2>::Identity() };
//     Eigen::Matrix<double, 2, 2> err_H_tf2D{ -Eigen::Matrix<double, 2, 2>::Identity() };
//     const Pixel error{ pt_proj - tf_2Doff };

//     // const Pixel error{ camera->reprojectionError(tf_3Doff, tf_2Doff, boost::none, err_H_tf3D, boost::none) };

//     if (Hx)
//     {
//       *Hx = err_H_tf2D * tf2D_H_x;
//     }
//     if (Hy)
//     {
//       *Hy = err_H_ptP * ptP_H_tf3D * tf3D_H_y;
//     }
//     return error;
//   }

//   virtual Eigen::VectorXd evaluateError(const SE2& x, const SE3& y,  // no-lint
//                                         boost::optional<Eigen::MatrixXd&> Hx = boost::none,
//                                         boost::optional<Eigen::MatrixXd&> Hy = boost::none) const override
//   {
//     const Eigen::VectorXd error{ compute_error(x, y, _scale, _offset2D, _offset3D, _camera, Hx, Hy) };

//     return error;
//   }

//   void print(const std::string& s, const gtsam::KeyFormatter& keyFormatter) const override
//   {
//     const std::string key_x{ keyFormatter(this->template key<1>()) };
//     const std::string key_y{ keyFormatter(this->template key<2>()) };

//     std::cout << s << " ";
//     std::cout << "[ " << key_x << " " << key_y << "]\n";

//     if (this->noiseModel_)
//       this->noiseModel_->print("  noise model: ");
//     else
//       std::cout << "no noise model" << std::endl;
//     std::cout << "\n";
//   }

// private:
//   const double _scale;
//   const Pixel _offset2D;
//   const Translation _offset3D;
//   const CameraPtr _camera;
// };

class bars2D_to_3d_factor_t : public gtsam::NoiseModelFactorN<gtsam::Pose3>
{
public:
  using SE2 = gtsam::Pose2;
  using SE3 = gtsam::Pose3;
  using Pixel = Eigen::Vector<double, 2>;
  using Translation = Eigen::Vector<double, 3>;
  using Rotation = gtsam::Rot2;
  using Base = gtsam::NoiseModelFactor1<SE3>;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;
  using Camera = gtsam::PinholeCamera<gtsam::Cal3_S2>;
  using CameraPtr = std::shared_ptr<Camera>;

  // using Vector = Eigen::Vector<double, Dim>;

  bars2D_to_3d_factor_t(const gtsam::Key key_se3, const SE2 pose2, const Pixel offset2D, const Translation offset3D,
                        const CameraPtr camera, const double scale, const NoiseModel& cost_model)
    : Base(cost_model, key_se3), _pose2(pose2), _offset2D(offset2D), _offset3D(offset3D), _camera(camera), _scale(scale)
  {
  }

  static Pixel compute_error(const SE2& x, const SE3& y,  // no-lint
                             const double& scale, const Pixel& offset2D, const Translation& offset3D,
                             const CameraPtr& camera,                         // no-lint
                             gtsam::OptionalJacobian<2, 3> Hx = boost::none,  // no-lint
                             gtsam::OptionalJacobian<2, 6> Hy = boost::none)
  {
    const bool compute_deriv{ Hx or Hy };

    // Eigen::Matrix<double, 2, 3> s2Doff_H_s{ offset2D };
    const Pixel scaled_2Doff{ scale * offset2D };

    Eigen::Matrix<double, 2, 3> tf2D_H_x;
    // Eigen::Matrix<double, 2, 2> tf2D_H_sOff;
    const Pixel tf_2Doff{ x.transformFrom(scaled_2Doff, tf2D_H_x) };

    Eigen::Matrix<double, 3, 6> tf3D_H_y;
    const Translation tf_3Doff{ y.transformFrom(offset3D, tf3D_H_y) };

    Eigen::Matrix<double, 2, 3> ptP_H_tf3D;
    const Pixel pt_proj{ camera->project2(tf_3Doff, boost::none, ptP_H_tf3D) };

    Eigen::Matrix<double, 2, 2> err_H_ptP{ Eigen::Matrix<double, 2, 2>::Identity() };
    Eigen::Matrix<double, 2, 2> err_H_tf2D{ -Eigen::Matrix<double, 2, 2>::Identity() };
    const Pixel error{ pt_proj - tf_2Doff };

    // const Pixel error{ camera->reprojectionError(tf_3Doff, tf_2Doff, boost::none, err_H_tf3D, boost::none) };

    if (Hx)
    {
      *Hx = err_H_tf2D * tf2D_H_x;
    }
    if (Hy)
    {
      *Hy = err_H_ptP * ptP_H_tf3D * tf3D_H_y;
    }
    return error;
  }

  virtual Eigen::VectorXd evaluateError(const SE3& y,  // no-lint
                                        boost::optional<Eigen::MatrixXd&> Hy = boost::none) const override
  {
    const Eigen::VectorXd error{ compute_error(_pose2, y, _scale, _offset2D, _offset3D, _camera, boost::none, Hy) };

    return error;
  }

  void print(const std::string& s, const gtsam::KeyFormatter& keyFormatter) const override
  {
    const std::string key_x{ keyFormatter(this->template key<1>()) };
    // const std::string key_y{ keyFormatter(this->template key<2>()) };

    std::cout << s << " ";
    std::cout << "[ " << key_x << "]\n";

    if (this->noiseModel_)
      this->noiseModel_->print("  noise model: ");
    else
      std::cout << "no noise model" << std::endl;
    std::cout << "\n";
  }

private:
  const double _scale;
  const Pixel _offset2D;
  const Translation _offset3D;
  const CameraPtr _camera;
  const SE2 _pose2;
};

class pose3_depth_factor_t : public gtsam::NoiseModelFactorN<gtsam::Pose3>
{
public:
  using SE2 = gtsam::Pose2;
  using SE3 = gtsam::Pose3;
  using Pixel = Eigen::Vector<double, 2>;
  using Translation = Eigen::Vector<double, 3>;
  using Rotation = gtsam::Rot2;
  using Base = gtsam::NoiseModelFactor1<SE3>;
  using NoiseModel = gtsam::noiseModel::Base::shared_ptr;
  using Camera = gtsam::PinholeCamera<gtsam::Cal3_S2>;
  using CameraPtr = std::shared_ptr<Camera>;

  // using Vector = Eigen::Vector<double, Dim>;

  pose3_depth_factor_t(const gtsam::Key key_se3, const Pixel offset2D, const Translation offset3D, const double z,
                       const CameraPtr camera, const NoiseModel cost_model = nullptr)
    : Base(cost_model, key_se3), _offset2D(offset2D), _offset3D(offset3D), _z(z), _camera(camera)
  {
  }

  static Eigen::Vector<double, 3> compute_error(const SE3& y, const double& z, const Pixel& offset2D,
                                                const Translation& offset3D, const CameraPtr& camera,  // no-lint
                                                gtsam::OptionalJacobian<3, 6> Hy = boost::none)
  {
    // const bool compute_deriv{ Hx or Hy };

    // Eigen::Matrix<double, 2, 3> s2Doff_H_s{ offset2D };
    // const Pixel scaled_2Doff{ scale * offset2D };

    // Eigen::Matrix<double, 2, 3> tf2D_H_x;
    // // Eigen::Matrix<double, 2, 2> tf2D_H_sOff;
    // const Pixel tf_2Doff{ x.transformFrom(scaled_2Doff, tf2D_H_x) };

    Eigen::Matrix<double, 3, 6> tf3D_H_y;
    const Translation tf_3Doff{ y.transformFrom(offset3D, tf3D_H_y) };

    // Eigen::Matrix<double, 3, 2> pt3_H_off2D;
    // Eigen::Matrix<double, 3, 1> pt3_H_dpt;
    // const Pixel pt_proj{ camera->project2(tf_3Doff, boost::none, ptP_H_tf3D) };
    const Translation pt3dp{ camera->backproject(offset2D, z) };  //, boost::none, pt3_H_dpt) };
    // DEBUG_VARS(z, pt3dp.transpose(), tf_3Doff.transpose());
    const Translation error{ tf_3Doff - pt3dp };
    // Eigen::Matrix<double, 1, 3> err_H_tf3D{ Eigen::Matrix<double, 1, 3>::Zero() };
    // err_H_tf3D(0, 2) = 1.0;
    // const Eigen::Vector<double, 1> error{ tf_3Doff[2] - z };

    if (Hy)
    {
      *Hy = tf3D_H_y;
    }
    return error;
  }

  virtual Eigen::VectorXd evaluateError(const SE3& y,  // no-lint
                                        boost::optional<Eigen::MatrixXd&> Hy = boost::none) const override
  {
    const Eigen::VectorXd error{ compute_error(y, _z, _offset2D, _offset3D, _camera, Hy) };

    return error;
  }

  void print(const std::string& s, const gtsam::KeyFormatter& keyFormatter) const override
  {
    const std::string key_x{ keyFormatter(this->template key<1>()) };
    // const std::string key_y{ keyFormatter(this->template key<2>()) };

    std::cout << s << " ";
    std::cout << "[ " << key_x << "]\n";

    if (this->noiseModel_)
      this->noiseModel_->print("  noise model: ");
    else
      std::cout << "no noise model" << std::endl;
    std::cout << "\n";
  }

private:
  const double _z;
  const Pixel _offset2D;
  const Translation _offset3D;
  const CameraPtr _camera;
  // const SE2 _pose2;
};
}  // namespace perception