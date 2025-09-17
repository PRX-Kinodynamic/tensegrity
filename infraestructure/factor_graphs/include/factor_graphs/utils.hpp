#pragma once
#include <random>
#include <tensegrity_utils/constants.hpp>

namespace factor_graphs
{
namespace internal_random
{
static std::uniform_real_distribution<double> uniform_zero_one(0.0,
                                                               std::nextafter(1.0, std::numeric_limits<double>::max()));
static std::normal_distribution<double> gaussian_zero_one(0.0, 1.0);
std::mt19937_64 global_generator;

}  // namespace internal_random
void set_seed(int seed)
{
  internal_random::global_generator.seed(seed);
}

double random_uniform()
{
  return internal_random::uniform_zero_one(internal_random::global_generator);
}

template <typename Type>
Type random_uniform(const Type min, const Type max)
{
  const double val{ internal_random::uniform_zero_one(internal_random::global_generator) };
  const Type r{ static_cast<Type>(val) * (max - min) + min };
  return r;
}

template <typename Container, typename Type>
void random_uniform(Container& container, const Type min, const Type max)
{
  for (int i = 0; i < container.size(); ++i)
  {
    container[i] = random_uniform(min, max);
  }
}

template <typename RotationType, std::enable_if_t<std::is_same<RotationType, gtsam::Rot3>::value, bool> = true>
void random(RotationType& rotation)
{
  using RotationMatrix = Eigen::Matrix3d;
  const double x1{ random_uniform() };  // \in [0.0, 1.0]
  const double x2{ random_uniform() };  // \in [0.0, 1.0]
  const double x3{ random_uniform() };  // \in [0.0, 1.0]
  const double theta{ 2.0 * tensegrity::constants::pi * x1 };
  const double phi{ 2.0 * tensegrity::constants::pi * x2 };
  const double z{ x3 };

  const double v1{ std::cos(phi) * std::sqrt(z) };
  const double v2{ std::sin(phi) * std::sqrt(z) };
  const double v3{ std::sqrt(1.0 - z) };
  const Eigen::Vector3d V(v1, v2, v3);

  const Eigen::Rotation2D<double> R2(theta);
  RotationMatrix R3{ RotationMatrix::Identity() };
  R3.topLeftCorner<2, 2>() = R2.matrix();
  const RotationMatrix M{ (2.0 * V * V.transpose() - RotationMatrix::Identity()) * R3 };
  // return std::move(RotationOut(M));
  rotation = RotationType(M);
}

template <typename PoseType, std::enable_if_t<std::is_same<PoseType, gtsam::Pose3>::value, bool> = true>
void random(PoseType& pose)
{
  // TODO: add limits
  gtsam::Rot3 rot;
  random(rot);
  const Eigen::Vector3d pos{ Eigen::Vector3d::Random() };
  pose = std::move(PoseType(rot, pos));
}

template <typename Type>
Type random()
{
  Type value;
  random(value);
  return value;
}
}  // namespace factor_graphs