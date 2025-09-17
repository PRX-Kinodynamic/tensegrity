#pragma once

#include <filesystem>

#include <tensegrity_utils/assert.hpp>

#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/Point.h>

namespace interface
{

template <typename Derived>
inline void copy(geometry_msgs::Vector3& msg, const Eigen::MatrixBase<Derived>& vec)
{
  msg.x = vec[0];
  msg.y = vec[1];
  msg.z = vec[2];
}

template <typename Derived>
inline void copy(Eigen::MatrixBase<Derived> const& vec, const geometry_msgs::Vector3& msg)
{
  const Eigen::Vector3d _vec{ Eigen::Vector3d(msg.x, msg.y, msg.z) };
  // vec[0] = msg.x;
  // vec[1] = msg.y;
  // vec[2] = msg.z;
  const_cast<Eigen::MatrixBase<Derived>&>(vec) = _vec;
}

inline void copy(geometry_msgs::Quaternion& msg, const Eigen::Quaterniond& quat)
{
  msg.x = quat.x();
  msg.y = quat.y();
  msg.z = quat.z();
  msg.w = quat.w();
}

inline void copy(Eigen::Quaterniond& quat, const geometry_msgs::Quaternion& msg)
{
  quat.x() = msg.x;
  quat.y() = msg.y;
  quat.z() = msg.z;
  quat.w() = msg.w;
}

template <typename Derived>
inline void copy(geometry_msgs::Point& msg, const Eigen::MatrixBase<Derived>& vec)
{
  msg.x = vec[0];
  msg.y = vec[1];
  msg.z = vec[2];
}

template <typename Derived>
inline void copy(Eigen::MatrixBase<Derived>& vec, const geometry_msgs::Point& msg)
{
  vec[0] = msg.x;
  vec[1] = msg.y;
  vec[2] = msg.z;
}

template <typename Derived>
inline void copy(Eigen::MatrixBase<Derived>& matrix, std::vector<double>& values)
{
  const Eigen::Index rows{ matrix.rows() };
  const Eigen::Index cols{ matrix.cols() };
  TENSEGRITY_ASSERT(rows * cols == values.size(), "Matrix (" << rows << " x " << cols << ") expected " << rows * cols
                                                             << ". Instead got: " << values.size());

  // Row first
  int idx{ 0 };
  for (int i = 0; i < rows; ++i)
  {
    for (int j = 0; j < cols; ++j)
    {
      matrix(i, j) = values[idx];
      idx++;
    }
  }
}

}  // namespace interface
