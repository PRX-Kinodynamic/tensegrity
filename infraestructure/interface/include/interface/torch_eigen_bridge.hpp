#ifndef TORCH_NOT_BUILT
#pragma once

/**
 * @file transforms.hpp
 * @author Edgar Granados
 * @brief <b> Bridge between Eigen and Torch
 * */

#include <Eigen/Dense>
#include <Eigen/Core>
#include <torch/torch.h>
#include <torch/script.h>
// #include "prx/utilities/defs.hpp"
namespace interface
{

template <typename Derived>
void copy(torch::Tensor& tensor, const Eigen::MatrixBase<Derived>& vec)
{
  static constexpr Eigen::Index RowsComp{ Eigen::MatrixBase<Derived>::RowsAtCompileTime };
  static constexpr Eigen::Index ColsComp{ Eigen::MatrixBase<Derived>::ColsAtCompileTime };
  // const std::pair<Eigen::Index, Eigen::Index> dimensions{ get_dimension(vec) };
  // const Eigen::Index rows{ dimensions.first };
  // const Eigen::Index cols{ dimensions.second };

  // DEBUG_VARS(vec.rows(), vec.cols());
  // DEBUG_VARS(tensor);
  // DEBUG_PRINT
  double* data{ tensor.data_ptr<double>() };
  Eigen::Map<Eigen::Matrix<double, RowsComp, ColsComp>> ef(data, vec.rows(), vec.cols());
  ef = vec.template cast<double>();
  // DEBUG_VARS(ef);
  // DEBUG_PRINT
}

template <typename Derived>
void copy(Eigen::MatrixBase<Derived> const& vec, const torch::Tensor& tensor)
{
  static constexpr Eigen::Index RowsComp{ Eigen::MatrixBase<Derived>::RowsAtCompileTime };
  static constexpr Eigen::Index ColsComp{ Eigen::MatrixBase<Derived>::ColsAtCompileTime };
  // const std::pair<Eigen::Index, Eigen::Index> dimensions{ get_dimension(vec) };
  // const Eigen::Index rows{ dimensions.first };
  // const Eigen::Index cols{ dimensions.second };
  // PRX_DBG_VARS(vec.transpose());

  // using RowMajorMat = Eigen::Matrix<float, RowsComp, ColsComp, Eigen::RowMajor>;
  using RowMajorMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  // PRX_DBG_VARS(tensor)
  // PRX_DBG_VARS(tensor.size(0))
  auto sizes = tensor.sizes();
  const int rows{ sizes.size() > 0 ? static_cast<int>(sizes[0]) : 0 };
  const int cols{ sizes.size() > 1 ? static_cast<int>(sizes[1]) : 1 };
  // PRX_DBG_VARS(rows, cols)
  // PRX_DBG_VARS(dimensions.first, dimensions.second)
  double* data{ tensor.data_ptr<double>() };
  // Eigen::Map<RowMajorMat> ef(data, rows, cols);
  Eigen::Map<RowMajorMat> ef(data, rows, cols);

  // PRX_DBG_VARS(ef);

  // ef = vec.template cast<float>();
  const_cast<Eigen::MatrixBase<Derived>&>(vec) = ef.template cast<double>();
}

}  // namespace interface
#endif
