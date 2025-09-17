#pragma once
#include <fstream>
#include <gtsam/nonlinear/Values.h>
#include <factor_graphs/symbols_factory.hpp>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

namespace factor_graphs
{
using SF = symbol_factory_t;

template <typename LinearGraph>
void indeterminant_linear_system_helper(const LinearGraph graph, std::ostream& os = std::cout)
{
  os << "Keys:\n";
  int idx{ 0 };
  for (auto k : graph->keys())
  {
    os << "[" << idx << "]: " << SF::formatter(k) << "\n";
    idx++;
  }
  os << "Use Matlab's 'sparse()':\n";
  os << graph->sparseJacobian_() << "\n";

  os << "Jacobian A|b':\n";
  os << "\tA:\n";
  auto Ab = graph->jacobian();
  os << Ab.first << "\n";
  os << "\tb:\n";
  os << Ab.second << "\n";

  auto H = graph->hessian();
  os << "Hessian A|b':\n";
  os << "\tA:\n";
  os << H.first << "\n";
  os << "\tb:\n";
  os << H.second << "\n";
  // std::pair<Matrix, Vector>
}
void indeterminant_linear_system_helper(const gtsam::NonlinearFactorGraph& graph, const gtsam::Values& values,
                                        std::ostream& os = std::cout)
{
  boost::shared_ptr<gtsam::GaussianFactorGraph> fgl{ graph.linearize(values) };
  indeterminant_linear_system_helper(fgl, os);
}
}  // namespace factor_graphs
