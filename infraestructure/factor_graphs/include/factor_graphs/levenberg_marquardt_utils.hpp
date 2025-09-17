#pragma once

#include <tensegrity_utils/dbg_utils.hpp>

#include <factor_graphs/FactorGraphResult.h>
#include <factor_graphs/symbols_factory.hpp>

#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

namespace factor_graphs
{
// utilities for LM optimizer, including publishing stats
struct levenberg_marquardt_t
{
  using Values = gtsam::Values;
  using Graph = gtsam::NonlinearFactorGraph;
  using LMParams = gtsam::LevenbergMarquardtParams;
  using SF = factor_graphs::symbol_factory_t;

  levenberg_marquardt_t(ros::NodeHandle& nh, std::string topicname, LMParams params = LMParams()) : _lm_params(params)
  {
    _publisher = nh.advertise<FactorGraphResult>(topicname, 1, true);
  }

  Values optimize(Graph& graph, Values& initial_values, const bool publish)
  {
    Values result;
    try
    {
      gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial_values, _lm_params);
      result = optimizer.optimize();
    }
    catch (gtsam::ValuesKeyDoesNotExist e)
    {
      PRINT_MSG("[factor_graphs::levenberg_marquardt_t] Optimization failed!")
      PRINT_MSG(e.what());
      PRINT_MSG("[gtsam::ValuesKeyDoesNotExist] Variable: " + SF::formatter(e.key()) + "\n");
      exit(-1);
    }
    if (publish)
    {
      _msg.initial_error = graph.error(initial_values);
      _msg.final_error = graph.error(result);
      _msg.size = graph.size();

      _publisher.publish(_msg);
    }
    return result;
  }

  gtsam::LevenbergMarquardtParams _lm_params;
  FactorGraphResult _msg;
  ros::Publisher _publisher;
};

}  // namespace factor_graphs