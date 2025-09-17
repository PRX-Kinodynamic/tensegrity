#pragma once

#include <factor_graphs/FactorGraphResult.h>

namespace factor_graphs
{
// utilities for LM optimizer, including publishing stats
struct levenberg_marquardt_t
{
  using Values = gtsam::Values;
  using Graph = gtsam::NonlinearFactorGraph;
  using LMParams = gtsam::LevenbergMarquardtParams;

  levenberg_marquardt_t(ros::NodeHandle& nh, std::string topicname, LMParams params = LMParams()) : _lm_params(params)
  {
    _publisher = nh.advertise<FactorGraphResult>(topicname, 1, true);
  }

  double compute_error(Graph& graph, gtsam::Values& values) const
  {
    return graph.error(values);
  }

  Values optimize(Graph& graph, Values& initial_values, const bool publish)
  {
    const Values result{ calculate_estimate_safe(graph, initial_values, _lm_params) };

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