#pragma once

#include <ros/ros.h>
#include <ros/node_handle.h>

#define GET_VARIABLE_NAME(Variable) (#Variable)

#define ROS_PARAM_SETUP(nh, var) (utils::get_param_or_exit(nh, GET_VARIABLE_NAME(var), var))
#define GLOBAL_PARAM_SETUP(nh, var) PARAM_NAME_SETUP(nh, "/" #var, var)
#define PARAM_SETUP(nh, var) PARAM_NAME_SETUP(nh, GET_VARIABLE_NAME(var), var)
#define NODELET_PARAM_SETUP(nh, var) PARAM_NAME_SETUP(nh, GET_VARIABLE_NAME(var), var)
#define PARAM_SETUP_WITH_DEFAULT(nh, var, default_value) NODELET_PARAM_SETUP_WITH_DEFAULT(nh, var, default_value)

namespace tensegrity
{
namespace utils
{

template <typename T>
void get_param_or_exit(ros::NodeHandle& nh, const std::string var_name, T& var)
{
  if (!nh.getParam(var_name, var))
  {
    ROS_FATAL_STREAM("Parameter " << var_name << " is needed.");
    exit(-1);
  }
}

// This function is to check without exiting. For example, if a parameter is set by another node and the current node
// needs to wait till it is set.
template <typename T>
bool param_check_then_get(const std::string param_path, T& var)
{
  if (ros::param::has(param_path))
  {
    ros::param::get(param_path, var);
    return true;
  }
  return false;
}

#define PARAM_NAME_SETUP(nh, name, var)                                                                                \
  if (!nh.getParam(name, var))                                                                                         \
  {                                                                                                                    \
    ROS_ERROR_STREAM_NAMED(ros::this_node::getName(), "Parameter " << GET_VARIABLE_NAME(var) << " is needed.");        \
    exit(-1);                                                                                                          \
  }

#define NODELET_PARAM_SETUP_WITH_DEFAULT(nh, var, default_value)                                                       \
  if (!nh.getParam(GET_VARIABLE_NAME(var), var))                                                                       \
  {                                                                                                                    \
    var = default_value;                                                                                               \
  }

}  // namespace utils
}  // namespace tensegrity