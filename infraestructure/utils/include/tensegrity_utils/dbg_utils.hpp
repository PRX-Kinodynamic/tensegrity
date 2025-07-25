#pragma once

#include <fstream>
#include <regex>

#include <ros/ros.h>

#include <tensegrity_utils/constants.hpp>
#include <tensegrity_utils/template_utils.hpp>

#define DEBUG_PRINT std::cout << __PRETTY_FUNCTION__ << ": " << __LINE__ << std::endl;

namespace tensegrity
{
static inline std::string lib_path_safe(std::string env_var)
{
  char* path = std::getenv(env_var.c_str());
  if (path == NULL)
  {
    std::cout << env_var << " environmental variable not set." << std::endl;
    exit(1);
  }

  auto p_str = std::string(path);
  return (p_str + (p_str.back() == '/' ? "" : "/"));
}

namespace dbg
{
namespace variables
{

inline static const std::string lib_path{ tensegrity::lib_path_safe("TENSEGRITY_ROS") };
inline static std::ofstream ofs_log;
}  // namespace variables

template <typename Value, std::enable_if_t<utils::is_streamable<Value>::value, bool> = true>
inline void print_value(std::ostream& stream, const Value& value)
{
  stream << value << " ";
}

template <typename Value,
          std::enable_if_t<utils::is_iterable<Value>::value and not utils::is_streamable<Value>::value, bool> = true>
inline void print_value(std::ostream& stream, const Value& value)
{
  for (auto e : value)
  {
    print_value(stream, e);
  }
  // stream << "\n";
}

inline void print_variables(std::ostream& stream, bool color, std::string name)
{
  stream << std::endl;
}

template <typename Var0, class... Vars>
inline void print_variables(std::ostream& stream, bool color, std::string name, Var0 var, Vars... vars)
{
  const std::regex regex(",(\\s*)+");
  std::string var_name{ name };
  std::string other_names{ "" };
  std::smatch match;  // <-- need a match object
  // std::cout << "name: " << name << std::endl;
  if (std::regex_search(name, match, regex))  // <-- use it here to get the match
  {
    const int split_on = match.position();  // <-- use the match position
    var_name = name.substr(0, split_on);
    other_names = name.substr(split_on + match.length());  // <-- also, skip the whole math
  }
  if (color)
  {
    stream << constants::color::yellow;
    stream << var_name << ": ";
    stream << constants::color::normal;
  }
  else
  {
    stream << var_name << ": ";
  }

  print_value(stream, var);
  print_variables(stream, color, other_names, vars...);
}

template <class... Vars>
inline void log_variables(std::string name, Vars... vars)
{
  using dbg::variables::ofs_log;
  if (not ofs_log.is_open())
  {
    ofs_log.open(dbg::variables::lib_path + "/log.txt");
  }
  dbg::print_variables(ofs_log, false, name, vars...);
}
}  // namespace dbg
}  // namespace tensegrity
#define DEBUG_VARS(...) tensegrity::dbg::print_variables(std::cout, true, #__VA_ARGS__, __VA_ARGS__);
#define LOG_VARS(...) tensegrity::dbg::log_variables(#__VA_ARGS__, __VA_ARGS__);
#define PRINT_MSG(MSG)                                                                                                 \
  {                                                                                                                    \
    const std::string msg{ MSG };                                                                                      \
    DEBUG_VARS(msg)                                                                                                    \
  };

#define PRINT_MSG_VARS(MSG, ...)                                                                                       \
  {                                                                                                                    \
    const std::string msg{ MSG };                                                                                      \
    tensegrity::dbg::print_variables(std::cout, true, "msg", msg, #__VA_ARGS__, __VA_ARGS__);                          \
  };

// #define PRINT_KEY(KEY)                                                                                                 \
//   {                                                                                                                    \
//     const std::string key{ SF::formatter(KEY) };                                                                       \
//     dbg::print_variables(std::cout, true, #KEY, key);                                                                  \
//   };

#define PRINT_MSG_ONCE(MSG)                                                                                            \
  static bool deprecated_print_once = []() {                                                                           \
    const std::string msg{ MSG };                                                                                      \
    DEBUG_VARS(msg)                                                                                                    \
    return true;                                                                                                       \
  }();
