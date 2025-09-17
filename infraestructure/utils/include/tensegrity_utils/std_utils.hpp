#pragma once

#include <tensegrity_utils/type_conversions.hpp>
#include <std_msgs/Header.h>
namespace tensegrity
{
namespace utils
{
template <typename T>
static std::vector<T> split(const std::string str, const char delimiter)
{
  std::vector<T> result;
  std::istringstream ss(str);
  std::string token;
  while (std::getline(ss, token, delimiter))
  {
    if (token.size() > 0)
    {
      std::istringstream ti(token);
      T x;
      if ((ti >> x))
        result.push_back(x);
    }
  }
  return result;
}

template <typename T>
void print_container(const std::string& name, const T& container)
{
  std::cout << name << ": ";
  for (auto e : container)
  {
    std::cout << e << ", ";
  }
  std::cout << std::endl;
}

static std::string timestamp()
{
  auto t = std::time(nullptr);
  std::tm tm = *std::localtime(&t);
  std::stringstream strstr{};
  strstr << std::put_time(&tm, "%y%m%d_%H%M%S");
  return strstr.str();
}

void init_header(std_msgs::Header& msg, const std::string& frame)
{
  msg.seq = 0;
  msg.stamp = ros::Time::now();
  msg.frame_id = frame;
}

void update_header(std_msgs::Header& msg)
{
  msg.seq++;
  msg.stamp = ros::Time::now();
}

template <typename TopicConstPtr>
void shutdown_callback(const TopicConstPtr& msg)
{
  std::cout << tensegrity::constants::color::red << " ";
  std::cout << "Shutdown Callback Received! ";
  std::cout << tensegrity::constants::color::normal << " ";
  std::cout << std::endl;
  ros::Rate rate(2);
  rate.sleep();
  ros::shutdown();
}

std::string time_to_string(const ros::Time& time)
{
  // uconvert_to;
  const std::string ti{ convert_to<std::string>(time.toSec()) };
  return ti;
}

}  // namespace utils
}  // namespace tensegrity