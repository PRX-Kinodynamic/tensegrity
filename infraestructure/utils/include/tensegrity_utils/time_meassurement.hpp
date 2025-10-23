#pragma once
#include <chrono>

namespace tensegrity
{
namespace utils
{
struct time_meassurement_t
{
  time_meassurement_t()
  {
  }

  void operator()(const std::string id, const bool end)
  {
    if (not end)
    {
      start = std::chrono::steady_clock::now();
    }
    else
    {
      finish = std::chrono::steady_clock::now();
      const std::chrono::duration<double> elapsed_seconds{ finish - start };
      const double dt{ elapsed_seconds.count() };
      DEBUG_VARS(id, dt);
    }
  }

private:
  std::chrono::time_point<std::chrono::steady_clock> start, finish;
};  // namespace struct time_meassurement_t
}  // namespace utils
}  // namespace tensegrity