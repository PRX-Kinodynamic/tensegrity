#pragma once

namespace tensegrity
{
namespace constants
{
const double pi{ 3.1415926535897932385 };
const double epsilon{ 1e-7 };
constexpr double infinity{ std::numeric_limits<double>::infinity() };

int precision{ 9 };
char separating_value{ ' ' };

namespace color
{
constexpr std::string_view normal{ "\033[0m" };
constexpr std::string_view red{ "\033[31m" };
constexpr std::string_view green{ "\033[32m" };
constexpr std::string_view yellow{ "\033[33m" };
}  // namespace color
}  // namespace constants
}  // namespace tensegrity