#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <limits>        // Added for std::numeric_limits
#include <string_view>   // Added for std::string_view

namespace tensegrity {
namespace constants {
const double pi{3.1415926535897932385};
const double epsilon{1e-7};
constexpr double infinity{std::numeric_limits<double>::infinity()};

// Added 'inline' to prevent Multiple Definition linker errors
inline int precision{9};
inline char separating_value{' '};

namespace color {
constexpr std::string_view normal{"\033[0m"};
constexpr std::string_view red{"\033[31m"};
constexpr std::string_view green{"\033[32m"};
constexpr std::string_view yellow{"\033[33m"};
} // namespace color
} // namespace constants
} // namespace tensegrity

// The actual patch for older Eigen 3.3 versions
namespace Eigen {
    template<typename T, int Size>
    using Vector = Eigen::Matrix<T, Size, 1>;
} // namespace Eigen