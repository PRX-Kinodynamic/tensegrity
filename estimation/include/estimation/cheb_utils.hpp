#pragma once
#include <factor_graphs/defs.hpp>
#include <interface/defs.hpp>

namespace estimation
{
template <typename ParamMatrix>
std::string cheb_to_json_array(ParamMatrix& mat)
{
  std::stringstream strstr;
  strstr << "[ ";
  for (int i = 0; i < mat.rows(); ++i)
  {
    auto row = mat.row(i);
    strstr << "[ ";
    for (int j = 0; j < row.cols(); ++j)
    {
      strstr << row[j];
      if (j < row.cols() - 1)
        strstr << ", ";
    }
    strstr << "]";
    if (i < mat.rows() - 1)
      strstr << ",";
    strstr << "\n";
  }
  strstr << "]";
  return strstr.str();
}
}  // namespace estimation