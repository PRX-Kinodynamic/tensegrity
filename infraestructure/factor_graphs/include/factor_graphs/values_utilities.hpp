#pragma once
#include <fstream>
#include <gtsam/nonlinear/Values.h>
#include "factor_graphs/symbols_factory.hpp"

namespace factor_graphs
{
using SF = symbol_factory_t;

template <typename Estimate>
void get_value_safe(gtsam::Values& values, Estimate& variable, const gtsam::Key& key)
{
  try
  {
    variable = values.at<Estimate>(key);
  }
  catch (gtsam::ValuesKeyDoesNotExist e)
  {
    const std::string msg{ "[EXCEPTION] Var:" + SF::formatter(e.key()) + "\n" };
    std::cout << msg << std::string(e.what()) << std::endl;
  }
}

template <typename Type>
void values_by_type_to_file(const gtsam::Values& values, const std::string filename,
                            const std::ios_base::openmode _mode = std::ofstream::trunc)
{
  std::ofstream ofs(filename.c_str(), _mode);
  for (auto pair : values.keys())
  {
    try
    {
      const Type value{ values.at<Type>(pair) };
      ofs << SF::formatter(pair) << " ";
      ofs << value << "\n";
    }
    catch (...)
    {
    }
  }
  ofs.close();
}

}  // namespace factor_graphs
