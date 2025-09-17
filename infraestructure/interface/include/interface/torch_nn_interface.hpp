#ifndef TORCH_NOT_BUILT
#pragma once
#include <array>
#include <numeric>

#include <gtsam/config.h>
#include <gtsam/base/Testable.h>

#include <torch/torch.h>
#include "tensegrity_utils/assert.hpp"
#include "interface/torch_eigen_bridge.hpp"
// #include "prx/utilities/defs.hpp"

namespace interface
{
template <typename Module, typename OutType, typename... InTypes>
class torch_nn_interface_t
{
  using Derived = torch_nn_interface_t<OutType, InTypes...>;

  static constexpr Eigen::Index DimOut{ OutType::ColsAtCompileTime };
  static constexpr std::size_t NumTypes{ sizeof...(InTypes) };

  using OptDeriv = boost::optional<Eigen::MatrixXd&>;
  template <typename T>
  using OptionalMatrix = boost::optional<Eigen::MatrixXd&>;

  // using Error = Eigen::Vector<double, DimX>;
  // template <typename Input>
  // using Partial = std::function<Error(const Input&)>;
  // template <typename Input>
  // using FirstOrderDerivative = prx::math::first_order_derivative_t<Partial<Input>, Input, 4>;
  // using DifferenceFunction = std::function<X(const X&, const X&)>;

  // using Module = torch::jit::script::Module;
  // using Module = torch::nn::Module;
  using ModulePtr = std::shared_ptr<Module>;
  using Inputs = std::vector<torch::jit::IValue>;

  using TensorArray = std::array<torch::Tensor, NumTypes + 1>;
  using GradientsArray = std::array<torch::Tensor, DimOut>;

  // inline static DifferenceFunction DefaultDiff = [](const X& a, const X& b) { return a - b; };

  torch_nn_interface_t(const torch_nn_interface_t& other) = delete;

public:
  // Make the interface the owner of the ptr. Would need multiple interfaces if multiple modules are needed
  // torch_nn_interface_t() : _module_ptr(nullptr)
  // {
  // }

  torch_nn_interface_t(const std::string filename) : _module_ptr(load_model_(filename))
  {
  }

  // Use a pointer own by someone else. The pointer should keep being valid through the existence of the interface
  torch_nn_interface_t(ModulePtr module_ptr = nullptr) : _module_ptr(module_ptr), _in_grads(create_in_grads())
  {
  }

  ~torch_nn_interface_t()
  {
  }

  void load_model(const std::string filename)
  {
    _module_ptr = torch_nn_interface_t::load_model_(filename);
  }

  // OutType operator()(const InTypes&... types, OptionalMatrix<ValueTypes>... H)
  OutType operator()(const InTypes&... types)
  {
    // _idx = 0;
    // torch::Tensor tensor_output{ eval(types...) };
    // interface::copy(_output, tensor_output);
    return eval(types..., OptionalMatrix<InTypes>()...);
  }

  inline OutType eval(const InTypes&... types)
  {
    // TensorArray tensor_array;
    return eval(types..., OptionalMatrix<InTypes>()...);
    // interface::copy(output, tensor_output);
  }
  // void eval(OutType output, const InTypes&... types)
  OutType eval(const InTypes&... types, OptionalMatrix<InTypes>... H)
  {
    _inputs.clear();
    torch::Tensor tensor_output{ eval_(_module_ptr, _tensor_array, 0, _inputs, types...) };

    compute_gradients(tensor_output, _tensor_array, 0, _in_grads, H...);
    // torch::Tensor grad{ tensor_output };
    // PRX_DBG_VARS(_in_grads);
    interface::copy(_output, tensor_output);

    return _output;
  }
  // inline void eval(OutType output, const InTypes&... types)
  // {
  // evaluateError(output, types..., boost::none);
  // }

protected:
  static ModulePtr load_model_(const std::string filename)
  {
    ModulePtr module;
    try
    {
      module = std::make_shared<Module>(torch::jit::load(filename, torch::kCPU));
    }
    catch (const c10::Error& e)
    {
      std::cout << e.what() << "\n";
      TENSEGRITY_THROW("Error loading the model\n");
    }
    return module;
  }

  static GradientsArray create_in_grads()
  {
    GradientsArray arr;
    for (int i = 0; i < DimOut; ++i)
    {
      arr[i] = torch::zeros({ DimOut });
      arr[i][i] = 1;
    }
    return arr;
  }

  template <typename Car, typename... Cdr, std::enable_if_t<(sizeof...(Cdr) == 0), bool> = true>
  static Eigen::Index dimension()
  {
    return Car::ColsAtCompileTime;
  }
  template <typename Car, typename... Cdr, std::enable_if_t<(sizeof...(Cdr) >= 1), bool> = true>
  static Eigen::Index dimension()
  {
    return Car::ColsAtCompileTime + dimension<Cdr...>();
  }

  template <typename Type>
  static void fill_inputs(Type& type, Inputs& inputs, TensorArray& tensor_array, const std::size_t idx)
  {
    tensor_array[idx] = torch::zeros({ type.size() }, torch::requires_grad());
    interface::copy(tensor_array[idx], type);
    // PRX_DBG_VARS(idx, tensor_array[idx]);
    inputs.push_back(tensor_array[idx]);
  }

  template <typename Car, typename... Cdr, std::enable_if_t<(sizeof...(Cdr) >= 1), bool> = true>
  static torch::Tensor eval_(ModulePtr model, TensorArray& tensor_array, const std::size_t idx, Inputs& inputs,
                             const Car& car, const Cdr&... cdr)
  {
    fill_inputs(car, inputs, tensor_array, idx);
    return eval_(model, tensor_array, idx + 1, inputs, cdr...);
  }

  template <typename Car, typename... Cdr, std::enable_if_t<(sizeof...(Cdr) == 0), bool> = true>
  static torch::Tensor eval_(ModulePtr model, TensorArray& tensor_array, const std::size_t idx, Inputs& inputs,
                             const Car& car, const Cdr&... types)
  {
    fill_inputs(car, inputs, tensor_array, idx);
    return model->forward(inputs).toTensor();
  }

  template <typename Hcar, typename... Hcdr, std::enable_if_t<(sizeof...(Hcdr) >= 1), bool> = true>
  static void compute_gradients(torch::Tensor& output, TensorArray& inputs, const std::size_t idx,
                                GradientsArray& grads, Hcar& H, Hcdr... Hs)
  {
    if (H)
    {
      // PRX_DEBUG_PRINT
      compute_grad(*H, output, inputs[idx], grads);
      // PRX_DBG_VARS(*H);
    }
    compute_gradients(output, inputs, idx + 1, grads, Hs...);
  }

  template <typename Hcar, typename... Hcdr, std::enable_if_t<(sizeof...(Hcdr) == 0), bool> = true>
  static void compute_gradients(torch::Tensor& output, TensorArray& inputs, const std::size_t idx,
                                GradientsArray& grads, Hcar& H, Hcdr... Hs)
  {
    if (H)
    {
      // PRX_DEBUG_PRINT
      compute_grad(*H, output, inputs[idx], grads);
      // PRX_DBG_VARS(*H);
    }
  }

  static void compute_grad(Eigen::MatrixXd& H, torch::Tensor& output, torch::Tensor& input, GradientsArray& grads)
  {
    // PRX_DBG_VARS(input)
    const std::size_t DimIn{ static_cast<size_t>(input.size(0)) };
    H = Eigen::MatrixXd::Zero(DimOut, DimIn);
    // PRX_DBG_VARS(DimOut, DimIn)
    // PRX_DBG_VARS(output)
    for (int i = 0; i < DimIn; ++i)
    {
      // PRX_DBG_VARS(i)
      compute_grad_column(H, output, input, grads, 0, i);
    }
  }

  static void compute_grad_column(Eigen::MatrixXd& H, torch::Tensor& output, torch::Tensor& input,
                                  GradientsArray& grads, const std::size_t& row, const std::size_t& col)
  {
    // PRX_DEBUG_PRINT
    auto partial = torch::autograd::grad({ output }, { input }, { grads[row] }, true);
    // PRX_DBG_VARS(partial.size(), partial);
    // PRX_DBG_VARS(H);
    interface::copy(H.block(row, 0, 1, H.cols()).transpose(), partial[0]);
    // PRX_DBG_VARS(H);
    // H(row, i) = partial[i].template item<double>();

    if (row + 1 < grads.size())
    {
      compute_grad_column(H, output, input, grads, row + 1, col);
    }
  }

  GradientsArray _in_grads;  // Input Gradients (torch wtf?) per column

  TensorArray _tensor_array;

  ModulePtr _module_ptr;
  // const Eigen::Index _in_dim;
  // Eigen::Index _idx;
  torch::jit::script::Module _module;

  OutType _output;
  // at::Tensor _tensor_output;
  std::vector<torch::jit::IValue> _inputs;
};

}  // namespace interface
#endif