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

  // static constexpr Eigen::Index DimOut{ OutType::ColsAtCompileTime };
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
  // using GradientsArray = std::array<torch::Tensor, DimOut>;

  // inline static DifferenceFunction DefaultDiff = [](const X& a, const X& b) { return a - b; };

  torch_nn_interface_t(const torch_nn_interface_t& other) = delete;

public:
  // Make the interface the owner of the ptr. Would need multiple interfaces if multiple modules are needed
  // torch_nn_interface_t() : _module_ptr(nullptr)
  // {
  // }

  torch_nn_interface_t(const std::string filename) : _module_ptr(load_model_(filename))
  {
    _torch_options = torch::TensorOptions()
                         .dtype(torch::kFloat64)
                         .layout(torch::kStrided)
                         .device(torch::kCPU, 1)
                         .requires_grad(true);
  }

  // Use a pointer own by someone else. The pointer should keep being valid through the existence of the interface
  torch_nn_interface_t(ModulePtr module_ptr = nullptr) : _module_ptr(module_ptr)
  // , _in_grads(create_in_grads())
  {
    _torch_options = torch::TensorOptions()
                         .dtype(torch::kFloat64)
                         .layout(torch::kStrided)
                         .device(torch::kCPU, 0)
                         .requires_grad(true);
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
    return eval(types...);
  }

  inline OutType eval(const InTypes&... types)
  {
    // TensorArray tensor_array;
    _inputs.clear();
    torch::IValue tensor_output{ eval_(_module_ptr, _tensor_array, 0, _inputs, _torch_options, types...) };
    if (tensor_output.isTuple())
    {
      auto tuple = tensor_output.toTuple();
      recover_torch_output<0>(_output, tuple->elements());
    }
    //   for (auto e : tuple->elements())
    //   {
    // if (torch_vec[idx].isTensor())
    // {

    // return eval(types..., OptionalMatrix<InTypes>()...);
    // interface::copy(output, tensor_output);
    return _output;
  }
  // void eval(OutType output, const InTypes&... types)
  // OutType eval(const InTypes&... types, OptionalMatrix<InTypes>... H)
  // {
  //   DEBUG_PRINT
  //   _inputs.clear();
  //   torch::Tensor tensor_output{ eval_(_module_ptr, _tensor_array, 0, _inputs, _torch_options, types...) };
  //   DEBUG_PRINT

  //   compute_gradients(tensor_output, _tensor_array, 0, _in_grads, H...);
  //   // torch::Tensor grad{ tensor_output };
  //   // PRX_DBG_VARS(_in_grads);
  //   interface::copy(_output, tensor_output);

  //   return _output;
  // }
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

  // static GradientsArray create_in_grads()
  // {
  //   GradientsArray arr;
  //   for (int i = 0; i < DimOut; ++i)
  //   {
  //     arr[i] = torch::zeros({ DimOut });
  //     arr[i][i] = 1;
  //   }
  //   return arr;
  // }

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
  static void fill_inputs(Type& type, Inputs& inputs, TensorArray& tensor_array, const std::size_t idx,
                          const at::TensorOptions& opts)
  {
    // DEBUG_VARS(idx, NumTypes);
    tensor_array[idx] = torch::zeros({ type.size() }, opts);

    // torch::Tensor tx{ torch::zeros({ 5 }, opts) };
    // DEBUG_PRINT
    // // std::cout << "tx: " << tx << std::endl;
    // DEBUG_VARS(type.size());
    // DEBUG_PRINT
    assert(tensor_array[idx].dtype() == torch::kFloat64);
    // DEBUG_PRINT
    interface::copy(tensor_array[idx], type);
    // DEBUG_PRINT
    // DEBUG_VARS(type);
    // DEBUG_PRINT
    // DEBUG_VARS(tensor_array[idx].sizes());
    // DEBUG_PRINT

    inputs.push_back(tensor_array[idx]);
  }

  template <typename Car, typename... Cdr, std::enable_if_t<(sizeof...(Cdr) >= 1), bool> = true>
  static torch::IValue eval_(ModulePtr model, TensorArray& tensor_array, const std::size_t idx, Inputs& inputs,
                             const at::TensorOptions& opts, const Car& car, const Cdr&... cdr)
  {
    fill_inputs(car, inputs, tensor_array, idx, opts);

    return eval_(model, tensor_array, idx + 1, inputs, opts, cdr...);
  }

  template <typename Car, typename... Cdr, std::enable_if_t<(sizeof...(Cdr) == 0), bool> = true>
  static torch::IValue eval_(ModulePtr model, TensorArray& tensor_array, const std::size_t idx, Inputs& inputs,
                             const at::TensorOptions& opts, const Car& car, const Cdr&... types)
  {
    fill_inputs(car, inputs, tensor_array, idx, opts);
    // auto tuple = model->forward(inputs).toTuple();
    const torch::IValue result{ model->forward(inputs) };
    // DEBUG_VARS(result);
    // DEBUG_VARS(result.isTensor());

    // DEBUG_PRINT
    // OutType _output;
    // std::size_t idx{ 0 };
    // for (auto e : tuple->elements())
    // {
    //   DEBUG_VARS(e.isTensor());
    //   Eigen::VectorXd vec;
    //   interface::copy(vec, e.toTensor());
    //   DEBUG_VARS(vec);
    // }
    // DEBUG_PRINT
    // recover_torch_output(_output, result);
    return result;
  }
  template <std::size_t idx, typename OutputType, typename TorchVector,
            std::enable_if_t<idx != std::tuple_size_v<OutputType>, bool> = true>
  void recover_torch_output(OutputType& out, const TorchVector& torch_vec)
  {
    // if (torch_out.isTuple())
    // {
    //   auto tuple = torch_out.toTuple();
    //   for (auto e : tuple->elements())
    //   {
    // if (torch_vec[idx].isTensor())
    // {
    auto tensor = torch_vec[idx].toTensor();
    interface::copy(std::get<idx>(out), tensor);
    // DEBUG_VARS(std::get<idx>(out));
    // }
    recover_torch_output<idx + 1>(out, torch_vec);

    //   }
    // }
  }
  template <std::size_t idx, typename OutputType, typename TorchVector,
            std::enable_if_t<idx == std::tuple_size_v<OutputType>, bool> = true>
  void recover_torch_output(OutputType& out, const TorchVector& torch_vec)
  {
  }

  // template <typename Hcar, typename... Hcdr, std::enable_if_t<(sizeof...(Hcdr) >= 1), bool> = true>
  // static void compute_gradients(torch::Tensor& output, TensorArray& inputs, const std::size_t idx,
  //                               GradientsArray& grads, Hcar& H, Hcdr... Hs)
  // {
  //   if (H)
  //   {
  //     compute_grad(*H, output, inputs[idx], grads);
  //   }
  //   compute_gradients(output, inputs, idx + 1, grads, Hs...);
  // }

  // template <typename Hcar, typename... Hcdr, std::enable_if_t<(sizeof...(Hcdr) == 0), bool> = true>
  // static void compute_gradients(torch::Tensor& output, TensorArray& inputs, const std::size_t idx,
  //                               GradientsArray& grads, Hcar& H, Hcdr... Hs)
  // {
  //   if (H)
  //   {
  //     compute_grad(*H, output, inputs[idx], grads);
  //   }
  // }

  // static void compute_grad(Eigen::MatrixXd& H, torch::Tensor& output, torch::Tensor& input, GradientsArray& grads)
  // {
  //   const std::size_t DimIn{ static_cast<size_t>(input.size(0)) };
  //   H = Eigen::MatrixXd::Zero(DimOut, DimIn);
  //   for (int i = 0; i < DimIn; ++i)
  //   {
  //     compute_grad_column(H, output, input, grads, 0, i);
  //   }
  // }

  // static void compute_grad_column(Eigen::MatrixXd& H, torch::Tensor& output, torch::Tensor& input,
  //                                 GradientsArray& grads, const std::size_t& row, const std::size_t& col)
  // {
  //   auto partial = torch::autograd::grad({ output }, { input }, { grads[row] }, true);
  //   interface::copy(H.block(row, 0, 1, H.cols()).transpose(), partial[0]);

  //   if (row + 1 < grads.size())
  //   {
  //     compute_grad_column(H, output, input, grads, row + 1, col);
  //   }
  // }

  // GradientsArray _in_grads;  // Input Gradients (torch wtf?) per column

  TensorArray _tensor_array;

  ModulePtr _module_ptr;
  // const Eigen::Index _in_dim;
  // Eigen::Index _idx;
  torch::jit::script::Module _module;

  OutType _output;
  // at::Tensor _tensor_output;
  std::vector<torch::jit::IValue> _inputs;
  at::TensorOptions _torch_options;
};

}  // namespace interface
#endif