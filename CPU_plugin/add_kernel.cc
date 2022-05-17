// add_kernel.cc

#include "paddle/phi/extension.h" // the header file on which the custom kernel depends

namespace custom_cpu {

// Kernel Implementation
template <typename T, typename Context>
void AddKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::DenseTensor& y,
               phi::DenseTensor* out) {
  // Use Alloc API of dev_ctx to allocate storage of the template parameter T for the output parameter--out.
  dev_ctx.template Alloc<T>(out);
  // Use numel API of DenseTensor to acquire the number of Tensor elements.
  auto numel = x.numel();
  // Use data API of DenseTensor to acquire the data pointer of the template parameter T of the input parameter--x.
  auto x_data = x.data<T>();
  // Use data API of DenseTensor to acquire the data pointer of the template parameter T of the input parameter--y.
  auto y_data = y.data<T>();
  // Use data API of DenseTensor to acquire the data pointer of the template parameter T of the output parameter--out.
  auto out_data = out->data<T>();
  // Get the computing logic done
  for (auto i = 0; i < numel; ++i) {
    out_data[i] = x_data[i] + y_data[i];
  }
}

} // namespace custom_cpu

// In the global namespace, use the macro of registration to register the kernel.
// Register AddKernel of CustomCPU
// Parameters： add - Kernel name
//       CustomCPU - Backend name
//       ALL_LAYOUT - Memory layout
//       custom_cpu::AddKernel - Name of the kernel function
//       int - Data type name
//       int64_t - Data type name
//       float - Data type name
//       double - Data type name
//       phi::dtype::float16 - Data type name
PD_REGISTER_PLUGIN_KERNEL(add,
                          CustomCPU,
                          ALL_LAYOUT,
                          custom_cpu::AddKernel,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16){}

