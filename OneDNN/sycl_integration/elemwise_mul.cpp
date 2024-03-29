#include <iostream>
#include <memory>
#include <CL/sycl.hpp>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"
#define show(x) std::cout << x << std::endl;

template<class T>
bool verify_on_cpu(T& out, T& in1, T& in2)
{
      if( in1.size()!=in2.size() || out.size() != in1.size() )
      {
         show("error: incorrect dim");
         return false;
      }

      for(size_t i=0;i<out.size();i++)
      {
           auto cmp = in1[i] * in2[i];
           if(cmp != out[i])
           {
            return false;
           }
      }

      return true;
}


static sycl::queue& getQ() {
   static sycl::queue q{sycl::gpu_selector{}};
   return q;
}

template<class T>
void sycl_delete(T *v)
{
    sycl::free(v,getQ());
    show("FreeGPU memory");
}


int items = 5;

template<class T>
std::ostream& operator<<(std::ostream& o, const std::unique_ptr<T[]>& ptr) {

   for(size_t i=0;i<items;++i) {
    std::cout << ptr[i]  << ",";
   }
    std::cout << std::endl;

  return o;
}

template<class T>
auto malloc_gpu(int N=64) {
    show("GPU allocate " << sizeof(T)*N << " bytes");
    return std::unique_ptr<T[],decltype(&sycl_delete<T>)>( sycl::malloc_device<T>(N,getQ()) , &sycl_delete<T>  );
}

template <class T>
std::ostream &operator<<(std::ostream &o, const std::vector<T> &ptr)
{

   for (size_t i = 0; i < ptr.size(); ++i)
   {
      std::cout << ptr[i] << ",";
   }
   std::cout << std::endl;

   return o;
}

int main(int argc, char **argv) {


auto gpu_mem_x = malloc_gpu<int>(items);
auto gpu_mem_y = malloc_gpu<int>(items);

// Fill memory

getQ().submit([&](sycl::handler& h){
   h.parallel_for(items,[x=gpu_mem_x.get(), y=gpu_mem_y.get()](sycl::id<1> i){
            x[i]=i;
            y[i]=i+8;
 });
});
getQ().wait();

auto cpu_mem_x = std::make_unique<int[]>(items);

getQ().submit([dst=cpu_mem_x.get(),src=gpu_mem_y.get(),size=sizeof(int)*items](sycl::handler& h){
   h.memcpy(dst,src,size);
});
getQ().wait();

show( cpu_mem_x );

// ####################################
using namespace dnnl;

using tag = memory::format_tag;
using dt = memory::data_type;

// dnnl::engine engine(dnnl::engine::kind::gpu, 0);
 dnnl::engine engine(dnnl::engine::kind::cpu, 0);

dnnl::stream engine_stream(engine);
const memory::dim N = 5;
// const memory::dim N = 3, // batch size
//             IC = 3, // channels
//             IH = 227, // tensor height
//             IW = 227; // tensor width



    // Source (src) and destination (dst) tensors dimensions.
//  memory::dims src_dims = {N, IC, IH, IW};
//  memory::dims dst_dims = {N, IC, IH, IW};

  memory::dims x_dims = {N};
  memory::dims y_dims = {N};

std::vector<int> x_data(product(x_dims));
std::vector<int> y_data(product(y_dims));

auto NextInt = []()
{
   static int i = 1;
   return i++;
};
// Initialize src tensor.
std::generate(x_data.begin(), x_data.end(), NextInt);
std::generate(y_data.begin(), y_data.end(), NextInt);

show("XINPUT =>" << x_data);
show("YINPUT =>" << y_data);

// Create src and dst memory descriptors and memory objects.
auto x_md = memory::desc(x_dims, dt::s32, tag::a);
auto y_md = memory::desc(y_dims, dt::s32, tag::a);
auto xy_md = memory::desc(y_dims, dt::s32, tag::a);

auto x_mem = memory(x_md, engine);
auto y_mem = memory(y_md, engine);
auto out_xy = memory(xy_md, engine);

// Write data to memory object's handle.
write_to_dnnl_memory(x_data.data(), x_mem);
write_to_dnnl_memory(y_data.data(), y_mem);

auto oper_desc = binary::desc(algorithm::binary_mul, x_md, y_md, xy_md);
auto prim_desc = binary::primitive_desc(oper_desc,engine);
auto prim = binary(prim_desc);

std::unordered_map<int, memory> binary_args;
binary_args.insert({DNNL_ARG_SRC_0, x_mem});
binary_args.insert({DNNL_ARG_SRC_1, y_mem});
binary_args.insert({DNNL_ARG_DST, out_xy});

prim.execute(engine, binary_args);
engine_stream.wait();

std::vector<int> cpu_out(N);

show("Final " << cpu_out);

read_from_dnnl_memory(cpu_out.data(),out_xy);

show("Final " << cpu_out);

if( verify_on_cpu(cpu_out, x_data, y_data) )
{
   show("SUCCESS");
} else {
   show("FAIL");
}


    // Create operation descriptor.

    //  auto eltwise_d = eltwise_forward::desc(prop_kind::forward_training,
    //          algorithm::eltwise_relu, src_md, 0.f, 0.f);

    //  // Create primitive descriptor.
    //  auto eltwise_pd = eltwise_forward::primitive_desc(eltwise_d, engine);

    //  // Create the primitive.
    //  auto eltwise_prim = eltwise_forward(eltwise_pd);

    //  // Primitive arguments.
    //  std::unordered_map<int, memory> eltwise_args;
    //  eltwise_args.insert({DNNL_ARG_SRC, src_mem});
    //  eltwise_args.insert({DNNL_ARG_DST, dst_mem});

    //  // Primitive execution: element-wise (ReLU).
    //  eltwise_prim.execute(engine_stream, eltwise_args);

    // Wait for the computation to finalize.
    // engine_stream.wait();

    return 0;
}
