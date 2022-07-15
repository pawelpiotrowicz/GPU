#include <iostream>
#include <memory>
#include <CL/sycl.hpp>
#include "oneapi/dnnl/dnnl_sycl.hpp"
#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"
#define show(x) std::cout << x << std::endl;
#include <stdio.h>
using CoreType = float;

using namespace dnnl;

using tag = memory::format_tag;
using dt = memory::data_type;

dnnl::engine eng(dnnl::engine::kind::gpu, 0);
//  dnnl::engine eng(dnnl::engine::kind::cpu, 0);
dnnl::stream engine_stream(eng);

//template<class T>
void readFromMem(void* dst, const dnnl::memory& mem)
{

   printf("============TUTAJ=========\n");
    for(size_t i=0;i<5;i++)
    {
       float k = 777777.44;
       k  = *(((float *)mem.get_data_handle()) + i);
                     printf("%f ", k);
                     fflush(stdout);
    }

  // show("ReadFromMem : " << mem.get_desc().get_size() << " tab[0]=" << *((float*)mem.get_data_handle()));
   memcpy(dst, mem.get_data_handle(), mem.get_desc().get_size());
}


void writeToMem(dnnl::memory& mem , const void* src, size_t bytes) {

        memcpy(mem.get_data_handle(),src,bytes);
}
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
    show("Before Free");
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
   // return std::unique_ptr<T[],decltype(&sycl_delete<T>)>( sycl::malloc_device<T>(N,getQ()) , &sycl_delete<T>  );
     //return std::unique_ptr<T[],decltype(&sycl_delete<T>)>( sycl::malloc_shared<T>(N,getQ()) , &sycl_delete<T>  );
     // return std::unique_ptr<T[],decltype(&sycl_delete<T>)>( sycl::malloc_shared<T>(N,getQ()) , &sycl_delete<T>  );
    T *ptr = reinterpret_cast<T *>(sycl::aligned_alloc_device(64, N * sizeof(T), getQ()));
    return std::unique_ptr<T[], decltype(&sycl_delete<T>)>(ptr, &sycl_delete<T>);

    // sycl::aligned_alloc_device(64, size, getQ());


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


template<class T>
struct toDnnType { };

template<>
struct toDnnType<int>
{
 const static dnnl::memory::data_type type = dnnl::memory::data_type::s32;
};

template <>
struct toDnnType<float>
{
   const static dnnl::memory::data_type type = dnnl::memory::data_type::f32;
};



int
main(int argc, char **argv)
{

   auto gpu_mem_x = malloc_gpu<CoreType>(items);
   auto gpu_mem_y = malloc_gpu<CoreType>(items);

   auto gpu_mem_output = malloc_gpu<CoreType>(items);

   // Fill memory with some numbers

   getQ().submit([&](sycl::handler &h)
                 { h.parallel_for(items, [x = gpu_mem_x.get(), y = gpu_mem_y.get()](sycl::id<1> i)
                                  {
            x[i]=i;
            y[i]=i+8; }); });
   getQ().wait();

   //show("GPU_MEM_Y " << gpu_mem_y.get() << " first value= " << *(reinterpret_cast<CoreType *>(gpu_mem_y.get())));
   auto cpu_mem_x = std::make_unique<CoreType[]>(items);

   getQ().submit([dst = cpu_mem_x.get(), src = gpu_mem_y.get(), size = sizeof(CoreType) * items](sycl::handler &h)
                 { h.memcpy(dst, src, size); });
   getQ().wait();

   show("Sycl Return " << cpu_mem_x);

   // ####################################



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

   std::vector<CoreType> x_data(product(x_dims));
   std::vector<CoreType> y_data(product(y_dims));

   auto NextInt = []()
   {
      static CoreType i = 1;
      return i++;
   };
   // Initialize src tensor.
   std::generate(x_data.begin(), x_data.end(), NextInt);
   std::generate(y_data.begin(), y_data.end(), NextInt);

   show("XINPUT =>" << x_data);
   show("YINPUT =>" << y_data);

   // Create src and dst memory descriptors and memory objects.
   auto x_md = memory::desc(x_dims, toDnnType<CoreType>::type, tag::a);
   auto y_md = memory::desc(y_dims, toDnnType<CoreType>::type, tag::a);
   auto xy_md = memory::desc(y_dims, toDnnType<CoreType>::type, tag::a);

   auto x_mem = memory(x_md, eng);

   // auto usm_mem = sycl_interop::make_memory(
   //     xy_md, eng, sycl_interop::memory_kind::usm, gpu_mem_y.get());

 //  auto y_mem = memory(y_md, eng, usm_mem.get());
   // auto y_mem = sycl_interop::make_memory(xy_md, eng, sycl_interop::memory_kind::usm, gpu_mem_y.get());
  // auto y_mem = memory(y_md, eng, gpu_mem_y.get());
   auto y_mem = memory(xy_md, eng, gpu_mem_y.get());

   auto out_xy = memory(xy_md, eng);

   // Write data to memory object's handle.

  // writeToMem(x_mem, x_data.data(), x_data.size() * sizeof(CoreType));
  // writeToMem(y_mem, y_data.data(), y_data.size() * sizeof(CoreType));

    write_to_dnnl_memory(x_data.data(), x_mem);
    write_to_dnnl_memory(y_data.data(), y_mem);

   auto oper_desc = binary::desc(algorithm::binary_mul, x_md, y_md, xy_md);
   auto prim_desc = binary::primitive_desc(oper_desc, eng);
   auto prim = binary(prim_desc);

   std::unordered_map<int, memory> binary_args;
   binary_args.insert({DNNL_ARG_SRC_0, x_mem});
   binary_args.insert({DNNL_ARG_SRC_1, y_mem});
   binary_args.insert({DNNL_ARG_DST, out_xy});

   prim.execute(eng, binary_args);
   engine_stream.wait();

   std::vector<CoreType> cpu_out(N);

   show("Final " << cpu_out);

  // read_from_dnnl_memory(cpu_out.data(), out_xy);
   readFromMem(cpu_out.data(), out_xy);
   show("Final " << cpu_out);

   if (verify_on_cpu(cpu_out, x_data, y_data))
   {
      show("SUCCESS");
   }
   else
   {
      show("FAIL");
      return 0;
   }

   //####################################

   // auto scl_mem = sycl_interop::make_memory(
   //     xy_md, eng, sycl_interop::memory_kind::usm, gpu_mem_y.get());

   //  read_from_dnnl_memory(cpu_out.data(), scl_mem);

   show("SCL_INTEROP_MEMORY : " << cpu_out );





   //auto syl_buf = sycl_interop::get_buffer<CoreType>(scl_mem);

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
