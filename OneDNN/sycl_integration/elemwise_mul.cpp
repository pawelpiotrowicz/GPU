#include <iostream>
#include <memory>
#include <CL/sycl.hpp>
#include "oneapi/dnnl/dnnl.hpp"
#define show(x) std::cout << x << std::endl;


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




 return 0;
}
