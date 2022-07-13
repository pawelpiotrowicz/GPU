
ONEAPI_DIR=/home/pawel/intel/oneapi



DPCPP=$(ONEAPI_DIR)/compiler/latest/linux/bin/dpcpp
#DPCPP=g++ -I/home/pawel/intel/oneapi/compiler/2022.1.0/linux/include/sycl/ -I/home/pawel/intel/oneapi/compiler/2022.1.0/linux/include/ -lsycl

main: elemwise_mul.cpp
	$(DPCPP) -std=c++17 -o elemwise_mul elemwise_mul.cpp -ldnnl -I/home/pawel/intel/oneapi/dnnl/2022.1.0/cpu_dpcpp_gpu_dpcpp/examples/ -g -O0 -lsycl


