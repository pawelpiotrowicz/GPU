
ONEAPI_DIR=/home/pawel/intel/oneapi


OSRC=../oneDNN
OINSTALL=../onednn_install


DPCPP=$(ONEAPI_DIR)/compiler/latest/linux/bin/dpcpp
#DPCPP=g++ -I/home/pawel/intel/oneapi/compiler/2022.1.0/linux/include/sycl/ -I/home/pawel/intel/oneapi/compiler/2022.1.0/linux/include/ -lsycl

main: elemwise_mul.cpp
	$(DPCPP) -std=c++17 -o elemwise_mul elemwise_mul.cpp  -I/home/pawel/intel/oneapi/dnnl/2022.1.0/cpu_dpcpp_gpu_dpcpp/examples/ -g -O0 -lsycl -L/home/pawel/intel/oneapi/dnnl/2022.1.0/cpu_dpcpp_gpu_dpcpp/lib/ -ldnnl


new: main.cpp
	$(DPCPP) -std=c++17 -o main2 main.cpp  -fsycl -I/home/pawel/intel/oneapi/dnnl/2022.1.0/cpu_dpcpp_gpu_dpcpp/examples/ -g -O0 -lsycl -L/home/pawel/intel/oneapi/dnnl/2022.1.0/cpu_dpcpp_gpu_dpcpp/lib/ -ldnnl

custom: main.cpp
	/home/pawel/intel/oneapi/compiler/2022.1.0/linux/bin/icpx -fsycl  -I$(OINSTALL)/include -L$(OINSTALL)/lib -I$(OSRC)/examples/ -o custom main.cpp -ldnnl -Wl,-rpath,$(OINSTALL)/lib
