OSRC=../oneDNN
OINSTALL=../onednn_install

CXX=g++ -I$(OINSTALL)/include -L$(OINSTALL)/lib -I$(OSRC)/examples/


root: main matmul gpu_opencl

main: cnn_training_f32.cpp
	$(CXX) -o cnn_training_f32 cnn_training_f32.cpp -ldnnl -Wl,-rpath,$(OINSTALL)/lib


matmul: cpu_sgemm_and_matmul.cpp 
	$(CXX) -o cpu_sgemm_and_matmul cpu_sgemm_and_matmul.cpp -ldnnl -Wl,-rpath,$(OINSTALL)/lib


gpu_opencl:  gpu_opencl_interop.cpp
	$(CXX) -I/usr/local/cuda-11.6/targets/x86_64-linux/include -L/usr/local/cuda-11.6/targets/x86_64-linux/lib -lOpenCL  -o  gpu_opencl_interop  gpu_opencl_interop.cpp -ldnnl -Wl,-rpath,$(OINSTALL)/lib

