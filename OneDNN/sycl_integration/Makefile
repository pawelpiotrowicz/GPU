
ONEAPI_DIR=/home/pawel/intel/oneapi



DPCPP=$(ONEAPI_DIR)/compiler/latest/linux/bin/dpcpp


main: elemwise_mul.cpp
	$(DPCPP) -std=c++17 -o elemwise_mul elemwise_mul.cpp


