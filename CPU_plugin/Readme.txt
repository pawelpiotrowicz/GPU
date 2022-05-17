>>> import paddle
>>> paddle.device.get_all_custom_device_type()
W0516 11:56:03.942762 109809 pybind.cc:2107] Cannot use get_all_custom_device_type because you have installedCPU/GPU version PaddlePaddle.
If you want to use get_all_custom_device_type, please try to install CustomDevice version PaddlePaddle by: pip install paddlepaddle-core
[]


###########


cmake_cmd="-DPY_VERSION=3.9 -DWITH_GPU=OFF -DWITH_TESTING=ON "




https://www.paddlepaddle.org.cn/documentation/docs/en/develop/dev_guides/custom_device_docs/index_en.html

https://github.com/PaddlePaddle/PaddleCustomDevice

https://github.com/PaddlePaddle/PaddleCustomDevice/blob/develop/backends/npu/README.md

########
