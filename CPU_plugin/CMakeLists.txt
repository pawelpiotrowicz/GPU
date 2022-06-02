cmake_minimum_required(VERSION 3.10)

project(paddle-custom_cpu CXX C)


set(ONEAPI_PATH "/opt/intel/oneapi")
set(ONEAPI_COMPILER_DIR "compiler/2022.1.0")
set(DPCPP_COMPIER_PATH "${ONEAPI_PATH}/${ONEAPI_COMPILER_DIR}/linux/bin/dpcpp")
set(ONEAPI_SYCL_INCLUDE "${ONEAPI_PATH}/${ONEAPI_COMPILER_DIR}/linux/include/sycl/")
set(ONEAPI_SYCL_LIBDIR "${ONEAPI_PATH}/${ONEAPI_COMPILER_DIR}/linux/lib/")
#set(CMAKE_CXX_COMPILER "${DPCPP_COMPIER_PATH}")



####################

set(PADDLE_BUILD_DIR "/backup/work/Paddle/build_gcc_Release")
#set(CND_PYTHON "/opt/conda/lib/python3.7")
set(CND_PYTHON "/home/pablo/miniconda3/envs/cpu/lib/python3.9")

set(ONEDNN_INC "${PADDLE_BUILD_DIR}/third_party/install/mkldnn/include")
set(ONEDNN_LIB "${PADDLE_BUILD_DIR}/third_party/install/mkldnn/lib")

set(PLUGIN_NAME        "paddle_custom_cpu")
set(PLUGIN_VERSION      "0.0.1")

set(PADDLE_PLUGIN_DIR  "${CND_PYTHON}/site-packages/paddle-plugins/")
set(PADDLE_INC_DIR     "${CND_PYTHON}/site-packages/paddle/include/")
set(PADDLE_LIB_DIR     "${CND_PYTHON}/site-packages/paddle/fluid/")

############ Third-party dependencies
set(BOOST_INC_DIR      "${PADDLE_BUILD_DIR}/third_party/boost/src/extern_boost")
set(GFLAGS_INC_DIR     "${PADDLE_BUILD_DIR}/third_party/install/gflags/include")
set(GLOG_INC_DIR       "${PADDLE_BUILD_DIR}/third_party/install/glog/include")
set(THREAD_INC_DIR     "${PADDLE_BUILD_DIR}/third_party/threadpool/src/extern_threadpool")
set(THIRD_PARTY_INC_DIR ${BOOST_INC_DIR} ${GFLAGS_INC_DIR} ${GLOG_INC_DIR} ${THREAD_INC_DIR})

include_directories( ${ONEAPI_SYCL_INCLUDE} ${PADDLE_INC_DIR} ${THIRD_PARTY_INC_DIR} ${ONEDNN_INC})
link_directories(${PADDLE_LIB_DIR} ${ONEDNN_LIB} ${ONEAPI_SYCL_LIBDIR})

add_definitions(-DPADDLE_WITH_CUSTOM_DEVICE)  # for out CustomContext temporarily
add_definitions(-DPADDLE_WITH_CUSTOM_KERNEL)  # for out fluid seperate temporarily

############ Compile plug-ins
add_library(${PLUGIN_NAME} SHARED runtime.cc add_kernel.cc)
target_link_libraries(${PLUGIN_NAME} PRIVATE :core_avx.so libdnnl.so)  # special name

if( "${CMAKE_CXX_COMPILER}" STREQUAL "${DPCPP_COMPIER_PATH}" )
        add_definitions(-DUSEDPCPP)
        set_target_properties(${PLUGIN_NAME} PROPERTIES CXX_STANDARD 17)
endif()

#set_target_properties(${PLUGIN_NAME} PROPERTIES CXX_STANDARD 17)

############ Assembly plug-ins
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

add_custom_command(TARGET ${PLUGIN_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_BINARY_DIR}/python/
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/python/
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/python/paddle-plugins/
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_BINARY_DIR}/lib${PLUGIN_NAME}.so ${CMAKE_CURRENT_BINARY_DIR}/python/paddle-plugins/
    COMMENT "Creating plugin dirrectories------>>>"
)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp
    COMMAND python3 ${CMAKE_CURRENT_BINARY_DIR}/setup.py bdist_wheel
    DEPENDS ${PLUGIN_NAME}
    COMMENT "Packing whl packages------>>>"
)

add_custom_target(python_package ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp)

