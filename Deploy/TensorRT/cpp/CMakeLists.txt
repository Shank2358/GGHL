cmake_minimum_required(VERSION 3.0.0)
project(GGHL)
add_definitions(-std=c++11)
include(CTest)
add_subdirectory(common)
enable_testing()
find_package(CUDA REQUIRED)
add_executable(GGHL main.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
set(OpenCV_DIR /home/crescent/build)



include(CPack)
include_directories(/home/crescent/GGHL/Deploy/TensorRT/common)
link_directories(/home/crescent/GGHL/Deploy/TensorRT/common)
# onnx2tensorrt
include_directories(/home/crescent/onnx-tensorrt)
# cuda
include_directories(/home/crescent/GGHL/Deploy/TensorRT/)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/cuda/lib64)
# cudnn
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/cuda/lib64)
# tensorrt
include_directories(/home/crescent/TensorRT-8.4.1.5/include)
link_directories(/home/crescent/TensorRT-8.4.1.5/lib)
# opencv
find_package(OpenCV REQUIRED)
# common
find_library(ONNX_PATH nvonnxparser /home/crescent/onnx-tensorrt/build)
target_link_libraries( GGHL ${ONNX_PATH})
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( GGHL ${OpenCV_LIBS} )
target_link_libraries( GGHL common)
target_include_directories(GGHL PUBLIC 
                           "${PROJECT_BINARY_DIR}" 
                           "${PROJECT_SOURCE_DIR}/common" 
                           ) 
target_link_libraries(GGHL nvinfer)
target_link_libraries(GGHL cudart)


add_definitions(-O2 -pthread)
