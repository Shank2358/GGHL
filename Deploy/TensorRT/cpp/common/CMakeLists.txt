include_directories(/home/crescent/TensorRT-8.4.1.5/include)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/cuda/lib64)
# cudnn
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/cuda/lib64)
link_directories(/home/crescent/TensorRT-8.4.1.5/lib)
add_library(common logger.cpp)