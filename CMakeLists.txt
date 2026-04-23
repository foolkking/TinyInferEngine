cmake_minimum_required(VERSION 3.10)
project(TinyInferEngine)

set(CMAKE_CXX_STANDARD 11)

# 指定头文件搜索路径
include_directories(include)

# 编译核心库
add_library(core_lib src/tensor.cpp src/relu.cpp src/linear.cpp src/model.cpp src/conv2d.cpp 
            src/maxpool2d.cpp src/flatten.cpp src/sgd.cpp src/loss.cpp src/silu.cpp)

# # 编译测试执行文件
# add_executable(test_tensor tests/test_tensor.cpp)
# target_link_libraries(test_tensor core_lib)
# 编译最终的可执行文件 main.cpp
add_executable(infer_engine src/main.cpp)
target_link_libraries(infer_engine core_lib)

#编译train_minist.cpp
add_executable(train_minist train_minist.cpp)
target_link_libraries(train_minist core_lib)

# OpenMP 支持,多线程加速卷积计算
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()