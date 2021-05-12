cmake_minimum_required(VERSION 3.10)
project(spc LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_package(Torch REQUIRED)
find_package(CNPY REQUIRED)

add_library(
    spc STATIC
    spc/SPC.h
    spc/SPC.cu
    spc/spc_math.h
    spc/spc_raytrace_cuda.cpp
    spc/spc_raytrace_cuda_kernel.cu    
)

set_property(TARGET spc PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(
    spc PUBLIC
    cub
)

target_include_directories(
    spc SYSTEM PUBLIC
    /usr/local/cuda/samples/common/inc  
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${TORCH_INCLUDE_DIRS}
    ${CNPY_INCLUDE_DIRS}
)

target_link_libraries(
    spc
    cnpy
)

