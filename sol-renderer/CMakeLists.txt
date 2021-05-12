# The MIT License (MIT)
#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

cmake_minimum_required(VERSION 3.14)
project(RENDERER LANGUAGES CXX CUDA)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

list(APPEND CMAKE_PREFIX_PATH 
    "include/libtorch" 
    "libtorch"
    "third-party"
    "third-party/libtorch")

# Dependencies for torch renderer
find_package(Torch REQUIRED)

# Dependencies for SPC renderer
find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
find_package(SOLR REQUIRED)
#find_package(CUB REQUIRED)
find_package(SPC REQUIRED)
find_package(CNPY REQUIRED)

add_compile_definitions(THRUST_IGNORE_CUB_VERSION_CHECK)

add_subdirectory("third-party/cnpy")
add_subdirectory("include/spc")

#######################################
# Rendering Library
#######################################

add_library(
    lib_renderer
    nvmath.h
    nvmath.cpp
    RenderCamera.h
    RenderCamera.cpp
)

set_property(TARGET lib_renderer PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(
    lib_renderer PUBLIC
    /usr/local/cuda/samples/common/inc   
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} 
)

#######################################
# SDF Library
#######################################

add_library(
    lib_sdf
    SDF.h
    SDF.cu
)

set_property(TARGET lib_sdf PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(
    lib_sdf PUBLIC
    cub
)

message(STATUS ${SOLR_INCLUDE_DIRS})

target_include_directories(
    lib_sdf SYSTEM PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${TORCH_INCLUDE_DIRS}
    ${SOLR_INCLUDE_DIRS}
    ${CNPY_INCLUDE_DIRS}
)

target_link_libraries(
    cnpy
)

#######################################
# SDF Renderer Executable
#######################################

add_executable(
    sdfRenderer
    sdfRenderer.cpp
    sdfRenderer.cu
)

set_property(TARGET sdfRenderer PROPERTY CXX_STANDARD 14)
set_property(TARGET sdfRenderer PROPERTY CUDA_ARCHITECTURES 70)

target_include_directories(
    sdfRenderer PUBLIC
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${SPC_INCLUDE_DIRS}
)

target_link_libraries(
    sdfRenderer PUBLIC
    lib_renderer
    lib_sdf
    spc
    ${GLUT_LIBRARIES} 
    ${OPENGL_LIBRARIES} 
    ${CUDART_LIBRARY}
    ${TORCH_LIBRARIES}
    ${ZLIB_LIBRARIES}
)
