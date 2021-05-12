cmake_minimum_required(VERSION 3.10)
project(cnpy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ZLIB REQUIRED)

#######################################
# Core Library
#######################################

add_library(
    cnpy STATIC
    cnpy/cnpy.cpp
)
target_link_libraries(cnpy ${ZLIB_LIBRARIES})
target_include_directories(cnpy PUBLIC ${ZLIB_INCLUDE_DIRS})

