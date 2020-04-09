cmake_minimum_required(VERSION 3.14)
project(bvh)

set(CMAKE_CXX_STANDARD 17)

add_library(bvh INTERFACE)
target_include_directories(bvh INTERFACE include/)

find_package(OpenMP QUIET)
if (OpenMP_CXX_FOUND)
    target_link_libraries(bvh INTERFACE OpenMP::OpenMP_CXX)
endif()

include(CTest)
add_subdirectory(test)
