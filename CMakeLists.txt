cmake_minimum_required(VERSION 3.13.4)

project(bvh CXX)

set(CMAKE_CXX_STANDARD 17)

if(CMAKE_BUILD_TYPE STREQUAL Release)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(bvh_cxxflags -march=native -mtune=native -ffast-math)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(bvh_cxxflags -march=native -ffast-math)
  endif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
endif(CMAKE_BUILD_TYPE STREQUAL Release)

add_library(bvh INTERFACE)
target_include_directories(bvh INTERFACE include/)

find_package(OpenMP QUIET)
if (OpenMP_CXX_FOUND)
    target_link_libraries(bvh INTERFACE OpenMP::OpenMP_CXX)
endif()

include(CTest)
add_subdirectory(test)
