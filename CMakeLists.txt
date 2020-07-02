cmake_minimum_required(VERSION 3.9)

project(bvh CXX)

set(CMAKE_CXX_STANDARD 17)

add_library(bvh INTERFACE)
target_include_directories(bvh INTERFACE include/)

if (NOT MSVC)
    find_package(OpenMP QUIET)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(bvh INTERFACE OpenMP::OpenMP_CXX)
    endif ()
else ()
    message(STATUS "OpenMP is disabled because MSVC only supports OpenMP 2.0.")
endif ()

# Make sure to only build tests when building this project,
# and not when importing it into another one.
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
    if (BUILD_TESTING)
        add_subdirectory(test)
    endif ()
endif ()
