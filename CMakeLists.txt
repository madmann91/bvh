cmake_minimum_required(VERSION 3.9)
project(bvh)

add_subdirectory(src/bvh/v2)

include(CTest)
if (BUILD_TESTING)
    add_subdirectory(test)
endif()
