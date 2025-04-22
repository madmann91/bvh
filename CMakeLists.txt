cmake_minimum_required(VERSION 3.21)
project(bvh VERSION 2.0.0 LANGUAGES CXX)

add_subdirectory(src/bvh/v2)

if (PROJECT_IS_TOP_LEVEL)
    include(CTest)
    if (BUILD_TESTING)
        add_subdirectory(test)
    endif()
endif()
