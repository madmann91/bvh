cmake_minimum_required(VERSION 3.9)

project(bvh CXX)

add_subdirectory(contrib)

add_library(bvh INTERFACE)
target_include_directories(bvh INTERFACE include/)
target_link_libraries(bvh INTERFACE proto)
set_target_properties(bvh PROPERTIES CXX_STANDARD 20)

# Make sure to only build tests when building this project,
# and not when importing it into another one.
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
    if (BUILD_TESTING)
        add_subdirectory(test)
    endif ()
endif ()
