cmake_minimum_required(VERSION 3.9)
project(bvh CXX)

add_subdirectory(contrib)

add_library(bvh INTERFACE)
target_include_directories(bvh INTERFACE include/)
target_link_libraries(bvh INTERFACE proto)

find_package(TBB QUIET)
if (TBB_FOUND)
    target_link_libraries(bvh INTERFACE TBB::tbb)
    target_compile_definitions(bvh INTERFACE -DBVH_ENABLE_TBB)
else ()
    message(INFO "TBB was not found. Some parallel algorithms might not be available.")
    if (CMAKE_CXX_COMPILER STREQUAL "GNU")
        message(WARNING
            "TBB is required for libstdc++'s implementation of C++ parallel algorithms."
            "If linking errors appear, please install TBB via the package manager on your system.")
    endif ()
endif ()

# Make sure to only build tests when building this project,
# and not when importing it into another one.
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
    if (BUILD_TESTING)
        add_subdirectory(test)
    endif ()
endif ()
