cmake_minimum_required(VERSION 3.13.4)

project(bvh CXX)

set(CMAKE_CXX_STANDARD 17)

add_library(bvh INTERFACE)
target_include_directories(bvh INTERFACE include/)

find_package(OpenMP QUIET)
if (OpenMP_CXX_FOUND)
    target_link_libraries(bvh INTERFACE OpenMP::OpenMP_CXX)
endif ()

include(cmake/FindRestrictKeyword.cmake)
find_restrict_keyword(restrict_keyword)
target_compile_definitions(bvh INTERFACE -Dbvh__restrict__=${restrict_keyword})

# Make sure to only build tests when building this project,
# and not when importing it into another one.
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
    if (BUILD_TESTING)
        add_subdirectory(test)
    endif ()
endif ()
