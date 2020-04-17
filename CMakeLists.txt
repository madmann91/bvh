cmake_minimum_required(VERSION 3.13)

project(bvh CXX)

set(CMAKE_CXX_STANDARD 17)

add_library(bvh INTERFACE)
target_include_directories(bvh INTERFACE include/)

if (NOT MSVC)
    find_package(OpenMP QUIET)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(bvh INTERFACE ${OpenMP_CXX_LIBRARIES})
        target_include_directories(bvh INTERFACE ${OpenMP_CXX_INCLUDE_DIRS})
        target_link_options(bvh INTERFACE ${OpenMP_CXX_FLAGS})
        target_compile_options(bvh INTERFACE ${OpenMP_CXX_FLAGS})
    endif ()
else ()
    message(STATUS "OpenMP is disabled because MSVC only supports OpenMP 2.0.")
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
