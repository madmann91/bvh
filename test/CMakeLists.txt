cmake_minimum_required(VERSION 3.13.4)

include("../cmake/AddBVHTest.cmake")

if (CMAKE_BUILD_TYPE STREQUAL Release)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(bvh_test_options -march=native -mtune=native)
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(bvh_test_options -march=native)
    endif ()
endif ()

add_bvh_test(NAME simple_example     SOURCES simple_example.cpp     OPTIONS ${bvh_test_options})
add_bvh_test(NAME custom_intersector SOURCES custom_intersector.cpp OPTIONS ${bvh_test_options})
add_bvh_test(NAME custom_primitive   SOURCES custom_primitive.cpp   OPTIONS ${bvh_test_options})

add_bvh_test(NAME benchmark
             SOURCES benchmark.cpp
             OPTIONS ${bvh_test_options}
             ARGS
                --eye 0 0.9 2.5
                --dir 0 0.001 -1
                --up 0 1 0
                --fov 60
                ${CMAKE_CURRENT_SOURCE_DIR}/scene/cornell_box.obj)
