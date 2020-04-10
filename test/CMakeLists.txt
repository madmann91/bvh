cmake_minimum_required(VERSION 3.13.4)

include("../cmake/AddBVHTest.cmake")

add_bvh_test(NAME simple_example SOURCES simple_example.cpp)

add_bvh_test(NAME custom_intersector SOURCES custom_intersector.cpp)

add_bvh_test(NAME custom_primitive SOURCES custom_primitive.cpp)

add_bvh_test(NAME benchmark
             SOURCES benchmark.cpp
             ARGS
              --eye 0 0.9 2.5
              --dir 0 0.001 -1
              --up 0 1 0
              --fov 60
              ${CMAKE_CURRENT_SOURCE_DIR}/scene/cornell_box.obj)
