if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-Wall -Wextra -pedantic -march=native)
endif()

add_executable(simple_example simple_example.cpp)
target_link_libraries(simple_example PUBLIC bvh)
set_target_properties(simple_example PROPERTIES CXX_STANDARD 20)
add_test(NAME simple_example COMMAND simple_example)

add_executable(serialize serialize.cpp)
target_link_libraries(serialize PUBLIC bvh)
set_target_properties(serialize PROPERTIES CXX_STANDARD 20)
add_test(NAME serialize COMMAND serialize)

add_executable(benchmark benchmark.cpp load_obj.cpp)
target_link_libraries(benchmark PUBLIC bvh)
set_target_properties(benchmark PROPERTIES CXX_STANDARD 20)
add_test(
    NAME benchmark
    COMMAND
        benchmark ${CMAKE_CURRENT_SOURCE_DIR}/scenes/cornell_box.obj
        --eye 0 1 2
        --dir 0 0 -1
        --up 0 1 0)

if (BVH_BUILD_C_API)
    add_executable(c_api_example c_api_example.c load_obj.cpp)
    target_link_libraries(c_api_example PUBLIC bvh_c)
    set_target_properties(c_api_example PROPERTIES CXX_STANDARD 20 C_STANDARD 11)

    add_test(
        NAME c_api_example
        COMMAND
            c_api_example
            ${CMAKE_CURRENT_SOURCE_DIR}/scenes/cornell_box.obj
            --eye 0 1 2
            --dir 0 0 -1
            --up 0 1 0)
endif()
