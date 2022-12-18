if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-Wall -Wextra -pedantic -march=native)
endif()

add_executable(simple_example simple_example.cpp)
target_link_libraries(simple_example PUBLIC bvh)
set_target_properties(simple_example PROPERTIES CXX_STANDARD 20)

add_executable(serialize serialize.cpp)
target_link_libraries(serialize PUBLIC bvh)
set_target_properties(serialize PROPERTIES CXX_STANDARD 20)

add_executable(benchmark benchmark.cpp load_obj.cpp)
target_link_libraries(benchmark PUBLIC bvh)
set_target_properties(benchmark PROPERTIES CXX_STANDARD 20)

add_test(NAME simple_example COMMAND simple_example)
add_test(NAME serialize COMMAND serialize)
