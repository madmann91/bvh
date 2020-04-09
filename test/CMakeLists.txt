
add_executable(simple_example simple_example.cpp)
target_link_libraries(simple_example PUBLIC bvh)
add_test(NAME simple_example_test COMMAND simple_example)

add_executable(custom_intersector custom_intersector.cpp)
target_link_libraries(custom_intersector PUBLIC bvh)
add_test(NAME custom_intersector_test COMMAND custom_intersector)

add_executable(custom_primitive custom_primitive.cpp)
target_link_libraries(custom_primitive PUBLIC bvh)
add_test(NAME custom_primitive_test COMMAND custom_primitive)

add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark PUBLIC bvh)
add_test(NAME benchmark_test COMMAND benchmark --eye 0 0.9 2.5 --dir 0 0.001 -1 --up 0 1 0 --fov 60 ${CMAKE_CURRENT_SOURCE_DIR}/scene/cornell_box.obj)
