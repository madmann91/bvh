find_package(OpenMP QUIET)
if (OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    link_libraries(${OpenMP_CXX_LIBRARIES})
endif()

add_executable(simple_example simple_example.cpp)
target_include_directories(simple_example PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
add_test(NAME simple_example_test COMMAND simple_example)

add_executable(custom_intersector custom_intersector.cpp)
target_include_directories(custom_intersector PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
add_test(NAME custom_intersector_test COMMAND custom_intersector)

add_executable(custom_primitive custom_primitive.cpp)
target_include_directories(custom_primitive PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
add_test(NAME custom_primitive_test COMMAND custom_primitive)

add_executable(benchmark_vs_fastbvh benchmark_vs_fastbvh.cpp)
target_include_directories(benchmark_vs_fastbvh PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
add_test(NAME benchmark_vs_fastbvh_test COMMAND benchmark_vs_fastbvh)
