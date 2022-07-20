add_library(bvh thread_pool.cpp)

target_include_directories(bvh PUBLIC ../..)
set_target_properties(bvh PROPERTIES CXX_STANDARD 20)
