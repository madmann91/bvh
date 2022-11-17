add_library(bvh INTERFACE)

target_include_directories(bvh INTERFACE ../..)
set_target_properties(bvh PROPERTIES CXX_STANDARD 20)
