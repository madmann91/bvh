find_package(OpenMP)

add_executable(bvh_test main.cpp load_obj.cpp)

target_link_libraries(bvh_test PUBLIC bvh)
set_target_properties(bvh_test PROPERTIES CXX_STANDARD 20)
if (TRUE)
    target_compile_options(bvh_test PUBLIC -DUSE_BVHLIB=0)
else()
    # TODO: Remove this
    target_include_directories(bvh_test PUBLIC ../../bvh_v1/include)
    target_link_libraries(bvh_test PUBLIC OpenMP::OpenMP_CXX)
    target_compile_options(bvh_test PUBLIC -DUSE_BVHLIB=1)
endif()
