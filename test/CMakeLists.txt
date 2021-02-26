find_package(ImageMagick COMPONENTS compare QUIET)
if (NOT ImageMagick_compare_FOUND)
    message(WARNING "Cannot find ImageMagick's compare tool. Images will not be compared to the reference.")
endif ()

# Adds an executable target, and links 'bvh' to it.
#
#   SOURCES : The source files used to compile the program.
#   NAME    : The name of the test program.
function (add_bvh_test_executable)
    set(single_value_args NAME)
    set(multi_value_args SOURCES)
    cmake_parse_arguments(bvh_test_executable "" "${single_value_args}" "${multi_value_args}" "${ARGN}")

    add_executable(${bvh_test_executable_NAME} ${bvh_test_executable_SOURCES})
    target_link_libraries(${bvh_test_executable_NAME} PRIVATE bvh)
endfunction ()

# Adds a test for the benchmark tool that compares the
# output to a reference picture. Arguments are:
# 
#   NAME      : The test name (e.g. benchmark_sweep_sah_cornell).
#   OPTIONS   : The options to pass to the benchmark tool.
#   REFERENCE : The reference image file.
function (add_benchmark_test)
    set(single_value_args NAME REFERENCE)
    set(multi_value_args OPTIONS)
    cmake_parse_arguments(benchmark_test "" "${single_value_args}" "${multi_value_args}" "${ARGN}")

    add_test(
        NAME ${benchmark_test_NAME}
        COMMAND
            ${CMAKE_COMMAND}
            -DImageMagick_compare_EXECUTABLE=${ImageMagick_compare_EXECUTABLE}
            -Dbenchmark_EXECUTABLE=$<TARGET_FILE:benchmark>
            "-Dbenchmark_OPTIONS=${benchmark_test_OPTIONS};-o;${CMAKE_CURRENT_BINARY_DIR}/render_${benchmark_test_NAME}.ppm"
            -Dbenchmark_REFERENCE=${benchmark_test_REFERENCE}
            -Dbenchmark_OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/render_${benchmark_test_NAME}.ppm
            -Dbenchmark_DIFFERENCE_RESULT=${CMAKE_CURRENT_BINARY_DIR}/difference_${benchmark_test_NAME}.png
            -P ${PROJECT_SOURCE_DIR}/cmake/RunBenchmarkTest.cmake
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endfunction ()

add_bvh_test_executable(NAME simple_example     SOURCES simple_example.cpp)
add_bvh_test_executable(NAME custom_intersector SOURCES custom_intersector.cpp)
add_bvh_test_executable(NAME custom_primitive   SOURCES custom_primitive.cpp)
add_bvh_test_executable(NAME node_intersectors  SOURCES node_intersectors.cpp)
add_bvh_test_executable(NAME refit_bvh          SOURCES refit_bvh.cpp)
add_bvh_test_executable(NAME benchmark          SOURCES benchmark.cpp)

add_test(NAME simple_example     COMMAND simple_example)
add_test(NAME custom_intersector COMMAND custom_intersector)
add_test(NAME custom_primitive   COMMAND custom_primitive)
add_test(NAME node_intersectors  COMMAND node_intersectors)
add_test(NAME refit_bvh          COMMAND refit_bvh)

set(cornell_scene_options 
    --eye 0 0.9 2.5
    --dir 0 0.001 -1
    --up 0 1 0
    --fov 60
    ${CMAKE_CURRENT_SOURCE_DIR}/scene/cornell_box.obj)
set(cornell_scene_reference ${CMAKE_CURRENT_SOURCE_DIR}/scene/cornell_box_reference.png)

foreach (build_options_as_string
    "--builder sweep_sah"
    "--builder binned_sah"
    "--builder binned_sah --permute"
    "--builder spatial_split"
    "--builder locally_ordered_clustering"
    "--builder locally_ordered_clustering --collapse-leaves"
    "--builder linear"
    "--builder linear --pre-split 30"
    "--builder sweep_sah --parallel-reinsertion"
    "--builder sweep_sah --optimize-layout")
    string(MAKE_C_IDENTIFIER ${build_options_as_string} benchmark_test_name)
    string(REPLACE " " ";" build_options ${build_options_as_string})
    add_benchmark_test(
        NAME ${benchmark_test_name}
        OPTIONS ${cornell_scene_options} ${build_options}
        REFERENCE ${cornell_scene_reference})
endforeach ()
