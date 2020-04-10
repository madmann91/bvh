# Adds an executable target, links 'bvh' to it, adds compiler flags,
# and makes it a test. Arguments are:
#
#   SOURCES : The source files used to compile the program.
#   OPTIONS : Optional. Compiler options to use when compiling the program.
#   ARGS    : Optional. The arguments to pass to the test invocation.
#   NAME    : The name of the test program. The prefix 'bvh_'
#             is prepended to this when declaring the CMake target.
function (add_bvh_test)
    set(single_value_args NAME)
    set(multi_value_args SOURCES ARGS OPTIONS)

    cmake_parse_arguments(bvh_test "" "${single_value_args}" "${multi_value_args}" "${ARGN}")

    if ("${bvh_test_NAME}" STREQUAL "")
        message(FATAL_ERROR "Missing 'NAME' parameter. Please specify test name.")
    endif ()

    if ("${bvh_test_SOURCES}" STREQUAL "")
        message(FATAL_ERROR "Missing 'SOURCES' parameter. Please specify what sources to compile.")
    endif ()

    # CMake doesn't scope target names.
    # When this project is added as a sub directory
    # to another project, adding the 'bvh_' prefix
    # prevents target names from colliding.
    set(target_name bvh_${bvh_test_NAME})

    add_executable(${target_name} ${bvh_test_SOURCES})
    target_link_libraries(${target_name} PRIVATE bvh)

    # Here we tell CMake that we want the filename to
    # be called ${bvh_test_NAME} (with .exe on Windows) and
    # that we want it to appear in the project binary directory
    # so that the user can easily run it.
    set_target_properties(${target_name} PROPERTIES
        OUTPUT_NAME ${bvh_test_NAME}
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

    target_compile_options(${target_name} PRIVATE ${bvh_test_OPTIONS})

    add_test(NAME ${target_name}
        COMMAND $<TARGET_FILE:${target_name}> ${bvh_test_ARGS}
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
endfunction ()
