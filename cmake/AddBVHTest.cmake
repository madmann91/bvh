# Adds an executable target, links 'bvh',
# adds compiler flags, and makes it a test.
# It takes several types of arguments.
#
#   SOURCES : The source files used to compile the program.
#   ARGS    : The arguments to pass to the test invocation.
#   NAME    : The name of the test program. The prefix 'bvh_'
#             is prepended to this when declaring the CMake target.
function(add_bvh_test)

  set(single_value_args NAME)

  set(multi_value_args SOURCES ARGS)

  cmake_parse_arguments(test "" "${single_value_args}" "${multi_value_args}" "${ARGN}")

  if("${test_NAME}" STREQUAL "")
    message(FATAL_ERROR "Missing 'NAME' parameter. Please specify test name.")
  endif("${test_NAME}" STREQUAL "")

  if("${test_SOURCES}" STREQUAL "")
    message(FATAL_ERROR "Missing 'SOURCES' parameter. Please specify what sources to compile.")
  endif("${test_SOURCES}" STREQUAL "")

  # CMake doesn't scope target names.
  # When this project is added as a sub directory
  # to another project, adding the 'bvh_' prefix
  # prevents target names from colliding.
  set(target_name bvh_${test_NAME})

  add_executable(${target_name} ${test_SOURCES})

  target_link_libraries(${target_name} PRIVATE bvh)

  # Here we tell CMake that we want the filename to
  # be called ${test_name} (with .exe on Windows) and
  # that we want it to appear in the project binary directory
  # so that the user can easily run it.
  set_target_properties(${target_name} PROPERTIES
    OUTPUT_NAME ${test_NAME}
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

  # 'bvh_cxxflags' is set in the parent CMakeLists.txt
  target_compile_options(${target_name} PRIVATE ${bvh_cxxflags})

  add_test(NAME ${target_name}
           COMMAND $<TARGET_FILE:${target_name}> ${test_ARGS}
           WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

endfunction(add_bvh_test test_name)
