cmake_minimum_required(VERSION 3.9)
project(bvh LANGUAGES CXX VERSION 2.0)

find_package(proto REQUIRED)

add_library(bvh INTERFACE)
target_include_directories(bvh INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(bvh INTERFACE proto::proto)

# Try to find TBB. It might be required when compiling with GCC's libstdc++,
# since parallel algorithms require TBB there.
find_package(TBB QUIET)
if (TBB_FOUND)
    target_link_libraries(bvh INTERFACE TBB::tbb)
    target_compile_definitions(bvh INTERFACE -DBVH_ENABLE_TBB)
    set(BVH_DEPENDS_TBB "find_dependency(TBB)")
else ()
    message(INFO "TBB was not found. Some parallel algorithms might not be available.")
    if (CMAKE_CXX_COMPILER STREQUAL "GNU")
        message(WARNING
            "TBB is required for libstdc++'s implementation of C++ parallel algorithms."
            "If linking errors appear, please install TBB via the package manager on your system.")
    endif ()
endif ()

# Make sure to only build tests when building this project,
# and not when importing it into another one.
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
    if (BUILD_TESTING)
        add_subdirectory(test)
    endif ()
endif ()

include(CMakePackageConfigHelpers)

# Allow using this library by simply calling `add_subdirectory(/path/to/bvh)`
add_library(bvh::bvh ALIAS bvh)

# Allow using this library by locating the config file from the build directory
export(TARGETS bvh NAMESPACE bvh:: FILE bvh-targets.cmake)
configure_package_config_file(
    cmake/bvh-config.cmake.in
    bvh-config.cmake
    INSTALL_DESTINATION .
    INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR})
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/bvh-config-version.cmake COMPATIBILITY SameMajorVersion)

# Allow using this library by installing it
configure_package_config_file(
    cmake/bvh-config.cmake.in
    bvh-config-install.cmake
    INSTALL_DESTINATION lib/cmake)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/bvh-config-install.cmake RENAME bvh-config.cmake DESTINATION lib/cmake)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/bvh-config-version.cmake DESTINATION lib/cmake)
install(DIRECTORY include DESTINATION .)
install(TARGETS bvh EXPORT bvh-targets)
install(EXPORT bvh-targets NAMESPACE bvh:: DESTINATION lib/cmake)
