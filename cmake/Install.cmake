install(
    TARGETS bvh_c
    EXPORT bvh_exports
    RUNTIME DESTINATION bin/
    LIBRARY DESTINATION lib/
    ARCHIVE DESTINATION lib/
    INCLUDES DESTINATION include/)
install(
    EXPORT bvh_exports
    FILE bvh-targets.cmake
    NAMESPACE bvh::v2
    DESTINATION lib/cmake/bvh/v2/)

include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/bvh-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/bvh-config.cmake"
    INSTALL_DESTINATION lib/cmake/bvh/v2/)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/bvh-config-version.cmake"
    COMPATIBILITY AnyNewerVersion)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/bvh-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/bvh-config-version.cmake"
    DESTINATION lib/cmake/bvh/v2/)
