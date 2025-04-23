include(GNUInstallDirs)
set(BVH_TARGET_NAME bvh)
set(BVH_INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_INCLUDEDIR}")
if(NOT DEFINED BVH_CONFIG_INSTALL_DIR )
  set(BVH_CONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}" CACHE INTERNAL "")
endif()
set(BVH_TARGETS_EXPORT_NAME "${PROJECT_NAME}Targets")
set(BVH_CMAKE_CONFIG_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(BVH_CMAKE_PROJECT_TARGETS_FILE "${BVH_CMAKE_CONFIG_DIR}/${PROJECT_NAME}Targets.cmake")

# —————  Package configuration  —————
include(CMakePackageConfigHelpers)

# Configure the config file that is to be installed
configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
  "${BVH_CMAKE_CONFIG_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION "${BVH_CONFIG_INSTALL_DIR}"
  PATH_VARS BVH_INCLUDE_INSTALL_DIR
)

# Generate the version file
write_basic_package_version_file(
  "${BVH_CMAKE_CONFIG_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

# —————  End package configuration  —————

add_library(${BVH_TARGET_NAME} INTERFACE)
find_package(Threads)
if (Threads_FOUND)
    # Link with the threading library of the system, which may
    # be required by standard header <thread> on some systems
    target_link_libraries(bvh INTERFACE Threads::Threads)
endif()

target_include_directories(${BVH_TARGET_NAME} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
  $<INSTALL_INTERFACE:include>
  )
set_target_properties(${BVH_TARGET_NAME} PROPERTIES CXX_STANDARD 20)


install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../bvh
  DESTINATION ${BVH_INCLUDE_INSTALL_DIR}
  )
export(
  TARGETS ${BVH_TARGET_NAME}
  NAMESPACE ${PROJECT_NAME}::
  FILE ${BVH_CMAKE_PROJECT_TARGETS_FILE}
  )
install(
  TARGETS ${BVH_TARGET_NAME}
  EXPORT ${BVH_TARGETS_EXPORT_NAME}
  INCLUDES DESTINATION ${BVH_INCLUDE_INSTALL_DIR}
  )
install(
  EXPORT ${BVH_TARGETS_EXPORT_NAME}
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${BVH_CONFIG_INSTALL_DIR}
  )
install(
  FILES
    "${BVH_CMAKE_CONFIG_DIR}/${PROJECT_NAME}Config.cmake"
    "${BVH_CMAKE_CONFIG_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  DESTINATION ${BVH_CONFIG_INSTALL_DIR}
)
target_include_directories(bvh INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>)

set_target_properties(bvh PROPERTIES CXX_STANDARD 20)

if (BVH_BUILD_C_API)
    add_subdirectory(c_api)
endif()
