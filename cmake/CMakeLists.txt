# retinifyConfig
install(EXPORT retinifyTargets
    FILE retinifyTargets.cmake
    NAMESPACE retinify::
    DESTINATION lib/cmake/retinify
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/retinifyConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/retinifyConfig.cmake"
    INSTALL_DESTINATION "lib/cmake/retinify"
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/retinifyConfig.cmake"
    DESTINATION "lib/cmake/retinify"
)