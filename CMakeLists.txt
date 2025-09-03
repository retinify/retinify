# CMAKE
cmake_minimum_required(VERSION 3.14)
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
set(CMAKE_BUILD_TYPE Release)

# PROJECT
project(retinify)
set(CMAKE_INSTALL_PREFIX "/usr")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# C++ SETTINGS
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# VERSION
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" VERSION_NUMBER)
string(STRIP "${VERSION_NUMBER}" PROJECT_VERSION)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/retinify_version.hpp.in"
    "${CMAKE_BINARY_DIR}/include/retinify/retinify_version.hpp"
    @ONLY
)

# CONFIG
install(EXPORT retinifyTargets
    FILE retinifyTargets.cmake
    NAMESPACE retinify::
    DESTINATION lib/cmake/retinify
)
include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/retinifyConfig.cmake.in"
    "${CMAKE_BINARY_DIR}/retinifyConfig.cmake"
    INSTALL_DESTINATION "lib/cmake/retinify"
)
install(FILES "${CMAKE_BINARY_DIR}/retinifyConfig.cmake"
    DESTINATION "lib/cmake/retinify"
)

# BUILD OPTIONS
option(BUILD_WITH_TENSORRT "Build with TensorRT" ON)
option(BUILD_SAMPLES "Build samples" ON)
option(BUILD_TESTS "Build tests" ON)

# ALL IN ONE HEADER
set(LIBRETINIFY_OPTION_INCLUDES "")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/retinify.hpp.in"
    "${CMAKE_BINARY_DIR}/include/retinify/retinify.hpp"
    @ONLY
)
install(
    FILES "${CMAKE_BINARY_DIR}/include/retinify/retinify.hpp"
    DESTINATION include/retinify
)

# ONNX MODEL
set(STEREO_MODEL_URL "https://github.com/retinify/retinify-models/releases/download/v${PROJECT_VERSION}/model.onnx")
set(STEREO_MODEL_HASH "SHA256=1b5de6f1d30187a7354750f20b65e7ecf507c46d288589a5833f75e4371242f6")
set(STEREO_MODEL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/weights/${PROJECT_VERSION}/model.onnx")
file(DOWNLOAD
    ${STEREO_MODEL_URL}
    ${STEREO_MODEL_PATH}
    EXPECTED_HASH ${STEREO_MODEL_HASH}
    SHOW_PROGRESS
    STATUS status
)
set(LIBRETINIFY_ONNX_PATH "${CMAKE_INSTALL_PREFIX}/share/retinify/weights/${PROJECT_VERSION}/model.onnx")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/retinify_onnx.hpp.in"
    "${CMAKE_BINARY_DIR}/include/retinify/retinify_onnx.hpp"
    @ONLY
)

# SUBDIRECTORIES
add_subdirectory(retinify)
add_subdirectory(3rdparty)
if(BUILD_SAMPLES)
    add_subdirectory(samples)
endif(BUILD_SAMPLES)

# SHARE
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE DESTINATION share/retinify)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/NOTICE.md DESTINATION share/retinify)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/VERSION DESTINATION share/retinify)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/weights/${PROJECT_VERSION}/model.onnx DESTINATION share/retinify/weights/${PROJECT_VERSION})

# CPACK
set(CPACK_GENERATOR "DEB")
set(CPACK_PACKAGE_NAME "libretinify-dev")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
  set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE amd64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE arm64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(armv6l|armv7l)$")
  set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE armhf)
else()
  set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "${CMAKE_SYSTEM_PROCESSOR}")
endif()

set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Sensui Yagi <contact@retinify.ai>")
set(CPACK_PACKAGE_CONTACT "contact@retinify.ai")
set(CPACK_PACKAGE_DESCRIPTION "Real-Time AI Stereo Vision Library")
set(CPACK_DEBIAN_PACKAGE_SECTION "libs")
set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://retinify.ai")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
include(CPack)

# MESSAGES
message(STATUS "================================================================================")
message(STATUS "                           RETINIFY Build Configuration                         ")
message(STATUS "================================================================================")
message(STATUS "PROJECT_VERSION        : ${PROJECT_VERSION}")
message(STATUS "BUILD_WITH_TENSORRT    : ${BUILD_WITH_TENSORRT}")
message(STATUS "BUILD_SAMPLES          : ${BUILD_SAMPLES}")
message(STATUS "BUILD_TESTS            : ${BUILD_TESTS}")
message(STATUS "STEREO_MODEL_PATH      : ${STEREO_MODEL_PATH}")
message(STATUS "================================================================================")