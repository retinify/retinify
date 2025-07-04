﻿# CMAKE
cmake_minimum_required(VERSION 3.22)
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
set(CMAKE_BUILD_TYPE Release)

# PROJECT
project(retinify)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# C++ SETTINGS
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -fopenmp -pthread")

# CONFIG
install(EXPORT retinifyTargets
    FILE retinifyTargets.cmake
    NAMESPACE retinify::
    DESTINATION lib/cmake/retinify
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_SOURCE_DIR}/cmake/retinifyConfig.cmake.in"
    "${CMAKE_BINARY_DIR}/retinifyConfig.cmake"
    INSTALL_DESTINATION "lib/cmake/retinify"
)

install(FILES "${CMAKE_BINARY_DIR}/retinifyConfig.cmake"
    DESTINATION "lib/cmake/retinify"
)

# BUILD OPTIONS
option(USE_NVIDIA_GPU "Use GPU acceleration if available" ON)
option(BUILD_PYTHON_BINDINGS "Build Python binding" ON)
option(BUILD_SAMPLES "Build samples" ON)
option(BUILD_TESTS "Build tests" ON)

# VERSION
file(READ "${CMAKE_SOURCE_DIR}/VERSION" VERSION_NUMBER)
string(STRIP "${VERSION_NUMBER}" PROJECT_VERSION)
configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/libretinify_version.hpp.in"
    "${CMAKE_BINARY_DIR}/include/retinify/libretinify_version.hpp"
    @ONLY
)

# ONNX MODEL
set(STEREO_MODEL_URL "https://github.com/retinify/retinify-models/releases/download/v0.0.1/model.onnx")
set(STEREO_MODEL_HASH "SHA256=3e50a878345a9792cae3c6f92919186ee1a162855f09c91eabb2f7cb86baeb9d")
set(STEREO_MODEL_PATH "${CMAKE_SOURCE_DIR}/weights/${PROJECT_VERSION}/model.onnx")
file(DOWNLOAD
    ${STEREO_MODEL_URL}
    ${STEREO_MODEL_PATH}
    EXPECTED_HASH ${STEREO_MODEL_HASH}
    SHOW_PROGRESS
    STATUS status
)
set(LIBRETINIFY_ONNX_PATH "${CMAKE_INSTALL_PREFIX}/share/retinify/weights/${PROJECT_VERSION}/model.onnx")
configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/libretinify_onnx.hpp.in"
    "${CMAKE_BINARY_DIR}/include/retinify/libretinify_onnx.hpp"
    @ONLY
)

# SUBDIRECTORIES
add_subdirectory(retinify)
if(BUILD_SAMPLES)
    add_subdirectory(samples)
endif(BUILD_SAMPLES)

# SHARE
install(FILES ${CMAKE_SOURCE_DIR}/LICENSE DESTINATION share/retinify)
install(FILES ${CMAKE_SOURCE_DIR}/NOTICE.md DESTINATION share/retinify)
install(FILES ${CMAKE_SOURCE_DIR}/VERSION DESTINATION share/retinify)
install(FILES ${CMAKE_SOURCE_DIR}/weights/${PROJECT_VERSION}/model.onnx DESTINATION share/retinify/weights/${PROJECT_VERSION})

# CPACK
set(CPACK_GENERATOR "DEB")
set(CPACK_PACKAGE_NAME "libretinify-dev")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
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
message(STATUS "USE_NVIDIA_GPU         : ${USE_NVIDIA_GPU}")
message(STATUS "BUILD_PYTHON_BINDINGS  : ${BUILD_PYTHON_BINDINGS}")
message(STATUS "BUILD_SAMPLES          : ${BUILD_SAMPLES}")
message(STATUS "BUILD_TESTS            : ${BUILD_TESTS}")
message(STATUS "STEREO_MODEL_PATH      : ${STEREO_MODEL_PATH}")
message(STATUS "================================================================================")