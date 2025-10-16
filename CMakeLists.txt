# CMAKE
cmake_minimum_required(VERSION 3.22)

# EULA
set(RETINIFY_EULA_DIR "${CMAKE_SOURCE_DIR}/3rdparty/retinify-eula")
set(RETINIFY_EULA_FILE "${RETINIFY_EULA_DIR}/EULA.md")
find_program(GIT_EXECUTABLE git REQUIRED)

execute_process(
  COMMAND "${GIT_EXECUTABLE}" submodule update --init --recursive -- 3rdparty/retinify-eula
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMAND_ERROR_IS_FATAL ANY
)
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${RETINIFY_EULA_DIR}" fetch origin main COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${RETINIFY_EULA_DIR}" reset --hard origin/main COMMAND_ERROR_IS_FATAL ANY)

if(NOT EXISTS "${RETINIFY_EULA_FILE}")
  message(FATAL_ERROR "EULA not found: ${RETINIFY_EULA_FILE}")
endif()

# BUILD OPTIONS
option(BUILD_WITH_TENSORRT "Build with TensorRT" ON)
option(BUILD_SAMPLES "Build samples" ON)
option(BUILD_TESTS "Build tests" ON)

# PROJECT
if(BUILD_WITH_TENSORRT)
    message(STATUS "BUILD_WITH_TENSORRT=ON -> enabling CXX and CUDA languages")
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
    project(retinify LANGUAGES CXX CUDA)
else()
    project(retinify LANGUAGES CXX)
endif()
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_INSTALL_PREFIX "/usr")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# VERSION
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" VERSION_NUMBER)
string(STRIP "${VERSION_NUMBER}" PROJECT_VERSION)
string(REPLACE "." ";" VERSION_LIST "${PROJECT_VERSION}")
list(GET VERSION_LIST 0 PROJECT_VERSION_MAJOR)
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
set(STEREO_MODEL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/weights/stereo-matching.onnx")
if(NOT EXISTS "${STEREO_MODEL_PATH}")
  message(FATAL_ERROR "Stereo model file not found: ${STEREO_MODEL_PATH}")
endif()

set(STEREO_MATCHING_ONNX_PATH "${CMAKE_INSTALL_PREFIX}/share/retinify/weights/${PROJECT_VERSION}/stereo-matching.onnx")
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
install(FILES ${RETINIFY_EULA_FILE} DESTINATION share/retinify)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/NOTICE.md DESTINATION share/retinify)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/VERSION DESTINATION share/retinify)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/weights/stereo-matching.onnx DESTINATION share/retinify/weights/${PROJECT_VERSION})

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
set(CPACK_RESOURCE_FILE_LICENSE "${RETINIFY_EULA_FILE}")
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
message(STATUS "EULA_PATH              : ${RETINIFY_EULA_FILE}")
message(STATUS "================================================================================")
