# CMAKE
cmake_minimum_required(VERSION 3.22)

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
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/retinifyVersion.hpp.in"
    "${CMAKE_BINARY_DIR}/include/retinify/retinifyVersion.hpp"
    @ONLY
)

# CONFIG
install(EXPORT retinifyTargets
    FILE retinifyTargets.cmake
    NAMESPACE retinify::
    DESTINATION lib/cmake/retinify
    COMPONENT dev
)
include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/retinifyConfig.cmake.in"
    "${CMAKE_BINARY_DIR}/retinifyConfig.cmake"
    INSTALL_DESTINATION "lib/cmake/retinify"
)
install(FILES "${CMAKE_BINARY_DIR}/retinifyConfig.cmake"
    DESTINATION "lib/cmake/retinify"
    COMPONENT dev
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
    COMPONENT dev
)
install(
    FILES
        "${CMAKE_BINARY_DIR}/include/retinify/retinifyVersion.hpp"
        "${CMAKE_BINARY_DIR}/include/retinify/retinifyModels.hpp"
    DESTINATION include/retinify
    COMPONENT dev
)

# DOWNLOAD MODELS
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/retinifyDownloadModels.cmake")

set(RETINIFY_MODELS_OWNER "retinify")
set(RETINIFY_MODELS_REPO  "retinify-models")
set(RETINIFY_MODELS_TAG   "v${PROJECT_VERSION}")
set(GITHUB_RELEASE_API_URL "https://api.github.com/repos/${RETINIFY_MODELS_OWNER}/${RETINIFY_MODELS_REPO}/releases/tags/${RETINIFY_MODELS_TAG}")
set(RETINIFY_DOWNLOADED_MODELS_DIR "${CMAKE_BINARY_DIR}/weights/${PROJECT_VERSION}")

retinify_download_models(
    "${GITHUB_RELEASE_API_URL}"
    "${RETINIFY_DOWNLOADED_MODELS_DIR}"
    DOWNLOADED_MODEL_ASSETS
)

set(STEREO_MATCHING_ONNX_FILE_NAME "stereo-matching.onnx")
set(STEREO_MATCHING_ONNX_DOWNLOADED_PATH "${RETINIFY_DOWNLOADED_MODELS_DIR}/${STEREO_MATCHING_ONNX_FILE_NAME}")

list(FIND DOWNLOADED_MODEL_ASSETS "${STEREO_MATCHING_ONNX_DOWNLOADED_PATH}" stereo_asset_index)
if(stereo_asset_index EQUAL -1)
    message(FATAL_ERROR "Asset '${STEREO_MATCHING_ONNX_FILE_NAME}' not found in downloaded release assets")
endif()

# VARIABLES CONFIGURATION
set(STEREO_MATCHING_ONNX_PATH "${CMAKE_INSTALL_PREFIX}/share/retinify/weights/${PROJECT_VERSION}/${STEREO_MATCHING_ONNX_FILE_NAME}")

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/retinifyModels.hpp.in"
    "${CMAKE_BINARY_DIR}/include/retinify/retinifyModels.hpp"
    @ONLY
)

# SUBDIRECTORIES
add_subdirectory(retinify)
add_subdirectory(3rdparty)
if(BUILD_SAMPLES)
    add_subdirectory(samples)
endif(BUILD_SAMPLES)

# SHARE
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE DESTINATION share/retinify COMPONENT runtime)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/NOTICE.md DESTINATION share/retinify COMPONENT runtime)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/VERSION DESTINATION share/retinify COMPONENT runtime)
install(FILES ${DOWNLOADED_MODEL_ASSETS} DESTINATION share/retinify/weights/${PROJECT_VERSION} COMPONENT runtime)

# CPACK
set(CPACK_GENERATOR "DEB")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_COMPONENTS_ALL runtime dev)
set(CPACK_COMPONENT_RUNTIME_DESCRIPTION "Runtime library for retinify")
set(CPACK_COMPONENT_DEV_DESCRIPTION "Development files for retinify")
set(CPACK_DEBIAN_ENABLE_COMPONENT_DEPENDS ON)
set(CPACK_COMPONENT_DEV_DEPENDS runtime)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
  set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE amd64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE arm64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(armv6l|armv7l)$")
  set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE armhf)
else()
  set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "${CMAKE_SYSTEM_PROCESSOR}")
endif()

set(_retinify_pkg_tag "cpu")
if(BUILD_WITH_TENSORRT)
  find_package(CUDAToolkit REQUIRED)

  set(_retinify_cuda_major "${CUDAToolkit_VERSION_MAJOR}")
  if(NOT _retinify_cuda_major)
    string(REGEX MATCH "^[0-9]+" _retinify_cuda_major "${CUDAToolkit_VERSION}")
  endif()
  if(NOT _retinify_cuda_major AND DEFINED CMAKE_CUDA_COMPILER_VERSION)
    string(REGEX MATCH "^[0-9]+" _retinify_cuda_major "${CMAKE_CUDA_COMPILER_VERSION}")
  endif()
  if(NOT _retinify_cuda_major)
    set(_retinify_cuda_major "unknown")
  endif()

  set(_retinify_tensorrt_major "")
  if(NVINFER)
    get_filename_component(_retinify_nvinfer_name "${NVINFER}" NAME)
    if(_retinify_nvinfer_name MATCHES "libnvinfer\\.so\\.([0-9]+)")
      set(_retinify_tensorrt_major "${CMAKE_MATCH_1}")
    elseif(EXISTS "${NVINFER}" AND IS_SYMLINK "${NVINFER}")
      file(READ_SYMLINK "${NVINFER}" _retinify_nvinfer_link)
      get_filename_component(_retinify_nvinfer_link_name "${_retinify_nvinfer_link}" NAME)
      if(_retinify_nvinfer_link_name MATCHES "libnvinfer\\.so\\.([0-9]+)")
        set(_retinify_tensorrt_major "${CMAKE_MATCH_1}")
      endif()
    endif()
  endif()
  if(NOT _retinify_tensorrt_major)
    set(_retinify_tensorrt_major "unknown")
  endif()

  set(_retinify_pkg_tag "cuda${_retinify_cuda_major}-tensorrt${_retinify_tensorrt_major}")
endif()

get_target_property(RETINIFY_SOVERSION retinify SOVERSION)
if(NOT RETINIFY_SOVERSION)
  set(RETINIFY_SOVERSION "${PROJECT_VERSION_MAJOR}")
endif()

set(CPACK_DEBIAN_RUNTIME_PACKAGE_NAME "libretinify${RETINIFY_SOVERSION}")
set(CPACK_DEBIAN_DEV_PACKAGE_NAME "libretinify-dev")
set(CPACK_DEBIAN_RUNTIME_FILE_NAME "${CPACK_DEBIAN_RUNTIME_PACKAGE_NAME}-${_retinify_pkg_tag}_${CPACK_PACKAGE_VERSION}_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}.deb")
set(CPACK_DEBIAN_DEV_FILE_NAME "${CPACK_DEBIAN_DEV_PACKAGE_NAME}-${_retinify_pkg_tag}_${CPACK_PACKAGE_VERSION}_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}.deb")

set(CPACK_DEBIAN_RUNTIME_PACKAGE_SHLIBDEPS ON)
set(CPACK_DEBIAN_DEV_PACKAGE_SHLIBDEPS ON)
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Sensui Yagi <contact@retinify.ai>")
set(CPACK_PACKAGE_CONTACT "contact@retinify.ai")
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
message(STATUS "STEREO_MATCHING_ONNX   : ${STEREO_MATCHING_ONNX_DOWNLOADED_PATH}")
message(STATUS "================================================================================")
