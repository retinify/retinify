# CMake
cmake_minimum_required(VERSION 3.22)
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
set(CMAKE_BUILD_TYPE Release)

# PROJECT
project(retinify)

# Enable Position Independent Code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# OpenCV
find_package(OpenCV REQUIRED)

# ONNX Runtime
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/onnxruntime.cmake)

# GTK4
find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK4 REQUIRED gtk4)
include_directories(${GTK4_INCLUDE_DIRS})
link_directories(${GTK4_LIBRARY_DIRS})
add_definitions(${GTK4_CFLAGS_OTHER})

# C++ STANDARD
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# SUBDIRECTORIES
add_subdirectory(cmake)
add_subdirectory(include)
add_subdirectory(modules)
add_subdirectory(apps)