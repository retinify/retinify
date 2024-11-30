# PROJECT
set(PROJECT_NAME retinify_engine)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)

# SOURCES
file(GLOB_RECURSE SRCS_ENGINE ./src/*.c* ./src/*.h* ./include/*.h*)

# LIBRARY
add_library(${PROJECT_NAME} SHARED ${SRCS_ENGINE})

# OPTIONS
option(USE_GPU "Use GPU acceleration if available" ON)
if(USE_GPU)
  add_definitions(-DUSE_GPU)
endif()

# DEPENDENCIES
if(USE_GPU)
    find_package(CUDAToolkit REQUIRED)
    project(${PROJECT_NAME} LANGUAGES CXX CUDA)
    target_link_libraries(${PROJECT_NAME} CUDA::cudart ${ONNXRUNTIME_LIBRARY} retinify_core)
    
    target_include_directories(${PROJECT_NAME}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${ONNXRUNTIME_INCLUDE_DIR}
    )
else()
    project(${PROJECT_NAME} LANGUAGES CXX)
    target_link_libraries(${PROJECT_NAME} retinify_core)

    target_include_directories(${PROJECT_NAME}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
endif()

# OUTPUT
message(STATUS "Building with GPU support: ${USE_GPU}")

# INSTALL
install(TARGETS ${PROJECT_NAME}
    EXPORT retinifyTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)