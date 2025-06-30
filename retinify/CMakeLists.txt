# SOURCES
file(GLOB_RECURSE SRCS_RETINIFY ./src/*.cpp ./src/*.hpp ./include/*.hpp)

# LIBRARY
add_library(${PROJECT_NAME} SHARED ${SRCS_RETINIFY})

# INCLUDES
target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_BINARY_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# COMPILE OPTIONS
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
target_compile_options(${PROJECT_NAME} 
    PRIVATE
        -fvisibility=hidden
        -fvisibility-inlines-hidden
        -O3
        -funroll-loops
        -fno-rtti
)

target_compile_options(${PROJECT_NAME}
    PRIVATE
        -Wall
        -Wextra
        -Werror
)

# BUILD OPTIONS
if(USE_NVIDIA_GPU)
    target_compile_definitions(${PROJECT_NAME} PUBLIC USE_NVIDIA_GPU)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cudart)

    find_library(NVINFER NAMES nvinfer
    HINTS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)
    find_library(NVINFERPLUGIN NAMES nvinfer_plugin
        HINTS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)
    find_library(NVPARSERS NAMES nvparsers
        HINTS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)
    find_library(NVONNXPARSER NAMES nvonnxparser
        HINTS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)
    find_library(NVONNXPARSERRUNTIME NAMES nvonnxparser_runtime
        HINTS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)

    find_path(TENSORRT_INCLUDE_DIR NAMES NvInfer.h
    HINTS /usr/include/x86_64-linux-gnu /usr/local/include)

    if(NVINFER AND NVINFERPLUGIN AND NVONNXPARSER AND TENSORRT_INCLUDE_DIR)
    target_include_directories(${PROJECT_NAME} PRIVATE ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${NVINFER} ${NVINFERPLUGIN} ${NVONNXPARSER})
    message(STATUS "TensorRT found and linked")
    else()
    message(WARNING "TensorRT libraries not found")
    endif()
else()
# ONNX RUNTIME
    include(${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime.cmake)
    target_link_libraries(${PROJECT_NAME}
        PRIVATE
            onnxruntime
    )
endif()

# INSTALL
install(TARGETS ${PROJECT_NAME}
    EXPORT retinifyTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

# TESTS
if(BUILD_TESTS)
    add_subdirectory(tests)
endif(BUILD_TESTS)

add_subdirectory(pyretinify)
