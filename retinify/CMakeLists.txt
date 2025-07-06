# SOURCES
file(GLOB_RECURSE SRCS_RETINIFY ./src/*.cpp ./src/*.hpp ./include/*.hpp)

# LIBRARY
add_library(retinify SHARED ${SRCS_RETINIFY})

# INCLUDES
target_include_directories(retinify
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
target_compile_options(retinify 
    PRIVATE
        -fvisibility=hidden
        -fvisibility-inlines-hidden
        -O3
        -funroll-loops
        -fno-rtti
)

target_compile_options(retinify
    PRIVATE
        -Wall
        -Wextra
        -Werror
)

# BUILD OPTIONS
if (USE_NVIDIA_GPU)
    target_compile_definitions(retinify PUBLIC USE_NVIDIA_GPU)

    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        find_package(CUDAToolkit REQUIRED)
        target_link_libraries(retinify PUBLIC CUDA::cudart)
    else()
        target_include_directories(retinify PUBLIC /usr/local/cuda/include)
        target_link_libraries(retinify PUBLIC /usr/local/cuda/lib64/libcudart.so)
    endif()

    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        set(_TRT_LIB_HINTS
            /usr/lib
            /usr/lib/x86_64-linux-gnu
            /usr/local/lib
        )
        set(_TRT_INC_HINTS
            /usr/include
            /usr/include/x86_64-linux-gnu
        )
    else()
        set(_TRT_LIB_HINTS
            /usr/lib/aarch64-linux-gnu
            /usr/local/lib
        )
        set(_TRT_INC_HINTS
            /usr/include/aarch64-linux-gnu
            /usr/local/include
        )
    endif()

    find_library(NVINFER       NAMES nvinfer       HINTS ${_TRT_LIB_HINTS})
    find_library(NVINFERPLUGIN NAMES nvinfer_plugin HINTS ${_TRT_LIB_HINTS})
    find_library(NVONNXPARSER  NAMES nvonnxparser  HINTS ${_TRT_LIB_HINTS})
    find_path  (TENSORRT_INCLUDE_DIR NAMES NvInfer.h  HINTS ${_TRT_INC_HINTS})

    if (NVINFER AND NVINFERPLUGIN AND NVONNXPARSER AND TENSORRT_INCLUDE_DIR)
        target_include_directories(retinify PRIVATE ${TENSORRT_INCLUDE_DIR})
        target_link_libraries(retinify PRIVATE
            ${NVINFER}
            ${NVINFERPLUGIN}
            ${NVONNXPARSER}
        )
        message(STATUS "TensorRT linked (${CMAKE_SYSTEM_PROCESSOR})")
    else()
        message(WARNING "TensorRT libraries not found (${CMAKE_SYSTEM_PROCESSOR})")
    endif()
else()
# ONNX RUNTIME
    include(${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime.cmake)
    target_link_libraries(retinify
        PRIVATE
            onnxruntime
    )
endif()

# INSTALL
install(TARGETS retinify
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

# PYTHON BINDINGS
if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(pyretinify)
endif(BUILD_PYTHON_BINDINGS)
