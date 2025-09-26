# CXX SOURCES
file(GLOB SRCS_CXX
    ./src/*.cpp
    ./src/*.hpp
    ./include/*.hpp
)

# LIBRARY
add_library(retinify SHARED ${SRCS_CXX})
set_target_properties(retinify 
    PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
)

# INCLUDES
target_include_directories(retinify
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_BINARY_DIR}/include
)

# SETTINGS
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

target_compile_features(retinify 
    PUBLIC 
        cxx_std_20
)

target_compile_options(retinify
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>
        $<$<COMPILE_LANGUAGE:CXX>:-fvisibility-inlines-hidden>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
        $<$<COMPILE_LANGUAGE:CXX>:-funroll-loops>
        $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>
)

if(BUILD_WITH_TENSORRT)
    add_subdirectory(src/cuda)
    target_link_libraries(retinify 
        PRIVATE 
            retinify-cuda
    )

    target_compile_definitions(retinify 
        PUBLIC 
            BUILD_WITH_TENSORRT
    )

    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(retinify
        PUBLIC
            CUDA::cudart
            CUDA::cuda_driver
            CUDA::nppc
            CUDA::nppidei
            CUDA::nppicc
            CUDA::nppig
            CUDA::nppial
    )

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
        set(_TRT_LIB_HINTS
            /usr/lib
            /usr/lib/x86_64-linux-gnu
            /usr/local/lib
        )
        set(_TRT_INC_HINTS
            /usr/include
            /usr/include/x86_64-linux-gnu
        )
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
        set(_TRT_LIB_HINTS
            /usr/lib
            /usr/lib/aarch64-linux-gnu
            /usr/local/lib
        )
        set(_TRT_INC_HINTS            
            /usr/include
            /usr/include/aarch64-linux-gnu
            /usr/local/include
        )
    endif()

    find_library(NVINFER       NAMES nvinfer        HINTS ${_TRT_LIB_HINTS})
    find_library(NVONNXPARSER  NAMES nvonnxparser   HINTS ${_TRT_LIB_HINTS})
    find_path(TENSORRT_INCLUDE_DIR NAMES NvInfer.h  HINTS ${_TRT_INC_HINTS})

    if (NVINFER AND NVONNXPARSER AND TENSORRT_INCLUDE_DIR)
        target_include_directories(retinify PRIVATE ${TENSORRT_INCLUDE_DIR})
        target_link_libraries(retinify PRIVATE
            ${NVINFER}
            ${NVONNXPARSER}
        )
        message(STATUS "TensorRT linked (${CMAKE_SYSTEM_PROCESSOR})")
    else()
        message(FATAL_ERROR "TensorRT libraries not found (${CMAKE_SYSTEM_PROCESSOR})")
    endif()
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
