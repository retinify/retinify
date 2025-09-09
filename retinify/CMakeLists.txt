# CXX SOURCES
file(GLOB SRCS_CXX
    ./src/*.cpp
    ./src/*.hpp
    ./include/*.hpp
)

# CUDA SOURCES
if (BUILD_WITH_TENSORRT)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)    
    file(GLOB SRCS_CUDA ./src/cuda/*.cu ./src/cuda/*.h)
    list(APPEND SRCS_CXX ${SRCS_CUDA})
endif()

# LIBRARY
add_library(retinify SHARED ${SRCS_CXX})
set_target_properties(retinify PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

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
        $<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>
        $<$<COMPILE_LANGUAGE:CXX>:-fvisibility-inlines-hidden>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
        $<$<COMPILE_LANGUAGE:CXX>:-funroll-loops>
        $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>
)

if(BUILD_WITH_TENSORRT)
    target_compile_options(retinify
        PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fvisibility=hidden>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fvisibility-inlines-hidden>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-funroll-loops>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-rtti>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3>
            $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
            $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
            $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
    )
    set_target_properties(retinify PROPERTIES
        CUDA_ARCHITECTURES "75;80;86;87;89"
    )    
endif()

# BUILD OPTIONS
if (BUILD_WITH_TENSORRT)
    target_compile_definitions(retinify PUBLIC BUILD_WITH_TENSORRT)

    target_include_directories(retinify PUBLIC /usr/local/cuda/include)
    target_link_libraries(retinify
        PUBLIC
            /usr/local/cuda/lib64/libcudart.so
            /usr/local/cuda/lib64/libnppc.so
            /usr/local/cuda/lib64/libnppidei.so
            /usr/local/cuda/lib64/libnppicc.so
            /usr/local/cuda/lib64/libnppig.so
            /usr/local/cuda/lib64/libnppial.so
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
        message(WARNING "TensorRT libraries not found (${CMAKE_SYSTEM_PROCESSOR})")
    endif()
else()
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
