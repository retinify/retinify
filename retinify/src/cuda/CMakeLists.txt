enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90 100)

file(GLOB SRCS_CUDA
    lrcheck.cu
    lrcheck.h
)

add_library(retinify-cuda STATIC ${SRCS_CUDA})

find_package(CUDAToolkit REQUIRED)
target_link_libraries(retinify-cuda
    PUBLIC
        CUDA::cudart
)

set_target_properties(retinify-cuda 
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)

target_include_directories(retinify-cuda
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_compile_options(retinify-cuda
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-std=c++17>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fvisibility=hidden>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fvisibility-inlines-hidden>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-funroll-loops>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-rtti>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)
