# SOURCES
file(GLOB_RECURSE SRCS_CORE ./src/*.cpp ./src/*.hpp ./include/*.hpp)

# LIBRARY
add_library(retinify-core SHARED ${SRCS_CORE})

# INCLUDES
target_include_directories(retinify-core
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
target_compile_options(retinify-core 
    PRIVATE
        -fvisibility=hidden
        -fvisibility-inlines-hidden
        -O3
        -funroll-loops
        -fno-rtti
)

target_compile_options(retinify-core
    PRIVATE
        -Wall
        -Wextra
        -Werror
)

# BUILD OPTIONS
if (BUILD_WITH_TENSORRT)
    target_compile_definitions(retinify-core PUBLIC BUILD_WITH_TENSORRT)

    target_include_directories(retinify-core PUBLIC /usr/local/cuda/include)
    target_link_libraries(retinify-core
        PUBLIC
            /usr/local/cuda/lib64/libcudart.so
            /usr/local/cuda/lib64/libnppc.so
            /usr/local/cuda/lib64/libnppidei.so
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
        target_include_directories(retinify-core PRIVATE ${TENSORRT_INCLUDE_DIR})
        target_link_libraries(retinify-core PRIVATE
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
install(TARGETS retinify-core
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
