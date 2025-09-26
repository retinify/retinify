# PACKAGE
find_package(OpenCV REQUIRED)

# SOURCES
file(GLOB SRCS_TESTS ./*.c* ./*.h*)

# EXECUTABLE
add_executable(retinify-tests ${SRCS_TESTS} ${SRC_RETINIFY})

# LINKS
target_link_libraries(retinify-tests 
    PUBLIC retinify ${OpenCV_LIBS} gtest gtest_main
)

# BUILD OPTIONS
if(BUILD_WITH_TENSORRT)
    target_compile_definitions(retinify-tests 
        PUBLIC 
            BUILD_WITH_TENSORRT
    )

    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(retinify-tests
        PRIVATE
            CUDA::cudart
    )
endif()

# INCLUDES
target_include_directories(retinify-tests
    PRIVATE 
        ../src # retinify source
)