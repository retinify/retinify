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

# COMPILE OPTIONS
target_compile_options(retinify-tests
    PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-std=c++20>
)

# INCLUDES
target_include_directories(retinify-tests
    PRIVATE 
        ../src # retinify source
)