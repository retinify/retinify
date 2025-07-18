add_subdirectory(googletest)

# PACKAGE
find_package(OpenCV REQUIRED)

# SOURCES
file(GLOB SRCS_TESTS ./*.c* ./*.h*)

# EXECUTABLE
add_executable(retinify_tests ${SRCS_TESTS} ${SRC_RETINIFY})

# LINKS
target_link_libraries(retinify_tests 
    PUBLIC retinify-core ${OpenCV_LIBS} gtest gtest_main
)

# INCLUDES
target_include_directories(retinify_tests
    PRIVATE 
        ../src # retinify source
)