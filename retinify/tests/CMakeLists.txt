add_subdirectory(googletest)

# PACKAGE
find_package(OpenCV REQUIRED)

# SOURCES
file(GLOB SRCS_TEST ./*.cpp ./*.h*)

# EXECUTABLE
add_executable(retinify_test ${SRCS_TEST} ${SRC_RETINIFY})

# LINKS
target_link_libraries(retinify_test 
    PUBLIC ${OpenCV_LIBS} gtest gtest_main
)

# INCLUDES
target_include_directories(retinify_test
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)