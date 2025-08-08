# stb
add_library(stb INTERFACE)
target_include_directories(stb INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/stb)

# google test
if(BUILD_TESTS)
    add_subdirectory(googletest)
endif(BUILD_TESTS)