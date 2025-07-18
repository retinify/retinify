# SOURCES
file(GLOB_RECURSE SRCS_TOOLS ./src/*.cpp ./src/*.hpp ./include/*.hpp)

# DEPENDENCIES
find_package(OpenCV REQUIRED)

# LIBRARY
add_library(retinify-tools SHARED ${SRCS_TOOLS})

# INCLUDES
target_include_directories(retinify-tools
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_BINARY_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# LINK LIBRARIES
target_link_libraries(retinify-tools
    PUBLIC
        retinify-core ${OpenCV_LIBS}
)

# COMPILE OPTIONS
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
target_compile_options(retinify-tools 
    PRIVATE
        -fvisibility=hidden
        -fvisibility-inlines-hidden
        -O3
        -funroll-loops
        -fno-rtti
)

target_compile_options(retinify-tools
    PRIVATE
        -Wall
        -Wextra
        -Werror
)

# INSTALL
install(TARGETS retinify-tools
    EXPORT retinifyTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)
