# PROJECT
set(PROJECT_NAME retinify_pipeline)

# SOURCES
file(GLOB_RECURSE SRCS_PIPELINE ./src/*.c* ./src/*.h* ./include/*.h*)

# LIBRARY
add_library(${PROJECT_NAME} SHARED ${SRCS_PIPELINE})

# LINKS
target_link_libraries(${PROJECT_NAME} 
    PUBLIC ${OpenCV_LIBS} retinify_core retinify_io retinify_engine
)

# INCLUDES
target_include_directories(${PROJECT_NAME}
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# INSTALL
install(TARGETS ${PROJECT_NAME}
    EXPORT retinifyTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h*"
)