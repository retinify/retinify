# PROJECT
set(PROJECT_NAME retinify_core)

# SOURCES
file(GLOB_RECURSE SRCS_CORE ./src/*.c* ./src/*.h* ./include/*.h*)

# LIBRARY
add_library(${PROJECT_NAME} SHARED ${SRCS_CORE})

# LINKS
target_link_libraries(${PROJECT_NAME} 
    PUBLIC ${OpenCV_LIBS}
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

# CONFIGURATION
set(RETINIFY_SHARE_DIR "${CMAKE_INSTALL_PREFIX}/share/retinify")

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/retinify/config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/retinify/config.h
    @ONLY
)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/include)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/retinify/config.h
    DESTINATION include/retinify
)