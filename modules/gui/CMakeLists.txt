# PROJECT
set(PROJECT_NAME retinify_gui)

# SOURCES
file(GLOB_RECURSE SRCS_GUI ./src/*.c* ./src/*.h* ./include/*.h*)

# LIBRARY
add_library(${PROJECT_NAME} SHARED ${SRCS_GUI})

# LINKS
target_link_libraries(${PROJECT_NAME}
    PUBLIC ${GTK4_LIBRARIES} GLESv2
    PRIVATE retinify_pipeline
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

# INSTALL RESOURCES
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/resources/
    DESTINATION share/retinify/resources/
)