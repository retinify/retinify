find_package(OpenCV REQUIRED)

add_executable(inference ./inference/main.cpp)
target_link_libraries(inference ${OpenCV_LIBS} retinify retinify-tools)
target_include_directories(inference
    PRIVATE
        ${CMAKE_BINARY_DIR}/include
)

add_executable(latency ./latency/main.cpp)
target_link_libraries(latency ${OpenCV_LIBS} retinify retinify-tools)
target_include_directories(latency
    PRIVATE
        ${CMAKE_BINARY_DIR}/include
)
