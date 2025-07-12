find_package(OpenCV REQUIRED)

add_executable(inference ./inference/main.cpp)
target_link_libraries(inference ${OpenCV_LIBS} retinify)

add_executable(latency ./latency/main.cpp)
target_link_libraries(latency ${OpenCV_LIBS} retinify)
