find_package(OpenCV REQUIRED)

file(GLOB SRCS_INFERENCE ./inference/*.cpp ./inference/*.hpp)
add_executable(inference ${SRCS_INFERENCE})
target_link_libraries(inference ${OpenCV_LIBS} retinify)

file(GLOB SRC_LATENCY ./latency/*.cpp ./latency/*.hpp)
add_executable(latency ${SRC_LATENCY})
target_link_libraries(latency ${OpenCV_LIBS} retinify)
