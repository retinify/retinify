project(gui_app LANGUAGES CXX)

file(GLOB SRCS_GUI_APP ./${PROJECT_NAME}/*.cpp ./${PROJECT_NAME}/*.hpp)

add_executable(${PROJECT_NAME} ${SRCS_GUI_APP})

target_link_libraries(${PROJECT_NAME} retinify_pipeline retinify_gui)