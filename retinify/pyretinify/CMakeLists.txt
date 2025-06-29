find_package(pybind11 REQUIRED)
pybind11_add_module(pyretinify pyretinify.cpp)
target_link_libraries(pyretinify PRIVATE retinify)

set_target_properties(pyretinify PROPERTIES
    INSTALL_RPATH "$ORIGIN/../lib"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

install(TARGETS pyretinify DESTINATION retinify)
install(FILES __init__.py DESTINATION retinify)
