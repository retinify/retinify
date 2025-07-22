# ONNXRuntime
include(ExternalProject)
set(ONNXRUNTIME_VERSION "1.20.1")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
        set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz")
        set(ONNXRUNTIME_SHA256 "67db4dc1561f1e3fd42e619575c82c601ef89849afc7ea85a003abbac1a1a105")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
        set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}.tgz")
        set(ONNXRUNTIME_SHA256 "ae4fedbdc8c18d688c01306b4b50c63de3445cdf2dbd720e01a2fa3810b8106a")
endif()
set(ONNXRUNTIME_INSTALL_DIR ${CMAKE_BINARY_DIR}/onnxruntime)

ExternalProject_Add(ONNXRuntime
    URL ${ONNXRUNTIME_URL}
    URL_HASH SHA256=${ONNXRUNTIME_SHA256}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR> ${ONNXRUNTIME_INSTALL_DIR}
    UPDATE_COMMAND ""
)

set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_INSTALL_DIR}/include)
set(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_INSTALL_DIR}/lib)

add_library(onnxruntime INTERFACE)
target_include_directories(onnxruntime INTERFACE $<BUILD_INTERFACE:${ONNXRUNTIME_INCLUDE_DIR}>)
target_link_libraries(onnxruntime INTERFACE ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so)

install(DIRECTORY ${ONNXRUNTIME_INSTALL_DIR}/include/
        DESTINATION include
)

install(DIRECTORY ${ONNXRUNTIME_INSTALL_DIR}/lib/
        DESTINATION lib
)

install(FILES 
        ${ONNXRUNTIME_INSTALL_DIR}/LICENSE
        ${ONNXRUNTIME_INSTALL_DIR}/ThirdPartyNotices.txt
        ${ONNXRUNTIME_INSTALL_DIR}/VERSION_NUMBER
        DESTINATION share/onnxruntime
)

add_dependencies(onnxruntime ONNXRuntime)

install(TARGETS onnxruntime
    EXPORT retinifyTargets
)