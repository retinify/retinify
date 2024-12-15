include(FetchContent)

set(ONNXRUNTIME_VERSION "1.20.1")
set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz")
set(DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/onnxruntime-download")
set(EXTRACT_DIR "${CMAKE_BINARY_DIR}/onnxruntime")
set(VERSION_FILE "${DOWNLOAD_DIR}/ONNXRUNTIME_VERSION.txt")

function(download_and_extract_onnxruntime)
    if(EXISTS ${VERSION_FILE})
        file(READ ${VERSION_FILE} CACHED_VERSION)
        if(CACHED_VERSION STREQUAL ONNXRUNTIME_VERSION)
            message(STATUS "ONNXRuntime ${ONNXRUNTIME_VERSION} is already downloaded and extracted")
            return()
        endif()
    endif()

    if(NOT EXISTS ${DOWNLOAD_DIR})
        file(MAKE_DIRECTORY ${DOWNLOAD_DIR})
    endif()

    if(NOT EXISTS "${DOWNLOAD_DIR}/onnxruntime.tgz")
        message(STATUS "Downloading ONNXRuntime from ${ONNXRUNTIME_URL}")
        file(DOWNLOAD
            ${ONNXRUNTIME_URL}
            "${DOWNLOAD_DIR}/onnxruntime.tgz"
            SHOW_PROGRESS
            STATUS DOWNLOAD_STATUS
        )
        list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
        if(NOT STATUS_CODE EQUAL 0)
            message(FATAL_ERROR "Failed to download ONNXRuntime")
        endif()
    endif()

    if(EXISTS ${EXTRACT_DIR})
        file(REMOVE_RECURSE ${EXTRACT_DIR})
    endif()
    file(MAKE_DIRECTORY ${EXTRACT_DIR})

    message(STATUS "Extracting ONNXRuntime to ${EXTRACT_DIR}")
    execute_process(
        COMMAND tar -xzvf "${DOWNLOAD_DIR}/onnxruntime.tgz" -C ${EXTRACT_DIR} --strip-components=1
        WORKING_DIRECTORY ${DOWNLOAD_DIR}
        RESULT_VARIABLE TAR_RESULT
    )

    if(NOT TAR_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to extract ONNXRuntime")
    endif()

    file(WRITE ${VERSION_FILE} ${ONNXRUNTIME_VERSION})
endfunction()

download_and_extract_onnxruntime()

set(ONNXRUNTIME_INCLUDE_DIR "${EXTRACT_DIR}/include")
set(ONNXRUNTIME_LIBRARY "${EXTRACT_DIR}/lib/libonnxruntime.so")

message(STATUS "ONNXRuntime include directory: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNXRuntime library: ${ONNXRUNTIME_LIBRARY}")

install(DIRECTORY "${EXTRACT_DIR}/"
        DESTINATION "."
        USE_SOURCE_PERMISSIONS
        COMPONENT onnxruntime)