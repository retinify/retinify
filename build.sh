#!/usr/bin/env bash
set -e

# PARAMETERS
BUILD_DIR="build"
INSTALL_PREFIX="/usr"
BUILD_WITH_TENSORRT=ON
BUILD_SAMPLES=OFF
BUILD_TESTS=OFF
DO_INSTALL=0
DEV_MODE=0

# ARGUMENTS
for arg in "$@"; do
    case "$arg" in
        --dev)
            DEV_MODE=1
            ;;
        --install)
            DO_INSTALL=1
            ;;    
        --tensorrt)
            BUILD_WITH_TENSORRT=ON
            ;;
        --cpu)
            BUILD_WITH_TENSORRT=OFF
            ;;
        --tests)
            BUILD_TESTS=ON
            ;;
        --samples)
            BUILD_SAMPLES=ON
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: $0 [--dev] [--install] [--tensorrt|--cpu] [--tests] [--samples]"
            exit 1
            ;;
    esac
done

# BUILD
echo -e "\033[1;32m[RETINIFY] STARTING BUILD PROCESS\033[0m"

# RESET BUILD DIR
if [[ -d "${BUILD_DIR}" ]]; then
    echo -e "\033[1;33m[RETINIFY] REMOVING OLD BUILD DIRECTORY\033[0m"
    rm -rf "${BUILD_DIR}"
fi
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
      -DBUILD_WITH_TENSORRT="${BUILD_WITH_TENSORRT}" \
      -DBUILD_SAMPLES="${BUILD_SAMPLES}" \
      -DBUILD_TESTS="${BUILD_TESTS}" \
      ..

make -j"$(nproc)"
cpack -G DEB

# INSTALL
if [[ "${DO_INSTALL}" -eq 1 ]]; then
    echo -e "\033[1;32m[RETINIFY] INSTALLING DEBIAN PACKAGE\033[0m"

    RUNTIME_DEB=$(ls -t libretinify[0-9]*-*.deb 2>/dev/null | head -n 1)
    if [[ -z "${RUNTIME_DEB}" ]]; then
        RUNTIME_DEB=$(ls -t libretinify-*.deb 2>/dev/null | grep -v "libretinify-dev" | head -n 1)
    fi
    DEV_DEB=$(ls -t libretinify-dev-*.deb 2>/dev/null | head -n 1)

    if [[ -z "${RUNTIME_DEB}" ]]; then
        echo -e "\033[1;31m[RETINIFY] ERROR: RETINIFY DEBIAN PACKAGE NOT FOUND.\033[0m"
        exit 1
    fi

    INSTALL_PKGS=("${RUNTIME_DEB}")
    if [[ "${DEV_MODE}" -eq 1 ]]; then
        if [[ -z "${DEV_DEB}" ]]; then
            echo -e "\033[1;31m[RETINIFY] ERROR: RETINIFY DEV PACKAGE NOT FOUND.\033[0m"
            exit 1
        fi
        INSTALL_PKGS+=("${DEV_DEB}")
    fi

    if [[ $EUID -eq 0 ]]; then
        dpkg -i "${INSTALL_PKGS[@]}"
    else
        sudo dpkg -i "${INSTALL_PKGS[@]}"
    fi
fi

cd ..

echo -e "\033[1;32m[RETINIFY] BUILD PROCESS COMPLETED\033[0m"
