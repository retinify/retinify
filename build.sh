#!/usr/bin/env bash
set -e

# PARAMETERS
BUILD_DIR="build"
INSTALL_PREFIX="/usr"
BUILD_WITH_TENSORRT=ON
BUILD_SAMPLES=OFF
BUILD_TESTS=OFF
DO_INSTALL=0

# ARGUMENTS
for arg in "$@"; do
    case "$arg" in
        --install)
            DO_INSTALL=1
            ;;    
        --gpu)
            BUILD_WITH_TENSORRT=ON
            ;;
        --cpu)
            BUILD_WITH_TENSORRT=OFF
            ;;
        --dev)
            BUILD_SAMPLES=ON
            BUILD_TESTS=ON
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: $0 [--install] [--gpu|--cpu] [--dev]"
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

    DEB_PACKAGE=$(ls -t libretinify-*.deb 2>/dev/null | head -n 1)

    if [[ -z "${DEB_PACKAGE}" ]]; then
        echo -e "\033[1;31m[RETINIFY] ERROR: RETINIFY DEBIAN PACKAGE NOT FOUND.\033[0m"
        exit 1
    fi

    if [[ $EUID -eq 0 ]]; then
        dpkg -i "${DEB_PACKAGE}"
    else
        sudo dpkg -i "${DEB_PACKAGE}"
    fi
fi

cd ..

echo -e "\033[1;32m[RETINIFY] BUILD PROCESS COMPLETED\033[0m"