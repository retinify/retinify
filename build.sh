#!/usr/bin/env bash
set -e

# PARAMETERS
BUILD_DIR="build"
INSTALL_PREFIX="/usr"
USE_NVIDIA_GPU=ON
BUILD_TESTS=OFF
DO_INSTALL=0

# ARGUMENTS
for arg in "$@"; do
    case "$arg" in
        --install)
            DO_INSTALL=1
            ;;
        --gpu)
            USE_NVIDIA_GPU=ON
            ;;
        --cpu)
            USE_NVIDIA_GPU=OFF
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: $0 [--install] [--gpu|--cpu]"
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
      -DUSE_NVIDIA_GPU="${USE_NVIDIA_GPU}" \
      -DBUILD_TESTS="${BUILD_TESTS}" \
      ..

make -j"$(nproc)"
cpack -G DEB

# INSTALL
if [[ "${DO_INSTALL}" -eq 1 ]]; then
    echo "Installing generated Debian package..."

    DEB_PACKAGE=$(ls -t libretinify-*.deb 2>/dev/null | head -n 1)

    if [[ -z "${DEB_PACKAGE}" ]]; then
        echo "Error: Retinify Debian package not found."
        exit 1
    fi

    sudo dpkg -i "${DEB_PACKAGE}"
fi

cd ..

echo -e "\033[1;32m[RETINIFY] BUILD PROCESS COMPLETED\033[0m"