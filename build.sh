#!/usr/bin/env bash
set -e

# PARAMETERS
BUILD_DIR="build"
INSTALL_PREFIX="/usr"
BUILD_WITH_TENSORRT=ON
BUILD_SAMPLES=OFF
BUILD_TESTS=OFF
DO_INSTALL=0
ACCEPT_EULA=0

# ARGUMENTS
for arg in "$@"; do
    case "$arg" in
        --install)
            DO_INSTALL=1
            ;;    
        --tensorrt)
            BUILD_WITH_TENSORRT=ON
            ;;
        --cpu)
            BUILD_WITH_TENSORRT=OFF
            ;;
        --dev)
            BUILD_SAMPLES=ON
            BUILD_TESTS=ON
            ;;
        --accept-eula)
            ACCEPT_EULA=1
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: $0 [--install] [--tensorrt|--cpu] [--dev] [--accept-eula]"
            exit 1
            ;;
    esac
done

# EULA ACCEPTANCE
EULA_REPO="retinify/retinify-eula"

EULA_URL="https://github.com/${EULA_REPO}/blob/main/EULA.md"
EULA_FAQ_URL="https://github.com/${EULA_REPO}/blob/main/FAQ.md"
RAW_URL="https://raw.githubusercontent.com/${EULA_REPO}/main/EULA.md"

if ! command -v sha256sum >/dev/null 2>&1; then
    echo -e "\033[1;31m\033[1m[RETINIFY] 'sha256sum' not found. Please install coreutils.\033[0m"
    exit 1
fi

EULA_SHA256="$(curl -fsSL --retry 3 --retry-delay 1 --max-time 15 "${RAW_URL}" | sha256sum | cut -d' ' -f1 || true)"
if [[ -z "${EULA_SHA256}" ]]; then
    echo -e "\033[1;31m\033[1m[RETINIFY] Failed to get EULA checksum.\033[0m"
    exit 1
fi

if [[ "${ACCEPT_EULA:-0}" == "1" ]]; then
    echo -e "\033[1;32m\033[1m[RETINIFY] EULA ACCEPTED VIA COMMAND LINE. CONTINUING BUILD.\033[0m"
else
    echo "----------------------------------------------------------------------"
    echo -e "\033[1;32m[RETINIFY] End User License Agreement (EULA)\033[0m"
    echo "EULA Text : ${EULA_URL}"
    echo "FAQ       : ${EULA_FAQ_URL}"
    echo "Checksum  : ${EULA_SHA256}"
    echo "----------------------------------------------------------------------"
    echo "By typing 'yes', you confirm that you have read and agree to the full"
    echo "text of the retinify End User License Agreement (EULA)."
    echo "If you do not agree, the build process will be aborted."
    echo "----------------------------------------------------------------------"

    if [[ ! -t 0 ]]; then
        echo -e "\033[1;31m\033[1m[RETINIFY] No TTY and no non-interactive acceptance provided. Aborting.\033[0m"
        exit 3
    fi
    read -r -p "Do you accept the retinify End User License Agreement? [yes/no]: " ans
    case "${ans}" in
        yes)
            echo -e "\033[1;32m[RETINIFY] EULA ACCEPTED. CONTINUING BUILD.\033[0m"
            ;;
        *)
            echo -e "\033[1;31m\033[1m[RETINIFY] EULA NOT ACCEPTED. ABORTING BUILD.\033[0m"
            exit 1
            ;;
    esac
fi

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