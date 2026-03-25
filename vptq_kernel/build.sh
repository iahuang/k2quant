#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

PLATFORM_FLAGS=()
BLAS_CFLAGS=()
BLAS_LDFLAGS=()

if [[ "$(uname)" == "Darwin" ]]; then
    PLATFORM_FLAGS+=(-undefined dynamic_lookup)
    BLAS_CFLAGS+=(-I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Headers)
    BLAS_LDFLAGS+=(-framework Accelerate)
else
    PLATFORM_FLAGS+=($(python3-config --ldflags --embed 2>/dev/null || python3-config --ldflags))
    BLAS_CFLAGS+=($(pkg-config --cflags openblas 2>/dev/null || echo "-I/usr/include/openblas"))
    BLAS_LDFLAGS+=($(pkg-config --libs openblas 2>/dev/null || echo "-lopenblas"))
fi

mkdir -p ../bin

OUTPUT_NAME=../bin/vptq$(python3 -m pybind11 --extension-suffix)
VPTQ_ASSEMBLY_NAME=../bin/vptq.S
KMEANS_ASSEMBLY_NAME=../bin/kmeans.S
COMMON_FLAGS=(-O3 -Wall -shared -std=c++17 -fPIC -march=native)
SOURCES=(vptq.cpp kmeans.cpp pcores.cpp)

# if -a is passed, output the assembly instead of the binary
if [[ "${1:-}" == "-a" ]]; then
    c++ -S -fverbose-asm -g "${COMMON_FLAGS[@]}" \
        $(python3 -m pybind11 --includes) \
        "${BLAS_CFLAGS[@]}" \
        "${PLATFORM_FLAGS[@]}" \
        vptq.cpp \
        -o ${VPTQ_ASSEMBLY_NAME}
    c++ -S -fverbose-asm -g "${COMMON_FLAGS[@]}" \
        $(python3 -m pybind11 --includes) \
        "${BLAS_CFLAGS[@]}" \
        "${PLATFORM_FLAGS[@]}" \
        kmeans.cpp \
        -o ${KMEANS_ASSEMBLY_NAME}
    c++ -S -fverbose-asm -g "${COMMON_FLAGS[@]}" \
        $(python3 -m pybind11 --includes) \
        "${BLAS_CFLAGS[@]}" \
        "${PLATFORM_FLAGS[@]}" \
        pcores.cpp \
        -o ${PCORES_ASSEMBLY_NAME}
else
    c++ "${COMMON_FLAGS[@]}" \
        $(python3 -m pybind11 --includes) \
        "${BLAS_CFLAGS[@]}" \
        "${PLATFORM_FLAGS[@]}" \
        "${SOURCES[@]}" \
        "${BLAS_LDFLAGS[@]}" \
        -o ${OUTPUT_NAME}
fi