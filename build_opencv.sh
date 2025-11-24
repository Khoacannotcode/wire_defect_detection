#!/bin/bash
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
#
# This script is a modified version of the one from JetsonHacksNano:
# https://github.com/JetsonHacksNano/buildOpenCV
# It has been adapted to be part of a larger installation process.

set -e

echo "Starting OpenCV build process..."

# It is recommended to run this script in its own directory
OPENCV_BUILD_DIR=$(pwd)
echo "Build directory: $OPENCV_BUILD_DIR"


# Function to configure and build OpenCV
function build_opencv {
    echo "Building OpenCV version: $OPENCV_VERSION"
    cd "$OPENCV_BUILD_DIR"
    
    # Download source code
    git clone --depth 1 --branch "$OPENCV_VERSION" https://github.com/opencv/opencv.git
    git clone --depth 1 --branch "$OPENCV_VERSION" https://github.com/opencv/opencv_contrib.git

    # Create a build directory
    cd opencv
    mkdir build
    cd build

    # Get the number of CPU cores
    NUM_CORES=$(nproc)
    echo "Using $NUM_CORES CPU cores for compilation."

    # CMake configuration
    # Note: Many flags are set here to optimize for Jetson, including CUDA support
    cmake \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_EXTRA_MODULES_PATH="$OPENCV_BUILD_DIR/opencv_contrib/modules" \
        -D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
        -D WITH_CUDA=ON \
        -D CUDA_ARCH_BIN=5.3 \
        -D CUDA_ARCH_PTX="" \
        -D WITH_CUDNN=ON \
        -D WITH_CUBLAS=ON \
        -D ENABLE_FAST_MATH=ON \
        -D CUDA_FAST_MATH=ON \
        -D OPENCV_DNN_CUDA=ON \
        -D ENABLE_NEON=ON \
        -D WITH_QT=OFF \
        -D WITH_OPENMP=ON \
        -D BUILD_TIFF=ON \
        -D WITH_FFMPEG=ON \
        -D WITH_GSTREAMER=ON \
        -D WITH_TBB=ON \
        -D BUILD_TBB=ON \
        -D BUILD_TESTS=OFF \
        -D WITH_EIGEN=ON \
        -D WITH_V4L=ON \
        -D WITH_LIBV4L=ON \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D INSTALL_C_EXAMPLES=OFF \
        -D INSTALL_PYTHON_EXAMPLES=OFF \
        -D BUILD_NEW_PYTHON_SUPPORT=ON \
        -D BUILD_opencv_python3=TRUE \
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        -D BUILD_EXAMPLES=OFF ..

    # Compile
    make -j$NUM_CORES
    
    # Install
    sudo make install
    
    echo "OpenCV installation complete."
}

# Set OpenCV version to a known stable version
OPENCV_VERSION="4.5.4" 

# Temporarily increase swap space for this script
echo "Increasing swap for build process..."
sudo fallocate -l 16G /mnt/16G.swap
sudo chmod 600 /mnt/16G.swap
sudo mkswap /mnt/16G.swap
sudo swapon /mnt/16G.swap
free -h

# Run the build
build_opencv

# Clean up swap space
echo "Cleaning up swap space..."
sudo swapoff /mnt/16G.swap
sudo rm /mnt/16G.swap
free -h

echo "OpenCV build script finished."
