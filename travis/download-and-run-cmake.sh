#!/bin/sh
mkdir -p build
cd build
wget https://github.com/Kitware/CMake/releases/download/v3.17.1/cmake-3.17.1-Linux-x86_64.tar.gz
tar -xvf cmake-3.17.1-Linux-x86_64.tar.gz
cmake-3.17.1-Linux-x86_64/bin/cmake ..
