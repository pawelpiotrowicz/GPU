#!/bin/bash

git clone https://github.com/oneapi-src/oneDNN.git
cd oneDNN

if [ -f build ]; then
 rm -rf build
fi
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=`pwd`/../../onednn_install ..
make -j20
make install



