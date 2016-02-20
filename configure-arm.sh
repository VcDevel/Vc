#!/bin/sh
srcdir=`dirname $0`
CXX=arm-linux-gnueabi-g++ CC=arm-linux-gnueabi-gcc cmake "-DCMAKE_TOOLCHAIN_FILE=$srcdir/cmake/toolchain-arm-linux.cmake" "$srcdir"
