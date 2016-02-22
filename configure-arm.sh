#!/bin/sh
srcdir=`dirname $0`
skip=false
test -n "$CXX" -a -n "$CC" && \
   case "`$CXX -dumpmachine`" in
      aarch64-linux-*)
         skip=true
         ;;
      arm-linux-*)
         skip=true
         ;;
   esac

if $skip; then
   echo "Using $CXX"
   # nothing
elif which aarch64-linux-gnueabi-g++ >/dev/null; then
   export CXX=aarch64-linux-gnueabi-g++ CC=aarch64-linux-gnueabi-gcc
elif which arm-linux-gnueabihf-g++ >/dev/null; then
   export CXX=arm-linux-gnueabihf-g++ CC=arm-linux-gnueabihf-gcc
elif which arm-linux-gnueabi-g++ >/dev/null; then
   export CXX=arm-linux-gnueabi-g++ CC=arm-linux-gnueabi-gcc
else
   echo "No suitable compiler found." 1>&2
   exit 1
fi
cmake "-DCMAKE_TOOLCHAIN_FILE=$srcdir/cmake/toolchain-arm-linux.cmake" "$srcdir"
