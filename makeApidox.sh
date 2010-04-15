#!/bin/sh

cd "`dirname "$0"`/doc"

rm -rf html latex man
doxygen
cd latex
make
