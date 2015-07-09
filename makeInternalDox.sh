#!/bin/sh

cd "`dirname "$0"`/doc"

rm -rf internal
doxygen DoxyInternal
