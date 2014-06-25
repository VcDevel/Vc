#!/bin/sh

cd "`dirname "$0"`/doc"

rm -rf html
doxygen
rm -f html/*.qhp
