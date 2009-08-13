#!/bin/sh
CXX=icpc CC=icc cmake -DCMAKE_BUILD_TYPE=Release "`dirname $0`"
