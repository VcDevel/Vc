#!/bin/bash

[[ -z "$CXX" ]] && CXX=`which c++`
which readlink>/dev/null && case `readlink -f $CXX` in
  *icecc)
    CXX=${ICECC_CXX:-g++}
    ;;
esac

[[ "${CXX:0:1}" != "/" ]] && CXX=`which $CXX`
which readlink>/dev/null && CXX=`readlink -f $CXX`

if [[ -z "$CXX" || "${CXX:0:1}" != "/" || "$CXX" == *$'\n'* ]]; then
  echo "Error in build_dir.sh: could not determine C++ compiler and thus the correct build dir to use" 1>&2
  exit 1
fi
CXX="${CXX#/}"
echo "build/${CXX//\//-}"
