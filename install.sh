#!/bin/sh
cd "${0%/*}"

test -x "$CXX" || CXX=g++

default_includedir() {
  $CXX --version >/dev/null || exit 1
  dirname `echo "#include <vector>"|"$CXX" -std=c++17 -x c++ -E -o- -|grep '^# 1 "/.*/vector"'|cut -d'"' -f2`
}

includedir=""
do_install=true

usage() {
  cat << EOF
Usage: $0 [options]

Options:
  -h|--help         This message.
  --gxx=FILEPATH    Compiler to use for querying the default includedir [\$CXX]
  --includedir=DIR  header files installation dir [$(default_includedir)]
  --uninstall       Remove the destination files instead of installing.
EOF
}

while test $# -gt 0; do
  case "$1" in
    --gxx=*)
      CXX="${1#--gxx=}"
      ;;
    --gxx)
      shift
      CXX="$1"
      ;;
    --includedir=*)
      includedir="${1#--includedir=}"
      ;;
    --includedir)
      shift
      includedir="$1"
      ;;
    --uninstall)
      do_install=false
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option \"$1\"." >&2
      usage
      exit 1
      ;;
  esac
  shift
done

test -z "$includedir" && includedir=`default_includedir` || exit 1
if ! test -w "$includedir"; then
  echo "Error: Destination directory '$includedir' is not writable by the current user."
  exit 1
fi

if $do_install; then
  echo "Testing that $CXX can include <experimental/simd> without errors:"
  echo '#include "experimental/simd"'|"$CXX" -fmax-errors=1 -fsyntax-only -std=c++17 -x c++ -o- - >/dev/null || { echo Failed.; exit 1; } || exit
  echo Passed.

  install -v -p -d "$includedir/experimental/bits"
  install -v -p experimental/bits/simd*.h "$includedir/experimental/bits/"
  install -v -p experimental/simd "$includedir/experimental/"
else
  for i in experimental/bits/simd*.h experimental/simd; do
    dst="$includedir/$i"
    if test -e "$dst"; then
      echo "Removing $dst"
      rm -f "$dst"
    fi
  done
fi
