#!/bin/sh
cd "`dirname "$0"`"
test -f .makeApidox.stamp || touch .makeApidox.stamp
while true; do
  ls -l -t Vc/* doc/dox.h doc/examples.h Vc/common/*|head -n1 > .makeApidox.stamp.new
  if ! diff -q .makeApidox.stamp.new .makeApidox.stamp; then
    ./makeApidox.sh
    mv -f .makeApidox.stamp.new .makeApidox.stamp
  else
    sleep 1s
  fi
done
