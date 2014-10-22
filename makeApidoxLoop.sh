#!/bin/sh
test -f .makeApidox.stamp || touch .makeApidox.stamp
while true; do
  ls -l -t include/Vc/* doc/dox.h doc/examples.h common/*|head -n1 > .makeApidox.stamp.new
  if ! diff -q .makeApidox.stamp.new .makeApidox.stamp; then
    ./makeApidox.sh
    mv -f .makeApidox.stamp.new .makeApidox.stamp
  else
    sleep 1s
  fi
done
