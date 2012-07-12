#!/bin/sh
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games"
export LANG="en_US.UTF-8"
export LANGUAGE="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8"
export LC_NUMERIC="en_US.UTF-8"
export LC_TIME="en_US.UTF-8"
export LC_MESSAGES="en_US.UTF-8"
unset CFLAGS CXXFLAGS

cd "`dirname "$0"`"

runTest() {
  CFLAGS="$1" CXXFLAGS="$1" ./Test_vc.sh Experimental
}

supports32Bit() {
  test `uname -m` = "x86_64" || return 1
  CXX=${CXX:-c++}
  cat > /tmp/m32test.cpp <<END
#include <iostream>
int main() { std::cout << "Hello World!\n"; return 0; }
END
  $CXX -m32 -o /tmp/m32test /tmp/m32test.cpp >/dev/null 2>&1 || return 1
  rm /tmp/m32test*
  return 0
}

cxxlist="`find /usr/bin/ /usr/local/bin/ -name 'g++-*'`"
if test -z "$cxxlist"; then
  cxxlist="`find /usr/bin/ /usr/local/bin/ -name 'g++'`"
fi
if test -z "$cxxlist"; then
  # default compiler
  runTest &
  supports32Bit && runTest -m32 &
  wait
else
  for CXX in $cxxlist; do
    CC=`echo "$CXX"|sed 's/g++/gcc/'`
    if test -x "$CC" -a -x "$CXX"; then (
      export CC
      export CXX
      runTest &
      supports32Bit && runTest -m32 &
      wait
    ) fi
  done
fi

for VcEnv in `find /opt/ -mindepth 2 -maxdepth 2 -name Vc.env`; do (
  . "$VcEnv"
  case "$VcEnv" in
    *-snapshot/Vc.env)
      ( cd $HOME/src/gcc-build && ./update.sh "`dirname "$VcEnv"`" )
      ;;
  esac
  runTest &
  supports32Bit && runTest -m32 &
  wait
) done

export CC=icc
export CXX=icpc
icclist="`find /opt/ -name 'iccvars.sh'`"
case x86_64 in
  x86_64)
    arch=intel64
    ;;
  i[345678]86)
    arch=ia32
    ;;
esac
test -n "$icclist" && for IccEnv in $icclist; do (
  . $IccEnv $arch
  runTest &
  supports32Bit && runTest -m32 &
  wait
) done
