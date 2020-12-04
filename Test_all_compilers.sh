#!/bin/sh -e
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games"
export LANG="en_US.UTF-8"
export LANGUAGE="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8"
export LC_NUMERIC="en_US.UTF-8"
export LC_TIME="en_US.UTF-8"
export LC_MESSAGES="en_US.UTF-8"
unset CFLAGS CXXFLAGS

cd "`dirname "$0"`"
test -z "dashboard_model" && export dashboard_model=Experimental

runTest() {
  libpath="$LD_LIBRARY_PATH"
  test -n "$1" && libpath="$(dirname $(realpath $($CXX $1 -print-file-name=libstdc++.so)))${libpath:+:}${libpath}"
  LD_LIBRARY_PATH="$libpath" CFLAGS="$1" CXXFLAGS="$1" ctest -S test.cmake || true
}

tested_compilers="lsakdfjwowleqirjodfisj"

runAllTests() {
  # first make sure we don't test a compiler a second time
  id="`which $CXX`"
  id="`readlink -f $id`"
  echo "$id"|grep -qF "$tested_compilers" && return
  tested_compilers="$tested_compilers
$id"

  # alright run the ctest script
  runTest
  supports32Bit && runTest -m32 || true
  supportsx32 && runTest -mx32 || true
}

supports32Bit() {
  test `uname -m` = "x86_64" || return 1
  CXX=${CXX:-c++}
  cat > /tmp/m32test.cpp <<END
#include <algorithm>
#include <string>
#include <iostream>
#include <cerrno>
void foo(int x) { switch (x) { case 0x0A: break; case 0x0B: break; case 0x0C: break; case 0x0D: break; case 0x0E: break; } }
int main() { std::cout << "Hello World!\n"; return 0; }
END
  $CXX -m32 -o /tmp/m32test /tmp/m32test.cpp >/dev/null 2>&1 || return 1
  rm /tmp/m32test*
  return 0
}

supportsx32() {
  test `uname -m` = "x86_64" || return 1
  CXX=${CXX:-c++}
  cat > /tmp/mx32test.cpp <<END
#include <algorithm>
#include <string>
#include <iostream>
#include <cerrno>
void foo(int x) { switch (x) { case 0x0A: break; case 0x0B: break; case 0x0C: break; case 0x0D: break; case 0x0E: break; } }
int main() { std::cout << "Hello World!\n"; return 0; }
END
  $CXX -mx32 -o /tmp/mx32test /tmp/mx32test.cpp >/dev/null 2>&1 || return 1
  rm /tmp/mx32test*
  return 0
}

system_compilers() {
  cxxlist="`find /usr/bin/ /usr/local/bin/ -name '*++-[0-9]*'|grep -v -- -linux-gnu`"
  if test -z "$cxxlist"; then
    cxxlist="`find /usr/bin/ /usr/local/bin/ -name '*++'|grep -v -- -linux-gnu`"
  fi
  if test -z "$cxxlist"; then
    # default compiler
    runAllTests
  else
    for CXX in $cxxlist; do
      CC=`echo "$CXX"|sed 's/clang++/clang/;s/g++/gcc/'`
      if test -x "$CC" -a -x "$CXX"; then
        export CC
        export CXX
        runAllTests
      fi
    done
  fi
}

modules_compilers() {
  if test -r /etc/profile.d/modules.sh; then
    source /etc/profile.d/modules.sh
    for mod in `module avail -t 2>&1`; do
      case `echo $mod|tr '[:upper:]' '[:lower:]'` in
        *intel*|*icc*) export CC=icc CXX=icpc;;
        *gnu*|*gcc*) export CC=gcc CXX=g++;;
        *llvm*|*clang*) export CC=clang CXX=clang++;;
        *) continue;;
      esac
      module load $mod
      runAllTests
      module unload $mod
    done
  fi
}

gccbuild_compilers() {
  for VcEnv in `find /opt/ -mindepth 2 -maxdepth 2 -name Vc.env`; do (
    . "$VcEnv"
    case "$VcEnv" in
      *-snapshot/Vc.env)
        ( cd $HOME/src/gcc-build && ./update.sh "`dirname "$VcEnv"`" )
        ;;
    esac
    runAllTests
  ) done
}

icc_compilers() {
  test -d /opt/intel || return
  export CC=icc
  export CXX=icpc
  icclist="`find /opt/intel/compiler* -name 'iccvars.sh' | xargs readlink -e | sort -ur`"
  case `uname -m` in
    x86_64)
      COMPILERVARS_ARCHITECTURE=intel64
      ;;
    i[345678]86)
      COMPILERVARS_ARCHITECTURE=ia32
      ;;
  esac
  export COMPILERVARS_ARCHITECTURE
  test -n "$icclist" && for IccEnv in $icclist; do (
    . $IccEnv $COMPILERVARS_ARCHITECTURE
    runAllTests
  ) done
}

system_compilers
modules_compilers
gccbuild_compilers
icc_compilers
