#!/bin/zsh

scriptName="${0##*/}"
function usage() {
  echo "Usage: $scriptName [--root <ROOT Prefix>]"
}

function fatal() {
  echo "${1:-Error. Quit.}" >&2
  exit 1
}

function readWithDefault() {
  local default="${(P)1}"
  test -n "$2" && echo -n "$2 " || echo -n "$1 "
  test -n "$default" && echo -n "[$default] "
  read $1
  test -z "${(P)1}" && eval ${1}="${default}" || eval ${1}="${(e)${(P)1}}"
}

function sourcesFor() {
  local pattern=$1
  local file=$2
  local output=$3
  list=()
  local inside=false
  for i in `grep -A20 "$pattern\>" "$file"`; do
    case "$i" in
      STATIC|SHARED|MODULE|EXCLUDE_FROM_ALL)
        ;;
      "$pattern")
        inside=true
        ;;
      *')')
        $inside && test -n "${i%)}" && list=(${list} ${i%)})
        inside=false
        ;;
      *)
        $inside && list=(${list} ${i})
        ;;
    esac
  done
  eval "${output}=("${(u)list[@]}")"
}

rootDir=
while [[ $# > 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit
      ;;
    -r|--root)
      if ! test -f "$2/core/base/inc/TObject.h"; then
        echo "$2/core/base/inc/TObject.h not found" >&2
        usage >&2
        exit 1
      fi
      rootDir="$2"
      shift
      ;;
  esac
  shift
done

if [[ -z "$rootDir" ]]; then
  rootDir="$HOME/src/root"
  readWithDefault rootDir "ROOT Sources"
fi
rootVcDir="$rootDir/misc/vc"

echo "Clean up $rootVcDir"
rm -r "$rootVcDir"

vcDir="`dirname "$0"`"
sourcesFor "add_library(Vc"    "$vcDir/CMakeLists.txt" libVc_files
pushd $vcDir
includes=({scalar,sse,avx}{,/*.{h,tcc}} common{,/*.h} include/Vc/**/*)
popd

mkdir -p $rootVcDir/{inc/Vc,src,test} || fatal "Failed to create directories inside ROOT"
for file in $includes; do
  src="$vcDir/$file"
  dst="$rootVcDir/inc/Vc/${file/include\/Vc\//}"
  if [[ -d "$src" ]]; then
    echo "mkdir $dst"
    mkdir -p "$dst" || fatal
  else
    echo "copying $dst"
    cp "$src" "$dst" || fatal
  fi
done

function copy() {
  while [[ $# > 0 ]]; do
    file="$1"; shift
    dstfile="src/${file//\//-}"
    src="$vcDir/$file"
    dst="$rootVcDir/$dstfile"
    echo "copying $dst"
    cp "$src" "$dst" || fatal
  done
}

copy "${libVc_files[@]}"

# TODO: copy cmake files for installation

# Read version number
eval `awk '/VC_VERSION_NUMBER 0x[0-9]+/ { h=$3 }
END {
major=strtonum(substr(h, 1, 4))
minor=strtonum("0x" substr(h, 5, 2))
patch=strtonum("0x" substr(h, 7, 2)) / 2
printf "vcVersion=\"%d.%d.%d\"\n", major, minor, patch
}' $vcDir/include/Vc/version.h`

rootVcVersion="${vcVersion%%-*}-root"
sed -i "s/${vcVersion}.*\"/$rootVcVersion\"/" $rootVcDir/inc/Vc/version.h

# TODO: generate $rootVcDir/Module.mk
cat > $rootVcDir/Module.mk <<EOF
# Module.mk for Vc module
# Generated on `date` by Vc/${scriptName}

MODNAME      := vc
VCVERS       := vc-${rootVcVersion}

MODDIR       := \$(ROOT_SRCDIR)/misc/\$(MODNAME)
MODDIRS      := \$(MODDIR)/src
MODDIRI      := \$(MODDIR)/inc
VCBUILDDIR   := build/misc/\$(MODNAME)


ifeq (\$(PLATFORM),win32)
#VCLIBVCA     := \$(call stripsrc,\$(MODDIRS)/win32/libVc-${vcVersion}.lib)
VCLIBVC      := \$(LPATH)/libVc.lib
else
VCLIBVC      := \$(LPATH)/libVc.a
endif

VCH          := \$(wildcard \$(MODDIRI)/Vc/*.h \$(MODDIRI)/Vc/*/*.h)

ALLHDRS      += \$(patsubst \$(MODDIRI)/%.h,include/%.h,\$(VCH))
ALLLIBS      += \$(VCLIBVC)

##### local rules #####
.PHONY:         all-\$(MODNAME) clean-\$(MODNAME) distclean-\$(MODNAME)

VCFLAGS      := -DVC_COMPILE_LIB \$(OPT) \$(CXXFLAGS) -Iinclude/Vc
VCLIBVCOBJ   := \$(patsubst \$(MODDIRS)/%.cpp,\$(VCBUILDDIR)/%.cpp.o,\$(wildcard \$(MODDIRS)/*.cpp))
ifndef AVXFLAGS
VCLIBVCOBJ   := \$(filter-out \$(MODDIRS)/avx-%.cpp,\$(VCLIBVCOBJ))
endif

\$(VCLIBVC): \$(VCLIBVCOBJ)
	\$(MAKEDIR)
	@echo "Create static library \$@"
	@ar r \$@ \$?
	@ranlib \$@

\$(VCBUILDDIR)/avx-%.cpp.o: \$(MODDIRS)/avx-%.cpp
	\$(MAKEDIR)
	@echo "Compiling (AVX) \$<"
	@\$(CXX) \$(VCFLAGS) \$(AVXFLAGS) -Iinclude/Vc/avx -c \$(CXXOUT)\$@ \$<

\$(VCBUILDDIR)/%.cpp.o: \$(MODDIRS)/%.cpp
	\$(MAKEDIR)
	@echo "Compiling \$<"
	@\$(CXX) \$(VCFLAGS) -c \$(CXXOUT)\$@ \$<

include/%.h: \$(MODDIRI)/%.h
	\$(MAKEDIR)
	cp \$< \$@

all-\$(MODNAME): \$(VCLIBVC)

clean-\$(MODNAME):
	@rm -f \$(VCLIBVC) \$(VCLIBVCOBJ)

clean:: clean-\$(MODNAME)

distclean-\$(MODNAME): clean-\$(MODNAME)
	@rm -rf include/Vc

distclean:: distclean-\$(MODNAME)

EOF

# vim: sw=2 noet ft=zsh
