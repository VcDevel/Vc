#!/bin/bash


function print_example(){
echo "##################################################################"
echo "# To set the required parameters for the source and the build    #"
echo "# directory for ctest, put the export commands below to a        #"
echo "# separate file which is read during execution and which is      #"
echo "# defined on the command line.                                   #"
echo "# Set all parameters according to your needs.                    #"
echo "# LINUX_FLAVOUR should be set to the distribution you are using  #"
echo "# eg Debian, SuSe etc.                                           #"
echo "# This information is only needed to get some information about  #"
echo "# systems which show problems                                    #"
echo "# For example                                                    #"
echo "#!/bin/bash                                                      #"
echo "#export LINUX_FLAVOUR=Etch32                                     #"
echo "#export VC_BUILDDIR=<path_to_your_build_directory>               #"
echo "#export VC_SOURCEDIR=<path_to_source_directory_of_vc>            #"
echo "##################################################################"
}

if [ "$#" -lt "1" ]; then
  echo "Usage: $0 <Experimental|Nightly|Continuous> [config]"
  echo ""
  echo "The first parameter is the ctest model:"
  echo "Possible arguments are Nightly, Continuous, or Experimental."
  echo ""
  echo "The second parameter is optional. You may specify a file"
  echo "containg the information about the setup at the client"
  echo "installation (see example below)."
  echo ""
  print_example
  exit 1
fi

# test if a ctest model is either Experimantal or Nightly
if [ "$1" == "Experimental" -o "$1" == "Nightly" -o "$1" == "Continuous" ]; then
  echo ""
else
  echo "-- Error -- The CTest model \"$1\" is not supported."
  echo "-- Error -- Possible arguments are Nightly, Continuous, or Experimental."
  exit 1
fi  

# test if the input file exists and execute it
test -f "$2" && source "$2"

# set the ctest model to command line parameter
export ctest_model=$1

# test for architecture
arch=$(uname -s | tr '[A-Z]' '[a-z]')
chip=$(uname -m | tr '[A-Z]' '[a-z]')

# extract information about the system and the machine and set
# environment variables used by ctest
SYSTEM=$arch-$chip
if test -z "$CXX" ; then
  COMPILER="`g++ --version|head -n1`"
else
  COMPILER="`"$CXX" --version|head -n1`"
fi
branch=`cat .git/HEAD|cut -d/ -f3`

LABEL1="$branch $arch $chip $COMPILER $CXXFLAGS"
if test "$arch" = "linux"; then
  test -z "$LINUX_FLAVOUR" && LINUX_FLAVOUR=`lsb_release -d`
  if test -n "$LINUX_FLAVOUR"; then
    LABEL1="$branch $LINUX_FLAVOUR $chip $COMPILER $CXXFLAGS"
  fi
fi
export LABEL=$(echo $LABEL1 | sed -e 's#/#_#g')

# get the number of processors
# and information about the host
if [ "$arch" = "linux" ];
then
  export number_of_processors=$(cat /proc/cpuinfo | grep processor | wc -l)
  export SITE=$(hostname -f 2>/dev/null || hostname)
elif [ "$arch" = "darwin" ];
then
  export number_of_processors=$(sysctl -n hw.ncpu)
  export SITE=$(hostname -s)
fi

echo "************************"
date
echo "LABEL: $LABEL"
echo "SITE:  $SITE"
echo "Model: ${ctest_model}"
echo "Nr. of processes: " $number_of_processors
echo "************************"

test -z "$VC_SOURCEDIR" && VC_SOURCEDIR="`dirname $0`"
cd "$VC_SOURCEDIR"
export VC_SOURCEDIR="$PWD" # making sure VC_SOURCEDIR is an absolute path

test -z "$VC_BUILDDIR" && export VC_BUILDDIR="$VC_SOURCEDIR/build-${ctest_model}-${LABEL//[\[\] ()]/_}"
test -d "$VC_BUILDDIR" || mkdir -p "$VC_BUILDDIR"

ctest -S VCTest.cmake -V --VV
