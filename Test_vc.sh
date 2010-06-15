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

if [ "$#" -lt "2" ]; then
  echo ""
  echo "-- Error -- Please start script with two parameters"
  echo "-- Error -- The first parameter is the ctest model."
  echo "-- Error -- Possible arguments are Nightly and Experimental."
  echo "-- Error -- The second parameter is the file containg the"
  echo "-- Error -- information about the setup at the client"
  echo "-- Error -- installation (see example below)."
  echo ""
  print_example
  exit 1
fi

# test if a ctest model is either Experimantal or Nightly
if [ "$1" == "Experimental" -o "$1" == "Nightly" ]; then
  echo ""
else
  echo "-- Error -- This ctest model is not supported."
  echo "-- Error -- Possible arguments are Nightly or Experimental."
  exit 1
fi  

# test if the input file exists and execute it
if [ -e $2 ];then
  source $2
else
  echo "-- Error -- Input file does not exist."
  echo "-- Error -- Please choose existing input file."
  exit 1
fi

# set the ctest model to command line parameter
export ctest_model=$1

# test for architecture
arch=$(uname -s | tr '[A-Z]' '[a-z]')
chip=$(uname -m | tr '[A-Z]' '[a-z]')

# extract information about the system and the machine and set
# environment variables used by ctest
SYSTEM=$arch-$chip
if test -z $CXX ; then
  COMPILER=gcc;
  GCC_VERSION=$(gcc -dumpversion)
else
  COMPILER=$CXX;
  GCC_VERSION=$($CXX -dumpversion)
fi

export LABEL1=${LINUX_FLAVOUR}-$SYSTEM-$COMPILER$GCC_VERSION
export LABEL=$(echo $LABEL1 | sed -e 's#/#_#g')

# get the number of processors
# and information about the host
if [ "$arch" = "linux" ];
then
  export number_of_processors=$(cat /proc/cpuinfo | grep processor | wc -l)
  export SITE=$(hostname -f)
elif [ "$arch" = "darwin" ];
then
  export number_of_processors=$(sysctl -n hw.ncpu)
  export SITE=$(hostname -s)
fi

echo "************************"
date
echo "LABEL: " $LABEL
echo "SITE: " $SITE
echo "Model: " ${ctest_model}
echo "Nr. of processes: " $number_of_processors
echo "************************"

cd $VC_SOURCEDIR

ctest -S $VC_SOURCEDIR/VCTest.cmake -V --VV
