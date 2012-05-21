#!/bin/bash

case "$1" in
   Experimental|Nightly|Continuous)
      export dashboard_model=$1
      ;;
   *)
      echo "Usage: $0 <model>"
      echo
      echo "Possible arguments for model are Nightly, Continuous, or Experimental."
      echo
      exit 1
      ;;
esac

ctest -S "`dirname $0`/test.cmake" 2>&1 | grep -v 'Error in read script:'
