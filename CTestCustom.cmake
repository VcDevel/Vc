set(CTEST_CUSTOM_WARNING_EXCEPTION ${CTEST_CUSTOM_WARNING_EXCEPTION}
   " C4723: " # MSVC 2012 can't suppress this warning
   " C4756: " # MSVC 2012 can't suppress this warning
   "used uninitialized in this function"
   "Skipping compilation of tests gatherStruct and gather2dim because of clang bug" # Not a helpful warning for the dashboard
   "warning is a GCC extension"
   "^-- "  # Ignore output from cmake
   "AVX disabled per default because of old/broken compiler" # This warning is meant for users not the dashboard
   "WARNING non-zero return value in ctest from: make" # Ignore output from ctest
   "ipo: warning #11010:" # Ignore warning about incompatible libraries with ICC -m32 on 64-bit system
   "include/qt4" # -Wuninitialized in QWeakPointer(X *ptr)
   " note: " # Notes are additional lines from errors (or warnings) that we don't want to count as additional warnings
   "clang: warning: argument unused during compilation: '-stdlib=libc"
   "clang 3.6.x miscompiles AVX code" # a preprocessor warning for users of Vc, irrelevant for the dashboard
   )

set(CTEST_CUSTOM_ERROR_EXCEPTION ${CTEST_CUSTOM_ERROR_EXCEPTION}
   "^ICECC"
   "^make\\[[1-9]\\]: "
   "^collect2: ld returned . exit status"
   "^make: \\*\\*\\* \\[.*\\] Error ")
