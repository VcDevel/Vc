set(CTEST_CUSTOM_WARNING_EXCEPTION ${CTEST_CUSTOM_WARNING_EXCEPTION}
   " C4723: " # MSVC 2012 can't suppress this warning
   "used uninitialized in this function"
   "Skipping compilation of tests gatherStruct and gather2dim because of clang bug" # Not a helpful warning for the dashboard
   "GCC < 4.3 does not have full support for SSE2 intrinsics." # Ignore self-made warning, though what I really want is a message when the warning is absent
   "call to .*Vc::Warnings::_operator_bracket_warning.* declared with attribute warning"
   "warning is a GCC extension"
   "^-- "  # Ignore output from cmake
   "GCC 4.6.0 is broken.  The following tests are therefore disabled" # This warning is meant for users not the dashboard
   "Your GCC is older than 4.4.6.  This is known to cause problems/bugs" # This warning is meant for users not the dashboard
   "GCC 4.4.x shows false positives for -Wparentheses, thus we rather disable" # This warning is meant for users not the dashboard
   "AVX disabled per default because of old/broken compiler" # This warning is meant for users not the dashboard
   "GCC 4.7.0 miscompiles at -O3, adding -fno-predictive-commoning to the" # This warning is meant for users not the dashboard
   "warning: the mangled name of .*typename Vc::{anonymous}::Decltype.* will change in a future version of GCC"
   "^\\*\\*\\* WARNING non-zero return value in ctest from: make") # Ignore output from ctest

set(CTEST_CUSTOM_ERROR_EXCEPTION ${CTEST_CUSTOM_ERROR_EXCEPTION}
   "^make\\[[1-9]\\]: "
   "^collect2: ld returned . exit status"
   "^make: \\*\\*\\* \\[all\\] Error ")
