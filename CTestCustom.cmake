set(CTEST_CUSTOM_WARNING_EXCEPTION ${CTEST_CUSTOM_WARNING_EXCEPTION}
   "used uninitialized in this function"
   "GCC < 4.3 does not have full support for SSE2 intrinsics." # Ignore self-made warning, though what I really want is a message when the warning is absent
   "call to .*Vc::Warnings::_operator_bracket_warning.* declared with attribute warning"
   "warning is a GCC extension"
   "^-- "  # Ignore output from cmake
   "suggest parentheses around .+&&.+ within .+\\|\\|.+" # GCC 4.4 gives bogus warnings
   "^\\*\\*\\* WARNING non-zero return value in ctest from: make") # Ignore output from ctest

set(CTEST_CUSTOM_ERROR_EXCEPTION ${CTEST_CUSTOM_ERROR_EXCEPTION}
   "^make\\[[1-9]\\]: "
   "^collect2: ld returned . exit status"
   "^make: \\*\\*\\* \\[all\\] Error ")
