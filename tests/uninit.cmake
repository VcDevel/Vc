#######################################################################
# test that uninitialized datapar/mask produces no code
#######################################################################

foreach(variant native_datapar native_mask fixed_datapar)
   # fixed_mask uses bitset, which always initializes to null
   execute_process(
      COMMAND ${OBJDUMP} --no-show-raw-insn -dC -j .text ${BINARY}
      COMMAND awk -v "RS=" "/ <test_${variant}/"
      COMMAND sed 1d
      COMMAND cut -d: -f2
      COMMAND xargs echo
      OUTPUT_VARIABLE asm)
   string(STRIP "${asm}" asm)

   if("${asm}" MATCHES "retq")
      set(x86_32 FALSE)
      set(reference "^(repz )?retq")
   else()
      set(x86_32 TRUE)
      set(reference "Not implemented yet. Please fill in the reference regex.")
   endif()

   if("${asm}" MATCHES "${reference}")
      if(expect_failure)
         message(FATAL_ERROR "Warning: unexpected pass. The test was flagged as EXPECT_FAILURE but passed instead.")
      else()
         message("PASS: ${variant}<float> uninitialized")
      endif()
   elseif(expect_failure)
      message("Expected Failure.\n'${asm}'\n  does not match\n'${reference}'")
   else()
      message(FATAL_ERROR "Failed.\n'${asm}'\n  does not match\n'${reference}'")
   endif()
endforeach()
