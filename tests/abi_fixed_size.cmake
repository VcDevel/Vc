#######################################################################
# test fixed_size_datapar<T> ABI
#######################################################################

foreach(fun 1 2 4 8 Max)
   execute_process(
      COMMAND ${OBJDUMP} --no-show-raw-insn -dC -j .text ${BINARY}
      COMMAND awk -v "RS=" "/ <test${fun}/"
      COMMAND sed 1d
      COMMAND cut -d: -f2
      COMMAND xargs echo
      OUTPUT_VARIABLE asm)
   string(STRIP "${asm}" asm)
   string(REGEX REPLACE " (vzeroupper )?ret.*$" "" asm "${asm}")

   if("${asm}" MATCHES "%esp")
      set(x86_32 TRUE)
      set(reference "mov[a-z]* +\\(%[a-z]+\\),%.*mov[a-z]* +%[a-z0-9]+,\\(%[a-z]+\\)")
   else()
      set(x86_32 FALSE)
      set(reference "mov[a-z0-9]* +\\(%rsi\\),%.*mov[a-z0-9]* +%[a-z0-9]+,\\(%rdi\\)")
   endif()

   if("${asm}" MATCHES "${reference}")
      message("PASS: datapar<T, fixed_size<${fun}>> ABI")
   else()
      message(FATAL_ERROR "Failed.\n'${asm}'\n  does not match\n'${reference}'")
   endif()
endforeach()
