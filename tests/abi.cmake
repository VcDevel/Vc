#######################################################################
# test native_datapar<T> ABI
#######################################################################

execute_process(
   COMMAND ${OBJDUMP} --no-show-raw-insn -dC -j .text ${BINARY}
   COMMAND grep -A3 " <test"
   COMMAND sed 1d
   COMMAND cut -d: -f2
   COMMAND xargs echo
   OUTPUT_VARIABLE asm)
string(STRIP "${asm}" asm)

if("${asm}" MATCHES "%esp")
   set(x86_32 TRUE)
else()
   set(x86_32 FALSE)
endif()
set(expect_failure FALSE)

# Note the regex parts looking for %esp can only match on ia32. (Not sure about x32, though)
# 'retq' on the other hand, can only match on x64
if("${IMPL}" STREQUAL SSE)
   set(reference "(^addps %xmm(0,%xmm1 movaps %xmm1,%xmm0 retq|1,%xmm0 retq)|vaddps %xmm(0,%xmm1|1,%xmm0),%xmm0 retq|.*v?movaps 0x[12]4\\(%esp\\),%xmm[01]( | .+ )v?(add|mova)ps 0x[12]4\\(%esp\\),%xmm[01])")
   if(x86_32 AND COMPILER_IS_CLANG AND "${SYSTEM_NAME}" STREQUAL "Linux")
      set(expect_failure TRUE)
   endif()
elseif("${IMPL}" STREQUAL AVX OR "${IMPL}" STREQUAL AVX2)
   set(reference "(^vaddps %ymm(0,%ymm1|1,%ymm0),%ymm0 retq|vmovaps 0x[24]4\\(%esp\\),%ymm[01]( | .+ )v(add|mova)ps 0x[24]4\\(%esp\\),%ymm[01],%ymm)")
   if(x86_32 AND COMPILER_IS_CLANG AND "${SYSTEM_NAME}" STREQUAL "Linux")
      set(expect_failure TRUE)
   endif()
elseif("${IMPL}" STREQUAL MIC)
   set(reference "^vaddps %zmm(0,%zmm1|1,%zmm0),%zmm0 retq")
elseif("${IMPL}" STREQUAL AVX512 OR "${IMPL}" STREQUAL KNL)
   set(reference "^vaddps %zmm(0,%zmm1|1,%zmm0),%zmm0 retq")
else()
   message(FATAL_ERROR "Unknown IMPL '${IMPL}'")
endif()

if("${asm}" MATCHES "${reference}")
   if(expect_failure)
      message(FATAL_ERROR "Warning: unexpected pass. The test was flagged as EXPECT_FAILURE but passed instead.")
   else()
      message("PASS: Vector<T> ABI")
   endif()
elseif(expect_failure)
   message("Expected Failure.\n'${asm}'\n  does not match\n'${reference}'")
else()
   message(FATAL_ERROR "Failed.\n'${asm}'\n  does not match\n'${reference}'")
endif()

#######################################################################
# test Mask<T> ABI
#######################################################################
if("${IMPL}" STREQUAL MIC)
   set(reference "%di,.*%si,") # needs to read from %di and %si
endif()

execute_process(
   COMMAND ${OBJDUMP} --no-show-raw-insn -dC -j .text ${BINARY}
   COMMAND grep -A3 " <mask_test"
   COMMAND sed 1d
   COMMAND cut -d: -f2
   COMMAND xargs echo
   OUTPUT_VARIABLE asm)
string(STRIP "${asm}" asm)
if("${IMPL}" STREQUAL AVX512 OR "${IMPL}" STREQUAL KNL)
   set(reference "^(and %e[sd]i,%e[sd]i mov %e[sd]i|mov %e[sd]i,%eax and %e[sd]i),%eax ret")
else()
   string(REPLACE "add" "and" reference "${reference}")
endif()
if("${asm}" MATCHES "${reference}")
   if(expect_failure)
      message(FATAL_ERROR "Warning: unexpected pass. The test was flagged as EXPECT_FAILURE but passed instead.")
   else()
      message("PASS: Mask<T> ABI")
   endif()
elseif(expect_failure)
   message("Expected Failure.\n'${asm}'\n  does not match\n'${reference}'")
else()
   message(FATAL_ERROR "Failed.\n'${asm}'\n  does not match\n'${reference}'")
endif()
