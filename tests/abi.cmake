execute_process(
   COMMAND ${OBJDUMP} --no-show-raw-insn -dC -j .text ${BINARY}
   COMMAND grep -A3 " <test"
   COMMAND sed 1d
   COMMAND cut -d: -f2
   COMMAND xargs echo
   OUTPUT_VARIABLE asm)
string(STRIP "${asm}" asm)

# Note the regex parts looking for %esp can only match on ia32. (Not sure about x32, though)
# 'retq' on the other hand, can only match on x64
if("${IMPL}" STREQUAL SSE)
   set(reference "(^addps %xmm(0,%xmm1 movaps %xmm1,%xmm0 retq|1,%xmm0 retq)|vaddps %xmm(0,%xmm1|1,%xmm0),%xmm0 retq|.*v?movaps 0x[12]4\\(%esp\\),%xmm[01]( | .+ )v?(add|mova)ps 0x[12]4\\(%esp\\),%xmm[01])")
elseif("${IMPL}" STREQUAL AVX OR "${IMPL}" STREQUAL AVX2)
   set(reference "(^vaddps %ymm(0,%ymm1|1,%ymm0),%ymm0 retq|vmovaps 0x[24]4\\(%esp\\),%ymm[01]( | .+ )v(add|mova)ps 0x[24]4\\(%esp\\),%ymm[01],%ymm)")
elseif("${IMPL}" STREQUAL MIC)
   set(reference "^vaddps %zmm(0,%zmm1|1,%zmm0),%zmm0 retq")
else()
   message(FATAL_ERROR "Unknown IMPL '${IMPL}'")
endif()

if("${asm}" MATCHES "${reference}")
   message(STATUS "Passed.")
else()
   message(FATAL_ERROR "Failed. ('${asm}' does not match '${reference}')")
endif()
