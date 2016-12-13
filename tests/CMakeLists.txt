include(AddFileDependencies)

add_definitions(-DCOMPILE_FOR_UNIT_TESTS)
if(Vc_COMPILER_IS_MSVC)
   AddCompilerFlag("/wd4267") # Disable warning "conversion from 'size_t' to 'int', possible loss of data"
   AddCompilerFlag("/wd4723") # Disable warning "potential divide by 0" (suppress doesn't work)
   AddCompilerFlag("/wd4503") # Disable warning "decorated name length exceeded, name was truncated"
endif()

# 32-bit x86 requires SSE for fp math to produce comparable results.
AddCompilerFlag("-mfpmath=sse" CXX_FLAGS Vc_ARCHITECTURE_FLAGS CXX_RESULT _fpmath_ok)

option(ENABLE_UBSAN "Enable build of unit tests with undefined behavior sanitizer" OFF)
if(ENABLE_UBSAN)
   AddCompilerFlag("-fsanitize=undefined")
endif()

if(Vc_COMPILER_IS_CLANG) # don't warn about sanity checks#{{{
   # GCC does not complain about the following flags until much later. Therefore
   # the AddCompilerFlag logic doesn't work as intended.
   AddCompilerFlag(-Wno-tautological-constant-out-of-range-compare)
   AddCompilerFlag(-Wno-tautological-compare)
endif()#}}}

function(get_target_executable VAR TARGET)#{{{
   get_target_property(exe "${TARGET}" OUTPUT_NAME)
   if(NOT exe)
      set(exe "${CMAKE_CURRENT_BINARY_DIR}/${TARGET}")
   endif()
   set(${VAR} "${exe}" PARENT_SCOPE)
endfunction()#}}}

# typeToString can use cxxabi for demangling
CHECK_CXX_SOURCE_COMPILES("#include <cxxabi.h>
int main() { return 0; }" cxx_abi_header_works)
if(cxx_abi_header_works)
   add_definitions(-DHAVE_CXX_ABI_H)
endif()

function(vc_target_setup name sde_cpuid)#{{{
   macro(_return_success)
      message(STATUS "Building tests for ${name}: enabled")
      list(APPEND vc_all_target_variants ${name})
      set(vc_all_target_variants ${vc_all_target_variants} PARENT_SCOPE)
      set(sde_cpuid_${name} "-${sde_cpuid}" PARENT_SCOPE)
      return()
   endmacro()

   if(${ARGC} EQUAL 2)
      _return_success()
   endif()
   foreach(flags ${ARGN})
      string(REPLACE "|" ";" flags "${flags}")
      set(result)
      AddCompilerFlag("${flags}" CXX_FLAGS result CXX_RESULT ok)
      if(ok)
         set(Vc_${name}_flags ${result} PARENT_SCOPE)
         _return_success()
      endif()
   endforeach()
   message(STATUS "Building tests for ${name}: disabled")
endfunction()#}}}

# sets Vc_nosimd_flags, Vc_sse2_flags, ...
if(Vc_X86)
   if(Vc_COMPILER_IS_MSVC)
      if(CMAKE_SYSTEM_PROCESSOR MATCHES 64)
         message(STATUS "Building tests for nosimd: disabled")
         # /arch:SSE2 is implied and an unknown option to cl
         vc_target_setup(sse2 mrm)
      else()
         vc_target_setup(nosimd mrm)
         # There's only /arch:SSE2, and no way to select ISA options for SSE3-4
         vc_target_setup(sse2 mrm "/arch:SSE2")
      endif()
   else()
      vc_target_setup(nosimd mrm "-mno-sse")
      vc_target_setup(sse2 mrm "-xSSE2" "-msse2")
      vc_target_setup(sse4_2 nhm "-xSSE4.2" "-msse4.2")
   endif()
   vc_target_setup(avx snb "-xAVX" "-mavx" "/arch:AVX")
   vc_target_setup(avx2 hsw "-xCORE-AVX2" "-mavx2|-mbmi|-mbmi2|-mlzcnt|-mfma" "/arch:AVX2")
   #vc_target_setup(mic "" "-mmic")
   vc_target_setup(knl knl "-xMIC-AVX512" "-march=knl" "/arch:KNL")
   vc_target_setup(avx512 skx "-xCORE-AVX512" "-march=skylake-avx512" "/arch:AVX512")
endif()

macro(vc_add_run_target _target emulator run_target_list)#{{{
   if("${CMAKE_GENERATOR}" MATCHES "Visual Studio")
      # do nothing. This just clutters the solution explorer
   else()
      if(emulator)
         get_target_executable(_exe "${_target}")
         add_custom_target(run_${_target}
            ${emulator} "${_exe}" "-v" # FIXME: MIC_NATIVELOAD needs `emulator exe -a -v`
            DEPENDS ${_target}
            COMMENT "Execute ${_target} test"
            VERBATIM
            )
      else()
         add_custom_target(run_${_target}
            ${_target} -v
            DEPENDS ${_target}
            COMMENT "Execute ${_target} test"
            VERBATIM
            )
      endif()
      list(APPEND ${run_target_list} "run_${target}")
   endif()
endmacro()#}}}

set(alias_strategies Union)#{{{
if(Vc_COMPILER_IS_CLANG OR Vc_COMPILER_IS_GCC)
   set(alias_strategies ${alias_strategies} MayAlias VectorBuiltin)
elseif(Vc_COMPILER_IS_INTEL)
   # ICC can parse (aka ignore) may_alias...
   set(alias_strategies ${alias_strategies} MayAlias)
elseif(Vc_COMPILER_IS_MSVC)
   set(alias_strategies ${alias_strategies} UnionMembers)
endif()#}}}

function(vc_add_test name) #{{{
   set(src "${name}.cpp")
   set(run_targets)
   foreach(impl ${vc_all_target_variants})
      set(_target "${name}_${impl}")
      list(FIND disabled_targets ${_target} disabled)
      if(NOT disabled EQUAL -1)
         continue()
      endif()
      set(flags "${Vc_COMPILE_FLAGS}")
      if(DEFINED Vc_${impl}_flags)
         list(APPEND flags "${Vc_${impl}_flags}")
      endif()
      string(TOUPPER "${impl}" label)
      set(use_var "USE_${label}")
      set(emulator FALSE)
      if(impl STREQUAL "nosimd")
         set(label NoSIMD)
         set(use_var TRUE)
      elseif(impl STREQUAL "mic")
         set(use_var "MIC_NATIVE_FOUND")
         if(MIC_NATIVELOAD)
            set(emulator ${MIC_NATIVELOAD})
         endif()
      elseif(Vc_X86 AND INTEL_SDE AND NOT ${use_var})
         set(use_var TRUE)
         set(emulator "${INTEL_SDE};${sde_cpuid_${impl}};--")
      endif()
      foreach(types "ldouble,float,double,schar,uchar" "llong,long,ullong,ulong" "int,short,uint,ushort")
         set(types_flag "TESTTYPES=${types}")
         foreach(alias ${alias_strategies})
            if(impl STREQUAL "nosimd")#{{{
               set(target ${_target})
            else()
               string(TOLOWER "${_target}_${alias}" target)
               string(TOUPPER ${alias} alias_flag)
               set(alias_flag "Vc_USE_ALIASSTRATEGY_${alias_flag}")
            endif()#}}}
            string(REPLACE "," "_" target "${target}_${types}")
            if(impl STREQUAL "mic")
               set(_objs)#{{{
               if("${_name}" STREQUAL "trigonometric")
                  foreach(fun sincos asin acos atan)
                     foreach(filename reference-${fun}-sp.dat reference-${fun}-dp.dat)
                        set(_objs ${_objs} ${CMAKE_CURRENT_BINARY_DIR}/${filename}.k1om.o)
                     endforeach()
                  endforeach()
               elseif("${_name}" STREQUAL "logarithm")
                  foreach(fun ln log2 log10)
                     foreach(filename reference-${fun}-sp.dat reference-${fun}-dp.dat)
                        set(_objs ${_objs} ${CMAKE_CURRENT_BINARY_DIR}/${filename}.k1om.o)
                     endforeach()
                  endforeach()
               endif()
               mic_add_executable(${target}
                  LINK_LIBRARIES Vc_MIC
                  COMPILE_FLAGS "-DNO_ISA_CHECK" "-D${types_flag}"
                  SOURCES ${src}
                  OBJECTS ${_objs}
                  EXCLUDE_FROM_ALL
                  )#}}}
            else()
               add_executable(${target} EXCLUDE_FROM_ALL ${src})
               #target_link_libraries(${target} Vc)
               set_property(TARGET ${target} APPEND PROPERTY COMPILE_OPTIONS "${flags}")
               if(NOT impl STREQUAL "nosimd")
                  set_property(TARGET ${target} APPEND PROPERTY COMPILE_DEFINITIONS "${alias_flag}")
               endif()
               set_property(TARGET ${target} APPEND PROPERTY COMPILE_DEFINITIONS "NO_ISA_CHECK")
               set_property(TARGET ${target} APPEND PROPERTY COMPILE_DEFINITIONS "${types_flag}")
            endif()
            add_target_property(${target} LABELS "${label}")
            add_dependencies(build_tests ${target})
            add_dependencies(${label} ${target})
            if(${use_var}) # add test and run target#{{{
               if(emulator)
                  get_target_executable(_exe "${target}")
                  add_test(NAME ${target}
                     COMMAND ${emulator} "${_exe}"
                     )
               else()
                  add_test(${target} "${CMAKE_CURRENT_BINARY_DIR}/${target}")
               endif()
               set_property(TEST ${target} PROPERTY LABELS "${label}")
               vc_add_run_target(${target} "${emulator}" run_targets)
            endif()#}}}
            if(impl STREQUAL "nosimd")
               break()
            endif()
         endforeach()
      endforeach()
   endforeach()
   if(run_targets)
      add_custom_target(run_${name}_all
         COMMENT "Execute all ${name} tests"
         VERBATIM
         )
      add_dependencies(run_${name}_all ${run_targets})
   endif()
endfunction() #}}}

vc_add_test(loadstore)
vc_add_test(datapar_mask)
vc_add_test(datapar)
vc_add_test(where)

function(vc_download_testdata)#{{{
   set(_deps)
   foreach(fun sincos asin acos atan ln log2 log10)
      foreach(filename reference-${fun}-sp.dat reference-${fun}-dp.dat)
         add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${filename}"
            COMMAND ${CMAKE_COMMAND} -Dfilename=${filename} -P ${CMAKE_CURRENT_SOURCE_DIR}/download.cmake
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/download.cmake
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Downloading Test Data: ${filename}"
            VERBATIM
            )
         if(MIC_OBJCOPY)
            add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${filename}.k1om.o"
               COMMAND ${MIC_OBJCOPY} -I binary -O elf64-k1om -B k1om ${filename} ${filename}.k1om.o
               DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${filename}"
               WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
               COMMENT "Converting Test Data: ${filename}.k1om.o"
               VERBATIM
               )
            list(APPEND _deps "${CMAKE_CURRENT_BINARY_DIR}/${filename}.k1om.o")
         endif()
         list(APPEND _deps "${CMAKE_CURRENT_BINARY_DIR}/${filename}")
      endforeach()
   endforeach()
   add_custom_target(download-testdata ALL
      DEPENDS ${_deps}
      )

   add_dependencies(NoSIMD download-testdata)
   if(Vc_X86)
      add_dependencies(SSE2 download-testdata)
      add_dependencies(SSE4_2 download-testdata)
      add_dependencies(AVX download-testdata)
      add_dependencies(AVX2 download-testdata)
      add_dependencies(MIC download-testdata)
      add_dependencies(KNL download-testdata)
      add_dependencies(AVX512 download-testdata)
   elseif(Vc_ARM)
      add_dependencies(NEON download-testdata)
   endif()
endfunction()
vc_download_testdata()#}}}

# ABI tests#{{{
find_program(OBJDUMP objdump)
mark_as_advanced(OBJDUMP)
if(OBJDUMP)
   macro(test_abi _target _impl)
      set(_test test_${_target})
      if("${_impl}" STREQUAL "MIC")
         mic_add_executable(${_target} SOURCES abi.cpp LINK_LIBRARIES Vc_MIC COMPILE_FLAGS "-DVc_IMPL=${_impl};${ARGN};-O2")
      else()
         add_executable(${_target} EXCLUDE_FROM_ALL abi.cpp)
         #target_link_libraries(${_target} Vc)
         set_property(TARGET ${_target} APPEND PROPERTY COMPILE_DEFINITIONS "Vc_IMPL=${_impl}")
         set_property(TARGET ${_target} APPEND PROPERTY COMPILE_OPTIONS "${ARGN};-O2")
      endif()
      add_target_property(${_target} LABELS "${_impl}")
      add_dependencies(${_impl} ${_target})
      add_dependencies(build_tests ${_target})

      get_target_executable(_exe "${_target}")

      add_test(NAME ${_test}
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
         COMMAND ${CMAKE_COMMAND} -DOBJDUMP=${OBJDUMP} -DBINARY=${_exe} -DIMPL=${_impl}
         -DCOMPILER_IS_CLANG=${Vc_COMPILER_IS_CLANG} -DSYSTEM_NAME=${CMAKE_SYSTEM_NAME}
         -P ${CMAKE_CURRENT_SOURCE_DIR}/abi.cmake
         )
      set_property(TEST ${_test} PROPERTY LABELS "${_impl}")
      add_custom_target(run_${_target}
         ${CMAKE_COMMAND} -DOBJDUMP=${OBJDUMP} -DBINARY=${_exe} -DIMPL=${_impl}
         -DCOMPILER_IS_CLANG=${Vc_COMPILER_IS_CLANG} -DSYSTEM_NAME=${CMAKE_SYSTEM_NAME}
         -P ${CMAKE_CURRENT_SOURCE_DIR}/abi.cmake
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
         DEPENDS ${_target}
         COMMENT "Execute ${_target} test"
         VERBATIM
         )
   endmacro()

   if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "[x3-7]86")
      test_abi(abi_SSE     SSE  ${Vc_sse2_flags})
      test_abi(abi_SSE_AVX SSE  ${Vc_avx_flags})
      if(NOT Vc_AVX_INTRINSICS_BROKEN)
         test_abi(abi_AVX     AVX  ${Vc_avx_flags})
         test_abi(abi_AVX2    AVX2 ${Vc_avx2_flags})
      endif()
   endif()
   if(MIC_NATIVE_FOUND)
      find_program(MIC_OBJDUMP x86_64-k1om-linux-objdump PATHS ENV PATH "${MIC_TARGET_TOOLS_DIR}/bin")
      mark_as_advanced(MIC_OBJDUMP)
      if(MIC_OBJDUMP)
         set(OBJDUMP "${MIC_OBJDUMP}")
         test_abi(abi_MIC MIC)
      endif()
   endif()
endif()#}}}

macro(vc_add_general_test _name)#{{{
   add_executable(${_name} EXCLUDE_FROM_ALL ${_name}.cpp )
   target_link_libraries(${_name} Vc)
   add_target_property(${_name} LABELS "other")
   set_property(TARGET ${_name} APPEND PROPERTY COMPILE_OPTIONS "${Vc_ARCHITECTURE_FLAGS}")
   add_dependencies(build_tests ${_name})
   add_dependencies(other ${_name})
   add_test(${_name} "${CMAKE_CURRENT_BINARY_DIR}/${_name}")
   set_property(TEST ${_name} PROPERTY LABELS "other")
   vc_add_run_target(${_name} FALSE _ignore)
endmacro()#}}}

if(NOT Vc_COMPILER_IS_MSVC)
   if(_last_target_arch STREQUAL "auto" AND NOT Vc_AVX_INTRINSICS_BROKEN)
      vc_add_general_test(supportfunctions)
   endif()
   vc_add_general_test(alignmentinheritance)
   vc_add_general_test(alignedbase)

# compile and link test for targets that need to link lots of stuff together#{{{
add_library(linkTestLibDynamic1 SHARED EXCLUDE_FROM_ALL linkTestLib0.cpp linkTestLib1.cpp)
add_library(linkTestLibDynamic2 SHARED EXCLUDE_FROM_ALL linkTestLib0.cpp linkTestLib1.cpp)
add_library(linkTestLibStatic STATIC EXCLUDE_FROM_ALL linkTestLib2.cpp linkTestLib3.cpp)
add_executable(linkTest EXCLUDE_FROM_ALL linkTest0.cpp linkTest1.cpp)
add_dependencies(build_tests linkTest)
add_dependencies(other linkTest)
target_link_libraries(linkTestLibDynamic1 Vc)
target_link_libraries(linkTestLibDynamic2 Vc)
add_target_property(linkTestLibDynamic1 COMPILE_FLAGS "-DPOSTFIX=A")
add_target_property(linkTestLibDynamic2 COMPILE_FLAGS "-DPOSTFIX=B")
set_property(TARGET linkTestLibDynamic1 APPEND PROPERTY COMPILE_OPTIONS "${Vc_ARCHITECTURE_FLAGS}")
set_property(TARGET linkTestLibDynamic2 APPEND PROPERTY COMPILE_OPTIONS "${Vc_ARCHITECTURE_FLAGS}")
set_property(TARGET linkTestLibStatic APPEND PROPERTY COMPILE_OPTIONS "${Vc_ARCHITECTURE_FLAGS}")
set_property(TARGET linkTest APPEND PROPERTY COMPILE_OPTIONS "${Vc_ARCHITECTURE_FLAGS}")
target_link_libraries(linkTestLibStatic Vc)
target_link_libraries(linkTest Vc linkTestLibDynamic1 linkTestLibDynamic2 linkTestLibStatic)#}}}
endif()

# Use the following program to generate the sincos-reference-*.dat files
#add_executable(convert-sincos-reference EXCLUDE_FROM_ALL convert-sincos-reference.cpp)

# vim: foldmethod=marker commentstring=#%s
