include(AddFileDependencies)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/virtest)

execute_process(
   COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=liblto_plugin.so
   OUTPUT_VARIABLE lto_plugin
   RESULT_VARIABLE found_lto_plugin
   OUTPUT_STRIP_TRAILING_WHITESPACE
   )
if(${found_lto_plugin} EQUAL 0)
   execute_process(COMMAND ${CMAKE_AR} --plugin ${lto_plugin} --version
      OUTPUT_VARIABLE tmp
      ERROR_VARIABLE tmp
      RESULT_VARIABLE ar_plugin_switch_available)
   if(${ar_plugin_switch_available} EQUAL 0)
      set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> qc --plugin ${lto_plugin} <TARGET> <LINK_FLAGS> <OBJECTS>")
      set(CMAKE_CXX_ARCHIVE_APPEND "<CMAKE_AR> q  --plugin ${lto_plugin} <TARGET> <LINK_FLAGS> <OBJECTS>")
      set(CMAKE_CXX_ARCHIVE_FINISH "<CMAKE_RANLIB> --plugin ${lto_plugin} <TARGET>")
   endif()
   AddCompilerFlag(-flto=4) # Hopefully faster "code build test" cycles
endif()

if(NOT "$ENV{TRAVIS_OS_NAME}" STREQUAL "")
   set(CMAKE_CXX_FLAGS_RELEASE "-O0 -DNDEBUG") # try to not time out, -O0 is faster
endif()

add_definitions(-DCOMPILE_FOR_UNIT_TESTS)
if(Vc_COMPILER_IS_MSVC)
   AddCompilerFlag("/wd4267") # Disable warning "conversion from 'size_t' to 'int', possible loss of data"
   AddCompilerFlag("/wd4723") # Disable warning "potential divide by 0" (suppress doesn't work)
   AddCompilerFlag("/wd4503") # Disable warning "decorated name length exceeded, name was truncated"
   AddCompilerFlag("/wd4752") # Disable warning "found Intel(R) Advanced Vector Extensions; consider using /arch:AVX"
   AddCompilerFlag("/bigobj") # work around "fatal error C1128: number of sections exceeded object file format limit: compile with /bigobj"
endif()
AddCompilerFlag("-fmax-errors=10") # Don't go too crazy on errors

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
elseif(Vc_ARM)
   vc_target_setup(neon "" "-march=armv8-a")
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

#alias_strategies{{{
if(APPLE)
   set(alias_strategies) # compiles too long and is not needed
else()
   set(alias_strategies Union)
endif()
if(Vc_COMPILER_IS_CLANG OR Vc_COMPILER_IS_GCC)
   list(APPEND alias_strategies MayAlias VectorBuiltin)
elseif(Vc_COMPILER_IS_INTEL)
   # ICC can parse (aka ignore) may_alias, the result is aliasing violations => UB => test failures. So skip it
elseif(Vc_COMPILER_IS_MSVC)
   list(APPEND alias_strategies UnionMembers)
endif()#}}}

function(vc_add_test name) #{{{
   if($ENV{VcSkipTests} MATCHES "${name}")
      return()
   endif()
   set(_only "$ENV{VcOnlyTests}")
   if(NOT ("${_only}" STREQUAL "" OR "${_only}" MATCHES "${name}"))
      return()
   endif()

   macro(check_target_disabled _t)
      list(FIND disabled_targets ${${_t}} disabled)
      if(disabled EQUAL -1)
         set(disabled FALSE)
      else()
         set(disabled TRUE)
         if(NOT DEFINED printed_${${_t}})
            message(STATUS "Skipping unit test ${${_t}}. Disabled.")
            set(printed_${${_t}} TRUE)
         endif()
      endif()
   endmacro()

   cmake_parse_arguments("" "NO_TESTTYPES;SINGLE_TESTTYPE;EXCLUDE_FROM_ALL" "SOURCE" "DEFINITIONS" ${ARGN})

   if(_SOURCE)
      set(src "${_SOURCE}")
   else()
      set(src "${name}.cpp")
   endif()

   set(run_targets)
   foreach(impl ${vc_all_target_variants})
      set(_target "${name}_${impl}")
      check_target_disabled(_target)
      if(disabled)
         continue()
      endif()
      set(flags "${Vc_COMPILE_FLAGS}")
      if(DEFINED Vc_${impl}_flags)
         list(APPEND flags "${Vc_${impl}_flags}")
      endif()
      string(REPLACE ";" " " link_flags "${flags}")
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
      if(_NO_TESTTYPES)
         set(type_split " ")
      elseif(_SINGLE_TESTTYPE)
         set(type_split "ldouble" "float" "double" "uint" "llong" "long" "uchar" "int" "ushort" "short" "ulong" "schar" "ullong")
      elseif(Vc_COMPILER_IS_MSVC)
         set(type_split "ldouble,float,double" "uint,llong" "long,uchar" "int,ushort" "short,ulong" "schar,ullong")
      else()
         set(type_split "ldouble,float,double,schar,uchar" "llong,long,ullong,ulong" "int,short,uint,ushort")
      endif()
      foreach(types ${type_split})
         foreach(alias ${alias_strategies})
            if(impl STREQUAL "nosimd")#{{{
               set(target ${_target})
            else()
               string(TOLOWER "${_target}_${alias}" target)
               string(TOUPPER ${alias} alias_flag)
               set(alias_flag "Vc_USE_ALIASSTRATEGY_${alias_flag}")
            endif()#}}}
            check_target_disabled(target)
            if(disabled)
               continue()
            endif()
            if(NOT _NO_TESTTYPES)
               string(REPLACE "," "_" target "${target}_${types}")
            endif()
            check_target_disabled(target)
            if(disabled)
               continue()
            endif()
            if(impl STREQUAL "mic")
               set(_objs)#{{{
               if(_NO_TESTTYPES)
                  unset(types_flag)
               else()
                  set(types_flag "-DTESTTYPES=${types}")
               endif()
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
                  COMPILE_FLAGS "-DNO_ISA_CHECK" "${types_flag}"
                  SOURCES ${src}
                  OBJECTS ${_objs}
                  EXCLUDE_FROM_ALL
                  )#}}}
            else()
               add_executable(${target} EXCLUDE_FROM_ALL ${src})
               set_property(TARGET ${target} APPEND PROPERTY COMPILE_DEFINITIONS "${_DEFINITIONS}")
               set_property(TARGET ${target} APPEND PROPERTY COMPILE_OPTIONS "${flags}")
               set_property(TARGET ${target} APPEND PROPERTY LINK_FLAGS "${link_flags}")
               if(NOT impl STREQUAL "nosimd")
                  set_property(TARGET ${target} APPEND PROPERTY COMPILE_DEFINITIONS "${alias_flag}")
               endif()
               set_property(TARGET ${target} APPEND PROPERTY COMPILE_DEFINITIONS "NO_ISA_CHECK")
               if(NOT _NO_TESTTYPES)
                  set_property(TARGET ${target} APPEND PROPERTY COMPILE_DEFINITIONS "TESTTYPES=${types}")
               endif()
            endif()
            if(NOT _EXCLUDE_FROM_ALL)
               add_target_property(${target} LABELS "${label}")
               add_dependencies(build_tests ${target})
               add_dependencies(${label} ${target})
            endif()
            if(${use_var}) # add test and run target#{{{
               if(NOT _EXCLUDE_FROM_ALL)
                  if(emulator)
                     get_target_executable(_exe "${target}")
                     add_test(NAME ${target}
                        COMMAND ${emulator} "${_exe}"
                        )
                  else()
                     add_test(${target} "${CMAKE_CURRENT_BINARY_DIR}/${target}")
                  endif()
                  set_property(TEST ${target} PROPERTY LABELS "${label}")
               endif()
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

vc_add_test(sfinae NO_TESTTYPES)
vc_add_test(loadstore SINGLE_TESTTYPE DEFINITIONS ONE_RANDOM_CONVERTING_LOAD_STORE)
vc_add_test(allcvt_loadstore SINGLE_TESTTYPE SOURCE loadstore.cpp SINGLE_TESTTYPE EXCLUDE_FROM_ALL)
vc_add_test(mask)
vc_add_test(datapar)
vc_add_test(operators)
vc_add_test(math)
vc_add_test(where)
vc_add_test(experimental)
vc_add_test(bitset_conversions)

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
#Not needed yet: vc_download_testdata()#}}}

# ABI tests#{{{
find_program(OBJDUMP objdump)
mark_as_advanced(OBJDUMP)
if(OBJDUMP AND NOT ENABLE_UBSAN) # ubsan changes codegen too much, just skip it
   set(run_targets)
   macro(test_abi target _impl abi)
      foreach(subtype native fixed_size)
         if("${subtype}" STREQUAL "fixed_size")
            set(suffix "_fixed_size")
         else()
            set(suffix "")
         endif()
         set(_target "${target}${suffix}")
         set(_test test_${_target})
         if("${_impl}" STREQUAL "MIC")
            mic_add_executable(${_target} SOURCES abi${suffix}.cpp LINK_LIBRARIES Vc_MIC COMPILE_FLAGS "-DVc_ABI=${abi};${ARGN};-O2")
         else()
            add_executable(${_target} EXCLUDE_FROM_ALL abi${suffix}.cpp)
            set_property(TARGET ${_target} APPEND PROPERTY COMPILE_DEFINITIONS "Vc_ABI=${abi}")
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
            -P ${CMAKE_CURRENT_SOURCE_DIR}/abi${suffix}.cmake
            )
         set_property(TEST ${_test} PROPERTY LABELS "${_impl}")
         add_custom_target(run_${_target}
            ${CMAKE_COMMAND} -DOBJDUMP=${OBJDUMP} -DBINARY=${_exe} -DIMPL=${_impl}
            -DCOMPILER_IS_CLANG=${Vc_COMPILER_IS_CLANG} -DSYSTEM_NAME=${CMAKE_SYSTEM_NAME}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/abi${suffix}.cmake
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS ${_target}
            COMMENT "Execute ${_target} test"
            VERBATIM
            )
         list(APPEND run_targets "run_${_target}")
      endforeach()
   endmacro()

   if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "[x3-7]86")
      test_abi(abi_SSE     SSE sse ${Vc_sse2_flags})
      test_abi(abi_SSE_AVX SSE sse ${Vc_avx_flags})
      if(NOT Vc_AVX_INTRINSICS_BROKEN)
         test_abi(abi_AVX     AVX avx ${Vc_avx_flags})
         test_abi(abi_AVX2    AVX2 avx ${Vc_avx2_flags})
         test_abi(abi_AVX512  AVX512 avx512 ${Vc_avx512_flags})
         test_abi(abi_KNL     KNL avx512 ${Vc_knl_flags})
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

   if(run_targets)
      add_custom_target(run_abi_all
         COMMENT "Execute all ABI tests"
         VERBATIM
         )
      add_dependencies(run_abi_all ${run_targets})
   endif()
endif()#}}}

# compile and link test for targets that need to link lots of stuff together#{{{
add_library(linkTestLibDynamic1 SHARED EXCLUDE_FROM_ALL linkTestLib0.cpp linkTestLib1.cpp)
add_library(linkTestLibDynamic2 SHARED EXCLUDE_FROM_ALL linkTestLib0.cpp linkTestLib1.cpp)
add_library(linkTestLibStatic STATIC EXCLUDE_FROM_ALL linkTestLib2.cpp linkTestLib3.cpp)
add_executable(linkTest EXCLUDE_FROM_ALL linkTest0.cpp linkTest1.cpp)
add_dependencies(build_tests linkTest)
add_dependencies(other linkTest)
add_target_property(linkTestLibDynamic1 COMPILE_FLAGS "-DPOSTFIX=A")
add_target_property(linkTestLibDynamic2 COMPILE_FLAGS "-DPOSTFIX=B")
set_property(TARGET linkTestLibDynamic1 APPEND PROPERTY COMPILE_OPTIONS "${Vc_ARCHITECTURE_FLAGS}")
set_property(TARGET linkTestLibDynamic2 APPEND PROPERTY COMPILE_OPTIONS "${Vc_ARCHITECTURE_FLAGS}")
set_property(TARGET linkTestLibStatic APPEND PROPERTY COMPILE_OPTIONS "${Vc_ARCHITECTURE_FLAGS}")
set_property(TARGET linkTest APPEND PROPERTY COMPILE_OPTIONS "${Vc_ARCHITECTURE_FLAGS}")
target_link_libraries(linkTest linkTestLibDynamic1 linkTestLibDynamic2 linkTestLibStatic)#}}}

# Use the following program to generate the sincos-reference-*.dat files
#add_executable(convert-sincos-reference EXCLUDE_FROM_ALL convert-sincos-reference.cpp)

# version.h test - make sure it can be included on its own.
add_executable(version_hTest EXCLUDE_FROM_ALL version_h.cpp)

# vim: foldmethod=marker commentstring=#%s
