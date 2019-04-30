include(CMakeParseArguments)
include(AddFileDependencies)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/virtest)

if(FALSE)
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
endif(FALSE)

if(NOT "$ENV{TRAVIS_OS_NAME}" STREQUAL "")
   set(CMAKE_CXX_FLAGS_RELEASE "-O1 -DNDEBUG") # try to not time out, -O0 is faster
endif()

add_definitions(-DCOMPILE_FOR_UNIT_TESTS -D_GLIBCXX_SIMD_NO_OPTIMIZATION_WARNINGS)
AddCompilerFlag("-fmax-errors=10") # Don't go too crazy on errors

# 32-bit x86 requires SSE for fp math to produce comparable results.
AddCompilerFlag("-mfpmath=sse" CXX_FLAGS Vc_ARCHITECTURE_FLAGS CXX_RESULT _fpmath_ok)

option(ENABLE_UBSAN "Enable build of unit tests with undefined behavior sanitizer" OFF)
if(ENABLE_UBSAN)
   AddCompilerFlag("-fsanitize=undefined")
endif()

function(vc_download_testdata)#{{{
   find_program(OBJCOPY objcopy)
   set(_deps)
   foreach(fun sincos asin acos atan ln log2 log10)
      foreach(type f d)
         if("${type}" STREQUAL "f")
            set(filename "reference-${fun}-sp.dat")
         else()
            set(filename "reference-${fun}-dp.dat")
         endif()
         add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${filename}"
            COMMAND ${CMAKE_COMMAND} -Dfilename=${filename} -P ${CMAKE_CURRENT_SOURCE_DIR}/download.cmake
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/download.cmake
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Downloading Test Data: ${filename}"
            VERBATIM
            )
         if(OBJCOPY)
            string(REGEX REPLACE "[.-]" "_" toreplace "_binary_${filename}_")
            string(LENGTH "${fun}" fun_len)
            set(replacement "_ZN14reference_dataIN8function${fun_len}${fun}E${type}E")
            add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${filename}.o"
               COMMAND ${OBJCOPY} -I binary -O elf64-x86-64 -B i386
               --redefine-sym ${toreplace}start=${replacement}6begin_E
               --redefine-sym ${toreplace}end=${replacement}4end_E
               --redefine-sym ${toreplace}size=${replacement}5size_E
               ${filename} ${filename}.o
               DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${filename}"
               WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
               COMMENT "Converting Test Data: ${filename}.o"
               VERBATIM
               )
            list(APPEND _deps "${CMAKE_CURRENT_BINARY_DIR}/${filename}.o")
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
      add_dependencies(KNL download-testdata)
      add_dependencies(AVX512 download-testdata)
   elseif(Vc_ARM)
      add_dependencies(NEON download-testdata)
   endif()
endfunction()
vc_download_testdata()#}}}

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
   vc_target_setup(nosimd mrm "-mno-sse")
   vc_target_setup(sse2 mrm "-xSSE2" "-march=athlon64")
   vc_target_setup(ssse3 mrm "-xSSSE3" "-march=core2")
   vc_target_setup(sse4_2 nhm "-xSSE4.2" "-march=nehalem")
   vc_target_setup(avx snb "-xAVX" "-march=sandybridge" "/arch:AVX")
   vc_target_setup(avx2 hsw "-xCORE-AVX2" "-march=haswell" "/arch:AVX2")
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
            ${emulator} "${_exe}" "-v"
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

function(vc_add_test name) #{{{
   if($ENV{VcSkipTests} MATCHES "${name}")
      return()
   endif()
   set(_only "$ENV{VcOnlyTests}")
   if(NOT ("${_only}" STREQUAL "" OR "${_only}" MATCHES "${name}"))
      return()
   endif()

   cmake_parse_arguments("" "NO_TESTTYPES;SINGLE_TESTTYPE;EXCLUDE_FROM_ALL" "SOURCE" "DEFINITIONS" ${ARGN})

   if(_SOURCE)
      set(src "${_SOURCE}")
   else()
      set(src "${name}.cpp")
   endif()

   set(run_targets)
   foreach(impl ${vc_all_target_variants})
      set(_target "${name}_${impl}")
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
      elseif(Vc_X86 AND INTEL_SDE AND NOT ${use_var})
         set(use_var TRUE)
         set(emulator "${INTEL_SDE};${sde_cpuid_${impl}};--")
      endif()
      if(_NO_TESTTYPES)
         set(type_split " ")
      elseif(_SINGLE_TESTTYPE)
         set(type_split "ldouble" "float" "double" "uint" "llong" "long" "uchar" "int" "ushort" "short" "ulong" "schar" "ullong" "wchar" "char16" "char32" "char")
      else()
         set(type_split "ldouble,float,double,schar,uchar" "llong,long,ullong,ulong,char,wchar" "int,short,uint,ushort,char16,char32")
      endif()
      foreach(types ${type_split})
         if(impl STREQUAL "nosimd")#{{{
            set(target ${_target})
         else()
            string(TOLOWER "${_target}" target)
         endif()#}}}
         if(NOT _NO_TESTTYPES)
            string(REPLACE "," "_" target "${target}_${types}")
         endif()

         if(OBJCOPY)
            list(APPEND _DEFINITIONS "_GLIBCXX_SIMD_LINK_TESTDATA")
            if("${name}" STREQUAL "math")
               foreach(fun sincos asin acos atan)
                  foreach(filename reference-${fun}-sp.dat reference-${fun}-dp.dat)
                     set(src ${src} ${CMAKE_CURRENT_BINARY_DIR}/${filename}.o)
                  endforeach()
               endforeach()
            elseif("${name}" STREQUAL "logarithm")
               foreach(fun ln log2 log10)
                  foreach(filename reference-${fun}-sp.dat reference-${fun}-dp.dat)
                     set(src ${src} ${CMAKE_CURRENT_BINARY_DIR}/${filename}.o)
                  endforeach()
               endforeach()
            endif()
         endif()

         add_executable(${target} EXCLUDE_FROM_ALL ${src})
         set_property(TARGET ${target} APPEND PROPERTY COMPILE_DEFINITIONS "${_DEFINITIONS}")
         set_property(TARGET ${target} APPEND PROPERTY COMPILE_OPTIONS "${flags}")
         set_property(TARGET ${target} APPEND PROPERTY LINK_FLAGS "${link_flags}")
         if(NOT _NO_TESTTYPES)
            set_property(TARGET ${target} APPEND PROPERTY COMPILE_DEFINITIONS "TESTTYPES=${types}")
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
   if(run_targets)
      add_custom_target(run_${name}_all
         COMMENT "Execute all ${name} tests"
         VERBATIM
         )
      add_dependencies(run_${name}_all ${run_targets})
   endif()
endfunction() #}}}

vc_add_test(sfinae NO_TESTTYPES)
vc_add_test(loadstore SINGLE_TESTTYPE DEFINITIONS ONE_RANDOM_ARITHMETIC_TYPE)
vc_add_test(allcvt_loadstore SINGLE_TESTTYPE SOURCE loadstore.cpp SINGLE_TESTTYPE EXCLUDE_FROM_ALL)
vc_add_test(mask NO_TESTTYPES)
vc_add_test(mask_conversions NO_TESTTYPES)
vc_add_test(simd NO_TESTTYPES)
vc_add_test(operators)
vc_add_test(integer_operators)
vc_add_test(casts DEFINITIONS ONE_RANDOM_ARITHMETIC_TYPE)
vc_add_test(allcvt_casts SOURCE casts.cpp EXCLUDE_FROM_ALL)
vc_add_test(math NO_TESTTYPES)
vc_add_test(specialmath NO_TESTTYPES)
vc_add_test(where NO_TESTTYPES)
#vc_add_test(experimental NO_TESTTYPES)
vc_add_test(bitset_conversions NO_TESTTYPES)

# ABI tests#{{{
find_program(OBJDUMP objdump)
mark_as_advanced(OBJDUMP)
if(OBJDUMP AND NOT ENABLE_UBSAN) # ubsan changes codegen too much, just skip it
   set(alias_strategies2)
   list(APPEND alias_strategies2 MayAlias VectorBuiltin)

   set(run_targets)
   macro(test_uninitialized _impl)
      set(_target "uninit_${_impl}")
      set(_test test_${_target})
      add_executable(${_target} EXCLUDE_FROM_ALL uninit.cpp)
      set_property(TARGET ${_target} APPEND PROPERTY COMPILE_OPTIONS "${ARGN};-O2")
      add_target_property(${_target} LABELS "${_impl}")
      add_dependencies(${_impl} ${_target})
      add_dependencies(build_tests ${_target})
      get_target_executable(_exe "${_target}")
      add_test(NAME ${_test}
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
         COMMAND ${CMAKE_COMMAND} -DOBJDUMP=${OBJDUMP} -DBINARY=${_exe} -DIMPL=${_impl}
         -DCOMPILER_IS_CLANG=${Vc_COMPILER_IS_CLANG} -DSYSTEM_NAME=${CMAKE_SYSTEM_NAME}
         -P ${CMAKE_CURRENT_SOURCE_DIR}/uninit.cmake
         )
      set_property(TEST ${_test} PROPERTY LABELS "${_impl}")
      add_custom_target(run_${_target}
         ${CMAKE_COMMAND} -DOBJDUMP=${OBJDUMP} -DBINARY=${_exe} -DIMPL=${_impl}
         -DCOMPILER_IS_CLANG=${Vc_COMPILER_IS_CLANG} -DSYSTEM_NAME=${CMAKE_SYSTEM_NAME}
         -P ${CMAKE_CURRENT_SOURCE_DIR}/uninit.cmake
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
         DEPENDS ${_target}
         COMMENT "Execute ${_target} test"
         VERBATIM
         )
      list(APPEND run_targets "run_${_target}")
   endmacro()

   macro(test_abi target _impl abi)
      foreach(subtype native fixed_size)
         if("${subtype}" STREQUAL "fixed_size")
            set(suffix "_fixed_size")
         else()
            set(suffix "")
         endif()
         set(_target "${target}${suffix}")
         set(_test test_${_target})
         add_executable(${_target} EXCLUDE_FROM_ALL abi${suffix}.cpp)
         set_property(TARGET ${_target} APPEND PROPERTY COMPILE_DEFINITIONS "_GLIBCXX_SIMD_ABI=${abi}")
         set_property(TARGET ${_target} APPEND PROPERTY COMPILE_OPTIONS "${ARGN};-O2")
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
      test_uninitialized(SSE ${Vc_sse2_flags})
      test_uninitialized(AVX ${Vc_avx_flags})
      test_uninitialized(KNL ${Vc_knl_flags})

      test_abi(abi_SSE     SSE __sse ${Vc_sse2_flags})
      test_abi(abi_SSE_AVX SSE __sse ${Vc_avx_flags})
      test_abi(abi_AVX     AVX __avx ${Vc_avx_flags})
      test_abi(abi_AVX2    AVX2 __avx ${Vc_avx2_flags})
      test_abi(abi_AVX512  AVX512 __avx512 ${Vc_avx512_flags})
      test_abi(abi_KNL     KNL __avx512 ${Vc_knl_flags})
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
