include(AddFileDependencies)

add_definitions(-DCOMPILE_FOR_UNIT_TESTS) # -DVC_CHECK_ALIGNMENT)
if(Vc_COMPILER_IS_MSVC)
   AddCompilerFlag("/wd4267") # Disable warning "conversion from 'size_t' to 'int', possible loss of data"
   AddCompilerFlag("/wd4723") # Disable warning "potential divide by 0" (suppress doesn't work)
endif()

# 32-bit x86 requires SSE for fp math to produce comparable results.
AddCompilerFlag("-mfpmath=sse" CXX_FLAGS Vc_ARCHITECTURE_FLAGS CXX_RESULT _fpmath_ok)

set(Vc_SCALAR_FLAGS "${Vc_ARCHITECTURE_FLAGS};-DVc_IMPL=Scalar")
set(Vc_SSE_FLAGS    "${Vc_ARCHITECTURE_FLAGS};-DVc_IMPL=SSE")
set(Vc_AVX_FLAGS    "${Vc_ARCHITECTURE_FLAGS};-DVc_IMPL=AVX")
set(Vc_AVX2_FLAGS   "${Vc_ARCHITECTURE_FLAGS};-DVc_IMPL=AVX2")
set(Vc_MIC_FLAGS    "-DVc_IMPL=MIC")

if(USE_XOP)
   set(Vc_SSE_FLAGS  "${Vc_SSE_FLAGS}+XOP")
   set(Vc_AVX_FLAGS  "${Vc_AVX_FLAGS}+XOP")
endif()
if(USE_FMA)
   set(Vc_SSE_FLAGS  "${Vc_SSE_FLAGS}+FMA")
   set(Vc_AVX_FLAGS  "${Vc_AVX_FLAGS}+FMA")
   set(Vc_AVX2_FLAGS "${Vc_AVX2_FLAGS}+FMA")
elseif(USE_FMA4)
   set(Vc_SSE_FLAGS  "${Vc_SSE_FLAGS}+FMA4")
   set(Vc_AVX_FLAGS  "${Vc_AVX_FLAGS}+FMA4")
endif()
if(USE_BMI2)
   set(Vc_AVX2_FLAGS "${Vc_AVX2_FLAGS}+BMI2")
endif()

if(DEFINED Vc_INSIDE_ROOT)
   set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "")  # Reset the ROOT default executable destination
   set(Vc_TEST_TARGET_PREFIX "vc-")
else()
   set(Vc_TEST_TARGET_PREFIX "")
endif()

# don't warn about sanity checks:
if(Vc_COMPILER_IS_CLANG)
   # GCC does not complain about the following flags until much later. Therefore
   # the AddCompilerFlag logic doesn't work as intended.
   AddCompilerFlag(-Wno-tautological-constant-out-of-range-compare)
   AddCompilerFlag(-Wno-tautological-compare)
endif()

CHECK_CXX_SOURCE_COMPILES("#include <cxxabi.h>
int main() { return 0; }" cxx_abi_header_works)
if(cxx_abi_header_works)
   add_definitions(-DHAVE_CXX_ABI_H)
endif()

set(all_target_variants)
function(vc_set_first_flag_of name)
   foreach(flags ${ARGN})
      string(REPLACE "|" ";" flags "${flags}")
      set(result)
      foreach(flag IN LISTS flags)
         AddCompilerFlag("${flag}" CXX_FLAGS result CXX_RESULT ok)
         if(NOT ok)
            break()
         endif()
      endforeach()
      if(ok)
         message(STATUS "Building tests for ${name}: enabled")
         set(Vc_${name}_flags ${result} PARENT_SCOPE)
         list(APPEND all_target_variants ${name})
         set(all_target_variants ${all_target_variants} PARENT_SCOPE)
         return()
      endif()
   endforeach()
   message(STATUS "Building tests for ${name}: disabled")
endfunction()

# sets Vc_scalar_flags, Vc_sse2_flags, ...
vc_set_first_flag_of(scalar "-mno-sse2")  # fails to compile without SSE in system headers
vc_set_first_flag_of(sse2 "-xSSE2" "-msse2" "/arch:SSE2")
vc_set_first_flag_of(sse4_2 "-xSSE4.2" "-msse4.2" "/arch:SSE2")
vc_set_first_flag_of(avx "-xAVX" "-mavx" "/arch:AVX")
vc_set_first_flag_of(avx2 "-xCORE-AVX2" "-mavx2|-mbmi2|-mfma" "/arch:AVX2")
vc_set_first_flag_of(mic "-mmic")
vc_set_first_flag_of(knl "-xMIC-AVX512" "-march=knl")
vc_set_first_flag_of(avx512 "-xCORE-AVX512" "-march=skylake-avx512")

macro(vc_add_run_target _target)
   if("${_target}" MATCHES "_mic$")
      if(MIC_NATIVELOAD)
         get_target_property(_exe "${_target}" OUTPUT_NAME)
         add_custom_target(run_${_target}
            ${MIC_NATIVELOAD} "${_exe}" -a "-v"
            DEPENDS ${_target}
            COMMENT "Execute ${_target} test"
            VERBATIM
            )
      endif()
   else()
      add_custom_target(run_${_target}
         ${_target} -v
         DEPENDS ${_target}
         COMMENT "Execute ${_target} test"
         VERBATIM
         )
   endif()
endmacro()

set(alias_strategies Union)
if(Vc_COMPILER_IS_CLANG OR Vc_COMPILER_IS_GCC)
   set(alias_strategies ${alias_strategies} MayAlias VectorBuiltin)
elseif(Vc_COMPILER_IS_INTEL)
   # ICC can parse (aka ignore) may_alias...
   set(alias_strategies ${alias_strategies} MayAlias VectorBuiltin)
elseif(Vc_COMPILER_IS_MSVC)
   set(alias_strategies ${alias_strategies} UnionMembers)
endif()

function(vc_add_test2 name)
   set(src "${name}.cpp")
   set(run_targets)
   foreach(impl ${all_target_variants})
      set(_target "${name}_${impl}")
      list(FIND disabled_targets ${_target} disabled)
      if(disabled EQUAL -1)
         set(flags "${Vc_COMPILE_FLAGS}")
         list(APPEND flags "${Vc_${impl}_flags}")
         string(TOUPPER "${impl}" label)
         set(use_var "USE_${label}")
         if(impl STREQUAL "scalar")
            set(label Scalar)
            set(use_var TRUE)
         elseif(impl STREQUAL "mic")
            set(use_var "MIC_NATIVE_FOUND")
         endif()
         if(NOT Vc_${impl}_flags)
            continue()
         endif()
         foreach(alias ${alias_strategies})
            if(impl STREQUAL "scalar")
               set(target ${_target})
            else()
               string(TOLOWER "${_target}_${alias}" target)
               string(TOUPPER ${alias} alias_flag)
               set(alias_flag "Vc_USE_ALIASSTRATEGY_${alias_flag}")
            endif()
            if(impl STREQUAL "mic")
               set(_objs)
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
                  COMPILE_FLAGS "${Vc_MIC_FLAGS}"
                  SOURCES ${src}
                  OBJECTS ${_objs}
                  EXCLUDE_FROM_ALL
                  )
            else()
               add_executable(${target} EXCLUDE_FROM_ALL ${src})
               #target_link_libraries(${target} Vc)
               set_property(TARGET ${target} APPEND PROPERTY COMPILE_OPTIONS "${flags}")
               if(NOT impl STREQUAL "scalar")
                  set_property(TARGET ${target} APPEND PROPERTY COMPILE_DEFINITIONS "${alias_flag}")
               endif()
            endif()
            add_target_property(${target} LABELS "${label}")
            add_dependencies(build_tests ${target})
            add_dependencies(${label} ${target})
            if(impl STREQUAL "mic")
               if(${use_var} AND MIC_NATIVELOAD)
                  get_target_property(_exe "${target}" OUTPUT_NAME)
                  add_test(NAME ${Vc_TEST_TARGET_PREFIX}${target}
                     COMMAND "${MIC_NATIVELOAD}" "${_exe}"
                     )
               endif()
            else()
               if(${use_var})
                  add_test(${Vc_TEST_TARGET_PREFIX}${target} "${CMAKE_CURRENT_BINARY_DIR}/${target}")
               endif()
            endif()
            if(${use_var} AND (NOT impl STREQUAL "mic" OR MIC_NATIVELOAD))
               set_property(TEST ${Vc_TEST_TARGET_PREFIX}${target} PROPERTY LABELS "${label}")
               vc_add_run_target(${target})
               list(APPEND run_targets "run_${target}")
            endif()
            if(impl STREQUAL "scalar")
               break()
            endif()
         endforeach()
      endif()
   endforeach()
   if(run_targets)
      add_custom_target(run_${name}_all
         COMMENT "Execute all ${name} tests"
         VERBATIM
         )
      add_dependencies(run_${name}_all ${run_targets})
   endif()
endfunction()

vc_add_test2(datapar_mask)
vc_add_test2(datapar)

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
add_dependencies(other download-testdata)

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

      get_target_property(_exe ${_target} OUTPUT_NAME)
      if(NOT _exe)
         set(_exe ${_target})
      endif()

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
endif()

macro(vc_add_general_test _name)
   add_executable(${_name} EXCLUDE_FROM_ALL ${_name}.cpp )
   target_link_libraries(${_name} Vc)
   add_target_property(${_name} LABELS "other")
   set_property(TARGET ${_name} APPEND PROPERTY COMPILE_OPTIONS "${Vc_ARCHITECTURE_FLAGS}")
   add_dependencies(build_tests ${_name})
   add_dependencies(other ${_name})
   add_test(${Vc_TEST_TARGET_PREFIX}${_name} "${CMAKE_CURRENT_BINARY_DIR}/${_name}")
   set_property(TEST ${Vc_TEST_TARGET_PREFIX}${_name} PROPERTY LABELS "other")
   vc_add_run_target(${_name})
endmacro()

if(_last_target_arch STREQUAL "auto" AND NOT Vc_AVX_INTRINSICS_BROKEN)
   vc_add_general_test(supportfunctions)
endif()
vc_add_general_test(alignmentinheritance)
vc_add_general_test(alignedbase)

get_property(_incdirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
set(incdirs)
foreach(_d ${_incdirs})
   list(APPEND incdirs "-I${_d}")
endforeach()

separate_arguments(_flags UNIX_COMMAND "${CMAKE_CXX_FLAGS} ${Vc_DEFINITIONS}")
list(APPEND _flags ${Vc_COMPILE_FLAGS})
foreach(_f ${_flags})
   if(_f MATCHES "^-m" OR _f MATCHES "^/arch:" OR _f MATCHES "^-x")
      list(REMOVE_ITEM _flags "${_f}")
   endif()
endforeach()

# compile and link test for targets that need to link lots of stuff together
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
target_link_libraries(linkTest Vc linkTestLibDynamic1 linkTestLibDynamic2 linkTestLibStatic)

# Use the following program to generate the sincos-reference-*.dat files
#add_executable(convert-sincos-reference EXCLUDE_FROM_ALL convert-sincos-reference.cpp)

