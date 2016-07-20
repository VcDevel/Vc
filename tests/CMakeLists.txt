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

macro(vc_set_first_flag_of _result)
   set(${_result})
   foreach(_flag ${ARGN})
      AddCompilerFlag("${_flag}" CXX_FLAGS ${_result} CXX_RESULT _ok)
      if(_ok)
         break()
      endif()
   endforeach()
endmacro()

vc_set_first_flag_of(_sse2_flag "-xSSE2" "-msse2" "/arch:SSE2")
vc_set_first_flag_of(_sse4_1_flag "-xSSE4.1" "-msse4.1" "/arch:SSE2")
vc_set_first_flag_of(_avx_flag "-xAVX" "-mavx" "/arch:AVX")
vc_set_first_flag_of(_avx2_flag "-xCORE-AVX2" "-mavx2" "/arch:AVX2")
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

macro(vc_set_test_target_properties _target _impl _compile_flags)
   target_link_libraries(${_target} Vc)
   set_target_properties(${_target} PROPERTIES XCODE_ATTRIBUTE_CLANG_CXX_LANGUAGE_STANDARD "c++0x")
   set_target_properties(${_target} PROPERTIES XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY "libc++")
   add_target_property(${_target} COMPILE_FLAGS "${_extra_flags}")
   set_property(TARGET ${_target} APPEND PROPERTY COMPILE_OPTIONS ${_compile_flags})
   add_target_property(${_target} LABELS "${_impl}")
   add_dependencies(build_tests ${_target})
   add_dependencies(${_impl} ${_target})
   add_test(${Vc_TEST_TARGET_PREFIX}${_target} "${CMAKE_CURRENT_BINARY_DIR}/${_target}")
   set_property(TEST ${Vc_TEST_TARGET_PREFIX}${_target} PROPERTY LABELS "${_impl}")
   vc_add_run_target(${_target})
endmacro()

macro(vc_add_test _name)
   set(_extra_flags)
   set(name ${_name})
   set(_state 0)
   set(_targets "Scalar;SSE;AVX1;AVX2;MIC")
   foreach(_arg ${ARGN})
      if("${_arg}" STREQUAL "FLAGS")
         set(_state 0)
      elseif("${_arg}" STREQUAL "TARGETS")
         set(_targets)
         set(_state 1)
      elseif(_state EQUAL 0)
         set(_extra_flags "${_extra_flags} -D${_arg}")
         set(name "${name}_${_arg}")
      elseif(_state EQUAL 1)
         if("${_arg}" STREQUAL "AVX")
            list(APPEND _targets "AVX1")
         else()
            list(APPEND _targets "${_arg}")
         endif()
      endif()
   endforeach()
   string(REPLACE "=" "_" name "${name}")

   set(_run_targets)

   set(_target "${name}_scalar")
   list(FIND disabled_targets ${_target} _disabled)
   if(_disabled EQUAL -1 AND "${_targets}" MATCHES "Scalar")
      file(GLOB _extra_deps "${CMAKE_SOURCE_DIR}/scalar/*.tcc" "${CMAKE_SOURCE_DIR}/scalar/*.h" "${CMAKE_SOURCE_DIR}/common/*.h")
      add_file_dependencies(${_name}.cpp "${_extra_deps}")
      add_executable(${_target} EXCLUDE_FROM_ALL ${_name}.cpp)
      vc_set_test_target_properties(${_target} Scalar "${Vc_SCALAR_FLAGS}")
      list(APPEND _run_targets "run_${_target}")
   endif()

   if(USE_SSE2 AND NOT Vc_SSE_INTRINSICS_BROKEN AND "${_targets}" MATCHES "SSE")
      set(_target "${name}_sse")
      list(FIND disabled_targets ${_target} _disabled)
      if(_disabled EQUAL -1)
         file(GLOB _extra_deps "${CMAKE_SOURCE_DIR}/sse/*.tcc" "${CMAKE_SOURCE_DIR}/sse/*.h" "${CMAKE_SOURCE_DIR}/common/*.h")
         add_file_dependencies(${_name}.cpp "${_extra_deps}")
         add_executable(${_target} EXCLUDE_FROM_ALL ${_name}.cpp)
         vc_set_test_target_properties(${_target} SSE "${Vc_SSE_FLAGS}")
         list(APPEND _run_targets "run_${_target}")
      endif()
   endif()

   if(USE_AVX AND "${_targets}" MATCHES "AVX1")
      set(_target "${name}_avx")
      list(FIND disabled_targets ${_target} _disabled)
      if(_disabled EQUAL -1)
         file(GLOB _extra_deps "${CMAKE_SOURCE_DIR}/avx/*.tcc" "${CMAKE_SOURCE_DIR}/avx/*.h" "${CMAKE_SOURCE_DIR}/common/*.h")
         add_file_dependencies(${_name}.cpp "${_extra_deps}")
         add_executable(${_target} EXCLUDE_FROM_ALL ${_name}.cpp)
         vc_set_test_target_properties(${_target} AVX "${Vc_AVX_FLAGS}")
         list(APPEND _run_targets "run_${_target}")
      endif()
   endif()

   if(USE_AVX2 AND "${_targets}" MATCHES "AVX2")
      set(_target "${name}_avx2")
      list(FIND disabled_targets ${_target} _disabled)
      if(_disabled EQUAL -1)
         file(GLOB _extra_deps "${CMAKE_SOURCE_DIR}/avx/*.tcc" "${CMAKE_SOURCE_DIR}/avx/*.h" "${CMAKE_SOURCE_DIR}/common/*.h")
         add_file_dependencies(${_name}.cpp "${_extra_deps}")
         add_executable(${_target} EXCLUDE_FROM_ALL ${_name}.cpp)
         vc_set_test_target_properties(${_target} AVX2 "${Vc_AVX2_FLAGS}")
         list(APPEND _run_targets "run_${_target}")
      endif()
   endif()

   if(MIC_NATIVE_FOUND AND "${_targets}" MATCHES "MIC")
      set(_target "${name}_mic")
      list(FIND disabled_targets ${_target} _disabled)
      file(GLOB _extra_deps "${CMAKE_SOURCE_DIR}/mic/*.tcc" "${CMAKE_SOURCE_DIR}/mic/*.h" "${CMAKE_SOURCE_DIR}/common/*.h")
      add_file_dependencies(${_name}.cpp "${_extra_deps}")
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
      mic_add_executable(${_target}
         LINK_LIBRARIES Vc_MIC
         COMPILE_FLAGS "${Vc_MIC_FLAGS} ${_extra_flags}"
         SOURCES ${_name}.cpp
         OBJECTS ${_objs}
         EXCLUDE_FROM_ALL
         )
      add_target_property(${_target} LABELS "MIC")
      if(_disabled EQUAL -1)
         add_dependencies(build_tests ${_target})
         add_dependencies(MIC ${_target})
         if(MIC_NATIVELOAD)
            get_target_property(_exe "${_target}" OUTPUT_NAME)
            add_test(NAME ${Vc_TEST_TARGET_PREFIX}${_target}
               COMMAND "${MIC_NATIVELOAD}" "${_exe}"
               )
            set_property(TEST ${Vc_TEST_TARGET_PREFIX}${_target} PROPERTY LABELS "MIC")
            vc_add_run_target(${_target})
            list(APPEND _run_targets "run_${_target}")
         endif()
      endif()
   endif()

   if(_run_targets)
      add_custom_target(run_${name}_all
         COMMENT "Execute all ${name} tests"
         VERBATIM
         )
      add_dependencies(run_${name}_all ${_run_targets})
   endif()

endmacro(vc_add_test)

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

vc_add_test(datapar_mask)
vc_add_test(types)
vc_add_test(subscript)
vc_add_test(type_traits)
vc_add_test(stlcontainer)
vc_add_test(scalaraccess)
vc_add_test(memory)
vc_add_test(arithmetics)
vc_add_test(simdize)
vc_add_test(implicit_type_conversion)
vc_add_test(iterators)
vc_add_test(load)
vc_add_test(store)
vc_add_test(gather)
vc_add_test(gather Vc_USE_BSF_GATHERS TARGETS SSE AVX AVX2)
vc_add_test(gather Vc_USE_POPCNT_BSF_GATHERS TARGETS SSE AVX AVX2)
vc_add_test(gather Vc_USE_SET_GATHERS TARGETS SSE AVX AVX2)
vc_add_test(scatter)
vc_add_test(scatter Vc_USE_BSF_SCATTERS TARGETS SSE AVX AVX2)
vc_add_test(scatter Vc_USE_POPCNT_BSF_SCATTERS TARGETS SSE AVX AVX2)
vc_add_test(ulp)
vc_add_test(logarithm)
vc_add_test(logarithm Vc_LOG_ILP TARGETS SSE AVX AVX2)
vc_add_test(logarithm Vc_LOG_ILP2 TARGETS SSE AVX AVX2)
vc_add_test(trigonometric)
vc_add_test(math)
vc_add_test(reductions)
vc_add_test(mask)
vc_add_test(utils)
vc_add_test(sorted)
vc_add_test(random)
vc_add_test(deinterleave)
vc_add_test(gatherinterleavedmemory)
vc_add_test(scatterinterleavedmemory)
vc_add_test(scatterinterleavedmemory Vc_USE_MASKMOV_SCATTER TARGETS SSE AVX AVX2)
vc_add_test(casts Vc_DEFAULT_TYPES)
option(BUILD_EXTRA_CAST_TESTS "build all possible combinations of simd_cast tests (compiles for a very long time)" FALSE)
if(BUILD_EXTRA_CAST_TESTS)
   vc_add_test(casts Vc_EXTRA_TYPES)
   foreach(n 1 2 3 4 5 8 16 17 31)
      vc_add_test(casts Vc_FROM_N=${n})
      vc_add_test(casts Vc_TO_N=${n})
      foreach(m 1 2 3 4 5 8 16 17 31)
         vc_add_test(casts Vc_FROM_N=${m} Vc_TO_N=${n})
      endforeach()
   endforeach()
endif()
vc_add_test(simdarray)

find_program(OBJDUMP objdump)
mark_as_advanced(OBJDUMP)

if(OBJDUMP)
   macro(test_abi _target _impl)
      set(_test test_${_target})
      if("${_impl}" STREQUAL "MIC")
         mic_add_executable(${_target} SOURCES abi.cpp LINK_LIBRARIES Vc_MIC COMPILE_FLAGS "-DVc_IMPL=${_impl};${ARGN};-O2")
      else()
         add_executable(${_target} EXCLUDE_FROM_ALL abi.cpp)
         target_link_libraries(${_target} Vc)
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
      test_abi(abi_SSE     SSE  ${_sse2_flag})
      test_abi(abi_SSE_AVX SSE  ${_avx_flag})
      if(NOT Vc_AVX_INTRINSICS_BROKEN)
         test_abi(abi_AVX     AVX  ${_avx_flag})
         test_abi(abi_AVX2    AVX2 ${_avx2_flag})
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

if(USE_SSE2 AND NOT Vc_SSE_INTRINSICS_BROKEN)
   list(FIND disabled_targets sse_blend _disabled)
   if(_disabled EQUAL -1)
      add_executable(sse2_blend EXCLUDE_FROM_ALL sse_blend.cpp)
      add_target_property(sse2_blend COMPILE_DEFINITIONS "Vc_IMPL=SSE2")
      set_property(TARGET sse2_blend APPEND PROPERTY COMPILE_OPTIONS "${_sse2_flag}")
      add_target_property(sse2_blend LABELS "SSE")
      add_dependencies(build_tests sse2_blend)
      add_dependencies(SSE sse2_blend)
      add_test(${Vc_TEST_TARGET_PREFIX}sse2_blend "${CMAKE_CURRENT_BINARY_DIR}/sse2_blend")
      set_property(TEST ${Vc_TEST_TARGET_PREFIX}sse2_blend PROPERTY LABELS "SSE")
      target_link_libraries(sse2_blend Vc)

      if(USE_SSE4_1)
         add_executable(sse4_blend EXCLUDE_FROM_ALL sse_blend.cpp)
         add_target_property(sse4_blend COMPILE_DEFINITIONS "Vc_IMPL=SSE4_1")
         set_property(TARGET sse4_blend APPEND PROPERTY COMPILE_OPTIONS ${Vc_ARCHITECTURE_FLAGS})
         add_target_property(sse4_blend LABELS "SSE")
         add_dependencies(build_tests sse4_blend)
         add_dependencies(SSE sse4_blend)
         add_test(${Vc_TEST_TARGET_PREFIX}sse4_blend "${CMAKE_CURRENT_BINARY_DIR}/sse4_blend")
         set_property(TEST ${Vc_TEST_TARGET_PREFIX}sse4_blend PROPERTY LABELS "SSE")
         target_link_libraries(sse4_blend Vc)
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

set(TEST_OPERATOR_FAILURES FALSE CACHE BOOL "Run implicit type conversion operator tests.")
if(TEST_OPERATOR_FAILURES)
   macro(vc_test_implicit_type_conversion_failures A B)
      get_property(_launch_rule GLOBAL PROPERTY RULE_LAUNCH_COMPILE)
      foreach(impl Scalar SSE AVX AVX2)
         if("${impl}" STREQUAL "Scalar")
            set(_implFlags)
         elseif("${impl}" STREQUAL "SSE")
            string(REPLACE " " ";" _implFlags "${_sse2_flag}")
         elseif("${impl}" STREQUAL "AVX")
            if(Vc_AVX_INTRINSICS_BROKEN)
               break()
            endif()
            string(REPLACE " " ";" _implFlags "${_avx_flag}")
         elseif("${impl}" STREQUAL "AVX2")
            if(Vc_AVX2_INTRINSICS_BROKEN)
               break()
            endif()
            string(REPLACE " " ";" _implFlags "${_avx2_flag}")
         endif()
         set(type_b "${B}")
         foreach(type_a "${A}" "${B}")
            foreach(op "^" "==" "*") # "/" "+" "-" "&" "|" "!=" "<=" ">=" "<" ">")
               set(name "implicit_type_conversion_failures_${type_a}_${op}_${type_b}_${impl}")
               string(REGEX REPLACE "[<>, ():;?]" "_" name "${name}")
               add_test(NAME "${Vc_TEST_TARGET_PREFIX}${name}" WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMAND
                  env LANG=C
                  ${_launch_rule}
                  ${CMAKE_CXX_COMPILER} ${_flags} ${_implFlags}
                  ${incdirs} -c -o "${name}.tmp"
                  "-DTYPE_A=${type_a}" "-DTEST_OP=${op}" "-DTYPE_B=${type_b}" "-DVc_IMPL=${impl}"
                  ${CMAKE_CURRENT_SOURCE_DIR}/implicit_type_conversion_failures.cpp
                  )
               set_property(TEST "${Vc_TEST_TARGET_PREFIX}${name}" PROPERTY LABELS "${impl}")
               set_property(TEST "${Vc_TEST_TARGET_PREFIX}${name}" PROPERTY PASS_REGULAR_EXPRESSION
                  "error: no operator .* matches these operands" # ICC
                  "error: invalid operands to binary expression" # Clang
                  "error: no match for 'operator..?' "           # GCC
                  "error: no match for ‘operator..?’ "           # GCC
                  )
            endforeach()
            set(type_b "${A}")
         endforeach()
      endforeach()
   endmacro()
   vc_test_implicit_type_conversion_failures("double_v"    "float_v")
   vc_test_implicit_type_conversion_failures("double_v"    "short_v")
   vc_test_implicit_type_conversion_failures("double_v"   "ushort_v")
   vc_test_implicit_type_conversion_failures("double_v"      "int_v")
   vc_test_implicit_type_conversion_failures("double_v"     "uint_v")
   vc_test_implicit_type_conversion_failures( "float_v"     "double")
   vc_test_implicit_type_conversion_failures( "float_v"    "short_v")
   vc_test_implicit_type_conversion_failures( "float_v"   "ushort_v")
   vc_test_implicit_type_conversion_failures(   "int_v"    "float_v")
   vc_test_implicit_type_conversion_failures(   "int_v"      "float")
   vc_test_implicit_type_conversion_failures(   "int_v"     "double")
   vc_test_implicit_type_conversion_failures(   "int_v"       "long")
   vc_test_implicit_type_conversion_failures(   "int_v"      "llong")
   vc_test_implicit_type_conversion_failures( "short_v"      "int_v")
   vc_test_implicit_type_conversion_failures( "short_v"     "uint_v")
   vc_test_implicit_type_conversion_failures( "short_v"      "float")
   vc_test_implicit_type_conversion_failures("ushort_v"      "int_v")
   vc_test_implicit_type_conversion_failures("ushort_v"     "uint_v")
   vc_test_implicit_type_conversion_failures("SimdArray<int, 3>" "int_v")
endif()

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

