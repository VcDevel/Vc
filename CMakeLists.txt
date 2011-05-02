cmake_minimum_required(VERSION 2.6.2)
project(Vc)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(disabled_targets)

if(CMAKE_CXX_COMPILER MATCHES "/(icpc|icc)$")
   set(COMPILER_IS_INTEL true)
   set(COMPILER_IS_OPEN64 false)
   set(COMPILER_IS_MSVC false)
   set(COMPILER_IS_GCC false)
elseif(CMAKE_CXX_COMPILER MATCHES "/(opencc|openCC)$")
   set(COMPILER_IS_INTEL false)
   set(COMPILER_IS_OPEN64 true)
   set(COMPILER_IS_MSVC false)
   set(COMPILER_IS_GCC false)
elseif(MSVC)
   set(COMPILER_IS_INTEL false)
   set(COMPILER_IS_OPEN64 false)
   set(COMPILER_IS_MSVC true)
   set(COMPILER_IS_GCC false)
elseif(CMAKE_COMPILER_IS_GNUCXX)
   set(COMPILER_IS_INTEL false)
   set(COMPILER_IS_OPEN64 false)
   set(COMPILER_IS_MSVC false)
   set(COMPILER_IS_GCC true)
else()
   message(ERROR "Unsupported Compiler. Please fill out the missing parts in the CMake scripts and submit a patch to http://code.compeng.uni-frankfurt.de/projects/vc")
endif()

if(NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

include (MacroEnsureVersion)
include (OptimizeForArchitecture)
include (AddCompilerFlag)
include (CheckCXXSourceRuns)
include (AddTargetProperty)

check_cxx_source_runs("int main() { return sizeof(void*) != 8; }" VOID_PTR_IS_64BIT)

find_package(Larrabee)
if(LARRABEE_FOUND)
   include_directories(${LRB_HOST_INCLUDE_DIR})
else(LARRABEE_FOUND)
   if(VOID_PTR_IS_64BIT) # LRBni will be x86_64, so we don't really care for 32-bit archs
      find_file(prototypes_header lrbni_prototypes.h PATHS "${CMAKE_SOURCE_DIR}/larrabee/" NO_DEFAULT_PATH)
      if(prototypes_header)
         set(LARRABEE_FOUND "prototype")
         add_definitions(-DVC_LRBni_PROTOTYPE_H)
      else(prototypes_header)
         message(STATUS "Download the LRBni Prototype Header from http://software.intel.com/en-us/articles/prototype-primitives-guide/ and save it as ${CMAKE_SOURCE_DIR}/larrabee/lrbni_prototypes.h\nYou will also need to apply the patch from http://software.intel.com/en-us/forums/showthread.php?t=66359")
      endif(prototypes_header)
   endif(VOID_PTR_IS_64BIT)
endif(LARRABEE_FOUND)

if(NO_BSF_LOOPS)
   add_definitions(-DVC_NO_BSF_LOOPS)
endif(NO_BSF_LOOPS)
if(NO_GATHER_TRICKS)
   add_definitions(-DVC_NO_GATHER_TRICKS)
endif(NO_GATHER_TRICKS)
if(GATHER_SET)
   add_definitions(-DVC_GATHER_SET)
endif(GATHER_SET)

set(SSE_INTRINSICS_BROKEN false)

if(COMPILER_IS_OPEN64)
   # Open64 is detected as GNUCXX :(
   AddCompilerFlag("-W")
   AddCompilerFlag("-Wall")
   AddCompilerFlag("-Wimplicit")
   AddCompilerFlag("-Wswitch")
   AddCompilerFlag("-Wformat")
   AddCompilerFlag("-Wchar-subscripts")
   AddCompilerFlag("-Wparentheses")
   AddCompilerFlag("-Wmultichar")
   AddCompilerFlag("-Wtrigraphs")
   AddCompilerFlag("-Wpointer-arith")
   AddCompilerFlag("-Wcast-align")
   AddCompilerFlag("-Wreturn-type")
   AddCompilerFlag("-Wno-unused-function")
   AddCompilerFlag("-ansi")
   AddCompilerFlag("-pedantic")
   AddCompilerFlag("-Wno-long-long")
   AddCompilerFlag("-Wshadow")
   AddCompilerFlag("-Wold-style-cast")
   AddCompilerFlag("-Wno-variadic-macros")
   AddCompilerFlag("-fno-threadsafe-statics")
   set(CMAKE_CXX_FLAGS_DEBUG          "${CMAKE_CXX_FLAGS_DEBUG}")
   set(CMAKE_CXX_FLAGS_MINSIZEREL     "${CMAKE_CXX_FLAGS_MINSIZEREL} ")
   set(CMAKE_CXX_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE} -O3")
   set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -DNDEBUG -O3")
   set(CMAKE_C_FLAGS_DEBUG          "${CMAKE_C_FLAGS_DEBUG}")
   set(CMAKE_C_FLAGS_MINSIZEREL     "${CMAKE_C_FLAGS_MINSIZEREL} ")
   set(CMAKE_C_FLAGS_RELEASE        "${CMAKE_C_FLAGS_RELEASE} -O3")
   set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -DNDEBUG -O3")

   if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
      set(ENABLE_STRICT_ALIASING true CACHE BOOL "Enables strict aliasing rules for more aggressive optimizations")
      if(NOT ENABLE_STRICT_ALIASING)
         set(CMAKE_CXX_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE} -fno-strict-aliasing ")
         set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fno-strict-aliasing ")
         set(CMAKE_C_FLAGS_RELEASE        "${CMAKE_C_FLAGS_RELEASE} -fno-strict-aliasing ")
         set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -fno-strict-aliasing ")
      endif(NOT ENABLE_STRICT_ALIASING)
   endif(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")

   # if compiling for 32 bit x86 we need to use the -mfpmath=sse since the x87 is broken by design
   if(NOT VOID_PTR_IS_64BIT)
      exec_program(${CMAKE_C_COMPILER} ARGS -dumpmachine OUTPUT_VARIABLE _gcc_machine)
      if(_gcc_machine MATCHES "[x34567]86")
         AddCompilerFlag("-mfpmath=sse")
      endif(_gcc_machine MATCHES "[x34567]86")
   endif(NOT VOID_PTR_IS_64BIT)

   # Open64 uses binutils' assembler. And it has to be recent enough otherwise it'll bail out at some of the instructions Vc uses
   find_program(_binutils_as as)
   if(NOT _binutils_as)
      message(WARNING "Could not find 'as', the assembly normally used by GCC. Hoping everything will work out...")
   else(NOT _binutils_as)
      if(APPLE)
         # it's not really binutils, but it'll give us the assembler version which is what we want
         exec_program(${_binutils_as} ARGS -v /dev/null OUTPUT_VARIABLE _as_version)
      else(APPLE)
         exec_program(${_binutils_as} ARGS --version OUTPUT_VARIABLE _as_version)
      endif(APPLE)
      string(REGEX REPLACE "\\([^\\)]*\\)" "" _as_version "${_as_version}")
      string(REGEX MATCH "[1-9]\\.[0-9]+(\\.[0-9]+)?" _as_version "${_as_version}")
      macro_ensure_version("2.18.93" "${_as_version}" _as_good)
      if(NOT _as_good)
         message(SEND_ERROR "Your binutils is too old (${_as_version}). The assembler will not be able to compile Vc.")
      endif(NOT _as_good)
   endif(NOT _binutils_as)
   mark_as_advanced(_binutils_as)
elseif(COMPILER_IS_GCC)
   ##################################################################################################
   #                                              GCC                                               #
   ##################################################################################################
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -W -Wall -Wswitch -Wformat -Wchar-subscripts -Wparentheses -Wmultichar -Wtrigraphs -Wpointer-arith -Wcast-align -Wreturn-type -Wno-unused-function -ansi -pedantic -Wno-long-long -Wshadow")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -Wswitch -Wformat -Wchar-subscripts -Wparentheses -Wmultichar -Wtrigraphs -Wpointer-arith -Wcast-align -Wreturn-type -Wno-unused-function -ansi -pedantic -Wno-long-long -Wshadow")
   AddCompilerFlag("-Wimplicit")
   AddCompilerFlag("-Wabi")
   AddCompilerFlag("-fabi-version=4") # this is required to make __m128 and __m256 appear as different types.
   AddCompilerFlag("-Wold-style-cast")
   AddCompilerFlag("-Wno-variadic-macros")
   AddCompilerFlag("-fno-threadsafe-statics")
   AddCompilerFlag("-frename-registers")
   set(CMAKE_CXX_FLAGS_DEBUG          "${CMAKE_CXX_FLAGS_DEBUG}")
   set(CMAKE_CXX_FLAGS_MINSIZEREL     "${CMAKE_CXX_FLAGS_MINSIZEREL} ")
   set(CMAKE_CXX_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE} -O3")
   set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -DNDEBUG -O3")
   set(CMAKE_C_FLAGS_DEBUG          "${CMAKE_C_FLAGS_DEBUG}")
   set(CMAKE_C_FLAGS_MINSIZEREL     "${CMAKE_C_FLAGS_MINSIZEREL} ")
   set(CMAKE_C_FLAGS_RELEASE        "${CMAKE_C_FLAGS_RELEASE} -O3")
   set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -DNDEBUG -O3")

   # check the GCC version
   exec_program(${CMAKE_C_COMPILER} ARGS -dumpversion OUTPUT_VARIABLE _gcc_version)
   macro_ensure_version("4.4.1" "${_gcc_version}" GCC_4_4_1)
   if(NOT GCC_4_4_1)
      message(STATUS "\n-- \n-- NOTE: Your GCC is older than 4.4.1. This is known to cause problems/bugs. Please update to the latest GCC if you can.\n-- \n-- ")
      macro_ensure_version("4.3.0" "${_gcc_version}" GCC_4_3_0)
      if(NOT GCC_4_3_0)
         message(STATUS "WARNING: Your GCC is older than 4.3.0. It is unable to handle all SSE2 intrinsics. All SSE code will be disabled. Please update to the latest GCC if you can.\n-- \n-- ")
         set(SSE_INTRINSICS_BROKEN true)
      endif(NOT GCC_4_3_0)
   endif(NOT GCC_4_4_1)

   if(_gcc_version STREQUAL "4.6.0")
      list(APPEND disabled_targets
         gather_avx
         gather_VC_GATHER_SET_avx
         gather_sse_LOOP
         scatter_avx
         dhryrock_avx_benchmark
         )
   endif()

   if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
      set(ENABLE_STRICT_ALIASING true CACHE BOOL "Enables strict aliasing rules for more aggressive optimizations")
      if(NOT ENABLE_STRICT_ALIASING)
         set(CMAKE_CXX_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE} -fno-strict-aliasing ")
         set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fno-strict-aliasing ")
         set(CMAKE_C_FLAGS_RELEASE        "${CMAKE_C_FLAGS_RELEASE} -fno-strict-aliasing ")
         set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -fno-strict-aliasing ")
      endif(NOT ENABLE_STRICT_ALIASING)
   endif(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")

   # if compiling for 32 bit x86 we need to use the -mfpmath=sse since the x87 is broken by design
   if(NOT VOID_PTR_IS_64BIT)
      exec_program(${CMAKE_C_COMPILER} ARGS -dumpmachine OUTPUT_VARIABLE _gcc_machine)
      if(_gcc_machine MATCHES "[x34567]86")
         AddCompilerFlag("-mfpmath=sse")
      endif(_gcc_machine MATCHES "[x34567]86")
   endif(NOT VOID_PTR_IS_64BIT)

   # GCC uses binutils' assembler. And it has to be recent enough otherwise it'll bail out at some of the instructions Vc uses
   exec_program(${CMAKE_C_COMPILER} ARGS -print-prog-name=as OUTPUT_VARIABLE _binutils_as)
   if(NOT _binutils_as)
      message(WARNING "Could not find 'as', the assembler used by GCC. Hoping everything will work out...")
   else(NOT _binutils_as)
      if(APPLE)
         # it's not really binutils, but it'll give us the assembler version which is what we want
         exec_program(${_binutils_as} ARGS -v /dev/null OUTPUT_VARIABLE _as_version)
      else(APPLE)
         exec_program(${_binutils_as} ARGS --version OUTPUT_VARIABLE _as_version)
      endif(APPLE)
      string(REGEX REPLACE "\\([^\\)]*\\)" "" _as_version "${_as_version}")
      string(REGEX MATCH "[1-9]\\.[0-9]+(\\.[0-9]+)?" _as_version "${_as_version}")
      macro_ensure_version("2.18.93" "${_as_version}" _as_good)
      if(NOT _as_good)
         message(WARNING "Your binutils is too old (${_as_version}). Some optimizations of Vc will be disabled.")
         add_definitions(-DVC_NO_XGETBV) # old assembler doesn't know the xgetbv instruction
      endif(NOT _as_good)
   endif(NOT _binutils_as)
   mark_as_advanced(_binutils_as)
elseif(COMPILER_IS_INTEL)
   ##################################################################################################
   #                                          Intel Compiler                                        #
   ##################################################################################################

   set(CMAKE_CXX_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE} -O3")
   set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -DNDEBUG -O3")
   set(CMAKE_C_FLAGS_RELEASE          "${CMAKE_C_FLAGS_RELEASE} -O3")
   set(CMAKE_C_FLAGS_RELWITHDEBINFO   "${CMAKE_C_FLAGS_RELWITHDEBINFO} -DNDEBUG -O3")

   set(ALIAS_FLAGS "-no-ansi-alias")
   if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
      # default ICC to -no-ansi-alias because otherwise tests/utils_sse fails. So far I suspect a miscompilation...
      set(ENABLE_STRICT_ALIASING false CACHE BOOL "Enables strict aliasing rules for more aggressive optimizations")
      if(ENABLE_STRICT_ALIASING)
         set(ALIAS_FLAGS "-ansi-alias")
      endif(ENABLE_STRICT_ALIASING)
   endif(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")

   exec_program(${CMAKE_C_COMPILER} ARGS -dumpversion OUTPUT_VARIABLE _icc_version)
   macro_ensure_version("12.0.0" "${_icc_version}" ICC_12_0_0)
   if(ICC_12_0_0)
      # iomanip from latest libstdc++ makes ICC fail unless C++0x is selected
      AddCompilerFlag("-std=c++0x")
   endif()

   # per default icc is not IEEE compliant, but we need that for verification
   set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${ALIAS_FLAGS} -w1 -fp-model precise")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ALIAS_FLAGS} -w1 -fp-model precise")
elseif(COMPILER_IS_MSVC)
   AddCompilerFlag("/wd4800") # Disable warning "forcing value to bool"
   add_definitions(-D_CRT_SECURE_NO_WARNINGS)
   # MSVC does not support inline assembly on 64 bit! :(
   # searching the help for xgetbv doesn't turn up anything. So just fall back to not supporting AVX on Windows :(
   add_definitions(-DVC_NO_XGETBV)
endif()

find_package(Qt4)

OptimizeForArchitecture()

if(CMAKE_BUILD_TYPE STREQUAL "" AND NOT CMAKE_CXX_FLAGS MATCHES "-O[123]")
   message(STATUS "WARNING! It seems you are compiling without optimization. Please set CMAKE_BUILD_TYPE.")
endif(CMAKE_BUILD_TYPE STREQUAL "" AND NOT CMAKE_CXX_FLAGS MATCHES "-O[123]")

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(CpuId STATIC cpuid.cpp)
add_target_property(CpuId LABELS "other")

if(USE_AVX)
   add_library(Vc STATIC vector.cpp common/support.cpp avx/sorthelper.cpp)
else()
   add_library(Vc STATIC vector.cpp common/support.cpp)
endif()
add_target_property(Vc COMPILE_FLAGS "-DVC_COMPILE_LIB")
add_target_property(Vc LABELS "other")

add_custom_target(other VERBATIM)
add_custom_target(Scalar COMMENT "build Scalar code" VERBATIM)
add_custom_target(SSE COMMENT "build SSE code" VERBATIM)
add_custom_target(AVX COMMENT "build AVX code" VERBATIM)
add_custom_target(LRBni COMMENT "build LRBni code" VERBATIM)
add_dependencies(other Vc)

install(TARGETS Vc DESTINATION lib/)
install(DIRECTORY include/Vc/ DESTINATION include/Vc)
install(DIRECTORY larrabee scalar sse avx common DESTINATION include/Vc FILES_MATCHING REGEX "/*.(h|tcc|def)$")

install(FILES
   cmake/FindVc.cmake
   cmake/OptimizeForArchitecture.cmake
   cmake/AddCompilerFlag.cmake
   cmake/MacroEnsureVersion.cmake
   cmake/CheckCCompilerFlag.cmake
   DESTINATION share/Vc/cmake/
   )

include (CTest)
configure_file(${CMAKE_SOURCE_DIR}/CTestCustom.cmake ${CMAKE_BINARY_DIR}/CTestCustom.cmake COPYONLY)
if(BUILD_TESTING)
   add_subdirectory(tests)
endif(BUILD_TESTING)

set(BUILD_BENCHMARKS TRUE CACHE BOOL "Build benchmarks.")
if(BUILD_BENCHMARKS)
   add_subdirectory(benchmarks)
endif(BUILD_BENCHMARKS)

set(BUILD_EXAMPLES FALSE CACHE BOOL "Build examples.")
if(BUILD_EXAMPLES)
   add_subdirectory(examples)
endif(BUILD_EXAMPLES)

mark_as_advanced(prototypes_header)
