# Macros for use with the Vc library. Vc can be found at http://code.compeng.uni-frankfurt.de/projects/vc
#
# The following macros are provided:
# vc_determine_compiler
# vc_set_preferred_compiler_flags
#
#=============================================================================
# Copyright 2009-2012   Matthias Kretz <kretz@kde.org>
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file CmakeCopyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================

include (MacroEnsureVersion)
include (AddCompilerFlag)

macro(vc_determine_compiler)
   set(VC_COMPILER_IS_INTEL false)
   set(VC_COMPILER_IS_OPEN64 false)
   set(VC_COMPILER_IS_MSVC false)
   set(VC_COMPILER_IS_GCC false)
   if(CMAKE_CXX_COMPILER MATCHES "/(icpc|icc)$")
      set(VC_COMPILER_IS_INTEL true)
   elseif(CMAKE_CXX_COMPILER MATCHES "/(opencc|openCC)$")
      set(VC_COMPILER_IS_OPEN64 true)
   elseif(MSVC)
      set(VC_COMPILER_IS_MSVC true)
   elseif(CMAKE_COMPILER_IS_GNUCXX)
      set(VC_COMPILER_IS_GCC true)
   else()
      message(ERROR "Unsupported Compiler for use with Vc.\nPlease fill out the missing parts in the CMake scripts and submit a patch to http://code.compeng.uni-frankfurt.de/projects/vc")
   endif()
endmacro()

macro(vc_set_preferred_compiler_flags)
   include (CheckCXXSourceRuns)
   check_cxx_source_runs("int main() { return sizeof(void*) != 8; }" VOID_PTR_IS_64BIT)

   set(SSE_INTRINSICS_BROKEN false)

   if(VC_COMPILER_IS_OPEN64)
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
   elseif(VC_COMPILER_IS_GCC)
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
      set(CMAKE_CXX_FLAGS_DEBUG          "-g3"          CACHE STRING "Flags used by the compiler during debug builds." FORCE)
      set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG" CACHE STRING "Flags used by the compiler during release minsize builds." FORCE)
      set(CMAKE_CXX_FLAGS_RELEASE        "-O3 -DNDEBUG" CACHE STRING "Flags used by the compiler during release builds (/MD /Ob1 /Oi /Ot /Oy /Gs will produce slightly less optimized but smaller files)." FORCE)
      set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} -g" CACHE STRING "Flags used by the compiler during Release with Debug Info builds." FORCE)
      set(CMAKE_C_FLAGS_DEBUG          "${CMAKE_CXX_FLAGS_DEBUG}"          CACHE STRING "Flags used by the compiler during debug builds." FORCE)
      set(CMAKE_C_FLAGS_MINSIZEREL     "${CMAKE_CXX_FLAGS_MINSIZEREL}"     CACHE STRING "Flags used by the compiler during release minsize builds." FORCE)
      set(CMAKE_C_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE}"        CACHE STRING "Flags used by the compiler during release builds (/MD /Ob1 /Oi /Ot /Oy /Gs will produce slightly less optimized but smaller files)." FORCE)
      set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}" CACHE STRING "Flags used by the compiler during Release with Debug Info builds." FORCE)

      # check the GCC version
      exec_program(${CMAKE_C_COMPILER} ARGS -dumpversion OUTPUT_VARIABLE _gcc_version)
      # some distributions patch their GCC to return nothing or only major and minor version on -dumpversion.
      # In that case we must extract the version number from --version.
      if(NOT _gcc_version OR _gcc_version MATCHES "^[0-9]\\.[0-9]+$")
         exec_program(${CMAKE_C_COMPILER} ARGS --version OUTPUT_VARIABLE _gcc_version)
         string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" _gcc_version "${_gcc_version}")
         message(STATUS "GCC Version from --version: ${_gcc_version}")
      endif()
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
            gather_sse
            gather_VC_USE_SET_GATHERS_avx
            gather_VC_USE_SET_GATHERS_sse
            gather_sse_LOOP
            scatter_avx
            scatter_sse
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
   elseif(VC_COMPILER_IS_INTEL)
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
   elseif(VC_COMPILER_IS_MSVC)
      AddCompilerFlag("/wd4800") # Disable warning "forcing value to bool"
      AddCompilerFlag("/wd4996") # Disable warning about strdup vs. _strdup
      AddCompilerFlag("/wd4244") # Disable warning "conversion from 'unsigned int' to 'float', possible loss of data"
      AddCompilerFlag("/wd4146") # Disable warning "unary minus operator applied to unsigned type, result still unsigned"
      add_definitions(-D_CRT_SECURE_NO_WARNINGS)
      # MSVC does not support inline assembly on 64 bit! :(
      # searching the help for xgetbv doesn't turn up anything. So just fall back to not supporting AVX on Windows :(
      add_definitions(-DVC_NO_XGETBV)
      # get rid of the min/max macros
      add_definitions(-DNOMINMAX)
   endif()
endmacro()
