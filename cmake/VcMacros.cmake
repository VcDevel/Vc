# Macros for use with the Vc library. Vc can be found at http://code.compeng.uni-frankfurt.de/projects/vc
#
# The following macros are provided:
# vc_determine_compiler
# vc_set_preferred_compiler_flags
#
#=============================================================================
# Copyright 2009-2015   Matthias Kretz <kretz@kde.org>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#  * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  * Neither the names of contributing organizations nor the
#    names of its contributors may be used to endorse or promote products
#    derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#=============================================================================

cmake_minimum_required(VERSION 2.8.3)

get_filename_component(_currentDir "${CMAKE_CURRENT_LIST_FILE}" PATH)
include ("${_currentDir}/UserWarning.cmake")
include ("${_currentDir}/AddCompilerFlag.cmake")
include ("${_currentDir}/OptimizeForArchitecture.cmake")

macro(vc_determine_compiler)
   if(NOT DEFINED Vc_COMPILER_IS_INTEL)
      execute_process(COMMAND "${CMAKE_CXX_COMPILER}" "--version" OUTPUT_VARIABLE _cxx_compiler_version ERROR_VARIABLE _cxx_compiler_version)
      set(Vc_COMPILER_IS_INTEL false)
      set(Vc_COMPILER_IS_OPEN64 false)
      set(Vc_COMPILER_IS_CLANG false)
      set(Vc_COMPILER_IS_MSVC false)
      set(Vc_COMPILER_IS_GCC false)
      if(CMAKE_CXX_COMPILER MATCHES "/(icpc|icc)$")
         set(Vc_COMPILER_IS_INTEL true)
         exec_program(${CMAKE_CXX_COMPILER} ARGS -dumpversion OUTPUT_VARIABLE Vc_ICC_VERSION)
         message(STATUS "Detected Compiler: Intel ${Vc_ICC_VERSION}")

         # break build with too old clang as early as possible.
         if(Vc_ICC_VERSION VERSION_LESS 15.0.3)
            message(FATAL_ERROR "Vc 1.x requires C++11 support. This requires at least ICC 15.0.3")
         endif()
      elseif(CMAKE_CXX_COMPILER MATCHES "(opencc|openCC)$")
         set(Vc_COMPILER_IS_OPEN64 true)
         message(STATUS "Detected Compiler: Open64")
      elseif(CMAKE_CXX_COMPILER MATCHES "clang\\+\\+$" OR "${_cxx_compiler_version}" MATCHES "clang")
         set(Vc_COMPILER_IS_CLANG true)
         exec_program(${CMAKE_CXX_COMPILER} ARGS --version OUTPUT_VARIABLE Vc_CLANG_VERSION)
         string(REGEX MATCH "[0-9]+\\.[0-9]+(\\.[0-9]+)?" Vc_CLANG_VERSION "${Vc_CLANG_VERSION}")
         message(STATUS "Detected Compiler: Clang ${Vc_CLANG_VERSION}")

         # break build with too old clang as early as possible.
         if(Vc_CLANG_VERSION VERSION_LESS 3.4)
            message(FATAL_ERROR "Vc 1.x requires C++11 support. This requires at least clang 3.4")
         endif()
      elseif(MSVC)
         set(Vc_COMPILER_IS_MSVC true)
         execute_process(COMMAND ${CMAKE_CXX_COMPILER} /nologo -EP "${_currentDir}/msvc_version.c" OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE Vc_MSVC_VERSION)
         string(STRIP "${Vc_MSVC_VERSION}" Vc_MSVC_VERSION)
         string(REPLACE "MSVC " "" Vc_MSVC_VERSION "${Vc_MSVC_VERSION}")
         message(STATUS "Detected Compiler: MSVC ${Vc_MSVC_VERSION}")
      elseif(CMAKE_COMPILER_IS_GNUCXX)
         set(Vc_COMPILER_IS_GCC true)
         exec_program(${CMAKE_CXX_COMPILER} ARGS -dumpversion OUTPUT_VARIABLE Vc_GCC_VERSION)
         message(STATUS "Detected Compiler: GCC ${Vc_GCC_VERSION}")

         # some distributions patch their GCC to return nothing or only major and minor version on -dumpversion.
         # In that case we must extract the version number from --version.
         if(NOT Vc_GCC_VERSION OR Vc_GCC_VERSION MATCHES "^[0-9]\\.[0-9]+$")
            exec_program(${CMAKE_CXX_COMPILER} ARGS --version OUTPUT_VARIABLE Vc_GCC_VERSION)
            string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" Vc_GCC_VERSION "${Vc_GCC_VERSION}")
            message(STATUS "GCC Version from --version: ${Vc_GCC_VERSION}")
         endif()

         # some distributions patch their GCC to be API incompatible to what the FSF released. In
         # those cases we require a macro to identify the distribution version
         find_program(Vc_lsb_release lsb_release)
         mark_as_advanced(Vc_lsb_release)
         if(Vc_lsb_release)
            if(NOT Vc_distributor_id)
              execute_process(COMMAND ${Vc_lsb_release} -is OUTPUT_VARIABLE Vc_distributor_id OUTPUT_STRIP_TRAILING_WHITESPACE)
              string(TOUPPER "${Vc_distributor_id}" Vc_distributor_id)
              set(Vc_distributor_id "${Vc_distributor_id}" CACHE STRING "lsb distribution id")
              execute_process(COMMAND ${Vc_lsb_release} -rs OUTPUT_VARIABLE Vc_distributor_release OUTPUT_STRIP_TRAILING_WHITESPACE)
              set(Vc_distributor_release "${Vc_distributor_release}" CACHE STRING "lsb release id")
            endif()
            if(Vc_distributor_id STREQUAL "UBUNTU")
               execute_process(COMMAND ${CMAKE_CXX_COMPILER} --version OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE _gcc_version)
               string(REGEX MATCH "\\(.* ${Vc_GCC_VERSION}-([0-9]+).*\\)" _tmp "${_gcc_version}")
               if(_tmp)
                  set(_patch ${CMAKE_MATCH_1})
                  string(REGEX MATCH "^([0-9]+)\\.([0-9]+)$" _tmp "${Vc_distributor_release}")
                  execute_process(COMMAND printf 0x%x%02x%02x ${CMAKE_MATCH_1} ${CMAKE_MATCH_2} ${_patch} OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE _tmp)
                  set(Vc_DEFINITIONS "${Vc_DEFINITIONS} -D__GNUC_UBUNTU_VERSION__=${_tmp}")
               endif()
            endif()
         endif()

         # break build with too old GCC as early as possible.
         if(Vc_GCC_VERSION VERSION_LESS 4.8.1)
            message(FATAL_ERROR "Vc 1.x requires C++11 support. This requires at least GCC 4.8.1")
         endif()
      else()
         message(WARNING "Untested/-supported Compiler (${CMAKE_CXX_COMPILER}) for use with Vc.\nPlease fill out the missing parts in the CMake scripts and submit a patch to http://code.compeng.uni-frankfurt.de/projects/vc")
      endif()
   endif()
endmacro()

macro(vc_set_gnu_buildtype_flags)
   set(CMAKE_CXX_FLAGS_DEBUG          "-g3"          CACHE STRING "Flags used by the compiler during debug builds." FORCE)
   set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG" CACHE STRING "Flags used by the compiler during release minsize builds." FORCE)
   set(CMAKE_CXX_FLAGS_RELEASE        "-O3 -DNDEBUG" CACHE STRING "Flags used by the compiler during release builds (/MD /Ob1 /Oi /Ot /Oy /Gs will produce slightly less optimized but smaller files)." FORCE)
   set(CMAKE_CXX_FLAGS_RELWITHDEBUG   "-O3"          CACHE STRING "Flags used by the compiler during release builds containing runtime checks." FORCE)
   set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBUG} -g" CACHE STRING "Flags used by the compiler during Release with Debug Info builds." FORCE)
   set(CMAKE_C_FLAGS_DEBUG          "${CMAKE_CXX_FLAGS_DEBUG}"          CACHE STRING "Flags used by the compiler during debug builds." FORCE)
   set(CMAKE_C_FLAGS_MINSIZEREL     "${CMAKE_CXX_FLAGS_MINSIZEREL}"     CACHE STRING "Flags used by the compiler during release minsize builds." FORCE)
   set(CMAKE_C_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE}"        CACHE STRING "Flags used by the compiler during release builds (/MD /Ob1 /Oi /Ot /Oy /Gs will produce slightly less optimized but smaller files)." FORCE)
   set(CMAKE_C_FLAGS_RELWITHDEBUG   "${CMAKE_CXX_FLAGS_RELWITHDEBUG}"   CACHE STRING "Flags used by the compiler during release builds containing runtime checks." FORCE)
   set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}" CACHE STRING "Flags used by the compiler during Release with Debug Info builds." FORCE)
   if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebug")
      set(ENABLE_STRICT_ALIASING true CACHE BOOL "Enables strict aliasing rules for more aggressive optimizations")
      if(NOT ENABLE_STRICT_ALIASING)
         AddCompilerFlag(-fno-strict-aliasing)
      endif(NOT ENABLE_STRICT_ALIASING)
   endif()
   mark_as_advanced(CMAKE_CXX_FLAGS_RELWITHDEBUG CMAKE_C_FLAGS_RELWITHDEBUG)
endmacro()

macro(vc_add_compiler_flag VAR _flag)
   AddCompilerFlag("${_flag}" CXX_FLAGS ${VAR})
endmacro()

macro(vc_check_assembler)
   exec_program(${CMAKE_CXX_COMPILER} ARGS -print-prog-name=as OUTPUT_VARIABLE _as)
   mark_as_advanced(_as)
   if(NOT _as)
      message(WARNING "Could not find 'as', the assembler used by GCC. Hoping everything will work out...")
   else()
      exec_program(${_as} ARGS --version OUTPUT_VARIABLE _as_version)
      string(REGEX REPLACE "\\([^\\)]*\\)" "" _as_version "${_as_version}")
      string(REGEX MATCH "[1-9]\\.[0-9]+(\\.[0-9]+)?" _as_version "${_as_version}")
      if(_as_version VERSION_LESS "2.18.93")
         UserWarning("Your binutils is too old (${_as_version}). Some optimizations of Vc will be disabled.")
         set(Vc_DEFINITIONS "${Vc_DEFINITIONS} -DVc_NO_XGETBV") # old assembler doesn't know the xgetbv instruction
         set(Vc_AVX_INTRINSICS_BROKEN true)
         set(Vc_XOP_INTRINSICS_BROKEN true)
         set(Vc_FMA4_INTRINSICS_BROKEN true)
      elseif(_as_version VERSION_LESS "2.21.0")
         UserWarning("Your binutils is too old (${_as_version}) for XOP and AVX2 instructions. They will therefore not be provided in libVc.")
         set(Vc_XOP_INTRINSICS_BROKEN true)
         set(Vc_AVX2_INTRINSICS_BROKEN true)
      endif()
   endif()
endmacro()

macro(vc_set_preferred_compiler_flags)
   vc_determine_compiler()

   set(_add_warning_flags false)
   set(_add_buildtype_flags false)
   foreach(_arg ${ARGN})
      if(_arg STREQUAL "WARNING_FLAGS")
         set(_add_warning_flags true)
      elseif(_arg STREQUAL "BUILDTYPE_FLAGS")
         set(_add_buildtype_flags true)
      endif()
   endforeach()

   set(Vc_SSE_INTRINSICS_BROKEN false)
   set(Vc_AVX_INTRINSICS_BROKEN false)
   set(Vc_AVX2_INTRINSICS_BROKEN false)
   set(Vc_XOP_INTRINSICS_BROKEN false)
   set(Vc_FMA4_INTRINSICS_BROKEN false)

   if(Vc_COMPILER_IS_OPEN64)
      ##################################################################################################
      #                                             Open64                                             #
      ##################################################################################################
      if(_add_warning_flags)
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
         AddCompilerFlag("-pedantic")
         AddCompilerFlag("-Wno-long-long")
         AddCompilerFlag("-Wshadow")
         AddCompilerFlag("-Wold-style-cast")
         AddCompilerFlag("-Wno-variadic-macros")
      endif()
      if(_add_buildtype_flags)
         vc_set_gnu_buildtype_flags()
      endif()

      vc_check_assembler()

      # Open64 4.5.1 still doesn't ship immintrin.h
      set(Vc_AVX_INTRINSICS_BROKEN true)
      set(Vc_AVX2_INTRINSICS_BROKEN true)
   elseif(Vc_COMPILER_IS_GCC)
      ##################################################################################################
      #                                              GCC                                               #
      ##################################################################################################
      if(_add_warning_flags)
         foreach(_f -W -Wall -Wswitch -Wformat -Wchar-subscripts -Wparentheses -Wmultichar -Wtrigraphs -Wpointer-arith -Wcast-align -Wreturn-type -pedantic -Wshadow -Wundef)
            AddCompilerFlag("${_f}")
         endforeach()
         foreach(_f -Wold-style-cast)
            AddCompilerFlag("${_f}" CXX_FLAGS CMAKE_CXX_FLAGS)
         endforeach()
      endif()
      vc_add_compiler_flag(Vc_COMPILE_FLAGS "-Wabi")
      vc_add_compiler_flag(Vc_COMPILE_FLAGS "-fabi-version=0") # ABI version 4 is required to make __m128 and __m256 appear as different types. 0 should give us the latest version.
      vc_add_compiler_flag(Vc_COMPILE_FLAGS "-fabi-compat-version=0") # GCC 5 introduced this switch
      # and defaults it to 2 if -fabi-version is 0. But in that case the bug -fabi-version=0 is
      # supposed to fix resurfaces. For now just make sure that it compiles and links.
      # Bug report pending.

      if(_add_buildtype_flags)
         vc_set_gnu_buildtype_flags()
      endif()

      if(APPLE)
         # The GNU assembler (as) on Mac OS X is hopelessly outdated. The -q flag
         # to as tells it to use the clang assembler, though, which is fine.
         # -Wa,-q tells GCC to pass -q to as.
         vc_add_compiler_flag(Vc_COMPILE_FLAGS "-Wa,-q")
         # Apparently the MacOS clang assember doesn't understand XOP instructions.
         set(Vc_XOP_INTRINSICS_BROKEN true)
      else()
         vc_check_assembler()
      endif()
   elseif(Vc_COMPILER_IS_INTEL)
      ##################################################################################################
      #                                          Intel Compiler                                        #
      ##################################################################################################

      if(_add_buildtype_flags)
         set(CMAKE_CXX_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE} -O3")
         set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -DNDEBUG -O3")
         set(CMAKE_C_FLAGS_RELEASE          "${CMAKE_C_FLAGS_RELEASE} -O3")
         set(CMAKE_C_FLAGS_RELWITHDEBINFO   "${CMAKE_C_FLAGS_RELWITHDEBINFO} -DNDEBUG -O3")
      endif()
      vc_add_compiler_flag(Vc_COMPILE_FLAGS "-diag-disable 913")
      # Disable warning #13211 "Immediate parameter to intrinsic call too large". (sse/vector.tcc rotated(int))
      vc_add_compiler_flag(Vc_COMPILE_FLAGS "-diag-disable 13211")
      vc_add_compiler_flag(Vc_COMPILE_FLAGS "-diag-disable 61") # warning #61: integer operation result is out of range
      vc_add_compiler_flag(Vc_COMPILE_FLAGS "-diag-disable 173") # warning #173: floating-point value does not fit in required integral type
      vc_add_compiler_flag(Vc_COMPILE_FLAGS "-diag-disable 264") # warning #264: floating-point value does not fit in required floating-point type
      if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
         set(ENABLE_STRICT_ALIASING true CACHE BOOL "Enables strict aliasing rules for more aggressive optimizations")
         if(ENABLE_STRICT_ALIASING)
            AddCompilerFlag(-ansi-alias CXX_FLAGS Vc_COMPILE_FLAGS)
         else()
            AddCompilerFlag(-no-ansi-alias CXX_FLAGS Vc_COMPILE_FLAGS)
         endif()
      endif()

      if(NOT "$ENV{DASHBOARD_TEST_FROM_CTEST}" STREQUAL "")
         # disable warning #2928: the __GXX_EXPERIMENTAL_CXX0X__ macro is disabled when using GNU version 4.6 with the c++0x option
         # this warning just adds noise about problems in the compiler - but I'm only interested in seeing problems in Vc
         vc_add_compiler_flag(Vc_COMPILE_FLAGS "-diag-disable 2928")
      endif()

      # Intel doesn't implement the XOP or FMA4 intrinsics
      set(Vc_XOP_INTRINSICS_BROKEN true)
      set(Vc_FMA4_INTRINSICS_BROKEN true)
   elseif(Vc_COMPILER_IS_MSVC)
      ##################################################################################################
      #                                      Microsoft Visual Studio                                   #
      ##################################################################################################

      if(_add_warning_flags)
         AddCompilerFlag("/wd4800") # Disable warning "forcing value to bool"
         AddCompilerFlag("/wd4996") # Disable warning about strdup vs. _strdup
         AddCompilerFlag("/wd4244") # Disable warning "conversion from 'unsigned int' to 'float', possible loss of data"
         AddCompilerFlag("/wd4146") # Disable warning "unary minus operator applied to unsigned type, result still unsigned"
         AddCompilerFlag("/wd4227") # Disable warning "anachronism used : qualifiers on reference are ignored" (this is about 'restrict' usage on references, stupid MSVC)
         AddCompilerFlag("/wd4722") # Disable warning "destructor never returns, potential memory leak" (warns about ~_UnitTest_Global_Object which we don't care about)
         AddCompilerFlag("/wd4748") # Disable warning "/GS can not protect parameters and local variables from local buffer overrun because optimizations are disabled in function" (I don't get it)
         add_definitions(-D_CRT_SECURE_NO_WARNINGS)
      endif()
      vc_add_compiler_flag(Vc_COMPILE_FLAGS "/Gv") # default to __vectorcall

      # get rid of the min/max macros
      set(Vc_DEFINITIONS "${Vc_DEFINITIONS} -DNOMINMAX")

      # MSVC doesn't implement the XOP or FMA4 intrinsics
      #set(Vc_XOP_INTRINSICS_BROKEN true)
      #set(Vc_FMA4_INTRINSICS_BROKEN true)

      if(MSVC_VERSION LESS 1900)
         UserWarning("MSVC before 2015 does not support enough of C++11")
      endif()
   elseif(Vc_COMPILER_IS_CLANG)
      ##################################################################################################
      #                                              Clang                                             #
      ##################################################################################################

      if(Vc_CLANG_VERSION VERSION_GREATER "3.5.99" AND Vc_CLANG_VERSION VERSION_LESS 3.7.0)
         UserWarning("Clang 3.6 has serious issues with AVX code generation, frequently losing 50% of the data. AVX is therefore disabled.\nPlease update to a more recent clang version.\n")
         set(Vc_AVX_INTRINSICS_BROKEN true)
         set(Vc_AVX2_INTRINSICS_BROKEN true)
      endif()

      # disable these warnings because clang shows them for function overloads that were discarded via SFINAE
      vc_add_compiler_flag(Vc_COMPILE_FLAGS "-Wno-local-type-template-args")
      vc_add_compiler_flag(Vc_COMPILE_FLAGS "-Wno-unnamed-type-template-args")
   endif()

   if(NOT Vc_COMPILER_IS_MSVC)
      vc_add_compiler_flag(Vc_COMPILE_FLAGS "-ffp-contract=fast")
   endif()

   OptimizeForArchitecture()
   set(Vc_IMPL "auto" CACHE STRING "Force the Vc implementation globally to the selected instruction set. \"auto\" lets Vc use the best available instructions.")
   if(NOT Vc_IMPL STREQUAL "auto")
      set(Vc_DEFINITIONS "${Vc_DEFINITIONS} -DVc_IMPL=${Vc_IMPL}")
      if(NOT Vc_IMPL STREQUAL "Scalar")
         set(_use_var "USE_${Vc_IMPL}")
         if(Vc_IMPL STREQUAL "SSE")
            set(_use_var "USE_SSE2")
         endif()
         if(NOT ${_use_var})
            message(WARNING "The selected value for Vc_IMPL (${Vc_IMPL}) will not work because the relevant instructions are not enabled via compiler flags.")
         endif()
      endif()
   endif()
endmacro()

# helper macro for vc_compile_for_all_implementations
macro(_vc_compile_one_implementation _srcs _impl)
   list(FIND _disabled_targets "${_impl}" _disabled_index)
   list(FIND _only_targets "${_impl}" _only_index)
   if(${_disabled_index} EQUAL -1 AND (NOT _only_targets OR ${_only_index} GREATER -1))
      set(_extra_flags)
      set(_ok FALSE)
      foreach(_flags_it ${ARGN})
         if(_flags_it STREQUAL "NO_FLAG")
            set(_ok TRUE)
            break()
         endif()
         string(REPLACE " " ";" _flag_list "${_flags_it}")
         foreach(_f ${_flag_list})
            AddCompilerFlag(${_f} CXX_RESULT _ok)
            if(NOT _ok)
               break()
            endif()
         endforeach()
         if(_ok)
            set(_extra_flags ${_flags_it})
            break()
         endif()
      endforeach()

      if(Vc_COMPILER_IS_MSVC)
         # MSVC for 64bit does not recognize /arch:SSE2 anymore. Therefore we set override _ok if _impl
         # says SSE
         if("${_impl}" MATCHES "SSE")
            set(_ok TRUE)
         endif()
      endif()

      if(_ok)
         get_filename_component(_out "${_vc_compile_src}" NAME_WE)
         get_filename_component(_ext "${_vc_compile_src}" EXT)
         set(_out "${CMAKE_CURRENT_BINARY_DIR}/${_out}_${_impl}${_ext}")
         add_custom_command(OUTPUT "${_out}"
            COMMAND ${CMAKE_COMMAND} -E copy "${_vc_compile_src}" "${_out}"
            DEPENDS "${_vc_compile_src}"
            COMMENT "Copy to ${_out}"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            VERBATIM)
         set_source_files_properties( "${_out}" PROPERTIES
            COMPILE_DEFINITIONS "Vc_IMPL=${_impl}"
            COMPILE_FLAGS "${_flags} ${_extra_flags}"
         )
         list(APPEND ${_srcs} "${_out}")
      endif()
   endif()
endmacro()

# Generate compile rules for the given C++ source file for all available implementations and return
# the resulting list of object files in _obj
# all remaining arguments are additional flags
# Example:
#   vc_compile_for_all_implementations(_objs src/trigonometric.cpp FLAGS -DCOMPILE_BLAH EXCLUDE Scalar)
#   add_executable(executable main.cpp ${_objs})
macro(vc_compile_for_all_implementations _srcs _src)
   set(_flags)
   unset(_disabled_targets)
   unset(_only_targets)
   set(_state 0)
   foreach(_arg ${ARGN})
      if(_arg STREQUAL "FLAGS")
         set(_state 1)
      elseif(_arg STREQUAL "EXCLUDE")
         set(_state 2)
      elseif(_arg STREQUAL "ONLY")
         set(_state 3)
      elseif(_state EQUAL 1)
         set(_flags "${_flags} ${_arg}")
      elseif(_state EQUAL 2)
         list(APPEND _disabled_targets "${_arg}")
      elseif(_state EQUAL 3)
         list(APPEND _only_targets "${_arg}")
      else()
         message(FATAL_ERROR "incorrect argument to vc_compile_for_all_implementations")
      endif()
   endforeach()

   set(_vc_compile_src "${_src}")

   _vc_compile_one_implementation(${_srcs} Scalar NO_FLAG)
   if(NOT Vc_SSE_INTRINSICS_BROKEN)
      _vc_compile_one_implementation(${_srcs} SSE2   "-xSSE2"   "-msse2"   "/arch:SSE2")
      _vc_compile_one_implementation(${_srcs} SSE3   "-xSSE3"   "-msse3"   "/arch:SSE2")
      _vc_compile_one_implementation(${_srcs} SSSE3  "-xSSSE3"  "-mssse3"  "/arch:SSE2")
      _vc_compile_one_implementation(${_srcs} SSE4_1 "-xSSE4.1" "-msse4.1" "/arch:SSE2")
      _vc_compile_one_implementation(${_srcs} SSE4_2 "-xSSE4.2" "-msse4.2" "/arch:SSE2")
      _vc_compile_one_implementation(${_srcs} SSE3+SSE4a  "-msse4a")
   endif()
   if(NOT Vc_AVX_INTRINSICS_BROKEN)
      _vc_compile_one_implementation(${_srcs} AVX      "-xAVX"    "-mavx"    "/arch:AVX")
      if(NOT Vc_XOP_INTRINSICS_BROKEN)
         if(NOT Vc_FMA4_INTRINSICS_BROKEN)
            _vc_compile_one_implementation(${_srcs} SSE+XOP+FMA4 "-mxop -mfma4"        ""    "")
            _vc_compile_one_implementation(${_srcs} AVX+XOP+FMA4 "-mavx -mxop -mfma4"  ""    "")
         endif()
         _vc_compile_one_implementation(${_srcs} SSE+XOP+FMA "-mxop -mfma"        ""    "")
         _vc_compile_one_implementation(${_srcs} AVX+XOP+FMA "-mavx -mxop -mfma"  ""    "")
      endif()
      _vc_compile_one_implementation(${_srcs} AVX+FMA "-mavx -mfma"  ""    "")
   endif()
   if(NOT Vc_AVX2_INTRINSICS_BROKEN)
      # The necessary list is not clear to me yet. At this point I'll only consider Intel CPUs, in
      # which case AVX2 implies the availability of FMA and BMI2
      #_vc_compile_one_implementation(${_srcs} AVX2  "-mavx2")
      #_vc_compile_one_implementation(${_srcs} AVX2+BMI2 "-mavx2 -mbmi2")
      _vc_compile_one_implementation(${_srcs} AVX2+FMA+BMI2 "-xCORE-AVX2" "-mavx2 -mfma -mbmi2" "/arch:AVX2")
      #_vc_compile_one_implementation(${_srcs} AVX2+FMA "-mavx2 -mfma")
   endif()
endmacro()
