#=============================================================================
# Copyright © 2010-2015 Matthias Kretz <kretz@kde.org>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the names of contributing organizations nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#=============================================================================
#
# This check will search for a MIC compiler and check whether the C and C++
# compilers are able to offload via offload pragma and target(mic) attribute.
# The project may choose to either build native MIC binaries, or offload
# binaries (hybrid code), or both. In the case where only native MIC binaries
# are built, the compiler does not need to support offloading
#
# MIC_NATIVE_FOUND is true if native MIC binaries can be built
# MIC_OFFLOAD_FOUND is true if hybrid host/MIC binaries via offload can be built
# MIC_FOUND is true if either MIC_NATIVE_FOUND or MIC_OFFLOAD_FOUND is true
#
# When MIC_NATIVE_FOUND is true you can use the macros
# mic_add_definitions
# mic_include_directories
# mic_set_link_libraries
# mic_add_library
# mic_add_executable
# for building native libraries/executables
#
# When MIC_OFFLOAD_FOUND is true you use the standard cmake macros to build
# libraries and executables but have to make sure manually that the necessary
# offload compiler switches are present. You might want to add something like:
# if(MIC_OFFLOAD_FOUND)
#    AddCompilerFlag("-offload-build")
#    AddCompilerFlag("-offload-copts=-vec-report=3 -H")
#    AddCompilerFlag("-offload-ldopts=-lmylib")
#    AddCompilerFlag("-opt-report-phase=offload")
# endif()

set(MIC_FOUND false)
set(MIC_NATIVE_FOUND false)
set(MIC_OFFLOAD_FOUND false)

if(CMAKE_GENERATOR MATCHES "Makefile")
   option(ENABLE_MIC "Enable native builds for the MIC architecture (Intel Knights Corner)" ON)
else()
   message(STATUS "MIC builds are only supported with a Makefile generator")
   set(ENABLE_MIC false)
endif()
if(ENABLE_MIC)
   file(GLOB _intel_dirs "/opt/intel/compilers_and_libraries_*/linux")
   if ("${_intel_dirs}" STREQUAL "")
      file(GLOB _intel_dirs "/opt/intel/composer_xe_*")
   endif()

   list(SORT _intel_dirs)
   list(REVERSE _intel_dirs)
   find_path(MIC_SDK_DIR bin/intel64_mic/icpc PATHS
      "$ENV{MIC_SDK_DIR}"
      ${_intel_dirs}
      )
   mark_as_advanced(MIC_SDK_DIR)

   ##############################################################################
   # First check whether offload works

   # For now offload is not supported so skip it
#   if(NOT DEFINED c_compiler_can_offload OR NOT DEFINED cxx_compiler_can_offload)
#      set(c_compiler_can_offload FALSE)
#      set(cxx_compiler_can_offload FALSE)
#
#      include(CheckCSourceCompiles)
#      include(CheckCXXSourceCompiles)
#
#      #find_library(MIC_HOST_IMF_LIBRARY   imf   HINTS ENV LIBRARY_PATH)
#      #find_library(MIC_HOST_SVML_LIBRARY  svml  HINTS ENV LIBRARY_PATH)
#      #find_library(MIC_HOST_INTLC_LIBRARY intlc HINTS ENV LIBRARY_PATH)
#
#      #set(MIC_HOST_LIBS ${MIC_HOST_IMF_LIBRARY} ${MIC_HOST_SVML_LIBRARY} ${MIC_HOST_INTLC_LIBRARY})
#
#      set(_mic_offload_test_source "
##ifdef __MIC__
##include <immintrin.h>
##endif
#__attribute__((target(mic))) void test()
#{
##ifdef __MIC__
# __m512 v = _mm512_setzero_ps();
# (void)v;
##endif
#}
#
#int main()
#{
##pragma offload target(mic)
# test();
# return 0;
#}
#")
#      set(CMAKE_REQUIRED_FLAGS "-offload-build")
#      check_c_source_compiles("${_mic_offload_test_source}" c_compiler_can_offload)
#      check_cxx_source_compiles("${_mic_offload_test_source}" cxx_compiler_can_offload)
#      set(CMAKE_REQUIRED_FLAGS)
#   endif()
#
#   if(c_compiler_can_offload AND cxx_compiler_can_offload)
#      message(STATUS "C/C++ Compiler can offload to MIC.")
#      set(MIC_OFFLOAD_FOUND true)
#   else()
#      message(STATUS "C/C++ Compiler can NOT offload to MIC.")
#   endif()

   ##############################################################################
   # Next check whether everything required for native builds is available

   find_path(MIC_TARGET_TOOLS_DIR bin/x86_64-k1om-linux-ar HINTS
      "$ENV{MIC_TARGET_TOOLS_DIR}"
      "${MIC_SDK_DIR}/target"
      "/usr/linux-k1om-4.7"
      )
   find_program(MIC_AR x86_64-k1om-linux-ar PATHS "${MIC_TARGET_TOOLS_DIR}/bin")
   find_program(MIC_RANLIB x86_64-k1om-linux-ranlib PATHS "${MIC_TARGET_TOOLS_DIR}/bin")
   find_program(MIC_OBJCOPY x86_64-k1om-linux-objcopy PATHS "${MIC_TARGET_TOOLS_DIR}/bin")
   find_program(MIC_NATIVELOAD micnativeloadex PATHS ENV PATH)
   mark_as_advanced(MIC_TARGET_TOOLS_DIR MIC_AR MIC_RANLIB MIC_NATIVELOAD MIC_OBJCOPY)

   if(MIC_SDK_DIR AND MIC_AR AND MIC_RANLIB)
      find_program(MIC_CC  icc  HINTS "${MIC_SDK_DIR}/bin" "${MIC_SDK_DIR}/bin/intel64")
      find_program(MIC_CXX icpc HINTS "${MIC_SDK_DIR}/bin" "${MIC_SDK_DIR}/bin/intel64")

      find_library(MIC_IMF_LIBRARY   imf      HINTS "${MIC_SDK_DIR}/compiler/lib/mic")
      find_library(MIC_SVML_LIBRARY  svml     HINTS "${MIC_SDK_DIR}/compiler/lib/mic")
      find_library(MIC_INTLC_LIBRARY intlc    HINTS "${MIC_SDK_DIR}/compiler/lib/mic")
      mark_as_advanced(MIC_CC MIC_CXX MIC_IMF_LIBRARY MIC_SVML_LIBRARY MIC_INTLC_LIBRARY)

      set(MIC_LIBS ${MIC_IMF_LIBRARY} ${MIC_SVML_LIBRARY} ${MIC_INTLC_LIBRARY})
      set(MIC_CFLAGS "-O2 -vec")

      exec_program(${MIC_CXX} ARGS -V OUTPUT_VARIABLE _mic_icc_version_string RETURN_VALUE _mic_icc_ok)
      if(0 EQUAL _mic_icc_ok)
         string(REGEX MATCH "Version (Mainline)?[0-9. a-zA-Z]+" Vc_MIC_ICC_VERSION "${_mic_icc_version_string}")
         string(SUBSTRING "${Vc_MIC_ICC_VERSION}" 8 -1 Vc_MIC_ICC_VERSION)
         message(STATUS "MIC ICC Version: \"${Vc_MIC_ICC_VERSION}\"")

         if(MIC_CC AND MIC_CXX AND MIC_IMF_LIBRARY AND MIC_SVML_LIBRARY AND MIC_INTLC_LIBRARY)
            set(MIC_NATIVE_FOUND true)
         endif()
      else()
         message(STATUS "MIC ICC found, but not usable.")
      endif()
   endif()
endif(ENABLE_MIC)

if(MIC_NATIVE_FOUND OR MIC_OFFLOAD_FOUND)
   set(MIC_FOUND true)

   list(APPEND CMAKE_MIC_CXX_FLAGS "-fp-model source") # fix IEEE FP comliance

   set(Vc_MIC_CXX_FLAGS "")

   macro(mic_add_definitions)
      add_definitions(${ARGN})
      foreach(_def ${ARGN})
         set(_mic_cflags ${_mic_cflags} "${_def}")
      endforeach()
   endmacro()
   macro(mic_include_directories)
      foreach(_dir ${ARGN})
         set(_mic_cflags ${_mic_cflags} "-I${_dir}")
      endforeach()
      include_directories(${ARGN})
   endmacro()
   if(NOT DEFINED MIC_C_FLAGS)
      set(MIC_C_FLAGS)
   endif()
   if(NOT DEFINED MIC_CXX_FLAGS)
      set(MIC_CXX_FLAGS)
   endif()
else()
   message(STATUS "MIC SDK was not found!")
endif()

if(MIC_NATIVE_FOUND)
   macro(_mic_add_object _target _source _output)
      get_property(_deps SOURCE "${_source}" PROPERTY OBJECT_DEPENDS)
      get_filename_component(_abs "${_source}" ABSOLUTE)
      get_filename_component(_ext "${_source}" EXT)
      get_filename_component(_tmp "${_source}" NAME_WE)
      set(${_output} "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${_target}.dir/${_tmp}${_ext}.mic.o")
      set(_lang CXX)
      set(_compiler "${MIC_CXX}")
      if(_ext STREQUAL "c")
         set(_lang C)
         set(_compiler "${MIC_CC}")
      endif()

      string(TOUPPER "${CMAKE_BUILD_TYPE}" _tmp)
      string(STRIP "${CMAKE_MIC_${_lang}_FLAGS} ${CMAKE_${_lang}_FLAGS_${_tmp}} ${_mic_cflags} ${Vc_MIC_CXX_FLAGS}" _flags)
      string(REPLACE " " ";" _flags "${_flags} ${ARGN}")
      get_directory_property(_inc INCLUDE_DIRECTORIES)
      foreach(_i ${_inc})
         list(APPEND _flags "-I${_i}")
      endforeach()

      get_property(_launch_rule GLOBAL PROPERTY RULE_LAUNCH_COMPILE)
      string(REPLACE "\"" "" _launch_rule "${_launch_rule}")
      string(REPLACE " " ";" _launch_rule "${_launch_rule}")
      string(REPLACE "<TARGET_NAME>" "${_target}" _launch_rule "${_launch_rule}")
      string(REPLACE "<CMAKE_CURRENT_BINARY_DIR>" "${CMAKE_CURRENT_BINARY_DIR}" _launch_rule "${_launch_rule}")
      string(REPLACE "<OBJECT>" "${${_output}}" _launch_rule "${_launch_rule}")
      string(REPLACE "<SOURCE>" "${_abs}" _launch_rule "${_launch_rule}")
      string(REPLACE "<LANGUAGE>" "C++" _launch_rule "${_launch_rule}")
      add_custom_command(OUTPUT "${${_output}}"
         COMMAND ${_launch_rule} "${_compiler}" -mmic
         -DVc_IMPL=MIC
         ${_flags} -c -o "${${_output}}" "${_abs}"
         DEPENDS "${_abs}" ${_deps}
         IMPLICIT_DEPENDS ${_lang} "${_abs}"
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
         COMMENT "Compiling (MIC) ${${_output}}"
         VERBATIM
         )
   endmacro()
   macro(mic_set_link_libraries)
      set(_mic_lflags)
      foreach(_lib ${ARGN})
         get_filename_component(_lpath "${_lib}" PATH)
         get_filename_component(_lname "${_lib}" NAME)
         set(_mic_lflags ${_mic_lflags} "-L${_lpath}" "-l${_lname}")
      endforeach()
   endmacro()
   macro(mic_add_library _target)
      set(_state 0)
      if(BUILD_SHARED_LIBS)
         set(_type SHARED)
      else()
         set(_type STATIC)
      endif()
      set(_all ALL)
      set(_srcs)
      set(_cflags)
      set(_libs)
      foreach(_arg ${ARGN})
         if(_arg MATCHES "^(STATIC|SHARED|MODULE)$")
            set(_type ${_arg})
         elseif(_arg STREQUAL "EXCLUDE_FROM_ALL")
            set(_all)
         elseif(_arg STREQUAL "COMPILE_FLAGS" OR _arg STREQUAL "COMPILE_OPTIONS")
            set(_state 1)
         elseif(_arg STREQUAL "LINK_LIBRARIES")
            set(_state 2)
         elseif(_arg STREQUAL "SOURCES")
            set(_state 0)
         elseif(_state EQUAL 0) # SOURCES
            set(_srcs ${_srcs} "${_arg}")
         elseif(_state EQUAL 1) # COMPILE_FLAGS
            list(APPEND _cflags ${_arg})
         elseif(_state EQUAL 2) # LINK_LIBRARIES
            get_filename_component(_lpath "${_arg}" PATH)
            get_filename_component(_lname "${_arg}" NAME)
            set(_libs ${_libs} "-L${_lpath}" "-l${_lname}")
         endif()
      endforeach()
      set(_objects)
      set(_objectsStr)
      foreach(_src ${_srcs})
         _mic_add_object("${_target}" "${_src}" _obj ${_cflags})
         list(APPEND _objects "${_obj}")
         set(_objectsStr "${_objectsStr} \"${_obj}\"")
      endforeach()

      set(_outdir "${CMAKE_CURRENT_BINARY_DIR}/x86_64-k1om-linux")
      file(MAKE_DIRECTORY "${_outdir}")

      #TODO: handle STATIC/SHARED/MODULE differently
      set(_output "lib${_target}.a")
      set(_linkscript "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${_target}.dir/link.txt")
      set(_cleanscript "CMakeFiles/${_target}.dir/cmake_clean_target.cmake")
      file(WRITE "${_linkscript}"
         "${MIC_AR} cr ${_output} ${_objectsStr}
${MIC_RANLIB} ${_output}
")
      file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${_cleanscript}"
         "FILE(REMOVE_RECURSE \"${_output}\")
")
      add_custom_command(OUTPUT "${_outdir}/${_output}"
         COMMAND "${CMAKE_COMMAND}" -E cmake_link_script "${_linkscript}" --verbose=$(VERBOSE)
         DEPENDS ${_objects}
         WORKING_DIRECTORY "${_outdir}"
         COMMENT "Linking (MIC) ${_output}"
         VERBATIM
         )
      add_custom_target("${_target}" ${_all}
         DEPENDS "${_outdir}/${_output}"
         COMMENT ""
         SOURCES ${_srcs}
         )
      set_target_properties("${_target}" PROPERTIES
         OUTPUT_NAME "${_outdir}/${_output}"
         )
   endmacro()

   macro(mic_add_executable _target)
      set(_state 0)
      set(_all ALL)
      set(_srcs)
      set(_cflags)
      set(_libs)
      set(_libTargets)
      set(_dump_asm false)
      set(_exec_output_name "${_target}")
      set(_objects)
      set(_objectsStr)
      foreach(_arg ${ARGN})
         if(_arg STREQUAL "EXCLUDE_FROM_ALL")
            set(_all)
         elseif(_arg STREQUAL "COMPILE_FLAGS")
            set(_state 1)
         elseif(_arg STREQUAL "LINK_LIBRARIES")
            set(_state 2)
         elseif(_arg STREQUAL "OUTPUT_NAME")
            set(_state 3)
         elseif(_arg STREQUAL "SOURCES")
            set(_state 0)
         elseif(_arg STREQUAL "OBJECTS")
            set(_state 4)
         elseif(_arg STREQUAL "DUMP_ASM")
            set(_dump_asm true)
         elseif(_state EQUAL 0) # SOURCES
            set(_srcs ${_srcs} "${_arg}")
         elseif(_state EQUAL 1) # COMPILE_FLAGS
            set(_cflags ${_cflags} "${_arg}")
         elseif(_state EQUAL 2) # LINK_LIBRARIES
            if(TARGET ${_arg})
               get_target_property(_tmp "${_arg}" OUTPUT_NAME)
               if(_tmp)
                  set(_libs ${_libs} "${_tmp}")
                  set(_libTargets ${_libTargets} "${_tmp}" "${_arg}")
               else()
                  set(_libs ${_libs} "${_arg}")
                  if(EXISTS "${_arg}")
                     set(_libTargets ${_libTargets} "${_arg}")
                  endif()
               endif()
            else()
               set(_libs ${_libs} "${_arg}")
               if(EXISTS "${_arg}")
                  set(_libTargets ${_libTargets} "${_arg}")
               endif()
            endif()
         elseif(_state EQUAL 3) # OUTPUT_NAME
            set(_exec_output_name "${_arg}")
         elseif(_state EQUAL 4) # OBJECTS
            set(_objects ${_objects} "${_arg}")
            set(_objectsStr "${_objectsStr} \"${_arg}\"")
         endif()
      endforeach()
      foreach(_src ${_srcs})
         _mic_add_object("${_target}" "${_src}" _obj ${_cflags})
         set(_objects ${_objects} "${_obj}")
         set(_objectsStr "${_objectsStr} \"${_obj}\"")
      endforeach()

      set(_exec_output "${CMAKE_CURRENT_BINARY_DIR}/${_exec_output_name}")
      get_property(_launch_rule GLOBAL PROPERTY RULE_LAUNCH_LINK)
      string(REPLACE "\"" "" _launch_rule "${_launch_rule}")
      string(REPLACE " " ";" _launch_rule "${_launch_rule}")
      string(REPLACE "<TARGET_NAME>" "${_target}" _launch_rule "${_launch_rule}")
      string(REPLACE "<CMAKE_CURRENT_BINARY_DIR>" "${CMAKE_CURRENT_BINARY_DIR}" _launch_rule "${_launch_rule}")
      add_custom_command(OUTPUT "${_exec_output}"
         COMMAND ${_launch_rule} "${MIC_CXX}" -mmic
         "-L${MIC_SDK_DIR}/compiler/lib/mic/"
         ${_mic_lflags} ${_objects} -o "${_exec_output}"
         ${_libs}
         DEPENDS ${_objects} ${_libTargets}
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
         COMMENT "Linking (MIC) ${_exec_output}"
         VERBATIM
         )
      set(_dump_asm_output)
      if(_dump_asm)
         foreach(_src ${_srcs})
            get_filename_component(_abs "${_src}" ABSOLUTE)
            get_filename_component(_name "${_src}" NAME)
            add_custom_command(OUTPUT "${_name}.s"
               COMMAND "${MIC_CXX}" -mmic
               -DVc_IMPL=MIC ${_mic_cflags} ${_cflags} ${Vc_MIC_CXX_FLAGS}
               ${_abs}
               -S -fsource-asm -fno-verbose-asm -o "${_name}.x"
               COMMAND sh -c "grep -v ___tag_value '${_name}.x' | c++filt > '${_name}.s'"
               COMMAND rm "${_name}.x"
               DEPENDS ${_abs}
               WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
               COMMENT "Creating MIC Assembly ${_name}.s"
               VERBATIM
               )
            set(_dump_asm_output ${_dump_asm_output} "${CMAKE_CURRENT_BINARY_DIR}/${_name}.s")
         endforeach()
      endif()
      add_custom_target("${_target}" ${_all}
         DEPENDS "${_exec_output}" ${_dump_asm_output}
         COMMENT ""
         SOURCES ${_srcs}
         )
      set_target_properties("${_target}" PROPERTIES OUTPUT_NAME "${_exec_output_name}")
   endmacro()
endif()

if(MIC_OFFLOAD_FOUND)
   macro(mic_offload _target)
      set(_mic_debug)
      if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
         set(_mic_debug "-g")
      endif()
      add_target_property(${_target} COMPILE_FLAGS "-offload-build -DCAN_OFFLOAD ${_mic_debug}")
      set(_offload_ldflags "${_mic_debug}")
      set(_libTargets)
      foreach(_lib ${ARGN})
         get_target_property(_tmp "${_lib}" OUTPUT_NAME)
         if(_tmp)
            set(_offload_ldflags "${_offload_ldflags} ${_tmp}")
            set(_libTargets ${_libTargets} "${_arg}")
         else()
            get_filename_component(_lpath "${_arg}" PATH)
            get_filename_component(_lname "${_arg}" NAME)
            set(_offload_ldflags "${_offload_ldflags} -L${_lpath} -l${_lname}")
         endif()
      endforeach()
      add_target_property(${_target} LINK_FLAGS "-offload-build -offload-ldopts=\"${_offload_ldflags}\" ${_mic_debug}")
      if(_libTargets)
         add_dependencies(${_target} ${_libTargets})
      endif()
   endmacro()
endif()
