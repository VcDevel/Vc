# Locate the Vc template library. Vc can be found at http://gitorious.org/Vc/
#
# Copyright 2009, 2010   Matthias Kretz <kretz@kde.org>
#
# This module defines
# VC_FOUND
# VC_INCLUDE_DIR
# VC_LIBRARIES
#
# Additionally it will set some defines to match the capabilities of your
# compiler/assembler

include (MacroEnsureVersion)

if(NOT DEFINED VC_INSTALL_PREFIX OR NOT VC_INSTALL_PREFIX)
   # we have to search for Vc ourself
   find_path(VC_INSTALL_PREFIX include/Vc/Vc HINTS $ENV{HOME} $ENV{HOME}/local)
   if(VC_INSTALL_PREFIX)
      set(VC_INCLUDE_DIR "${VC_INSTALL_PREFIX}/include" CACHE PATH "The include directory for Vc")
      find_library(VC_LIBRARIES Vc HINTS "${VC_INSTALL_PREFIX}/lib")
   endif(VC_INSTALL_PREFIX)
endif(NOT DEFINED VC_INSTALL_PREFIX OR NOT VC_INSTALL_PREFIX)

if(VC_INSTALL_PREFIX AND VC_INCLUDE_DIR AND VC_LIBRARIES)
   message(STATUS "Vc template library found at ${VC_INSTALL_PREFIX}.")
   set(VC_FOUND true)

   if(CMAKE_COMPILER_IS_GNUCXX)
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
   elseif(MSVC)
      # MSVC does not support inline assembly on 64 bit! :(
      # searching the help for xgetbv doesn't turn up anything. So just fall back to not supporting AVX on Windows :(
      add_definitions(-DVC_NO_XGETBV)
   endif(CMAKE_COMPILER_IS_GNUCXX)
else(VC_INSTALL_PREFIX AND VC_INCLUDE_DIR AND VC_LIBRARIES)
   set(VC_FOUND false)
   if(Vc_FIND_REQUIRED)
      message(FATAL_ERROR "Could not find required Vc template library.")
   else(Vc_FIND_REQUIRED)
      message(STATUS "Could not find Vc template library.")
   endif(Vc_FIND_REQUIRED)
endif(VC_INSTALL_PREFIX AND VC_INCLUDE_DIR AND VC_LIBRARIES)

mark_as_advanced(VC_INCLUDE_DIR VC_LIBRARIES)
