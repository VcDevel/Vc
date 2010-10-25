# Locate the Vc template library. Vc can be found at http://gitorious.org/Vc/
#
# Copyright 2009, 2010   Matthias Kretz <kretz@kde.org>
#
# This module defines
# VC_FOUND
# VC_INCLUDE_DIR
# VC_LIBRARIES

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
else(VC_INSTALL_PREFIX AND VC_INCLUDE_DIR AND VC_LIBRARIES)
   set(VC_FOUND false)
   if(Vc_FIND_REQUIRED)
      message(FATAL_ERROR "Could not find required Vc template library.")
   else(Vc_FIND_REQUIRED)
      message(STATUS "Could not find Vc template library.")
   endif(Vc_FIND_REQUIRED)
endif(VC_INSTALL_PREFIX AND VC_INCLUDE_DIR AND VC_LIBRARIES)

mark_as_advanced(VC_INCLUDE_DIR VC_LIBRARIES)
