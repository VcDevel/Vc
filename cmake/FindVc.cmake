# Locate the Vc template library. Vc can be found at http://gitorious.org/Vc/
#
# Copyright 2009-2012   Matthias Kretz <kretz@kde.org>
#
# This file is meant to be copied into projects that want to use Vc. It will
# search for VcConfig.cmake, which ships with Vc and will provide up-to-date
# buildsystem changes. Thus there should not be any need to update FindVc.cmake
# again after you integrated it into your project.
#
# This module defines the following variables:
# VC_FOUND
# VC_INCLUDE_DIR
# VC_LIBRARIES
# VC_DEFINITIONS
# VC_VERSION_MAJOR
# VC_VERSION_MINOR
# VC_VERSION_PATCH
# VC_VERSION
# VC_VERSION_STRING
# VC_INSTALL_DIR
# VC_LIB_DIR
# VC_CMAKE_MODULES_DIR
#
# The following two variables are set according to the compiler used. Feel free
# to use them to skip whole compilation units.
# VC_SSE_INTRINSICS_BROKEN
# VC_AVX_INTRINSICS_BROKEN

find_package(Vc ${Vc_FIND_VERSION} QUIET NO_MODULE PATHS $ENV{HOME} /opt/Vc)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Vc CONFIG_MODE)
