cmake_minimum_required(VERSION 3.0)
cmake_policy(SET CMP0054 NEW)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
   message(FATAL_ERROR "You don't want to configure in the source directory!")
endif()

project(Vc)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(CMAKE_REQUIRED_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}")
include (AddTargetProperty)
include (OptimizeForArchitecture)
OptimizeForArchitecture()

set(Vc_X86 FALSE)
set(Vc_ARM FALSE)
if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "(x86|AMD64|amd64)")
   set(Vc_X86 TRUE)
elseif("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "(arm|aarch32|aarch64)")
   set(Vc_ARM TRUE)
else()
   message(WARNING "No optimized implementation of the Vc types available for ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# set up ccache#{{{
##################################################################
option(USE_CCACHE "If enabled, ccache will be used (if it exists on the system) to speed up recompiles." OFF)
if(USE_CCACHE)
   find_program(CCACHE_COMMAND ccache)
   if(CCACHE_COMMAND)
      mark_as_advanced(CCACHE_COMMAND)
      set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_COMMAND}")
   endif()
endif()#}}}

if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS "8.0")
   message(FATAL_ERROR "This std::experimental::simd implementation requires GCC 8 or higher.")
endif()#}}}

# Set default CMAKE_BUILD_TYPE to Release#{{{
##################################################################
if(NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebug RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

if(CMAKE_BUILD_TYPE STREQUAL "" AND NOT CMAKE_CXX_FLAGS MATCHES "-O[123]")
   message(STATUS "WARNING! It seems you are compiling without optimization. Please set CMAKE_BUILD_TYPE.")
endif(CMAKE_BUILD_TYPE STREQUAL "" AND NOT CMAKE_CXX_FLAGS MATCHES "-O[123]")#}}}

# Set default C(XX)FLAGS and includes#{{{
##################################################################
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -W -Wall -Wextra -Wpedantic -Wcast-align -Wshadow -Wundef -Wold-style-cast -fabi-version=0 -fabi-compat-version=0 -ffp-contract=fast -ftemplate-depth=512")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
#}}}

# Look for the Intel SDE if we're running on x86 and Linux#{{{
##################################################################
if(Vc_X86 AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
   macro(Vc_find_sde)
      find_program(INTEL_SDE sde64 ${ARGN} DOC "Path to the Intel Software Development Emulator")
      if(INTEL_SDE)
         execute_process(COMMAND ${INTEL_SDE} -help RESULT_VARIABLE _sde_usable OUTPUT_QUIET ERROR_QUIET)
         if(NOT _sde_usable LESS 2)
            unset(INTEL_SDE CACHE)
            find_program(INTEL_SDE sde ${ARGN} NO_DEFAULT_PATH DOC "Path to the Intel Software Development Emulator")
            if(INTEL_SDE)
               execute_process(COMMAND ${INTEL_SDE} -help RESULT_VARIABLE _sde_usable OUTPUT_QUIET ERROR_QUIET)
               if(NOT _sde_usable LESS 2)
                  set(INTEL_SDE FALSE)
               endif()
            endif()
         endif()
      endif()
   endmacro()

   file(GLOB SDE_DIRS /opt/intel/*)
   list(SORT SDE_DIRS)
   list(REVERSE SDE_DIRS)
   Vc_find_sde(${SDE_DIRS} "$ENV{HOME}/.cache/sde/")

   # if running a dashboard cycle and no installed SDE was found, download and use it automatically
   if(NOT "$ENV{DASHBOARD_TEST_FROM_CTEST}" STREQUAL "" AND NOT INTEL_SDE)
      find_program(_bin_sh sh)
      find_program(_wget wget)
      if(_bin_sh AND _wget AND DEFINED ENV{HOME})
         option(DOWNLOAD_INTEL_SDE "Download the Intel Software Development Emulator to execute tests on different ISAs" ON)
         if(DOWNLOAD_INTEL_SDE)
            message(STATUS "Downloading Intel SDE:")
            file(MAKE_DIRECTORY "$ENV{HOME}/.cache/sde")
            execute_process(COMMAND ${_bin_sh} "${CMAKE_SOURCE_DIR}/scripts/download_intel_sde.sh"
               WORKING_DIRECTORY "$ENV{HOME}/.cache/sde"
               TIMEOUT 20
               RESULT_VARIABLE res)
            if(${res} EQUAL 0)
               Vc_find_sde("$ENV{HOME}/.cache/sde/" NO_DEFAULT_PATH)
            endif()
         endif()
      endif()
      mark_as_advanced(_bin_sh _wget)
   endif()

   if(INTEL_SDE)
      execute_process(COMMAND ${INTEL_SDE} -version OUTPUT_VARIABLE sde_version ERROR_QUIET)
      string(REGEX MATCH "Version: +[0-9.]+" sde_version "${sde_version}")
      message(STATUS "Intel SDE ${sde_version}")
   else()
      message(STATUS "Intel SDE not found.")
   endif()
endif()#}}}

# Set up make targets#{{{
##################################################################
add_custom_target(other VERBATIM)
add_custom_target(NoSIMD COMMENT "build non-SIMD code" VERBATIM)
if(Vc_X86)
   add_custom_target(SSE COMMENT "build SSE code" VERBATIM)
   add_custom_target(SSE1 COMMENT "build SSE1 code" VERBATIM)
   add_custom_target(SSE2 COMMENT "build SSE2 code" VERBATIM)
   add_custom_target(SSSE3 COMMENT "build SSSE3 code" VERBATIM)
   add_custom_target(SSE4_2 COMMENT "build SSE4.2 code" VERBATIM)
   add_dependencies(SSE SSE1 SSE2 SSSE3 SSE4_2)
   add_custom_target(AVX COMMENT "build AVX code" VERBATIM)
   add_custom_target(AVX2 COMMENT "build AVX2 code" VERBATIM)
   add_custom_target(KNL COMMENT "build KNL code" VERBATIM)
   add_custom_target(AVX512 COMMENT "build AVX512 code" VERBATIM)
elseif(Vc_ARM)
   add_custom_target(NEON COMMENT "build NEON code" VERBATIM)
endif()#}}}

# Unit tests#{{{
##################################################################
#Release# option(BUILD_TESTING "Build the testing tree." OFF)
include (CTest)
configure_file(${CMAKE_SOURCE_DIR}/scripts/CTestCustom.cmake ${CMAKE_BINARY_DIR}/CTestCustom.cmake COPYONLY)
if(BUILD_TESTING)
   if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/virtest/vir/test.h")
      # first try to resolve the issue automatically
      message(STATUS "virtest missing: calling 'git submodule update --init'")
      execute_process(
         COMMAND git submodule update --init
         WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
         )
   endif()
   if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/virtest/vir/test.h")
      add_custom_target(build_tests ALL VERBATIM)
      add_subdirectory(tests)
   else()
      message(FATAL_ERROR "Unit tests enabled but virtest is missing. Call 'git submodule update' to clone the required unit test framework.")
   endif()
endif(BUILD_TESTING)#}}}

# Hack for caching `make help`#{{{
##################################################################
find_program(BIN_CAT cat)
mark_as_advanced(BIN_CAT)
if(BIN_CAT)
   file(REMOVE ${CMAKE_BINARY_DIR}/help.txt)
   add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/help.txt
      COMMAND ${CMAKE_MAKE_PROGRAM} help > ${CMAKE_BINARY_DIR}/help.txt
      VERBATIM
      )
   add_custom_target(cached_help
      ${BIN_CAT} ${CMAKE_BINARY_DIR}/help.txt
      DEPENDS ${CMAKE_BINARY_DIR}/help.txt
      VERBATIM
      )
endif()#}}}

# vim: foldmethod=marker commentstring=#%s
