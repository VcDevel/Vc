cmake_minimum_required(VERSION 3.0)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
   message(FATAL_ERROR "You don't want to configure in the source directory!")
endif()

project(Vc)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(disabled_targets)

include (VcMacros)
include (AddTargetProperty)
include (OptimizeForArchitecture)

vc_determine_compiler()
vc_determine_target_isa()
find_package(MIC)

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

# Check for C++14 support in the compiler#{{{
##################################################################
if(NOT Vc_COMPILER_IS_MSVC) # MSVC doesn't provide a switch to turn C++11 on/off AFAIK
   AddCompilerFlag("-std=c++14" CXX_RESULT _ok MIC_CXX_RESULT _mic_ok CXX_FLAGS CMAKE_CXX_FLAGS MIC_CXX_FLAGS Vc_MIC_CXX_FLAGS)
   if(MIC_NATIVE_FOUND AND NOT _mic_ok)
      AddCompilerFlag("-std=c++1y" MIC_CXX_RESULT _mic_ok MIC_CXX_FLAGS Vc_MIC_CXX_FLAGS)
      if(NOT _mic_ok)
         message(STATUS "Vc 2.x requires C++14. The MIC native compiler does not support C++14. MIC build disabled.")
         set(ENABLE_MIC FALSE)
         set(MIC_NATIVE_FOUND FALSE)
         set(MIC_FOUND FALSE)
      endif()
   endif()
   if(NOT _ok)
      AddCompilerFlag("-std=c++1y" CXX_RESULT _ok CXX_FLAGS CMAKE_CXX_FLAGS)
      if(NOT _ok)
         message(FATAL_ERROR "Vc 2.x requires C++14. It seems this is not available. If this was incorrectly determined please open an issue on https://github.com/VcDevel/Vc")
      endif()
   endif()
elseif(Vc_MSVC_VERSION LESS 180021114)
   message(FATAL_ERROR "Vc 2.x requires C++14 support.")
endif()#}}}

# Compiler workarounds that are not necessary for users#{{{
##################################################################
if(Vc_COMPILER_IS_GCC)
   if(Vc_GCC_VERSION VERSION_GREATER "5.99")
      AddCompilerFlag(-Wno-ignored-attributes)
   endif()
elseif(Vc_COMPILER_IS_CLANG)
   if(Vc_CLANG_VERSION VERSION_LESS 4.0.50)
      # Clang ICEs with AVX512 on the loadstore test
      list(APPEND disabled_targets
         loadstore_avx512_vectorbuiltin_int_short_uint_ushort
         loadstore_avx512_vectorbuiltin_ldouble_float_double_schar_uchar
         datapar_avx512_vectorbuiltin_int_short_uint_ushort
         datapar_avx512_vectorbuiltin_ldouble_float_double_schar_uchar
         )
   endif()
elseif(Vc_COMPILER_IS_MSVC)
   # Disable warning "C++ exception specification ignored except to indicate a function is not __declspec(nothrow)"
   # MSVC emits the warning for the _UnitTest_Compare desctructor which needs the throw declaration so that it doesn't std::terminate
   AddCompilerFlag("/wd4290")
elseif(Vc_COMPILER_IS_INTEL)
   # per default icc is not IEEE compliant, but we need that for verification
   AddCompilerFlag("-fp-model source")
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
vc_set_preferred_compiler_flags(WARNING_FLAGS BUILDTYPE_FLAGS)
add_definitions(${Vc_DEFINITIONS})
add_compile_options(${Vc_COMPILE_FLAGS})
AddCompilerFlag(-ftemplate-depth=512 CXX_FLAGS CMAKE_CXX_FLAGS MIC_CXX_FLAGS CMAKE_MIC_CXX_FLAGS)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})#}}}

# Option for building against libc++ with Clang#{{{
##################################################################
if(UNIX AND Vc_COMPILER_IS_CLANG)
   # On UNIX (Linux) the standard library used by default typically is libstdc++ (GCC).
   # To get the full clang deal we rather want to build against libc++. This requires
   # additionally the libc++abi and libsupc++ libraries in all linker invokations.
   option(USE_LIBC++ "Use libc++ instead of the system default C++ standard library." ON)
   if(USE_LIBC++)
      AddCompilerFlag(-stdlib=libc++ CXX_FLAGS CMAKE_CXX_FLAGS CXX_RESULT _use_libcxx)
      if(_use_libcxx)
         find_library(LIBC++ABI c++abi)
         mark_as_advanced(LIBC++ABI)
         if(LIBC++ABI)
            set(CMAKE_REQUIRED_LIBRARIES "${LIBC++ABI};supc++")
            CHECK_CXX_SOURCE_COMPILES("#include <stdexcept>
            #include <iostream>
            void foo() {
              std::cout << 'h' << std::flush << std::endl;
              throw std::exception();
            }
            int main() {
              try { foo(); }
              catch (int) { return 0; }
              return 1;
            }" libcxx_compiles)
            unset(CMAKE_REQUIRED_LIBRARIES)
            if(libcxx_compiles)
               link_libraries(${LIBC++ABI} supc++)
            endif()
         endif()
      endif()
   endif()
endif()#}}}

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
endif()#}}}

# Set up make targets#{{{
##################################################################
add_custom_target(other VERBATIM)
add_custom_target(NoSIMD COMMENT "build non-SIMD code" VERBATIM)
if(Vc_X86)
   add_custom_target(SSE COMMENT "build SSE code" VERBATIM)
   add_custom_target(SSE1 COMMENT "build SSE1 code" VERBATIM)
   add_custom_target(SSE2 COMMENT "build SSE2 code" VERBATIM)
   add_custom_target(SSE4_2 COMMENT "build SSE4.2 code" VERBATIM)
   add_dependencies(SSE SSE1 SSE2 SSE4_2)
   add_custom_target(AVX COMMENT "build AVX code" VERBATIM)
   add_custom_target(AVX2 COMMENT "build AVX2 code" VERBATIM)
   add_custom_target(MIC COMMENT "build MIC code" VERBATIM)
   add_custom_target(KNL COMMENT "build KNL code" VERBATIM)
   add_custom_target(AVX512 COMMENT "build AVX512 code" VERBATIM)
elseif(Vc_ARM)
   add_custom_target(NEON COMMENT "build NEON code" VERBATIM)
endif()#}}}

if(FALSE) # disabled for now
# Build libVc#{{{
##################################################################
set(libvc_compile_flags "-DVc_COMPILE_LIB")
set(libvc_mic_compile_flags "-DVc_COMPILE_LIB")
AddCompilerFlag("-fPIC" CXX_FLAGS libvc_compile_flags MIC_CXX_FLAGS libvc_mic_compile_flags)
set(libVc_srcs src/const.cpp)
if(Vc_X86)
   list(APPEND libVc_srcs src/cpuid.cpp src/support_x86.cpp)
   vc_compile_for_all_isa(libVc_srcs src/trigonometric.cpp ONLY SSE2 SSE3 SSSE3 SSE4_1 AVX SSE+XOP+FMA4 AVX+XOP+FMA4 AVX+XOP+FMA AVX+FMA AVX2)
   vc_compile_for_all_isa(libVc_srcs src/sse_sorthelper.cpp ONLY SSE2 SSE4_1 AVX AVX2)
   vc_compile_for_all_isa(libVc_srcs src/avx_sorthelper.cpp ONLY AVX AVX2)
elseif(Vc_ARM)
   list(APPEND libVc_srcs src/support_arm.cpp)
   vc_compile_for_all_isa(libVc_srcs src/trigonometric.cpp ONLY NEON)
else()
   message(FATAL_ERROR "Unsupported target architecture '${CMAKE_SYSTEM_PROCESSOR}'. No support_???.cpp file exists for this architecture.")
endif()
add_library(Vc STATIC ${libVc_srcs})
set_property(TARGET Vc APPEND PROPERTY COMPILE_OPTIONS ${libvc_compile_flags})
add_target_property(Vc LABELS "other")
add_dependencies(other Vc)
if(XCODE)
   # TODO: document what this does and why it has no counterpart in the non-XCODE logic
   set_target_properties(Vc PROPERTIES XCODE_ATTRIBUTE_GCC_INLINES_ARE_PRIVATE_EXTERN "NO")
   set_target_properties(Vc PROPERTIES XCODE_ATTRIBUTE_GCC_SYMBOLS_PRIVATE_EXTERN "YES")
   set_target_properties(Vc PROPERTIES XCODE_ATTRIBUTE_CLANG_CXX_LANGUAGE_STANDARD "c++14")
   set_target_properties(Vc PROPERTIES XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY "libc++")
endif()#}}}

# Build libVc_MIC#{{{
##################################################################
if(MIC_NATIVE_FOUND)
   mic_add_library(Vc_MIC STATIC src/mic_const.cpp src/cpuid.cpp src/support_x86.cpp src/mic_sorthelper.cpp
      COMPILE_FLAGS ${libvc_mic_compile_flags})
   add_target_property(Vc_MIC LABELS "MIC")
   add_dependencies(MIC Vc_MIC)
endif()#}}}

# Install libVc(_MIC)#{{{
##################################################################
install(TARGETS Vc DESTINATION lib${LIB_SUFFIX})
if(MIC_NATIVE_FOUND)
   get_target_property(outputName Vc_MIC OUTPUT_NAME)
   install(FILES ${outputName} DESTINATION lib${LIB_SUFFIX})
endif()#}}}
endif()

# Install headers#{{{
##################################################################
install(DIRECTORY Vc/ DESTINATION include/Vc)
# }}}

# Generate & install cmake config#{{{
##################################################################
# read version parts from version.h to be put into VcConfig.cmake
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/Vc/version.h _version_lines REGEX "^#define Vc_VERSION_STRING ")
string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)" _version_matches "${_version_lines}")
set(Vc_VERSION_MAJOR ${CMAKE_MATCH_1})
set(Vc_VERSION_MINOR ${CMAKE_MATCH_2})
set(Vc_VERSION_PATCH ${CMAKE_MATCH_3})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/VcConfig.cmake.in
   ${CMAKE_CURRENT_BINARY_DIR}/cmake/VcConfig.cmake @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/VcConfigVersion.cmake.in
   ${CMAKE_CURRENT_BINARY_DIR}/cmake/VcConfigVersion.cmake @ONLY)

install(FILES
   cmake/UserWarning.cmake
   cmake/VcMacros.cmake
   cmake/AddCompilerFlag.cmake
   cmake/CheckCCompilerFlag.cmake
   cmake/CheckCXXCompilerFlag.cmake
   cmake/CheckMicCCompilerFlag.cmake
   cmake/CheckMicCXXCompilerFlag.cmake
   cmake/FindMIC.cmake
   cmake/OptimizeForArchitecture.cmake
   cmake/FindVc.cmake
   ${CMAKE_CURRENT_BINARY_DIR}/cmake/VcConfig.cmake
   ${CMAKE_CURRENT_BINARY_DIR}/cmake/VcConfigVersion.cmake
   DESTINATION lib${LIB_SUFFIX}/cmake/Vc
   )#}}}

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

# Examples#{{{
##################################################################
set(BUILD_EXAMPLES FALSE CACHE BOOL "Build examples.")
if(BUILD_EXAMPLES)
   message(STATUS "Examples not ported to Vc 2. Disabled.")
   #add_subdirectory(examples)
endif(BUILD_EXAMPLES)#}}}

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
