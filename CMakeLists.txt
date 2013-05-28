cmake_minimum_required(VERSION 2.8.3)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BUILD_DIR)
   message(ERROR "You don't want to configure in the source directory!")
endif()

project(Vc)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(ROOT_RELEASE FALSE CACHE BOOL "Set up for creating a Vc copy inside ROOT/AliRoot.")
mark_as_advanced(ROOT_RELEASE)

set(disabled_targets)

include (VcMacros)
include (AddTargetProperty)
include (OptimizeForArchitecture)

vc_determine_compiler()
find_package(MIC)

if(ROOT_RELEASE)
   file(READ "${CMAKE_INSTALL_PREFIX}/Module.mk" ROOT_MODULE_MK)
   if(NOT ROOT_MODULE_MK MATCHES "\nMODNAME *:= *vc *\n")
      message(FATAL_ERROR "CMAKE_INSTALL_PREFIX is incorrect. It must point to the Vc subdirectory inside ROOT/AliRoot")
   endif()
else()

   # TODO: skip the C++11 check for C
   # TODO: check that 'decltype' compiles
   # TODO: check that 'constexpr' compiles
   if(NOT Vc_COMPILER_IS_MSVC) # MSVC doesn't provide a switch to turn C++11 on/off AFAIK
      if(Vc_COMPILER_IS_GCC AND Vc_GCC_VERSION VERSION_LESS 4.6.0)
         message(FATAL_ERROR "Vc 1.x requires C++11 support. This requires at least GCC 4.6.")
      endif()

      AddCompilerFlag("-std=c++11" _ok)
      if(NOT _ok)
         AddCompilerFlag("-std=c++0x" _ok)
         if(NOT _ok)
            message(FATAL_ERROR "Vc 1.x requires C++11 support. It seems this is not available. If this was incorrectly determined please notify vc-devel@compeng.uni-frankfurt.de")
         endif()
      endif()
   endif()

   if(Vc_COMPILER_IS_GCC)
      if(Vc_GCC_VERSION STREQUAL "4.6.0")
         UserWarning("GCC 4.6.0 is broken. The following tests are therefore disabled:
         gather_avx, gather_sse, gather_VC_USE_SET_GATHERS_avx, gather_VC_USE_SET_GATHERS_sse,
         gather_sse_LOOP, scatter_avx, and scatter_sse")
         list(APPEND disabled_targets
            gather_avx
            gather_sse
            gather_VC_USE_SET_GATHERS_avx
            gather_VC_USE_SET_GATHERS_sse
            scatter_avx
            scatter_sse
            )
      elseif(Vc_GCC_VERSION STREQUAL "4.8.0")
         UserWarning("GCC 4.8.0 is broken. The following tests are therefore disabled:
         memory_avx2, memory_avx, memory_sse, memory_scalar")
         list(APPEND disabled_targets
            memory_avx2
            memory_avx
            memory_sse
            memory_scalar
            )
      endif()
   elseif(Vc_COMPILER_IS_INTEL)
      UserWarning("ICC does not support initializer lists (C++11). The following tests are therefore disabled:
      stlcontainer_avx stlcontainer_scalar stlcontainer_sse.")
      list(APPEND disabled_targets
         stlcontainer_avx
         stlcontainer_scalar
         stlcontainer_sse
         )
   elseif(Vc_COMPILER_IS_MSVC)
      if(MSVC_VERSION LESS 1700)
         # MSVC before 2012 has a broken std::vector::resize implementation. STL + Vc code will probably not compile.
         # UserWarning in VcMacros.cmake
         list(APPEND disabled_targets
            stlcontainer_sse
            stlcontainer_avx
            )
      endif()
   endif()
endif()

if(NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebug RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

vc_set_preferred_compiler_flags(WARNING_FLAGS BUILDTYPE_FLAGS)
add_definitions("${Vc_DEFINITIONS}")

if(Vc_COMPILER_IS_INTEL)
   # per default icc is not IEEE compliant, but we need that for verification
   set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -fp-model source")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fp-model source")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "" AND NOT CMAKE_CXX_FLAGS MATCHES "-O[123]")
   message(STATUS "WARNING! It seems you are compiling without optimization. Please set CMAKE_BUILD_TYPE.")
endif(CMAKE_BUILD_TYPE STREQUAL "" AND NOT CMAKE_CXX_FLAGS MATCHES "-O[123]")

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(NOT ROOT_RELEASE)
   add_custom_target(other VERBATIM)
   add_custom_target(Scalar COMMENT "build Scalar code" VERBATIM)
   add_custom_target(SSE COMMENT "build SSE code" VERBATIM)
   add_custom_target(AVX COMMENT "build AVX code" VERBATIM)
   add_custom_target(AVX2 COMMENT "build AVX2 code" VERBATIM)
   if(MIC_FOUND)
      add_custom_target(MIC COMMENT "build MIC code" VERBATIM)
      mic_add_library(Vc_MIC STATIC src/mic_const.cpp src/cpuid.cpp src/support.cpp)
      add_dependencies(MIC Vc_MIC)
   endif()

   set(libvc_compile_flags "-DVC_COMPILE_LIB")
   AddCompilerFlag("-fPIC" CXX_FLAGS libvc_compile_flags)
   vc_compile_for_all_implementations(_objs src/trigonometric.cpp FLAGS ${libvc_compile_flags}
      ONLY SSE2 SSE3 SSSE3 SSE4_1 AVX SSE+XOP+FMA4 AVX+XOP+FMA4 AVX+XOP+FMA AVX+FMA AVX2)
   set(_srcs src/const.cpp src/cpuid.cpp src/support.cpp ${_objs})
   if(USE_AVX)
      list(APPEND _srcs src/avx_sorthelper.cpp)
   else()
      if(NOT Vc_AVX_INTRINSICS_BROKEN)
         # we'd still like to have avx_sorthelper.cpp built in, but that requires compilation with -mavx (or a comparable flag)
         foreach(_flag "-xAVX" "-mavx" "-arch:AVX")
            check_cxx_compiler_flag("${_flag}" check_cxx_compiler_flag_${_flag})
            if(check_cxx_compiler_flag_${_flag})
               if(_flag STREQUAL "-xAVX")
                  set(_flag "${_flag} -diag-disable 10121") # disable the warning "overriding -xSSE4.2 with -xAVX"
               endif()
               list(APPEND _srcs src/avx_sorthelper.cpp)
               set_source_files_properties(src/avx_sorthelper.cpp PROPERTIES COMPILE_FLAGS "${_flag}")
               break()
            endif()
         endforeach()
      endif()
      if(NOT Vc_AVX2_INTRINSICS_BROKEN)
         # we'd still like to have avx_sorthelper.cpp built in, but that requires compilation with -mavx (or a comparable flag)
         foreach(_flag "-xCORE-AVX2" "-mavx2" "-arch:AVX2")
            check_cxx_compiler_flag("${_flag}" check_cxx_compiler_flag_${_flag})
            if(check_cxx_compiler_flag_${_flag})
               if(_flag STREQUAL "-xCORE-AVX2")
                  set(_flag "${_flag} -diag-disable 10121") # disable the warning "overriding -xSSE4.2 with -xAVX"
               endif()
               list(APPEND _srcs src/avx2_sorthelper.cpp)
               set_source_files_properties(src/avx2_sorthelper.cpp PROPERTIES COMPILE_FLAGS "${_flag}")
               break()
            endif()
         endforeach()
      endif()
   endif()
   add_library(Vc STATIC ${_srcs})
   add_target_property(Vc COMPILE_FLAGS ${libvc_compile_flags})
   add_target_property(Vc LABELS "other")
   add_dependencies(other Vc)

   install(TARGETS Vc DESTINATION lib${LIB_SUFFIX})
   install(DIRECTORY include/Vc/ DESTINATION include/Vc)
else()
   # libVc should be compiled in the ROOT/AliRoot tree, so we need to install the sources
   #
   # Sadly there are messed up systems where putting include/Vc in the include paths will
   # break the standard library (e.g. MacOS X Lion with case insensitive filesystem).
   # Thus, we modify the includes such that include/Vc never needs to be in the path.
   file(GLOB _srcs RELATIVE "${CMAKE_SOURCE_DIR}" src/*.cpp tests/*.cpp tests/*.h)
   foreach(_src ${_srcs})
      install(CODE "
      message(\"-- Rewriting and Installing: ${CMAKE_INSTALL_PREFIX}/${_src}\")
         file(READ \"${CMAKE_SOURCE_DIR}/${_src}\" data)
         string(REGEX REPLACE \"#include (.)(common|avx|sse|scalar)\"
            \"#include \\\\1Vc/\\\\2\" data \"\${data}\")
            file(WRITE \"${CMAKE_INSTALL_PREFIX}/${_src}\" \"\${data}\")
      ")
   endforeach()
   install(DIRECTORY ${CMAKE_BINARY_DIR}/src/ DESTINATION src)

   install(DIRECTORY include/Vc/ DESTINATION include/Vc
      PATTERN "global.h" EXCLUDE
      PATTERN "*.swp" EXCLUDE
      PATTERN "*.bak" EXCLUDE
      )
   install(CODE "
   message(\"-- Rewriting and Installing: ${CMAKE_INSTALL_PREFIX}/include/Vc/global.h\")
      file(READ \"${CMAKE_SOURCE_DIR}/include/Vc/global.h\" data)
      string(REGEX REPLACE \"#define Vc__SYMBOL_VERSION ([^\\n]*)\"
         \"#define Vc__SYMBOL_VERSION ROOT\\\\1\" data \"\${data}\")
         file(WRITE \"${CMAKE_INSTALL_PREFIX}/include/Vc/global.h\" \"\${data}\")
   ")
endif()

# Install all implementation headers
install(DIRECTORY scalar sse avx common DESTINATION include/Vc FILES_MATCHING REGEX "/*.(h|tcc|def)$")

# read version parts from version.h to be put into VcConfig.cmake
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/include/Vc/version.h _version_lines REGEX "^#define VC_VERSION_STRING ")
string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)" _version_matches "${_version_lines}")
set(Vc_VERSION_MAJOR ${CMAKE_MATCH_1})
set(Vc_VERSION_MINOR ${CMAKE_MATCH_2})
set(Vc_VERSION_PATCH ${CMAKE_MATCH_3})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/VcConfig.cmake.in
   ${CMAKE_CURRENT_BINARY_DIR}/cmake/VcConfig.cmake @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/VcConfigVersion.cmake.in
   ${CMAKE_CURRENT_BINARY_DIR}/cmake/VcConfigVersion.cmake @ONLY)

set(cmake_install_files
   cmake/UserWarning.cmake
   cmake/VcMacros.cmake
   cmake/AddCompilerFlag.cmake
   cmake/CheckCCompilerFlag.cmake
   cmake/CheckCXXCompilerFlag.cmake
   )
if(ROOT_RELEASE)
   execute_process(
      COMMAND sed "s, \"auto\" CACHE, \"none\" CACHE,"
      INPUT_FILE ${CMAKE_SOURCE_DIR}/cmake/OptimizeForArchitecture.cmake
      OUTPUT_FILE ${CMAKE_BINARY_DIR}/cmake/OptimizeForArchitecture.cmake
      )
   install(FILES
      ${cmake_install_files}
      cmake/AddTargetProperty.cmake
      ${CMAKE_BINARY_DIR}/cmake/OptimizeForArchitecture.cmake
      DESTINATION cmake
      )
   install(DIRECTORY examples/ DESTINATION examples)
   install(DIRECTORY ${CMAKE_BINARY_DIR}/tests/ DESTINATION tests)
   install(FILES tests/CMakeLists.txt tests/download.cmake DESTINATION tests)
else()
   install(FILES
      ${cmake_install_files}
      ${CMAKE_CURRENT_BINARY_DIR}/cmake/VcConfig.cmake
      ${CMAKE_CURRENT_BINARY_DIR}/cmake/VcConfigVersion.cmake
      cmake/OptimizeForArchitecture.cmake
      cmake/FindVc.cmake
      DESTINATION lib/cmake/Vc
      )
endif()

if(NOT ROOT_RELEASE)
   include (CTest)
   configure_file(${CMAKE_SOURCE_DIR}/CTestCustom.cmake ${CMAKE_BINARY_DIR}/CTestCustom.cmake COPYONLY)
   if(BUILD_TESTING)
      add_subdirectory(tests)
   endif(BUILD_TESTING)

   set(BUILD_EXAMPLES FALSE CACHE BOOL "Build examples.")
   if(BUILD_EXAMPLES)
      add_subdirectory(examples)
   endif(BUILD_EXAMPLES)
endif()

# Hide VC_IMPL as it is only meant for users of Vc
mark_as_advanced(VC_IMPL)
