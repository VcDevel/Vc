cmake_minimum_required(VERSION 2.8.3)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BUILD_DIR)
   message(ERROR "You don't want to configure in the source directory!")
endif()

project(Vc)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(ALIROOT_RELEASE FALSE CACHE BOOL "Set up for creating a Vc copy inside AliRoot.")
mark_as_advanced(ALIROOT_RELEASE)

set(disabled_targets)

include (VcMacros)
include (AddTargetProperty)
include (OptimizeForArchitecture)

vc_determine_compiler()

if(Vc_COMPILER_IS_GCC AND NOT ALIROOT_RELEASE)
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
         c++11_gather_avx
         c++11_gather_sse
         c++11_gather_VC_USE_SET_GATHERS_avx
         c++11_gather_VC_USE_SET_GATHERS_sse
         c++11_scatter_avx
         c++11_scatter_sse
         )
   elseif(Vc_GCC_VERSION STREQUAL "4.5.0" OR Vc_GCC_VERSION STREQUAL "4.5.1")
      UserWarning("GCC 4.5.[12] are known to generate an internal compiler error on the memory unit test.
      (http://gcc.gnu.org/bugzilla/show_bug.cgi?id=46723)
      The test will therefore not be compiled and executed.")
      list(APPEND disabled_targets
         memory_scalar
         memory_sse
         memory_avx
         c++11_memory_scalar
         c++11_memory_sse
         c++11_memory_avx
         )
   elseif(Vc_GCC_VERSION STREQUAL "4.5.2")
      UserWarning("GCC 4.5.2 generates an internal compiler error on the memory_scalar unit test. The test will not be compiled and executed.")
      list(APPEND disabled_targets
         memory_scalar
         c++11_memory_scalar
         )
   endif()
endif()

if(NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

vc_set_preferred_compiler_flags(WARNING_FLAGS BUILDTYPE_FLAGS)
add_definitions("${Vc_DEFINITIONS}")

if(Vc_COMPILER_IS_INTEL)
   # per default icc is not IEEE compliant, but we need that for verification
   set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -w1 -fp-model precise")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w1 -fp-model precise")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "" AND NOT CMAKE_CXX_FLAGS MATCHES "-O[123]")
   message(STATUS "WARNING! It seems you are compiling without optimization. Please set CMAKE_BUILD_TYPE.")
endif(CMAKE_BUILD_TYPE STREQUAL "" AND NOT CMAKE_CXX_FLAGS MATCHES "-O[123]")

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_custom_target(other VERBATIM)
add_custom_target(Scalar COMMENT "build Scalar code" VERBATIM)
add_custom_target(SSE COMMENT "build SSE code" VERBATIM)
add_custom_target(AVX COMMENT "build AVX code" VERBATIM)

if(NOT ALIROOT_RELEASE)
   if(USE_AVX)
      add_library(Vc STATIC vector.cpp cpuid.cpp support.cpp avx/sorthelper.cpp)
   else()
      set(_srcs vector.cpp cpuid.cpp support.cpp)

      if(NOT Vc_AVX_INTRINSICS_BROKEN)
         # we'd still like to have avx/sorthelper.cpp built in, but that requires compilation with -mavx (or a comparable flag)
         foreach(_flag "-xAVX" "-mavx" "-arch:AVX")
            check_cxx_compiler_flag("${_flag}" check_cxx_compiler_flag_${_flag})
            if(check_cxx_compiler_flag_${_flag})
               if(_flag STREQUAL "-xAVX")
                  set(_flag "${_flag} -diag-disable 10121") # disable the warning "overriding -xSSE4.2 with -xAVX"
               endif()
               set(_srcs ${_srcs} avx/sorthelper.cpp)
               set_source_files_properties(avx/sorthelper.cpp PROPERTIES COMPILE_FLAGS "${_flag}")
               break()
            endif()
         endforeach()
      endif()
      add_library(Vc STATIC ${_srcs})
   endif()
   add_target_property(Vc COMPILE_FLAGS "-DVC_COMPILE_LIB")
   add_target_property(Vc LABELS "other")
   AddCompilerFlag("-fPIC" CXX_RESULT _has_fPIC)
   if(_has_fPIC)
      add_target_property(Vc COMPILE_FLAGS "-fPIC")
   endif()
   add_dependencies(other Vc)

   install(TARGETS Vc DESTINATION lib/)
else()
   # libVc should be compiled in the AliRoot tree, so we need to install the sources
   install(FILES vector.cpp cpuid.cpp support.cpp DESTINATION src)
   install(FILES avx/sorthelper.cpp DESTINATION src/avx)
endif()

install(DIRECTORY include/Vc/ DESTINATION include/Vc)
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
   cmake/OptimizeForArchitecture.cmake
   cmake/AddCompilerFlag.cmake
   cmake/CheckCCompilerFlag.cmake
   cmake/CheckCXXCompilerFlag.cmake
   )
if(ALIROOT_RELEASE)
   install(FILES
      ${cmake_install_files}
      DESTINATION cmake
      )
else()
   install(FILES
      ${cmake_install_files}
      ${CMAKE_CURRENT_BINARY_DIR}/cmake/VcConfig.cmake
      ${CMAKE_CURRENT_BINARY_DIR}/cmake/VcConfigVersion.cmake
      cmake/FindVc.cmake
      DESTINATION lib/cmake/Vc
      )
endif()

if(NOT ALIROOT_RELEASE)
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
