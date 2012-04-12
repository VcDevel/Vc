cmake_minimum_required(VERSION 2.8.1)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BUILD_DIR)
   message(ERROR "You don't want to configure in the source directory!")
endif()

project(Vc)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(disabled_targets)

include (VcMacros)
include (AddTargetProperty)
include (OptimizeForArchitecture)

find_package(Qt4)

vc_determine_compiler()

if(VC_COMPILER_IS_GCC)
   if(VC_GCC_VERSION STREQUAL "4.6.0")
      message(WARNING "GCC 4.6.0 is broken. The following tests are therefore disabled:
      gather_avx, gather_sse, gather_VC_USE_SET_GATHERS_avx, gather_VC_USE_SET_GATHERS_sse,
      gather_sse_LOOP, scatter_avx, and scatter_sse")
      list(APPEND disabled_targets
         gather_avx
         gather_sse
         gather_VC_USE_SET_GATHERS_avx
         gather_VC_USE_SET_GATHERS_sse
         gather_sse_LOOP
         scatter_avx
         scatter_sse
         )
   elseif(VC_GCC_VERSION STREQUAL "4.5.0" OR VC_GCC_VERSION STREQUAL "4.5.1")
      message(WARNING "GCC 4.5.[12] are known to generate an internal compiler error on the memory unit test.
      (http://gcc.gnu.org/bugzilla/show_bug.cgi?id=46723)
      The test will therefore not be compiled and executed.")
      list(APPEND disabled_targets
         memory_scalar
         memory_sse
         memory_avx
         )
   endif()
endif()

if(NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

vc_set_preferred_compiler_flags()

OptimizeForArchitecture()

if(CMAKE_BUILD_TYPE STREQUAL "" AND NOT CMAKE_CXX_FLAGS MATCHES "-O[123]")
   message(STATUS "WARNING! It seems you are compiling without optimization. Please set CMAKE_BUILD_TYPE.")
endif(CMAKE_BUILD_TYPE STREQUAL "" AND NOT CMAKE_CXX_FLAGS MATCHES "-O[123]")

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(USE_AVX)
   add_library(Vc STATIC vector.cpp cpuid.cpp support.cpp avx/sorthelper.cpp)
else()
   set(_srcs vector.cpp cpuid.cpp support.cpp)

   # we'd still like to have avx/sorthelper.cpp built in, but that requires compilation with -mavx (or a comparable flag)
   foreach(_flag "-mavx" "-xAVX" "-arch:AVX")
      check_cxx_compiler_flag("${_flag}" check_cxx_compiler_flag_${_flag})
      if(check_cxx_compiler_flag_${_flag})
         set(_srcs ${_srcs} avx/sorthelper.cpp)
         set_source_files_properties(avx/sorthelper.cpp PROPERTIES COMPILE_FLAGS "${_flag}")
         break()
      endif()
   endforeach()
   add_library(Vc STATIC ${_srcs})
endif()
add_target_property(Vc COMPILE_FLAGS "-DVC_COMPILE_LIB")
add_target_property(Vc LABELS "other")

add_custom_target(other VERBATIM)
add_custom_target(Scalar COMMENT "build Scalar code" VERBATIM)
add_custom_target(SSE COMMENT "build SSE code" VERBATIM)
add_custom_target(AVX COMMENT "build AVX code" VERBATIM)
add_dependencies(other Vc)

install(TARGETS Vc DESTINATION lib/)
install(DIRECTORY include/Vc/ DESTINATION include/Vc)
install(DIRECTORY scalar sse avx common DESTINATION include/Vc FILES_MATCHING REGEX "/*.(h|tcc|def)$")

install(FILES
   cmake/VcMacros.cmake
   cmake/FindVc.cmake
   cmake/OptimizeForArchitecture.cmake
   cmake/AddCompilerFlag.cmake
   cmake/MacroEnsureVersion.cmake
   cmake/CheckCCompilerFlag.cmake
   DESTINATION share/Vc/cmake/
   )

include (CTest)
configure_file(${CMAKE_SOURCE_DIR}/CTestCustom.cmake ${CMAKE_BINARY_DIR}/CTestCustom.cmake COPYONLY)
if(BUILD_TESTING)
   add_subdirectory(tests)
endif(BUILD_TESTING)

set(BUILD_EXAMPLES FALSE CACHE BOOL "Build examples.")
if(BUILD_EXAMPLES)
   add_subdirectory(examples)
endif(BUILD_EXAMPLES)

mark_as_advanced(prototypes_header)
