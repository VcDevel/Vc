cmake_minimum_required(VERSION 2.8.1)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BUILD_DIR)
   message(ERROR "You don't want to configure in the source directory!")
endif()

project(Vc)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(disabled_targets)

include(VcMacros)
vc_determine_compiler()

if(NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

include (OptimizeForArchitecture)
include (AddTargetProperty)

vc_set_preferred_compiler_flags()

find_package(Qt4)

OptimizeForArchitecture()

if(CMAKE_BUILD_TYPE STREQUAL "" AND NOT CMAKE_CXX_FLAGS MATCHES "-O[123]")
   message(STATUS "WARNING! It seems you are compiling without optimization. Please set CMAKE_BUILD_TYPE.")
endif(CMAKE_BUILD_TYPE STREQUAL "" AND NOT CMAKE_CXX_FLAGS MATCHES "-O[123]")

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(CpuId STATIC cpuid.cpp)
add_target_property(CpuId LABELS "other")

if(USE_AVX)
   add_library(Vc STATIC vector.cpp common/support.cpp avx/sorthelper.cpp)
else()
   add_library(Vc STATIC vector.cpp common/support.cpp)
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
