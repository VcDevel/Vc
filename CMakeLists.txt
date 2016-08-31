cmake_minimum_required(VERSION 3.0)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
   message(FATAL_ERROR "You don't want to configure in the source directory!")
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
   if(EXISTS "${CMAKE_INSTALL_PREFIX}/Module.mk")
      file(READ "${CMAKE_INSTALL_PREFIX}/Module.mk" ROOT_MODULE_MK)
      if(NOT "${ROOT_MODULE_MK}" MATCHES "\nMODNAME *:= *vc *\n")
         message(FATAL_ERROR "CMAKE_INSTALL_PREFIX is incorrect. It must point to the Vc subdirectory inside ROOT/AliRoot")
      endif()
      set(_extra_namespace "ROOT")
   endif()
   if(EXISTS "${CMAKE_INSTALL_PREFIX}/Vc.cmake")
      file(READ "${CMAKE_INSTALL_PREFIX}/Vc.cmake" ALIROOT_VC_CMAKE)
      if(NOT "${ALIROOT_VC_CMAKE}" MATCHES "\nmacro\\(ALICE_UseVc\\)\n")
         message(FATAL_ERROR "CMAKE_INSTALL_PREFIX is incorrect. It must point to the Vc subdirectory inside ROOT/AliRoot")
      endif()
      set(_extra_namespace "AliRoot")
   endif()
else()
   option(USE_CCACHE "If enabled, ccache will be used (if it exists on the system) to speed up recompiles." OFF)
   if(USE_CCACHE)
      find_program(CCACHE_COMMAND ccache)
      if(CCACHE_COMMAND)
         mark_as_advanced(CCACHE_COMMAND)
         set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_COMMAND}")
      endif()
   endif()

   # TODO: check that 'decltype' compiles
   # TODO: check that 'constexpr' compiles
   if(NOT Vc_COMPILER_IS_MSVC) # MSVC doesn't provide a switch to turn C++11 on/off AFAIK
      AddCompilerFlag("-std=c++17" CXX_RESULT _ok MIC_CXX_RESULT _mic_ok CXX_FLAGS CMAKE_CXX_FLAGS MIC_CXX_FLAGS Vc_MIC_CXX_FLAGS)
      if(MIC_NATIVE_FOUND AND NOT _mic_ok)
         AddCompilerFlag("-std=c++1z" MIC_CXX_RESULT _mic_ok MIC_CXX_FLAGS Vc_MIC_CXX_FLAGS)
         if(NOT _mic_ok)
            message(FATAL_ERROR "Vc 2.x requires C++17. The MIC native compiler does not support any of the C++17 language flags.")
         endif()
      endif()
      if(NOT _ok)
         AddCompilerFlag("-std=c++1z" CXX_RESULT _ok CXX_FLAGS CMAKE_CXX_FLAGS)
         if(NOT _ok)
            message(FATAL_ERROR "Vc 2.x requires C++17. It seems this is not available. If this was incorrectly determined please notify vc-devel@compeng.uni-frankfurt.de")
         endif()
      endif()
   elseif(Vc_MSVC_VERSION LESS 180021114)
      message(FATAL_ERROR "Vc 2.x requires C++17 support.")
   endif()

   if(Vc_COMPILER_IS_GCC)
      if(Vc_GCC_VERSION VERSION_GREATER "5.0.0" AND Vc_GCC_VERSION VERSION_LESS "6.0.0")
         UserWarning("GCC 5 goes into an endless loop comiling example_scaling_scalar. Therefore, this target is disabled.")
         list(APPEND disabled_targets
            example_scaling_scalar
            )
      elseif(Vc_GCC_VERSION VERSION_GREATER "5.99")
         AddCompilerFlag(-Wno-ignored-attributes)
      endif()
   elseif(Vc_COMPILER_IS_MSVC)
      # Disable warning "C++ exception specification ignored except to indicate a function is not __declspec(nothrow)"
      # MSVC emits the warning for the _UnitTest_Compare desctructor which needs the throw declaration so that it doesn't std::terminate
      AddCompilerFlag("/wd4290")
   endif()
   if(MIC_NATIVE_FOUND)
      if("${Vc_MIC_ICC_VERSION}" VERSION_LESS "16.1.0")
         UserWarning("ICC for MIC uses an incompatible STL. Disabling simdize_mic.")
         list(APPEND disabled_targets
            simdize_mic
            example_simdize_mic
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

add_definitions(${Vc_DEFINITIONS})
add_compile_options(${Vc_COMPILE_FLAGS})

if(Vc_COMPILER_IS_INTEL)
   # per default icc is not IEEE compliant, but we need that for verification
   AddCompilerFlag("-fp-model source")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "" AND NOT CMAKE_CXX_FLAGS MATCHES "-O[123]")
   message(STATUS "WARNING! It seems you are compiling without optimization. Please set CMAKE_BUILD_TYPE.")
endif(CMAKE_BUILD_TYPE STREQUAL "" AND NOT CMAKE_CXX_FLAGS MATCHES "-O[123]")

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(NOT ROOT_RELEASE)
   add_custom_target(other VERBATIM)
   add_custom_target(Scalar COMMENT "build Scalar code" VERBATIM)
   add_custom_target(SSE COMMENT "build SSE code" VERBATIM)
   add_custom_target(SSE2 COMMENT "build SSE2 code" VERBATIM)
   add_custom_target(SSE4_2 COMMENT "build SSE4.2 code" VERBATIM)
   add_dependencies(SSE SSE2 SSE4_2)
   add_custom_target(AVX COMMENT "build AVX code" VERBATIM)
   add_custom_target(AVX2 COMMENT "build AVX2 code" VERBATIM)
   add_custom_target(MIC COMMENT "build MIC code" VERBATIM)
   add_custom_target(KNL COMMENT "build KNL code" VERBATIM)
   add_custom_target(AVX512 COMMENT "build AVX512 code" VERBATIM)

   AddCompilerFlag(-ftemplate-depth=128 CXX_FLAGS CMAKE_CXX_FLAGS MIC_CXX_FLAGS CMAKE_MIC_CXX_FLAGS)

   set(libvc_compile_flags "-DVc_COMPILE_LIB")
   set(libvc_mic_compile_flags "-DVc_COMPILE_LIB")
   AddCompilerFlag("-fPIC" CXX_FLAGS libvc_compile_flags MIC_CXX_FLAGS libvc_mic_compile_flags)

   if(MIC_FOUND)
      mic_add_library(Vc_MIC STATIC src/mic_const.cpp src/cpuid.cpp src/support_x86.cpp src/mic_sorthelper.cpp
         COMPILE_FLAGS ${libvc_mic_compile_flags})
      add_target_property(Vc_MIC LABELS "MIC")
      add_dependencies(MIC Vc_MIC)
      get_target_property(outputName Vc_MIC OUTPUT_NAME)
      install(FILES ${outputName} DESTINATION lib${LIB_SUFFIX})
   endif()

   set(_srcs src/const.cpp)
   if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "([x3-7]86|AMD64)")

      list(APPEND _srcs src/cpuid.cpp src/support_x86.cpp)
      vc_compile_for_all_implementations(_srcs src/trigonometric.cpp ONLY SSE2 SSE3 SSSE3 SSE4_1 AVX SSE+XOP+FMA4 AVX+XOP+FMA4 AVX+XOP+FMA AVX+FMA)# AVX2+FMA+BMI2)
      vc_compile_for_all_implementations(_srcs src/sse_sorthelper.cpp ONLY SSE2 SSE4_1 AVX AVX2+FMA+BMI2)
      vc_compile_for_all_implementations(_srcs src/avx_sorthelper.cpp ONLY AVX AVX2+FMA+BMI2)
   else()
      message(FATAL_ERROR "Unsupported target architecture '${CMAKE_SYSTEM_PROCESSOR}'. No support_???.cpp file exists for this architecture.")
   endif()
   add_library(Vc STATIC ${_srcs})
   set_property(TARGET Vc APPEND PROPERTY COMPILE_OPTIONS ${libvc_compile_flags})
   add_target_property(Vc LABELS "other")
   if(XCODE)
      # TODO: document what this does and why it has no counterpart in the non-XCODE logic
      set_target_properties(Vc PROPERTIES XCODE_ATTRIBUTE_GCC_INLINES_ARE_PRIVATE_EXTERN "NO")
      set_target_properties(Vc PROPERTIES XCODE_ATTRIBUTE_GCC_SYMBOLS_PRIVATE_EXTERN "YES")
      set_target_properties(Vc PROPERTIES XCODE_ATTRIBUTE_CLANG_CXX_LANGUAGE_STANDARD "c++0x")
      set_target_properties(Vc PROPERTIES XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY "libc++")
   elseif(UNIX AND Vc_COMPILER_IS_CLANG)
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
   endif()
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
         \"#define Vc__SYMBOL_VERSION ${_extra_namespace}\\\\1\" data \"\${data}\")
         file(WRITE \"${CMAKE_INSTALL_PREFIX}/include/Vc/global.h\" \"\${data}\")
   ")
endif()

# Install all implementation headers
install(DIRECTORY scalar sse avx mic common traits DESTINATION include/Vc FILES_MATCHING REGEX "/*.(h|tcc|def)$")

# read version parts from version.h to be put into VcConfig.cmake
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/include/Vc/version.h _version_lines REGEX "^#define Vc_VERSION_STRING ")
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
   cmake/CheckMicCCompilerFlag.cmake
   cmake/CheckMicCXXCompilerFlag.cmake
   cmake/FindMIC.cmake
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
      DESTINATION lib${LIB_SUFFIX}/cmake/Vc
      )
endif()

if(NOT ROOT_RELEASE)
   #Release# option(BUILD_TESTING "Build the testing tree." OFF)
   include (CTest)
   configure_file(${CMAKE_SOURCE_DIR}/CTestCustom.cmake ${CMAKE_BINARY_DIR}/CTestCustom.cmake COPYONLY)
   if(BUILD_TESTING)
      add_custom_target(build_tests ALL VERBATIM)
      add_subdirectory(tests)
   endif(BUILD_TESTING)

   set(BUILD_EXAMPLES FALSE CACHE BOOL "Build examples.")
   if(BUILD_EXAMPLES)
      add_subdirectory(examples)
   endif(BUILD_EXAMPLES)
endif()

# Hide Vc_IMPL as it is only meant for users of Vc
mark_as_advanced(Vc_IMPL)
