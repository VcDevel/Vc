cmake_minimum_required(VERSION 2.6.2)
project(Vc)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include (MacroEnsureVersion)
include (OptimizeForArchitecture)
include (AddCompilerFlag)
include (CheckCXXSourceRuns)
include (AddTargetProperty)

find_package(Larrabee)
if(LARRABEE_FOUND)
   include_directories(${LRB_HOST_INCLUDE_DIR})
else(LARRABEE_FOUND)
   find_file(prototypes_header lrbni_prototypes.h PATHS "${CMAKE_SOURCE_DIR}/larrabee/" NO_DEFAULT_PATH)
   if(prototypes_header)
      set(LARRABEE_FOUND "prototype")
      add_definitions(-DVC_LRBni_PROTOTYPE_H)
   else(prototypes_header)
      message(STATUS "Download the LRBni Prototype Header from http://software.intel.com/en-us/articles/prototype-primitives-guide/ and save it as ${CMAKE_SOURCE_DIR}/larrabee/lrbni_prototypes.h\nYou will also need to apply the patch from http://software.intel.com/en-us/forums/showthread.php?t=66359")
   endif(prototypes_header)
endif(LARRABEE_FOUND)

if(NO_BSF_LOOPS)
   add_definitions(-DVC_NO_BSF_LOOPS)
endif(NO_BSF_LOOPS)
if(NO_GATHER_TRICKS)
   add_definitions(-DVC_NO_GATHER_TRICKS)
endif(NO_GATHER_TRICKS)
if(GATHER_SET)
   add_definitions(-DVC_GATHER_SET)
endif(GATHER_SET)

set(SSE_INTRINSICS_BROKEN false)

if(CMAKE_COMPILER_IS_GNUCXX)
   ##################################################################################################
   #                                              GCC                                               #
   ##################################################################################################
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -W -Wall -Wimplicit -Wswitch -Wformat -Wchar-subscripts -Wparentheses -Wmultichar -Wtrigraphs -Wpointer-arith -Wcast-align -Wreturn-type -Wno-unused-function -ansi -pedantic -Wno-long-long -Wshadow")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -Wimplicit -Wswitch -Wformat -Wchar-subscripts -Wparentheses -Wmultichar -Wtrigraphs -Wpointer-arith -Wcast-align -Wreturn-type -Wno-unused-function -ansi -pedantic -Wno-long-long -Wshadow")
   AddCompilerFlag("-Wno-variadic-macros")
   AddCompilerFlag("-fno-threadsafe-statics")
   set(CMAKE_CXX_FLAGS_DEBUG          "${CMAKE_CXX_FLAGS_DEBUG}")
   set(CMAKE_CXX_FLAGS_MINSIZEREL     "${CMAKE_CXX_FLAGS_MINSIZEREL} ")
   set(CMAKE_CXX_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE} -O3")
   set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -DNDEBUG -O3")
   set(CMAKE_C_FLAGS_DEBUG          "${CMAKE_C_FLAGS_DEBUG}")
   set(CMAKE_C_FLAGS_MINSIZEREL     "${CMAKE_C_FLAGS_MINSIZEREL} ")
   set(CMAKE_C_FLAGS_RELEASE        "${CMAKE_C_FLAGS_RELEASE} -O3")
   set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -DNDEBUG -O3")

   # check the GCC version
   exec_program(${CMAKE_C_COMPILER} ARGS -dumpversion OUTPUT_VARIABLE _gcc_version)
   macro_ensure_version("4.4.1" "${_gcc_version}" GCC_4_4_1)
   if(NOT GCC_4_4_1)
      message(STATUS "\n-- \n-- NOTE: Your GCC is older than 4.4.1. This is known to cause problems/bugs. Please update to the latest GCC if you can.\n-- \n-- ")
      macro_ensure_version("4.3.0" "${_gcc_version}" GCC_4_3_0)
      if(NOT GCC_4_3_0)
         message(STATUS "WARNING: Your GCC is older than 4.3.0. It is unable to handle all SSE2 intrinsics. All SSE code will be disabled. Please update to the latest GCC if you can.\n-- \n-- ")
         set(SSE_INTRINSICS_BROKEN true)
      endif(NOT GCC_4_3_0)
   endif(NOT GCC_4_4_1)

   if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
      set(ENABLE_STRICT_ALIASING true CACHE BOOL "Enables strict aliasing rules for more aggressive optimizations")
      if(NOT ENABLE_STRICT_ALIASING)
         set(CMAKE_CXX_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE} -fno-strict-aliasing ")
         set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fno-strict-aliasing ")
         set(CMAKE_C_FLAGS_RELEASE        "${CMAKE_C_FLAGS_RELEASE} -fno-strict-aliasing ")
         set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -fno-strict-aliasing ")
      endif(NOT ENABLE_STRICT_ALIASING)
   endif(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")

   # if compiling for 32 bit x86 we need to use the -mfpmath=sse since the x87 is broken by design
   CHECK_CXX_SOURCE_RUNS("int main() { return sizeof(void*) != 8; }" VOID_PTR_IS_64BIT)
   if(NOT VOID_PTR_IS_64BIT)
      exec_program(${CMAKE_C_COMPILER} ARGS -dumpmachine OUTPUT_VARIABLE _gcc_machine)
      if(_gcc_machine MATCHES "[x34567]86")
         AddCompilerFlag("-mfpmath=sse")
      endif(_gcc_machine MATCHES "[x34567]86")
   endif(NOT VOID_PTR_IS_64BIT)

   # GCC uses binutils' assembler. And it has to be recent enough otherwise it'll bail out at some of the instructions Vc uses
   find_program(_binutils_as as)
   if(NOT _binutils_as)
      message(WARNING "Could not find 'as', the assembly normally used by GCC. Hoping everything will work out...")
   else(NOT _binutils_as)
      exec_program(${_binutils_as} ARGS --version OUTPUT_VARIABLE _as_version)
      string(REGEX REPLACE "\\([^\\)]*\\)" "" _as_version "${_as_version}")
      string(REGEX MATCH "[1-9]\\.[0-9]+(\\.[0-9]+)?" _as_version "${_as_version}")
      macro_ensure_version("2.18.93" "${_as_version}" _as_good)
      if(NOT _as_good)
         message(SEND_ERROR "Your binutils is too old (${_as_version}). The assembler will not be able to compile Vc.")
      endif(NOT _as_good)
   endif(NOT _binutils_as)
elseif(CMAKE_CXX_COMPILER MATCHES "/(icpc|icc)$")
   ##################################################################################################
   #                                          Intel Compiler                                        #
   ##################################################################################################

   set(CMAKE_CXX_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE} -O3")
   set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -DNDEBUG -O3")
   set(CMAKE_C_FLAGS_RELEASE          "${CMAKE_C_FLAGS_RELEASE} -O3")
   set(CMAKE_C_FLAGS_RELWITHDEBINFO   "${CMAKE_C_FLAGS_RELWITHDEBINFO} -DNDEBUG -O3")

   set(ALIAS_FLAGS "-no-ansi-alias")
   if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
      set(ENABLE_STRICT_ALIASING true CACHE BOOL "Enables strict aliasing rules for more aggressive optimizations")
      if(ENABLE_STRICT_ALIASING)
         set(ALIAS_FLAGS "-ansi-alias")
      endif(ENABLE_STRICT_ALIASING)
   endif(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")

   # per default icc is not IEEE compliant, but we need that for verification
   set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${ALIAS_FLAGS} -w1 -fp-model precise")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ALIAS_FLAGS} -w1 -fp-model precise")
elseif(MSVC)
   AddCompilerFlag("/wd4800") # Disable warning "forcing value to bool"
   add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif(CMAKE_COMPILER_IS_GNUCXX)

OptimizeForArchitecture()

if(CMAKE_BUILD_TYPE STREQUAL "" AND NOT CMAKE_CXX_FLAGS MATCHES "-O[123]")
   message(STATUS "WARNING! It seems you are compiling without optimization. Please set CMAKE_BUILD_TYPE.")
endif(CMAKE_BUILD_TYPE STREQUAL "" AND NOT CMAKE_CXX_FLAGS MATCHES "-O[123]")

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(Vc STATIC vector.cpp common/support.cpp)
add_target_property(Vc COMPILE_FLAGS "-DVC_COMPILE_LIB")

install(TARGETS Vc DESTINATION lib/)
install(DIRECTORY include/Vc/ DESTINATION include/Vc)
install(DIRECTORY larrabee simple sse common DESTINATION include/Vc FILES_MATCHING REGEX "/*.(h|tcc|def)$")

include (CTest)
configure_file(${CMAKE_SOURCE_DIR}/CTestCustom.cmake ${CMAKE_BINARY_DIR}/CTestCustom.cmake COPYONLY)
if(BUILD_TESTING)
   add_subdirectory(tests)
endif(BUILD_TESTING)

set(BUILD_BENCHMARKS TRUE CACHE BOOL "Build benchmarks.")
if(BUILD_BENCHMARKS)
   add_subdirectory(benchmarks)
endif(BUILD_BENCHMARKS)

set(BUILD_EXAMPLES FALSE CACHE BOOL "Build examples.")
if(BUILD_EXAMPLES)
   add_subdirectory(examples)
endif(BUILD_EXAMPLES)

mark_as_advanced(prototypes_header)
