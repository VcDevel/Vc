#!ctest -S
if(NOT CTEST_SOURCE_DIRECTORY)
   get_filename_component(CTEST_SOURCE_DIRECTORY "${CMAKE_CURRENT_LIST_FILE}" PATH)
endif()

macro(read_argument name)
   if(NOT ${name})
      if(DEFINED ENV{${name}})
         set(${name} "$ENV{${name}}")
      elseif(${ARGC} GREATER 1)
         set(${name} "${ARGV1}")
      endif()
   endif()
   #message(STATUS "argument ${name}: '${${name}}'")
endmacro()

# Dashboard Model
################################################################################
read_argument(dashboard_model "Experimental")
set(is_continuous FALSE)
set(is_experimental FALSE)
set(is_nightly FALSE)
if(${dashboard_model} STREQUAL "Continuous")
   set(is_continuous TRUE)
elseif(${dashboard_model} STREQUAL "Experimental")
   set(is_experimental TRUE)
elseif(${dashboard_model} STREQUAL "Nightly")
   set(is_nightly TRUE)
else()
   message(FATAL_ERROR "Unknown dashboard_model '${dashboard_model}'. Please use one of Experimental, Continuous, Nightly.")
endif()

# Set build variables from environment variables
################################################################################
read_argument(target_architecture)
read_argument(skip_tests FALSE)
read_argument(subset)
read_argument(CXXFLAGS "")
set(ENV{CXXFLAGS} "${CXXFLAGS}")

# Set CMAKE_BUILD_TYPE from environment variable
################################################################################
set(appveyor $ENV{APPVEYOR})
if(appveyor)
   set(build_type "$ENV{CONFIGURATION}")
   if(NOT build_type)
      set(build_type "Release")
   endif()
else()
   read_argument(build_type "Release")
endif()

# better make sure we get english output (this is vital for the implicit_type_conversion_failures tests)
################################################################################
set(ENV{LANG} "en_US")

# determine the git branch we're testing
################################################################################
file(READ "${CTEST_SOURCE_DIRECTORY}/.git/HEAD" git_branch)
string(STRIP "${git_branch}" git_branch)                         # -> ref: refs/heads/user/foobar
string(REPLACE "ref: refs/heads/" "" git_branch "${git_branch}") # -> user/foobar
if(NOT git_branch MATCHES "^[0-9a-f]+$")                         # an actual branch name, not a hash
   # read the associated hash
   if(EXISTS "${CTEST_SOURCE_DIRECTORY}/.git/refs/heads/${git_branch}")
      set(git_hash_file "${CTEST_SOURCE_DIRECTORY}/.git/refs/heads/${git_branch}")
      file(READ "${git_hash_file}" git_hash LIMIT 7)
      string(STRIP "${git_hash}" git_hash)
   endif()
   string(REGEX REPLACE "^.*/" "" git_branch "${git_branch}")
else()
   # a long hash -> make it shorter
   string(SUBSTRING "${git_branch}" 0 7 git_hash)

   # try harder to find a branch name
   find_program(GIT git)
   unset(git_branch_out)
   if(GIT)
      execute_process(COMMAND
         "${GIT}" branch -a --contains=${git_branch} -v --abbrev=0
         WORKING_DIRECTORY "${PROJECT_DIRECTORY}"
         OUTPUT_VARIABLE git_branch_out
         OUTPUT_STRIP_TRAILING_WHITESPACE
         )
      #message("${git_branch_out}")
      string(REPLACE "\n" ";" git_branch_out "${git_branch_out}")
      #message("${git_branch_out}")
      foreach(i IN LISTS git_branch_out)
         string(REGEX MATCH "[^() ]+ +${git_branch}" i "${i}")
         string(REGEX REPLACE "^.*/" "" i "${i}")
         string(REGEX REPLACE " *${git_branch}.*$" "" i "${i}")
         if(NOT "${i}" MATCHES "^ *$")
            set(git_branch "${i}")
            break()
         endif()
      endforeach()
   endif()
   if(git_branch MATCHES "^[0-9a-f]+$")
      # still just a hash -> make it empty because the hash is communicated via git_hash
      set(git_branch "")
   endif()
endif()
if(git_branch MATCHES "^gh-[0-9]+")
   # it's a feature branch referencing a GitHub issue number -> make it short
   string(REGEX MATCH "^gh-[0-9]+" git_branch "${git_branch}")
endif()

# determine the (short) hostname of the build machine
################################################################################
set(travis_os $ENV{TRAVIS_OS_NAME})
set(github_ci $ENV{GITHUB_ACTIONS})
if(travis_os)
   set(CTEST_SITE "Travis CI")
elseif(appveyor)
   set(CTEST_SITE "AppVeyor CI")
elseif(github_ci)
   set(CTEST_SITE "GitHub Actions")
else()
   execute_process(COMMAND hostname -s RESULT_VARIABLE ok OUTPUT_VARIABLE CTEST_SITE ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
   if(NOT ok EQUAL 0)
      execute_process(COMMAND hostname OUTPUT_VARIABLE CTEST_SITE ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
   endif()
endif()

# collect system information for the "Build Name" (and build setup, e.g. -j)
################################################################################
find_program(UNAME uname)
if(UNAME)
   execute_process(COMMAND ${UNAME} -s OUTPUT_VARIABLE arch OUTPUT_STRIP_TRAILING_WHITESPACE)
   string(TOLOWER "${arch}" arch)
   execute_process(COMMAND ${UNAME} -m OUTPUT_VARIABLE chip OUTPUT_STRIP_TRAILING_WHITESPACE)
   string(TOLOWER "${chip}" chip)
   string(REPLACE "x86_64" "amd64" chip "${chip}")
else()
   find_program(CMD cmd)
   if(CMD)
      execute_process(COMMAND cmd /D /Q /C ver OUTPUT_VARIABLE arch OUTPUT_STRIP_TRAILING_WHITESPACE)
      string(REGEX REPLACE "^.*Windows[^0-9]*([.0-9]+).*$" "Windows \\1" arch "${arch}")
   else()
      string(TOLOWER "$ENV{TARGET_PLATFORM}" arch)
      if(arch)
         if("$ENV{WindowsSDKVersionOverride}")
            set(arch "${arch} SDK $ENV{WindowsSDKVersionOverride}")
         endif()
      else()
         string(TOLOWER "${CMAKE_SYSTEM_NAME}" arch)
      endif()
   endif()
   execute_process(COMMAND
      reg query "HKLM\\HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0" /v Identifier
      OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE processorId)
   if("${processorId}" MATCHES "(Intel|AMD)64")
      set(chip "amd64")
   elseif("${processorId}" MATCHES "x86")
      set(chip "x86")
   else()
      set(chip "unknown")
   endif()
endif()

# Determine the processor count (number_of_processors)
################################################################################
read_argument(number_of_processors)
if(NOT number_of_processors)
   if("${arch}" MATCHES "[Ww]indows" OR "${arch}" MATCHES "win7" OR "${arch}" MATCHES "mingw" OR "${arch}" MATCHES "msys")
      execute_process(COMMAND
         reg query "HKLM\\HARDWARE\\DESCRIPTION\\System\\CentralProcessor"
         OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE number_of_processors)
      string(REGEX REPLACE "[^0-9]+" "," number_of_processors "${number_of_processors}")
      string(REGEX REPLACE "^.*," "" number_of_processors "${number_of_processors}")
      math(EXPR number_of_processors "1 + ${number_of_processors}")
   elseif("${arch}" STREQUAL "darwin")
      execute_process(COMMAND sysctl -n hw.ncpu OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE number_of_processors)
   else()
      execute_process(COMMAND grep -c processor /proc/cpuinfo OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE number_of_processors)
   endif()
endif()

# Determine the compiler version string
################################################################################
set(compiler)
macro(extract_msvc_compiler_info CL)
   if(CL)
      execute_process(COMMAND ${CL} OUTPUT_QUIET ERROR_VARIABLE COMPILER_VERSION)
      string(REGEX MATCH "Version ([0-9\\.]+)" COMPILER_VERSION ${COMPILER_VERSION})
      string(REPLACE "Version " "" COMPILER_VERSION "${COMPILER_VERSION}")
      string(STRIP "${COMPILER_VERSION}" COMPILER_VERSION)
      set(COMPILER_VERSION "MSVC ${COMPILER_VERSION}")
      if("${CL}" MATCHES "amd64")
         set(COMPILER_VERSION "${COMPILER_VERSION} amd64")
      elseif("${CL}" MATCHES "ia64")
         set(COMPILER_VERSION "${COMPILER_VERSION} ia64")
      else()
         set(COMPILER_VERSION "${COMPILER_VERSION} x86")
      endif()
      set(compiler "MSVC")
   else()
      message(FATAL_ERROR "unknown compiler")
   endif()
endmacro()
macro(extract_gnuc_compiler_info CXX)
   string(REPLACE " " ";" CXX_AS_LIST "${CXX}")
   execute_process(COMMAND ${CXX_AS_LIST} --version OUTPUT_VARIABLE COMPILER_VERSION_COMPLETE ERROR_VARIABLE COMPILER_VERSION_COMPLETE OUTPUT_STRIP_TRAILING_WHITESPACE)
   string(REPLACE "\n" ";" COMPILER_VERSION "${COMPILER_VERSION_COMPLETE}")
   list(GET COMPILER_VERSION 0 COMPILER_VERSION)
   string(REPLACE "Open64 Compiler Suite: Version" "Open64" COMPILER_VERSION "${COMPILER_VERSION}")
   string(REPLACE "icpc" "ICC" COMPILER_VERSION "${COMPILER_VERSION}")
   string(REPLACE "gxx" "GCC" COMPILER_VERSION "${COMPILER_VERSION}")
   string(REPLACE "g++" "GCC" COMPILER_VERSION "${COMPILER_VERSION}")
   string(REPLACE ".exe" "" COMPILER_VERSION "${COMPILER_VERSION}")
   string(REPLACE " version" "" COMPILER_VERSION "${COMPILER_VERSION}")
   if(COMPILER_VERSION MATCHES "based on LLVM ")
      # e.g. "Apple LLVM version 6.0 (clang-600.0.57) (based on LLVM 3.5svn)"
      string(REGEX REPLACE ".*based on LLVM" "clang" COMPILER_VERSION "${COMPILER_VERSION}")
      string(REPLACE ")" "" COMPILER_VERSION "${COMPILER_VERSION}")
   endif()
   string(REGEX REPLACE " \\([^()]*\\)" "" COMPILER_VERSION "${COMPILER_VERSION}")
   string(REGEX REPLACE " \\[.*\\]" "" COMPILER_VERSION "${COMPILER_VERSION}")
   string(REGEX REPLACE "GCC-[0-9](\\.[0-9])?" "GCC" COMPILER_VERSION "${COMPILER_VERSION}")
   string(STRIP "${COMPILER_VERSION}" COMPILER_VERSION)
   if(COMPILER_VERSION_COMPLETE MATCHES "Free Software Foundation, Inc." AND NOT COMPILER_VERSION MATCHES "GCC")
      # it's GCC but without "GCC" in the COMPILER_VERSION string - fix it up
      string(REPLACE "cxx" "GCC" COMPILER_VERSION "${COMPILER_VERSION}")
      string(REPLACE "c++" "GCC" COMPILER_VERSION "${COMPILER_VERSION}")
      if(NOT COMPILER_VERSION MATCHES "GCC")
         set(COMPILER_VERSION "GCC (${COMPILER_VERSION})")
      endif()
   endif()
   if(COMPILER_VERSION MATCHES "clang")
      set(compiler "clang")
   elseif(COMPILER_VERSION MATCHES "ICC")
      set(compiler "ICC")
   elseif(COMPILER_VERSION MATCHES "Open64")
      set(compiler "Open64")
   elseif(COMPILER_VERSION MATCHES "GCC")
      if(WIN32)
         set(compiler "MinGW")
      else()
         set(compiler "GCC")
      endif()
   endif()
endmacro()
if("${arch}" MATCHES "[Ww]indows" OR "${arch}" MATCHES "win7" OR "${arch}" MATCHES "msys")
   find_program(CL cl)
   extract_msvc_compiler_info(${CL})
elseif(arch MATCHES "mingw")
   if("$ENV{CXX}" MATCHES "g\\+\\+" OR "$ENV{CXX}" MATCHES "gxx")
      set(GXX "$ENV{CXX}")
   else()
      find_program(GXX NAMES "g++" "gxx")
   endif()
   if(GXX)
      extract_gnuc_compiler_info("${GXX}")
   else()
      find_program(CL cl)
      extract_msvc_compiler_info(${CL})
   endif()
else()
   set(CXX "$ENV{CXX}")
   if(NOT CXX)
      unset(CXX)
      find_program(CXX NAMES c++ g++ clang++ icpc)
      if(NOT CXX)
         message(FATAL_ERROR "No C++ compiler found (arch: ${arch})")
      endif()
   endif()
   extract_gnuc_compiler_info("${CXX}")
endif()

# Build the CTEST_BUILD_NAME string
################################################################################
if(DEFINED target_architecture)
   set(tmp ${target_architecture})
else()
   execute_process(COMMAND cmake -Darch=${arch} -P ${CTEST_SOURCE_DIRECTORY}/print_target_architecture.cmake OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE tmp)
   string(REPLACE "-- " "" tmp "${tmp}")
endif()
if(build_type STREQUAL "Release")
   set(build_type_short "Rel")
elseif(build_type STREQUAL "Debug")
   set(build_type_short "Deb")
elseif(build_type STREQUAL "RelWithDebInfo")
   set(build_type_short "RDI")
elseif(build_type STREQUAL "RelWithDebug")
   set(build_type_short "RWD")
elseif(build_type STREQUAL "None")
   set(build_type_short "Non")
else()
   set(build_type_short "${build_type}")
endif()

string(STRIP "${git_branch} ${COMPILER_VERSION} ${CXXFLAGS} ${build_type_short} ${tmp}" CTEST_BUILD_NAME)
if(NOT is_nightly)
   # nightly builds shouldn't contain the git hash, since it makes expected builds impossible
   string(STRIP "${git_hash} ${CTEST_BUILD_NAME}" CTEST_BUILD_NAME)
   # instead make sure the hash is included in the notes
   if(DEFINED git_hash_file)
      list(APPEND CTEST_NOTES_FILES "${git_hash_file}")
   endif()
endif()
if(DEFINED subset)
   set(CTEST_BUILD_NAME "${CTEST_BUILD_NAME} ${subset}")
endif()

string(REPLACE "  " " " CTEST_BUILD_NAME "${CTEST_BUILD_NAME}")

# work around CDash limitations:
string(REPLACE "/" "_" CTEST_BUILD_NAME "${CTEST_BUILD_NAME}")
string(REPLACE "+" "x" CTEST_BUILD_NAME "${CTEST_BUILD_NAME}")

# Determine build directory
################################################################################
string(REGEX REPLACE "[][ ():, |!*]" "" CTEST_BINARY_DIRECTORY "${CTEST_BUILD_NAME}")
set(CTEST_BINARY_DIRECTORY "${CTEST_SOURCE_DIRECTORY}/build-${dashboard_model}/${CTEST_BINARY_DIRECTORY}")

# Give user feedback
################################################################################
#message("src:        ${CTEST_SOURCE_DIRECTORY}")
#message("obj:        ${CTEST_BINARY_DIRECTORY}")
message("build name: ${CTEST_BUILD_NAME}")
message("site:       ${CTEST_SITE}")
message("model:      ${dashboard_model}")

Set(CTEST_START_WITH_EMPTY_BINARY_DIRECTORY_ONCE TRUE)

list(APPEND CTEST_NOTES_FILES "${CTEST_SOURCE_DIRECTORY}/.git/HEAD")

# attach information on the OS
################################################################################
if(arch STREQUAL "linux")
   execute_process(COMMAND lsb_release -idrc OUTPUT_VARIABLE os_ident)
elseif(arch STREQUAL "darwin")
   execute_process(COMMAND system_profiler SPSoftwareDataType OUTPUT_VARIABLE os_ident)
else()
   set(os_ident "${arch}")
endif()
file(WRITE "${CTEST_SOURCE_DIRECTORY}/os_ident.txt" "${os_ident}")
list(APPEND CTEST_NOTES_FILES "${CTEST_SOURCE_DIRECTORY}/os_ident.txt")

include(${CTEST_SOURCE_DIRECTORY}/CTestConfig.cmake)
ctest_read_custom_files(${CTEST_SOURCE_DIRECTORY})
set(CTEST_USE_LAUNCHERS TRUE)
if(WIN32)
   set(MAKE_ARGS "-k")
else()
   set(MAKE_ARGS "-j${number_of_processors} -l${number_of_processors} -k")
endif()

if(WIN32)
   if("${compiler}" STREQUAL "MSVC")
      find_program(NINJA ninja)
      find_program(JOM jom)
      if(NINJA)
         set(CTEST_CMAKE_GENERATOR "Ninja")
         set(CMAKE_MAKE_PROGRAM "${NINJA}")
         set(MAKE_ARGS "-k50")
      elseif(JOM)
         set(CTEST_CMAKE_GENERATOR "NMake Makefiles JOM")
         set(CMAKE_MAKE_PROGRAM "${JOM}")
      else()
         set(CTEST_CMAKE_GENERATOR "NMake Makefiles")
         set(CMAKE_MAKE_PROGRAM "nmake")
         set(MAKE_ARGS "-I")
      endif()
   elseif("${compiler}" STREQUAL "MinGW")
      set(CTEST_CMAKE_GENERATOR "MSYS Makefiles")
      set(CMAKE_MAKE_PROGRAM "make")
   else()
      message(FATAL_ERROR "unknown cmake generator required (compiler: ${compiler})")
   endif()
else()
   set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
   set(CMAKE_MAKE_PROGRAM "make")
endif()

set(configure_options "-DCTEST_USE_LAUNCHERS=${CTEST_USE_LAUNCHERS}")
list(APPEND configure_options "-DCMAKE_BUILD_TYPE=${build_type}")
list(APPEND configure_options "-DBUILD_TESTING=TRUE")
list(APPEND configure_options "-DBUILD_EXAMPLES=TRUE")
list(APPEND configure_options "-DUSE_CCACHE=ON")
list(APPEND configure_options "-DCMAKE_INSTALL_PREFIX=${CTEST_BINARY_DIRECTORY}/installed")
if(DEFINED target_architecture)
   list(APPEND configure_options "-DTARGET_ARCHITECTURE=${target_architecture}")
endif()
if(NOT "$ENV{CMAKE_TOOLCHAIN_FILE}" STREQUAL "")
   set(skip_tests TRUE)  # cross-compiling, can't run the tests
   list(APPEND configure_options "-DCMAKE_TOOLCHAIN_FILE=${CTEST_SOURCE_DIRECTORY}/$ENV{CMAKE_TOOLCHAIN_FILE}")
endif()

if("${COMPILER_VERSION}" MATCHES "(GCC|Open64).*4\\.[01234567]\\."
      OR "${COMPILER_VERSION}" MATCHES "GCC 4.8.0"
      OR "${COMPILER_VERSION}" MATCHES "ICC 1[01234567]"
      OR "${COMPILER_VERSION}" MATCHES "clang 3\\.[0123](\\.[0-9])?$")
   message(FATAL_ERROR "Compiler too old for C++11 (${COMPILER_VERSION})")
endif()

if(chip STREQUAL "x86")
   set(arch_abi "x86 32-bit")
elseif(chip STREQUAL "amd64")
   if(CXXFLAGS MATCHES "-m32")
      set(arch_abi "x86 32-bit")
   elseif(CXXFLAGS MATCHES "-mx32")
      set(arch_abi "x86 x32")
   else()
      set(arch_abi "x86 64-bit")
   endif()
else()
   set(arch_abi "${chip}")
endif()

macro(go)
   # On Continuous builds this string may change and thus must be inside go()
   file(STRINGS "${CTEST_SOURCE_DIRECTORY}/Vc/version.h"
      Vc_VERSION_STRING
      REGEX "#define +Vc_VERSION_STRING "
      )
   string(REGEX REPLACE "\"$" "" Vc_VERSION_STRING "${Vc_VERSION_STRING}")
   string(REGEX REPLACE "^.*\"" "" Vc_VERSION_STRING "${Vc_VERSION_STRING}")

   set_property(GLOBAL PROPERTY Label other)
   CTEST_START (${dashboard_model} TRACK "${dashboard_model} ${Vc_VERSION_STRING} ${arch_abi}")
   set(res 0)
   if(NOT is_experimental)
      CTEST_UPDATE (SOURCE "${CTEST_SOURCE_DIRECTORY}" RETURN_VALUE res)
      if(res GREATER 0)
         ctest_submit(PARTS Update)
      endif()
   endif()

   # enter the following section for Continuous builds only if the CTEST_UPDATE above found changes
   if(NOT is_continuous OR res GREATER 0)
      CTEST_CONFIGURE (BUILD "${CTEST_BINARY_DIRECTORY}"
         OPTIONS "${configure_options}"
         APPEND
         RETURN_VALUE res)
      list(APPEND CTEST_NOTES_FILES
         #"${CTEST_BINARY_DIRECTORY}/CMakeFiles/CMakeOutput.log"
         "${CTEST_BINARY_DIRECTORY}/CMakeFiles/CMakeError.log"
         )
      ctest_submit(PARTS Notes Configure)
      unset(CTEST_NOTES_FILES) # less clutter in ctest -V output
      if(res EQUAL 0)
         set(test_results 0)
         if(travis_os OR github_ci)
            set(CTEST_BUILD_COMMAND "${CMAKE_MAKE_PROGRAM} ${MAKE_ARGS}")
            ctest_build(
               BUILD "${CTEST_BINARY_DIRECTORY}"
               APPEND
               RETURN_VALUE res)
            ctest_submit(PARTS Build)
            if(NOT skip_tests)
               ctest_test(
                  BUILD "${CTEST_BINARY_DIRECTORY}"
                  APPEND
                  RETURN_VALUE test_results
                  PARALLEL_LEVEL ${number_of_processors})
            endif()
            ctest_submit(PARTS Test)
         else()
            if("${subset}" STREQUAL "sse")
               set(label_list other Scalar SSE)
            elseif("${subset}" STREQUAL "avx")
               set(label_list AVX AVX2)
            else()
               set(label_list other Scalar SSE AVX AVX2)
            endif()
            foreach(label ${label_list})
               set_property(GLOBAL PROPERTY Label ${label})
               set(CTEST_BUILD_TARGET "${label}")
               set(CTEST_BUILD_COMMAND "${CMAKE_MAKE_PROGRAM} ${MAKE_ARGS} ${CTEST_BUILD_TARGET}")
               ctest_build(
                  BUILD "${CTEST_BINARY_DIRECTORY}"
                  APPEND
                  RETURN_VALUE res)
               ctest_submit(PARTS Build)
               if(res EQUAL 0 AND NOT skip_tests)
                  execute_process(
                     COMMAND ${CMAKE_CTEST_COMMAND} -N -L "^${label}$"
                     WORKING_DIRECTORY "${CTEST_BINARY_DIRECTORY}"
                     OUTPUT_VARIABLE tmp
                     OUTPUT_STRIP_TRAILING_WHITESPACE)
                  if(tmp MATCHES "Total Tests: 0")
                     message("No tests were defined. Skipping tests.")
                  else()
                     ctest_test(
                        BUILD "${CTEST_BINARY_DIRECTORY}"
                        APPEND
                        RETURN_VALUE res
                        PARALLEL_LEVEL ${number_of_processors}
                        INCLUDE_LABEL "^${label}$")
                     ctest_submit(PARTS Test)
                     if(NOT res EQUAL 0)
                        message("ctest_test returned non-zero result: ${res}")
                        set(test_results ${res})
                     endif()
                  endif()
               endif()
               if(label STREQUAL "other" AND CTEST_CMAKE_GENERATOR MATCHES "Make")
                  set(CTEST_BUILD_TARGET "install/fast")
                  set(CTEST_BUILD_COMMAND "${CMAKE_MAKE_PROGRAM} ${MAKE_ARGS} ${CTEST_BUILD_TARGET}")
                  ctest_build(
                     BUILD "${CTEST_BINARY_DIRECTORY}"
                     APPEND
                     RETURN_VALUE res)
                  ctest_submit(PARTS Build)
               endif()
            endforeach()
         endif()
      endif()
   endif()
endmacro()

if(is_continuous)
   while(${CTEST_ELAPSED_TIME} LESS 64800)
      set(START_TIME ${CTEST_ELAPSED_TIME})
      go()
      ctest_sleep(${START_TIME} 1200 ${CTEST_ELAPSED_TIME})
   endwhile()
else()
   if(EXISTS "${CTEST_BINARY_DIRECTORY}")
      CTEST_EMPTY_BINARY_DIRECTORY(${CTEST_BINARY_DIRECTORY})
   endif()
   go()
   if(NOT test_results EQUAL 0)
      message(FATAL_ERROR "One or more tests failed.")
   endif()
endif()
