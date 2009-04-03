cmake_minimum_required(VERSION 2.6.2)
project(vector)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(Larrabee)
if(LARRABEE_FOUND)
   include_directories(${LRB_HOST_INCLUDE_DIR})
   add_definitions(-DENABLE_LARRABEE)
endif(LARRABEE_FOUND)

macro(add_target_property _target _prop _value)
   get_target_property(_oldprop "${_target}" ${_prop})
   if(NOT _oldprop)
      set_target_properties("${_target}" PROPERTIES ${_prop} "${_value}")
   else(NOT _oldprop)
      set_target_properties("${_target}" PROPERTIES ${_prop} "${_oldprop} ${_value}")
   endif(NOT _oldprop)
endmacro(add_target_property)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
   set(CMAKE_C_FLAGS "-W -Wall -Wimplicit -Wswitch -Wformat -Wchar-subscripts -Wparentheses -Wmultichar -Wtrigraphs -Wpointer-arith -Wcast-align -Wreturn-type -Wno-unused-function -ansi -pedantic -Wno-long-long -Wno-variadic-macros -msse -msse2")
   set(CMAKE_CXX_FLAGS "-W -Wall -Wimplicit -Wswitch -Wformat -Wchar-subscripts -Wparentheses -Wmultichar -Wtrigraphs -Wpointer-arith -Wcast-align -Wreturn-type -Wno-unused-function -ansi -pedantic -Wno-long-long -Wno-variadic-macros -msse -msse2")
   set(CMAKE_CXX_FLAGS_DEBUG          "${CMAKE_CXX_FLAGS_DEBUG} -DTBB_DO_ASSERT=1")
   set(CMAKE_CXX_FLAGS_MINSIZEREL     "${CMAKE_CXX_FLAGS_MINSIZEREL} ")
   set(CMAKE_CXX_FLAGS_RELEASE        "${CMAKE_CXX_FLAGS_RELEASE} ")
   set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -DNDEBUG")
   set(CMAKE_C_FLAGS_DEBUG          "${CMAKE_C_FLAGS_DEBUG} -DTBB_DO_ASSERT=1")
   set(CMAKE_C_FLAGS_MINSIZEREL     "${CMAKE_C_FLAGS_MINSIZEREL} ")
   set(CMAKE_C_FLAGS_RELEASE        "${CMAKE_C_FLAGS_RELEASE} ")
   set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -DNDEBUG")
endif(CMAKE_SYSTEM_NAME STREQUAL "Linux")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(vector STATIC vector.cpp)

add_subdirectory(tests)
enable_testing()

# install dir is not interesting for now
mark_as_advanced(CMAKE_INSTALL_PREFIX)
