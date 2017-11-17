find_package(Qt4)
set(SAFE_CMAKE_REQUIRED_INCLUDES "${CMAKE_REQUIRED_INCLUDES}")
set(SAFE_CMAKE_REQUIRED_LIBRARIES "${CMAKE_REQUIRED_LIBRARIES}")
set(CMAKE_REQUIRED_INCLUDES "${QT_INCLUDES}")
set(CMAKE_REQUIRED_LIBRARIES "${QT_QTCORE_LIBRARY}")
CHECK_CXX_SOURCE_COMPILES("#include <QObject>
int main() { QObject o; return 0;}" QT4_USABLE)
mark_as_advanced(QT4_USABLE)
set(CMAKE_REQUIRED_INCLUDES "${SAFE_CMAKE_REQUIRED_INCLUDES}")
set(CMAKE_REQUIRED_LIBRARIES "${SAFE_CMAKE_REQUIRED_LIBRARIES}")

add_custom_target(Examples COMMENT "build all examples" VERBATIM)

AddCompilerFlag(-ftemplate-depth=1024 CXX_FLAGS CMAKE_CXX_FLAGS MIC_CXX_FLAGS CMAKE_MIC_CXX_FLAGS)

macro(vc_add_run_target _target)
   if("${_target}" MATCHES "_mic$")
      if(MIC_NATIVELOAD)
         get_target_property(_exe "${_target}" OUTPUT_NAME)
         add_custom_target(run_${_target}
            ${MIC_NATIVELOAD} "${_exe}"
            DEPENDS ${_target}
            COMMENT "Execute ${_target} example"
            VERBATIM
            )
      endif()
   else()
      add_custom_target(run_${_target}
         ${_target}
         DEPENDS ${_target}
         COMMENT "Execute ${_target} example"
         VERBATIM
         )
   endif()
endmacro()

macro(_build_one_example_target _name _impl)
   set(_target "example_${_name}_${_impl}")
   string(TOLOWER "${_target}" _target)
   list(FIND _disabled_impl "${_impl}" _index1)
   list(FIND disabled_targets "${_target}" _index2)
   if(_index1 EQUAL -1 AND USE_${_impl} AND _index2 EQUAL -1)
      add_executable(${_target} ${ARGN})
      add_target_property(${_target} COMPILE_DEFINITIONS "Vc_IMPL=${_impl}")
      set_property(TARGET ${_target} APPEND PROPERTY COMPILE_OPTIONS ${Vc_ARCHITECTURE_FLAGS})
      add_target_property(${_target} LABELS "${_impl}")
      add_dependencies(${_impl} ${_target})
      add_dependencies(Examples ${_target})
      target_link_libraries(${_target} Vc ${_LIBS})
      vc_add_run_target(${_target})
   endif()
endmacro()

function(build_example name)
   set(_SRCS)
   set(_LIBS)
   set(_disabled_impl)
   set(_state 1)
   set(USE_Scalar TRUE)
   set(USE_SSE ${USE_SSE2})
   foreach(ARG ${ARGN})
      if(ARG STREQUAL "LIBS")
         set(_state 2)
      elseif(ARG STREQUAL "DISABLE")
         set(_state 3)
      elseif(_state EQUAL 1)
         set(_SRCS ${_SRCS} ${ARG})
      elseif(_state EQUAL 2)
         set(_LIBS ${_LIBS} ${ARG})
      elseif(_state EQUAL 3)
         list(APPEND _disabled_impl ${ARG})
      endif()
   endforeach()

   _build_one_example_target("${name}" Scalar ${_SRCS})
   _build_one_example_target("${name}" SSE ${_SRCS})
   _build_one_example_target("${name}" AVX ${_SRCS})
   _build_one_example_target("${name}" AVX2 ${_SRCS})

   set(_target "example_${name}_mic")
   list(FIND _disabled_impl "MIC" _index1)
   list(FIND disabled_targets "${_target}" _index2)
   if(MIC_NATIVE_FOUND AND "${_LIBS}" STREQUAL "" AND _index1 EQUAL -1 AND _index2 EQUAL -1)
      mic_add_executable(${_target}
         SOURCES ${_SRCS}
         LINK_LIBRARIES Vc_MIC
         )
      add_target_property(${_target} LABELS "MIC")
      add_dependencies(MIC ${_target})
      vc_add_run_target(${_target})
   endif()
endfunction(build_example)

file(GLOB examples RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/*/CMakeLists.txt")
foreach(example ${examples})
   string(REPLACE "/CMakeLists.txt" "" example "${example}")
   list(FIND disabled_targets "example_${example}" _disabled)
   if(_disabled EQUAL -1)
      add_subdirectory(${example})
   endif()
endforeach()
