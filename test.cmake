Set(CTEST_SOURCE_DIRECTORY $ENV{VC_SOURCEDIR})
Set(CTEST_BINARY_DIRECTORY $ENV{VC_BUILDDIR})
Set(CTEST_SITE $ENV{SITE})
Set(CTEST_BUILD_NAME $ENV{LABEL})
Set(CTEST_START_WITH_EMPTY_BINARY_DIRECTORY_ONCE TRUE)
set(CTEST_NOTES_FILES ${CTEST_SOURCE_DIRECTORY}/.git/HEAD ${CTEST_SOURCE_DIRECTORY}/.git/refs/heads/$ENV{git_branch})

include(CTestCustom.cmake)
include(CTestConfig.cmake)
set(CTEST_USE_LAUNCHERS 1) # much improved error/warning message logging

if(WIN32)
   find_program(JOM jom)
   if(JOM)
      set(CTEST_CMAKE_GENERATOR "NMake Makefiles JOM")
      set(CMAKE_MAKE_PROGRAM "jom -j$ENV{number_of_processors} -i")
   else()
      set(CTEST_CMAKE_GENERATOR "NMake Makefiles")
      set(CMAKE_MAKE_PROGRAM "nmake -I")
   endif()
else()
   set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
   if(NOT CMAKE_MAKE_PROGRAM)
      set(CMAKE_MAKE_PROGRAM "make -j$ENV{number_of_processors} -i")
   endif()
endif()

macro(go)
   CTEST_START ($ENV{ctest_model})
   set(res 0)
   if(NOT $ENV{ctest_model} STREQUAL "Experimental")
      CTEST_UPDATE (SOURCE "${CTEST_SOURCE_DIRECTORY}" RETURN_VALUE res)
      if(res GREATER 0)
         ctest_submit(PARTS Update)
      endif()
   endif()
   if(NOT $ENV{ctest_model} STREQUAL "Continuous" OR res GREATER 0)
      CTEST_CONFIGURE (BUILD "${CTEST_BINARY_DIRECTORY}"
         OPTIONS "-DCTEST_USE_LAUNCHERS=${CTEST_USE_LAUNCHERS} -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=TRUE"
         APPEND
         RETURN_VALUE res)
      ctest_submit(PARTS Notes Configure)
      if(res EQUAL 0)
         foreach(subproject ${CTEST_PROJECT_SUBPROJECTS})
            set_property(GLOBAL PROPERTY SubProject ${subproject})
            set_property(GLOBAL PROPERTY Label ${subproject})
            set(CTEST_BUILD_TARGET "${subproject}")
            set(CTEST_BUILD_COMMAND "${CMAKE_MAKE_PROGRAM} ${CTEST_BUILD_TARGET}")
            ctest_build(
               BUILD "${CTEST_BINARY_DIRECTORY}"
               APPEND
               RETURN_VALUE res)
            ctest_submit(PARTS Build)
            if(res EQUAL 0)
               ctest_test(
                  BUILD "${CTEST_BINARY_DIRECTORY}"
                  APPEND
                  RETURN_VALUE res
                  PARALLEL_LEVEL $ENV{number_of_processors}
                  INCLUDE_LABEL "${subproject}")
               ctest_submit(PARTS Test)
            endif()
         endforeach()
      endif()
   endif()
endmacro()

if($ENV{ctest_model} STREQUAL "Continuous")
   while(${CTEST_ELAPSED_TIME} LESS 64800)
      set(START_TIME ${CTEST_ELAPSED_TIME})
      go()
      ctest_sleep(${START_TIME} 1200 ${CTEST_ELAPSED_TIME})
   endwhile()
else()
   CTEST_EMPTY_BINARY_DIRECTORY(${CTEST_BINARY_DIRECTORY})
   go()
endif()
