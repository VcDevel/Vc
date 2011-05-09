if(QT4_FOUND)
   include(${QT_USE_FILE})
   include_directories(${CMAKE_CURRENT_BINARY_DIR})
   qt4_generate_moc(main.h moc_main.cpp)
   qt4_generate_moc(mandel.h moc_mandel.cpp)

   set(SOURCES main.cpp ${CMAKE_CURRENT_BINARY_DIR}/moc_main.cpp mandel.cpp
      ${CMAKE_CURRENT_BINARY_DIR}/moc_mandel.cpp)

   add_executable(mandelbrot_sse ${SOURCES})
   add_target_property(mandelbrot_sse COMPILE_FLAGS "-DVC_IMPL=SSE")
   target_link_libraries(mandelbrot_sse ${QT_LIBRARIES} Vc)

   add_executable(mandelbrot_scalar ${SOURCES})
   add_target_property(mandelbrot_scalar COMPILE_FLAGS "-DVC_IMPL=Scalar")
   target_link_libraries(mandelbrot_scalar ${QT_LIBRARIES} Vc)

   # It is an interesting test if we can compare against the autovect
   # capabilities of Open64 and ICC, so we try to find those and compile extra
   # binaries with them
   find_program(O64_CXX openCC HINTS /opt/x86_open64-4.2.4/bin /opt/open64/bin)
   find_program(ICC_CXX icpc HINTS
      $ENV{HOME}/intel/Compiler/11.1/072/bin/intel64)

   if(O64_CXX)
      add_custom_command(OUTPUT mandelbrot_open64
         COMMAND ${O64_CXX} -O3 -Wall -msse3
         -o ${CMAKE_CURRENT_BINARY_DIR}/mandelbrot_open64
         -I ${CMAKE_CURRENT_BINARY_DIR} -I ${QT_INCLUDE_DIR}
         -I ${QT_QTCORE_INCLUDE_DIR} -I ${QT_QTGUI_INCLUDE_DIR}
         -I ${CMAKE_SOURCE_DIR} -I ${CMAKE_SOURCE_DIR}/include
         -L ${QT_LIBRARY_DIR} -lQtGui ${SOURCES}
         -DVC_IMPL=Scalar
         DEPENDS ${SOURCES}
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
         COMMENT "Bulding mandelbrot_open64"
         VERBATIM)
      add_custom_target(build_mandelbrot_open64 ALL DEPENDS mandelbrot_open64)
      add_target_property(build_mandelbrot_open64 LABELS "other")
      add_dependencies(other build_mandelbrot_open64)
   endif(O64_CXX)

   if(ICC_CXX)
      add_custom_command(OUTPUT mandelbrot_icc
         COMMAND ${ICC_CXX} -O3 -xSSE3
         -o ${CMAKE_CURRENT_BINARY_DIR}/mandelbrot_icc
         -I ${CMAKE_CURRENT_BINARY_DIR} -I ${QT_INCLUDE_DIR}
         -I ${QT_QTCORE_INCLUDE_DIR} -I ${QT_QTGUI_INCLUDE_DIR}
         -I ${CMAKE_SOURCE_DIR} -I ${CMAKE_SOURCE_DIR}/include
         -L ${QT_LIBRARY_DIR} -lQtGui ${SOURCES}
         -DVC_IMPL=Scalar
         DEPENDS ${SOURCES}
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
         COMMENT "Bulding mandelbrot_icc"
         VERBATIM)
      add_custom_target(build_mandelbrot_icc ALL DEPENDS mandelbrot_icc)
      add_target_property(build_mandelbrot_icc LABELS "other")
      add_dependencies(other build_mandelbrot_icc)
   endif(ICC_CXX)
endif(QT4_FOUND)
