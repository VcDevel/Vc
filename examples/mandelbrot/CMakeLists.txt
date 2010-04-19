find_package(Qt4)
if(QT4_FOUND)
   include(${QT_USE_FILE})
   include_directories(${CMAKE_CURRENT_BINARY_DIR})
   qt4_generate_moc(main.h moc_main.cpp)
   qt4_generate_moc(mandel.h moc_mandel.cpp)

   set(SOURCES main.cpp moc_main.cpp mandel.cpp moc_mandel.cpp)

   add_executable(mandelbrot_sse ${SOURCES})
   add_target_property(mandelbrot_sse COMPILE_FLAGS "-DVC_IMPL=SSE")
   target_link_libraries(mandelbrot_sse ${QT_LIBRARIES} Vc)

   add_executable(mandelbrot_scalar ${SOURCES})
   add_target_property(mandelbrot_scalar COMPILE_FLAGS "-DVC_IMPL=Scalar")
   target_link_libraries(mandelbrot_scalar ${QT_LIBRARIES} Vc)
endif(QT4_FOUND)
