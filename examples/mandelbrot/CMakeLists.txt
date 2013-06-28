if(QT4_FOUND AND QT4_USABLE)
   include(${QT_USE_FILE})
   include_directories(${CMAKE_CURRENT_BINARY_DIR})
   qt4_generate_moc(main.h moc_main.cpp)
   qt4_generate_moc(mandel.h moc_mandel.cpp)

   set(SOURCES main.cpp ${CMAKE_CURRENT_BINARY_DIR}/moc_main.cpp mandel.cpp
      ${CMAKE_CURRENT_BINARY_DIR}/moc_mandel.cpp)

   build_example(mandelbrot ${SOURCES} LIBS ${QT_LIBRARIES})
endif()
