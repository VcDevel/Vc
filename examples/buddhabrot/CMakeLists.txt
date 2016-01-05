if(QT4_FOUND AND QT4_USABLE)
   include(${QT_USE_FILE})

   build_example(buddhabrot main.cpp LIBS ${QT_LIBRARIES})

   # Special -DScalar target that uses fundamental types directly
   add_executable(example_buddhabrot_scalar2 main.cpp)
   add_target_property(example_buddhabrot_scalar2 COMPILE_DEFINITIONS "Scalar")
   set_property(TARGET example_buddhabrot_scalar2 APPEND PROPERTY COMPILE_OPTIONS ${Vc_ARCHITECTURE_FLAGS})
   add_target_property(example_buddhabrot_scalar2 LABELS "Scalar")
   add_dependencies(Scalar example_buddhabrot_scalar2)
   add_dependencies(Examples example_buddhabrot_scalar2)
   target_link_libraries(example_buddhabrot_scalar2 Vc ${QT_LIBRARIES})
   vc_add_run_target(example_buddhabrot_scalar2)
endif()
