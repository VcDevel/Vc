macro(build_example name)
   add_executable("example_${name}_default" main.cpp)
   target_link_libraries("example_${name}_default" Vc)

   add_executable("example_${name}_sse" main.cpp)
   add_target_property("example_${name}_sse" COMPILE_FLAGS "-DVC_IMPL=SSE")
   target_link_libraries("example_${name}_sse" Vc)

   add_executable("example_${name}_scalar" main.cpp)
   add_target_property("example_${name}_scalar" COMPILE_FLAGS "-DVC_IMPL=Scalar")

   if(LARRABEE_FOUND)
      add_executable("example_${name}_lrbni" main.cpp)
      add_target_property("example_${name}_lrbni" COMPILE_FLAGS "-DVC_IMPL=LRBni")
   endif(LARRABEE_FOUND)
endmacro(build_example)

build_example(matrix)
