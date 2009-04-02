add_definitions(-DCOMPILE_FOR_UNIT_TESTS)

macro(vc_add_test name)
   add_executable("${name}_simple" ${name}.cpp)
   add_test("${name}_simple" "${CMAKE_CURRENT_BINARY_DIR}/${name}")

   add_executable("${name}_sse" ${name}.cpp)
   add_target_property("${name}_sse" COMPILE_FLAGS "-DUSE_SSE")
   add_test("${name}_sse" "${CMAKE_CURRENT_BINARY_DIR}/${name}_sse")
endmacro(vc_add_test)

vc_add_test(arithmetics)
