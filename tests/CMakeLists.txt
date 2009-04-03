add_definitions(-DCOMPILE_FOR_UNIT_TESTS)

macro(vc_add_test name)
   add_executable("${name}_simple" ${name}.cpp)
   add_test("${name}_simple" "${CMAKE_CURRENT_BINARY_DIR}/${name}_simple")

   add_executable("${name}_sse" ${name}.cpp)
   add_target_property("${name}_sse" COMPILE_FLAGS "-DUSE_SSE")
   add_test("${name}_sse" "${CMAKE_CURRENT_BINARY_DIR}/${name}_sse")
endmacro(vc_add_test)

vc_add_test(arithmetics)
vc_add_test(load)
vc_add_test(store)
vc_add_test(gather)
vc_add_test(scatter)

enable_testing()
