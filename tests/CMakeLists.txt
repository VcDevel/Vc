add_definitions(-DCOMPILE_FOR_UNIT_TESTS)

macro(vc_add_test name)
   set(simpleTarget "${name}_simple")
   set(sseTarget "${name}_sse")

   add_executable(${simpleTarget} ${name}.cpp)
   target_link_libraries(${simpleTarget} vector)
   add_test(${simpleTarget} "${CMAKE_CURRENT_BINARY_DIR}/${simpleTarget}")

   add_executable(${sseTarget} ${name}.cpp)
   target_link_libraries(${sseTarget} vector)
   add_target_property(${sseTarget} COMPILE_FLAGS "-DUSE_SSE")
   add_test(${sseTarget} "${CMAKE_CURRENT_BINARY_DIR}/${sseTarget}")
endmacro(vc_add_test)

vc_add_test(arithmetics)
vc_add_test(load)
vc_add_test(store)
vc_add_test(gather)
vc_add_test(scatter)
#vc_add_test(pack)

enable_testing()
