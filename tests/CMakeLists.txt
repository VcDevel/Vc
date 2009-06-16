add_definitions(-DCOMPILE_FOR_UNIT_TESTS)

macro(vc_add_test name)
   set(simpleTarget "${name}_simple")
   set(sseTarget "${name}_sse")
   set(lrbTarget "${name}_lrb")

   add_executable(${simpleTarget} ${name}.cpp)
   target_link_libraries(${simpleTarget} vector)
   add_test(${simpleTarget} "${CMAKE_CURRENT_BINARY_DIR}/${simpleTarget}")

   add_executable(${sseTarget} ${name}.cpp)
   target_link_libraries(${sseTarget} vector)
   add_target_property(${sseTarget} COMPILE_FLAGS "-DUSE_SSE")
   add_test(${sseTarget} "${CMAKE_CURRENT_BINARY_DIR}/${sseTarget}")

   if(LARRABEE_FOUND)
      add_executable(${lrbTarget} ${name}.cpp)
      target_link_libraries(${lrbTarget} vector)
      add_target_property(${lrbTarget} COMPILE_FLAGS "-DENABLE_LARRABEE")
      add_test(${lrbTarget} "${CMAKE_CURRENT_BINARY_DIR}/${lrbTarget}")
   endif(LARRABEE_FOUND)
endmacro(vc_add_test)

vc_add_test(arithmetics)
add_target_property(arithmetics_sse COMPILE_FLAGS "-DUSE_CORRECT_UNSIGNED_COMPARE")

vc_add_test(expandandmerge)
vc_add_test(load)
vc_add_test(store)
vc_add_test(gather)
vc_add_test(scatter)
vc_add_test(math)
vc_add_test(mask)

enable_testing()
