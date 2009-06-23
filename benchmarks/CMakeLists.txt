add_definitions(-DCOMPILE_FOR_BENCHMARKS)

macro(vc_add_benchmark name)
   set(simpleTarget "${name}_simple")
   set(sseTarget "${name}_sse")
   set(lrbTarget "${name}_lrb")

   add_executable(${simpleTarget} ${name}.cpp)
   target_link_libraries(${simpleTarget} Vc rt)

   add_executable(${sseTarget} ${name}.cpp)
   target_link_libraries(${sseTarget} Vc rt)
   add_target_property(${sseTarget} COMPILE_FLAGS "-DUSE_SSE")

   if(LARRABEE_FOUND)
      add_executable(${lrbTarget} ${name}.cpp)
      target_link_libraries(${lrbTarget} Vc rt)
      add_target_property(${lrbTarget} COMPILE_FLAGS "-DENABLE_LARRABEE")
   endif(LARRABEE_FOUND)
endmacro(vc_add_benchmark)

vc_add_benchmark(flops)
