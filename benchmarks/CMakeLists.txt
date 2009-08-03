add_definitions(-DCOMPILE_FOR_BENCHMARKS)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
   set(LIBS rt)
endif(CMAKE_SYSTEM_NAME STREQUAL "Linux")

set(BENCHMARK_COMMANDS "")
set(ALL_BENCHMARK_TARGETS "")

macro(vc_add_benchmark name)
   set(simpleExec "${name}_simple")
   set(sseExec "${name}_sse")
   set(lrbExec "${name}_lrb")

   set(simpleTarget "${name}_simple_benchmark")
   set(sseTarget "${name}_sse_benchmark")
   set(lrbTarget "${name}_lrb_benchmark")

   add_executable(${simpleTarget} ${name}.cpp)
   add_target_property(${simpleTarget} OUTPUT_NAME ${simpleExec})
   target_link_libraries(${simpleTarget} Vc ${LIBS})
   add_custom_target("run_${simpleTarget}" ${simpleTarget} DEPENDS ${simpleTarget} COMMENT "Running ${simpleExec}")

   add_executable(${sseTarget} ${name}.cpp)
   add_target_property(${sseTarget} OUTPUT_NAME ${sseExec})
   target_link_libraries(${sseTarget} Vc ${LIBS})
   add_target_property(${sseTarget} COMPILE_FLAGS "-DUSE_SSE")
   add_custom_target("run_${sseTarget}" ${sseTarget} DEPENDS ${sseTarget} COMMENT "Running ${sseExec}")

   set(ALL_BENCHMARK_TARGETS ${ALL_BENCHMARK_TARGETS} "run_${simpleTarget}" "run_${sseTarget}")

   if(LARRABEE_FOUND)
      add_executable(${lrbTarget} ${name}.cpp)
      add_target_property(${lrbTarget} OUTPUT_NAME ${lrbExec})
      target_link_libraries(${lrbTarget} Vc ${LIBS} ${LRB_LIBS})
      add_target_property(${lrbTarget} COMPILE_FLAGS "-DENABLE_LARRABEE")

      add_custom_target("run_${lrbTarget}" ${lrbTarget} DEPENDS ${lrbTarget} COMMENT "Running ${lrbExec}")

      set(ALL_BENCHMARK_TARGETS ${ALL_BENCHMARK_TARGETS} "run_${lrbTarget}")
   endif(LARRABEE_FOUND)
endmacro(vc_add_benchmark)

vc_add_benchmark(flops)
vc_add_benchmark(gather)
vc_add_benchmark(mask)

add_custom_target(benchmark
   COMMENT "Run all benchmarks"
   VERBATIM
   )

add_dependencies(benchmark ${ALL_BENCHMARK_TARGETS})
