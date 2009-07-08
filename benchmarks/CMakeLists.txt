add_definitions(-DCOMPILE_FOR_BENCHMARKS)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
   set(LIBS rt)
endif(CMAKE_SYSTEM_NAME STREQUAL "Linux")

set(BENCHMARK_COMMANDS "")
set(ALL_BENCHMARK_TARGETS "")

macro(vc_add_benchmark name)
   set(simpleTarget "${name}_simple")
   set(sseTarget "${name}_sse")
   set(lrbTarget "${name}_lrb")

   add_executable(${simpleTarget} ${name}.cpp)
   target_link_libraries(${simpleTarget} Vc ${LIBS})
   add_custom_target("run_${simpleTarget}" ${simpleTarget} DEPENDS ${simpleTarget} COMMENT "Running ${simpleTarget}")

   add_executable(${sseTarget} ${name}.cpp)
   target_link_libraries(${sseTarget} Vc ${LIBS})
   add_target_property(${sseTarget} COMPILE_FLAGS "-DUSE_SSE")
   add_custom_target("run_${sseTarget}" ${sseTarget} DEPENDS ${sseTarget} COMMENT "Running ${sseTarget}")

   set(ALL_BENCHMARK_TARGETS ${ALL_BENCHMARK_TARGETS} "run_${simpleTarget}" "run_${sseTarget}")

   if(LARRABEE_FOUND)
      add_executable(${lrbTarget} ${name}.cpp)
      target_link_libraries(${lrbTarget} Vc ${LIBS} ${LRB_LIBS})
      add_target_property(${lrbTarget} COMPILE_FLAGS "-DENABLE_LARRABEE")

      add_custom_target("run_${lrbTarget}" ${lrbTarget} DEPENDS ${lrbTarget} COMMENT "Running ${lrbTarget}")

      set(ALL_BENCHMARK_TARGETS ${ALL_BENCHMARK_TARGETS} "run_${lrbTarget}")
   endif(LARRABEE_FOUND)
endmacro(vc_add_benchmark)

vc_add_benchmark(flops)

message(STATUS "${ALL_BENCHMARK_TARGETS}")
add_custom_target(benchmark
   COMMENT "Run all benchmarks"
   VERBATIM
   )

add_dependencies(benchmark ${ALL_BENCHMARK_TARGETS})
