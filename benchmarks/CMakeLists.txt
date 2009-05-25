add_definitions(-DCOMPILE_FOR_BENCHMARKS)
find_package(TBB)
if(TBB_FOUND)
   include_directories(${TBB_INCLUDE_DIRS})
endif(TBB_FOUND)

macro(vc_add_benchmark name)
   set(simpleTarget "${name}_simple")
   set(sseTarget "${name}_sse")
   set(lrbTarget "${name}_lrb")

   add_executable(${simpleTarget} ${name}.cpp)
   target_link_libraries(${simpleTarget} vector rt ${TBB_RELEASE_LIBRARIES})

   add_executable(${sseTarget} ${name}.cpp)
   target_link_libraries(${sseTarget} vector rt ${TBB_RELEASE_LIBRARIES})
   add_target_property(${sseTarget} COMPILE_FLAGS "-DUSE_SSE")

   add_executable(${lrbTarget} ${name}.cpp)
   target_link_libraries(${lrbTarget} vector rt ${TBB_RELEASE_LIBRARIES})
   add_target_property(${lrbTarget} COMPILE_FLAGS "-DENABLE_LARRABEE")# -fno-strict-aliasing")
endmacro(vc_add_benchmark)

vc_add_benchmark(flops)
