add_custom_target(benchmarks COMMENT "Run all benchmarks" VERBATIM)
add_target_property(benchmarks EXCLUDE_FROM_DEFAULT_BUILD 1)
set(ENABLE_BENCHMARKS FALSE)
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo" OR MSVC)
   set(ENABLE_BENCHMARKS TRUE)
endif(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo" OR MSVC)

if(ENABLE_BENCHMARKS)
   add_custom_target(benchmark_data COMMENT "Create data for later processing for all benchmarks" VERBATIM)
   add_target_property(benchmark_data EXCLUDE_FROM_DEFAULT_BUILD 1)
endif(ENABLE_BENCHMARKS)

macro(vc_generate_datafile target outfilepath)
   get_target_property(exec ${target} OUTPUT_NAME)
   set(outfile "${exec}.dat")
   set(${outfilepath} "${CMAKE_CURRENT_BINARY_DIR}/${exec}.dat")
   add_custom_command(OUTPUT "${${outfilepath}}"
      COMMAND ${target}
      ARGS -o "${${outfilepath}}"
      DEPENDS ${target}
      COMMENT "Running Benchmark ${exec} to generate ${outfile}"
      VERBATIM
      )
endmacro(vc_generate_datafile)

macro(vc_generate_plots name)
   if(ENABLE_BENCHMARKS)
      set(dataFilePaths)

      set(lrb)
      if(LARRABEE_FOUND)
         set(lrb "lrb")
      endif(LARRABEE_FOUND)

      foreach(def "scalar" "sse" ${lrb} ${ARGN})
         set(_t "${name}_${def}_benchmark")
         vc_generate_datafile(${_t} dfp)
         set(dataFilePaths ${dataFilePaths} ${dfp})
      endforeach(def)

      set(generate_target "generate_${name}_data")
      add_custom_target(${generate_target} DEPENDS ${dataFilePaths} "${scriptfile}")
      add_dependencies(benchmark_data "${generate_target}")
   endif(ENABLE_BENCHMARKS)
endmacro(vc_generate_plots)

macro(vc_add_benchmark name)
   set(LIBS cpuset)
   if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      set(LIBS ${LIBS} rt)
   endif(CMAKE_SYSTEM_NAME STREQUAL "Linux")

   set(lrb)
   if(LARRABEE_FOUND)
      set(lrb "lrb")
   endif(LARRABEE_FOUND)

   foreach(def "scalar" "sse" ${lrb} ${ARGN})
      set(exec "${name}_${def}")
      set(target "${name}_${def}_benchmark")

      add_executable(${target} ${name}.cpp)
      target_link_libraries(${target} Vc ${LIBS})
      add_target_property(${target} OUTPUT_NAME ${exec})
      if(def STREQUAL "scalar")
         add_target_property(${target} COMPILE_FLAGS "-DVC_IMPL=Scalar")
      elseif(def STREQUAL "sse")
         add_target_property(${target} COMPILE_FLAGS "-DVC_IMPL=SSE")
      elseif(def STREQUAL "lrb")
         add_target_property(${target} COMPILE_FLAGS "-DVC_IMPL=LRBni")
      else(def STREQUAL "scalar")
         add_target_property(${target} COMPILE_FLAGS "-D${def}")
      endif(def STREQUAL "scalar")
      add_custom_target("run_${target}" ${target} DEPENDS ${target} COMMENT "Running ${exec}")
      add_target_property("run_${target}" EXCLUDE_FROM_DEFAULT_BUILD 1)
      add_dependencies(benchmarks "run_${target}")
   endforeach(def)
endmacro(vc_add_benchmark)
