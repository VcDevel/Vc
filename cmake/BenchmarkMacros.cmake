
find_program(R_COMMAND "R" DOC "http://www.r-project.org/")
find_program(PDFTK_COMMAND "pdftk" DOC "http://www.accesspdf.com/pdftk/")
mark_as_advanced(R_COMMAND PDFTK_COMMAND)

add_custom_target(benchmarks COMMENT "Run all benchmarks" VERBATIM)
add_target_property(benchmarks EXCLUDE_FROM_DEFAULT_BUILD 1)
set(ENABLE_BENCHMARKS FALSE)
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo" OR MSVC)
   set(ENABLE_BENCHMARKS TRUE)
endif(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo" OR MSVC)

if(ENABLE_BENCHMARKS)
   if(R_COMMAND)
      add_custom_target(benchmark_plots COMMENT "Create PDF plots for all benchmarks" VERBATIM)
      add_target_property(benchmark_plots EXCLUDE_FROM_DEFAULT_BUILD 1)
   else(R_COMMAND)
      add_custom_target(benchmark_data COMMENT "Create data for later processing for all benchmarks" VERBATIM)
      add_target_property(benchmark_data EXCLUDE_FROM_DEFAULT_BUILD 1)
   endif(R_COMMAND)
endif(ENABLE_BENCHMARKS)

macro(vc_generate_datafile target outfilepath outfile)
   get_target_property(exec ${target} OUTPUT_NAME)
   set(${outfile} "${exec}.dat")
   set(${outfilepath} "${CMAKE_CURRENT_BINARY_DIR}/${exec}.dat")
   add_custom_command(OUTPUT "${${outfilepath}}"
      COMMAND ${target}
      ARGS -o "${${outfilepath}}"
      DEPENDS ${target}
      COMMENT "Running Benchmark ${exec} to generate ${${outfile}}"
      VERBATIM
      )
endmacro(vc_generate_datafile)

macro(vc_generate_plots name)
   if(ENABLE_BENCHMARKS)
      set(simpleTarget "${name}_simple_benchmark")
      set(sseTarget "${name}_sse_benchmark")
      set(lrbTarget "${name}_lrb_benchmark")

      vc_generate_datafile(${simpleTarget} simple_datafilepath simple_datafile)
      vc_generate_datafile(${sseTarget} sse_datafilepath sse_datafile)
      if(LARRABEE_FOUND)
         vc_generate_datafile(${lrbTarget} lrb_datafilepath lrb_datafile)
      else(LARRABEE_FOUND)
         set(lrb_datafile "")
         set(lrb_datafilepath "")
      endif(LARRABEE_FOUND)

      set(scriptfile "${CMAKE_CURRENT_BINARY_DIR}/plot_${name}.r")
      add_custom_command(OUTPUT "${scriptfile}"
         COMMAND "${CMAKE_COMMAND}"
         ARGS
         "-Dscriptfile=${scriptfile}"
         "-Dcommon=${CMAKE_CURRENT_SOURCE_DIR}/common.r"
         "-Dappend=${CMAKE_CURRENT_SOURCE_DIR}/${name}.r"
         "-Dsimple_datafile=${simple_datafile}"
         "-Dsse_datafile=${sse_datafile}"
         "-Dlrb_datafile=${lrb_datafile}"
         -P "${CMAKE_SOURCE_DIR}/cmake/generate_plot_script.cmake"
         DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/common.r" "${CMAKE_CURRENT_SOURCE_DIR}/${name}.r"
         VERBATIM
         )

      if(R_COMMAND)
         set(tmpfile "${CMAKE_CURRENT_BINARY_DIR}/${name}_tmp.pdf")
         set(pdffile "${CMAKE_CURRENT_BINARY_DIR}/${name}.pdf")
         if(NOT PDFTK_COMMAND)
            set(tmpfile "${pdffile}")
         endif(NOT PDFTK_COMMAND)
         add_custom_command(OUTPUT "${tmpfile}"
            COMMAND "${R_COMMAND}" ARGS --quiet --slave --vanilla -f "${scriptfile}"
            COMMAND "${CMAKE_COMMAND}" ARGS -E copy "Rplots.pdf" "${tmpfile}"
            COMMAND "${CMAKE_COMMAND}" ARGS -E remove "Rplots.pdf"
            DEPENDS "${simple_datafilepath}" "${sse_datafilepath}" "${lrb_datafilepath}" "${CMAKE_CURRENT_SOURCE_DIR}/common.r" "${CMAKE_CURRENT_SOURCE_DIR}/${name}.r" "${scriptfile}"
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
            COMMENT "Generating PDF plots for the ${name} benchmark"
            VERBATIM
            )
         if(PDFTK_COMMAND)
            set(metafile "${CMAKE_CURRENT_BINARY_DIR}/${name}.meta")
            add_custom_command(OUTPUT "${pdffile}"
               COMMAND "${PDFTK_COMMAND}" ARGS "${tmpfile}" dump_data output "${metafile}"
               COMMAND "${CMAKE_COMMAND}" ARGS -D metafile=${metafile} -D name=${name} -P "${CMAKE_CURRENT_SOURCE_DIR}/processPdfMetaData.cmake"
               COMMAND "${PDFTK_COMMAND}" ARGS "${tmpfile}" update_info "${metafile}" output "${pdffile}"
               COMMAND "${CMAKE_COMMAND}" ARGS -E remove "${tmpfile}"
               DEPENDS "${tmpfile}"
               COMMENT "Adjusting PDF Title of ${pdffile}"
               VERBATIM
               )
         endif(PDFTK_COMMAND)

         set(generate_target "generate_${name}_plot")
         add_custom_target(${generate_target} DEPENDS "${pdffile}")
         add_dependencies(benchmark_plots "${generate_target}")
      else(R_COMMAND)
         set(generate_target "generate_${name}_data")
         add_custom_target(${generate_target} DEPENDS "${simple_datafilepath}" "${sse_datafilepath}" "${lrb_datafilepath}" "${scriptfile}")
         add_dependencies(benchmark_data "${generate_target}")
      endif(R_COMMAND)
   endif(ENABLE_BENCHMARKS)
endmacro(vc_generate_plots)

macro(vc_add_benchmark name)
   if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      set(LIBS rt)
   endif(CMAKE_SYSTEM_NAME STREQUAL "Linux")

   set(simpleExec "${name}_simple")
   set(sseExec "${name}_sse")
   set(lrbExec "${name}_lrb")

   set(simpleTarget "${name}_simple_benchmark")
   set(sseTarget "${name}_sse_benchmark")
   set(lrbTarget "${name}_lrb_benchmark")

   add_executable(${simpleTarget} ${name}.cpp)
   target_link_libraries(${simpleTarget} Vc ${LIBS})
   add_target_property(${simpleTarget} OUTPUT_NAME ${simpleExec})
   add_target_property(${simpleTarget} COMPILE_FLAGS "-DVC_IMPL=Scalar")
   add_custom_target("run_${simpleTarget}" ${simpleTarget} DEPENDS ${simpleTarget} COMMENT "Running ${simpleExec}")
   add_target_property("run_${simpleTarget}" EXCLUDE_FROM_DEFAULT_BUILD 1)
   add_dependencies(benchmarks "run_${simpleTarget}")

   add_executable(${sseTarget} ${name}.cpp)
   target_link_libraries(${sseTarget} Vc ${LIBS})
   add_target_property(${sseTarget} OUTPUT_NAME ${sseExec})
   add_custom_target("run_${sseTarget}" ${sseTarget} DEPENDS ${sseTarget} COMMENT "Running ${sseExec}")
   add_target_property("run_${sseTarget}" EXCLUDE_FROM_DEFAULT_BUILD 1)
   add_dependencies(benchmarks "run_${sseTarget}")

   if(LARRABEE_FOUND)
      add_executable(${lrbTarget} ${name}.cpp)
      target_link_libraries(${lrbTarget} Vc ${LIBS} ${LRB_LIBS})
      add_target_property(${lrbTarget} OUTPUT_NAME ${lrbExec})
      add_target_property(${lrbTarget} COMPILE_FLAGS "-DVC_IMPL=LRBni")
      add_custom_target("run_${lrbTarget}" ${lrbTarget} DEPENDS ${lrbTarget} COMMENT "Running ${lrbExec}")
      add_target_property("run_${lrbTarget}" EXCLUDE_FROM_DEFAULT_BUILD 1)
      add_dependencies(benchmarks "run_${lrbTarget}")
   endif(LARRABEE_FOUND)
endmacro(vc_add_benchmark)
