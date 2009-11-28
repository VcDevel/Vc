
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
   endif(R_COMMAND)
   add_custom_target(benchmark_data COMMENT "Create data for later processing for all benchmarks" VERBATIM)
   add_target_property(benchmark_data EXCLUDE_FROM_DEFAULT_BUILD 1)
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
      set(targets)
      set(dataFilePaths)
      set(dataFiles)

      set(lrb)
      if(LARRABEE_FOUND)
         set(lrb "lrb")
      endif(LARRABEE_FOUND)

      set(scriptArgs)

      foreach(def "scalar" "sse" ${lrb} ${ARGN})
         set(_t "${name}_${def}_benchmark")
         vc_generate_datafile(${_t} dfp df)

         set(targets ${targets} ${_t})
         set(dataFilePaths ${dataFilePaths} ${dfp})
         set(dataFiles ${dataFiles} ${df})
         set(scriptArgs ${scriptArgs} "-D${def}_datafile=${df}")
      endforeach(def)

      set(scriptfile "${CMAKE_CURRENT_BINARY_DIR}/plot_${name}.r")
      add_custom_command(OUTPUT "${scriptfile}"
         COMMAND "${CMAKE_COMMAND}"
         ARGS
         "-Dscriptfile=${scriptfile}"
         "-Dcommon=${CMAKE_CURRENT_SOURCE_DIR}/common.r"
         "-Dappend=${CMAKE_CURRENT_SOURCE_DIR}/${name}.r"
         ${scriptArgs}
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
            DEPENDS "${scalar_datafilepath}" "${sse_datafilepath}" "${lrb_datafilepath}" "${CMAKE_CURRENT_SOURCE_DIR}/common.r" "${CMAKE_CURRENT_SOURCE_DIR}/${name}.r" "${scriptfile}"
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
      endif(R_COMMAND)

      set(generate_target "generate_${name}_data")
      add_custom_target(${generate_target} DEPENDS ${dataFilePaths} "${scriptfile}")
      add_dependencies(benchmark_data "${generate_target}")
   endif(ENABLE_BENCHMARKS)
endmacro(vc_generate_plots)

macro(vc_add_benchmark name)
   if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      set(LIBS rt)
   endif(CMAKE_SYSTEM_NAME STREQUAL "Linux")

   set(execs)
   set(targets)

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
