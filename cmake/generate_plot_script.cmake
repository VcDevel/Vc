configure_file("${common}" "${scriptfile}" @ONLY)
file(READ "${append}" r_code)
file(APPEND "${scriptfile}" "${r_code}")
