if(Vc_X86)
   add_executable(cpuid main.cpp)
   target_link_libraries(cpuid Vc)
endif()
