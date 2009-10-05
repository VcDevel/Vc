add_executable(default main.cpp)
target_link_libraries(default Vc)

add_executable(sse main.cpp)
add_target_property(sse COMPILE_FLAGS "-DVC_IMPL=SSE")
target_link_libraries(sse Vc)

add_executable(scalar main.cpp)
add_target_property(scalar COMPILE_FLAGS "-DVC_IMPL=Scalar")

if(LARRABEE_FOUND)
   add_executable(lrb main.cpp)
   add_target_property(lrb COMPILE_FLAGS "-DVC_IMPL=LRBni")
endif(LARRABEE_FOUND)
