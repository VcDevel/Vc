add_definitions(-fno-stack-protector) # Ubuntu adds -fstack-protector-strong / it's not a GCC default
build_example(spline main.cpp spline.cpp)
