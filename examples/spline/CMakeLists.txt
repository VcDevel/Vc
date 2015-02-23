AddCompilerFlag(-fno-stack-protector) # Ubuntu adds -fstack-protector-strong / it's not a GCC default
AddCompilerFlag(-mno-vzeroupper)
build_example(spline main.cpp spline.cpp)
