include(BenchmarkMacros)

vc_add_benchmark(flops)
vc_add_benchmark(gather)
vc_add_benchmark(mask)
vc_add_benchmark(compare)

vc_generate_plots(flops)
vc_generate_plots(gather)
vc_generate_plots(mask)
vc_generate_plots(compare)
