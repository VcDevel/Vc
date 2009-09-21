colors  <- rep(rainbow(3, v=0.25), each=2)
lcolors <- rep(rainbow(3, v=0.25, alpha=0.5), each=2)
bgs     <- rep(rainbow(3, v=0.8), each=2)
pch     <- rep(c(21, 23), times=3)

for(data in list(rbind(sse, simple, lrb))) {
    attach(data)

    keys <- paste(benchmark.arch, benchmark.name, sep=", ")
    groups <- tapply(FLOP_per_cycle, keys, median)
    GFLOP_per_sec_Real <- FLOP_per_sec_Real / 1000000000
    n <- nlevels(factor(keys))

    mychart2(FLOP_per_cycle, keys, xlab="FLOP per Cycle", main="Multiply - Add", groups=n:1, lcolor=lcolors, bg=bgs, color=colors, pch=pch)
    mychart2(GFLOP_per_sec_Real, keys, xlab="GFLOP per Second", main="Multiply - Add", groups=n:1, lcolor=lcolors, bg=bgs, color=colors, pch=pch)

    detach(data)
}

# vim: sw=4 et filetype=r sts=4 ai
