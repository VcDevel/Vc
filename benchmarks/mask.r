colors  <- rainbow(3, v=0.5)
sse$color <- colors[[3]]
simple$color <- colors[[1]]
lrb$color <- colors[[2]]

maskProcessData <- function(d) {
    d <- processData(d, paste(d$datatype, d$benchmark.arch, d$benchmark.name),
        pchkey = "benchmark.name", sortkey = "key")
    d$key <- paste(d$datatype, d$benchmark.arch)
    d
}

for(data in list(rbind(sse, simple, lrb))) {
    data <- maskProcessData(data)

    mychart4(data, data$benchmark.name, column = "Op_per_cycle", xlab = "Operations per Cycle", main = "Masked Operations")
}

lrb    <- maskProcessData(lrb)
sse    <- maskProcessData(sse)
simple <- maskProcessData(simple)
lrb$color <- colors[as.integer(factor(lrb$datatype))]
sse$color <- colors[as.integer(factor(sse$datatype))]

lrb    <- split(lrb   , lrb$datatype)
sse    <- split(sse   , sse$datatype)
simple <- split(simple, simple$datatype)

speedupOf <- function(data, reference) {
    data.frame(
        key = data$datatype,
        benchmark.name = data$benchmark.name,
        datatype = data$datatype,
        pch = data$pch,
        color = data$color,
        speedup.median = data$Op_per_cycle.median / reference$Op_per_cycle.median,
        speedup.mean = data$Op_per_cycle.mean / reference$Op_per_cycle.mean,
        speedup.stddev = sqrt(
                (data$Op_per_cycle.stddev / data$Op_per_cycle.mean) ^ 2 +
                (reference$Op_per_cycle.stddev / reference$Op_per_cycle.mean) ^ 2
            ) * data$Op_per_cycle.mean / reference$Op_per_cycle.mean,
        stringsAsFactors = FALSE
        )
}

speedup <- rbind(
    speedupOf(sse[["float_v"]], simple[["float_v"]]),
    speedupOf(sse[["sfloat_v"]], simple[["float_v"]]),
    speedupOf(sse[["short_v"]], simple[["short_v"]])
    )
mychart4(speedup, data$benchmark.name, column = "speedup", xlab = "Speedup",
    main = "Masked Operations: SSE vs. Scalar")
abline(v = 1, lty = "dashed", col = hsv(s = 1, v = 0, alpha = 0.4))

# vim: sw=4 et filetype=r sts=4 ai
