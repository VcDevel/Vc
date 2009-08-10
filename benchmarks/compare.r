for(data in list(rbind(sse, simple, lrb))) {
    data <- processData(data, factor(paste(data$benchmark.arch, data$datatype, data$benchmark.name, sep=", ")))
    data$pch     <- (21:31)[as.integer(factor(data$datatype))]

    data <- sortBy(data, data$cmp_per_cycle.median)
    mychart3(data, "cmp_per_cycle", xlab = "Compares per Cycle", main = "operator<")

    data <- sortBy(data, data$benchmark.arch)
    mychart3(data, "cmp_per_cycle", xlab = "Compares per Cycle", main = "operator<")
}

# vim: sw=4 et filetype=r sts=4 ai
