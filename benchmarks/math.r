mathProcessData <- function(d) {
    d <- processData(d, paste(d$datatype, d$benchmark.arch, d$benchmark.name),
        pchkey = "benchmark.name")
    if(length(d) > 0) d$key <- paste(d$datatype, d$benchmark.arch)
    d
}

for(data in list(rbind(sse, simple, lrb))) {
    data <- mathProcessData(data)
    mychart4(data, data$benchmark.name, column = "Op_per_cycle", xlab = "Operations per Cycle",
        main = "Math Operations")
}

lrb    <- mathProcessData(lrb)
sse    <- mathProcessData(sse)
simple <- mathProcessData(simple)

plotSpeedup(sse, simple, lrb, main = "Math Operations", speedupColumn = "Op_per_cycle",
    datafun = function(d, ref) list(key = d$datatype)
    )


# vim: sw=4 et filetype=r sts=4 ai
