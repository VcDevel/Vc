maskProcessData <- function(d) {
    d <- processData(d, paste(d$datatype, d$benchmark.arch, d$benchmark.name),
        pchkey = "benchmark.name", sortkey = "key", colorkey = "datatype")
    if(length(d) > 0 ) d$key <- paste(d$datatype, d$benchmark.arch)
    d
}

for(data in list(rbind(sse, simple, lrb))) {
    data <- maskProcessData(data)

    mychart4(data, data$benchmark.name, column = "Op_per_cycle", xlab = "Operations per Cycle", main = "Masked Operations")
}

lrb    <- maskProcessData(lrb)
sse    <- maskProcessData(sse)
simple <- maskProcessData(simple)

plotSpeedup(sse, simple, lrb, main = "Masked Operations",
    speedupColumn = "Op_per_cycle",
    datafun = function(d, ref) list(key = d$datatype)
    )

# vim: sw=4 et filetype=r sts=4 ai
