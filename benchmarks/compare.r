compareProcessData <- function(d) {
    d <- processData(d, paste(d$benchmark.arch, d$datatype, d$benchmark.name),
        pchkey = "benchmark.name", colorkey = "datatype")
    d$key <- pasteNN(d$datatype, d$benchmark.arch)
    d
}

for(data in list(rbind(sse, simple, lrb))) {
    data <- compareProcessData(data)

    mychart4(data, data$benchmark.name, orderfun = function(d) order(d$Op_per_cycle.median),
        column = "Op_per_cycle", xlab = "Operations per Cycle", main = "Comparisons")

    data$key <- paste(data$datatype)
    for(part in split(data, data$benchmark.name)) {
        mybarplot(part, "benchmark.arch", column = "Op_per_cycle", ylab = "Operations per Cycle",
            main = paste("Comparisons (", part$benchmark.name[[1]], ")", sep=""),
            orderfun = function(d) sort.list(logicalSortkeyForDatatype(d$datatype)),
            maxlaboffset = 1)
    }
}

sse    <- compareProcessData(sse)
lrb    <- compareProcessData(lrb)
simple <- compareProcessData(simple)

plotSpeedup(sse, simple, lrb, main = "Comparisons",
    speedupColumn = "Op_per_cycle",
    datafun = function(d, ref) list(key = d$datatype)
    )

# vim: sw=4 et filetype=r sts=4 ai
