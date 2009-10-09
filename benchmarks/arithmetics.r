arithmeticsProcessData <- function(d) {
    d <- processData(d, paste(d$benchmark.arch, d$datatype, d$benchmark.name),
        pchkey = "benchmark.name", colorkey = "datatype")
    d$key <- pasteNN(d$datatype, d$benchmark.arch)
    d
}

for(data in list(rbind(sse, simple, lrb), rbind(sse, simple))) {
    data <- arithmeticsProcessData(data)

    mychart4(data, data$benchmark.name, orderfun = function(d) order(d$Op_per_cycle.median),
        column = "Op_per_cycle", xlab = "Operations per Cycle", main = "Arithmetics")

    data$key <- paste(data$datatype)
    for(part in split(data, data$benchmark.name)) {
        mybarplot(part,
            splitcolumn = "benchmark.arch",
            column = "cycles_per_Op",
            ylab = "Cycles per Operation",
            main = paste("Arithmetics (", part$benchmark.name[[1]], ")", sep=""),
            orderfun = function(d) sort.list(logicalSortkeyForDatatype(d$datatype)),
            maxlaboffset = 1)
    }

    data$key <- paste(data$benchmark.name, data$benchmark.arch)
    data <- sortBy(data, logicalSortkeyForDatatype(data$datatype))
    mybarplot(data,
            splitcolumn = "datatype",
            column = "cycles_per_Op",
            ylab = "Cycles per Operation",
            main = "Arithmetics",
            orderfun = function(d) sort.list(logicalSortkeyForDatatype(d$datatype)),
            maxlaboffset = 4)
}

sse    <- arithmeticsProcessData(sse)
lrb    <- arithmeticsProcessData(lrb)
simple <- arithmeticsProcessData(simple)

plotSpeedup(sse, simple, lrb, main = "Arithmetics",
    speedupColumn = "Op_per_cycle",
    datafun = function(d, ref) list(key = d$datatype),
    maxlaboffset = 1
    )

# vim: sw=4 et filetype=r sts=4 ai
