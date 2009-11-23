memioProcessData <- function(d) {
    d <- processData(d, paste(d$benchmark.arch, d$datatype, d$benchmark.name),
        pchkey = "benchmark.name", colorkey = "datatype")
    d$key <- pasteNN(d$benchmark.arch, d$datatype)
    d
}

for(data in list(rbind(sse, simple), rbind(sse, simple, lrb))) {
    data <- memioProcessData(data)

    mychart4(data, data$benchmark.name, orderfun = function(d) order(d$key),
        column = "Byte_per_cycle", xlab = "Byte per Cycle", main = "L1 I/O")

    data$key <- paste(data$datatype)
    for(part in split(data, data$benchmark.name)) {
        mybarplot(part, "benchmark.arch", column = "Byte_per_cycle", ylab = "Byte per Cycle",
            main = paste("L1 I/O (", part$benchmark.name[[1]], ")", sep=""),
            orderfun = function(d) sort.list(logicalSortkeyForDatatype(d$datatype)),
            maxlaboffset = 1)
    }
}

sse    <- memioProcessData(sse)
lrb    <- memioProcessData(lrb)
simple <- memioProcessData(simple)

plotSpeedup(sse, simple, lrb, main = "L1 I/O",
    speedupColumn = "Byte_per_cycle",
    datafun = function(d, ref) list(key = d$datatype)
    )

# vim: sw=4 et filetype=r sts=4 ai
