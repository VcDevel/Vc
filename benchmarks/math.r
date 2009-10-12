mathProcessData <- function(d) {
    d <- processData(d, paste(d$datatype, d$benchmark.arch, d$benchmark.name),
        pchkey = "benchmark.name", colorkey = "datatype")
    if(length(d) > 0) d$key <- paste(d$datatype, d$benchmark.arch)
    d
}

set1 <- function(d) {
    if(length(d) == 0) return(NULL)
    subset(d, !(d$benchmark.name == "rsqrt" | d$benchmark.name == "recip" | d$benchmark.name == "round"))
}
set2 <- function(d) {
    if(length(d) == 0) return(NULL)
    subset(d,   d$benchmark.name == "rsqrt" | d$benchmark.name == "recip" | d$benchmark.name == "round" )
}

data <- rbind(sse, simple, lrb)
for(data in list(set1(data), set2(data))) {
    data <- mathProcessData(data)
    mychart4(data, data$benchmark.name, column = "Op_per_cycle", xlab = "Operations per Cycle",
        main = "Math Operations", orderfun = function(d) order(d$key), maxlaboffset = 1)
}

lrb1    <- mathProcessData(set1(lrb))
sse1    <- mathProcessData(set1(sse))
simple1 <- mathProcessData(set1(simple))
lrb2    <- mathProcessData(set2(lrb))
sse2    <- mathProcessData(set2(sse))
simple2 <- mathProcessData(set2(simple))

plotSpeedup(sse1, simple1, lrb1, main = "Math Operations", speedupColumn = "Op_per_cycle",
    datafun = function(d, ref) list(key = d$datatype), maxlaboffset = 1
    )
plotSpeedup(sse2, simple2, lrb2, main = "Math Operations", speedupColumn = "Op_per_cycle",
    datafun = function(d, ref) list(key = d$datatype), maxlaboffset = 1
    )


# vim: sw=4 et filetype=r sts=4 ai
