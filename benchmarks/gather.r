par(cex=0.75)
#par(mfrow=c(2,1))

gatherProcessData <- function(data) {
    processData(data, paste(
                data$benchmark.name,
                data$datatype,
                data$benchmark.arch,
                data$L1.size,
                data$L2.size,
                data$Cacheline.size,
                data$mask,
                sep=", "),
            skip = c("Cacheline.size", "L1.size", "L2.size"),
            pchkey = "mask", colorkey = "benchmark.name"
        )
}

sse$Cacheline.size <- paste(as.character(sse$Cacheline.size), "B")
if(length(lrb) > 0) lrb$Cacheline.size <- paste(as.character(lrb$Cacheline.size), "B")
simple$Cacheline.size <- paste(as.character(simple$Cacheline.size), "B")

sse$L1.size <- paste(as.character(sse$L1.size / 1024), "KiB")
if(length(lrb) > 0) lrb$L1.size <- paste(as.character(lrb$L1.size / 1024), "KiB")
simple$L1.size <- paste(as.character(simple$L1.size / 1024), "KiB")

sse$L2.size <- sse$L2.size / 1024
if(length(lrb) > 0) lrb$L2.size <- lrb$L2.size / 1024
simple$L2.size <- simple$L2.size / 1024
if(sse$L2.size[[1]] >= 1024) {
    sse$L2.size <- paste(as.character(sse$L2.size / 1024), "MiB")
    if(length(lrb) > 0) lrb$L2.size <- paste(as.character(lrb$L2.size / 1024), "MiB")
    simple$L2.size <- paste(as.character(simple$L2.size / 1024), "MiB")
} else {
    sse$L2.size <- paste(as.character(sse$L2.size), "KiB")
    if(length(lrb) > 0) lrb$L2.size <- paste(as.character(lrb$L2.size), "KiB")
    simple$L2.size <- paste(as.character(simple$L2.size), "KiB")
}

sse$L3.size <- sse$L3.size / 1024
if(length(lrb) > 0) lrb$L3.size <- lrb$L3.size / 1024
simple$L3.size <- simple$L3.size / 1024
if(sse$L3.size[[1]] >= 1024) {
    sse$L3.size <- paste(as.character(sse$L3.size / 1024), "MiB")
    if(length(lrb) > 0) lrb$L3.size <- paste(as.character(lrb$L3.size / 1024), "MiB")
    simple$L3.size <- paste(as.character(simple$L3.size / 1024), "MiB")
} else {
    sse$L3.size <- paste(as.character(sse$L3.size), "KiB")
    if(length(lrb) > 0) lrb$L3.size <- paste(as.character(lrb$L3.size), "KiB")
    simple$L3.size <- paste(as.character(simple$L3.size), "KiB")
}

l3info <- NULL
l2info <- NULL
l1info <- NULL
clinfo <- NULL
if(!is.null(simple$L3.size)) l3info <- paste("L3:", as.character(simple$L3.size[[1]]))
if(!is.null(simple$L2.size)) l2info <- paste("L2:", as.character(simple$L2.size[[1]]))
if(!is.null(simple$L1.size)) l1info <- paste("L1:", as.character(simple$L1.size[[1]]))
if(!is.null(simple$Cacheline.size)) clinfo <- paste("Cacheline:", as.character(simple$Cacheline.size[[1]]))
cacheinfo <- paste(c(l3info, l2info, l1info, clinfo), collapse=", ")
if(nchar(cacheinfo) > 0) {
    cacheinfo <- paste("(", cacheinfo, ")", sep="")
}

gatherChart <- function(data, orderfun, legendpos = "bottomright", column = "Values_per_cycle",
    xlab = "Values per Cycle", main = "Vector Gather", splitfactor = NULL)
{
    main <- paste(main, cacheinfo)
    mychart4(data, data$mask, orderfun, legendpos = legendpos, legendcol = "mask", column = column,
        xlab = xlab, main = main, offsetInY = FALSE)
}

for(data in list(rbind(sse, simple, lrb), rbind(sse, simple))) {
    data <- gatherProcessData(data)

    data$key <- paste(data$datatype, data$benchmark.arch, data$benchmark.name)
    data <- sortBy(data, data$key)

    gatherChart(data, function(d) { order(d$datatype, d$Values_per_cycle.median) })
    data$key <- paste(data$benchmark.name, data$datatype, data$benchmark.arch)
    gatherChart(data, function(d) { order(d$benchmark.name, d$datatype, d$benchmark.arch) }, legendpos="topright")

    data$key <- paste(data$benchmark.name)
    data$splitkey <- paste(data$datatype, data$mask, sep= " | ")
    for(part in split(data, data$splitkey)) {
        mybarplot(part, "benchmark.arch", column = "Values_per_cycle", ylab = "Values per Cycle",
            main = part$splitkey[[1]],
            orderfun = function(d) order(d$benchmark.name),
            maxlaboffset = 4)
    }
}

lrb    <- gatherProcessData(lrb)
sse    <- gatherProcessData(sse)
simple <- gatherProcessData(simple)

plotSpeedup(sse, simple, lrb, plotfun = gatherChart, main = "Vector Gather",
    speedupColumn = "Values_per_cycle",
    datafun = function(data, reference) list(
        key = paste(data$datatype, data$benchmark.name),
        mask = data$mask,
        L1.size = data$L1.size,
        L2.size = data$L2.size,
        Cacheline.size = data$Cacheline.size
        )
    )

# vim: sw=4 et filetype=r sts=4 ai
