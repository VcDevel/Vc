par(cex=0.75)
#par(mfrow=c(2,1))

gatherProcessData <- function(data) {
    data <- processData(data, paste(
            data$benchmark.name,
            data$datatype,
            data$benchmark.arch,
            data$L1.size,
            data$L2.size,
            data$Cacheline.size,
            data$mask,
            sep=", "),
            skip = c("Cacheline.size", "L1.size", "L2.size")
        )
    data$pch <- (21:31)[as.integer(factor(data$mask))]
    data$mask <- c("not masked", "masked with one", "random mask")[as.integer(factor(data$mask))]
    data$color <- rainbow(5, v=0.5)[as.integer(factor(data$benchmark.name))]
    data
}

gatherChart <- function(data, orderfun, legendpos = "bottomright", column = "Values_per_cycle",
    xlab = "Values per Cycle", main = "Vector Gather")
{
    l2info <- NULL
    l1info <- NULL
    clinfo <- NULL
    if(!is.null(data$L2.size)) l2info <- paste("L2:", as.character(data$L2.size[[1]]))
    if(!is.null(data$L1.size)) l1info <- paste("L1:", as.character(data$L1.size[[1]]))
    if(!is.null(data$Cacheline.size)) clinfo <- paste("Cacheline:", as.character(data$Cacheline.size[[1]]))
    cacheinfo <- paste(c(l2info, l1info, clinfo), collapse=", ")
    if(nchar(cacheinfo) > 0) {
        main <- paste(main, " (", cacheinfo, ")", sep="")
    }

    mychart4(data, data$mask, orderfun, legendpos = legendpos, legendcol = "mask", column = column,
        xlab = xlab, main = main)
}

sse$Cacheline.size <- paste(as.character(sse$Cacheline.size), "B")
lrb$Cacheline.size <- paste(as.character(lrb$Cacheline.size), "B")
simple$Cacheline.size <- paste(as.character(simple$Cacheline.size), "B")

sse$L1.size <- paste(as.character(sse$L1.size / 1024), "KiB")
lrb$L1.size <- paste(as.character(lrb$L1.size / 1024), "KiB")
simple$L1.size <- paste(as.character(simple$L1.size / 1024), "KiB")

sse$L2.size <- sse$L2.size / 1024
lrb$L2.size <- lrb$L2.size / 1024
simple$L2.size <- simple$L2.size / 1024
if(sse$L2.size[[1]] >= 1024) {
    sse$L2.size <- paste(as.character(sse$L2.size / 1024), "MiB")
    lrb$L2.size <- paste(as.character(lrb$L2.size / 1024), "MiB")
    simple$L2.size <- paste(as.character(simple$L2.size / 1024), "MiB")
} else {
    sse$L2.size <- paste(as.character(sse$L2.size), "KiB")
    lrb$L2.size <- paste(as.character(lrb$L2.size), "KiB")
    simple$L2.size <- paste(as.character(simple$L2.size), "KiB")
}

for(data in list(rbind(sse, simple, lrb), rbind(sse, simple))) {
    data <- gatherProcessData(data)

    data$key <- paste(data$datatype, data$benchmark.arch, data$benchmark.name)
    data <- sortBy(data, data$key)

    gatherChart(data, function(d) { order(d$datatype, d$Values_per_cycle.median) })
    data$key <- paste(data$benchmark.name, data$datatype, data$benchmark.arch)
    gatherChart(data, function(d) { order(d$benchmark.name, d$datatype, d$benchmark.arch) }, legendpos="topright")
}

lrb    <- gatherProcessData(lrb)
sse    <- gatherProcessData(sse)
simple <- gatherProcessData(simple)

lrb    <- split(lrb   , lrb$datatype)
sse    <- split(sse   , sse$datatype)
simple <- split(simple, simple$datatype)

speedupOf <- function(data, reference) {
    data.frame(
        key = paste(data$datatype, data$benchmark.name),
        benchmark.name = data$benchmark.name,
        datatype = data$datatype,
        mask = data$mask,
        L1.size = data$L1.size,
        L2.size = data$L2.size,
        pch = data$pch,
        color = data$color,
        Cacheline.size = data$Cacheline.size,
        speedup.median = data$Values_per_cycle.median / reference$Values_per_cycle.median,
        speedup.mean = data$Values_per_cycle.mean / reference$Values_per_cycle.mean,
        speedup.stddev = sqrt(
                (data$Values_per_cycle.stddev / data$Values_per_cycle.mean) ^ 2 +
                (reference$Values_per_cycle.stddev / reference$Values_per_cycle.mean) ^ 2
            ) * data$Values_per_cycle.mean / reference$Values_per_cycle.mean,
        stringsAsFactors = FALSE
        )
}

speedup <- rbind(
    speedupOf(sse[["float_v"]], simple[["float_v"]]),
    speedupOf(sse[["sfloat_v"]], simple[["float_v"]]),
    speedupOf(sse[["short_v"]], simple[["short_v"]])
    )

gatherChart(speedup, function(d) { order(d$datatype, d$benchmark.name) }, column = "speedup", xlab =
"Speedup", main = "Vector Gather: SSE vs. Scalar")
abline(v = 1, lty = "dashed", col = hsv(s = 1, v = 0, alpha = 0.4))

speedup <- rbind(
    speedupOf(lrb[["float_v"]], simple[["float_v"]]),
    speedupOf(lrb[["short_v"]], simple[["short_v"]])
    )

gatherChart(speedup, function(d) { order(d$datatype, d$benchmark.name) }, column = "speedup", xlab =
"Speedup", main = "Vector Gather: LRB Prototype vs. Scalar")
abline(v = 1, lty = "dashed", col = hsv(s = 1, v = 0, alpha = 0.4))

speedup <- rbind(
    speedupOf(lrb[["float_v"]], sse[["sfloat_v"]]),
    speedupOf(lrb[["short_v"]], sse[["short_v"]])
    )

gatherChart(speedup, function(d) { order(d$datatype, d$benchmark.name) }, column = "speedup", xlab =
"Speedup", main = "Vector Gather: LRB Prototype vs. SSE")
abline(v = 1, lty = "dashed", col = hsv(s = 1, v = 0, alpha = 0.4))

# vim: sw=4 et filetype=r sts=4 ai
