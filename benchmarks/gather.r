par(cex=0.75)
#par(mfrow=c(2,1))

colors  <- rainbow(5, v=0.3)
lcolors <- rainbow(5, v=0.3, alpha=0.5)
bgs     <- rainbow(5, v=0.8)

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
    medianCol <- paste(column, ".median", sep="")
    meanCol   <- paste(column, ".mean"  , sep="")
    stddevCol <- paste(column, ".stddev", sep="")
    chart <- NULL
    legendtext <- rep("not set", times=50)
    xlim <- range(0, data[medianCol], data[meanCol] + data[stddevCol])
    for(d in split(data, data$mask)) {
        legendtext[d$pch[[1]]] <- as.character(d$mask[[1]])
        if(is.null(chart)) {
            permutation <- orderfun(d)
            d <- permute(d, permutation)
            chart <- mychart3(d, column, xlab = xlab, main = main, xlim = xlim)
        } else {
            d <- permute(d, permutation)
            addPoints(chart, d[[medianCol]], d[[stddevCol]], d[[meanCol]], color = d$color, pch = d$pch)
        }
    }
    legend(legendpos, legendtext[21:23], pch=21:23)
    invisible(chart)
}

for(data in list(rbind(sse, simple, lrb), rbind(sse, simple))) {
    data <- gatherProcessData(data)

    data$key <- paste(data$datatype, data$benchmark.arch, data$benchmark.name)
    data <- sortBy(data, data$key)

    gatherChart(data, function(d) { order(d$datatype, d$Values_per_cycle.median) })
    data$key <- paste(data$benchmark.name, data$datatype, data$benchmark.arch)
    gatherChart(data, function(d) { order(d$benchmark.name, d$datatype, d$benchmark.arch) }, legendpos="topright")
}

sse    <- gatherProcessData(sse)
simple <- gatherProcessData(simple)

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
print(speedup)

gatherChart(speedup, function(d) { order(d$datatype, d$benchmark.name) }, column = "speedup", xlab = "Speedup")
abline(v = 1, lty = "dashed", col = hsv(s = 1, v = 0, alpha = 0.4))


#    data$fullkey  <- sub(" Masked", "", paste(data$benchmark.name, data$datatype, data$benchmark.arch, sep=", "))
#    data$fullkey2 <- sub(" Masked", "", paste(data$datatype, data$benchmark.arch, data$benchmark.name, sep=", "))
#    data$fullkey3 <- sub(" Masked", "", paste(data$benchmark.arch, data$datatype, data$benchmark.name, sep=", "))
#        mychart2(Values_per_cycle, fullkey2, xlab="Values per Cycle", main=main[[1]], lcolor=lcolors, bg=bgs, color=colors, pch=pch)
#        mychart2(Values_per_cycle, fullkey2, xlab="Values per Cycle", main=main[[1]], lcolor=lcolors, bg=bgs, color=colors, groups=sortkey(datatype, Values_per_cycle, fullkey2) * -1, pch=pch)
#        mychart2(Values_per_cycle, fullkey3, xlab="Values per Cycle", main=main[[1]], lcolor=lcolors, bg=bgs, color=colors, groups=sortkey(benchmark.arch, Values_per_cycle, fullkey3) * -1, pch=pch)
#        mychart2(Values_per_cycle, fullkey, xlab="Values per Cycle", main=main[[1]],  lcolor=rep(lcolors, each=n/5), bg=rep(bgs, each=n/5), color=rep(colors, each=n/5), groups=sortkey(benchmark.name, Values_per_cycle, fullkey) * -1, pch=pch)

# vim: sw=4 et filetype=r sts=4 ai
