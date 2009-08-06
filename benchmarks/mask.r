par(cex=0.75)

colors  <- rainbow(2, v=0.3)
lcolors <- rainbow(2, v=0.3, alpha=0.5)
bgs     <- rainbow(2, v=0.8)

for(data in list(rbind(sse, simple, lrb), rbind(sse, simple))) {
    data$main <- factor(sub(" \\(.*?\\)", "", data$benchmark.name, perl=TRUE))
    data$benchmark.name <- sub("^.*?\\((.*)\\).*?", "\\1", data$benchmark.name, perl=TRUE)

    data$fullkey  <- paste(data$benchmark.name, data$datatype, data$benchmark.arch, sep=", ")
    data$fullkey2 <- paste(data$datatype, data$benchmark.arch, data$benchmark.name, sep=", ")
    data$fullkey3 <- paste(data$benchmark.arch, data$datatype, data$benchmark.name, sep=", ")
    splitdata <- split(data, data$main)
    for(i in 1:length(splitdata)) {
        splitdata[[i]]$pch = 20 + i
    }

    for(part in splitdata) {
        attach(part)
        n <- nlevels(factor(fullkey))
        mychart2(Values_per_cycle, fullkey2, xlab="Values per Cycle", main=main[[1]], lcolor=lcolors, bg=bgs, color=colors, pch=pch)
        mychart2(Values_per_cycle, fullkey2, xlab="Values per Cycle", main=main[[1]], lcolor=lcolors, bg=bgs, color=colors, groups=sortkey(datatype, Values_per_cycle, fullkey2) * -1, pch=pch)
        mychart2(Values_per_cycle, fullkey3, xlab="Values per Cycle", main=main[[1]], lcolor=lcolors, bg=bgs, color=colors, groups=sortkey(benchmark.arch, Values_per_cycle, fullkey3) * -1, pch=pch)
        mychart2(Values_per_cycle, fullkey, xlab="Values per Cycle", main=main[[1]],  lcolor=rep(lcolors, each=n/2), bg=rep(bgs, each=n/2), color=rep(colors, each=n/2), groups=sortkey(benchmark.name, Values_per_cycle, fullkey) * -1, pch=pch)
        detach(part)
    }
}

# vim: sw=4 et filetype=r sts=4 ai
