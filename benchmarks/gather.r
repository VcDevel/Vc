par(cex=0.75)

colors  <- rainbow(5, v=0.3)
lcolors <- rainbow(5, v=0.3, alpha=0.5)
bgs     <- rainbow(5, v=0.8)

for(data in list(rbind(sse, simple, lrb), rbind(sse, simple))) {
    data$GValues_per_sec_Real <- data$Values_per_sec_Real / 1000000000
    data$GValues_per_sec_CPU <- data$Values_per_sec_CPU / 1000000000
    data$main <- factor(sub("^.*?(Masked)?$", "\\1 Vector Gather", data$benchmark.name, perl=TRUE))

    data$fullkey  <- sub(" Masked", "", paste(data$benchmark.name, data$datatype, data$benchmark.arch, sep=", "))
    data$fullkey2 <- sub(" Masked", "", paste(data$datatype, data$benchmark.arch, data$benchmark.name, sep=", "))
    data$fullkey3 <- sub(" Masked", "", paste(data$benchmark.arch, data$datatype, data$benchmark.name, sep=", "))
    splitdata <- split(data, data$main)
    splitdata[[1]]$pch = 21
    splitdata[[2]]$pch = 24

    n <- nlevels(factor(data$fullkey))

    for(part in splitdata) {
        attach(part)
        mychart2(Values_per_cycle, fullkey2, xlab="Values per Cycle", main=main[[1]], lcolor=lcolors, bg=bgs, color=colors, pch=pch)
        mychart2(Values_per_cycle, fullkey2, xlab="Values per Cycle", main=main[[1]], lcolor=lcolors, bg=bgs, color=colors, groups=sortkey(datatype, Values_per_cycle, fullkey2) * -1, pch=pch)
        mychart2(Values_per_cycle, fullkey3, xlab="Values per Cycle", main=main[[1]], lcolor=lcolors, bg=bgs, color=colors, groups=sortkey(benchmark.arch, Values_per_cycle, fullkey3) * -1, pch=pch)
        mychart2(Values_per_cycle, fullkey, xlab="Values per Cycle", main=main[[1]],  lcolor=rep(lcolors, each=n/5), bg=rep(bgs, each=n/5), color=rep(colors, each=n/5), groups=sortkey(benchmark.name, Values_per_cycle, fullkey) * -1, pch=pch)
        detach(part)
    }
}

data <- rbind(sse, simple)
data$key <- paste(data$benchmark.arch, data$datatype, data$benchmark.name)
data$key2 <- sub(" Masked", "", data$key)
masked <- factor(sub("^.*?(Masked)?$", "\\1 Vector Gather", data$benchmark.name, perl=TRUE))
tmp <- split(data, masked)
data1 <- tmp[[1]]
data2 <- tmp[[2]]
rm(tmp)
groups <- sortkey(data1$key, data1$Values_per_cycle, data1$key)

n <- nlevels(factor(data1$key2))
y <- 1:n
ylim <- c(0, n + 1)
x1 <- tapply(data1$Values_per_cycle, data1$key2, median)
x2 <- tapply(data2$Values_per_cycle, data2$key2, median)
labels <- names(x1)
linch <- max(strwidth(labels, "inch"), na.rm = TRUE)
o <- sort.list(groups, decreasing = TRUE)
x1 <- x1[o]
x2 <- x2[o]
color <- rep(colors, length.out = n)[o]
lcolor <- rep(lcolors, length.out = n)[o]
bg <- rep(bgs, length.out = n)[o]
offset <- cumsum(c(0, diff(groups) != 0))

par(yaxs = "i")
plot.new()
nmai <- par("mai")
nmai[2] <- nmai[4] + linch + 0.1
par(mai = nmai)

plot.window(xlim = range(0, max(c(x1, x2))), ylim = ylim, log = "")
lheight <- par("csi")

linch <- max(strwidth(labels, "inch"), na.rm = TRUE)
loffset <- (linch + 0.1)/lheight
labs <- labels[o]
mtext(labs, side = 2, line = loffset, at = y, adj = 0, col = color, las = 2, cex = par("cex"))
abline(h = y, lty = "dotted", col = lcolor)
points(x1, y, pch = 21, col = color, bg = bg)
points(x2, y, pch = 24, col = color, bg = bg)

axis(1)
box()
title(main = "Full vs. Masked Vector Gather", xlab = "Values per Cycle", sub="median")

# vim: sw=4 et filetype=r sts=4 ai
