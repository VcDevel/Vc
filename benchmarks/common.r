errorbars <- function(x, y, xerr = NULL, yerr = NULL, ...) {
    if(!is.null(xerr)) {
        arrows(x - xerr, y, x + xerr, y, angle=90, code=3, length=0.05, ...)
    }
    if(!is.null(yerr)) {
        arrows(x, y - yerr, x, y + yerr, angle=90, code=3, length=0.05, ...)
    }
}

mydotchart <- function(x, labels = NULL, groups = NULL, gdata = NULL, errors = NULL,
    mean = NULL, cex = par("cex"),
    pch = 21, gpch = 21, bg = par("bg"), color = par("fg"), gcolor = par("fg"),
    lcolor = "gray", xlim = range(x[is.finite(x)]), main = NULL,
    xlab = NULL, ylab = NULL, yoffset = 0, ...)
{
    opar <- par("mai", "mar", "cex", "yaxs")
    on.exit(par(opar))
    par(cex = cex, yaxs = "i")
    if (!is.numeric(x))
        stop("'x' must be a numeric vector or matrix")
    n <- length(x)
    if (is.matrix(x)) {
        if (is.null(labels))
            labels <- rownames(x)
        if (is.null(labels))
            labels <- as.character(1:nrow(x))
        labels <- rep(labels, length.out = n)
        if (is.null(groups))
            groups <- col(x, as.factor = TRUE)
        glabels <- levels(groups)
    } else {
        if (is.null(labels))
            labels <- names(x)
        glabels <- if (!is.null(groups))
            levels(groups)
    }
    plot.new()
    linch <- if (!is.null(labels))
        max(strwidth(labels, "inch"), na.rm = TRUE)
    else 0
    if (is.null(glabels)) {
        ginch <- 0
        goffset <- 0
    } else {
        ginch <- max(strwidth(glabels, "inch"), na.rm = TRUE)
        goffset <- 0.4
    }
    nmai <- par("mai")
    if (!(is.null(labels) && is.null(glabels))) {
        nmai[2] <- nmai[4] + max(linch + goffset, ginch) + 0.1
        par(mai = nmai)
    }
    if (is.null(groups)) {
        o <- 1:n
        y <- o
        ylim <- c(0, n + 1)
    } else {
        o <- sort.list(as.numeric(groups), decreasing = TRUE)
        x <- x[o]
        groups <- groups[o]
        errors <- errors[o]
        mean <- mean[o]
        color <- rep(color, length.out = length(groups))[o]
        lcolor <- rep(lcolor, length.out = length(groups))[o]
        bg <- rep(bg, length.out = length(groups))[o]
        offset <- cumsum(c(0, diff(as.numeric(groups)) != 0))
        y <- 1:n + 2 * offset
        ylim <- range(0, y + 2)
    }
    plot.window(xlim = xlim, ylim = ylim, log = "")
    lheight <- par("csi")
    if (!is.null(labels)) {
        linch <- max(strwidth(labels, "inch"), na.rm = TRUE)
        loffset <- (linch + 0.1)/lheight
        labs <- labels[o]
        mtext(labs, side = 2, line = loffset, at = y, adj = 0,
            col = color, las = 2, cex = cex, ...)
    }
    abline(h = y + yoffset, lty = "dotted", col = lcolor)
    if (!is.null(errors)) {
        if (is.null(mean)) mean <- x
        errorbars(mean, y + yoffset, xerr=errors, col = color)
    }
    points(x, y + yoffset, pch = pch, col = color, bg = bg)
    if (!is.null(groups)) {
        gpos <- rev(cumsum(rev(tapply(groups, groups, length)) +
            2) - 1)
        ginch <- max(strwidth(glabels, "inch"), na.rm = TRUE)
        goffset <- (max(linch + 0.2, ginch, na.rm = TRUE) + 0.1)/lheight
        mtext(glabels, side = 2, line = goffset, at = gpos, adj = 0,
            col = gcolor, las = 2, cex = cex, ...)
        if (!is.null(gdata)) {
            abline(h = gpos, lty = "dotted")
            points(gdata, gpos, pch = gpch, col = gcolor, bg = bg,
                ...)
        }
    }
    axis(1)
    box()
    title(main = main, xlab = xlab, ylab = ylab, ...)
    invisible(list(y=y + yoffset, o=o, cex=cex, nmai=nmai))
}

addPoints <- function(chart, x, errors = NULL, mean = NULL,
    color = par("fg"), bg = NULL, pch = 21, ...)
{
    opar <- par("mai", "mar", "cex", "yaxs")
    on.exit(par(opar))

    attach(chart)
    par(cex = cex, yaxs = "i", mai = nmai)

    if(is.null(bg)) bg <- {
            tmp <- rgb2hsv(col2rgb(color))
            hsv(h=tmp[row(tmp) == 1], s=tmp[row(tmp) == 2], v=min(1, 1.6 * tmp[row(tmp) == 3]), alpha=0.5)
        }

    x <- x[o]
    errors <- errors[o]
    mean <- mean[o]
    color <- color[o]
    bg <- bg[o]

    if (!is.null(errors)) {
        if (is.null(mean)) mean <- x
        errorbars(mean, y, xerr=errors, col = color)
    }
    points(x, y, pch = pch, col = color, bg = bg)

    detach(chart)
    invisible()
}

addPoints2 <- function(chart, data, key, ...) {
    median <- data[[paste(key, ".median", sep = "")]]
    mean   <- data[[paste(key, ".mean",   sep = "")]]
    stddev <- data[[paste(key, ".stddev", sep = "")]]

    color <- if(is.null(data$color)) par("fg") else data$color
    pch   <- if(is.null(data$pch  )) 21        else data$pch

    addPoints(chart, median, stddev, mean, color, pch = pch, ...)
}

mychart2 <- function(data, keys, groups = NULL, ...) {
keys <- factor(as.vector(keys), ordered=FALSE)
    median <- tapply(data, keys, median)
    mean   <- tapply(data, keys, mean)
    errors <- tapply(data, keys, sd)

    if(is.null(groups)) groups <- median * -1

    mydotchart(median, groups=groups, xlim=c(0, (max(mean + errors))), xaxs="i", sub="median", mean=mean, errors=errors, ...)
}

mychart3 <- function(data, key, ncolors = NULL, xlim = NULL, ...) {
    median <- data[[paste(key, ".median", sep = "")]]
    mean   <- data[[paste(key, ".mean",   sep = "")]]
    stddev <- data[[paste(key, ".stddev", sep = "")]]

    colorkey <- if(is.null(data$benchmark.arch)) data$benchmark.name else data$benchmark.arch
    colorkey <- factor(colorkey)
    if(is.null(ncolors)) ncolors <- nlevels(colorkey)
    if(is.null(data$color  )) data$color   <- rainbow(ncolors, v = 0.5)[as.integer(colorkey)]
    if(is.null(data$bgcolor)) data$bgcolor <- {
            tmp <- rgb2hsv(col2rgb(data$color))
            hsv(h=tmp[row(tmp) == 1], s=tmp[row(tmp) == 2], v=min(1, 1.6 * tmp[row(tmp) == 3]), alpha=0.5)
        }
    if(is.null(data$lcolor )) data$lcolor  <- {
            tmp <- rgb2hsv(col2rgb(data$color))
            hsv(h=tmp[row(tmp) == 1], s=tmp[row(tmp) == 2], v=0.8 * tmp[row(tmp) == 3], alpha=0.5)
        }
    if(is.null(data$pch    )) data$pch     <- c(21:25, 3:20)[as.integer(factor(data$benchmark.name))]

    if(is.null(xlim)) xlim <- range(0, mean + stddev, median)

    chart <- mydotchart(
        median,
        labels = data$key,
        xlim = xlim,
        xaxs = "i",
        sub = "median",
        mean = median,
        errors = stddev,
        color = data$color,
        lcolor = data$lcolor,
        bg = data$bgcolor,
        pch = data$pch,
        ...
        )
    invisible(chart)
}

mychart4 <- function(data, splitfactor, orderfun = function(d) { order(d[[medianCol]]) }, legendpos = "bottomright",
    legendcol = "benchmark.name", column = "Values_per_cycle", xlab = "Values per Cycle", offsetInY = TRUE, ...)
{
    colorkey <- if(is.null(data$benchmark.arch)) data$benchmark.name else data$benchmark.arch
    colorkey <- factor(colorkey)
    ncolors <- nlevels(colorkey)
    if(is.null(data$color  )) data$color   <- rainbow(ncolors, v = 0.5)[as.integer(colorkey)]
    if(is.null(data$lcolor )) data$lcolor  <- {
            tmp <- rgb2hsv(col2rgb(data$color))
            hsv(h=tmp[row(tmp) == 1], s=tmp[row(tmp) == 2], v=0.8 * tmp[row(tmp) == 3], alpha=0.5)
        }
    medianCol <- paste(column, ".median", sep="")
    meanCol   <- paste(column, ".mean"  , sep="")
    stddevCol <- paste(column, ".stddev", sep="")
    chart <- NULL
    legendtext <- rep("unknown", times=50)
    xlim <- range(0, data[medianCol], data[meanCol] + data[stddevCol])
    splitted <- split(data, splitfactor)
    wrap <- (length(splitted) - 1) * 0.5
    steps <- if(offsetInY) 0.8 / length(splitted) else 0
    for(d in splitted) {
        legendtext[d$pch[[1]]] <- as.character(d[[legendcol]][[1]])
        if(is.null(chart)) {
            permutation <- orderfun(d)
            d <- permute(d, permutation)
            chart <- mychart3(d, column, xlab = xlab, xlim = xlim, yoffset = steps * wrap, ...)
        } else {
            d <- permute(d, permutation)
            chart$y <- chart$y - steps
            addPoints2(chart, d, column)
            if(offsetInY) abline(h = chart$y, lty = "dotted", col = d$lcolor)
        }
    }
    pchLevels <- as.numeric(levels(factor(data$pch)))
    pchLevels <- c(pchLevels[pchLevels > 20], pchLevels[pchLevels <= 20])
    cex <- par("cex")
    if(cex >= 0.8) cex <- max(0.75, cex * 0.8)
    legend(legendpos, legendtext[pchLevels], pch=pchLevels, cex = cex)
    invisible(chart)
}

sortkey <- function(string, values, keys) {
    foo <- function(x) x[[1]]
    values <- as.vector(tapply(values, keys, median))
    string <- as.vector(as.numeric(factor(tapply(string, keys, foo))) - 1)

    string <- string * max(values) * 2
    string + values
}

permute <- function(data, o) {
    if(is.data.frame(data)) {
        for(col in colnames(data)) {
            data[col] <- data[[col]][o]
        }
    } else {
        data <- data[o]
    }
    data
}

sortBy <- function(data, key) {
    o <- sort.list(key)
    permute(data, o)
}

processData <- function(data, keys, skip = c(""), pchkey = NULL, sortkey = NULL, colorkey = NULL) {
    keys <- factor(keys) # no empty levels
    l <- levels(keys)
    n <- length(l)
    result <- data.frame(key = l)
    for(col in colnames(data)) {
        v <- as.vector(data[[col]])
        v2 <- NULL
        if(is.character(v) || 1 == max(col == skip)) {
            j <- 1
            for(i in as.integer(keys)) {
                v2[i] <- v[j]
                j <- j + 1
            }
            result[col] <- as.vector(v2)
        } else {
            result[paste(col, sep=".", "median")] <- as.vector(tapply(v, keys, median))
            result[paste(col, sep=".", "mean"  )] <- as.vector(tapply(v, keys, mean))
            result[paste(col, sep=".", "stddev")] <- as.vector(tapply(v, keys, sd))
        }
    }
    if(!is.null(pchkey)) {
        result$pch <- c(21:25, 3:20)[as.integer(factor(result[[pchkey]]))]
    }
    if(!is.null(colorkey)) {
        tmp <- factor(result[[colorkey]])
        result$color <- rainbow(nlevels(tmp), v = 0.5)[as.integer(tmp)]
    }
    if(!is.null(sortkey)) {
        result <- sortBy(result, result[[sortkey]])
    }
    result
}

plotSpeedup <- function(sse, simple, lrb = data.frame(), datafun, plotfun = mychart4, main,
    orderfun = function(d) order(d$datatype, d$benchmark.name), speedupColumn = NULL)
{
    lrb    <- split(lrb   , lrb$datatype)
    sse    <- split(sse   , sse$datatype)
    simple <- split(simple, simple$datatype)

    speedupOf <- function(data, reference) {
        tmp <- datafun(data, reference)
        if(is.null(tmp$datatype) && !is.null(data$datatype)) tmp$datatype <- data$datatype
        if(is.null(tmp$benchmark.name) && !is.null(data$benchmark.name)) tmp$benchmark.name <- data$benchmark.name
        if(is.null(tmp$pch) && !is.null(data$pch)) tmp$pch <- data$pch
        if(is.null(tmp$color) && !is.null(data$color)) tmp$color <- data$color
        if(is.null(tmp$speedup) && !is.null(speedupColumn)) {
            medianCol <- paste(speedupColumn, ".median", sep="")
            meanCol   <- paste(speedupColumn, ".mean"  , sep="")
            stddevCol <- paste(speedupColumn, ".stddev", sep="")
            tmp$speedup.median <- data[[medianCol]] / reference[[medianCol]]
            tmp$speedup.mean <- data[[meanCol]] / reference[[meanCol]]
            tmp$speedup.stddev <- sqrt(
                    (data[[stddevCol]] / data[[meanCol]]) ^ 2 +
                    (reference[[stddevCol]] / reference[[meanCol]]) ^ 2
                ) * data[[meanCol]] / reference[[meanCol]]
        }
        if(is.null(tmp$split)) tmp$split <- tmp$benchmark.name
        data.frame(
            tmp,
            stringsAsFactors = FALSE
            )
    }

    speedup <- rbind(
        speedupOf(sse[["float_v"]], simple[["float_v"]]),
        speedupOf(sse[["sfloat_v"]], simple[["float_v"]]),
        speedupOf(sse[["short_v"]], simple[["short_v"]])
        )

    plotfun(speedup, splitfactor = speedup$split, orderfun = orderfun,
        column = "speedup", xlab = "Speedup",
        main = paste(main, ": SSE vs. Scalar", sep=""))
    abline(v = 1, lty = "dashed", col = hsv(s = 1, v = 0, alpha = 0.4))

    speedup <- rbind(
        speedupOf(lrb[["float_v"]], simple[["float_v"]]),
        speedupOf(lrb[["short_v"]], simple[["short_v"]])
        )

    plotfun(speedup, splitfactor = speedup$split, orderfun = orderfun,
        column = "speedup", xlab = "Speedup",
        main = paste(main, ": LRB Prototype vs. Scalar", sep=""))
    abline(v = 1, lty = "dashed", col = hsv(s = 1, v = 0, alpha = 0.4))

    speedup <- rbind(
        speedupOf(lrb[["float_v"]], sse[["sfloat_v"]]),
        speedupOf(lrb[["short_v"]], sse[["short_v"]])
        )

    plotfun(speedup, splitfactor = speedup$split, orderfun = orderfun,
        column = "speedup", xlab = "Speedup",
        main = paste(main, ": LRB Prototype vs. SSE", sep=""))
    abline(v = 1, lty = "dashed", col = hsv(s = 1, v = 0, alpha = 0.4))
}

par(family="serif")

sse    <- read.table("@sse_datafile@")
simple <- read.table("@simple_datafile@")
lrb    <- if(nchar("@lrb_datafile@") > 0) read.table("@lrb_datafile@") else data.frame()

# vim: sw=4 et filetype=r sts=4 ai
