pasteNN <- function(..., sep = " ") {
    tmp <- NULL
    for(i in list(...)) {
        if(!is.null(i)) {
            tmp <- if(is.null(tmp)) i else paste(tmp, i, sep = sep)
        }
    }
    tmp
}

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
    invisible(list(y=y + yoffset, o=o, cex=cex, nmai=nmai, points=data.frame(x=x, y=y + yoffset)))
}

addPoints <- function(chart, addx, errors = NULL, mean = NULL,
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

    addx <- addx[o]
    errors <- errors[o]
    mean <- mean[o]
    color <- color[o]
    bg <- bg[o]

    if (!is.null(errors)) {
        if (is.null(mean)) mean <- addx
        errorbars(mean, y, xerr=errors, col = color)
    }
    points(addx, y, pch = pch, col = color, bg = bg)

    detach(chart)
    chart$points = rbind(chart$points, data.frame(x=addx, y=chart$y))
    invisible(chart)
}

addPoints2 <- function(chart, data, key, ...) {
    median <- data[[paste(key, ".median", sep = "")]]
    mean   <- data[[paste(key, ".mean",   sep = "")]]
    stddev <- data[[paste(key, ".stddev", sep = "")]]

    color <- if(is.null(data$color)) par("fg") else data$color
    pch   <- if(is.null(data$pch  )) 21        else data$pch

    chart <- addPoints(chart, median, stddev, mean, color, pch = pch, ...)
    invisible(chart)
}

mychart2 <- function(data, keys, groups = NULL, ...) {
keys <- factor(as.vector(keys), ordered=FALSE)
    median <- tapply(data, keys, median)
    mean   <- tapply(data, keys, mean)
    errors <- tapply(data, keys, sd)

    if(is.null(groups)) groups <- median * -1

    maxx <- max(mean + errors)
    mydotchart(median, groups=groups, xlim=c(0, maxx), xaxs="i", sub="median", mean=mean, errors=errors, ...)

    y <- 1.2
    for(x in median) {
        pos <- if(x > maxx * 0.5) 2 else 4
        text(x = x, y = y, pos = pos, labels = x)
        y <- y + 3
    }
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
            chart <- addPoints2(chart, d, column)
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

printValuesInChart <- function(chart) {
    x <- chart$points$x
    y <- chart$points$y
    maxx <- max(x)
    pos <- 4
    pos[x <= maxx * 0.5] <- 4
    pos[x > maxx * 0.5] <- 2
    text(x = x, y = y, pos = pos, labels = x)
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
    if(length(data) == 0) return(NULL)
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

logicalSortkeyForDatatype <- function(type) {
    type[type == "double_v"] <- 0
    type[type == "float_v" ] <- 1
    type[type == "sfloat_v"] <- 2
    type[type == "int_v"   ] <- 3
    type[type == "uint_v"  ] <- 4
    type[type == "short_v" ] <- 5
    type[type == "ushort_v"] <- 6
    type
}

speedupBarPlot <- function(speedup, ylab = "Speedup", maxlaboffset = 4, ...) {
    speedup <- permute(speedup, sort.list(logicalSortkeyForDatatype(speedup$datatype)))
    speedup <- sortBy(speedup, speedup$benchmark.name)
    datatypes <- unique.default(speedup$datatype)
    n <- length(datatypes)
    m <- matrix(data = speedup$speedup.median, nrow = n)
    barplot(
        height = m,
        beside = TRUE,
        col = speedup$color,
        legend = datatypes,
        ylim = c(0,ceiling(max(m))),
        ylab = ylab,
        ...
    )
    x <- (n + 2) * 0.5
    offset <- 0.5
    for(name in levels(as.factor(speedup$benchmark.name))) {
        text(x = x, y = 0,
            labels = name,
            pos = 1,
            offset = offset,
            xpd = TRUE
            )
        x <- x + n + 1
        offset <- offset + 1
        if (offset >= min(maxlaboffset, nlevels(as.factor(speedup$benchmark.name)) / 2)) offset <- 0.5
    }
    abline(h = 1, lty = "dashed", col = hsv(s = 1, v = 0, alpha = 0.8))
}

plotSpeedup <- function(sse, simple, lrb = data.frame(), datafun, plotfun = mychart4, main,
    orderfun = function(d) order(logicalSortkeyForDatatype(d$datatype), d$benchmark.name), speedupColumn = NULL, ...)
{
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
        if(is.null(tmp$mainpostfix)) tmp$mainpostfix <- paste(data$benchmark.arch, reference$benchmark.arch, sep = " vs. ")
        data.frame(
            tmp,
            stringsAsFactors = FALSE
            )
    }

    simpledatatypes <- levels(as.factor(simple$datatype))
    sse    <- split(sse   , sse$datatype)
    simple <- split(simple, simple$datatype)

    speedup <- NULL
    for(datatype in simpledatatypes) {
        tmp <- speedupOf(sse[[datatype]], simple[[datatype]])
        if(is.null(speedup)) {
            speedup <- tmp
        } else {
            speedup <- rbind(speedup, tmp)
        }
    }
    if(min(simpledatatypes != "sfloat_v")) {
        speedup <- rbind(speedup, speedupOf(sse[["sfloat_v"]], simple[["float_v"]]))
    }

    plotfun(speedup, splitfactor = speedup$split, orderfun = orderfun,
        column = "speedup", xlab = "Speedup",
        main = paste(main, speedup$mainpostfix[[1]], sep=": "))
    abline(v = 1, lty = "dashed", col = hsv(s = 1, v = 0, alpha = 0.4))

    speedupBarPlot(speedup,
            main = paste(main, speedup$mainpostfix[[1]], sep=": "), ...
            )

    if(length(lrb) > 0) {
        lrb <- split(lrb, lrb$datatype)
        speedup <- rbind(
            speedupOf(lrb[["float_v"]], simple[["float_v"]]),
            speedupOf(lrb[["short_v"]], simple[["short_v"]])
            )

        plotfun(speedup, splitfactor = speedup$split, orderfun = orderfun,
            column = "speedup", xlab = "Speedup",
            main = paste(main, speedup$mainpostfix[[1]], sep=": "))
        abline(v = 1, lty = "dashed", col = hsv(s = 1, v = 0, alpha = 0.4))

        speedupBarPlot(speedup,
            main = paste(main, speedup$mainpostfix[[1]], sep=": "), ...
                )

        speedup <- rbind(
            speedupOf(lrb[["float_v"]], sse[["sfloat_v"]]),
            speedupOf(lrb[["short_v"]], sse[["short_v"]])
            )

        plotfun(speedup, splitfactor = speedup$split, orderfun = orderfun,
            column = "speedup", xlab = "Speedup",
            main = paste(main, speedup$mainpostfix[[1]], sep=": "))
        abline(v = 1, lty = "dashed", col = hsv(s = 1, v = 0, alpha = 0.4))

    speedupBarPlot(speedup,
            main = paste(main, speedup$mainpostfix[[1]], sep=": "), ...
            )
    }
}

barPlotCyclesPerOp <- function(data) {
    data <- sortBy(sortBy(data, data$datatype), data$benchmark.name)

    narch <- length(levels(as.factor(data$benchmark.arch)))

    key <- levels(as.factor(data$key))
    nrow <- length(key)

    barplot(
        height = matrix(data = 1 / data$Op_per_cycle.median, nrow = nrow),
        beside = TRUE,
        col = rainbow(narch, v = 0.8)[as.integer(as.factor(data$benchmark.arch))],
        legend = data$key[1:nrow],
        ylab = "Cycles per Operation"
        )
}

orderAs <- function(data, colName, reference) {
    tmp <- data[[colName]]
    n <- length(tmp)
    tmp <- matrix(tmp, nrow=n+1, ncol=n)
    tmp <- matrix(tmp[row(tmp) <= n], nrow=n)
    reference <- matrix(reference, byrow=TRUE, nrow=n, ncol=n)
    h <- matrix(1:n, byrow=TRUE, nrow=n, ncol=n+1)
    h <- matrix(h[col(h) <= n], nrow=n)
    data[colName] <- data[[colName]][h[tmp == reference]]
    data
}

nicerLimit <- function(max) {
    factor <- 1.0
    while(max < 1.0) {
        factor <- factor * 0.1
        max <- max * 10
    }
    c(0, ceiling(max) * factor)
}

mybarplot <- function(data, column, splitcolumn = NULL, orderfun = function(d) order(-d[[medianCol]]),
        maxlaboffset = 4, ...) {
    medianCol <- paste(column, ".median", sep="")
    meanCol   <- paste(column, ".mean"  , sep="")
    stddevCol <- paste(column, ".stddev", sep="")

    if(!is.null(orderfun)) data <- permute(data, orderfun(data))

    colors <- NULL
    xlab <- NULL
    m <- NULL
    n <- 0
    if(!is.null(splitcolumn)) {
        ncolors <- nlevels(as.factor(data[[splitcolumn]]))
        data$color <- rainbow(ncolors, v = 0.5)[as.integer(as.factor(data[[splitcolumn]]))]
        splitted <- split(data, as.factor(data[[splitcolumn]]))
        legend <- NULL
        for(s in splitted) {
            legend <- c(legend, s[[splitcolumn]][[1]])
            if(!is.null(xlab)) {
                s <- orderAs(s, "key", xlab)
            } else {
                xlab <- s$key
            }
            m <- rbind(m, s[[medianCol]])
            colors <- rbind(colors, s$color)
            n <- n + 1
        }
        colors <- as.vector(colors)
    } else {
        m <- data[[medianCol]]
        colors <- data$colors
        legend <- data$key
    }
    barplot(
        height = m,
        beside = TRUE,
        col = colors,
        legend = legend,
        ylim = nicerLimit(max(m)),
        ...
        )
    if(!is.null(xlab)) {
        x <- (n + 2) * 0.5
        offset <- 0.5
        for(name in xlab) {
            text(x = x, y = 0,
                labels = name,
                pos = 1,
                offset = offset,
                xpd = TRUE
                )
            x <- x + n + 1
            offset <- offset + 1
            if (offset >= min(maxlaboffset, length(xlab) / 2)) offset <- 0.5
        }
    }
}

par(family="serif")

sse    <- read.table("@sse_datafile@", header = TRUE)
simple <- read.table("@simple_datafile@", header = TRUE)
lrb    <- if(nchar("@lrb_datafile@") > 0) read.table("@lrb_datafile@", header = TRUE) else data.frame()

# vim: sw=4 et filetype=r sts=4 ai
