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
    xlab = NULL, ylab = NULL, ...)
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
    if (!(is.null(labels) && is.null(glabels))) {
        nmai <- par("mai")
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
    abline(h = y, lty = "dotted", col = lcolor)
    if (!is.null(errors)) {
        if (is.null(mean)) mean <- x
        errorbars(mean, y, xerr=errors, col = color)
    }
    points(x, y, pch = pch, col = color, bg = bg)
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
    invisible()
}

mychart2 <- function(data, keys, groups = NULL, ...) {
keys <- factor(as.vector(keys), ordered=FALSE)
    median <- tapply(data, keys, median)
    mean   <- tapply(data, keys, mean)
    errors <- tapply(data, keys, sd)

    if(is.null(groups)) groups <- median * -1

    mydotchart(median, groups=groups, xlim=c(0, (max(mean + errors))), xaxs="i", sub="median", mean=mean, errors=errors, ...)
}

mychart3 <- function(data, key, ...) {
    medianOf <- function(data, key) data[[paste(key, ".median", sep = "")]]
    meanOf   <- function(data, key) data[[paste(key, ".mean",   sep = "")]]
    stddevOf <- function(data, key) data[[paste(key, ".stddev", sep = "")]]

    arch <- factor(data$benchmark.arch)
    narch <- nlevels(arch)
    if(is.null(data$color  )) data$color   <- rainbow(narch, v = 0.5)[as.integer(arch)]
    if(is.null(data$lcolor )) data$lcolor  <- rainbow(narch, v = 0.5, alpha = 0.5)[as.integer(arch)]
    if(is.null(data$bgcolor)) data$bgcolor <- rainbow(narch, v = 0.8)[as.integer(arch)]
    if(is.null(data$pch    )) data$pch     <- c(21:31, 1:20)[as.integer(factor(data$benchmark.name))]

    mydotchart(
        medianOf(data, key),
        labels = data$key,
        xlim = range(0, meanOf(data, key) + stddevOf(data, key), medianOf(data, key)),
        xaxs = "i",
        sub = "median",
        mean = medianOf(data, key),
        errors = stddevOf(data, key),
        color = data$color,
        lcolor = data$lcolor,
        bg = data$bgcolor,
        pch = data$pch,
        ...
        )
}

sortkey <- function(string, values, keys) {
    foo <- function(x) x[[1]]
    values <- as.vector(tapply(values, keys, median))
    string <- as.vector(as.numeric(factor(tapply(string, keys, foo))) - 1)

    string <- string * max(values) * 2
    string + values
}

processData <- function(data, keys) {
    keys <- factor(keys) # no empty levels
    l <- levels(keys)
    n <- length(l)
    result <- data.frame(key = l)
    for(col in colnames(data)) {
        v <- as.vector(data[[col]])
        v2 <- NULL
        if(is.character(v)) {
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
    result
}

sortBy <- function(data, key) {
    o <- sort.list(key)
    if(is.data.frame(data)) {
        for(col in colnames(data)) {
            data[col] <- data[[col]][o]
        }
    } else {
        data <- data[o]
    }
    data
}

par(family="serif")

sse    <- read.table("@sse_datafile@")
simple <- read.table("@simple_datafile@")
lrb    <- if(nchar("@lrb_datafile@") > 0) read.table("@lrb_datafile@") else data.frame()

# vim: sw=4 et filetype=r sts=4 ai
