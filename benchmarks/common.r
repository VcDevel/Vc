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

sortkey <- function(string, values, keys) {
    foo <- function(x) x[[1]]
    values <- as.vector(tapply(values, keys, median))
    string <- as.vector(as.numeric(factor(tapply(string, keys, foo))) - 1)

    string <- string * max(values) * 2
    string + values
}

par(family="serif")

sse    <- read.table("@sse_datafile@")
simple <- read.table("@simple_datafile@")
lrb    <- read.table("@lrb_datafile@")

# vim: sw=4 et filetype=r sts=4 ai
