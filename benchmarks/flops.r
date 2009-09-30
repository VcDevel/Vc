flopsProcessData <- function(d) {
    d$GFLOP_per_sec_Real <- d$FLOP_per_sec_Real / 1000000000
    d <- processData(d, paste(d$benchmark.arch, d$benchmark.name, sep=", "),
        pchkey = "benchmark.name", colorkey = "benchmark.arch")
    d$key <- d$benchmark.arch
    d
}

for(data in list(rbind(sse, simple, lrb))) {
    data <- flopsProcessData(data)

    chart <- mychart4(data, data$benchmark.name,
        column = "FLOP_per_cycle", xlab = "FLOP per Cycle", main = "Multiply - Add (Peak FLOP)")
    printValuesInChart(chart)

    chart <- mychart4(data, data$benchmark.name,
        column = "GFLOP_per_sec_Real", xlab = "GFLOP per Second", main = "Multiply - Add (Peak FLOP)")
    printValuesInChart(chart)

    chart <- mybarplot(data, splitcolumn = "benchmark.name",
        column = "FLOP_per_cycle", ylab = "FLOP per Cycle", main = "Multiply - Add (Peak FLOP)")
}

# vim: sw=4 et filetype=r sts=4 ai
