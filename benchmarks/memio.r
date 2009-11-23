memioProcessData <- function(d) {
    d <- processData(d, paste(d$benchmark.arch, d$MemorySize, d$benchmark.name),
        pchkey = "benchmark.arch", colorkey = "MemorySize")
    d$key <- pasteNN(d$MemorySize, d$benchmark.name)
    d
}


for(data in list(rbind(sse, simple), rbind(sse, simple, lrb))) {
    for(data in split(data, data$benchmark.name)) {
        data <- memioProcessData(data)

        sortkey <- function(s, n) {
          s[s == "half L1"] <- 0
          s[s == "L1"]      <- 2
          s[s == "half L2"] <- 4
          s[s == "L2"]      <- 6
          s[s == "half L3"] <- 8
          s[s == "L3"]      <- 10
          s[s == "4x L2"]   <- 12
          s[s == "4x L3"]   <- 14
          n[n == "read"]  <- 0
          n[n == "write"] <- 1
          n[n == "r/w"] <- 1.5
          20 - (as.numeric(s) + as.numeric(n))
        }
        mychart4(data, data$benchmark.arch,
            orderfun = function(d) order(sortkey(d$MemorySize, d$benchmark.name)),
            column = "Byte_per_cycle", xlab = "Byte per Cycle",
            main = paste(data$benchmark.name[[1]], "Throughput"),
            legendcol = "benchmark.arch")
        abline(v= 0, lty = "dashed", col = hsv(s = 1, v = 0, alpha = 0.3))
        abline(v= 5, lty = "dashed", col = hsv(s = 1, v = 0, alpha = 0.3))
        abline(v=10, lty = "dashed", col = hsv(s = 1, v = 0, alpha = 0.3))
        abline(v=15, lty = "dashed", col = hsv(s = 1, v = 0, alpha = 0.3))
        #abline(v=16, lty = "dashed", col = hsv(s = 1, v = 0, alpha = 0.7))

        #data$key <- paste(data$MemorySize)
        #for(part in split(data, data$benchmark.name)) {
            #mybarplot(part, "benchmark.arch", column = "Byte_per_cycle", ylab = "Byte per Cycle",
                #main = paste("L1 I/O (", part$benchmark.name[[1]], ")", sep=""),
                #orderfun = function(d) sort.list(logicalSortkeyForDatatype(d$MemorySize)),
                #maxlaboffset = 1)
        #}
    }
}

sse    <- memioProcessData(sse)
lrb    <- memioProcessData(lrb)
simple <- memioProcessData(simple)

#plotSpeedup(sse, simple, lrb, main = "L1 I/O",
    #speedupColumn = "Byte_per_cycle",
    #datafun = function(d, ref) list(key = d$MemorySize)
    #)

# vim: sw=4 et filetype=r sts=4 ai
