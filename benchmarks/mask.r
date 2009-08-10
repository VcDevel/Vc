colors  <- rainbow(3, v=0.5)
sse$color <- colors[[3]]
simple$color <- colors[[1]]
lrb$color <- colors[[2]]

for(data in list(rbind(sse, simple, lrb), rbind(sse, simple))) {
    data <- processData(data, paste(data$benchmark.name, data$datatype, data$benchmark.arch))

    data <- sortBy(data, data$key)
    data$key <- paste(data$datatype, data$benchmark.arch)
    data$pch <- (21:30)[as.integer(factor(data$benchmark.name))]

    mychart4(data, data$benchmark.name, main = "Masked Operations")
}

# vim: sw=4 et filetype=r sts=4 ai
