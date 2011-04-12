#!/usr/bin/env ruby

require 'pp'

benchmarks = {
    'memio' => {
        :pageColumn => 'MemorySize',
        :groupColumn => 'benchmark.name',
        :titleColumn => 'Implementation',
        :clusterColumn => 'datatype',
        :groupTranslation => {
            'read' => 'load',
            'write' => 'store',
            'r/w' => 'load \\& store'
        }
    },
    'arithmetics' => {
        :groupColumn => 'benchmark.name',
        :titleColumn => 'Implementation',
        :clusterColumn => 'datatype'
    },
    'flops' => {
        :titleColumn => 'Implementation'
    },
    'gather' => {
    },
    'mask' => {
    },
    'compare' => {
    },
    'math' => {
    },
    'dhryrock' => {
    },
    'whetrock' => {
    },
    'mandelbrotbench' => {
    }
}

class ColumnFilter
    def initialize(grep, name_column)
        @name_column = name_column.map {|i| [i[0], i[1], -1]}
        @grep = grep.map do |v|
            if v.is_a? String then
                "\"#{v}\""
            else
                v
            end
        end
    end

    def prepare(colnames)
        @name_column.map! do |col|
            col[2] = colnames.index col[1]
            fail if col[2] === nil
            col
        end
    end

    def process(row)
        if (@grep.map{|i| row.include? i}).reduce(:&)
            @name_column.each do |col|
                yield(col[0], row[col[2]])
            end
        end
    end

    def headers
        @name_column.map { |col| col[0] }
    end
end

class TableRow
    def initialize(keys)
        @keys = keys
        @data = Hash.new
    end

    def []=(key, value)
        @data[key] = value
    end

    def [](key)
        @data[key]
    end

    def ==(rhs)
        fail unless rhs.is_a? Array
        @keys == rhs
    end

    def fields(headers)
        @keys + headers[@keys.size..-1].map do |h|
            self[h]
        end
    end
end

class DataParser
    def parseFields(fields)
        r = Array.new
        fields.each do |f|
            if f =~ /^"(.*)"$/
                r.push $~[1]
            elsif f =~ /^\d*$/
                r.push f.to_i
            else
                r.push f.to_f
            end
        end
        return r;
    end

    def initialize(bench)
        @bench = bench
        @impl = Array.new
        @data = Array.new
        @colnames = Array.new
        Dir.glob("#{bench}_*.dat").each do |filename|
            dat = File.new(filename, "r")
            versionline = dat.readline.strip.match /^Version (\d+)$/
            tmp = $~[1].to_i
            colheads = dat.readline.strip[1..-2]
            if @version === nil
                @version = tmp
                @colnames = colheads.split("\"\t\"")
            else
                fail if @version != tmp
                fail if @colnames != colheads.split("\"\t\"")
            end
            impl = '"' + filename[bench.length + 1..-5] + '"'
            @impl << impl
            dat.readlines.each do |line|
                @data.push(line.strip.split("\t") + [impl])
            end
        end
        @colnames << "Implementation"
    end

    def empty?
        return @data.empty?
    end

    def write(keys, columnFilters)
        s = ''
        columnFilters.each {|cf| cf.prepare(@colnames)}
        # { ['half L1'] => { 'read' => 1.2, 'write' => 2.3 }, ... }

        keyIndexes = keys.map {|i| @colnames.index i}
        keyIndexes.compact!

        table = Array.new

        @data.each do |row|
            keyValues = keyIndexes.map {|i| row[i]}
            index = table.index keyValues
            if index === nil
                index = table.size
                table[index] = TableRow.new keyValues
            end
            columnFilters.each do |cf|
                cf.process(row) {|name, value| table[index][name] = value}
            end
        end

        headers = keys + (columnFilters.map {|cf| cf.headers}).flatten
        s << '"' << headers.join("\"\t\"") << "\"\n"
        table.each do |row|
            fields = row.fields headers
            s << fields.join("\t") << "\n"
        end
        return s
    end

    def list(columnname)
        i = @colnames.index(columnname)
        if i
            contents = @data.map { |x| x[i] }
            contents.uniq.map { |x| x[1..-2] }
        else
            nil
        end
    end

    attr_reader :version, :colnames
end

gnuplot = IO.popen("gnuplot", 'w')
#gnuplot = STDOUT
gnuplot.print <<EOF
set style line  1 lc rgbcolor "#CCCCCC"
set grid y ls 1
set autoscale y

set style line  1 lc rgbcolor "#9F2020"
set style line  2 lc rgbcolor "#409494"
set style line  3 lc rgbcolor "#949440"
set style line  4 lc rgbcolor "#20209F"
set style line  5 lc rgbcolor "#209F20"
set style line  6 lc rgbcolor "#9F2020"
set style line  7 lc rgbcolor "#409494"
set style line  8 lc rgbcolor "#949440"
set style line  9 lc rgbcolor "#20209F"
set style line 10 lc rgbcolor "#209F20"
set style line 11 lc rgbcolor "#9F2020"
set style line 12 lc rgbcolor "#409494"
set style line 13 lc rgbcolor "#949440"
set style line 14 lc rgbcolor "#20209F"
set style line 15 lc rgbcolor "#209F20"
set style increment user

set terminal pdf color noenhanced font "CM Sans,5" size 18cm,7cm
set pointsize 0.6
set style histogram clustered gap 1
set style data histogram
set style fill transparent solid 0.85 noborder
set key right top
set border 10       # only lines on the left and right
set boxwidth 0.91      # width of the bars
set xtics scale 0 # no tics on below histogram bars
set ytics scale 0
set y2tics scale 0
set bmargin 3.5

#set yrange [0:36]
#set xtics nomirror rotate by -40 scale 0
EOF

benchmarks.each do |bench, opt|
    dp = DataParser.new(bench)
    next if dp.empty?

    opt[:outname] = bench if opt[:outname] === nil
    gnuplot.print <<EOF
set output "#{opt[:outname]}.pdf"
set ylabel "Bytes / Cycle"
EOF
    col = if dp.version == 3 then 'Byte/Cycles' else 'Bytes/Cycle' end

    pageNames = dp.list(opt[:pageColumn])
    pageNames = [bench] if pageNames === nil
    groupNames = dp.list(opt[:groupColumn])
    groupNames = [''] if groupNames === nil
    titleNames = dp.list(opt[:titleColumn])
    clusterNames = dp.list(opt[:clusterColumn])

    pageNames.each do |page|
        data = ''
        gnuplot_print = Array.new
        at = 0
        groupNames.each do |group|
            filters = Array.new
            titleNames.each do |title|
                filters \
                    << ColumnFilter.new([page, group, title],
                                        [[title, col]]) \
                    << ColumnFilter.new([page, group, title],
                                        [[title + ' stddev', col + '_stddev']])
            end
            tmp = dp.write([opt[:clusterColumn]], filters)
            titleNames.size.times { data << tmp << "e\n" }

            groupName = if opt[:groupTranslation]
                opt[:groupTranslation][group]
            else
                group
            end
            gnuplot_print << "  newhistogram \" \\r#{groupName}\" lt 1 at #{at}"
            1.upto(titleNames.size) do |i|
                i2 = i * 2
                gnuplot_print << "  '-' using #{i2}:xtic(1) " +
                if at == 0
                    "title columnheader(#{i2})"
                else
                    "notitle"
                    "title columnheader(#{i2})" # required because of a gnuplot bug
                end
            end
            at += clusterNames.size + 2.0 / (titleNames.size + 1)
        end
        gnuplot.print <<EOF
set title "#{page}"
plot \
#{gnuplot_print.join(", \\\n")}
#{data}
EOF
    end
end

# vim: sw=4 et
