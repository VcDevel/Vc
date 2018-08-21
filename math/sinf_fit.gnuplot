log2(x) = log(x)/log(2)
exponent(x) = floor(log2(abs(x)))
mantissa(x) = int(abs(x) * 2**(23-exponent(x)) + .5) * 2**-23
round_to_float(x) = sgn(x) * mantissa(x) * 2**exponent(x)
sqr(x) = round_to_float(x * x)
ulp(x) = x == 0 ? 0 : 2**(exponent(x)-23)

b = round_to_float(0.166667)
c = round_to_float(1/5!)
d = round_to_float(1/7!)
e = round_to_float(1/9!)

sinf_fit(x) = \
  round_to_float( \
    round_to_float( \
      round_to_float( \
        round_to_float( \
        round_to_float( \
          round_to_float(0 \
                       + round_to_float(e)) * sqr(x) \
                       - round_to_float(d)) * sqr(x) \
                       + round_to_float(c)) * sqr(x) \
                       - round_to_float(b)) * sqr(x) \
                       + 1) * x);

fit sinf_fit(x) 'sincosd.dat' using 1:2 via b,c,d,e
#plot 'sincosd.dat' using 1:($2-sinf_fit($1)),ulp(sin(x))
plot 'sincosd.dat' using 1:(sinf_fit($1)-$2) with dots, '' using 1:(.5*ulp($2)) with dots, '' using 1:(-.5*ulp($2)) with dots
#pause -1

# vim: ft=gnuplot
