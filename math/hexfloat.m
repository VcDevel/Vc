function r = hexfloat(x)
  if (x == 0)
    r = "0x0.000000p0";
    return;
  endif
  if (signbit(x))
    s = "-";
  else
    s = "";
  endif
  x = abs(x);
  e = floor(log2(x));
  m = roundb((x*2^-e - 1) * 2^23) * 2;
  r = sprintf("%s0x1.%06xp%i", s, m, e);
endfunction
