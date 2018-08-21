function r = hexdouble(x)
  if (signbit(x))
    s = "-";
  else
    s = "";
  endif
  x = abs(x);
  e = floor(log2(x));
  m = round((x*2^-e - 1) * 2^52);
  r = sprintf("%s0x1.%013xp%i", s, m, e);
endfunction