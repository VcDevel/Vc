function r = hexfloat(x)
  if (signbit(x))
    s = "-";
  else
    s = "";
  endif
  x = abs(x);
  e = floor(log2(x));
  m = round((x*2^-e - 1) * 2^23) * 2;
  r = sprintf("%s0x1.%xp%i", s, m, e);
endfunction