function x = round_to_float(x)
  if(x == 0)
    return;
  endif
  s = sign(x);
  x = abs(x);
  e = floor(log2(x));
  m = roundb((x.*2.^-e - 1) .* 2.^23);
  x = s .* (m .* 2.^-23 + 1) .* 2.^e;
endfunction
