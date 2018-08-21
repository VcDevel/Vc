function y = sinf_fit (x, c0, c1, c2, c3)
  x2 = x.*x;
  y = 0;
  y = (y - round_to_float(c3)) .* x2;
  y = (y + round_to_float(c2)) .* x2;
  y = (y - round_to_float(c1)) .* x2;
  y = (y + round_to_float(c0)) .* x;
endfunction
