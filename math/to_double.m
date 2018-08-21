function r = to_double(m, e)
  r = (0x10000000000000 + m) * 2^(e-52);
endfunction