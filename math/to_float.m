function r = to_float(m, e)
  r = (0x800000 + m) * 2^(e-23);
endfunction