function y = sinf (x)
  x2 = x.*x;
  y = 0;
  #y = (y - to_float(0x573F9F, -41)) .* x2;
  #y = (y + to_float(0x309231, -33)) .* x2;
  #y = (y - to_float(0x57322B, -26)) .* x2;
  #y = (y + to_float(0x38E000, -19)) .* x2;
  y = (y - to_float(0x4E6000, -13)) .* x2;
  y = (y + to_float(0x088880,  -7)) .* x2;
  y = (y - to_float(0x2AAAAB,  -3)) .* x2;
  y = (y + 1) .* x;
endfunction
