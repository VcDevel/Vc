function y = mycos (x)
  x2 = x.*x;
  y = 0;
  y = y .* x2 + to_double(0xAC00000000000, -45);
  y = y .* x2 - to_double(0x9394000000000, -37);
  y = y .* x2 + to_double(0x1EED8C0000000, -29);
  y = y .* x2 - to_double(0x27E4FB7400000, -22);
  y = y .* x2 + to_double(0xA01A01A018000, -16);
  y = y .* x2 - to_double(0x6C16C16C16C00, -10);
  y = y .* x2 + to_double(0x5555555555554,  -5);
  y = y .* (x2 .* x2) - .5 * x2 + 1.;
endfunction
