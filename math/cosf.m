function y = cosf (x)
  x2 = x.*x;
  y = 0;
  y = y .* x2 + to_float(0x573F9F, -45);
  y = y .* x2 - to_float(0x49CBA5, -37);
  y = y .* x2 + to_float(0x0F76C7, -29);
  y = y .* x2 - to_float(0x13F27E, -22);
  y = y .* x2 + to_float(0x500D01, -16);
  y = y .* x2 - to_float(0x360B61, -10);
  y = y .* x2 + to_float(0x2AAAAB,  -5);
  y = y .* (x2 .* x2) - .5 * x2 + 1.;
endfunction
