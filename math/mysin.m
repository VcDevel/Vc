function y = mysin (x)
  x2 = x.*x;
  y = 0;
  y = (y - to_double(0xACF0000000000 + 0, -41)) .* x2;
  y = (y + to_double(0x6124400000000 + 0, -33)) .* x2;
  y = (y - to_double(0xAE64567000000 + 0, -26)) .* x2;
  y = (y + to_double(0x71DE3A5540000 + 0, -19)) .* x2;
  y = (y - to_double(0xa01a01a01a000 + 0, -13)) .* x2;
  y = (y + to_double(0x1111111111110 + 0, -7)) .* x2;
  y = (y - to_double(0x5555555555555 + 0, -3)) .* x2;
  y = (y + 1) .* x;
endfunction
