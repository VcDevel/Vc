function y = sin_pi4(_x)
  x = _x - pi/4;
  y =          1/(sqrt(2)*factorial(8));
  y = y .* x - 1/(sqrt(2)*factorial(7));
  y = y .* x - 1/(sqrt(2)*factorial(6));
  y = y .* x + 1/(sqrt(2)*factorial(5));
  y = y .* x + 1/(sqrt(2)*factorial(4));
  y = y .* x - 1/(sqrt(2)*factorial(3));
  y = y .* x - 1/(sqrt(2)*factorial(2));
  y = y .* x + 1/(sqrt(2)*factorial(1));
  y = y .* x + 1/sqrt(2);
endfunction