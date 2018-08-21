function x = fold_pi4_single(x)
  x = round_to_float(x);
  expected = round_to_float(x - floor(x / (pi/2) + .5)*(pi/2));
  y = round_to_float(floor(x * round_to_float(2/pi) + .5)) * 2;
  printf("y = %i = %s\n", y, hexfloat(y));
  while(y > 0)
    p2 = 2^floor(log2(y));
    y -= p2;
    diff = p2 * round_to_float(pi/4);
    x -= diff;
    printf("(x -= %s) -> %s\n", hexfloat(diff), hexfloat(x));
  endwhile
  diff = p2 * round_to_float(pi/4 - round_to_float(pi/4));
  x -= diff;
  printf("(x -= %s) -> %s\n", hexfloat(diff), hexfloat(x));
  printf("expected result:         %s\n", hexfloat(expected));
endfunction
