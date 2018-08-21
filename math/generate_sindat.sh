#!/bin/zsh

cd "${0%/*}"

function usage() {
  cat <<EOF
Usage: $0

Options:
  --help|-h        this message

EOF
}

while (( $# > 0 )); do
  case "$1" in
    --help|-h) usage; exit ;;
  esac
  shift
done

octave <<EOF
ff=fopen("sincosf.dat", "w");
fd=fopen("sincosd.dat", "w");

printf("Generating 100000 dp & sp sincos values [0, π/4]     ");
for i = 1:100000
  x=i * pi / 400000;
  fprintf(fd, "%.60f\t%.60f\t%.60f\n", x, sin(x), cos(x));
  x=round_to_float(x);
  fprintf(ff, "%.40f\t%.50f\t%.50f\n", x, sin(x), cos(x));
  printf("\033[4D%3i%%", floor(i/1000));
endfor
EOF

# vim: sw=2 et
