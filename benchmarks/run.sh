#!/bin/bash

no_turbo=/sys/devices/system/cpu/intel_pstate/no_turbo

turn_on() {
  echo "enabling benchmark mode (no clock scaling/turbo)"
  echo performance | sudo tee /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor >/dev/null
  if test -f $no_turbo; then
    echo 1 | sudo tee $no_turbo >/dev/null
  else
    freq=$(cut -d" " -f1,2 /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies)
    freq1=${freq% *}
    freq2=${freq#* }
    test $(($freq2+1000)) -eq $freq1 && \
      echo $freq2 | sudo tee /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_max_freq >/dev/null
  fi
}

turn_off() {
  echo "disabling benchmark mode"
  if test -f $no_turbo; then
    echo powersave | sudo tee /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor >/dev/null
    echo 0 | sudo tee $no_turbo >/dev/null
  else
    freq=$(cut -d" " -f1 /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies)
    echo $freq | sudo tee /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_max_freq >/dev/null
    echo ondemand | sudo tee /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor >/dev/null
  fi
}

name=$1
flags=()
shift
while (($# > 0)); do
    case "$1" in
        -f|-ffast-math|-fast-math)
            flags=(${flags[@]} -ffast-math)
            shift
            ;;
        -finite|-ffinite-math-only|-finite-math-only)
            flags=(${flags[@]} -ffinite-math-only)
            shift
            ;;
        *)
            arch_list="$arch_list $1"
            shift
            ;;
    esac
done
test -z "$arch_list" && arch_list="native westmere k8"

test -r ${name}.cpp
turn_on
for arch in ${arch_list}; do
  CXXFLAGS="-g0 -O2 -std=gnu++17 -march=$arch"

  $CXX $CXXFLAGS ${flags[@]} ${name}.cpp -o ${name} && \
    echo "-march=$arch $flags:" && \
    sudo chrt --fifo 50 ./${name}
done
turn_off

# vim: tw=0
