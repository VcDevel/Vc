/*  This file is part of the Vc library. {{{
Copyright © 2019 Matthias Kretz <kretz@kde.org>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the names of contributing organizations nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

}}}*/

#ifndef VC_BENCHMARKS_BENCH_H_
#define VC_BENCHMARKS_BENCH_H_

#include "../experimental/simd"
#include <iostream>

template <class T> struct size {
  static inline constexpr int value = 1;
};
template <class T, class Abi> struct size<std::experimental::simd<T, Abi>> {
  static inline constexpr int value = std::experimental::simd_size_v<T, Abi>;
};
template <class T> constexpr inline int size_v = size<T>::value;

template <bool Latency, class T> double benchmark();

struct Times {
    double lat, thr;
};

template <class T, class = decltype(T())>
Times bench_lat_thr(const char* id, const Times ref = {})
{
    const double lat = benchmark<true, T>();
    const double thr = benchmark<false, T>();
    std::cout << id << std::setprecision(3) << std::setw(15) << lat << std::setw(12);
    if (ref.lat > 0)
        std::cout << ref.lat * size_v<T> / lat;
    else
        std::cout << ' ';
    std::cout << std::setw(15) << thr;
    if (ref.lat > 0)
        std::cout << std::setw(12) << ref.thr * size_v<T> / thr;
    std::cout << std::endl;
    return {lat, thr};
}

template <class> void bench_lat_thr(...) {}

template <std::size_t N> using cstr = char[N];

template <std::size_t N>
void print_header(const cstr<N> &id_name)
{
    std::cout << id_name
        << std::setw(15) << "Latency"
        << std::setw(12) << "Speedup"
        << std::setw(15) << "Throughput"
        << std::setw(12) << "Speedup" << '\n';

    char pad[N] = {};
    std::memset(pad, ' ', N - 1);
    pad[N - 1] = '\0';
    std::cout << pad
        << std::setw(15) << "[cycles/call]"
        << std::setw(12) << ""
        << std::setw(15) << "[cycles/call]"
        << std::setw(12) << "" << '\n';
}

template <class T> void bench_all()
{
    using namespace std::experimental::simd_abi;
    using std::experimental::simd;
    constexpr std::size_t N = 23;
    char id[N];
    std::strncpy(id, "                  TYPE", N);
    print_header(id);
    std::memset(id, ' ', N - 1);
    if constexpr (std::is_same_v<T, float>) {
        std::strncpy(id + 5, " float", 6);
    } else if constexpr (std::is_same_v<T, double>) {
        std::strncpy(id + 5, "double", 6);
    } else if constexpr (std::is_same_v<T, long double>) {
        std::strncpy(id + 5, "ldoubl", 6);
    } else if constexpr (std::is_same_v<T, int>) {
        std::strncpy(id + 5, "   int", 6);
    } else if constexpr (std::is_same_v<T, unsigned>) {
        std::strncpy(id + 5, "  uint", 6);
    } else if constexpr (std::is_same_v<T, short>) {
        std::strncpy(id + 5, " short", 6);
    } else if constexpr (std::is_same_v<T, unsigned short>) {
        std::strncpy(id + 5, "ushort", 6);
    } else if constexpr (std::is_same_v<T, char>) {
        std::strncpy(id + 5, "  char", 6);
    } else if constexpr (std::is_same_v<T, signed char>) {
        std::strncpy(id + 5, " schar", 6);
    } else if constexpr (std::is_same_v<T, unsigned char>) {
        std::strncpy(id + 5, " uchar", 6);
    } else {
        std::strncpy(id + 5, "??????", 6);
    }
    const auto ref = bench_lat_thr<T>(id);
    std::strncpy(id, "simd<", 5);
    std::strncpy(id + 11, ",   scalar>", 11);
    bench_lat_thr<simd<T, scalar>>(id, ref);
    std::strncpy(id + 13, "   __sse>", 9);
    bench_lat_thr<simd<T, __sse>>(id, ref);
    std::strncpy(id + 13, "   __avx>", 9);
    bench_lat_thr<simd<T, __avx>>(id, ref);
    std::strncpy(id + 13, "__avx512>", 9);
    bench_lat_thr<simd<T, __avx512>>(id, ref);
    char sep[N + 2 * 15 + 2 * 12];
    std::memset(sep, '-', sizeof(sep) - 1);
    sep[sizeof(sep) - 1] = '\0';
    std::cout << sep << std::endl;
}

template <long Iterations, class F> double time_mean(F&& fun)
{
    unsigned int tmp;
    const auto start = __rdtscp(&tmp);
    for (int i = 0; i < Iterations; ++i) {
        fun();
    }
    const auto end       = __rdtscp(&tmp);
    const double elapsed = end - start;
    return elapsed / Iterations;
}

#endif  // VC_BENCHMARKS_BENCH_H_
