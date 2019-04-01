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
    static inline constexpr int value_impl() {
        if constexpr(std::experimental::__is_vector_type_v<T>)
            return std::experimental::_VectorTraits<T>::_S_width;
        else
            return 1;
    }
    static inline constexpr int value = value_impl();
};
template <class T, class Abi> struct size<std::experimental::simd<T, Abi>> {
  static inline constexpr int value = std::experimental::simd_size_v<T, Abi>;
};
template <class T> constexpr inline int size_v = size<T>::value;

template <bool Latency, class T> double benchmark();
template <bool Latency, class T, class> double benchmark();
template <bool Latency, class T, class, class> double benchmark();
template <bool Latency, class T, class, class, class> double benchmark();

struct Times {
    double lat, thr;
    constexpr operator bool() const { return true; }
};

template <class T, class... ExtraFlags, class = decltype(T())>
Times bench_lat_thr(const char* id, const Times ref = {})
{
    static constexpr char red[] = "\033[1;40;31m";
    static constexpr char green[] = "\033[1;40;32m";
    static constexpr char normal[] = "\033[0m";

    const double lat = benchmark<true, T, ExtraFlags...>();
    const double thr = benchmark<false, T, ExtraFlags...>();
    std::cout << id << std::setprecision(3) << std::setw(15) << lat;
    if (ref.lat > 0) {
        const double speedup = ref.lat * size_v<T> / lat;
        if (speedup >= size_v<T> * 0.95 && speedup >= 1.5) {
            std::cout << green;
        }
        if (speedup < 0.95) {
            std::cout << red;
        }
        std::cout << std::setw(12) << speedup << normal;
    } else {
        std::cout << "            ";
    }
    std::cout << std::setw(15) << thr;
    if (ref.lat > 0) {
        const double speedup = ref.thr * size_v<T> / thr;
        if (speedup >= size_v<T> * 0.95 && speedup >= 1.5) {
            std::cout << green;
        }
        if (speedup < 0.95) {
            std::cout << red;
        }
        std::cout << std::setw(12) << speedup << normal;
    }
    std::cout << std::endl;
    return {lat, thr};
}

template <class, class...> bool bench_lat_thr(...) { return false; }

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

template <class T, class... ExtraFlags> void bench_all()
{
    using namespace std::experimental::simd_abi;
    using std::experimental::simd;
    constexpr std::size_t N = 8 + 18 + (1 + ... + sizeof(ExtraFlags::name));
    char id[N];
    std::memset(id, ' ', N - 1);
    id[N - 1] = '\0';
    std::strncpy(id + 18, "TYPE", 4);
    print_header(id);
    std::strncpy(id + 18, "    ", 4);
    char* const typestr = id;
    char* const abistr  = id + 8;
    id[6]         = ',';
    char* extraflags = id + 8 + 18;
    if constexpr (std::is_same_v<T, float>) {
        std::strncpy(typestr, " float", 6);
    } else if constexpr (std::is_same_v<T, double>) {
        std::strncpy(typestr, "double", 6);
    } else if constexpr (std::is_same_v<T, long double>) {
        std::strncpy(typestr, "ldoubl", 6);
    } else if constexpr (std::is_same_v<T, long long>) {
        std::strncpy(typestr, " llong", 6);
    } else if constexpr (std::is_same_v<T, unsigned long long>) {
        std::strncpy(typestr, "ullong", 6);
    } else if constexpr (std::is_same_v<T, long>) {
        std::strncpy(typestr, "  long", 6);
    } else if constexpr (std::is_same_v<T, unsigned long>) {
        std::strncpy(typestr, " ulong", 6);
    } else if constexpr (std::is_same_v<T, int>) {
        std::strncpy(typestr, "   int", 6);
    } else if constexpr (std::is_same_v<T, unsigned>) {
        std::strncpy(typestr, "  uint", 6);
    } else if constexpr (std::is_same_v<T, short>) {
        std::strncpy(typestr, " short", 6);
    } else if constexpr (std::is_same_v<T, unsigned short>) {
        std::strncpy(typestr, "ushort", 6);
    } else if constexpr (std::is_same_v<T, char>) {
        std::strncpy(typestr, "  char", 6);
    } else if constexpr (std::is_same_v<T, signed char>) {
        std::strncpy(typestr, " schar", 6);
    } else if constexpr (std::is_same_v<T, unsigned char>) {
        std::strncpy(typestr, " uchar", 6);
    } else {
        std::strncpy(typestr, "??????", 6);
    }
    {
        [&](const std::initializer_list<int>&) {}({[&]() {
            std::strncpy(extraflags, ExtraFlags::name, sizeof(ExtraFlags::name) - 1);
            extraflags += sizeof(ExtraFlags::name);
            return 0;
        }()...});
    }
    const auto ref = bench_lat_thr<T, ExtraFlags...>(id);
    std::strncpy(abistr, "simd_abi::scalar  ", 18);
    bench_lat_thr<simd<T, scalar>, ExtraFlags...>(id, ref);
    std::strncpy(abistr, "simd_abi::__sse   ", 18);
    if (bench_lat_thr<simd<T, __sse>, ExtraFlags...>(id, ref)) {
        using V [[gnu::vector_size(16)]] = T;
        if constexpr (alignof(V) == sizeof(V)) {
            std::strncpy(abistr, "vector_size(16)   ", 18);
            bench_lat_thr<V, ExtraFlags...>(id, ref);
        }
    }
    std::strncpy(abistr, "simd_abi::__avx   ", 18);
    if (bench_lat_thr<simd<T, __avx>, ExtraFlags...>(id, ref)) {
        using V [[gnu::vector_size(32)]] = T;
        if constexpr (alignof(V) == sizeof(V)) {
            std::strncpy(abistr, "vector_size(32)   ", 18);
            bench_lat_thr<V, ExtraFlags...>(id, ref);
        }
    }
    std::strncpy(abistr, "simd_abi::__avx512", 18);
    if (bench_lat_thr<simd<T, __avx512>, ExtraFlags...>(id, ref)) {
        using V [[gnu::vector_size(64)]] = T;
        if constexpr (alignof(V) == sizeof(V)) {
            std::strncpy(abistr, "vector_size(64)   ", 18);
            bench_lat_thr<V, ExtraFlags...>(id, ref);
        }
    }
    char sep[N + 2 * 15 + 2 * 12];
    std::memset(sep, '-', sizeof(sep) - 1);
    sep[sizeof(sep) - 1] = '\0';
    std::cout << sep << std::endl;
}

template <long Iterations, class F, class... Args>
double time_mean(F&& fun, Args&&... args)
{
    unsigned int tmp;
    long i = Iterations;
    const auto start = __rdtscp(&tmp);
    for (; i; --i) {
        fun(std::forward<Args>(args)...);
    }
    const auto end       = __rdtscp(&tmp);
    const double elapsed = end - start;
    return elapsed / Iterations;
}

template <typename T, typename... Ts> void fake_modify(T& x, Ts&... more)
{
    if constexpr (sizeof(T) >= 16 || std::is_floating_point_v<T>) {
        asm volatile("" : "+x"(x));
    } else {
        asm volatile("" : "+g"(x));
    }
    if constexpr (sizeof...(Ts) > 0) {
        fake_modify(more...);
    }
}

template <typename T, typename... Ts> void fake_read(const T& x, const Ts&... more)
{
    if constexpr (sizeof(T) >= 16 || std::is_floating_point_v<T>) {
        asm volatile("" ::"x"(x));
    } else {
        asm volatile("" ::"g"(x));
    }
    if constexpr (sizeof...(Ts) > 0) {
        fake_read(more...);
    }
}

#define MAKE_VECTORMATH_OVERLOAD(name)                                                   \
    template <class T, class... More, class VT = std::experimental::_VectorTraits<T>>    \
    T hypot(T a, More... more)                                                           \
    {                                                                                    \
        T r;                                                                             \
        for (int i = 0; i < VT::_S_width; ++i) {                                         \
            r[i] = std::hypot(a[i], more[i]...);                                         \
        }                                                                                \
        return r;                                                                        \
    }

#endif  // VC_BENCHMARKS_BENCH_H_
