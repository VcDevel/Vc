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

#include "bench.h"

template <bool Latency, class T> double benchmark()
{
    T a = 23;
    T b = 7;
    return time_mean<50'000'000>([&]() {
        if constexpr (sizeof(T) >= 16) {
            asm volatile("" : "+x"(a), "+x"(b));
        } else {
            asm volatile("" : "+g"(a), "+g"(b));
        }
        T r = a / b;
        if constexpr (Latency)
            a = r;
        else if constexpr (sizeof(T) >= 16)
            asm volatile("" :: "x"(r));
        else
            asm volatile("" :: "g"(r));
    });
}

int main()
{
    bench_all<signed char>();
    bench_all<unsigned char>();
    bench_all<signed short>();
    bench_all<unsigned short>();
    bench_all<signed int>();
    bench_all<unsigned int>();
}
