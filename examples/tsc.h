/*  This file is part of the Vc library. {{{
Copyright © 2009-2017 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_TSC_H_
#define VC_TSC_H_

#ifdef _MSC_VER
#include <intrin.h>
#pragma intrinsic(__rdtsc)
#endif

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

class TimeStampCounter
{
    public:
        void start();
        void stop();
        unsigned long long cycles() const;

    private:
        union Data {
            unsigned long long a;
            unsigned int b[2];
        } m_start, m_end;
};

inline void TimeStampCounter::start()
{
#if defined __MIC__
    asm volatile("xor %%eax,%%eax\n\tcpuid\n\trdtsc" : "=a"(m_start.b[0]), "=d"(m_start.b[1]) :: "ebx", "ecx" );
#elif defined _MSC_VER
	unsigned int tmp;
    m_start.a = __rdtscp(&tmp);
#elif defined __x86_64__ || defined __i386__
    asm volatile("rdtscp" : "=a"(m_start.b[0]), "=d"(m_start.b[1]) :: "ecx" );
#else
    m_start = {};
#endif
}

inline void TimeStampCounter::stop()
{
#if defined __MIC__
    asm volatile("xor %%eax,%%eax\n\tcpuid\n\trdtsc" : "=a"(m_end.b[0]), "=d"(m_end.b[1]) :: "ebx", "ecx" );
#elif defined _MSC_VER
	unsigned int tmp;
    m_end.a = __rdtscp(&tmp);
#elif defined __x86_64__ || defined __i386__
    asm volatile("rdtscp" : "=a"(m_end.b[0]), "=d"(m_end.b[1]) :: "ecx" );
#else
    m_end = {};
#endif
}

inline unsigned long long TimeStampCounter::cycles() const
{
    return m_end.a - m_start.a;
}

inline std::ostream &operator<<(std::ostream &out, const TimeStampCounter &tsc)
{
    std::ostringstream o;
    auto c = tsc.cycles();
    int blocks[10];
    int n = 0;
    for (int digits = std::ceil(std::log10(c)); digits > 0; digits -= 3) {
        blocks[n++] = c % 1000;
        c /= 1000;
    }
    if (n == 0) {
        return out;
    }
    o.fill('0');
    o << blocks[--n];
    while (n > 0) {
        o << '\'' << std::setw(3) << blocks[--n];
    }
    return out << o.str();
}

#endif  // VC_TSC_H_

// vim: foldmethod=marker
