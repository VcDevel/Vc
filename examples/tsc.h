/*
    Copyright (C) 2009-2014 Matthias Kretz <kretz@kde.org>

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) version 3.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Library General Public License for more details.

    You should have received a copy of the GNU Library General Public License
    along with this library; see the file COPYING.LIB.  If not, write to
    the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
    Boston, MA 02110-1301, USA.

*/

#ifndef TSC_H
#define TSC_H

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
#if defined VC_IMPL_MIC || defined __MIC__
    asm volatile("xor %%eax,%%eax\n\tcpuid\n\trdtsc" : "=a"(m_start.b[0]), "=d"(m_start.b[1]) :: "ebx", "ecx" );
#elif defined _MSC_VER
	unsigned int tmp;
    m_start.a = __rdtscp(&tmp);
#else
    asm volatile("rdtscp" : "=a"(m_start.b[0]), "=d"(m_start.b[1]) :: "ecx" );
#endif
}

inline void TimeStampCounter::stop()
{
#if defined VC_IMPL_MIC || defined __MIC__
    asm volatile("xor %%eax,%%eax\n\tcpuid\n\trdtsc" : "=a"(m_end.b[0]), "=d"(m_end.b[1]) :: "ebx", "ecx" );
#elif defined _MSC_VER
	unsigned int tmp;
    m_end.a = __rdtscp(&tmp);
#else
    asm volatile("rdtscp" : "=a"(m_end.b[0]), "=d"(m_end.b[1]) :: "ecx" );
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
    for (int digits = std::log10(c); digits > 0; digits -= 3) {
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

#endif // TSC_H
