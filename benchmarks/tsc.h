/*
    Copyright (C) 2009 Matthias Kretz <kretz@kde.org>

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

class TimeStampCounter
{
    public:
        void Start();
        void Stop();
        unsigned long long Cycles() const;

    private:
        union Data {
            unsigned long long a;
            unsigned int b[2];
        } start, end;
};

inline void TimeStampCounter::Start()
{
#ifdef _MSC_VER
    start.a = __rdtsc();
#else
    asm volatile("rdtsc" : "=a"(start.b[0]), "=d"(start.b[1]) );
#endif
}

inline void TimeStampCounter::Stop()
{
#ifdef _MSC_VER
    end.a = __rdtsc();
#else
    asm volatile("rdtsc" : "=a"(end.b[0]), "=d"(end.b[1]) );
#endif
}

inline unsigned long long TimeStampCounter::Cycles() const
{
    return end.a - start.a;
}

#endif // TSC_H
