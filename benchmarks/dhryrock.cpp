/*  This file is part of the Vc library.

    Copyright (C) 2009 Matthias Kretz <kretz@kde.org>

    Vc is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    Vc is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with Vc.  If not, see <http://www.gnu.org/licenses/>.

*/

#include <Vc/Vc>
#include "benchmark.h"
#include "random.h"
#include <limits>

using namespace Vc;

enum {
    MemorySize = 1024 * 1024 * 64 // 64 MiB
};

void *blackHolePtr = 0;

template<typename V>
static void doBlah()
{
    typedef typename V::EntryType T;
    typedef typename V::IndexType I;
    typedef typename V::Mask M;

    const T two = 2;
    const T divider = std::numeric_limits<T>::max() / 1500;
    const T bound = 1000;

    enum {
        ArraySize = MemorySize / sizeof(V)
    };
    union {
        V *v;
        T *t;
    } mem = { new V[ArraySize] };
    for (int i = 0; i < ArraySize; ++i) {
        mem.v[i] = PseudoRandom<V>::next();
    }
    V t = mem.v[0];
    mem.v[0].makeZero();
    for (int i = 1; i < ArraySize; ++i) {
        t = (t * mem.v[i] + t) / divider;
        t(t >= bound || t < 0) = two;
        mem.v[i] = t;
        t(t == 0) += V(One);
    }
    M mask;
    T data[1000];
    for (I i(Vc::Zero); !(mask = i < 1000).isEmpty(); i += I::Size) {
        I i2 = static_cast<I>(V(mem.t, i, mask));
        V v(data, i2, mask, Vc::Zero);
        v = v * two + t;
        v.scatter(data, i2, mask);
    }
    asm volatile(""::"r"(&data[0]));
    blackHolePtr = mem.v;
}

template<typename V>
static double opsFactor()
{
    enum {
        ArraySize = MemorySize / sizeof(V)
    };
    return V::Size * (ArraySize * (5 + 10) + (1000 + V::Size - 1) / V::Size * 5);
}

int bmain()
{
    Benchmark timer("DhryRock", opsFactor<int_v>() + opsFactor<uint_v>() + opsFactor<short_v>() + opsFactor<ushort_v>(), "Op");
    while (timer.wantsMoreDataPoints()) {
        timer.Start();
        doBlah<int_v>();
        delete[] static_cast<int_v *>(blackHolePtr);
        doBlah<uint_v>();
        delete[] static_cast<uint_v *>(blackHolePtr);
        doBlah<short_v>();
        delete[] static_cast<short_v *>(blackHolePtr);
        doBlah<ushort_v>();
        delete[] static_cast<ushort_v *>(blackHolePtr);
        timer.Stop();
    }
    timer.Print();
    return 0;
}
