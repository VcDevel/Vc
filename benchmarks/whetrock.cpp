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
    typedef typename I::Mask IM;

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
        t = Vc::abs(t * mem.v[i] + t) / divider;
        t(t >= bound) = two;
        mem.v[i] = t;
        t(t == 0) += V(One);
    }
    IM mask;
    T data[1000];
    for (I i(Vc::Zero); !(mask = i < 1000).isEmpty(); i += V::Size) {
        M mask2(mask);
        I i2 = static_cast<I>(V(mem.t, i, mask2));
        V v(data, i2, mask2, Vc::Zero);
        v = v * two + t;
        v.scatter(data, i2, mask2);
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
    return V::Size * (ArraySize * (5 + 9) + (1000 + V::Size - 1) / V::Size * 5);
}

int bmain()
{
    Benchmark timer("WhetRock", opsFactor<float_v>() + opsFactor<double_v>(), "Op");
    while (timer.wantsMoreDataPoints()) {
        timer.Start();
        doBlah<float_v>();
        delete[] static_cast<float_v *>(blackHolePtr);
        doBlah<double_v>();
        delete[] static_cast<double_v *>(blackHolePtr);
        timer.Stop();
    }
    timer.Print();
    return 0;
}
