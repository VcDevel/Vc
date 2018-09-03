/*  This file is part of the Vc library. {{{
Copyright © 2015 Matthias Kretz <kretz@kde.org>

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

#include "unittest.h"

template <std::size_t N> using Int = std::integral_constant<std::size_t, N>;

template <std::size_t A> struct Foo : public Vc::AlignedBase<A>
{
    static constexpr std::size_t alignment() { return A; }
    int x;
};

std::uintptr_t addressOf(void *ptr)
{
    return reinterpret_cast<std::uintptr_t>(ptr);
}

void *blackhole = nullptr;

TEST_TYPES(A, alignedbase, Int<4>, Int<8>, Int<16>, Int<32>, Int<64>, Int<128>)
{
    using T = Foo<A::value>;
    COMPARE(alignof(T), A::value);
    char padding = 1;
    blackhole = &padding;
    T onStack;
    COMPARE(addressOf(&onStack) & (A::value - 1), 0u);

    std::unique_ptr<T[]> onHeap{new T[3]};
    COMPARE(addressOf(&onHeap[0]) & (A::value - 1), 0u);
    COMPARE(addressOf(&onHeap[1]) & (A::value - 1), 0u);
    COMPARE(addressOf(&onHeap[2]) & (A::value - 1), 0u);
}

template <typename T> struct VectorAligned : public Vc::VectorAlignedBaseT<T>
{
    int x;
};

TEST_TYPES(N, vectoralignedbase,
           Int<4>, Int<8>, Int<16>, Int<32>, Int<64>, Int<128>, Int<256>)
{
    using V = Vc::SimdMaskArray<int, N::value>;
    using T = VectorAligned<V>;
    constexpr auto A = alignof(V);
    COMPARE(alignof(T), A);
    char padding = 1;
    blackhole = &padding;
    T onStack;
    COMPARE(addressOf(&onStack) & (A - 1), 0u);

    std::unique_ptr<T[]> onHeap{new T[3]};
    COMPARE(addressOf(&onHeap[0]) & (A - 1), 0u);
    COMPARE(addressOf(&onHeap[1]) & (A - 1), 0u);
    COMPARE(addressOf(&onHeap[2]) & (A - 1), 0u);
}

template <typename T> struct MemoryAligned : public Vc::MemoryAlignedBaseT<T>
{
    int x;
};

TEST_TYPES(N, memoryalignedbase,
           Int<4>, Int<8>, Int<16>, Int<32>, Int<64>, Int<128>, Int<256>)
{
    using V = Vc::SimdMaskArray<int, N::value>;
    using T = MemoryAligned<V>;
    constexpr auto A = V::MemoryAlignment;
    COMPARE(alignof(T), A);
    char padding = 1;
    blackhole = &padding;
    T onStack;
    COMPARE(addressOf(&onStack) & (A - 1), 0u);

    std::unique_ptr<T[]> onHeap{new T[3]};
    COMPARE(addressOf(&onHeap[0]) & (A - 1), 0u);
    COMPARE(addressOf(&onHeap[1]) & (A - 1), 0u);
    COMPARE(addressOf(&onHeap[2]) & (A - 1), 0u);
}

TEST_TYPES(T, alignedbasealiases,
           float, double, unsigned long long, long long, unsigned long, long,
           unsigned int, int, unsigned short, short, unsigned char, signed char)
{
    using V = Vc::Vector<T>;
    VERIFY(alignof(Vc::VectorAlignedBase) >= alignof(V));
    VERIFY(alignof(Vc::MemoryAlignedBase) >= V::MemoryAlignment);
    VERIFY(Vc::VectorAlignment >= alignof(V));
    VERIFY(Vc::MemoryAlignment >= V::MemoryAlignment);
}

// vim: foldmethod=marker
