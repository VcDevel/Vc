/*  This file is part of the Vc library. {{{
Copyright © 2009-2016 Matthias Kretz <kretz@kde.org>

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
#include <Vc/datapar>

template<typename T, int N>
using fixed_vec = Vc::datapar<T, Vc::datapar_abi::fixed_size<N>>;
template<typename T, int N>
using fixed_mask = Vc::mask<T, Vc::datapar_abi::fixed_size<N>>;

#define ALL_TYPES                                                                        \
    (fixed_mask<int, 3>, fixed_mask<int, 1>, fixed_mask<int, 8>, fixed_mask<int, 32>,    \
     Vc::mask<int>)

TEST_TYPES(M, default_constructors, ALL_TYPES)
{
    M x;
    x = M{};
    x = M();
    x = x;
    for (std::size_t i = 0; i < M::size(); ++i) {
        COMPARE(x[i], false);
    }
}

TEST_TYPES(M, broadcast, ALL_TYPES)
{
    M x = true;
    M y = false;
    for (std::size_t i = 0; i < M::size(); ++i) {
        COMPARE(x[i], true);
        COMPARE(y[i], false);
    }
    y = true;
    COMPARE(x, y);
}

TEST_TYPES(M, compares, ALL_TYPES)
{
    M x = true, y = false;
    VERIFY(x == x);
    VERIFY(x != y);
    VERIFY(y != x);
    VERIFY(!(x != x));
    VERIFY(!(x == y));
    VERIFY(!(y == x));
}

TEST_TYPES(M, subscript, ALL_TYPES)
{
    M x=true;
    for (std::size_t i = 0; i < M::size(); ++i) {
        COMPARE(x[i], true);
        x[i] = !x[i];
    }
    for (std::size_t i = 0; i < M::size(); ++i) {
        COMPARE(x[i], false);
    }
}

TEST_TYPES(M, convert, ALL_TYPES)
{
    using M2 = fixed_mask<float, M::size()>;
    M2 x = true;
    M y = x;
    COMPARE(y, M{true});
    x[0] = false;
    COMPARE(x[0], false);
    y = x;
    COMPARE(y[0], false);
    for (std::size_t i = 1; i < M::size(); ++i) {
        COMPARE(y[i], true);
    }
    M2 z = y;
    COMPARE(z, x);
}

TEST_TYPES(M, load, ALL_TYPES)
{
    alignas(Vc::memory_alignment<M> * 2) bool mem[3 * M::size()] = {};
    using Vc::flags::element_aligned;
    using Vc::flags::vector_aligned;
    using Vc::flags::overaligned;
    M x(&mem[M::size()], vector_aligned);
    COMPARE(x, M{false});
    x = {&mem[1], element_aligned};
    COMPARE(x, M{false});
    x = {mem, overaligned<Vc::memory_alignment<M> * 2>};
    COMPARE(x, M{false});
}

TEST_TYPES(M, negate, ALL_TYPES)
{
    M x = false;
    M y = !x;
    COMPARE(y, M{true});
    COMPARE(!y, x);
}

// vim: foldmethod=marker
