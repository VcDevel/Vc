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

//#define UNITTEST_ONLY_XTEST 1
#include <vir/testassert.h>
#include <vir/test.h>
#include <Vc/simd>
#include "make_vec.h"

template <class... Ts> using base_template = Vc::simd<Ts...>;
#include "testtypes.h"
#include "conversions.h"

// loads & stores {{{1
using MemTypes =
    std::conditional_t<arithmetic_types::size() == 1,
                       vir::concat<testtypes, arithmetic_types>, arithmetic_types>;

TEST_TYPES(VU, load_store, outer_product<all_test_types, MemTypes>)
{
#ifdef Vc_MSVC
// disable "warning C4756: overflow in constant arithmetic" - I don't care.
#pragma warning(disable : 4756)
#endif

    // types, tags, and constants {{{2
    using V = typename VU::template at<0>;
    using U = typename VU::template at<1>;
    using T = typename V::value_type;
    using M = typename V::mask_type;
    auto &&gen = make_vec<V>;
    using Vc::element_aligned;
    using Vc::vector_aligned;
    constexpr size_t stride_alignment =
        V::size() & 1 ? 1 : V::size() & 2
                                ? 2
                                : V::size() & 4
                                      ? 4
                                      : V::size() & 8
                                            ? 8
                                            : V::size() & 16
                                                  ? 16
                                                  : V::size() & 32
                                                        ? 32
                                                        : V::size() & 64
                                                              ? 64
                                                              : V::size() & 128
                                                                    ? 128
                                                                    : V::size() & 256
                                                                          ? 256
                                                                          : 512;
    using stride_aligned_t =
        std::conditional_t<V::size() == stride_alignment, decltype(vector_aligned),
                           Vc::overaligned_tag<stride_alignment * sizeof(U)>>;
    constexpr stride_aligned_t stride_aligned = {};
    constexpr size_t alignment = 2 * Vc::memory_alignment_v<V, U>;
#ifdef Vc_MSVC
    using TT = Vc::overaligned_tag<alignment>;
    constexpr TT overaligned = {};
#else
    constexpr auto overaligned = Vc::overaligned<alignment>;
#endif
    const V indexes_from_0 = gen({0, 1, 2, 3}, 4);
    for (std::size_t i = 0; i < V::size(); ++i) {
        COMPARE(indexes_from_0[i], T(i));
    }
    const V indexes_from_1 = gen({1, 2, 3, 4}, 4);
    const V indexes_from_size = gen({T(V::size())}, 1);
    const M alternating_mask = make_mask<M>({0, 1});

    // loads {{{2
    cvt_inputs<T, U> test_values;

    constexpr auto mem_size =
        test_values.size() > 3 * V::size() ? test_values.size() : 3 * V::size();
    alignas(Vc::memory_alignment_v<V, U> * 2) U mem[mem_size] = {};
    alignas(Vc::memory_alignment_v<V, T> * 2) T reference[mem_size] = {};
    for (std::size_t i = 0; i < test_values.size(); ++i) {
        const U value = test_values[i];
        mem[i] = value;
        reference[i] = static_cast<T>(value);
    }
    for (std::size_t i = test_values.size(); i < mem_size; ++i) {
        mem[i] = U(i);
        reference[i] = mem[i];
    }

    V x(&mem[V::size()], stride_aligned);
    auto &&compare = [&](const std::size_t offset) {
        static int n = 0;
        const V ref(&reference[offset], element_aligned);
        for (auto i = 0ul; i < V::size(); ++i) {
            if (is_conversion_undefined<T>(mem[i + offset])) {
                continue;
            }
            COMPARE(x[i], reference[i + offset])
                << "\nbefore conversion: " << mem[i + offset]
                << "\n   offset = " << offset
                << "\n        x = " << asBytes(x) << " = " << x
                << "\nreference = " << asBytes(ref) << " = " << ref
                << "\nx == ref  = " << asBytes(x == ref) << " = " << (x == ref)
                << "\ncall no. " << n;
        }
        ++n;
    };
    compare(V::size());
    x = V{mem, overaligned};
    compare(0);
    x = {&mem[1], element_aligned};
    compare(1);

    x.copy_from(&mem[V::size()], stride_aligned);
    compare(V::size());
    x.copy_from(&mem[1], element_aligned);
    compare(1);
    x.copy_from(mem, vector_aligned);
    compare(0);

    for (std::size_t i = 0; i < mem_size - V::size(); ++i) {
        x.copy_from(&mem[i], element_aligned);
        compare(i);
    }

    for (std::size_t i = 0; i < test_values.size(); ++i) {
        mem[i] = U(i);
    }
    x = indexes_from_0;
    where(alternating_mask, x).copy_from(&mem[V::size()], stride_aligned);
    COMPARE(x == indexes_from_size, alternating_mask);
    COMPARE(x == indexes_from_0, !alternating_mask);
    where(alternating_mask, x).copy_from(&mem[1], element_aligned);
    COMPARE(x == indexes_from_1, alternating_mask);
    COMPARE(x == indexes_from_0, !alternating_mask);
    where(!alternating_mask, x).copy_from(mem, overaligned);
    COMPARE(x == indexes_from_0, !alternating_mask);
    COMPARE(x == indexes_from_1, alternating_mask);

    x = where(alternating_mask, V()).copy_from(&mem[V::size()], stride_aligned);
    COMPARE(x == indexes_from_size, alternating_mask);
    COMPARE(x == 0, !alternating_mask);

    x = where(!alternating_mask, V()).copy_from(&mem[1], element_aligned);
    COMPARE(x == indexes_from_1, !alternating_mask);
    COMPARE(x == 0, alternating_mask);

    // stores {{{2
    memset(mem, 0, sizeof(mem));
    x = indexes_from_1;
    x.copy_to(&mem[V::size()], stride_aligned);
    std::size_t i = 0;
    for (; i < V::size(); ++i) {
        COMPARE(mem[i], U(0)) << "i: " << i;
    }
    for (; i < 2 * V::size(); ++i) {
        COMPARE(mem[i], U(i - V::size() + 1)) << "i: " << i;
    }
    for (; i < 3 * V::size(); ++i) {
        COMPARE(mem[i], U(0)) << "i: " << i;
    }

    memset(mem, 0, sizeof(mem));
    x.copy_to(&mem[1], element_aligned);
    COMPARE(mem[0], U(0));
    for (i = 1; i <= V::size(); ++i) {
        COMPARE(mem[i], U(i));
    }
    for (; i < 3 * V::size(); ++i) {
        COMPARE(mem[i], U(0));
    }

    memset(mem, 0, sizeof(mem));
    x.copy_to(mem, vector_aligned);
    for (i = 0; i < V::size(); ++i) {
        COMPARE(mem[i], U(i + 1));
    }
    for (; i < 3 * V::size(); ++i) {
        COMPARE(mem[i], U(0));
    }

    memset(mem, 0, sizeof(mem));
    where(alternating_mask, indexes_from_0).copy_to(&mem[V::size()], stride_aligned);
    for (i = 0; i < V::size() + 1; ++i) {
        COMPARE(mem[i], U(0));
    }
    for (; i < 2 * V::size(); i += 2) {
        COMPARE(mem[i], U(i - V::size()));
    }
    for (i = V::size() + 2; i < 2 * V::size(); i += 2) {
        COMPARE(mem[i], U(0));
    }
    for (; i < 3 * V::size(); ++i) {
        COMPARE(mem[i], U(0));
    }
}

