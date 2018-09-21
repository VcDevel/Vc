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

using namespace Vc;

template<typename T, size_t N> struct SomeStruct
{
    T d[N];
};

size_t createNMask(size_t N)
{
    size_t NMask = (N >> 1) | (N >> 2);
    for (size_t shift = 2; shift < sizeof(size_t) * 8; shift *= 2) {
        NMask |= NMask >> shift;
    }
    return NMask;
}

#if !defined __GNUC__ || defined __OPTIMIZE__
static constexpr int TotalRetests = 10000;
#else
static constexpr int TotalRetests = 1000;
#endif

template <typename T> T rotate(T x)
{
    return x.rotated(1);
}
template <typename T, std::size_t N> Vc::SimdArray<T, N> rotate(const Vc::SimdArray<T, N> &x)
{
    Vc::SimdArray<T, N> r;
    r[0] = x[N - 1];
    for (std::size_t i = 0; i < N - 1; ++i) {
        r[i + 1] = x[i];
    }
    return r;
}

namespace std
{
template <typename T, std::size_t N>
inline std::ostream &operator<<(std::ostream &out, const std::array<T, N> &a)
{
    out << "\narray{" << a[0];
    for (std::size_t i = 1; i < N; ++i) {
        out << ", " << a[i];
    }
    return out << '}';
}
template <typename T, std::size_t N>
inline bool operator==(const std::array<Vc::Vector<T>, N> &a, const std::array<Vc::Vector<T>, N> &b)
{
    for (std::size_t i = 0; i < N; ++i) {
        if (!Vc::all_of(a[i] == b[i])) {
            return false;
        }
    }
    return true;
}
}  // namespace std

template <typename V, typename Wrapper, typename IndexType, std::size_t... Indexes>
void testInterleavingScatterCompare(Wrapper &data, const IndexType &i,
                                    Vc::index_sequence<Indexes...>)
{
    const std::array<V, sizeof...(Indexes)> reference = {
        {((void)Indexes, V::Random())...}};
        //{V([](size_t i) { return i + Indexes * V::size(); })...}};

    data[i] = tie(reference[Indexes]...);
    std::array<V, sizeof...(Indexes)> t = data[i];
    COMPARE(t, reference) << "i: " << i;

    for (auto &x : t) {
        x.setZero();
    }
    tie(t[Indexes]...) = data[i];
    COMPARE(t, reference);

    if (sizeof...(Indexes) > 2) {
        testInterleavingScatterCompare<V>(
            data, i, Vc::make_index_sequence<
                         (sizeof...(Indexes) > 2 ? sizeof...(Indexes) - 1 : 2)>());
    }
}

TEST_TYPES(Param, testInterleavingScatter,
           outer_product<AllVectors, Typelist<std::integral_constant<std::size_t, 2>,
                                              std::integral_constant<std::size_t, 3>,
                                              std::integral_constant<std::size_t, 4>,
                                              std::integral_constant<std::size_t, 5>,
                                              std::integral_constant<std::size_t, 6>,
                                              std::integral_constant<std::size_t, 7>,
                                              std::integral_constant<std::size_t, 8>>>)
{
    typedef typename Param::template at<0> V;
    constexpr auto StructSize = Param::template at<1>::value;
    typedef typename V::EntryType T;
    typedef typename V::IndexType I;
    typedef SomeStruct<T, StructSize> S;
    typedef Vc::InterleavedMemoryWrapper<S, V> Wrapper;
    const size_t N =
        std::min(static_cast<size_t>(std::numeric_limits<typename I::EntryType>::max()),
                 1024 * 1024 / sizeof(S));
    const typename I::value_type NN = N;
    const size_t NMask = createNMask(N);

    S *data = Vc::malloc<S, Vc::AlignOnVector>(N);
    std::memset(data, 0, sizeof(S) * N);
    Wrapper data_v(data);

    try {
        testInterleavingScatterCompare<V>(data_v, (NN - 1) - I([](int n) { return n; }),
                                          Vc::make_index_sequence<StructSize>());
        for (int retest = 0; retest < TotalRetests; ++retest) {
            I indexes = (I::Random() >> 10) & I(NMask);
            if (I::Size != 1) {
                // ensure the indexes are unique
                while (any_of(indexes.sorted() == rotate(indexes.sorted()))) {
                    indexes = (I::Random() >> 10) & I(NMask);
                }
            }
            VERIFY(all_of(indexes >= 0));
            VERIFY(all_of(indexes < NN));

            testInterleavingScatterCompare<V>(data_v, indexes,
                                              Vc::make_index_sequence<StructSize>());
        }

        for (size_t i = 0; i < N - V::Size; ++i) {
            //std::memset(data, 0, sizeof(S) * N); // useful when debugging
            testInterleavingScatterCompare<V>(data_v, i,
                                              Vc::make_index_sequence<StructSize>());
        }
    } catch (...) {
        std::cout << "data was:";
        std::cout << '\n';
        for (size_t i = 0; i < 16; ++i) {
            for (size_t n = 0; n < StructSize; ++n) {
                std::cout << data[i].d[n] << ' ';
            }
        }
        std::cout << '\n';
        for (size_t n = 0; n < StructSize; ++n) {
            std::cout << '\n' << n << ": ";
            for (size_t i = 0; i < N; ++i) {
                std::cout << data[i].d[n] << ' ';
            }
        }
        std::cout << '\n';
        throw;
    }
}

// vim: foldmethod=marker
