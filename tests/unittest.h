/*  This file is part of the Vc library. {{{
Copyright © 2009-2015 Matthias Kretz <kretz@kde.org>

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

#include <Vc/Vc>
#include <Vc/support.h>
#include "virtest/vir/test.h"
#include <random>

// pre-defined type lists {{{
using RealVectors = vir::Typelist<Vc::native_simd<double>, Vc::native_simd<float>>;
using IntVectors = vir::Typelist<Vc::native_simd<int>, Vc::native_simd<unsigned short>,
                                 Vc::native_simd<unsigned int>, Vc::native_simd<short>>;
using AllVectors = vir::concat<RealVectors, IntVectors>;
using AllMasks = vir::Typelist<Vc::double_m, Vc::float_m, Vc::int_m, Vc::short_m>;
template <int N>
using RealSimdArrays =
    vir::Typelist<Vc::fixed_size_simd<double, N>, Vc::fixed_size_simd<float, N>>;
template <int N>
using IntSimdArrays =
    vir::Typelist<Vc::fixed_size_simd<int, N>, Vc::fixed_size_simd<unsigned short, N>,
                  Vc::fixed_size_simd<unsigned int, N>, Vc::fixed_size_simd<short, N>>;
template <int N>
using OddIntSimdArrays =
    IntSimdArrays<N>;
template <int N> using SimdArrays = vir::concat<RealSimdArrays<N>, IntSimdArrays<N>>;
template <int N>
using OddSimdArrays = vir::concat<RealSimdArrays<N>, OddIntSimdArrays<N>>;
using SimdArrayList = vir::concat<
#ifdef Vc_IMPL_Scalar
    SimdArrays<3>, SimdArrays<1>
#else
# if Vc_FLOAT_V_SIZE > 4
    SimdArrays<32>,
# endif
    OddSimdArrays<19>, SimdArrays<9>, SimdArrays<8>, SimdArrays<5>, SimdArrays<4>,
    SimdArrays<3>
#endif
    >;
using RealSimdArrayList = vir::concat<
#ifdef Vc_IMPL_Scalar
    RealSimdArrays<3>, RealSimdArrays<1>
#else
# if Vc_FLOAT_V_SIZE > 4
    RealSimdArrays<32>,
# endif
    RealSimdArrays<19>, RealSimdArrays<9>, RealSimdArrays<8>, RealSimdArrays<5>,
    RealSimdArrays<4>, RealSimdArrays<3>
#endif
    >;

using RealTypes = vir::concat<RealVectors, RealSimdArrayList>;
using AllTypes = vir::concat<AllVectors, SimdArrayList>;
// }}}
// allMasks {{{
template <typename Vec> static typename Vec::Mask allMasks(size_t i)
{
    static_assert(Vec::size() <= 8 * sizeof(i),
                  "allMasks cannot create all possible masks for the given type Vec.");
    using M = typename Vec::Mask;
    const Vec indexes(Vc::IndexesFromZero);
    M mask(true);

    for (int j = 0; j < int(Vec::size()); ++j) {
        if (i & (size_t(1) << j)) {
            mask ^= indexes == j;
        }
    }
    return mask;
}

#define for_all_masks(VecType, _mask_)                                                   \
    static_assert(VecType::size() <= 16, "for_all_masks takes too long with "            \
                                         "VecType::size > 16. Use withRandomMask "       \
                                         "instead.");                                    \
    for (int _Vc_for_all_masks_i = 0; _Vc_for_all_masks_i == 0; ++_Vc_for_all_masks_i)   \
        for (typename VecType::Mask _mask_ =                                             \
                 allMasks<VecType>(_Vc_for_all_masks_i++);                     \
             !_mask_.isEmpty();                                                          \
             _mask_ = allMasks<VecType>(_Vc_for_all_masks_i++))

template <typename V, int Repetitions = 10000, typename F> void withRandomMask(F &&f)
{
    std::default_random_engine engine;
    std::uniform_int_distribution<std::size_t> dist(0, (1ull << V::Size) - 1);
    for (int repetition = 0; repetition < Repetitions; ++repetition) {
        f(allMasks<V>(dist(engine)));
    }
}
// }}}
// vir::test::compare_traits specialization {{{
template <class Lhs, class Rhs, class = void> struct vc1_compare_traits {
    using common_type = decltype(std::declval<Lhs>() + std::declval<Rhs>());
    using value_type = typename common_type::value_type;
    static constexpr bool use_memcompare = false;
    static constexpr bool is_fuzzy_comparable = std::is_floating_point<value_type>::value;
    static inline bool is_equal(const common_type &a, const common_type &b)
    {
        return all_of(a == b);
    }

    static inline common_type ulp_distance(const common_type &a, const common_type &b)
    {
        static_assert(std::is_floating_point<value_type>::value, "");
        return vir::detail::ulpDiffToReference(a, b);
    }

    static inline common_type ulp_distance_signed(const common_type &a,
                                                  const common_type &b)
    {
        static_assert(std::is_floating_point<value_type>::value, "");
        return vir::detail::ulpDiffToReferenceSigned(a, b);
    }

    static inline bool ulp_compare_and_log(const common_type &ulp,
                                           const common_type &allowed_distance)
    {
        using delegate_traits = vir::test::compare_traits<value_type, value_type>;
        for (std::size_t i = 0; i < common_type::size(); ++i) {
            if (!delegate_traits::ulp_compare_and_log(ulp[i], allowed_distance[i])) {
                return false;
            }
        }
        return true;
    }

    template <class... Ts>
    static inline std::string to_datafile_string(const common_type &d0,
                                                 const Ts &... data)
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < common_type::size(); ++i) {
            ss << std::setprecision(50) << d0[i];
            auto unused = {((ss << '\t' << data[i]), 0)...};
            ss << ((void)unused, '\n');
        }
        return ss.str();
    }
};

// SFINAE for invalid Vector<T, Abi>
template <class Lhs, class Rhs>
struct vc1_compare_traits<
    Lhs, Rhs,
    typename std::enable_if<!Vc::is_simd_vector<Lhs>::value &&
                            !Vc::is_simd_vector<Rhs>::value>::type> {
};

namespace vir
{
namespace test
{
template <class T0, class A0, class T1, class A1>
struct compare_traits<Vc::Vector<T0, A0>, Vc::Vector<T1, A1>>
    : vc1_compare_traits<Vc::Vector<T0, A0>, Vc::Vector<T1, A1>> {
};
template <class Lhs, class T1, class A1>
struct compare_traits<Lhs, Vc::Vector<T1, A1>>
    : vc1_compare_traits<Lhs, Vc::Vector<T1, A1>> {
};
template <class T0, class A0, class Rhs>
struct compare_traits<Vc::Vector<T0, A0>, Rhs>
    : vc1_compare_traits<Vc::Vector<T0, A0>, Rhs> {
};
template <class T0, class T1, std::size_t N, class V0, class V1>
struct compare_traits<Vc::SimdArray<T0, N, V0>, Vc::SimdArray<T1, N, V1>>
    : vc1_compare_traits<Vc::SimdArray<T0, N, V0>, Vc::SimdArray<T1, N, V1>> {
};

template <class T0, class V0, Vc::SimdizeDetail::IteratorDetails::Mutable M0, class T1,
          class V1, Vc::SimdizeDetail::IteratorDetails::Mutable M1>
struct compare_traits<Vc::SimdizeDetail::IteratorDetails::Reference<T0, V0, M0>,
                      Vc::SimdizeDetail::IteratorDetails::Reference<T1, V1, M1>>
    : vc1_compare_traits<V0, V1> {
};

}  // namespace test
}  // namespace vir

// }}}
// verify_vector_unit_supported {{{
namespace
{
struct verify_vector_unit_supported
{
    verify_vector_unit_supported()
    {
        if (!Vc::currentImplementationSupported()) {
            std::cerr
                << "CPU or OS requirements not met for the compiled in vector unit!\n";
            exit(-1);
        }
    }
} verify_vector_unit_supported_;
}  // unnamed namespace
// }}}

// vim: foldmethod=marker
