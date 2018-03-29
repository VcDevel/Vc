/*  This file is part of the Vc library. {{{
Copyright © 2016-2017 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_SIMD_GENERICIMPL_H_
#define VC_SIMD_GENERICIMPL_H_

#include "detail.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail {
// simd impl {{{1
template <class Derived> struct generic_simd_impl {
    // member types {{{2
    template <size_t N> using size_tag = size_constant<N>;
    template <class T> using type_tag = T *;

    template <class T, size_t N>
    static Vc_INTRINSIC auto Vc_VDECL simd(Storage<T, N> x)
    {
        return Derived::make_simd(x);
    }

    // adjust_for_long{{{2
    template <size_t Size>
    static Vc_INTRINSIC Storage<equal_int_type_t<long>, Size> Vc_VDECL
    adjust_for_long(Storage<long, Size> x)
    {
        return {x.intrin()};
    }
    template <size_t Size>
    static Vc_INTRINSIC Storage<equal_int_type_t<ulong>, Size> Vc_VDECL
    adjust_for_long(Storage<ulong, Size> x)
    {
        return {x.intrin()};
    }
    template <class T, size_t Size>
    static Vc_INTRINSIC const Storage<T, Size> &adjust_for_long(const Storage<T, Size> &x)
    {
        return x;
    }

    // broadcast {{{2
    template <class T, size_t N>
    static constexpr Vc_INTRINSIC Storage<T, N> broadcast(T x, size_tag<N>) noexcept
    {
        return Storage<T, N>::broadcast(x);
    }

    // generator {{{2
    template <class F, class T, size_t N>
    static Vc_INTRINSIC Storage<T, N> generator(F &&gen, type_tag<T>, size_tag<N>)
    {
        return detail::generate_storage<T, N>(std::forward<F>(gen));
    }

    // complement {{{2
    template <class T, size_t N>
    static constexpr Vc_INTRINSIC Storage<T, N> complement(Storage<T, N> x) noexcept
    {
        return detail::x86::complement(x);
    }

    // unary minus {{{2
    template <class T, size_t N>
    static constexpr Vc_INTRINSIC Storage<T, N> unary_minus(Storage<T, N> x) noexcept
    {
        return detail::x86::unary_minus(x);
    }

    // arithmetic operators {{{2
#define Vc_ARITHMETIC_OP_(name_)                                                         \
    template <class T, size_t N>                                                         \
    static Vc_INTRINSIC Storage<T, N> name_(Storage<T, N> x, Storage<T, N> y)            \
    {                                                                                    \
        return detail::x86::name_(x, y);                                                 \
    }                                                                                    \
    Vc_NOTHING_EXPECTING_SEMICOLON
    Vc_ARITHMETIC_OP_(plus);
    Vc_ARITHMETIC_OP_(minus);
    Vc_ARITHMETIC_OP_(multiplies);
    Vc_ARITHMETIC_OP_(divides);
    Vc_ARITHMETIC_OP_(modulus);
    Vc_ARITHMETIC_OP_(bit_and);
    Vc_ARITHMETIC_OP_(bit_or);
    Vc_ARITHMETIC_OP_(bit_xor);
    Vc_ARITHMETIC_OP_(bit_shift_left);
    Vc_ARITHMETIC_OP_(bit_shift_right);
#undef Vc_ARITHMETIC_OP_

    template <class T, size_t N>
    static constexpr Vc_INTRINSIC Storage<T, N> Vc_VDECL bit_shift_left(Storage<T, N> x,
                                                                        int y)
    {
        return detail::x86::bit_shift_left(x, y);
    }
    template <class T, size_t N>
    static constexpr Vc_INTRINSIC Storage<T, N> Vc_VDECL bit_shift_right(Storage<T, N> x,
                                                                         int y)
    {
        return detail::x86::bit_shift_right(x, y);
    }

    // min, max, clamp {{{2
    template <class T, size_t N>
    Vc_NORMAL_MATH static constexpr Vc_INTRINSIC Storage<T, N> min(Storage<T, N> a,
                                                                   Storage<T, N> b)
    {
        return a.d < b.d ? a.d : b.d;
    }
    template <class T, size_t N>
    Vc_NORMAL_MATH static constexpr Vc_INTRINSIC Storage<T, N> max(Storage<T, N> a,
                                                                   Storage<T, N> b)
    {
        return a.d > b.d ? a.d : b.d;
    }

    template <class T, size_t N>
    Vc_NORMAL_MATH static constexpr Vc_INTRINSIC std::pair<Storage<T, N>, Storage<T, N>>
    minmax(Storage<T, N> a, Storage<T, N> b)
    {
        return {a.d < b.d ? a.d : b.d, a.d < b.d ? b.d : a.d};
    }

    // sqrt {{{2
    template <class T, size_t N>
    static Vc_INTRINSIC Storage<T, N> sqrt(Storage<T, N> x) noexcept
    {
        return detail::x86::sqrt(x);
    }

    // abs {{{2
    template <class T, size_t N>
    static Vc_INTRINSIC Storage<T, N> abs(Storage<T, N> x) noexcept
    {
        return detail::x86::abs(adjust_for_long(x));
    }

    // increment & decrement{{{2
    template <class T, size_t N> static Vc_INTRINSIC void increment(Storage<T, N> &x)
    {
        x = plus(x, Storage<T, N>(Derived::broadcast(T(1), size_tag<N>())));
    }
    template <class T, size_t N> static Vc_INTRINSIC void decrement(Storage<T, N> &x)
    {
        x = minus(x, Storage<T, N>(Derived::broadcast(T(1), size_tag<N>())));
    }

    // masked_assign{{{2
    template <class T, class K, size_t N>
    static Vc_INTRINSIC void Vc_VDECL masked_assign(Storage<K, N> k, Storage<T, N> &lhs,
                                             detail::id<Storage<T, N>> rhs)
    {
        lhs = detail::x86::blend(k, lhs, rhs);
    }
    template <class T, class K, size_t N>
    static Vc_INTRINSIC void Vc_VDECL masked_assign(Storage<K, N> k, Storage<T, N> &lhs,
                                                    detail::id<T> rhs)
    {
#ifdef __GNUC__
        if (__builtin_constant_p(rhs) && rhs == 0 && std::is_same<K, T>::value) {
            lhs = x86::andnot_(k, lhs);
            return;
        }
#endif  // __GNUC__
        lhs =
            detail::x86::blend(k, lhs, x86::broadcast(rhs, size_constant<sizeof(lhs)>()));
    }

    // masked_cassign {{{2
    template <template <typename> class Op, class T, class K, size_t N>
    static Vc_INTRINSIC void Vc_VDECL masked_cassign(const Storage<K, N> k,
                                                     Storage<T, N> &lhs,
                                                     const detail::id<Storage<T, N>> rhs)
    {
        lhs = detail::x86::blend(k, lhs,
                                 detail::data(Op<void>{}(simd(lhs), simd(rhs))));
    }

    template <template <typename> class Op, class T, class K, size_t N>
    static Vc_INTRINSIC void Vc_VDECL masked_cassign(const Storage<K, N> k,
                                                     Storage<T, N> &lhs,
                                                     const detail::id<T> rhs)
    {
        lhs = detail::x86::blend(
            k, lhs,
            detail::data(Op<void>{}(
                simd(lhs), simd<T, N>(Derived::broadcast(rhs, size_tag<N>())))));
    }

    // masked_unary {{{2
    template <template <typename> class Op, class T, class K, size_t N>
    static Vc_INTRINSIC Storage<T, N> Vc_VDECL masked_unary(const Storage<K, N> k,
                                                            const Storage<T, N> v)
    {
        auto vv = simd(v);
        Op<decltype(vv)> op;
        return detail::x86::blend(k, v, detail::data(op(vv)));
    }

    //}}}2
};

// simd_mask impl {{{1
template <class abi, template <class> class mask_member_type> struct generic_mask_impl {
    // member types {{{2
    template <size_t N> using size_tag = size_constant<N>;
    template <class T> using type_tag = T *;
    template <class T> using simd_mask = Vc::simd_mask<T, abi>;

    // masked load (AVX512 has its own overloads) {{{2
    template <class T, size_t N, class F>
    static inline void Vc_VDECL masked_load(Storage<T, N> &merge, Storage<T, N> mask,
                                            const bool *mem, F) noexcept
    {
        if constexpr (have_avx512bw_vl && N == 32 && sizeof(T) == 1) {
            const auto k = convert_any_mask<Storage<bool, N>>(mask);
            merge = to_storage(
                _mm256_mask_sub_epi8(to_m256i(merge), k, __m256i(),
                                     _mm256_mask_loadu_epi8(__m256i(), k, mem)));
        } else if constexpr (have_avx512bw_vl && N == 16 && sizeof(T) == 1) {
            const auto k = convert_any_mask<Storage<bool, N>>(mask);
            merge = to_storage(_mm_mask_sub_epi8(to_m128i(merge), k, __m128i(),
                                                 _mm_mask_loadu_epi8(__m128i(), k, mem)));
        } else if constexpr (have_avx512bw_vl && N == 16 && sizeof(T) == 2) {
            const auto k = convert_any_mask<Storage<bool, N>>(mask);
            merge = to_storage(_mm256_mask_sub_epi16(
                to_m256i(merge), k, __m256i(),
                _mm256_cvtepi8_epi16(_mm_mask_loadu_epi8(__m128i(), k, mem))));
        } else if constexpr (have_avx512bw_vl && N == 8 && sizeof(T) == 2) {
            const auto k = convert_any_mask<Storage<bool, N>>(mask);
            merge = to_storage(_mm_mask_sub_epi16(
                to_m128i(merge), k, __m128i(),
                _mm_cvtepi8_epi16(_mm_mask_loadu_epi8(__m128i(), k, mem))));
        } else if constexpr (have_avx512bw_vl && N == 8 && sizeof(T) == 4) {
            const auto k = convert_any_mask<Storage<bool, N>>(mask);
            merge = to_storage(_mm256_mask_sub_epi32(
                to_m256i(merge), k, __m256i(),
                _mm256_cvtepi8_epi32(_mm_mask_loadu_epi8(__m128i(), k, mem))));
        } else if constexpr (have_avx512bw_vl && N == 4 && sizeof(T) == 4) {
            const auto k = convert_any_mask<Storage<bool, N>>(mask);
            merge = to_storage(_mm_mask_sub_epi32(
                to_m128i(merge), k, __m128i(),
                _mm_cvtepi8_epi32(_mm_mask_loadu_epi8(__m128i(), k, mem))));
        } else if constexpr (have_avx512bw_vl && N == 4 && sizeof(T) == 8) {
            const auto k = convert_any_mask<Storage<bool, N>>(mask);
            merge = to_storage(_mm256_mask_sub_epi64(
                to_m256i(merge), k, __m256i(),
                _mm256_cvtepi8_epi64(_mm_mask_loadu_epi8(__m128i(), k, mem))));
        } else if constexpr (have_avx512bw_vl && N == 2 && sizeof(T) == 8) {
            const auto k = convert_any_mask<Storage<bool, N>>(mask);
            merge = to_storage(_mm_mask_sub_epi64(
                to_m128i(merge), k, __m128i(),
                _mm_cvtepi8_epi64(_mm_mask_loadu_epi8(__m128i(), k, mem))));
        } else {
            // AVX(2) has 32/64 bit maskload, but nothing at 8 bit granularity
            auto tmp = storage_bitcast<detail::int_for_sizeof_t<T>>(merge);
            detail::execute_n_times<N>([&](auto i) {
                if (mask[i]) {
                    tmp.set(i, -mem[i]);
                }
            });
            merge = storage_bitcast<T>(tmp);
        }
    }

    // masked store {{{2
    template <class T, size_t N, class F>
    static inline void Vc_VDECL masked_store(const Storage<T, N> v, bool *mem, F,
                                             const Storage<T, N> k) noexcept
    {
        detail::execute_n_times<N>([&](auto i) {
            if (k[i]) {
                mem[i] = v[i];
            }
        });
    }

    // to_bitset {{{2
    template <class T, size_t N>
    static Vc_INTRINSIC std::bitset<N> to_bitset(Storage<T, N> v) noexcept
    {
        static_assert(N <= sizeof(uint) * CHAR_BIT,
                      "Needs missing 64-bit implementation");
        if constexpr (std::is_integral_v<T> == (sizeof(T) == 1)) {
            return x86::movemask(v);
        } else if constexpr (sizeof(T) == 2) {
            return x86::movemask_epi16(v);
        } else {
            static_assert(std::is_integral_v<T>);
            using U = std::conditional_t<sizeof(T) == 4, float, double>;
            return x86::movemask(storage_bitcast<U>(v));
        }
#if 0 //defined Vc_HAVE_BMI2
            switch (sizeof(T)) {
            case 2: return _pext_u32(x86::movemask(v), 0xaaaaaaaa);
            case 4: return _pext_u32(x86::movemask(v), 0x88888888);
            case 8: return _pext_u32(x86::movemask(v), 0x80808080);
            default: Vc_UNREACHABLE();
            }
#endif
    }

    // from_bitset{{{2
    template <size_t N, class T>
    static Vc_INTRINSIC mask_member_type<T> from_bitset(std::bitset<N> bits, type_tag<T>)
    {
#ifdef Vc_HAVE_AVX512_ABI
        return to_storage(bits);
#else  // Vc_HAVE_AVX512_ABI
        using U = std::make_unsigned_t<detail::int_for_sizeof_t<T>>;
        using V = simd<U, abi>;
        constexpr size_t bits_per_element = sizeof(U) * CHAR_BIT;
        if (bits_per_element >= N) {
            V tmp(static_cast<U>(bits.to_ullong()));                  // broadcast
            tmp &= V([](auto i) { return static_cast<U>(1ull << i); });  // mask bit index
            return storage_bitcast<T>(detail::data(tmp != V()));
        } else {
            V tmp([&](auto i) {
                return static_cast<U>(bits.to_ullong() >>
                                      (bits_per_element * (i / bits_per_element)));
            });
            tmp &= V([](auto i) {
                return static_cast<U>(1ull << (i % bits_per_element));
            });  // mask bit index
            return storage_bitcast<T>(detail::data(tmp != V()));
        }
#endif  // Vc_HAVE_AVX512_ABI
    }

    // masked_assign{{{2
    template <class T, size_t N>
    static Vc_INTRINSIC void Vc_VDECL masked_assign(Storage<T, N> k, Storage<T, N> &lhs,
                                                    detail::id<Storage<T, N>> rhs)
    {
        lhs = detail::x86::blend(k, lhs, rhs);
    }
    template <class T, size_t N>
    static Vc_INTRINSIC void Vc_VDECL masked_assign(Storage<T, N> k, Storage<T, N> &lhs,
                                                    bool rhs)
    {
#ifdef __GNUC__
        if (__builtin_constant_p(rhs)) {
            if (rhs == false) {
                lhs = x86::andnot_(k, lhs);
            } else {
                lhs = x86::or_(k, lhs);
            }
            return;
        }
#endif  // __GNUC__
        lhs = detail::x86::blend(k, lhs, detail::data(simd_mask<T>(rhs)));
    }

    //}}}2
};

//}}}1
}  // namespace detail
Vc_VERSIONED_NAMESPACE_END


#endif  // VC_SIMD_GENERICIMPL_H_
