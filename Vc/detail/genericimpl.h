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
// ISA & type detection {{{1
template <class T, size_t N> constexpr bool is_sse_ps()
{
    return have_sse && std::is_same_v<T, float> && N == 4;
}
template <class T, size_t N> constexpr bool is_sse_pd()
{
    return have_sse2 && std::is_same_v<T, double> && N == 2;
}
template <class T, size_t N> constexpr bool is_avx_ps()
{
    return have_avx && std::is_same_v<T, float> && N == 8;
}
template <class T, size_t N> constexpr bool is_avx_pd()
{
    return have_avx && std::is_same_v<T, double> && N == 4;
}
template <class T, size_t N> constexpr bool is_avx512_ps()
{
    return have_avx512f && std::is_same_v<T, float> && N == 16;
}
template <class T, size_t N> constexpr bool is_avx512_pd()
{
    return have_avx512f && std::is_same_v<T, double> && N == 8;
}

template <class T, size_t N> constexpr bool is_neon_ps()
{
    return have_neon && std::is_same_v<T, float> && N == 4;
}
template <class T, size_t N> constexpr bool is_neon_pd()
{
    return have_neon && std::is_same_v<T, double> && N == 2;
}

// simd impl {{{1
template <class Derived, class Abi> struct generic_simd_impl {
    // member types {{{2
    template <size_t N> using size_tag = size_constant<N>;
    template <class T> using type_tag = T *;
    template <class T>
    using simd_member_type = typename Abi::template traits<T>::simd_member_type;
    template <class T>
    using mask_member_type = typename Abi::template traits<T>::mask_member_type;

    // make_simd(Storage) {{{2
    template <class T, size_t N>
    static Vc_INTRINSIC auto make_simd(Storage<T, N> x)
    {
        return Derived::make_simd(x);
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
    static constexpr Vc_INTRINSIC Storage<T, N> bit_shift_left(Storage<T, N> x,
                                                                        int y)
    {
        return detail::x86::bit_shift_left(x, y);
    }
    template <class T, size_t N>
    static constexpr Vc_INTRINSIC Storage<T, N> bit_shift_right(Storage<T, N> x,
                                                                         int y)
    {
        return detail::x86::bit_shift_right(x, y);
    }

    // compares {{{2
    // equal_to {{{3
    template <class T, size_t N>
    static constexpr Vc_INTRINSIC mask_member_type<T> equal_to(Storage<T, N> x,
                                                               Storage<T, N> y)
    {
        if constexpr (sizeof(x) == 64) {  // AVX512
            if constexpr (std::is_floating_point_v<T>) {
                       if constexpr (sizeof(T) == 8) { return _mm512_cmp_pd_mask(x, y, _CMP_EQ_OQ);
                } else if constexpr (sizeof(T) == 4) { return _mm512_cmp_ps_mask(x, y, _CMP_EQ_OQ);
                } else { assert_unreachable<T>(); }
            } else {
                       if constexpr (sizeof(T) == 8) { return _mm512_cmpeq_epi64_mask(x, y);
                } else if constexpr (sizeof(T) == 4) { return _mm512_cmpeq_epi32_mask(x, y);
                } else if constexpr (sizeof(T) == 2) { return _mm512_cmpeq_epi16_mask(x, y);
                } else if constexpr (sizeof(T) == 1) { return _mm512_cmpeq_epi8_mask(x, y);
                } else { assert_unreachable<T>(); }
            }
        } else {
            return to_storage(x.d == y.d);
        }
    }

    // not_equal_to {{{3
    template <class T, size_t N>
    static constexpr Vc_INTRINSIC mask_member_type<T> not_equal_to(Storage<T, N> x,
                                                                   Storage<T, N> y)
    {
        if constexpr (sizeof(x) == 64) {  // AVX512
            if constexpr (std::is_floating_point_v<T>) {
                       if constexpr (sizeof(T) == 8) { return _mm512_cmp_pd_mask(x, y, _CMP_NEQ_UQ);
                } else if constexpr (sizeof(T) == 4) { return _mm512_cmp_ps_mask(x, y, _CMP_NEQ_UQ);
                } else { assert_unreachable<T>(); }
            } else {
                       if constexpr (sizeof(T) == 8) { return ~_mm512_cmpeq_epi64_mask(x, y);
                } else if constexpr (sizeof(T) == 4) { return ~_mm512_cmpeq_epi32_mask(x, y);
                } else if constexpr (sizeof(T) == 2) { return ~_mm512_cmpeq_epi16_mask(x, y);
                } else if constexpr (sizeof(T) == 1) { return ~_mm512_cmpeq_epi8_mask(x, y);
                } else { assert_unreachable<T>(); }
            }
        } else {
            return to_storage(x.d != y.d);
        }
    }

    // less {{{3
    template <class T, size_t N>
    static constexpr Vc_INTRINSIC mask_member_type<T> less(Storage<T, N> x,
                                                           Storage<T, N> y)
    {
        if constexpr (sizeof(x) == 64) {  // AVX512
            if constexpr (std::is_floating_point_v<T>) {
                       if constexpr (sizeof(T) == 8) { return _mm512_cmp_pd_mask(x, y, _CMP_LT_OS);
                } else if constexpr (sizeof(T) == 4) { return _mm512_cmp_ps_mask(x, y, _CMP_LT_OS);
                } else { assert_unreachable<T>(); }
            } else if constexpr (std::is_signed_v<T>) {
                       if constexpr (sizeof(T) == 8) { return _mm512_cmplt_epi64_mask(x, y);
                } else if constexpr (sizeof(T) == 4) { return _mm512_cmplt_epi32_mask(x, y);
                } else if constexpr (sizeof(T) == 2) { return _mm512_cmplt_epi16_mask(x, y);
                } else if constexpr (sizeof(T) == 1) { return _mm512_cmplt_epi8_mask(x, y);
                } else { assert_unreachable<T>(); }
            } else {
                static_assert(std::is_unsigned_v<T>);
                       if constexpr (sizeof(T) == 8) { return _mm512_cmplt_epu64_mask(x, y);
                } else if constexpr (sizeof(T) == 4) { return _mm512_cmplt_epu32_mask(x, y);
                } else if constexpr (sizeof(T) == 2) { return _mm512_cmplt_epu16_mask(x, y);
                } else if constexpr (sizeof(T) == 1) { return _mm512_cmplt_epu8_mask(x, y);
                } else { assert_unreachable<T>(); }
            }
        } else {
            return to_storage(x.d < y.d);
        }
    }

    // less_equal {{{3
    template <class T, size_t N>
    static constexpr Vc_INTRINSIC mask_member_type<T> less_equal(Storage<T, N> x,
                                                                 Storage<T, N> y)
    {
        if constexpr (sizeof(x) == 64) {  // AVX512
            if constexpr (std::is_floating_point_v<T>) {
                       if constexpr (sizeof(T) == 8) { return _mm512_cmp_pd_mask(x, y, _CMP_LE_OS);
                } else if constexpr (sizeof(T) == 4) { return _mm512_cmp_ps_mask(x, y, _CMP_LE_OS);
                } else { assert_unreachable<T>(); }
            } else if constexpr (std::is_signed_v<T>) {
                       if constexpr (sizeof(T) == 8) { return _mm512_cmple_epi64_mask(x, y);
                } else if constexpr (sizeof(T) == 4) { return _mm512_cmple_epi32_mask(x, y);
                } else if constexpr (sizeof(T) == 2) { return _mm512_cmple_epi16_mask(x, y);
                } else if constexpr (sizeof(T) == 1) { return _mm512_cmple_epi8_mask(x, y);
                } else { assert_unreachable<T>(); }
            } else {
                static_assert(std::is_unsigned_v<T>);
                       if constexpr (sizeof(T) == 8) { return _mm512_cmple_epu64_mask(x, y);
                } else if constexpr (sizeof(T) == 4) { return _mm512_cmple_epu32_mask(x, y);
                } else if constexpr (sizeof(T) == 2) { return _mm512_cmple_epu16_mask(x, y);
                } else if constexpr (sizeof(T) == 1) { return _mm512_cmple_epu8_mask(x, y);
                } else { assert_unreachable<T>(); }
            }
        } else {
            return to_storage(x.d <= y.d);
        }
    }

    // negation {{{2
    template <class T, size_t N>
    static constexpr Vc_INTRINSIC mask_member_type<T> negate(Storage<T, N> x) noexcept
    {
        return detail::to_storage(!x.d);
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

    // reductions {{{2
    template <class T, class BinaryOperation, size_t N>
    static Vc_INTRINSIC T reduce(size_tag<N>, simd<T, Abi> x, BinaryOperation &&binary_op)
    {
        if constexpr (sizeof(x) > 16) {
            using A = simd_abi::deduce_t<T, N / 2>;
            using V = Vc::simd<T, A>;
            return traits<T, A>::simd_impl_type::reduce(
                size_tag<N / 2>(),
                binary_op(V(detail::private_init, extract<0, 2>(data(x).d)),
                          V(detail::private_init, extract<1, 2>(data(x).d))),
                std::forward<BinaryOperation>(binary_op));
        } else {
            const auto &xx = x.d;
            if constexpr (N == 16) {
                x = binary_op(make_simd<T, N>(_mm_unpacklo_epi8(xx, xx)),
                              make_simd<T, N>(_mm_unpackhi_epi8(xx, xx)));
            }
            if constexpr (N >= 8) {
                x = binary_op(make_simd<T, N>(_mm_unpacklo_epi16(xx, xx)),
                              make_simd<T, N>(_mm_unpackhi_epi16(xx, xx)));
            }
            if constexpr (N >= 4) {
                using U = std::conditional_t<std::is_floating_point_v<T>, float, int>;
                const auto y = builtin_cast<U>(xx.d);
                x = binary_op(x, make_simd<T, N>(to_storage(
                                     builtin_type_t<U, 4>{y[3], y[2], y[1], y[0]})));
            }
            const auto y = builtin_cast<llong>(xx.d);
            return binary_op(
                x, make_simd<T, N>(to_storage(builtin_type_t<llong, 2>{y[1], y[1]})))[0];
        }
    }

    // math {{{2
    // sqrt {{{3
    template <class T, size_t N> static Vc_INTRINSIC Storage<T, N> sqrt(Storage<T, N> x)
    {
               if constexpr (is_sse_ps   <T, N>()) { return _mm_sqrt_ps(x);
        } else if constexpr (is_sse_pd   <T, N>()) { return _mm_sqrt_pd(x);
        } else if constexpr (is_avx_ps   <T, N>()) { return _mm256_sqrt_ps(x);
        } else if constexpr (is_avx_pd   <T, N>()) { return _mm256_sqrt_pd(x);
        } else if constexpr (is_avx512_ps<T, N>()) { return _mm512_sqrt_ps(x);
        } else if constexpr (is_avx512_pd<T, N>()) { return _mm512_sqrt_pd(x);
        } else { assert_unreachable<T>(); }
    }

    // abs {{{3
    template <class T, size_t N>
    static Vc_INTRINSIC Storage<T, N> abs(Storage<T, N> x) noexcept
    {
        return detail::x86::abs(x);
    }

    // trunc {{{3
    template <class T, size_t N> static Vc_INTRINSIC Storage<T, N> trunc(Storage<T, N> x)
    {
        if constexpr (is_avx512_ps<T, N>()) {
            return _mm512_roundscale_round_ps(x, 0x03, _MM_FROUND_CUR_DIRECTION);
        } else if constexpr (is_avx512_pd<T, N>()) {
            return _mm512_roundscale_round_pd(x, 0x03, _MM_FROUND_CUR_DIRECTION);
        } else if constexpr (is_avx_ps<T, N>()) {
            return _mm256_round_ps(x, 0x3);
        } else if constexpr (is_avx_pd<T, N>()) {
            return _mm256_round_pd(x, 0x3);
        } else if constexpr (have_sse4_1 && is_sse_ps<T, N>()) {
            return _mm_round_ps(x, 0x3);
        } else if constexpr (have_sse4_1 && is_sse_pd<T, N>()) {
            return _mm_round_pd(x, 0x3);
        } else if constexpr (is_sse_ps<T, N>()) {
            auto truncated = _mm_cvtepi32_ps(_mm_cvttps_epi32(x));
            const auto no_fractional_values = builtin_cast<float>(
                builtin_cast<int>(builtin_cast<uint>(x.d) & 0x7f800000u) <
                0x4b000000);  // the exponent is so large that no mantissa bits signify
                              // fractional values (0x3f8 + 23*8 = 0x4b0)
            return x86::blend(no_fractional_values, x, truncated);
        } else if constexpr (is_sse_pd<T, N>()) {
            const auto abs_x = abs(x).d;
            const auto min_no_fractional_bits = builtin_cast<double>(
                builtin_broadcast<2>(0x4330'0000'0000'0000ull));  // 0x3ff + 52 = 0x433
            builtin_type16_t<double> truncated =
                (abs_x + min_no_fractional_bits) - min_no_fractional_bits;
            // due to rounding, the result can be too large. In this case `truncated >
            // abs(x)` holds, so subtract 1 to truncated if `abs(x) < truncated`
            truncated -=
                and_(builtin_cast<double>(abs_x < truncated), builtin_broadcast<2>(1.));
            // finally, fix the sign bit:
            return or_(
                and_(builtin_cast<double>(builtin_broadcast<2>(0x8000'0000'0000'0000ull)),
                     x),
                truncated);
        } else {
            assert_unreachable<T>();
        }
    }

    // floor {{{3
    template <class T, size_t N> static Vc_INTRINSIC Storage<T, N> floor(Storage<T, N> x)
    {
        if constexpr (is_avx512_ps<T, N>()) {
            return _mm512_roundscale_round_ps(x, 0x01, _MM_FROUND_CUR_DIRECTION);
        } else if constexpr (is_avx512_pd<T, N>()) {
            return _mm512_roundscale_round_pd(x, 0x01, _MM_FROUND_CUR_DIRECTION);
        } else if constexpr (is_avx_ps<T, N>()) {
            return _mm256_round_ps(x, 0x1);
        } else if constexpr (is_avx_pd<T, N>()) {
            return _mm256_round_pd(x, 0x1);
        } else if constexpr (have_sse4_1 && is_sse_ps<T, N>()) {
            return _mm_floor_ps(x);
        } else if constexpr (have_sse4_1 && is_sse_pd<T, N>()) {
            return _mm_floor_pd(x);
        } else {
            const auto y = trunc(x).d;
            const auto negative_input = builtin_cast<T>(x.d < builtin_broadcast<N, T>(0));
            const auto mask = andnot_(builtin_cast<T>(y == x.d), negative_input);
            return or_(andnot_(mask, y), and_(mask, y - builtin_broadcast<N, T>(1)));
        }
    }

    // ceil {{{3
    template <class T, size_t N> static Vc_INTRINSIC Storage<T, N> ceil(Storage<T, N> x)
    {
        if constexpr (is_avx512_ps<T, N>()) {
            return _mm512_roundscale_round_ps(x, 0x02, _MM_FROUND_CUR_DIRECTION);
        } else if constexpr (is_avx512_pd<T, N>()) {
            return _mm512_roundscale_round_pd(x, 0x02, _MM_FROUND_CUR_DIRECTION);
        } else if constexpr (is_avx_ps<T, N>()) {
            return _mm256_round_ps(x, 0x2);
        } else if constexpr (is_avx_pd<T, N>()) {
            return _mm256_round_pd(x, 0x2);
        } else if constexpr (have_sse4_1 && is_sse_ps<T, N>()) {
            return _mm_ceil_ps(x);
        } else if constexpr (have_sse4_1 && is_sse_pd<T, N>()) {
            return _mm_ceil_pd(x);
        } else {
            const auto y = trunc(x).d;
            const auto negative_input = builtin_cast<T>(x.d < builtin_broadcast<N, T>(0));
            const auto inv_mask = or_(builtin_cast<T>(y == x.d), negative_input);
            return or_(and_(inv_mask, y),
                       andnot_(inv_mask, y + builtin_broadcast<N, T>(1)));
        }
    }

    // isnan {{{3
    template <class T, size_t N>
    static Vc_INTRINSIC mask_member_type<T> isnan(Storage<T, N> x)
    {
             if constexpr (is_sse_ps   <T, N>()) { return _mm_cmpunord_ps(x, x); }
        else if constexpr (is_avx_ps   <T, N>()) { return _mm256_cmp_ps(x, x, _CMP_UNORD_Q); }
        else if constexpr (is_avx512_ps<T, N>()) { return _mm512_cmp_ps_mask(x, x, _CMP_UNORD_Q); }
        else if constexpr (is_sse_pd   <T, N>()) { return _mm_cmpunord_pd(x, x); }
        else if constexpr (is_avx_pd   <T, N>()) { return _mm256_cmp_pd(x, x, _CMP_UNORD_Q); }
        else if constexpr (is_avx512_pd<T, N>()) { return _mm512_cmp_pd_mask(x, x, _CMP_UNORD_Q); }
        else { assert_unreachable<T>(); }
    }

    // isfinite {{{3
    template <class T, size_t N>
    static Vc_INTRINSIC mask_member_type<T> isfinite(Storage<T, N> x)
    {
        return x86::cmpord(x, x.d * T());
    }

    // isunordered {{{3
    template <class T, size_t N>
    static Vc_INTRINSIC mask_member_type<T> isunordered(Storage<T, N> x, Storage<T, N> y)
    {
        return x86::cmpunord(x, y);
    }

    // signbit {{{3
    template <class T, size_t N>
    static Vc_INTRINSIC mask_member_type<T> signbit(Storage<T, N> x)
    {
        using I = int_for_sizeof_t<T>;
        if constexpr (have_avx512dq && is_avx512_ps<T, N>()) {
            return _mm512_movepi32_mask(to_m512i(x));
        } else if constexpr (have_avx512dq && is_avx512_pd<T, N>()) {
            return _mm512_movepi64_mask(to_m512i(x));
        } else if constexpr (sizeof(x) == 64) {
            const auto signmask = builtin_broadcast<N>(std::numeric_limits<I>::min());
            return equal_to(Storage<I, N>(builtin_cast<I>(x.d) & signmask),
                            Storage<I, N>(signmask));
        } else {
            const auto xx = builtin_cast<I>(x.d);
            constexpr I signmask = std::numeric_limits<I>::min();
            if constexpr ((sizeof(T) == 4 && (have_avx2 || sizeof(x) == 16)) ||
                          have_avx512vl) {
                (void)signmask;
                return builtin_cast<T>(xx >> std::numeric_limits<I>::digits);
            } else if constexpr ((have_avx2 || (have_ssse3 && sizeof(x) == 16))) {
                return builtin_cast<T>((xx & signmask) == signmask);
            } else {  // SSE2/3 or AVX (w/o AVX2)
                constexpr auto one = builtin_broadcast<N, T>(1);
                return builtin_cast<T>(
                    builtin_cast<T>((xx & signmask) | builtin_cast<I>(one))  // -1 or 1
                    != one);
            }
        }
    }

    // isnonzerovalue (isnormal | is subnormal == !isinf & !isnan & !is zero) {{{3
    template <class T> static Vc_INTRINSIC auto isnonzerovalue(T x)
    {
        using U = typename builtin_traits<T>::value_type;
        return x86::cmpord(x * std::numeric_limits<U>::infinity(),  // NaN if x == 0
                           x * U()                                  // NaN if x == inf
        );
    }

    template <class T> static Vc_INTRINSIC auto isnonzerovalue_mask(T x)
    {
        using U = typename builtin_traits<T>::value_type;
        constexpr size_t N = builtin_traits<T>::width;
        const auto a = x * std::numeric_limits<U>::infinity();  // NaN if x == 0
        const auto b = x * U();                                 // NaN if x == inf
        if constexpr (have_avx512vl && is_sse_ps<U, N>()) {
            return _mm_cmp_ps_mask(a, b, _CMP_ORD_Q);
        } else if constexpr (have_avx512f && is_sse_ps<U, N>()) {
            return __mmask8(0xf &
                            _mm512_cmp_ps_mask(auto_cast(a), auto_cast(b), _CMP_ORD_Q));
        } else if constexpr (have_avx512vl && is_sse_pd<U, N>()) {
            return _mm_cmp_pd_mask(a, b, _CMP_ORD_Q);
        } else if constexpr (have_avx512f && is_sse_pd<U, N>()) {
            return __mmask8(0x3 &
                            _mm512_cmp_pd_mask(auto_cast(a), auto_cast(b), _CMP_ORD_Q));
        } else if constexpr (have_avx512vl && is_avx_ps<U, N>()) {
            return _mm256_cmp_ps_mask(a, b, _CMP_ORD_Q);
        } else if constexpr (have_avx512f && is_avx_ps<U, N>()) {
            return __mmask8(_mm512_cmp_ps_mask(auto_cast(a), auto_cast(b), _CMP_ORD_Q));
        } else if constexpr (have_avx512vl && is_avx_pd<U, N>()) {
            return _mm256_cmp_pd_mask(a, b, _CMP_ORD_Q);
        } else if constexpr (have_avx512f && is_avx_pd<U, N>()) {
            return __mmask8(0xf &
                            _mm512_cmp_pd_mask(auto_cast(a), auto_cast(b), _CMP_ORD_Q));
        } else if constexpr (is_avx512_ps<U, N>()) {
            return _mm512_cmp_ps_mask(a, b, _CMP_ORD_Q);
        } else if constexpr (is_avx512_pd<U, N>()) {
            return _mm512_cmp_pd_mask(a, b, _CMP_ORD_Q);
        } else {
            assert_unreachable<T>();
        }
    }

    // isinf {{{3
    template <class T, size_t N>
    static Vc_INTRINSIC mask_member_type<T> isinf(Storage<T, N> x)
    {
        if constexpr (is_avx512_pd<T, N>()) {
            if constexpr (have_avx512dq) {
                return _mm512_fpclass_pd_mask(x, 0x08) | _mm512_fpclass_pd_mask(x, 0x10);
            } else {
                return _mm512_cmp_epi64_mask(to_m512i(x86::abs(x)),
                                             builtin_broadcast<N>(0x7ff0000000000000ll),
                                             _CMP_EQ_OQ);
            }
        } else if constexpr (is_avx512_ps<T, N>()) {
            if constexpr (have_avx512dq) {
                return _mm512_fpclass_ps_mask(x, 0x08) | _mm512_fpclass_ps_mask(x, 0x10);
            } else {
                return _mm512_cmp_epi32_mask(to_m512i(x86::abs(x)),
                                             auto_cast(builtin_broadcast<N>(0x7f800000u)),
                                             _CMP_EQ_OQ);
            }
        } else if constexpr (have_avx512dq_vl) {
            if constexpr (is_sse_pd<T, N>()) {
                return builtin_cast<double>(_mm_movm_epi64(_mm_fpclass_pd_mask(x, 0x08) |
                                                           _mm_fpclass_pd_mask(x, 0x10)));
            } else if constexpr (is_avx_pd<T, N>()) {
                return builtin_cast<double>(_mm256_movm_epi64(
                    _mm256_fpclass_pd_mask(x, 0x08) | _mm256_fpclass_pd_mask(x, 0x10)));
            } else if constexpr (is_sse_ps<T, N>()) {
                return builtin_cast<float>(_mm_movm_epi32(_mm_fpclass_ps_mask(x, 0x08) |
                                                          _mm_fpclass_ps_mask(x, 0x10)));
            } else if constexpr (is_avx_ps<T, N>()) {
                return builtin_cast<float>(_mm256_movm_epi32(
                    _mm256_fpclass_ps_mask(x, 0x08) | _mm256_fpclass_ps_mask(x, 0x10)));
            } else {
                assert_unreachable<T>();
            }
        } else {
            // compares to inf using the corresponding integer type
            return builtin_cast<T>(builtin_cast<int_for_sizeof_t<T>>(abs(x).d) ==
                                   builtin_cast<int_for_sizeof_t<T>>(builtin_broadcast<N>(
                                       std::numeric_limits<T>::infinity())));
            // alternative:
            //return builtin_cast<T>(abs(x).d > std::numeric_limits<T>::max());
        }
    }
    // isnormal {{{3
    template <class T, size_t N>
    static Vc_INTRINSIC mask_member_type<T> isnormal(Storage<T, N> x)
    {
        // subnormals -> 0
        // 0 -> 0
        // inf -> inf
        // -inf -> inf
        // nan -> inf
        // normal value -> positive value / not 0
        return isnonzerovalue(
            and_(x.d, builtin_broadcast<N>(std::numeric_limits<T>::infinity())));
    }

    // fpclassify {{{3
    template <class T, size_t N>
    static Vc_INTRINSIC fixed_size_storage<int, N> fpclassify(Storage<T, N> x)
    {
        if constexpr (is_avx512_pd<T, N>()) {
            // AVX512 is special because we want to use an __mmask to blend int vectors
            // (coming from double vectors). GCC doesn't allow this combination on the
            // ternary operator. Thus, resort to intrinsics:
            if constexpr (have_avx512vl) {
                auto &&b = [](int y) { return to_intrin(builtin_broadcast<N>(y)); };
                return {_mm256_mask_mov_epi32(
                    _mm256_mask_mov_epi32(
                        _mm256_mask_mov_epi32(b(FP_NORMAL), isnan(x), b(FP_NAN)),
                        isinf(x), b(FP_INFINITE)),
                    _mm512_cmp_pd_mask(
                        abs(x), builtin_broadcast<N>(std::numeric_limits<double>::min()),
                        _CMP_LT_OS),
                    _mm256_mask_mov_epi32(
                        b(FP_SUBNORMAL),
                        _mm512_cmp_pd_mask(x, _mm512_setzero_pd(), _CMP_EQ_OQ),
                        b(FP_ZERO)))};
            } else {
                auto &&b = [](int y) {
                    return _mm512_castsi256_si512(to_intrin(builtin_broadcast<N>(y)));
                };
                return {lo256(_mm512_mask_mov_epi32(
                    _mm512_mask_mov_epi32(
                        _mm512_mask_mov_epi32(b(FP_NORMAL), isnan(x), b(FP_NAN)),
                        isinf(x), b(FP_INFINITE)),
                    _mm512_cmp_pd_mask(
                        abs(x), builtin_broadcast<N>(std::numeric_limits<double>::min()),
                        _CMP_LT_OS),
                    _mm512_mask_mov_epi32(
                        b(FP_SUBNORMAL),
                        _mm512_cmp_pd_mask(x, _mm512_setzero_pd(), _CMP_EQ_OQ),
                        b(FP_ZERO))))};
            }
        } else {
            constexpr auto fp_normal =
                builtin_cast<T>(builtin_broadcast<N, int_for_sizeof_t<T>>(FP_NORMAL));
            constexpr auto fp_nan =
                builtin_cast<T>(builtin_broadcast<N, int_for_sizeof_t<T>>(FP_NAN));
            constexpr auto fp_infinite =
                builtin_cast<T>(builtin_broadcast<N, int_for_sizeof_t<T>>(FP_INFINITE));
            constexpr auto fp_subnormal =
                builtin_cast<T>(builtin_broadcast<N, int_for_sizeof_t<T>>(FP_SUBNORMAL));
            constexpr auto fp_zero =
                builtin_cast<T>(builtin_broadcast<N, int_for_sizeof_t<T>>(FP_ZERO));

            const auto tmp = builtin_cast<llong>(
                abs(x).d < std::numeric_limits<T>::min()
                    ? (x.d == 0 ? fp_zero : fp_subnormal)
                    : x86::blend(isinf(x).d, x86::blend(isnan(x).d, fp_normal, fp_nan),
                                 fp_infinite));
            if constexpr (std::is_same_v<T, float>) {
                if constexpr (fixed_size_storage<int, N>::tuple_size == 1) {
                    return {tmp};
                } else if constexpr (fixed_size_storage<int, N>::tuple_size == 2) {
                    return {extract<0, 2>(tmp), extract<1, 2>(tmp)};
                } else {
                    assert_unreachable<T>();
                }
            } else if constexpr (is_sse_pd<T, N>()) {
                static_assert(fixed_size_storage<int, N>::tuple_size == 2);
                return {_mm_cvtsi128_si32(tmp),
                        {_mm_cvtsi128_si32(_mm_unpackhi_epi64(tmp, tmp))}};
            } else if constexpr (is_avx_pd<T, N>()) {
                static_assert(fixed_size_storage<int, N>::tuple_size == 1);
                return {_mm_packs_epi32(lo128(tmp), hi128(tmp))};
            } else {
                assert_unreachable<T>();
            }
        }
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

    // smart_reference access {{{2
    template <class T, size_t N, class U>
    static Vc_INTRINSIC void set(Storage<T, N> &v, int i, U &&x) noexcept
    {
        v.set(i, std::forward<U>(x));
    }

    // masked_assign{{{2
    template <class T, class K, size_t N>
    static Vc_INTRINSIC void masked_assign(Storage<K, N> k, Storage<T, N> &lhs,
                                           detail::id<Storage<T, N>> rhs)
    {
        lhs = detail::x86::blend(k.d, lhs.d, rhs.d);
    }

    template <class T, class K, size_t N>
    static Vc_INTRINSIC void masked_assign(Storage<K, N> k, Storage<T, N> &lhs,
                                           detail::id<T> rhs)
    {
        if (__builtin_constant_p(rhs) && rhs == 0 && std::is_same<K, T>::value) {
            if constexpr (sizeof(k) >= 16) {
                // the andnot_ optimization only makes sense if k.d is a vector register
                lhs.d = andnot_(k.d, lhs.d);
                return;
            } else {
                // for AVX512/__mmask, a _mm512_maskz_mov is best
                lhs = detail::x86::blend(k, lhs, intrinsic_type_t<T, N>());
                return;
            }
        }
        lhs = detail::x86::blend(k.d, lhs.d, builtin_broadcast<N>(rhs));
    }

    // masked_cassign {{{2
    template <template <typename> class Op, class T, class K, size_t N>
    static Vc_INTRINSIC void masked_cassign(const Storage<K, N> k, Storage<T, N> &lhs,
                                            const detail::id<Storage<T, N>> rhs)
    {
        lhs = detail::x86::blend(
            k.d, lhs.d, detail::data(Op<void>{}(make_simd(lhs), make_simd(rhs))).d);
    }

    template <template <typename> class Op, class T, class K, size_t N>
    static Vc_INTRINSIC void masked_cassign(const Storage<K, N> k, Storage<T, N> &lhs,
                                            const detail::id<T> rhs)
    {
        lhs = detail::x86::blend(
            k.d, lhs.d,
            detail::data(
                Op<void>{}(make_simd(lhs), make_simd<T, N>(builtin_broadcast<N>(rhs))))
                .d);
    }

    // masked_unary {{{2
    template <template <typename> class Op, class T, class K, size_t N>
    static Vc_INTRINSIC Storage<T, N> masked_unary(const Storage<K, N> k,
                                                            const Storage<T, N> v)
    {
        auto vv = make_simd(v);
        Op<decltype(vv)> op;
        return detail::x86::blend(k, v, detail::data(op(vv)));
    }

    //}}}2
};

// simd_mask impl {{{1
template <class Abi> struct generic_mask_impl {
    // member types {{{2
    template <size_t N> using size_tag = size_constant<N>;
    template <class T> using type_tag = T *;
    template <class T> using simd_mask = Vc::simd_mask<T, Abi>;
    template <class T>
    using simd_member_type = typename Abi::template traits<T>::simd_member_type;
    template <class T>
    using mask_member_type = typename Abi::template traits<T>::mask_member_type;
    template <class T>
    using int_builtin_type =
        builtin_type_t<detail::int_for_sizeof_t<T>, simd_member_type<T>::width>;

    // broadcast {{{2
    template <class T>
    static Vc_INTRINSIC mask_member_type<T> broadcast(bool x, type_tag<T>) noexcept
    {
        return to_storage(x ? ~int_builtin_type<T>() : int_builtin_type<T>());
    }

    // load {{{2
    template <class F, size_t N>
    static Vc_INTRINSIC auto load(const bool *mem, F f, size_tag<N>) noexcept
    {
        if constexpr (std::is_same_v<Abi, simd_abi::__sse>) {
            if constexpr (N == 2 && have_sse2) {
                return _mm_set_epi32(-int(mem[1]), -int(mem[1]), -int(mem[0]),
                                     -int(mem[0]));
            } else if constexpr (N == 4 && have_sse2) {
                    __m128i k = _mm_cvtsi32_si128(*reinterpret_cast<const int *>(mem));
                    k = _mm_cmpgt_epi16(_mm_unpacklo_epi8(k, k), _mm_setzero_si128());
                    return intrin_cast<__m128>(_mm_unpacklo_epi16(k, k));
            } else if constexpr (N == 4 && have_mmx) {
                __m128 k =
                    _mm_cvtpi8_ps(_mm_cvtsi32_si64(*reinterpret_cast<const int *>(mem)));
                _mm_empty();
                return _mm_cmpgt_ps(k, __m128());
            } else if constexpr (N == 8 && have_sse2) {
                const auto k = make_builtin<long long>(
                    *reinterpret_cast<const may_alias<long long> *>(mem), 0);
                if constexpr (have_sse2) {
                    return to_intrin(builtin_cast<short>(_mm_unpacklo_epi8(k, k)) != 0);
                }
            } else if constexpr (N == 16 && have_sse2) {
                return _mm_cmpgt_epi8(x86::load16(mem, f), __m128i());
            } else {
                assert_unreachable<F>();
            }
        } else if constexpr (std::is_same_v<Abi, simd_abi::__avx>) {
            if constexpr (N == 4 && have_avx) {
                int bool4;
                if constexpr (detail::is_aligned_v<F, 4>) {
                    bool4=*reinterpret_cast<const may_alias<int> *>(mem);
                } else {
                    std::memcpy(&bool4, mem, 4);
                }
                const auto k =
                    to_intrin((builtin_broadcast<4>(bool4) &
                               make_builtin<int>(0x1, 0x100, 0x10000, 0x1000000)) != 0);
                return concat(_mm_unpacklo_epi32(k, k), _mm_unpackhi_epi32(k, k));
            } else if constexpr (N == 8 && have_avx) {
                auto k = x86::load8(mem, f);
                k = _mm_cmpgt_epi16(_mm_unpacklo_epi8(k, k), __m128i());
                return concat(_mm_unpacklo_epi16(k, k), _mm_unpackhi_epi16(k, k));
            } else if constexpr (N == 16 && have_avx) {
                const auto k = _mm_cmpgt_epi8(x86::load16(mem, f), __m128i());
                return concat(_mm_unpacklo_epi8(k, k), _mm_unpackhi_epi8(k, k));
            } else if constexpr (N == 32 && have_avx2) {
                return _mm256_cmpgt_epi8(x86::load32(mem, f), __m256i());
            } else {
                assert_unreachable<F>();
            }
        } else if constexpr (std::is_same_v<Abi, simd_abi::__avx512>) {
        } else {
            assert_unreachable<F>();
        }
        detail::unused(f); // not true, see PR85827
    }

    // masked load (AVX512 has its own overloads) {{{2
    template <class T, size_t N, class F>
    static inline void masked_load(Storage<T, N> &merge, Storage<T, N> mask,
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

    // store {{{2
    template <class F> static Vc_INTRINSIC void store4(int x, bool *mem, F)
    {
        if constexpr (detail::is_aligned_v<F, 4>) {
            *reinterpret_cast<may_alias<int> *>(mem) = x;
        } else {
            std::memcpy(mem, &x, 4);
        }
    }
    template <class T, class F, size_t N>
    static Vc_INTRINSIC void store(Storage<T, N> v, bool *mem, F f, size_tag<N>) noexcept
    {
        if constexpr (sizeof(v) == 16) {
            if constexpr (N == 2 && have_sse2) {
                const auto k = builtin_cast<int>(v.d);
                mem[0] = -k[1];
                mem[1] = -k[3];
            } else if constexpr (N == 4 && have_sse2) {
                const unsigned bool4 =
                    builtin_cast<uint>(_mm_packs_epi16(
                        _mm_packs_epi32(to_m128i(v), __m128i()), __m128i()))[0] &
                    0x01010101u;
                std::memcpy(mem, &bool4, 4);
            } else if constexpr (std::is_same_v<T, float> && have_mmx) {
                const __m128 k(v);
                const __m64 kk = _mm_cvtps_pi8(and_(k, detail::one16(float())));
                builtin_store<4>(kk, mem, f);
                _mm_empty();
            } else if constexpr (N == 8 && have_sse2) {
                builtin_store<8>(
                    _mm_packs_epi16(to_intrin(builtin_cast<ushort>(v.d) >> 15),
                                    __m128i()),
                    mem, f);
            } else if constexpr (N == 16 && have_sse2) {
                builtin_store(v.d & 1, mem, f);
            } else {
                assert_unreachable<T>();
            }
        } else if constexpr (sizeof(v) == 32) {
            if constexpr (N == 4 && have_avx) {
                auto k = to_m256i(v);
                int bool4;
                if constexpr (have_avx2) {
                    bool4 = _mm256_movemask_epi8(k);
                } else {
                    bool4 = (_mm_movemask_epi8(lo128(k)) |
                             (_mm_movemask_epi8(hi128(k)) << 16));
                }
                bool4 &= 0x01010101;
                std::memcpy(mem, &bool4, 4);
            } else if constexpr (N == 8 && have_avx) {
                const auto k = to_m256i(v);
                const auto k2 = _mm_srli_epi16(_mm_packs_epi16(lo128(k), hi128(k)), 15);
                const auto k3 = _mm_packs_epi16(k2, __m128i());
                builtin_store<8>(k3, mem, f);
            } else if constexpr (N == 16 && have_avx2) {
                const auto x = _mm256_srli_epi16(v, 15);
                const auto bools = _mm_packs_epi16(lo128(x), hi128(x));
                builtin_store<16>(bools, mem, f);
            } else if constexpr (N == 16 && have_avx) {
                const auto bools = 1 & builtin_cast<uchar>(_mm_packs_epi16(
                                           lo128(v.intrin()), hi128(v.intrin())));
                builtin_store<16>(bools, mem, f);
            } else if constexpr (N == 32 && have_avx) {
                builtin_store<32>(1 & v.d, mem, f);
            } else {
                assert_unreachable<T>();
            }
        //} else if constexpr (sizeof(v) == 64) {
        } else {
            assert_unreachable<T>();
        }
        detail::unused(f); // not true, see PR85827
    }

    // masked store {{{2
    template <class T, size_t N, class F>
    static inline void masked_store(const Storage<T, N> v, bool *mem, F,
                                             const Storage<T, N> k) noexcept
    {
        detail::bit_iteration(mask_to_int<N>(k), [&](auto i) { mem[i] = v[i]; });
    }

    // negation {{{2
    template <class T, size_t N, class SizeTag>
    static Vc_INTRINSIC mask_member_type<T> negate(const Storage<T, N> &x,
                                                   SizeTag) noexcept
    {
        return to_storage(~storage_bitcast<uint>(x).d);
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
        using V = simd<U, Abi>;
        constexpr size_t bits_per_element = sizeof(U) * CHAR_BIT;
        if constexpr (!have_avx2 && have_avx && sizeof(V) == 32) {
            if constexpr (N == 8) {
                return _mm256_cmp_ps(
                    _mm256_and_ps(_mm256_castsi256_ps(_mm256_set1_epi32(bits.to_ulong())),
                                  _mm256_castsi256_ps(_mm256_setr_epi32(
                                      0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80))),
                    _mm256_setzero_ps(), _CMP_NEQ_UQ);
            } else if constexpr (N == 4) {
                return _mm256_cmp_pd(
                    _mm256_and_pd(
                        _mm256_castsi256_pd(_mm256_set1_epi64x(bits.to_ulong())),
                        _mm256_castsi256_pd(_mm256_setr_epi64x(0x01, 0x02, 0x04, 0x08))),
                    _mm256_setzero_pd(), _CMP_NEQ_UQ);
            } else {
                assert_unreachable<T>();
            }
        } else if constexpr (bits_per_element >= N) {
            constexpr auto bitmask = generate_builtin<builtin_type_t<U, N>>(
                [](auto i) -> U { return 1ull << i; });
            return builtin_cast<T>(
                (builtin_broadcast<N, U>(bits.to_ullong()) & bitmask) != 0);
        } else if constexpr (sizeof(V) == 16 && sizeof(T) == 1 && have_ssse3) {
            const auto bitmask = to_intrin(make_builtin<uchar>(
                1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128));
            return to_storage(
                builtin_cast<T>(
                    _mm_shuffle_epi8(
                        to_intrin(builtin_type_t<ullong, 2>{bits.to_ullong()}),
                        _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1)) &
                    bitmask) != 0);
        } else if constexpr (sizeof(V) == 32 && sizeof(T) == 1 && have_avx2) {
            const auto bitmask =
                _mm256_broadcastsi128_si256(to_intrin(make_builtin<uchar>(
                    1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128)));
            return to_storage(
                builtin_cast<T>(
                    _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(to_intrin(
                                            builtin_type_t<ullong, 2>{bits.to_ullong()})),
                                        _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                                                         1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                                                         2, 2, 3, 3, 3, 3, 3, 3, 3, 3)) &
                    bitmask) != 0);
            /* TODO:
            } else if constexpr (sizeof(V) == 32 && sizeof(T) == 2 && have_avx2) {
                constexpr auto bitmask = _mm256_broadcastsi128_si256(
                    _mm_setr_epi8(0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x100,
                                  0x200, 0x400, 0x800, 0x1000, 0x2000, 0x4000, 0x8000));
                return builtin_cast<T>(
                           _mm256_shuffle_epi8(
                               _mm256_broadcastsi128_si256(__m128i{bits.to_ullong()}),
                               _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
            1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3)) & bitmask) != 0;
            */
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

    // logical and bitwise operators {{{2
    template <class T, size_t N>
    static constexpr Vc_INTRINSIC Storage<T, N> logical_and(const Storage<T, N> &x,
                                                            const Storage<T, N> &y)
    {
        if constexpr (std::is_same_v<T, bool>) {
            return x.d & y.d;
        } else {
            return detail::and_(x.d, y.d);
        }
    }

    template <class T, size_t N>
    static constexpr Vc_INTRINSIC Storage<T, N> logical_or(const Storage<T, N> &x,
                                                           const Storage<T, N> &y)
    {
        if constexpr (std::is_same_v<T, bool>) {
            return x.d | y.d;
        } else {
            return detail::or_(x.d, y.d);
        }
    }

    template <class T, size_t N>
    static constexpr Vc_INTRINSIC Storage<T, N> bit_and(const Storage<T, N> &x,
                                                        const Storage<T, N> &y)
    {
        if constexpr (std::is_same_v<T, bool>) {
            return x.d & y.d;
        } else {
            return detail::and_(x.d, y.d);
        }
    }

    template <class T, size_t N>
    static constexpr Vc_INTRINSIC Storage<T, N> bit_or(const Storage<T, N> &x,
                                                       const Storage<T, N> &y)
    {
        if constexpr (std::is_same_v<T, bool>) {
            return x.d | y.d;
        } else {
            return detail::or_(x.d, y.d);
        }
    }

    template <class T, size_t N>
    static constexpr Vc_INTRINSIC Storage<T, N> bit_xor(const Storage<T, N> &x,
                                                        const Storage<T, N> &y)
    {
        if constexpr (std::is_same_v<T, bool>) {
            return x.d ^ y.d;
        } else {
            return detail::xor_(x.d, y.d);
        }
    }

    // smart_reference access {{{2
    template <class T, size_t N> static void set(Storage<T, N> &k, int i, bool x) noexcept
    {
        using int_t = builtin_type_t<int_for_sizeof_t<T>, N>;
        auto tmp = reinterpret_cast<int_t>(k.d);
        tmp[i] = -x;
        k.d = auto_cast(tmp);
    }
    // masked_assign{{{2
    template <class T, size_t N>
    static Vc_INTRINSIC void masked_assign(Storage<T, N> k, Storage<T, N> &lhs,
                                           detail::id<Storage<T, N>> rhs)
    {
        lhs = detail::x86::blend(k.d, lhs.d, rhs.d);
    }

    template <class T, size_t N>
    static Vc_INTRINSIC void masked_assign(Storage<T, N> k, Storage<T, N> &lhs, bool rhs)
    {
        if (__builtin_constant_p(rhs)) {
            if (rhs == false) {
                lhs = andnot_(k.d, lhs.d);
            } else {
                lhs = or_(k.d, lhs.d);
            }
            return;
        }
        lhs = detail::x86::blend(k, lhs, detail::data(simd_mask<T>(rhs)));
    }

    //}}}2
};

//}}}1
}  // namespace detail
Vc_VERSIONED_NAMESPACE_END


#endif  // VC_SIMD_GENERICIMPL_H_
