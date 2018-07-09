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

#ifndef VC_SIMD_X86_CONVERT_H_
#define VC_SIMD_X86_CONVERT_H_

#include "storage.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
namespace x86
{
// converts_via_decomposition{{{
template <class From, class To, size_t ToSize> struct converts_via_decomposition {
private:
    static constexpr bool i_to_i = std::is_integral_v<From> && std::is_integral_v<To>;
    static constexpr bool f_to_i =
        std::is_floating_point_v<From> && std::is_integral_v<To>;
    static constexpr bool f_to_f =
        std::is_floating_point_v<From> && std::is_floating_point_v<To>;
    static constexpr bool i_to_f =
        std::is_integral_v<From> && std::is_floating_point_v<To>;

    template <size_t A, size_t B>
    static constexpr bool sizes = sizeof(From) == A && sizeof(To) == B;

public:
    static constexpr bool value =
        (i_to_i && sizes<8, 2> && !have_ssse3 && ToSize == 16) ||
        (i_to_i && sizes<8, 1> && !have_avx512f && ToSize == 16) ||
        (f_to_i && sizes<4, 8> && !have_avx512dq) ||
        (f_to_i && sizes<8, 8> && !have_avx512dq) ||
        (f_to_i && sizes<8, 4> && !have_sse4_1 && ToSize == 16) ||
        (i_to_f && sizes<8, 4> && !have_avx512dq && ToSize == 16) ||
        (i_to_f && sizes<8, 8> && !have_avx512dq && ToSize < 64);
};

template <class From, class To, size_t ToSize>
inline constexpr bool converts_via_decomposition_v =
    converts_via_decomposition<From, To, ToSize>::value;

// }}}
// convert_builtin{{{1
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
template <typename To, typename From, size_t... I>
constexpr Vc_INTRINSIC To convert_builtin(From v0, std::index_sequence<I...>)
{
    using T = typename To::value_type;
    return detail::make_storage<T>(v0[I]...);
}

template <typename To, typename From, size_t... I, size_t... Z>
constexpr Vc_INTRINSIC To convert_builtin_z(From v0, std::index_sequence<I...>,
                                            std::index_sequence<Z...>)
{
    using T = typename To::value_type;
    return detail::make_storage<T>(v0[I]..., ((void)Z, 0)...);
}

template <typename To, typename From, size_t... I>
constexpr Vc_INTRINSIC To convert_builtin(From v0, From v1, std::index_sequence<I...>)
{
    using T = typename To::value_type;
    return detail::make_storage<T>(v0[I]..., v1[I]...);
}

template <typename To, typename From, size_t... I>
constexpr Vc_INTRINSIC To convert_builtin(From v0, From v1, From v2, From v3,
                                          std::index_sequence<I...>)
{
    using T = typename To::value_type;
    return detail::make_storage<T>(static_cast<T>(v0[I])..., static_cast<T>(v1[I])...,
                                   static_cast<T>(v2[I])..., static_cast<T>(v3[I])...);
}

template <typename To, typename From, size_t... I>
constexpr Vc_INTRINSIC To convert_builtin(From v0, From v1, From v2, From v3, From v4,
                                          From v5, From v6, From v7,
                                          std::index_sequence<I...>)
{
    using T = typename To::value_type;
    return detail::make_storage<T>(static_cast<T>(v0[I])..., static_cast<T>(v1[I])...,
                                   static_cast<T>(v2[I])..., static_cast<T>(v3[I])...,
                                   static_cast<T>(v4[I])..., static_cast<T>(v5[I])...,
                                   static_cast<T>(v6[I])..., static_cast<T>(v7[I])...);
}

template <typename To, typename From, size_t... I0, size_t... I1>
constexpr Vc_INTRINSIC To convert_builtin(From v0, From v1, std::index_sequence<I0...>,
                                          std::index_sequence<I1...>)
{
    using T = typename To::value_type;
    return detail::make_storage<T>(static_cast<T>(v0[I0])..., static_cast<T>(v1[I0])...,
                                   (I1, T{})...);
}

template <typename To, typename From, size_t... I0, size_t... I1>
constexpr Vc_INTRINSIC To convert_builtin(From v0, From v1, From v2, From v3,
                                          std::index_sequence<I0...>,
                                          std::index_sequence<I1...>)
{
    using T = typename To::value_type;
    return detail::make_storage<T>(static_cast<T>(v0[I0])..., static_cast<T>(v1[I0])...,
                                   static_cast<T>(v2[I0])..., static_cast<T>(v3[I0])...,
                                   (I1, T{})...);
}

template <typename To, typename From, size_t... I0, size_t... I1>
constexpr Vc_INTRINSIC To convert_builtin(From v0, From v1, From v2, From v3, From v4,
                                          From v5, From v6, From v7,
                                          std::index_sequence<I0...>,
                                          std::index_sequence<I1...>)
{
    using T = typename To::value_type;
    return detail::make_storage<T>(
        static_cast<T>(v0[I0])..., static_cast<T>(v1[I0])..., static_cast<T>(v2[I0])...,
        static_cast<T>(v3[I0])..., static_cast<T>(v4[I0])..., static_cast<T>(v5[I0])...,
        static_cast<T>(v6[I0])..., static_cast<T>(v7[I0])..., (I1, T{})...);
}
#endif  // Vc_USE_BUILTIN_VECTOR_TYPES

#ifdef Vc_WORKAROUND_PR85048
// convert_to declarations {{{1
template <class To, class T, size_t N> To convert_to(Storage<T, N>);
template <class To, class T, size_t N> To convert_to(Storage<T, N>, Storage<T, N>);
template <class To, class T, size_t N>
To convert_to(Storage<T, N>, Storage<T, N>, Storage<T, N>, Storage<T, N>);
template <class To, class T, size_t N>
To convert_to(Storage<T, N>, Storage<T, N>, Storage<T, N>, Storage<T, N>, Storage<T, N>,
              Storage<T, N>, Storage<T, N>, Storage<T, N>);

//}}}1
// work around PR85827
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wunused-variable"
// 1-arg convert_to {{{1
template <class To, class T, size_t N> Vc_INTRINSIC To convert_to(Storage<T, N> v)
{
    using U = typename To::value_type;
    constexpr size_t M = To::width;

    using std::is_integral_v;
    using std::is_signed_v;
    using std::is_unsigned_v;
    using std::is_floating_point_v;

    // [xyz]_to_[xyz] {{{2
    constexpr bool x_to_x = sizeof(v) == 16 && sizeof(To) == 16;
    constexpr bool x_to_y = sizeof(v) == 16 && sizeof(To) == 32;
    constexpr bool x_to_z = sizeof(v) == 16 && sizeof(To) == 64;
    constexpr bool y_to_x = sizeof(v) == 32 && sizeof(To) == 16;
    constexpr bool y_to_y = sizeof(v) == 32 && sizeof(To) == 32;
    constexpr bool y_to_z = sizeof(v) == 32 && sizeof(To) == 64;
    constexpr bool z_to_x = sizeof(v) == 64 && sizeof(To) == 16;
    constexpr bool z_to_y = sizeof(v) == 64 && sizeof(To) == 32;
    constexpr bool z_to_z = sizeof(v) == 64 && sizeof(To) == 64;

    // iX_to_iX {{{2
    constexpr bool i_to_i = is_integral_v<U> && is_integral_v<T>;
    constexpr bool i8_to_i16  = i_to_i && sizeof(T) == 1 && sizeof(U) == 2;
    constexpr bool i8_to_i32  = i_to_i && sizeof(T) == 1 && sizeof(U) == 4;
    constexpr bool i8_to_i64  = i_to_i && sizeof(T) == 1 && sizeof(U) == 8;
    constexpr bool i16_to_i8  = i_to_i && sizeof(T) == 2 && sizeof(U) == 1;
    constexpr bool i16_to_i32 = i_to_i && sizeof(T) == 2 && sizeof(U) == 4;
    constexpr bool i16_to_i64 = i_to_i && sizeof(T) == 2 && sizeof(U) == 8;
    constexpr bool i32_to_i8  = i_to_i && sizeof(T) == 4 && sizeof(U) == 1;
    constexpr bool i32_to_i16 = i_to_i && sizeof(T) == 4 && sizeof(U) == 2;
    constexpr bool i32_to_i64 = i_to_i && sizeof(T) == 4 && sizeof(U) == 8;
    constexpr bool i64_to_i8  = i_to_i && sizeof(T) == 8 && sizeof(U) == 1;
    constexpr bool i64_to_i16 = i_to_i && sizeof(T) == 8 && sizeof(U) == 2;
    constexpr bool i64_to_i32 = i_to_i && sizeof(T) == 8 && sizeof(U) == 4;

    // [fsu]X_to_[fsu]X {{{2
    // ibw = integral && byte or word, i.e. char and short with any signedness
    constexpr bool s64_to_f32 = is_integral_v<T> &&   is_signed_v<T> && sizeof(T) == 8 && is_floating_point_v<U> && sizeof(U) == 4;
    constexpr bool s32_to_f32 = is_integral_v<T> &&   is_signed_v<T> && sizeof(T) == 4 && is_floating_point_v<U> && sizeof(U) == 4;
    constexpr bool s16_to_f32 = is_integral_v<T> &&   is_signed_v<T> && sizeof(T) == 2 && is_floating_point_v<U> && sizeof(U) == 4;
    constexpr bool  s8_to_f32 = is_integral_v<T> &&   is_signed_v<T> && sizeof(T) == 1 && is_floating_point_v<U> && sizeof(U) == 4;
    constexpr bool u64_to_f32 = is_integral_v<T> && is_unsigned_v<T> && sizeof(T) == 8 && is_floating_point_v<U> && sizeof(U) == 4;
    constexpr bool u32_to_f32 = is_integral_v<T> && is_unsigned_v<T> && sizeof(T) == 4 && is_floating_point_v<U> && sizeof(U) == 4;
    constexpr bool u16_to_f32 = is_integral_v<T> && is_unsigned_v<T> && sizeof(T) == 2 && is_floating_point_v<U> && sizeof(U) == 4;
    constexpr bool  u8_to_f32 = is_integral_v<T> && is_unsigned_v<T> && sizeof(T) == 1 && is_floating_point_v<U> && sizeof(U) == 4;
    constexpr bool s64_to_f64 = is_integral_v<T> &&   is_signed_v<T> && sizeof(T) == 8 && is_floating_point_v<U> && sizeof(U) == 8;
    constexpr bool s32_to_f64 = is_integral_v<T> &&   is_signed_v<T> && sizeof(T) == 4 && is_floating_point_v<U> && sizeof(U) == 8;
    constexpr bool u64_to_f64 = is_integral_v<T> && is_unsigned_v<T> && sizeof(T) == 8 && is_floating_point_v<U> && sizeof(U) == 8;
    constexpr bool u32_to_f64 = is_integral_v<T> && is_unsigned_v<T> && sizeof(T) == 4 && is_floating_point_v<U> && sizeof(U) == 8;
    constexpr bool f32_to_s64 = is_integral_v<U> &&   is_signed_v<U> && sizeof(U) == 8 && is_floating_point_v<T> && sizeof(T) == 4;
    constexpr bool f32_to_s32 = is_integral_v<U> &&   is_signed_v<U> && sizeof(U) == 4 && is_floating_point_v<T> && sizeof(T) == 4;
    constexpr bool f32_to_u64 = is_integral_v<U> && is_unsigned_v<U> && sizeof(U) == 8 && is_floating_point_v<T> && sizeof(T) == 4;
    constexpr bool f32_to_u32 = is_integral_v<U> && is_unsigned_v<U> && sizeof(U) == 4 && is_floating_point_v<T> && sizeof(T) == 4;
    constexpr bool f64_to_s64 = is_integral_v<U> &&   is_signed_v<U> && sizeof(U) == 8 && is_floating_point_v<T> && sizeof(T) == 8;
    constexpr bool f64_to_s32 = is_integral_v<U> &&   is_signed_v<U> && sizeof(U) == 4 && is_floating_point_v<T> && sizeof(T) == 8;
    constexpr bool f64_to_u64 = is_integral_v<U> && is_unsigned_v<U> && sizeof(U) == 8 && is_floating_point_v<T> && sizeof(T) == 8;
    constexpr bool f64_to_u32 = is_integral_v<U> && is_unsigned_v<U> && sizeof(U) == 4 && is_floating_point_v<T> && sizeof(T) == 8;
    constexpr bool ibw_to_f32 = is_integral_v<T> && sizeof(T) <= 2 && is_floating_point_v<U> && sizeof(U) == 4;
    constexpr bool ibw_to_f64 = is_integral_v<T> && sizeof(T) <= 2 && is_floating_point_v<U> && sizeof(U) == 8;
    constexpr bool f32_to_ibw = is_integral_v<U> && sizeof(U) <= 2 && is_floating_point_v<T> && sizeof(T) == 4;
    constexpr bool f64_to_ibw = is_integral_v<U> && sizeof(U) <= 2 && is_floating_point_v<T> && sizeof(T) == 8;
    constexpr bool f32_to_f64 = is_floating_point_v<T> && sizeof(T) == 4 && is_floating_point_v<U> && sizeof(U) == 8;
    constexpr bool f64_to_f32 = is_floating_point_v<T> && sizeof(T) == 8 && is_floating_point_v<U> && sizeof(U) == 4;

    if constexpr (i_to_i && y_to_x && !have_avx2) {  //{{{2
        return convert_to<To>(lo128(v), hi128(v));
    } else if constexpr (i_to_i && x_to_y && !have_avx2) {  //{{{2
        return detail::concat(detail::convert_to<detail::Storage<U, M / 2>>(v),
                              detail::convert_to<detail::Storage<U, M / 2>>(
                                  x86::extract_part<1, N / M * 2>(v)));
    } else if constexpr (i_to_i) {  //{{{2
        static_assert(x_to_x || have_avx2,
                      "integral conversions with ymm registers require AVX2");
        static_assert(have_avx512bw || ((sizeof(T) >= 4 || sizeof(v) < 64) &&
                                        (sizeof(U) >= 4 || sizeof(To) < 64)),
                      "8/16-bit integers in zmm registers require AVX512BW");
        static_assert((sizeof(v) < 64 && sizeof(To) < 64) || have_avx512f,
                      "integral conversions with ymm registers require AVX2");
    }

    if constexpr (is_floating_point_v<T> == is_floating_point_v<U> &&  //{{{2
                  sizeof(T) == sizeof(U)) {
        // conversion uses simple bit reinterpretation (or no conversion at all)
        if constexpr (N == M) {
            return to_storage(v.d);
        } else if constexpr (N > M) {
            return to_storage(v.d);
        } else {
            return x86::zeroExtend(v.intrin());
        }
    } else if constexpr (N < M && sizeof(To) > 16) {  // zero extend (eg. xmm -> ymm){{{2
        return x86::zeroExtend(
            convert_to<Storage<U, (16 / sizeof(U) > N) ? 16 / sizeof(U) : N>>(v)
                .intrin());
    } else if constexpr (N > M && sizeof(v) > 16) {  // partial input (eg. ymm -> xmm){{{2
        return convert_to<To>(extract_part<0, N / M>(v));
    } else if constexpr (i64_to_i32) {  //{{{2
        if constexpr (x_to_x && have_avx512vl) {
            return _mm_cvtepi64_epi32(v);
        } else if constexpr (x_to_x) {
            return to_storage(_mm_shuffle_ps(auto_cast(v), __m128(), 8));
            // return _mm_unpacklo_epi64(_mm_shuffle_epi32(v, 8), __m128i());
        } else if constexpr (y_to_x && have_avx512vl) {
            return _mm256_cvtepi64_epi32(v);
        } else if constexpr (y_to_x && have_avx512f) {
            return lo128(_mm512_cvtepi64_epi32(auto_cast(v.d)));
        } else if constexpr (y_to_x) {
            return lo128(_mm256_permute4x64_epi64(_mm256_shuffle_epi32(v, 8), 0 + 4 * 2));
        } else if constexpr (z_to_y) {
            return _mm512_cvtepi64_epi32(v);
        }
    } else if constexpr (i64_to_i16) {  //{{{2
        if constexpr (x_to_x && have_avx512vl) {
            return _mm_cvtepi64_epi16(v);
        } else if constexpr (x_to_x && have_avx512f) {
            return lo128(_mm512_cvtepi64_epi16(auto_cast(v)));
        } else if constexpr (x_to_x && have_ssse3) {
            return _mm_shuffle_epi8(
                v, _mm_setr_epi8(0, 1, 8, 9, -0x80, -0x80, -0x80, -0x80, -0x80, -0x80,
                                 -0x80, -0x80, -0x80, -0x80, -0x80, -0x80));
            // fallback without SSSE3
        } else if constexpr (y_to_x && have_avx512vl) {
            return _mm256_cvtepi64_epi16(v);
        } else if constexpr (y_to_x && have_avx512f) {
            return lo128(_mm512_cvtepi64_epi16(auto_cast(v)));
        } else if constexpr (y_to_x) {
            const auto a = _mm256_shuffle_epi8(
                v, _mm256_setr_epi8(0, 1, 8, 9, -0x80, -0x80, -0x80, -0x80, -0x80, -0x80,
                                    -0x80, -0x80, -0x80, -0x80, -0x80, -0x80, -0x80,
                                    -0x80, -0x80, -0x80, 0, 1, 8, 9, -0x80, -0x80, -0x80,
                                    -0x80, -0x80, -0x80, -0x80, -0x80));
            return lo128(a) | hi128(a);
        } else if constexpr (z_to_x) {
            return _mm512_cvtepi64_epi16(v);
        }
    } else if constexpr (i64_to_i8) {   //{{{2
        if constexpr (x_to_x && have_avx512vl) {
            return _mm_cvtepi64_epi8(v);
        } else if constexpr (x_to_x && have_avx512f) {
            return lo128(_mm512_cvtepi64_epi8(zeroExtend(v.intrin())));
        } else if constexpr (y_to_x && have_avx512vl) {
            return _mm256_cvtepi64_epi8(v);
        } else if constexpr (y_to_x && have_avx512f) {
            return _mm512_cvtepi64_epi8(zeroExtend(v.intrin()));
        } else if constexpr (z_to_x) {
            return _mm512_cvtepi64_epi8(v);
        }
    } else if constexpr (i32_to_i64) {    //{{{2
        if constexpr (have_sse4_1 && x_to_x) {
            return is_signed_v<T> ? _mm_cvtepi32_epi64(v) : _mm_cvtepu32_epi64(v);
        } else if constexpr (x_to_x) {
            return _mm_unpacklo_epi32(v,
                                      is_signed_v<T> ? _mm_srai_epi32(v, 31) : __m128i());
        } else if constexpr (x_to_y) {
            return is_signed_v<T> ? _mm256_cvtepi32_epi64(v) : _mm256_cvtepu32_epi64(v);
        } else if constexpr (y_to_z) {
            return is_signed_v<T> ? _mm512_cvtepi32_epi64(v) : _mm512_cvtepu32_epi64(v);
        }
    } else if constexpr (i32_to_i16) {  //{{{2
        if constexpr (x_to_x && have_avx512vl) {
            return _mm_cvtepi32_epi16(v);
        } else if constexpr (x_to_x && have_avx512f) {
            return lo128(_mm512_cvtepi32_epi16(auto_cast(v)));
        } else if constexpr (x_to_x && have_ssse3) {
            return _mm_shuffle_epi8(
                v, _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, -0x80, -0x80, -0x80, -0x80,
                                 -0x80, -0x80, -0x80, -0x80));
        } else if constexpr (x_to_x) {
            auto a = _mm_unpacklo_epi16(v, __m128i());  // 0o.o 1o.o
            auto b = _mm_unpackhi_epi16(v, __m128i());  // 2o.o 3o.o
            auto c = _mm_unpacklo_epi16(a, b);          // 02oo ..oo
            auto d = _mm_unpackhi_epi16(a, b);          // 13oo ..oo
            return _mm_unpacklo_epi16(c, d);            // 0123 oooo
        } else if constexpr (y_to_x && have_avx512vl) {
            return _mm256_cvtepi32_epi16(v);
        } else if constexpr (y_to_x && have_avx512f) {
            return lo128(_mm512_cvtepi32_epi16(auto_cast(v)));
        } else if constexpr (y_to_x) {
            auto a = _mm256_shuffle_epi8(
                v,
                _mm256_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, -0x80, -0x80, -0x80, -0x80,
                                 -0x80, -0x80, -0x80, -0x80, 0, 1, 4, 5, 8, 9, 12, 13,
                                 -0x80, -0x80, -0x80, -0x80, -0x80, -0x80, -0x80, -0x80));
            return lo128(_mm256_permute4x64_epi64(a, 0xf8));  // a[0] a[2] | a[3] a[3]
        } else if constexpr (z_to_y) {
            return _mm512_cvtepi32_epi16(v);
        }
    } else if constexpr (i32_to_i8) {   //{{{2
        if constexpr (x_to_x && have_avx512vl) {
            return _mm_cvtepi32_epi8(v);
        } else if constexpr (x_to_x && have_avx512f) {
            return lo128(_mm512_cvtepi32_epi8(zeroExtend(v.intrin())));
        } else if constexpr (x_to_x && have_ssse3) {
            return _mm_shuffle_epi8(
                v, _mm_setr_epi8(0, 4, 8, 12, -0x80, -0x80, -0x80, -0x80, -0x80, -0x80,
                                 -0x80, -0x80, -0x80, -0x80, -0x80, -0x80));
        } else if constexpr (x_to_x) {
            const auto a = _mm_unpacklo_epi8(v, v);  // 0... .... 1... ....
            const auto b = _mm_unpackhi_epi8(v, v);  // 2... .... 3... ....
            const auto c = _mm_unpacklo_epi8(a, b);  // 02.. .... .... ....
            const auto d = _mm_unpackhi_epi8(a, b);  // 13.. .... .... ....
            const auto e = _mm_unpacklo_epi8(c, d);  // 0123 .... .... ....
            return e & _mm_cvtsi32_si128(-1);
        } else if constexpr (y_to_x && have_avx512vl) {
            return _mm256_cvtepi32_epi8(v);
        } else if constexpr (y_to_x && have_avx512f) {
            return _mm512_cvtepi32_epi8(zeroExtend(v.intrin()));
        } else if constexpr (z_to_x) {
            return _mm512_cvtepi32_epi8(v);
        }
    } else if constexpr (i16_to_i64) {  //{{{2
        if constexpr (x_to_x && have_sse4_1) {
            return is_signed_v<T> ? _mm_cvtepi16_epi64(v) : _mm_cvtepu16_epi64(v);
        } else if constexpr (x_to_x && is_signed_v<T>) {
            auto x = _mm_srai_epi16(v, 15);
            auto y = _mm_unpacklo_epi16(v, x);
            x = _mm_unpacklo_epi16(x, x);
            return _mm_unpacklo_epi32(y, x);
        } else if constexpr (x_to_x) {
            return _mm_unpacklo_epi32(_mm_unpacklo_epi16(v, __m128i()), __m128i());
        } else if constexpr (x_to_y) {
            return is_signed_v<T> ? _mm256_cvtepi16_epi64(v) : _mm256_cvtepu16_epi64(v);
        } else if constexpr (x_to_z) {
            return is_signed_v<T> ? _mm512_cvtepi16_epi64(v) : _mm512_cvtepu16_epi64(v);
        }
    } else if constexpr (i16_to_i32) {  //{{{2
        if constexpr (x_to_x && have_sse4_1) {
            return is_signed_v<T> ? _mm_cvtepi16_epi32(v) : _mm_cvtepu16_epi32(v);
        } else if constexpr (x_to_x && is_signed_v<T>) {
            return _mm_srai_epi32(_mm_unpacklo_epi16(v, v), 16);
        } else if constexpr (x_to_x && is_unsigned_v<T>) {
            return _mm_unpacklo_epi16(v, __m128i());
        } else if constexpr (x_to_y) {
            return is_signed_v<T> ? _mm256_cvtepi16_epi32(v) : _mm256_cvtepu16_epi32(v);
        } else if constexpr (y_to_z) {
            return is_signed_v<T> ? _mm512_cvtepi16_epi32(v) : _mm512_cvtepu16_epi32(v);
        }
    } else if constexpr (i16_to_i8) {   //{{{2
        if constexpr (x_to_x && have_avx512bw_vl) {
            return _mm_cvtepi16_epi8(v);
        } else if constexpr (x_to_x && have_avx512bw) {
            return lo128(_mm512_cvtepi16_epi8(zeroExtend(v.intrin())));
        } else if constexpr (x_to_x && have_ssse3) {
            return _mm_shuffle_epi8(
                v, _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, -0x80, -0x80, -0x80, -0x80,
                                 -0x80, -0x80, -0x80, -0x80));
        } else if constexpr (x_to_x) {
            auto a = _mm_unpacklo_epi8(v, v);  // 00.. 11.. 22.. 33..
            auto b = _mm_unpackhi_epi8(v, v);  // 44.. 55.. 66.. 77..
            auto c = _mm_unpacklo_epi8(a, b);  // 0404 .... 1515 ....
            auto d = _mm_unpackhi_epi8(a, b);  // 2626 .... 3737 ....
            auto e = _mm_unpacklo_epi8(c, d);  // 0246 0246 .... ....
            auto f = _mm_unpackhi_epi8(c, d);  // 1357 1357 .... ....
            return _mm_unpacklo_epi8(e, f);
        } else if constexpr (y_to_x && have_avx512bw_vl) {
            return _mm256_cvtepi16_epi8(v);
        } else if constexpr (y_to_x && have_avx512bw) {
            return lo256(_mm512_cvtepi16_epi8(zeroExtend(v.intrin())));
        } else if constexpr (y_to_x) {
            auto a = _mm256_shuffle_epi8(
                v,
                _mm256_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, -0x80, -0x80, -0x80, -0x80,
                                 -0x80, -0x80, -0x80, -0x80, -0x80, -0x80, -0x80, -0x80,
                                 -0x80, -0x80, -0x80, -0x80, 0, 2, 4, 6, 8, 10, 12, 14));
            return lo128(a) | hi128(a);
        } else if constexpr (z_to_y && have_avx512bw) {
            return _mm512_cvtepi16_epi8(v);
        } else if constexpr (z_to_y)  {
            assert_unreachable<T>();
        }
    } else if constexpr (i8_to_i64) {  //{{{2
        if constexpr (x_to_x && have_sse4_1) {
            return is_signed_v<T> ? _mm_cvtepi8_epi64(v) : _mm_cvtepu8_epi64(v);
        } else if constexpr (x_to_x && is_signed_v<T>) {
            if constexpr (have_ssse3) {
                auto dup = _mm_unpacklo_epi8(v, v);
                auto epi16 = _mm_srai_epi16(dup, 8);
                _mm_shuffle_epi8(
                    epi16, _mm_setr_epi8(0, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 3, 3));
            } else {
                auto x = _mm_unpacklo_epi8(v, v);
                x = _mm_unpacklo_epi16(x, x);
                return _mm_unpacklo_epi32(_mm_srai_epi32(x, 24), _mm_srai_epi32(x, 31));
            }
        } else if constexpr (x_to_x) {
            return _mm_unpacklo_epi32(
                _mm_unpacklo_epi16(_mm_unpacklo_epi8(v, __m128i()), __m128i()),
                __m128i());
        } else if constexpr (x_to_y) {
            return is_signed_v<T> ? _mm256_cvtepi8_epi64(v) : _mm256_cvtepu8_epi64(v);
        } else if constexpr (x_to_z) {
            return is_signed_v<T> ? _mm512_cvtepi8_epi64(v) : _mm512_cvtepu8_epi64(v);
        }
    } else if constexpr (i8_to_i32) {  //{{{2
        if constexpr (x_to_x && have_sse4_1) {
            return is_signed_v<T> ? _mm_cvtepi8_epi32(v) : _mm_cvtepu8_epi32(v);
        } else if constexpr (x_to_x && is_signed_v<T>) {
            const auto x = _mm_unpacklo_epi8(v, v);
            return _mm_srai_epi32(_mm_unpacklo_epi16(x, x), 24);
        } else if constexpr (x_to_x && is_unsigned_v<T>) {
            return _mm_unpacklo_epi16(_mm_unpacklo_epi8(v, __m128i()), __m128i());
        } else if constexpr (x_to_y) {
            return is_signed_v<T> ? _mm256_cvtepi8_epi32(v) : _mm256_cvtepu8_epi32(v);
        } else if constexpr (x_to_z) {
            return is_signed_v<T> ? _mm512_cvtepi8_epi32(v) : _mm512_cvtepu8_epi32(v);
        }
    } else if constexpr (i8_to_i16) {   //{{{2
        if constexpr (x_to_x && have_sse4_1) {
            return is_signed_v<T> ? _mm_cvtepi8_epi16(v) : _mm_cvtepu8_epi16(v);
        } else if constexpr (x_to_x && is_signed_v<T>) {
            return _mm_srai_epi16(_mm_unpacklo_epi8(v, v), 8);
        } else if constexpr (x_to_x && is_unsigned_v<T>) {
            return _mm_unpacklo_epi8(v, __m128i());
        } else if constexpr (x_to_y) {
            return is_signed_v<T> ? _mm256_cvtepi8_epi16(v) : _mm256_cvtepu8_epi16(v);
        } else if constexpr (y_to_z && have_avx512bw) {
            return is_signed_v<T> ? _mm512_cvtepi8_epi16(v) : _mm512_cvtepu8_epi16(v);
        } else if constexpr (y_to_z) {
            assert_unreachable<T>();
        }
    } else if constexpr (f32_to_s64) {  //{{{2
        if constexpr (have_avx512dq_vl && x_to_x) {
            return _mm_cvttps_epi64(v);
        } else if constexpr (have_avx512dq_vl && x_to_y) {
            return _mm256_cvttps_epi64(v);
        } else if constexpr (have_avx512dq && y_to_z) {
            return _mm512_cvttps_epi64(v);
        } // else use scalar fallback
    } else if constexpr (f32_to_u64) {  //{{{2
        if constexpr (have_avx512dq_vl && x_to_x) {
            return _mm_cvttps_epu64(v);
        } else if constexpr (have_avx512dq_vl && x_to_y) {
            return _mm256_cvttps_epu64(v);
        } else if constexpr (have_avx512dq && y_to_z) {
            return _mm512_cvttps_epu64(v);
        } // else use scalar fallback
    } else if constexpr (f32_to_s32) {  //{{{2
        if constexpr (x_to_x || y_to_y || z_to_z) {
            // go to fallback, it does the right thing
        } else {
            assert_unreachable<T>();
        }
    } else if constexpr (f32_to_u32) {  //{{{2
        // the __builtin_constant_p hack enables constant propagation
        if constexpr (have_avx512vl && x_to_x) {
            const builtin_type_t<float, 4> x = v.d;
            return __builtin_constant_p(x) ? make_builtin<uint>(x[0], x[1], x[2], x[3])
                                           : builtin_cast<uint>(_mm_cvttps_epu32(v));
        } else if constexpr (have_avx512f && x_to_x) {
            const builtin_type_t<float, 4> x = v.d;
            return __builtin_constant_p(x)
                       ? make_builtin<uint>(x[0], x[1], x[2], x[3])
                       : builtin_cast<uint>(lo128(_mm512_cvttps_epu32(auto_cast(v))));
        } else if constexpr (have_avx512vl && y_to_y) {
            const builtin_type_t<float, 8> x = v.d;
            return __builtin_constant_p(x) ? make_builtin<uint>(x[0], x[1], x[2], x[3],
                                                                x[4], x[5], x[6], x[7])
                                           : builtin_cast<uint>(_mm256_cvttps_epu32(v));
        } else if constexpr (have_avx512f && y_to_y) {
            const builtin_type_t<float, 8> x = v.d;
            return __builtin_constant_p(x)
                       ? make_builtin<uint>(x[0], x[1], x[2], x[3], x[4], x[5], x[6],
                                            x[7])
                       : builtin_cast<uint>(lo256(_mm512_cvttps_epu32(auto_cast(v))));
        } else if constexpr (x_to_x || y_to_y || z_to_z) {
            // go to fallback, it does the right thing. We can't use the _mm_floor_ps -
            // 0x8000'0000 trick for f32->u32 because it would discard small input values
            // (only 24 mantissa bits)
        } else {
            assert_unreachable<T>();
        }
    } else if constexpr (f32_to_ibw) {  //{{{2
        return convert_to<To>(convert_to<Storage<int, N>>(v));
    } else if constexpr (f64_to_s64) {  //{{{2
        if constexpr (have_avx512dq_vl && x_to_x) {
            return _mm_cvttpd_epi64(v);
        } else if constexpr (have_avx512dq_vl && y_to_y) {
            return _mm256_cvttpd_epi64(v);
        } else if constexpr (have_avx512dq && z_to_z) {
            return _mm512_cvttpd_epi64(v);
        } // else use scalar fallback
    } else if constexpr (f64_to_u64) {  //{{{2
        if constexpr (have_avx512dq_vl && x_to_x) {
            return _mm_cvttpd_epu64(v);
        } else if constexpr (have_avx512dq_vl && y_to_y) {
            return _mm256_cvttpd_epu64(v);
        } else if constexpr (have_avx512dq && z_to_z) {
            return _mm512_cvttpd_epu64(v);
        } // else use scalar fallback
    } else if constexpr (f64_to_s32) {  //{{{2
        if constexpr (x_to_x) {
            return _mm_cvttpd_epi32(v);
        } else if constexpr (y_to_x) {
            return _mm256_cvttpd_epi32(v);
        } else if constexpr (z_to_y) {
            return _mm512_cvttpd_epi32(v);
        }
    } else if constexpr (f64_to_u32) {  //{{{2
        if constexpr (have_avx512vl && x_to_x) {
            return _mm_cvttpd_epu32(v);
        } else if constexpr (have_sse4_1 && x_to_x) {
            return builtin_cast<uint>(_mm_cvttpd_epi32(_mm_floor_pd(v) - 0x8000'0000u)) ^
                   0x8000'0000u;
        } else if constexpr (x_to_x) {
            // use scalar fallback: it's only 2 values to convert, can't get much better
            // than scalar decomposition
        } else if constexpr (have_avx512vl && y_to_x) {
            return _mm256_cvttpd_epu32(v);
        } else if constexpr (y_to_x) {
            return builtin_cast<uint>(
                       _mm256_cvttpd_epi32(_mm256_floor_pd(v) - 0x8000'0000u)) ^
                   0x8000'0000u;
        } else if constexpr (z_to_y) {
            return _mm512_cvttpd_epu32(v);
        }
    } else if constexpr (f64_to_ibw) {  //{{{2
        return convert_to<To>(convert_to<Storage<int, (N < 4 ? 4 : N)>>(v));
    } else if constexpr (s64_to_f32) {  //{{{2
        if constexpr (x_to_x && have_avx512dq_vl) {
            return _mm_cvtepi64_ps(v);
        } else if constexpr (y_to_x && have_avx512dq_vl) {
            return _mm256_cvtepi64_ps(v);
        } else if constexpr (z_to_y && have_avx512dq) {
            return _mm512_cvtepi64_ps(v);
        } else if constexpr (z_to_y) {
            return _mm512_cvtpd_ps(convert_to<z_f64>(v));
        }
    } else if constexpr (u64_to_f32) {  //{{{2
        if constexpr (x_to_x && have_avx512dq_vl) {
            return _mm_cvtepu64_ps(v);
        } else if constexpr (y_to_x && have_avx512dq_vl) {
            return _mm256_cvtepu64_ps(v);
        } else if constexpr (z_to_y && have_avx512dq) {
            return _mm512_cvtepu64_ps(v);
        } else if constexpr (z_to_y) {
            return lo256(_mm512_cvtepu32_ps(
                       auto_cast(_mm512_cvtepi64_epi32(_mm512_srai_epi64(v, 32))))) *
                       0x100000000LL +
                   lo256(_mm512_cvtepu32_ps(auto_cast(_mm512_cvtepi64_epi32(v))));
        }
    } else if constexpr (s32_to_f32) {  //{{{2
        // use fallback (builtin conversion)
    } else if constexpr (u32_to_f32) {  //{{{2
        if constexpr(x_to_x && have_avx512vl) {
            // use fallback
        } else if constexpr(x_to_x && have_avx512f) {
            return lo128(_mm512_cvtepu32_ps(auto_cast(v)));
        } else if constexpr(x_to_x && (have_fma || have_fma4)) {
            // work around PR85819
            return 0x10000 * _mm_cvtepi32_ps(to_intrin(v.d >> 16)) +
                   _mm_cvtepi32_ps(to_intrin(v.d & 0xffff));
        } else if constexpr(y_to_y && have_avx512vl) {
            // use fallback
        } else if constexpr(y_to_y && have_avx512f) {
            return lo256(_mm512_cvtepu32_ps(auto_cast(v)));
        } else if constexpr(y_to_y) {
            // work around PR85819
            return 0x10000 * _mm256_cvtepi32_ps(to_intrin(v.d >> 16)) +
                   _mm256_cvtepi32_ps(to_intrin(v.d & 0xffff));
        } // else use fallback (builtin conversion)
    } else if constexpr (ibw_to_f32) {  //{{{2
        if constexpr (M == 4 || have_avx2) {
            return convert_to<To>(convert_to<Storage<int, M>>(v));
        } else {
            static_assert(x_to_y);
            x_i32 a, b;
            if constexpr (have_sse4_1) {
                a = sizeof(T) == 2
                        ? (is_signed_v<T> ? _mm_cvtepi16_epi32(v) : _mm_cvtepu16_epi32(v))
                        : (is_signed_v<T> ? _mm_cvtepi8_epi32(v) : _mm_cvtepu8_epi32(v));
                const auto w = _mm_shuffle_epi32(v, sizeof(T) == 2 ? 0xee : 0xe9);
                b = sizeof(T) == 2
                        ? (is_signed_v<T> ? _mm_cvtepi16_epi32(w) : _mm_cvtepu16_epi32(w))
                        : (is_signed_v<T> ? _mm_cvtepi8_epi32(w) : _mm_cvtepu8_epi32(w));
            } else {
                __m128i tmp;
                if constexpr (sizeof(T) == 1) {
                    tmp = is_signed_v<T> ? _mm_srai_epi16(_mm_unpacklo_epi8(v, v), 8):
                        _mm_unpacklo_epi8(v, __m128i());
                } else {
                    static_assert(sizeof(T) == 2);
                    tmp = v;
                }
                a = is_signed_v<T> ? _mm_srai_epi32(_mm_unpacklo_epi16(tmp, tmp), 16)
                                   : _mm_unpacklo_epi16(tmp, __m128i());
                b = is_signed_v<T> ? _mm_srai_epi32(_mm_unpackhi_epi16(tmp, tmp), 16)
                                   : _mm_unpackhi_epi16(tmp, __m128i());
            }
            return convert_to<To>(a, b);
        }
    } else if constexpr (s64_to_f64) {  //{{{2
        if constexpr (x_to_x && have_avx512dq_vl) {
            return _mm_cvtepi64_pd(v);
        } else if constexpr (y_to_y && have_avx512dq_vl) {
            return _mm256_cvtepi64_pd(v);
        } else if constexpr (z_to_z && have_avx512dq) {
            return _mm512_cvtepi64_pd(v);
        } else if constexpr (z_to_z) {
            return _mm512_cvtepi32_pd(_mm512_cvtepi64_epi32(to_intrin(v.d >> 32))) *
                       0x100000000LL +
                   _mm512_cvtepu32_pd(_mm512_cvtepi64_epi32(v));
        }
    } else if constexpr (u64_to_f64) {  //{{{2
        if constexpr (x_to_x && have_avx512dq_vl) {
            return _mm_cvtepu64_pd(v);
        } else if constexpr (y_to_y && have_avx512dq_vl) {
            return _mm256_cvtepu64_pd(v);
        } else if constexpr (z_to_z && have_avx512dq) {
            return _mm512_cvtepu64_pd(v);
        } else if constexpr (z_to_z) {
            return _mm512_cvtepu32_pd(_mm512_cvtepi64_epi32(to_intrin(v.d >> 32))) *
                       0x100000000LL +
                   _mm512_cvtepu32_pd(_mm512_cvtepi64_epi32(v));
        }
    } else if constexpr (s32_to_f64) {  //{{{2
        if constexpr (x_to_x) {
            return _mm_cvtepi32_pd(v);
        } else if constexpr (x_to_y) {
            return _mm256_cvtepi32_pd(v);
        } else if constexpr (y_to_z) {
            return _mm512_cvtepi32_pd(v);
        }
    } else if constexpr (u32_to_f64) {  //{{{2
        if constexpr (x_to_x && have_avx512vl) {
            return _mm_cvtepu32_pd(v);
        } else if constexpr (x_to_x && have_avx512f) {
            return lo128(_mm512_cvtepu32_pd(auto_cast(v)));
        } else if constexpr (x_to_x) {
            return _mm_cvtepi32_pd(to_intrin(v.d ^ 0x8000'0000u)) + 0x8000'0000u;
        } else if constexpr (x_to_y && have_avx512vl) {
            return _mm256_cvtepu32_pd(v);
        } else if constexpr (x_to_y && have_avx512f) {
            return lo256(_mm512_cvtepu32_pd(auto_cast(v)));
        } else if constexpr (x_to_y) {
            return _mm256_cvtepi32_pd(to_intrin(v.d ^ 0x8000'0000u)) + 0x8000'0000u;
        } else if constexpr (y_to_z) {
            return _mm512_cvtepu32_pd(v);
        }
    } else if constexpr (ibw_to_f64) {  //{{{2
        return convert_to<To>(convert_to<Storage<int, std::max(size_t(4), M)>>(v));
    } else if constexpr (f32_to_f64) {  //{{{2
        if constexpr (x_to_x) {
            return _mm_cvtps_pd(v);
        } else if constexpr (x_to_y) {
            return _mm256_cvtps_pd(v);
        } else if constexpr (y_to_z) {
            return _mm512_cvtps_pd(v);
        }
    } else if constexpr (f64_to_f32) {  //{{{2
        if constexpr (x_to_x) {
            return _mm_cvtpd_ps(v);
        } else if constexpr (y_to_x) {
            return _mm256_cvtpd_ps(v);
        } else if constexpr (z_to_y) {
            return _mm512_cvtpd_ps(v);
        }
    } else {  //{{{2
        assert_unreachable<T>();
    }

    // fallback:{{{2
    if constexpr (N >= M) {
        return convert_builtin<To>(v.d, std::make_index_sequence<M>());
    } else {
        return convert_builtin_z<To>(v.d, std::make_index_sequence<N>(),
                                     std::make_index_sequence<M - N>());
    }
    //}}}
} // }}}
// 2-arg convert_to {{{1
template <class To, class T, size_t N>
Vc_INTRINSIC To convert_to(Storage<T, N> v0, Storage<T, N> v1)
{
    using U = typename To::value_type;
    constexpr size_t M = To::width;

    using std::is_integral_v;
    using std::is_signed_v;
    using std::is_unsigned_v;
    using std::is_floating_point_v;

    static_assert(
        2 * N <= M,
        "v1 would be discarded; use the one-argument convert_to overload instead");

    // [xyz]_to_[xyz] {{{2
    constexpr bool x_to_x = sizeof(v0) == 16 && sizeof(To) == 16;
    constexpr bool x_to_y = sizeof(v0) == 16 && sizeof(To) == 32;
    constexpr bool x_to_z = sizeof(v0) == 16 && sizeof(To) == 64;
    constexpr bool y_to_x = sizeof(v0) == 32 && sizeof(To) == 16;
    constexpr bool y_to_y = sizeof(v0) == 32 && sizeof(To) == 32;
    constexpr bool y_to_z = sizeof(v0) == 32 && sizeof(To) == 64;
    constexpr bool z_to_x = sizeof(v0) == 64 && sizeof(To) == 16;
    constexpr bool z_to_y = sizeof(v0) == 64 && sizeof(To) == 32;
    constexpr bool z_to_z = sizeof(v0) == 64 && sizeof(To) == 64;

    // iX_to_iX {{{2
    constexpr bool i_to_i = std::is_integral_v<U> && std::is_integral_v<T>;
    constexpr bool i8_to_i16  = i_to_i && sizeof(T) == 1 && sizeof(U) == 2;
    constexpr bool i8_to_i32  = i_to_i && sizeof(T) == 1 && sizeof(U) == 4;
    constexpr bool i8_to_i64  = i_to_i && sizeof(T) == 1 && sizeof(U) == 8;
    constexpr bool i16_to_i8  = i_to_i && sizeof(T) == 2 && sizeof(U) == 1;
    constexpr bool i16_to_i32 = i_to_i && sizeof(T) == 2 && sizeof(U) == 4;
    constexpr bool i16_to_i64 = i_to_i && sizeof(T) == 2 && sizeof(U) == 8;
    constexpr bool i32_to_i8  = i_to_i && sizeof(T) == 4 && sizeof(U) == 1;
    constexpr bool i32_to_i16 = i_to_i && sizeof(T) == 4 && sizeof(U) == 2;
    constexpr bool i32_to_i64 = i_to_i && sizeof(T) == 4 && sizeof(U) == 8;
    constexpr bool i64_to_i8  = i_to_i && sizeof(T) == 8 && sizeof(U) == 1;
    constexpr bool i64_to_i16 = i_to_i && sizeof(T) == 8 && sizeof(U) == 2;
    constexpr bool i64_to_i32 = i_to_i && sizeof(T) == 8 && sizeof(U) == 4;

    // [fsu]X_to_[fsu]X {{{2
    // ibw = integral && byte or word, i.e. char and short with any signedness
    constexpr bool i64_to_f32 = is_integral_v<T> &&                     sizeof(T) == 8 && is_floating_point_v<U> && sizeof(U) == 4;
    constexpr bool s32_to_f32 = is_integral_v<T> &&   is_signed_v<T> && sizeof(T) == 4 && is_floating_point_v<U> && sizeof(U) == 4;
    constexpr bool s16_to_f32 = is_integral_v<T> &&   is_signed_v<T> && sizeof(T) == 2 && is_floating_point_v<U> && sizeof(U) == 4;
    constexpr bool  s8_to_f32 = is_integral_v<T> &&   is_signed_v<T> && sizeof(T) == 1 && is_floating_point_v<U> && sizeof(U) == 4;
    constexpr bool u32_to_f32 = is_integral_v<T> && is_unsigned_v<T> && sizeof(T) == 4 && is_floating_point_v<U> && sizeof(U) == 4;
    constexpr bool u16_to_f32 = is_integral_v<T> && is_unsigned_v<T> && sizeof(T) == 2 && is_floating_point_v<U> && sizeof(U) == 4;
    constexpr bool  u8_to_f32 = is_integral_v<T> && is_unsigned_v<T> && sizeof(T) == 1 && is_floating_point_v<U> && sizeof(U) == 4;
    constexpr bool s64_to_f64 = is_integral_v<T> &&   is_signed_v<T> && sizeof(T) == 8 && is_floating_point_v<U> && sizeof(U) == 8;
    constexpr bool s32_to_f64 = is_integral_v<T> &&   is_signed_v<T> && sizeof(T) == 4 && is_floating_point_v<U> && sizeof(U) == 8;
    constexpr bool s16_to_f64 = is_integral_v<T> &&   is_signed_v<T> && sizeof(T) == 2 && is_floating_point_v<U> && sizeof(U) == 8;
    constexpr bool  s8_to_f64 = is_integral_v<T> &&   is_signed_v<T> && sizeof(T) == 1 && is_floating_point_v<U> && sizeof(U) == 8;
    constexpr bool u64_to_f64 = is_integral_v<T> && is_unsigned_v<T> && sizeof(T) == 8 && is_floating_point_v<U> && sizeof(U) == 8;
    constexpr bool u32_to_f64 = is_integral_v<T> && is_unsigned_v<T> && sizeof(T) == 4 && is_floating_point_v<U> && sizeof(U) == 8;
    constexpr bool u16_to_f64 = is_integral_v<T> && is_unsigned_v<T> && sizeof(T) == 2 && is_floating_point_v<U> && sizeof(U) == 8;
    constexpr bool  u8_to_f64 = is_integral_v<T> && is_unsigned_v<T> && sizeof(T) == 1 && is_floating_point_v<U> && sizeof(U) == 8;
    constexpr bool f32_to_s64 = is_integral_v<U> &&   is_signed_v<U> && sizeof(U) == 8 && is_floating_point_v<T> && sizeof(T) == 4;
    constexpr bool f32_to_s32 = is_integral_v<U> &&   is_signed_v<U> && sizeof(U) == 4 && is_floating_point_v<T> && sizeof(T) == 4;
    constexpr bool f32_to_u64 = is_integral_v<U> && is_unsigned_v<U> && sizeof(U) == 8 && is_floating_point_v<T> && sizeof(T) == 4;
    constexpr bool f32_to_u32 = is_integral_v<U> && is_unsigned_v<U> && sizeof(U) == 4 && is_floating_point_v<T> && sizeof(T) == 4;
    constexpr bool f64_to_s64 = is_integral_v<U> &&   is_signed_v<U> && sizeof(U) == 8 && is_floating_point_v<T> && sizeof(T) == 8;
    constexpr bool f64_to_s32 = is_integral_v<U> &&   is_signed_v<U> && sizeof(U) == 4 && is_floating_point_v<T> && sizeof(T) == 8;
    constexpr bool f64_to_u64 = is_integral_v<U> && is_unsigned_v<U> && sizeof(U) == 8 && is_floating_point_v<T> && sizeof(T) == 8;
    constexpr bool f64_to_u32 = is_integral_v<U> && is_unsigned_v<U> && sizeof(U) == 4 && is_floating_point_v<T> && sizeof(T) == 8;
    constexpr bool f32_to_ibw = is_integral_v<U> && sizeof(U) <= 2 && is_floating_point_v<T> && sizeof(T) == 4;
    constexpr bool f64_to_ibw = is_integral_v<U> && sizeof(U) <= 2 && is_floating_point_v<T> && sizeof(T) == 8;
    constexpr bool f32_to_f64 = is_floating_point_v<T> && sizeof(T) == 4 && is_floating_point_v<U> && sizeof(U) == 8;
    constexpr bool f64_to_f32 = is_floating_point_v<T> && sizeof(T) == 8 && is_floating_point_v<U> && sizeof(U) == 4;

    if constexpr (i_to_i && y_to_x && !have_avx2) {  //{{{2
        // <double, 4>, <double, 4> => <short, 8>
        return convert_to<To>(lo128(v0), hi128(v0), lo128(v1), hi128(v1));
    } else if constexpr (i_to_i) {  // assert ISA {{{2
        static_assert(x_to_x || have_avx2,
                      "integral conversions with ymm registers require AVX2");
        static_assert(have_avx512bw || ((sizeof(T) >= 4 || sizeof(v0) < 64) &&
                                        (sizeof(U) >= 4 || sizeof(To) < 64)),
                      "8/16-bit integers in zmm registers require AVX512BW");
        static_assert((sizeof(v0) < 64 && sizeof(To) < 64) || have_avx512f,
                      "integral conversions with ymm registers require AVX2");
    }
    // concat => use 1-arg convert_to {{{2
    if constexpr ((sizeof(v0) == 16 && have_avx2) ||
                  (sizeof(v0) == 16 && have_avx && std::is_floating_point_v<T>) ||
                  (sizeof(v0) == 32 && have_avx512f && (sizeof(T) >= 4 || have_avx512bw))) {
        // The ISA can handle wider input registers, so concat and use one-arg
        // implementation. This reduces code duplication considerably.
        return convert_to<To>(detail::concat(v0, v1));
    } else {  //{{{2
        // conversion using bit reinterpretation (or no conversion at all) should all go
        // through the concat branch above:
        static_assert(!(std::is_floating_point_v<T> == std::is_floating_point_v<U> &&
                        sizeof(T) == sizeof(U)));
        if constexpr (2 * N < M && sizeof(To) > 16) {  // handle all zero extension{{{2
            constexpr size_t Min = 16 / sizeof(U);
            return x86::zeroExtend(
                convert_to<Storage<U, (Min > 2 * N) ? Min : 2 * N>>(v0, v1).intrin());
        } else if constexpr (i64_to_i32) {  //{{{2
            if constexpr (x_to_x) {
                return to_storage(_mm_shuffle_ps(auto_cast(v0), auto_cast(v1), 0x88));
            } else if constexpr (y_to_y) {
                // AVX512F is not available (would concat otherwise)
                return to_storage(fixup_avx_xzyw(
                    _mm256_shuffle_ps(auto_cast(v0), auto_cast(v1), 0x88)));
                // alternative:
                // const auto v0_abxxcdxx = _mm256_shuffle_epi32(v0, 8);
                // const auto v1_efxxghxx = _mm256_shuffle_epi32(v1, 8);
                // const auto v_abefcdgh = _mm256_unpacklo_epi64(v0_abxxcdxx,
                // v1_efxxghxx); return _mm256_permute4x64_epi64(v_abefcdgh,
                // 0x01 * 0 + 0x04 * 2 + 0x10 * 1 + 0x40 * 3);  // abcdefgh
            } else if constexpr (z_to_z) {
                return detail::concat(_mm512_cvtepi64_epi32(v0),
                                      _mm512_cvtepi64_epi32(v1));
            }
        } else if constexpr (i64_to_i16) {  //{{{2
            if constexpr (x_to_x) {
                // AVX2 is not available (would concat otherwise)
                if constexpr (have_sse4_1) {
                    return _mm_shuffle_epi8(
                        _mm_blend_epi16(v0, _mm_slli_si128(v1, 4), 0x44),
                        _mm_setr_epi8(0, 1, 8, 9, 4, 5, 12, 13, -0x80, -0x80, -0x80,
                                      -0x80, -0x80, -0x80, -0x80, -0x80));
                } else {
                    return builtin_type_t<U, M>{U(v0[0]), U(v0[1]), U(v1[0]), U(v1[1])};
                }
            } else if constexpr (y_to_x) {
                auto a = _mm256_unpacklo_epi16(v0, v1);         // 04.. .... 26.. ....
                auto b = _mm256_unpackhi_epi16(v0, v1);         // 15.. .... 37.. ....
                auto c = _mm256_unpacklo_epi16(a, b);           // 0145 .... 2367 ....
                return _mm_unpacklo_epi32(lo128(c), hi128(c));  // 0123 4567
            } else if constexpr (z_to_y) {
                return detail::concat(_mm512_cvtepi64_epi16(v0),
                                      _mm512_cvtepi64_epi16(v1));
            }
        } else if constexpr (i64_to_i8) {  //{{{2
            if constexpr (x_to_x && have_sse4_1) {
                return _mm_shuffle_epi8(
                    _mm_blend_epi16(v0, _mm_slli_si128(v1, 4), 0x44),
                    _mm_setr_epi8(0, 8, 4, 12, -0x80, -0x80, -0x80, -0x80, -0x80, -0x80,
                                  -0x80, -0x80, -0x80, -0x80, -0x80, -0x80));
            } else if constexpr (x_to_x && have_ssse3) {
                return _mm_unpacklo_epi16(
                    _mm_shuffle_epi8(
                        v0, _mm_setr_epi8(0, 8, -0x80, -0x80, -0x80, -0x80, -0x80, -0x80,
                                          -0x80, -0x80, -0x80, -0x80, -0x80, -0x80, -0x80,
                                          -0x80)),
                    _mm_shuffle_epi8(
                        v1, _mm_setr_epi8(0, 8, -0x80, -0x80, -0x80, -0x80, -0x80, -0x80,
                                          -0x80, -0x80, -0x80, -0x80, -0x80, -0x80, -0x80,
                                          -0x80)));
            } else if constexpr (x_to_x) {
                return builtin_type_t<U, M>{U(v0[0]), U(v0[1]), U(v1[0]), U(v1[1])};
            } else if constexpr (y_to_x) {
                const auto a = _mm256_shuffle_epi8(
                    _mm256_blend_epi32(v0, _mm256_slli_epi64(v1, 32), 0xAA),
                    _mm256_setr_epi8(0, 8, -0x80, -0x80, 4, 12, -0x80, -0x80, -0x80,
                                     -0x80, -0x80, -0x80, -0x80, -0x80, -0x80, -0x80,
                                     -0x80, -0x80, 0, 8, -0x80, -0x80, 4, 12, -0x80,
                                     -0x80, -0x80, -0x80, -0x80, -0x80, -0x80, -0x80));
                return lo128(a) | hi128(a);
            } // z_to_x uses concat fallback
        } else if constexpr (i32_to_i16) {  //{{{2
            if constexpr (x_to_x) {
                // AVX2 is not available (would concat otherwise)
                if constexpr (have_sse4_1) {
                    return _mm_shuffle_epi8(
                        _mm_blend_epi16(v0, _mm_slli_si128(v1, 2), 0xaa),
                        _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14,
                                      15));
                } else if constexpr (have_ssse3) {
                    return _mm_hadd_epi16(to_intrin(v0.d << 16), to_intrin(v1.d << 16));
                    /*
                    return _mm_unpacklo_epi64(
                        _mm_shuffle_epi8(v0, _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 8, 9,
                                                           12, 13, 12, 13, 14, 15)),
                        _mm_shuffle_epi8(v1, _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 8, 9,
                                                           12, 13, 12, 13, 14, 15)));
                                                           */
                } else {
                    auto a = _mm_unpacklo_epi16(v0, v1);  // 04.. 15..
                    auto b = _mm_unpackhi_epi16(v0, v1);  // 26.. 37..
                    auto c = _mm_unpacklo_epi16(a, b);    // 0246 ....
                    auto d = _mm_unpackhi_epi16(a, b);    // 1357 ....
                    return _mm_unpacklo_epi16(c, d);      // 0123 4567
                }
            } else if constexpr (y_to_y) {
                const auto shuf = _mm256_setr_epi8(
                    0, 1, 4, 5, 8, 9, 12, 13, -0x80, -0x80, -0x80, -0x80, -0x80, -0x80,
                    -0x80, -0x80, 0, 1, 4, 5, 8, 9, 12, 13, -0x80, -0x80, -0x80, -0x80,
                    -0x80, -0x80, -0x80, -0x80);
                auto a = _mm256_shuffle_epi8(v0, shuf);
                auto b = _mm256_shuffle_epi8(v1, shuf);
                return fixup_avx_xzyw(_mm256_unpacklo_epi64(a, b));
            } // z_to_z uses concat fallback
        } else if constexpr (i32_to_i8) {  //{{{2
            if constexpr (x_to_x && have_ssse3) {
                const auto shufmask =
                    _mm_setr_epi8(0, 4, 8, 12, -0x80, -0x80, -0x80, -0x80, -0x80, -0x80,
                                  -0x80, -0x80, -0x80, -0x80, -0x80, -0x80);
                return _mm_unpacklo_epi32(_mm_shuffle_epi8(v0, shufmask),
                                          _mm_shuffle_epi8(v1, shufmask));
            } else if constexpr (x_to_x) {
                auto a = _mm_unpacklo_epi8(v0, v1);  // 04.. .... 15.. ....
                auto b = _mm_unpackhi_epi8(v0, v1);  // 26.. .... 37.. ....
                auto c = _mm_unpacklo_epi8(a, b);    // 0246 .... .... ....
                auto d = _mm_unpackhi_epi8(a, b);    // 1357 .... .... ....
                auto e = _mm_unpacklo_epi8(c, d);    // 0123 4567 .... ....
                return e & __m128i{-1, 0};
            } else if constexpr (y_to_x) {
                const auto a = _mm256_shuffle_epi8(
                    _mm256_blend_epi16(v0, _mm256_slli_epi32(v1, 16), 0xAA),
                    _mm256_setr_epi8(0, 4, 8, 12, -0x80, -0x80, -0x80, -0x80, 2, 6, 10,
                                     14, -0x80, -0x80, -0x80, -0x80, -0x80, -0x80, -0x80,
                                     -0x80, 0, 4, 8, 12, -0x80, -0x80, -0x80, -0x80, 2, 6,
                                     10, 14));
                return lo128(a) | hi128(a);
            } // z_to_y uses concat fallback
        } else if constexpr (i16_to_i8) {  //{{{2
            if constexpr (x_to_x && have_ssse3) {
                const auto shuf =
                    reinterpret_cast<__m128i>(sse_const::cvti16_i08_shuffle);
                return _mm_unpacklo_epi64(_mm_shuffle_epi8(v0, shuf),
                                          _mm_shuffle_epi8(v1, shuf));
            } else if constexpr (x_to_x) {
                auto a = _mm_unpacklo_epi8(v0, v1);  // 08.. 19.. 2A.. 3B..
                auto b = _mm_unpackhi_epi8(v0, v1);  // 4C.. 5D.. 6E.. 7F..
                auto c = _mm_unpacklo_epi8(a, b);    // 048C .... 159D ....
                auto d = _mm_unpackhi_epi8(a, b);    // 26AE .... 37BF ....
                auto e = _mm_unpacklo_epi8(c, d);    // 0246 8ACE .... ....
                auto f = _mm_unpackhi_epi8(c, d);    // 1357 9BDF .... ....
                return _mm_unpacklo_epi8(e, f);
            } else if constexpr (y_to_y) {
                return fixup_avx_xzyw(_mm256_shuffle_epi8(
                    (v0.intrin() & _mm256_set1_epi32(0x00ff00ff)) |
                        _mm256_slli_epi16(v1, 8),
                    _mm256_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15,
                                     0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13,
                                     15)));
            } // z_to_z uses concat fallback
        } else if constexpr (i64_to_f32) {  //{{{2
            if constexpr (x_to_x) {
                return make_storage<float>(v0[0], v0[1], v1[0], v1[1]);
            } else if constexpr (y_to_y) {
                static_assert(y_to_y && have_avx2);
                const auto a = _mm256_unpacklo_epi32(v0, v1);   // aeAE cgCG
                const auto b = _mm256_unpackhi_epi32(v0, v1);   // bfBF dhDH
                const auto lo32 = _mm256_unpacklo_epi32(a, b);  // abef cdgh
                const std::conditional_t<is_signed_v<T>, y_i32, y_u32> hi32 =
                    _mm256_unpackhi_epi32(a, b);  // ABEF CDGH
                const auto hi = 0x100000000LL * convert_to<y_f32>(hi32).d;
                const auto mid =
                    0x10000 * _mm256_cvtepi32_ps(_mm256_srli_epi32(lo32, 16));
                const auto lo = _mm256_cvtepi32_ps(_mm256_set1_epi32(0x0000ffffu) & lo32);
                return fixup_avx_xzyw((hi + mid) + lo);
            } else if constexpr (z_to_z && have_avx512dq) {
                return std::is_signed_v<T> ? detail::concat(_mm512_cvtepi64_ps(v0),
                                                            _mm512_cvtepi64_ps(v1))
                                           : detail::concat(_mm512_cvtepu64_ps(v0),
                                                            _mm512_cvtepu64_ps(v1));
            } else if constexpr (z_to_z && std::is_signed_v<T>) {
                const __m512 hi32 = _mm512_cvtepi32_ps(
                    detail::concat(_mm512_cvtepi64_epi32(to_intrin(v0.d >> 32)),
                                   _mm512_cvtepi64_epi32(to_intrin(v1.d >> 32))));
                const __m512i lo32 =
                    detail::concat(_mm512_cvtepi64_epi32(v0), _mm512_cvtepi64_epi32(v1));
                // split low 32-bits, because if hi32 is a small negative number, the
                // 24-bit mantissa may lose important information if any of the high 8
                // bits of lo32 is set, leading to catastrophic cancelation in the FMA
                const __m512 hi16 =
                    _mm512_cvtepu32_ps(_mm512_set1_epi32(0xffff0000u) & lo32);
                const __m512 lo16 =
                    _mm512_cvtepi32_ps(_mm512_set1_epi32(0x0000ffffu) & lo32);
                return (hi32 * 0x100000000LL + hi16) + lo16;
            } else if constexpr (z_to_z && std::is_unsigned_v<T>) {
                return _mm512_cvtepu32_ps(detail::concat(
                           _mm512_cvtepi64_epi32(_mm512_srai_epi64(v0, 32)),
                           _mm512_cvtepi64_epi32(_mm512_srai_epi64(v1, 32)))) *
                           0x100000000LL +
                       _mm512_cvtepu32_ps(detail::concat(_mm512_cvtepi64_epi32(v0),
                                                         _mm512_cvtepi64_epi32(v1)));
            }
        } else if constexpr (f64_to_s32) {  //{{{2
            // use concat fallback
        } else if constexpr (f64_to_u32) {  //{{{2
            if constexpr (x_to_x && have_sse4_1) {
                return builtin_cast<uint>(_mm_unpacklo_epi64(
                           _mm_cvttpd_epi32(_mm_floor_pd(v0) - 0x8000'0000u),
                           _mm_cvttpd_epi32(_mm_floor_pd(v1) - 0x8000'0000u))) ^
                       0x8000'0000u;
                // without SSE4.1 just use the scalar fallback, it's only four values
            } else if constexpr (y_to_y) {
                return builtin_cast<uint>(detail::concat(
                           _mm256_cvttpd_epi32(_mm256_floor_pd(v0) - 0x8000'0000u),
                           _mm256_cvttpd_epi32(_mm256_floor_pd(v1) - 0x8000'0000u))) ^
                       0x8000'0000u;
            } // z_to_z uses fallback
        } else if constexpr (f64_to_ibw) {  //{{{2
            // one-arg f64_to_ibw goes via Storage<int, ?>. The fallback would go via two
            // independet conversions to Storage<To> and subsequent interleaving. This is
            // better, because f64->i32 allows to combine v0 and v1 into one register:
            //if constexpr (z_to_x || y_to_x) {
            return convert_to<To>(convert_to<Storage<int, N * 2>>(v0, v1));
            //}
        } else if constexpr (f32_to_ibw) {  //{{{2
            return convert_to<To>(convert_to<Storage<int, N>>(v0),
                                  convert_to<Storage<int, N>>(v1));
        //}}}
        }

        // fallback: {{{2
        if constexpr (sizeof(To) >= 32) {
            // if To is ymm or zmm, then Storage<U, M / 2> is xmm or ymm
            return detail::concat(convert_to<Storage<U, M / 2>>(v0),
                                  convert_to<Storage<U, M / 2>>(v1));
        } else if constexpr (sizeof(To) == 16) {
            const auto lo = convert_to<To>(v0);
            const auto hi = convert_to<To>(v1);
            if constexpr (sizeof(U) * N == 8) {
                if constexpr (is_floating_point_v<U>) {
                    return to_storage(_mm_unpacklo_pd(auto_cast(lo), auto_cast(hi)));
                } else {
                    return _mm_unpacklo_epi64(lo, hi);
                }
            } else if constexpr (sizeof(U) * N == 4) {
                if constexpr (is_floating_point_v<U>) {
                    return to_storage(_mm_unpacklo_ps(auto_cast(lo), auto_cast(hi)));
                } else {
                    return _mm_unpacklo_epi32(lo, hi);
                }
            } else if constexpr (sizeof(U) * N == 2) {
                return _mm_unpacklo_epi16(lo, hi);
            } else {
                assert_unreachable<T>();
            }
        } else {
            return convert_builtin<To>(v0.d, v1.d, std::make_index_sequence<N>(),
                                       std::make_index_sequence<M - 2 * N>());
        }  //}}}
    }
}//}}}1
// 4-arg convert_to {{{1
template <class To, class T, size_t N>
Vc_INTRINSIC To convert_to(Storage<T, N> v0, Storage<T, N> v1, Storage<T, N> v2,
                           Storage<T, N> v3)
{
    using U = typename To::value_type;
    constexpr size_t M = To::width;

    using std::is_integral_v;
    using std::is_signed_v;
    using std::is_unsigned_v;
    using std::is_floating_point_v;

    static_assert(
        4 * N <= M,
        "v2/v3 would be discarded; use the two/one-argument convert_to overload instead");

    // [xyz]_to_[xyz] {{{2
    constexpr bool x_to_x = sizeof(v0) == 16 && sizeof(To) == 16;
    constexpr bool x_to_y = sizeof(v0) == 16 && sizeof(To) == 32;
    constexpr bool x_to_z = sizeof(v0) == 16 && sizeof(To) == 64;
    constexpr bool y_to_x = sizeof(v0) == 32 && sizeof(To) == 16;
    constexpr bool y_to_y = sizeof(v0) == 32 && sizeof(To) == 32;
    constexpr bool y_to_z = sizeof(v0) == 32 && sizeof(To) == 64;
    constexpr bool z_to_x = sizeof(v0) == 64 && sizeof(To) == 16;
    constexpr bool z_to_y = sizeof(v0) == 64 && sizeof(To) == 32;
    constexpr bool z_to_z = sizeof(v0) == 64 && sizeof(To) == 64;

    // iX_to_iX {{{2
    constexpr bool i_to_i = std::is_integral_v<U> && std::is_integral_v<T>;
    constexpr bool i8_to_i16  = i_to_i && sizeof(T) == 1 && sizeof(U) == 2;
    constexpr bool i8_to_i32  = i_to_i && sizeof(T) == 1 && sizeof(U) == 4;
    constexpr bool i8_to_i64  = i_to_i && sizeof(T) == 1 && sizeof(U) == 8;
    constexpr bool i16_to_i8  = i_to_i && sizeof(T) == 2 && sizeof(U) == 1;
    constexpr bool i16_to_i32 = i_to_i && sizeof(T) == 2 && sizeof(U) == 4;
    constexpr bool i16_to_i64 = i_to_i && sizeof(T) == 2 && sizeof(U) == 8;
    constexpr bool i32_to_i8  = i_to_i && sizeof(T) == 4 && sizeof(U) == 1;
    constexpr bool i32_to_i16 = i_to_i && sizeof(T) == 4 && sizeof(U) == 2;
    constexpr bool i32_to_i64 = i_to_i && sizeof(T) == 4 && sizeof(U) == 8;
    constexpr bool i64_to_i8  = i_to_i && sizeof(T) == 8 && sizeof(U) == 1;
    constexpr bool i64_to_i16 = i_to_i && sizeof(T) == 8 && sizeof(U) == 2;
    constexpr bool i64_to_i32 = i_to_i && sizeof(T) == 8 && sizeof(U) == 4;

    // [fsu]X_to_[fsu]X {{{2
    // ibw = integral && byte or word, i.e. char and short with any signedness
    constexpr bool i64_to_f32 = is_integral_v<T> &&                     sizeof(T) == 8 && is_floating_point_v<U> && sizeof(U) == 4;
    constexpr bool s32_to_f32 = is_integral_v<T> &&   is_signed_v<T> && sizeof(T) == 4 && is_floating_point_v<U> && sizeof(U) == 4;
    constexpr bool s16_to_f32 = is_integral_v<T> &&   is_signed_v<T> && sizeof(T) == 2 && is_floating_point_v<U> && sizeof(U) == 4;
    constexpr bool  s8_to_f32 = is_integral_v<T> &&   is_signed_v<T> && sizeof(T) == 1 && is_floating_point_v<U> && sizeof(U) == 4;
    constexpr bool u32_to_f32 = is_integral_v<T> && is_unsigned_v<T> && sizeof(T) == 4 && is_floating_point_v<U> && sizeof(U) == 4;
    constexpr bool u16_to_f32 = is_integral_v<T> && is_unsigned_v<T> && sizeof(T) == 2 && is_floating_point_v<U> && sizeof(U) == 4;
    constexpr bool  u8_to_f32 = is_integral_v<T> && is_unsigned_v<T> && sizeof(T) == 1 && is_floating_point_v<U> && sizeof(U) == 4;
    constexpr bool s64_to_f64 = is_integral_v<T> &&   is_signed_v<T> && sizeof(T) == 8 && is_floating_point_v<U> && sizeof(U) == 8;
    constexpr bool s32_to_f64 = is_integral_v<T> &&   is_signed_v<T> && sizeof(T) == 4 && is_floating_point_v<U> && sizeof(U) == 8;
    constexpr bool s16_to_f64 = is_integral_v<T> &&   is_signed_v<T> && sizeof(T) == 2 && is_floating_point_v<U> && sizeof(U) == 8;
    constexpr bool  s8_to_f64 = is_integral_v<T> &&   is_signed_v<T> && sizeof(T) == 1 && is_floating_point_v<U> && sizeof(U) == 8;
    constexpr bool u64_to_f64 = is_integral_v<T> && is_unsigned_v<T> && sizeof(T) == 8 && is_floating_point_v<U> && sizeof(U) == 8;
    constexpr bool u32_to_f64 = is_integral_v<T> && is_unsigned_v<T> && sizeof(T) == 4 && is_floating_point_v<U> && sizeof(U) == 8;
    constexpr bool u16_to_f64 = is_integral_v<T> && is_unsigned_v<T> && sizeof(T) == 2 && is_floating_point_v<U> && sizeof(U) == 8;
    constexpr bool  u8_to_f64 = is_integral_v<T> && is_unsigned_v<T> && sizeof(T) == 1 && is_floating_point_v<U> && sizeof(U) == 8;
    constexpr bool f32_to_s64 = is_integral_v<U> &&   is_signed_v<U> && sizeof(U) == 8 && is_floating_point_v<T> && sizeof(T) == 4;
    constexpr bool f32_to_s32 = is_integral_v<U> &&   is_signed_v<U> && sizeof(U) == 4 && is_floating_point_v<T> && sizeof(T) == 4;
    constexpr bool f32_to_u64 = is_integral_v<U> && is_unsigned_v<U> && sizeof(U) == 8 && is_floating_point_v<T> && sizeof(T) == 4;
    constexpr bool f32_to_u32 = is_integral_v<U> && is_unsigned_v<U> && sizeof(U) == 4 && is_floating_point_v<T> && sizeof(T) == 4;
    constexpr bool f64_to_s64 = is_integral_v<U> &&   is_signed_v<U> && sizeof(U) == 8 && is_floating_point_v<T> && sizeof(T) == 8;
    constexpr bool f64_to_s32 = is_integral_v<U> &&   is_signed_v<U> && sizeof(U) == 4 && is_floating_point_v<T> && sizeof(T) == 8;
    constexpr bool f64_to_u64 = is_integral_v<U> && is_unsigned_v<U> && sizeof(U) == 8 && is_floating_point_v<T> && sizeof(T) == 8;
    constexpr bool f64_to_u32 = is_integral_v<U> && is_unsigned_v<U> && sizeof(U) == 4 && is_floating_point_v<T> && sizeof(T) == 8;
    constexpr bool f32_to_ibw = is_integral_v<U> && sizeof(U) <= 2 && is_floating_point_v<T> && sizeof(T) == 4;
    constexpr bool f64_to_ibw = is_integral_v<U> && sizeof(U) <= 2 && is_floating_point_v<T> && sizeof(T) == 8;
    constexpr bool f32_to_f64 = is_floating_point_v<T> && sizeof(T) == 4 && is_floating_point_v<U> && sizeof(U) == 8;
    constexpr bool f64_to_f32 = is_floating_point_v<T> && sizeof(T) == 8 && is_floating_point_v<U> && sizeof(U) == 4;

    if constexpr (i_to_i && y_to_x && !have_avx2) {  //{{{2
        // <double, 4>, <double, 4>, <double, 4>, <double, 4> => <char, 16>
        return convert_to<To>(lo128(v0), hi128(v0), lo128(v1), hi128(v1), lo128(v2),
                              hi128(v2), lo128(v3), hi128(v3));
    } else if constexpr (i_to_i) {  // assert ISA {{{2
        static_assert(x_to_x || have_avx2,
                      "integral conversions with ymm registers require AVX2");
        static_assert(have_avx512bw || ((sizeof(T) >= 4 || sizeof(v0) < 64) &&
                                        (sizeof(U) >= 4 || sizeof(To) < 64)),
                      "8/16-bit integers in zmm registers require AVX512BW");
        static_assert((sizeof(v0) < 64 && sizeof(To) < 64) || have_avx512f,
                      "integral conversions with ymm registers require AVX2");
    }
    // concat => use 2-arg convert_to {{{2
    if constexpr ((sizeof(v0) == 16 && have_avx2) ||
                  (sizeof(v0) == 16 && have_avx && std::is_floating_point_v<T>) ||
                  (sizeof(v0) == 32 && have_avx512f)) {
        // The ISA can handle wider input registers, so concat and use two-arg
        // implementation. This reduces code duplication considerably.
        return convert_to<To>(detail::concat(v0, v1), detail::concat(v2, v3));
    } else {  //{{{2
        // conversion using bit reinterpretation (or no conversion at all) should all go
        // through the concat branch above:
        static_assert(!(std::is_floating_point_v<T> == std::is_floating_point_v<U> &&
                        sizeof(T) == sizeof(U)));
        if constexpr (4 * N < M && sizeof(To) > 16) {  // handle all zero extension{{{2
            constexpr size_t Min = 16 / sizeof(U);
            return x86::zeroExtend(
                convert_to<Storage<U, (Min > 4 * N) ? Min : 4 * N>>(v0, v1, v2, v3)
                    .intrin());
        } else if constexpr (i64_to_i16) {  //{{{2
            if constexpr (x_to_x && have_sse4_1) {
                return _mm_shuffle_epi8(
                    _mm_blend_epi16(_mm_blend_epi16(v0, _mm_slli_si128(v1, 2), 0x22),
                                    _mm_blend_epi16(_mm_slli_si128(v2, 4),
                                                    _mm_slli_si128(v3, 6), 0x88),
                                    0xcc),
                    _mm_setr_epi8(0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15));
            } else if constexpr (y_to_y) {
                return _mm256_shuffle_epi8(
                    fixup_avx_xzyw(_mm256_blend_epi16(
                        auto_cast(_mm256_shuffle_ps(to_m256(v0), to_m256(v2),
                                                    0x88)),  // 0.1. 8.9. 2.3. A.B.
                        to_intrin(builtin_cast<int>(
                                      _mm256_shuffle_ps(to_m256(v1), to_m256(v3), 0x88))
                                  << 16),  // .4.5 .C.D .6.7 .E.F
                        0xaa)              // 0415 8C9D 2637 AEBF
                                   ),      // 0415 2637 8C9D AEBF
                    _mm256_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15,
                                     0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14,
                                     15));
                /*
                auto a = _mm256_unpacklo_epi16(v0, v1);  // 04.. .... 26.. ....
                auto b = _mm256_unpackhi_epi16(v0, v1);  // 15.. .... 37.. ....
                auto c = _mm256_unpacklo_epi16(v2, v3);  // 8C.. .... AE.. ....
                auto d = _mm256_unpackhi_epi16(v2, v3);  // 9D.. .... BF.. ....
                auto e = _mm256_unpacklo_epi16(a, b);    // 0145 .... 2367 ....
                auto f = _mm256_unpacklo_epi16(c, d);    // 89CD .... ABEF ....
                auto g = _mm256_unpacklo_epi64(e, f);    // 0145 89CD 2367 ABEF
                return detail::concat(
                    _mm_unpacklo_epi32(lo128(g), hi128(g)),
                    _mm_unpackhi_epi32(lo128(g), hi128(g)));  // 0123 4567 89AB CDEF
                    */
            }  // else use fallback
        } else if constexpr (i64_to_i8) {  //{{{2
            if constexpr (x_to_x) {
                // TODO: use fallback for now
            } else if constexpr (y_to_x) {
                auto a =
                    _mm256_srli_epi32(_mm256_slli_epi32(v0, 24), 24) |
                    _mm256_srli_epi32(_mm256_slli_epi32(v1, 24), 16) |
                    _mm256_srli_epi32(_mm256_slli_epi32(v2, 24), 8) |
                    _mm256_slli_epi32(v3, 24);  // 048C .... 159D .... 26AE .... 37BF ....
                /*return _mm_shuffle_epi8(
                    _mm_blend_epi32(lo128(a) << 32, hi128(a), 0x5),
                    _mm_setr_epi8(4, 12, 0, 8, 5, 13, 1, 9, 6, 14, 2, 10, 7, 15, 3, 11));*/
                auto b = _mm256_unpackhi_epi64(a, a);  // 159D .... 159D .... 37BF .... 37BF ....
                auto c = _mm256_unpacklo_epi8(a, b);  // 0145 89CD .... .... 2367 ABEF .... ....
                return _mm_unpacklo_epi16(lo128(c), hi128(c));  // 0123 4567 89AB CDEF
            }
        } else if constexpr (i32_to_i8) {  //{{{2
            if constexpr (x_to_x) {
                if constexpr (have_ssse3) {
                    const auto x0 =  builtin_cast<uint>(v0.d) & 0xff;
                    const auto x1 = (builtin_cast<uint>(v1.d) & 0xff) << 8;
                    const auto x2 = (builtin_cast<uint>(v2.d) & 0xff) << 16;
                    const auto x3 =  builtin_cast<uint>(v3.d)         << 24;
                    return _mm_shuffle_epi8(to_intrin(x0 | x1 | x2 | x3),
                                            _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6,
                                                          10, 14, 3, 7, 11, 15));
                } else {
                    auto a = _mm_unpacklo_epi8(v0, v2);  // 08.. .... 19.. ....
                    auto b = _mm_unpackhi_epi8(v0, v2);  // 2A.. .... 3B.. ....
                    auto c = _mm_unpacklo_epi8(v1, v3);  // 4C.. .... 5D.. ....
                    auto d = _mm_unpackhi_epi8(v1, v3);  // 6E.. .... 7F.. ....
                    auto e = _mm_unpacklo_epi8(a, c);    // 048C .... .... ....
                    auto f = _mm_unpackhi_epi8(a, c);    // 159D .... .... ....
                    auto g = _mm_unpacklo_epi8(b, d);    // 26AE .... .... ....
                    auto h = _mm_unpackhi_epi8(b, d);    // 37BF .... .... ....
                    return _mm_unpacklo_epi8(
                        _mm_unpacklo_epi8(e, g),  // 0246 8ACE .... ....
                        _mm_unpacklo_epi8(f, h)   // 1357 9BDF .... ....
                    );                            // 0123 4567 89AB CDEF
                }
            } else if constexpr (y_to_y) {
                const auto a = _mm256_shuffle_epi8(
                    to_intrin((builtin_cast<ushort>(_mm256_blend_epi16(
                                   v0, _mm256_slli_epi32(v1, 16), 0xAA)) &
                               0xff) |
                              (builtin_cast<ushort>(_mm256_blend_epi16(
                                   v2, _mm256_slli_epi32(v3, 16), 0xAA))
                               << 8)),
                    _mm256_setr_epi8(0, 4, 8, 12, 2, 6, 10, 14, 1, 5, 9, 13, 3, 7, 11, 15,
                                     0, 4, 8, 12, 2, 6, 10, 14, 1, 5, 9, 13, 3, 7, 11, 15));
                return _mm256_permutevar8x32_epi32(
                    a, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
            }
        } else if constexpr (i64_to_f32) {  //{{{2
            // this branch is only relevant with AVX and w/o AVX2 (i.e. no ymm integers)
            if constexpr (x_to_y) {
                return make_storage<float>(v0[0], v0[1], v1[0], v1[1], v2[0], v2[1], v3[0], v3[1]);

                const auto a = _mm_unpacklo_epi32(v0, v1);   // acAC
                const auto b = _mm_unpackhi_epi32(v0, v1);   // bdBD
                const auto c = _mm_unpacklo_epi32(v2, v3);   // egEG
                const auto d = _mm_unpackhi_epi32(v2, v3);   // fhFH
                const auto lo32a = _mm_unpacklo_epi32(a, b);  // abcd
                const auto lo32b = _mm_unpacklo_epi32(c, d);  // efgh
                const std::conditional_t<is_signed_v<T>, y_i32, y_u32> hi32 = concat(
                    _mm_unpackhi_epi32(a, b), _mm_unpackhi_epi32(c, d));  // ABCD EFGH
                const auto hi = 0x100000000LL * convert_to<y_f32>(hi32).d;
                const auto mid =
                    0x10000 * _mm256_cvtepi32_ps(concat(_mm_srli_epi32(lo32a, 16),
                                                        _mm_srli_epi32(lo32b, 16)));
                const auto lo =
                    _mm256_cvtepi32_ps(concat(_mm_set1_epi32(0x0000ffffu) & lo32a,
                                              _mm_set1_epi32(0x0000ffffu) & lo32b));
                return (hi + mid) + lo;
            }
        } else if constexpr (f64_to_ibw) {  //{{{2
            return convert_to<To>(convert_to<Storage<int, N * 2>>(v0, v1),
                                  convert_to<Storage<int, N * 2>>(v2, v3));
        } else if constexpr (f32_to_ibw) {  //{{{2
            return convert_to<To>(
                convert_to<Storage<int, N>>(v0), convert_to<Storage<int, N>>(v1),
                convert_to<Storage<int, N>>(v2), convert_to<Storage<int, N>>(v3));
        }  //}}}

        // fallback: {{{2
        if constexpr (sizeof(To) >= 32) {
            // if To is ymm or zmm, then Storage<U, M / 2> is xmm or ymm
            return detail::concat(convert_to<Storage<U, M / 2>>(v0, v1),
                                  convert_to<Storage<U, M / 2>>(v2, v3));
        } else if constexpr (sizeof(To) == 16) {
            const auto lo = convert_to<To>(v0, v1);
            const auto hi = convert_to<To>(v2, v3);
            if constexpr (sizeof(U) * N * 2 == 8) {
                if constexpr (is_floating_point_v<U>) {
                    return to_storage(_mm_unpacklo_pd(auto_cast(lo), auto_cast(hi)));
                } else {
                    return _mm_unpacklo_epi64(lo, hi);
                }
            } else if constexpr (sizeof(U) * N * 2 == 4) {
                if constexpr (is_floating_point_v<U>) {
                    return to_storage(_mm_unpacklo_ps(auto_cast(lo), auto_cast(hi)));
                } else {
                    return _mm_unpacklo_epi32(lo, hi);
                }
            } else {
                assert_unreachable<T>();
            }
        } else {
            return convert_builtin<To>(v0.d, v1.d, v2.d, v3.d,
                                       std::make_index_sequence<N>(),
                                       std::make_index_sequence<M - 4 * N>());
        }  //}}}2
    }
}//}}}
// 8-arg convert_to {{{1
template <class To, class T, size_t N>
Vc_INTRINSIC To convert_to(Storage<T, N> v0, Storage<T, N> v1, Storage<T, N> v2,
                           Storage<T, N> v3, Storage<T, N> v4, Storage<T, N> v5,
                           Storage<T, N> v6, Storage<T, N> v7)
{
    using U = typename To::value_type;
    constexpr size_t M = To::width;

    using std::is_integral_v;
    using std::is_signed_v;
    using std::is_unsigned_v;
    using std::is_floating_point_v;

    static_assert(8 * N <= M, "v4-v7 would be discarded; use the four/two/one-argument "
                              "convert_to overload instead");

    // [xyz]_to_[xyz] {{{2
    constexpr bool x_to_x = sizeof(v0) == 16 && sizeof(To) == 16;
    constexpr bool x_to_y = sizeof(v0) == 16 && sizeof(To) == 32;
    constexpr bool x_to_z = sizeof(v0) == 16 && sizeof(To) == 64;
    constexpr bool y_to_x = sizeof(v0) == 32 && sizeof(To) == 16;
    constexpr bool y_to_y = sizeof(v0) == 32 && sizeof(To) == 32;
    constexpr bool y_to_z = sizeof(v0) == 32 && sizeof(To) == 64;
    constexpr bool z_to_x = sizeof(v0) == 64 && sizeof(To) == 16;
    constexpr bool z_to_y = sizeof(v0) == 64 && sizeof(To) == 32;
    constexpr bool z_to_z = sizeof(v0) == 64 && sizeof(To) == 64;

    // [if]X_to_i8 {{{2
    constexpr bool i_to_i = std::is_integral_v<U> && std::is_integral_v<T>;
    constexpr bool i64_to_i8 = i_to_i && sizeof(T) == 8 && sizeof(U) == 1;
    constexpr bool f64_to_i8 = is_integral_v<U> && sizeof(U) == 1 && is_floating_point_v<T> && sizeof(T) == 8;

    if constexpr (i_to_i) {  // assert ISA {{{2
        static_assert(x_to_x || have_avx2,
                      "integral conversions with ymm registers require AVX2");
        static_assert(have_avx512bw || ((sizeof(T) >= 4 || sizeof(v0) < 64) &&
                                        (sizeof(U) >= 4 || sizeof(To) < 64)),
                      "8/16-bit integers in zmm registers require AVX512BW");
        static_assert((sizeof(v0) < 64 && sizeof(To) < 64) || have_avx512f,
                      "integral conversions with ymm registers require AVX2");
    }
    // concat => use 4-arg convert_to {{{2
    if constexpr ((sizeof(v0) == 16 && have_avx2) ||
                  (sizeof(v0) == 16 && have_avx && std::is_floating_point_v<T>) ||
                  (sizeof(v0) == 32 && have_avx512f)) {
        // The ISA can handle wider input registers, so concat and use two-arg
        // implementation. This reduces code duplication considerably.
        return convert_to<To>(detail::concat(v0, v1), detail::concat(v2, v3),
                              detail::concat(v4, v5), detail::concat(v6, v7));
    } else {  //{{{2
        // conversion using bit reinterpretation (or no conversion at all) should all go
        // through the concat branch above:
        static_assert(!(std::is_floating_point_v<T> == std::is_floating_point_v<U> &&
                        sizeof(T) == sizeof(U)));
        static_assert(!(8 * N < M && sizeof(To) > 16),
                      "zero extension should be impossible");
        if constexpr (i64_to_i8) {  //{{{2
            if constexpr (x_to_x && have_ssse3) {
                // unsure whether this is better than the variant below
                return _mm_shuffle_epi8(
                    to_intrin(((( v0.intrin() & 0xff       ) | ((v1.intrin() & 0xff) <<  8)) |
                               (((v2.intrin() & 0xff) << 16) | ((v3.intrin() & 0xff) << 24))) |
                              ((((v4.intrin() & 0xff) << 32) | ((v5.intrin() & 0xff) << 40)) |
                               (((v6.intrin() & 0xff) << 48) | ( v7.intrin() << 56)))),
                    _mm_setr_epi8(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15));
            } else if constexpr (x_to_x) {
                const auto a = _mm_unpacklo_epi8(v0, v1); // ac
                const auto b = _mm_unpackhi_epi8(v0, v1); // bd
                const auto c = _mm_unpacklo_epi8(v2, v3); // eg
                const auto d = _mm_unpackhi_epi8(v2, v3); // fh
                const auto e = _mm_unpacklo_epi8(v4, v5); // ik
                const auto f = _mm_unpackhi_epi8(v4, v5); // jl
                const auto g = _mm_unpacklo_epi8(v6, v7); // mo
                const auto h = _mm_unpackhi_epi8(v6, v7); // np
                return _mm_unpacklo_epi64(
                    _mm_unpacklo_epi32(_mm_unpacklo_epi8(a, b),   // abcd
                                       _mm_unpacklo_epi8(c, d)),  // efgh
                    _mm_unpacklo_epi32(_mm_unpacklo_epi8(e, f),   // ijkl
                                       _mm_unpacklo_epi8(g, h))   // mnop
                );
            } else if constexpr (y_to_y) {
                auto a =  // 048C GKOS 159D HLPT 26AE IMQU 37BF JNRV
                    to_intrin(((( v0.intrin() & 0xff       ) | ((v1.intrin() & 0xff) <<  8)) |
                               (((v2.intrin() & 0xff) << 16) | ((v3.intrin() & 0xff) << 24))) |
                              ((((v4.intrin() & 0xff) << 32) | ((v5.intrin() & 0xff) << 40)) |
                               (((v6.intrin() & 0xff) << 48) | ((v7.intrin() << 56)))));
                /*
                auto b = _mm256_unpackhi_epi64(a, a);  // 159D HLPT 159D HLPT 37BF JNRV 37BF JNRV
                auto c = _mm256_unpacklo_epi8(a, b);  // 0145 89CD GHKL OPST 2367 ABEF IJMN QRUV
                auto d = fixup_avx_xzyw(c); // 0145 89CD 2367 ABEF GHKL OPST IJMN QRUV
                return _mm256_shuffle_epi8(
                    d, _mm256_setr_epi8(0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14,
                                        15, 0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7,
                                        14, 15));
                                        */
                auto b = _mm256_shuffle_epi8( // 0145 89CD GHKL OPST 2367 ABEF IJMN QRUV
                    a, _mm256_setr_epi8(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15,
                                        0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15));
                auto c = fixup_avx_xzyw(b); // 0145 89CD 2367 ABEF GHKL OPST IJMN QRUV
                return _mm256_shuffle_epi8(
                    c, _mm256_setr_epi8(0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14,
                                        15, 0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7,
                                        14, 15));
            } else if constexpr(z_to_z) {
                return detail::concat(convert_to<Storage<U, M / 2>>(v0, v1, v2, v3),
                                      convert_to<Storage<U, M / 2>>(v4, v5, v6, v7));
            }
        } else if constexpr (f64_to_i8) {  //{{{2
            return convert_to<To>(convert_to<Storage<int, N * 2>>(v0, v1),
                                  convert_to<Storage<int, N * 2>>(v2, v3),
                                  convert_to<Storage<int, N * 2>>(v4, v5),
                                  convert_to<Storage<int, N * 2>>(v6, v7));
        } else { // unreachable {{{2
            assert_unreachable<T>();
        }  //}}}

        // fallback: {{{2
        if constexpr (sizeof(To) >= 32) {
            // if To is ymm or zmm, then Storage<U, M / 2> is xmm or ymm
            return detail::concat(convert_to<Storage<U, M / 2>>(v0, v1, v2, v3),
                                  convert_to<Storage<U, M / 2>>(v4, v5, v6, v7));
        } else if constexpr (sizeof(To) == 16) {
            const auto lo = convert_to<To>(v0, v1, v2, v3);
            const auto hi = convert_to<To>(v4, v5, v6, v7);
            static_assert(sizeof(U) == 1 && N == 2);
            return _mm_unpacklo_epi64(lo, hi);
        } else {
            assert_unreachable<T>();
            /*return convert_builtin<To>(v0.d, v1.d, v2.d, v3.d, v4.d, v5.d, v6.d, v7.d,
                                       std::make_index_sequence<N>(),
                                       std::make_index_sequence<M - 8 * N>());*/
        }  //}}}2
    }
}//}}}
#pragma GCC diagnostic pop
#endif  // Vc_WORKAROUND_PR85048

// convert from scalars{{{1
template <typename To, typename... From>
[[deprecated("use make_storage instead")]]
Vc_INTRINSIC To Vc_VDECL convert_to(vectorizable<From>... scalars)
{
    return x86::set(static_cast<typename To::value_type>(scalars)...);
}

// convert function{{{1
template <class To, class From> Vc_INTRINSIC To Vc_VDECL convert(From v)
{
    if constexpr (detail::is_builtin_vector_v<From>) {
        using Trait = detail::builtin_traits<From>;
        return convert<To>(Storage<typename Trait::value_type, Trait::width>(v));
    } else if constexpr (detail::is_builtin_vector_v<To>) {
        using Trait = detail::builtin_traits<To>;
        return convert<Storage<typename Trait::value_type, Trait::width>>(v).d;
    } else {
#ifdef Vc_WORKAROUND_PR85048
        return convert_to<To>(v);
#else
        if constexpr (From::width >= To::width) {
            return convert_builtin<To>(v.d, std::make_index_sequence<To::width>());
        } else {
            return convert_builtin_z<To>(
                v.d, std::make_index_sequence<From::width>(),
                std::make_index_sequence<To::width - From::width>());
        }
#endif
    }
}

template <class To, class From> Vc_INTRINSIC To Vc_VDECL convert(From v0, From v1)
{
    if constexpr (detail::is_builtin_vector_v<From>) {
        using Trait = detail::builtin_traits<From>;
        using S = Storage<typename Trait::value_type, Trait::width>;
        return convert<To>(S(v0), S(v1));
    } else if constexpr (std::is_arithmetic_v<From>) {
        using T = typename To::value_type;
        return make_storage<T>(v0, v1);
    } else {
        static_assert(To::width >= 2 * From::width,
                      "convert(v0, v1) requires the input to fit into the output");
#ifdef Vc_WORKAROUND_PR85048
        return convert_to<To>(v0, v1);
#else
        return convert_builtin<To>(
            v0.d, v1.d, std::make_index_sequence<From::width>(),
            std::make_index_sequence<To::width - 2 * From::width>());
#endif
    }
}

template <class To, class From>
Vc_INTRINSIC To Vc_VDECL convert(From v0, From v1, From v2, From v3)
{
    if constexpr (detail::is_builtin_vector_v<From>) {
        using Trait = detail::builtin_traits<From>;
        using S = Storage<typename Trait::value_type, Trait::width>;
        return convert<To>(S(v0), S(v1), S(v2), S(v3));
    } else if constexpr (std::is_arithmetic_v<From>) {
        using T = typename To::value_type;
        return make_storage<T>(v0, v1, v2, v3);
    } else {
        static_assert(
            To::width >= 4 * From::width,
            "convert(v0, v1, v2, v3) requires the input to fit into the output");
#ifdef Vc_WORKAROUND_PR85048
        return convert_to<To>(v0, v1, v2, v3);
#else
        return convert_builtin<To>(
            v0.d, v1.d, v2.d, v3.d, std::make_index_sequence<From::width>(),
            std::make_index_sequence<To::width - 4 * From::width>());
#endif
    }
}

template <class To, class From>
Vc_INTRINSIC To Vc_VDECL convert(From v0, From v1, From v2, From v3, From v4, From v5, From v6,
                        From v7)
{
    if constexpr (detail::is_builtin_vector_v<From>) {
        using Trait = detail::builtin_traits<From>;
        using S = Storage<typename Trait::value_type, Trait::width>;
        return convert<To>(S(v0), S(v1), S(v2), S(v3), S(v4), S(v5), S(v6), S(v7));
    } else if constexpr (std::is_arithmetic_v<From>) {
        using T = typename To::value_type;
        return make_storage<T>(v0, v1, v2, v3, v4, v5, v6, v7);
    } else {
        static_assert(To::width >= 8 * From::width,
                      "convert(v0, v1, v2, v3, v4, v5, v6, v7) "
                      "requires the input to fit into the output");
#ifdef Vc_WORKAROUND_PR85048
        return convert_to<To>(v0, v1, v2, v3, v4, v5, v6, v7);
#else
        return convert_builtin<To>(
            v0.d, v1.d, v2.d, v3.d, v4.d, v5.d, v6.d, v7.d,
            std::make_index_sequence<From::width>(),
            std::make_index_sequence<To::width - 8 * From::width>());
#endif
    }
}

// convert_all function{{{1
template <typename To, typename From> Vc_INTRINSIC auto convert_all(From v)
{
    static_assert(detail::is_builtin_vector_v<To>);
    if constexpr (detail::is_builtin_vector_v<From>) {
        using Trait = detail::builtin_traits<From>;
        using S = Storage<typename Trait::value_type, Trait::width>;
        return convert_all<To>(S(v));
    } else if constexpr (From::width > builtin_traits<To>::width) {
        constexpr size_t N = From::width / builtin_traits<To>::width;
        return generate_from_n_evaluations<N, std::array<To, N>>([&](auto i) {
            auto part = x86::extract_part<decltype(i)::value, N>(v);
            return convert<To>(part);
        });
    } else {
        return convert<To>(v);
    }
}

// }}}1
}}  // namespace detail::x86
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_SIMD_X86_CONVERT_H_

// vim: foldmethod=marker
