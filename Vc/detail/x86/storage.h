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

#ifndef VC_SIMD_X86_STORAGE_H_
#define VC_SIMD_X86_STORAGE_H_

#ifndef VC_SIMD_STORAGE_H_
#error "Do not include detail/x86/storage.h directly. Include detail/storage.h instead."
#endif
#include "intrinsics.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
namespace x86
{
// lo/hi/extract128 {{{
template <typename T, size_t N>
constexpr Vc_INTRINSIC storage16_t<T> lo128(Storage<T, N> x)
{
    return detail::x86::lo128(x.d);
}
template <typename T, size_t N>
constexpr Vc_INTRINSIC storage16_t<T> hi128(Storage<T, N> x)
{
    return detail::x86::hi128(x.d);
}

template <int offset, typename T, size_t N>
constexpr Vc_INTRINSIC storage16_t<T> extract128(Storage<T, N> x)
{
    return detail::x86::extract128<offset>(x.d);
}

//}}}

// lo/hi256 {{{
template <typename T, size_t N>
constexpr Vc_INTRINSIC storage32_t<T> lo256(Storage<T, N> x)
{
    return detail::x86::lo256(x.d);
}
template <typename T, size_t N>
constexpr Vc_INTRINSIC storage32_t<T> hi256(Storage<T, N> x)
{
    return detail::x86::hi256(x.d);
}
//}}}

// extract_part {{{1
// identity {{{2
template <class T>
constexpr Vc_INTRINSIC const storage16_t<T>& extract_part_impl(std::true_type,
                                                               size_constant<0>,
                                                               size_constant<1>,
                                                               const storage16_t<T>& x)
{
    return x;
}

// by 2 and by 4 splits {{{2
template <class T, size_t N, size_t Index, size_t Total>
constexpr Vc_INTRINSIC Storage<T, N / Total> extract_part_impl(std::true_type,
                                                               size_constant<Index>,
                                                               size_constant<Total>,
                                                               Storage<T, N> x)
{
    return detail::extract<Index, Total>(x.d);
}

// partial SSE (shifts) {{{2
template <class T, size_t Index, size_t Total, size_t N>
Vc_INTRINSIC Storage<T, 16 / sizeof(T)> extract_part_impl(std::false_type,
                                                                   size_constant<Index>,
                                                                   size_constant<Total>,
                                                                   Storage<T, N> x)
{
    constexpr int split = sizeof(x) / 16;
    constexpr int shift = (sizeof(x) / Total * Index) % 16;
    return x86::shift_right<shift>(
        extract_part_impl<T>(std::true_type(), size_constant<Index * split / Total>(),
                             size_constant<split>(), x));
}

// public interface {{{2
template <class T> constexpr T constexpr_max(T a, T b) { return a > b ? a : b; }

template <size_t Index, size_t Total, class T, size_t N>
Vc_INTRINSIC Vc_CONST Storage<T, constexpr_max(16 / sizeof(T), N / Total)> extract_part(
    Storage<T, N> x)
{
    constexpr size_t NewN = N / Total;
    static_assert(Total > 1, "Total must be greater than 1");
    static_assert(NewN * Total == N, "N must be divisible by Total");
    return extract_part_impl<T>(
        bool_constant<(sizeof(T) * NewN >= 16)>(),  // dispatch on whether the result is a
                                                    // partial SSE register or larger
        size_constant<Index>(), size_constant<Total>(), x);
}

// }}}1

}  // namespace x86

// to_<intrin> {{{
template <class T, size_t N> constexpr Vc_INTRINSIC __m128 to_m128(Storage<T, N> a)
{
    static_assert(N <= 16 / sizeof(T));
    return a.template intrin<__m128>();
}
template <class T, size_t N> constexpr Vc_INTRINSIC __m128d to_m128d(Storage<T, N> a)
{
    static_assert(N <= 16 / sizeof(T));
    return a.template intrin<__m128d>();
}
template <class T, size_t N> constexpr Vc_INTRINSIC __m128i to_m128i(Storage<T, N> a)
{
    static_assert(N <= 16 / sizeof(T));
    return a.template intrin<__m128i>();
}

template <class T, size_t N> constexpr Vc_INTRINSIC __m256 to_m256(Storage<T, N> a)
{
    static_assert(N <= 32 / sizeof(T) && N > 16 / sizeof(T));
    return a.template intrin<__m256>();
}
template <class T, size_t N> constexpr Vc_INTRINSIC __m256d to_m256d(Storage<T, N> a)
{
    static_assert(N <= 32 / sizeof(T) && N > 16 / sizeof(T));
    return a.template intrin<__m256d>();
}
template <class T, size_t N> constexpr Vc_INTRINSIC __m256i to_m256i(Storage<T, N> a)
{
    static_assert(N <= 32 / sizeof(T) && N > 16 / sizeof(T));
    return a.template intrin<__m256i>();
}

template <class T, size_t N> constexpr Vc_INTRINSIC __m512 to_m512(Storage<T, N> a)
{
    static_assert(N <= 64 / sizeof(T) && N > 32 / sizeof(T));
    return a.template intrin<__m512>();
}
template <class T, size_t N> constexpr Vc_INTRINSIC __m512d to_m512d(Storage<T, N> a)
{
    static_assert(N <= 64 / sizeof(T) && N > 32 / sizeof(T));
    return a.template intrin<__m512d>();
}
template <class T, size_t N> constexpr Vc_INTRINSIC __m512i to_m512i(Storage<T, N> a)
{
    static_assert(N <= 64 / sizeof(T) && N > 32 / sizeof(T));
    return a.template intrin<__m512i>();
}

// }}}
// to_storage specializations for bitset and __mmask<N> {{{
#ifdef Vc_HAVE_AVX512_ABI
template <size_t N> class to_storage<std::bitset<N>>
{
    std::bitset<N> d;

public:
    constexpr to_storage(std::bitset<N> x) : d(x) {}
    template <class U> constexpr operator Storage<U, N>() const
    {
        return reinterpret_cast<builtin_type_t<U, N>>(
            detail::x86::convert_mask<sizeof(U), sizeof(builtin_type_t<U, N>)>(d));
    }
};

#define Vc_TO_STORAGE(type_)                                                             \
    template <> class to_storage<type_>                                                  \
    {                                                                                    \
        type_ d;                                                                         \
                                                                                         \
    public:                                                                              \
        constexpr to_storage(type_ x) : d(x) {}                                          \
                                                                                         \
        template <class U, size_t N> constexpr operator Storage<U, N>() const            \
        {                                                                                \
            return reinterpret_cast<builtin_type_t<U, N>>(                               \
                detail::x86::convert_mask<sizeof(U), sizeof(builtin_type_t<U, N>)>(d));  \
        }                                                                                \
                                                                                         \
        template <size_t N> constexpr operator Storage<bool, N>() const                  \
        {                                                                                \
            static_assert(                                                               \
                std::is_same_v<type_, typename bool_storage_member_type<N>::type>);      \
            return d;                                                                    \
        }                                                                                \
    }
Vc_TO_STORAGE(__mmask8);
Vc_TO_STORAGE(__mmask16);
Vc_TO_STORAGE(__mmask32);
Vc_TO_STORAGE(__mmask64);
#undef Vc_TO_STORAGE
#endif  // Vc_HAVE_AVX512_ABI

// }}}
// concat {{{
// These functions are part of the Storage interface => same namespace.
// These functions are only available when AVX or higher is enabled. In the future there
// may be more cases (e.g. half SSE -> full SSE or even MMX -> SSE).
#if 0//def Vc_HAVE_SSE2
template <class T>
Vc_INTRINSIC Vc_CONST Storage<T, 4 / sizeof(T)> Vc_VDECL
    concat(Storage<T, 2 / sizeof(T)> a, Storage<T, 2 / sizeof(T)> b)
{
    static_assert(std::is_integral_v<T>);
    return to_storage_unsafe(_mm_unpacklo_epi16(to_m128i(a), to_m128i(b)));
}

template <class T>
Vc_INTRINSIC Vc_CONST Storage<T, 8 / sizeof(T)> Vc_VDECL
    concat(Storage<T, 4 / sizeof(T)> a, Storage<T, 4 / sizeof(T)> b)
{
    static_assert(std::is_integral_v<T>);
    return to_storage_unsafe(_mm_unpacklo_epi32(to_m128i(a), to_m128i(b)));
}

Vc_INTRINSIC Vc_CONST Storage<float, 4> Vc_VDECL concat(Storage<float, 2> a,
                                                        Storage<float, 2> b)
{
    return to_storage(_mm_unpacklo_pd(to_m128d(a), to_m128d(b)));
}

template <class T>
Vc_INTRINSIC Vc_CONST Storage<T, 16 / sizeof(T)> Vc_VDECL
    concat(Storage<T, 8 / sizeof(T)> a, Storage<T, 8 / sizeof(T)> b)
{
    static_assert(std::is_integral_v<T>);
    return to_storage(_mm_unpacklo_epi64(to_m128d(a), to_m128d(b)));
}
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
template <class T>
Vc_INTRINSIC Vc_CONST Storage<T, 32 / sizeof(T)> Vc_VDECL
    concat(Storage<T, 16 / sizeof(T)> a, Storage<T, 16 / sizeof(T)> b)
{
    return concat(a.d, b.d);
}
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
template <class T>
Vc_INTRINSIC Vc_CONST Storage<T, 64 / sizeof(T)> Vc_VDECL
    concat(Storage<T, 32 / sizeof(T)> a, Storage<T, 32 / sizeof(T)> b)
{
    return concat(a.d, b.d);
}
#endif  // Vc_HAVE_AVX512F

template <class T, size_t N>
Vc_INTRINSIC Vc_CONST Storage<T, 4 * N> Vc_VDECL concat(Storage<T, N> a, Storage<T, N> b,
                                                        Storage<T, N> c, Storage<T, N> d)
{
    return concat(concat(a, b), concat(c, d));
}

template <class T, size_t N>
Vc_INTRINSIC Vc_CONST Storage<T, 8 * N> Vc_VDECL concat(Storage<T, N> a, Storage<T, N> b,
                                                        Storage<T, N> c, Storage<T, N> d,
                                                        Storage<T, N> e, Storage<T, N> f,
                                                        Storage<T, N> g, Storage<T, N> h)
{
    return concat(concat(concat(a, b), concat(c, d)), concat(concat(e, f), concat(g, h)));
}

//}}}
// convert_any_mask{{{
template <class To,  // required to be a detail::Storage specialization
          class T, size_t FromN>
To convert_any_mask(Storage<T, FromN> x)
{
    if constexpr (sizeof(T) == sizeof(typename To::value_type) &&
                  sizeof(To) == sizeof(x)) {
        // no change
        return to_storage(x.d);
    }
    if constexpr (sizeof(To) < 16) { // convert to __mmaskXX {{{
        if constexpr (sizeof(x) < 16) {
            // convert from __mmaskYY
            return x.d;
        } else {
            constexpr size_t cvt_id = FromN * 10 + sizeof(T);

            if constexpr (have_avx512bw_vl) {
                if constexpr (cvt_id == 16'1) { return    _mm_movepi8_mask(x); }
                if constexpr (cvt_id == 32'1) { return _mm256_movepi8_mask(x); }
                if constexpr (cvt_id ==  8'2) { return    _mm_movepi16_mask(x); }
                if constexpr (cvt_id == 16'2) { return _mm256_movepi16_mask(x); }
            }
            if constexpr (have_avx512dq_vl) {
                if constexpr (cvt_id ==  4'4) { return    _mm_movepi32_mask(to_m128i(x)); }
                if constexpr (cvt_id ==  8'4) { return _mm256_movepi32_mask(to_m256i(x)); }
                if constexpr (cvt_id ==  2'8) { return    _mm_movepi64_mask(to_m128i(x)); }
                if constexpr (cvt_id ==  4'8) { return _mm256_movepi64_mask(to_m256i(x)); }
            }
            if constexpr(have_avx512bw) {
                if constexpr (cvt_id == 16'1) { return _mm512_movepi8_mask(zeroExtend(x.intrin())); }
                if constexpr (cvt_id == 32'1) { return _mm512_movepi8_mask(zeroExtend(x.intrin())); }
                if constexpr (cvt_id == 64'1) { return _mm512_movepi8_mask(x); }
                if constexpr (cvt_id ==  8'2) { return _mm512_movepi16_mask(zeroExtend(x.intrin())); }
                if constexpr (cvt_id == 16'2) { return _mm512_movepi16_mask(zeroExtend(x.intrin())); }
                if constexpr (cvt_id == 32'2) { return _mm512_movepi16_mask(x); }
            }
            if constexpr (have_avx512dq) {
                if constexpr (cvt_id ==  4'4) { return _mm512_movepi32_mask(zeroExtend(to_m128i(x))); }
                if constexpr (cvt_id ==  8'4) { return _mm512_movepi32_mask(zeroExtend(to_m256i(x))); }
                if constexpr (cvt_id == 16'4) { return _mm512_movepi32_mask(to_m512i(x)); }
                if constexpr (cvt_id ==  2'8) { return _mm512_movepi64_mask(zeroExtend(to_m128i(x))); }
                if constexpr (cvt_id ==  4'8) { return _mm512_movepi64_mask(zeroExtend(to_m256i(x))); }
                if constexpr (cvt_id ==  8'8) { return _mm512_movepi64_mask(to_m512i(x)); }
            }
            if constexpr (have_avx512vl) {
                if constexpr (cvt_id ==  4'4) { return    _mm_cmp_epi32_mask(to_m128i(x), __m128i(), _MM_CMPINT_LT); }
                if constexpr (cvt_id ==  8'4) { return _mm256_cmp_epi32_mask(to_m256i(x), __m256i(), _MM_CMPINT_LT); }
                if constexpr (cvt_id ==  2'8) { return    _mm_cmp_epi64_mask(to_m128i(x), __m128i(), _MM_CMPINT_LT); }
                if constexpr (cvt_id ==  4'8) { return _mm256_cmp_epi64_mask(to_m256i(x), __m256i(), _MM_CMPINT_LT); }
            }
            if constexpr (cvt_id == 16'4) { return _mm512_cmp_epi32_mask(to_m512i(x), __m512i(), _MM_CMPINT_LT); }
            if constexpr (cvt_id ==  8'8) { return _mm512_cmp_epi64_mask(to_m512i(x), __m512i(), _MM_CMPINT_LT); }
            if constexpr (std::is_integral_v<T>) {
                if constexpr (cvt_id == 4'4) { return _mm512_cmp_epi32_mask(zeroExtend(x.intrin()), __m512i(), _MM_CMPINT_LT); }
                if constexpr (cvt_id == 8'4) { return _mm512_cmp_epi32_mask(zeroExtend(x.intrin()), __m512i(), _MM_CMPINT_LT); }
                if constexpr (cvt_id == 2'8) { return _mm512_cmp_epi64_mask(zeroExtend(x.intrin()), __m512i(), _MM_CMPINT_LT); }
                if constexpr (cvt_id == 4'8) { return _mm512_cmp_epi64_mask(zeroExtend(x.intrin()), __m512i(), _MM_CMPINT_LT); }
            } else {
                if constexpr (cvt_id == 4'4) { return    _mm_movemask_ps(x); }
                if constexpr (cvt_id == 8'4) { return _mm256_movemask_ps(x); }
                if constexpr (cvt_id == 2'8) { return    _mm_movemask_pd(x); }
                if constexpr (cvt_id == 4'8) { return _mm256_movemask_pd(x); }
            }

            if constexpr (cvt_id == 16'1) { return    _mm_movemask_epi8(x); }
            if constexpr (cvt_id == 32'1) { return _mm256_movemask_epi8(x); }
            if constexpr (cvt_id ==  8'2) { return x86::movemask_epi16(x); }
            if constexpr (cvt_id == 16'2) { return x86::movemask_epi16(x); }
        }
        // }}}
    } else if constexpr (sizeof(x) < 16) { // convert from __mmaskXX {{{
        // convert to __mm(128|256|512)
#ifdef Vc_HAVE_AVX512F
        return to_storage(
            detail::x86::convert_mask<sizeof(typename To::value_type), sizeof(To)>(
                static_cast<std::conditional_t<
                    (To::width <= 16),
                    std::conditional_t<To::width == 16, __mmask16, __mmask8>,
                    std::conditional_t<To::width == 32, __mmask32, __mmask64>>>(x)));
#endif
        // }}}
    } else { // convert __mmXXX to __mmXXX {{{
        using ToT = typename To::value_type;
        constexpr int FromBytes = sizeof(T);
        constexpr int ToBytes = sizeof(ToT);
        if constexpr (FromN == To::width && sizeof(To) == sizeof(x)) {
            // reinterpret the bits
            return storage_bitcast<ToT>(x);
        } else if constexpr (sizeof(To) == 16 && sizeof(x) == 16) {
            // SSE -> SSE {{{
            if constexpr (FromBytes == 4 && ToBytes == 8) {
                if constexpr(std::is_integral_v<T>) {
                    return to_storage(_mm_unpacklo_epi32(x, x));
                } else {
                    return to_storage(_mm_unpacklo_ps(x, x));
                }
            } else if constexpr (FromBytes == 2 && ToBytes == 8) {
                const auto y = _mm_unpacklo_epi16(x, x);
                return to_storage(_mm_unpacklo_epi32(y, y));
            } else if constexpr (FromBytes == 1 && ToBytes == 8) {
                auto y = _mm_unpacklo_epi8(x, x);
                y = _mm_unpacklo_epi16(y, y);
                return to_storage(_mm_unpacklo_epi32(y, y));
            } else if constexpr (FromBytes == 8 && ToBytes == 4) {
                if constexpr (std::is_floating_point_v<T>) {
                    return to_storage(_mm_shuffle_ps(to_m128(x), __m128(),
                                                     make_immediate<4>(1, 3, 1, 3)));
                } else {
                    auto y = to_m128i(x);
                    return to_storage(_mm_packs_epi32(y, __m128i()));
                }
            } else if constexpr (FromBytes == 2 && ToBytes == 4) {
                return to_storage(_mm_unpacklo_epi16(x, x));
            } else if constexpr (FromBytes == 1 && ToBytes == 4) {
                const auto y = _mm_unpacklo_epi8(x, x);
                return to_storage(_mm_unpacklo_epi16(y, y));
            } else if constexpr (FromBytes == 8 && ToBytes == 2) {
                if constexpr(have_ssse3) {
                    return _mm_shuffle_epi8(
                        to_m128i(x), _mm_setr_epi8(6, 7, 14, 15, -1, -1, -1, -1, -1, -1,
                                                   -1, -1, -1, -1, -1, -1));
                } else {
                    const auto y = _mm_packs_epi32(to_m128i(x), __m128i());
                    return _mm_packs_epi32(y, __m128i());
                }
            } else if constexpr (FromBytes == 4 && ToBytes == 2) {
                return _mm_packs_epi32(to_m128i(x), __m128i());
            } else if constexpr (FromBytes == 1 && ToBytes == 2) {
                return _mm_unpacklo_epi8(x, x);
            } else if constexpr (FromBytes == 8 && ToBytes == 1) {
                if constexpr(have_ssse3) {
                    return _mm_shuffle_epi8(
                        to_m128i(x), _mm_setr_epi8(7, 15, -1, -1, -1, -1, -1, -1, -1, -1,
                                                   -1, -1, -1, -1, -1, -1));
                } else {
                    auto y = _mm_packs_epi32(to_m128i(x), __m128i());
                    y = _mm_packs_epi32(y, __m128i());
                    return _mm_packs_epi16(y, __m128i());
                }
            } else if constexpr (FromBytes == 4 && ToBytes == 1) {
                if constexpr(have_ssse3) {
                    return _mm_shuffle_epi8(
                        to_m128i(x), _mm_setr_epi8(3, 7, 11, 15, -1, -1, -1, -1, -1, -1,
                                                   -1, -1, -1, -1, -1, -1));
                } else {
                    const auto y = _mm_packs_epi32(to_m128i(x), __m128i());
                    return _mm_packs_epi16(y, __m128i());
                }
            } else if constexpr (FromBytes == 2 && ToBytes == 1) {
                return _mm_packs_epi16(x, __m128i());
            } else {
                static_assert(!std::is_same_v<T, T>, "should be unreachable");
            }
            // }}}
        } else if constexpr (sizeof(To) == 32 && sizeof(x) == 32) {
            // AVX -> AVX {{{
            if constexpr (FromBytes == ToBytes) {  // keep low 1/2
                static_assert(!std::is_same_v<T, T>, "should be unreachable");
            } else if constexpr (FromBytes == ToBytes * 2) {
                const auto y = to_m256i(x);
                return to_storage(
                    _mm256_castsi128_si256(_mm_packs_epi16(lo128(y), hi128(y))));
            } else if constexpr (FromBytes == ToBytes * 4) {
                const auto y = to_m256i(x);
                return _mm256_castsi128_si256(
                    _mm_packs_epi16(_mm_packs_epi16(lo128(y), hi128(y)), __m128i()));
            } else if constexpr (FromBytes == ToBytes * 8) {
                const auto y = to_m256i(x);
                return _mm256_castsi128_si256(
                    _mm_shuffle_epi8(_mm_packs_epi16(lo128(y), hi128(y)),
                                     _mm_setr_epi8(3, 7, 11, 15, -1, -1, -1, -1, -1, -1,
                                                   -1, -1, -1, -1, -1, -1)));
            } else if constexpr (FromBytes * 2 == ToBytes) {
                auto y = fixup_avx_xzyw(x.intrin());
                if constexpr(std::is_floating_point_v<T>) {
                    return to_storage(_mm256_unpacklo_ps(y, y));
                } else {
                    return to_storage(_mm256_unpacklo_epi8(y, y));
                }
            } else if constexpr (FromBytes * 4 == ToBytes) {
                auto y = _mm_unpacklo_epi8(lo128(to_m256i(x)),
                                           lo128(to_m256i(x)));  // drops 3/4 of input
                return to_storage(
                    concat(_mm_unpacklo_epi16(y, y), _mm_unpackhi_epi16(y, y)));
            } else if constexpr (FromBytes == 1 && ToBytes == 8) {
                auto y = _mm_unpacklo_epi8(lo128(to_m256i(x)),
                                           lo128(to_m256i(x)));  // drops 3/4 of input
                y = _mm_unpacklo_epi16(y, y);  // drops another 1/2 => 7/8 total
                return to_storage(
                    concat(_mm_unpacklo_epi32(y, y), _mm_unpackhi_epi32(y, y)));
            } else {
                static_assert(!std::is_same_v<T, T>, "should be unreachable");
            }
            // }}}
        } else if constexpr (sizeof(To) == 32 && sizeof(x) == 16) {
            // SSE -> AVX {{{
            if constexpr (FromBytes == ToBytes) {
                return to_storage(
                    intrinsic_type_t<T, 32 / sizeof(T)>(zeroExtend(x.intrin())));
            } else if constexpr (FromBytes * 2 == ToBytes) {  // keep all
                return to_storage(concat(_mm_unpacklo_epi8(to_m128i(x), to_m128i(x)),
                                         _mm_unpackhi_epi8(to_m128i(x), to_m128i(x))));
            } else if constexpr (FromBytes * 4 == ToBytes) {
                if constexpr (have_avx2) {
                    return to_storage(_mm256_shuffle_epi8(
                        concat(to_m128i(x), to_m128i(x)),
                        _mm256_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                         4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7,
                                         7)));
                } else {
                    return to_storage(
                        concat(_mm_shuffle_epi8(to_m128i(x),
                                                _mm_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2,
                                                              2, 2, 2, 3, 3, 3, 3)),
                               _mm_shuffle_epi8(to_m128i(x),
                                                _mm_setr_epi8(4, 4, 4, 4, 5, 5, 5, 5, 6,
                                                              6, 6, 6, 7, 7, 7, 7))));
                }
            } else if constexpr (FromBytes * 8 == ToBytes) {
                if constexpr (have_avx2) {
                    return to_storage(_mm256_shuffle_epi8(
                        concat(to_m128i(x), to_m128i(x)),
                        _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                         2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3,
                                         3)));
                } else {
                    return to_storage(
                        concat(_mm_shuffle_epi8(to_m128i(x),
                                                _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1,
                                                              1, 1, 1, 1, 1, 1, 1)),
                               _mm_shuffle_epi8(to_m128i(x),
                                                _mm_setr_epi8(2, 2, 2, 2, 2, 2, 2, 2, 3,
                                                              3, 3, 3, 3, 3, 3, 3))));
                }
            } else if constexpr (FromBytes == ToBytes * 2) {
                return to_storage(
                    __m256i(zeroExtend(_mm_packs_epi16(auto_cast(x), __m128i()))));
            } else if constexpr (FromBytes == 8 && ToBytes == 2) {
                return __m256i(zeroExtend(_mm_shuffle_epi8(
                    to_m128i(x), _mm_setr_epi8(6, 7, 14, 15, -1, -1, -1, -1, -1, -1, -1,
                                               -1, -1, -1, -1, -1))));
            } else if constexpr (FromBytes == 4 && ToBytes == 1) {
                return __m256i(zeroExtend(_mm_shuffle_epi8(
                    to_m128i(x), _mm_setr_epi8(3, 7, 11, 15, -1, -1, -1, -1, -1, -1, -1,
                                               -1, -1, -1, -1, -1))));
            } else if constexpr (FromBytes == 8 && ToBytes == 1) {
                return __m256i(zeroExtend(_mm_shuffle_epi8(
                    to_m128i(x), _mm_setr_epi8(7, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                               -1, -1, -1, -1, -1))));
            } else {
                static_assert(!std::is_same_v<T, T>, "should be unreachable");
            }
            // }}}
        } else if constexpr (sizeof(To) == 16 && sizeof(x) == 32) {
            // AVX -> SSE {{{
            if constexpr (FromBytes == ToBytes) {  // keep low 1/2
                return to_storage(lo128(x.d));
            } else if constexpr (FromBytes == ToBytes * 2) {  // keep all
                auto y = to_m256i(x);
                return to_storage(_mm_packs_epi16(lo128(y), hi128(y)));
            } else if constexpr (FromBytes == ToBytes * 4) {  // add 1/2 undef
                auto y = to_m256i(x);
                return to_storage(
                    _mm_packs_epi16(_mm_packs_epi16(lo128(y), hi128(y)), __m128i()));
            } else if constexpr (FromBytes == 8 && ToBytes == 1) {  // add 3/4 undef
                auto y = to_m256i(x);
                return _mm_shuffle_epi8(_mm_packs_epi16(lo128(y), hi128(y)),
                                        _mm_setr_epi8(3, 7, 11, 15, -1, -1, -1, -1, -1,
                                                      -1, -1, -1, -1, -1, -1, -1));
            } else if constexpr (FromBytes * 2 == ToBytes) {  // keep low 1/4
                auto y = lo128(to_m256i(x));
                return to_storage(_mm_unpacklo_epi8(y, y));
            } else if constexpr (FromBytes * 4 == ToBytes) {  // keep low 1/8
                auto y = lo128(to_m256i(x));
                y = _mm_unpacklo_epi8(y, y);
                return to_storage(_mm_unpacklo_epi8(y, y));
            } else if constexpr (FromBytes * 8 == ToBytes) {  // keep low 1/16
                auto y = lo128(to_m256i(x));
                y = _mm_unpacklo_epi8(y, y);
                y = _mm_unpacklo_epi8(y, y);
                return to_storage(_mm_unpacklo_epi8(y, y));
            } else {
                static_assert(!std::is_same_v<T, T>, "should be unreachable");
            }
            // }}}
        }
        // }}}
    }
} //}}}

}  // namespace detail
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_SIMD_X86_STORAGE_H_
