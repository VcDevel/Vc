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

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
template <class T> using sse_simd_member_type = Storage<T, 16 / sizeof(T)>;
template <class T> using sse_mask_member_type = Storage<T, 16 / sizeof(T)>;

template <class T> using avx_simd_member_type = Storage<T, 32 / sizeof(T)>;
template <class T> using avx_mask_member_type = Storage<T, 32 / sizeof(T)>;

template <class T> using avx512_simd_member_type = Storage<T, 64 / sizeof(T)>;
template <class T> using avx512_mask_member_type = Storage<bool, 64 / sizeof(T)>;
template <size_t N> using avx512_mask_member_type_n = Storage<bool, N>;

namespace x86
{

// x_ aliases {{{
#ifdef Vc_HAVE_SSE
using x_f32 = Storage< float,  4>;
#ifdef Vc_HAVE_SSE2
using x_f64 = Storage<double,  2>;
using x_i08 = Storage< schar, 16>;
using x_u08 = Storage< uchar, 16>;
using x_i16 = Storage< short,  8>;
using x_u16 = Storage<ushort,  8>;
using x_i32 = Storage<   int,  4>;
using x_u32 = Storage<  uint,  4>;
using x_i64 = Storage< llong,  2>;
using x_u64 = Storage<ullong,  2>;
using x_long = Storage<long,   16 / sizeof(long)>;
using x_ulong = Storage<ulong, 16 / sizeof(ulong)>;
using x_long_equiv = Storage<equal_int_type_t<long>, x_long::size()>;
using x_ulong_equiv = Storage<equal_int_type_t<ulong>, x_ulong::size()>;
#endif  // Vc_HAVE_SSE2
#endif  // Vc_HAVE_SSE

//}}}
#ifdef Vc_HAVE_AVX
// y_ aliases {{{
using y_f32 = Storage< float,  8>;
using y_f64 = Storage<double,  4>;
using y_i08 = Storage< schar, 32>;
using y_u08 = Storage< uchar, 32>;
using y_i16 = Storage< short, 16>;
using y_u16 = Storage<ushort, 16>;
using y_i32 = Storage<   int,  8>;
using y_u32 = Storage<  uint,  8>;
using y_i64 = Storage< llong,  4>;
using y_u64 = Storage<ullong,  4>;
using y_long = Storage<long,   32 / sizeof(long)>;
using y_ulong = Storage<ulong, 32 / sizeof(ulong)>;
using y_long_equiv = Storage<equal_int_type_t<long>, y_long::size()>;
using y_ulong_equiv = Storage<equal_int_type_t<ulong>, y_ulong::size()>;

//}}}
// lo/hi/extract128 {{{
template <typename T, size_t N>
Vc_INTRINSIC Vc_CONST Storage<T, 16 / sizeof(T)> Vc_VDECL lo128(Storage<T, N> x)
{
    return lo128(x.v());
}
template <typename T, size_t N>
Vc_INTRINSIC Vc_CONST Storage<T, 16 / sizeof(T)> Vc_VDECL hi128(Storage<T, N> x)
{
    return hi128(x.v());
}

template <int offset, typename T, size_t N>
Vc_INTRINSIC Vc_CONST Storage<T, 16 / sizeof(T)> Vc_VDECL extract128(Storage<T, N> x)
{
    return extract128<offset>(x.v());
}

//}}}
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
// z_ aliases {{{
using z_f32 = Storage< float, 16>;
using z_f64 = Storage<double,  8>;
using z_i32 = Storage<   int, 16>;
using z_u32 = Storage<  uint, 16>;
using z_i64 = Storage< llong,  8>;
using z_u64 = Storage<ullong,  8>;
using z_long = Storage<long,   64 / sizeof(long)>;
using z_ulong = Storage<ulong, 64 / sizeof(ulong)>;
using z_i08 = Storage< schar, 64>;
using z_u08 = Storage< uchar, 64>;
using z_i16 = Storage< short, 32>;
using z_u16 = Storage<ushort, 32>;
using z_long_equiv = Storage<equal_int_type_t<long>, z_long::size()>;
using z_ulong_equiv = Storage<equal_int_type_t<ulong>, z_ulong::size()>;

//}}}
// lo/hi256 {{{
template <typename T, size_t N>
Vc_INTRINSIC Vc_CONST Storage<T, 32 / sizeof(T)> Vc_VDECL lo256(Storage<T, N> x)
{
    return lo256(x.v());
}
template <typename T, size_t N>
Vc_INTRINSIC Vc_CONST Storage<T, 32 / sizeof(T)> Vc_VDECL hi256(Storage<T, N> x)
{
    return hi256(x.v());
}
//}}}
#endif  // Vc_HAVE_AVX512F

// concat {{{
#ifdef Vc_HAVE_AVX
template <class T>
Vc_INTRINSIC Vc_CONST Storage<T, 32 / sizeof(T)> Vc_VDECL
    concat(Storage<T, 16 / sizeof(T)> a, Storage<T, 16 / sizeof(T)> b)
{
    return concat(a.v(), b.v());
}
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
template <class T>
Vc_INTRINSIC Vc_CONST Storage<T, 64 / sizeof(T)> Vc_VDECL
    concat(Storage<T, 32 / sizeof(T)> a, Storage<T, 32 / sizeof(T)> b)
{
    return concat(a.v(), b.v());
}
#endif  // Vc_HAVE_AVX512F

//}}}
// extract_part {{{1
// identity {{{2
template <class T>
Vc_INTRINSIC const Storage<T, 16 / sizeof(T)>& Vc_VDECL
    extract_part_impl(std::true_type, size_constant<0>, size_constant<1>,
                      const Storage<T, 16 / sizeof(T)>& x)
{
    return x;
}

// AVX to SSE splits {{{2
#ifdef Vc_HAVE_AVX
template <class T>
Vc_INTRINSIC Storage<T, 16 / sizeof(T)> Vc_VDECL extract_part_impl(
    std::true_type, size_constant<0>, size_constant<2>, Storage<T, 32 / sizeof(T)> x)
{
    return lo128(x);
}
template <class T>
Vc_INTRINSIC Storage<T, 16 / sizeof(T)> Vc_VDECL extract_part_impl(
    std::true_type, size_constant<1>, size_constant<2>, Storage<T, 32 / sizeof(T)> x)
{
    return hi128(x);
}
#endif  // Vc_HAVE_AVX

// AVX512 to AVX or SSE splits {{{2
#ifdef Vc_HAVE_AVX512F
template <class T, size_t Index>
Vc_INTRINSIC Storage<T, 16 / sizeof(T)> Vc_VDECL extract_part_impl(
    std::true_type, size_constant<Index>, size_constant<4>, Storage<T, 64 / sizeof(T)> x)
{
    return extract128<Index>(x);
}

template <class T>
Vc_INTRINSIC Storage<T, 32 / sizeof(T)> Vc_VDECL extract_part_impl(
    std::true_type, size_constant<0>, size_constant<2>, Storage<T, 64 / sizeof(T)> x)
{
    return lo256(x);
}
template <class T>
Vc_INTRINSIC Storage<T, 32 / sizeof(T)> Vc_VDECL extract_part_impl(
    std::true_type, size_constant<1>, size_constant<2>, Storage<T, 64 / sizeof(T)> x)
{
    return hi256(x);
}
#endif  // Vc_HAVE_AVX512F

// partial SSE (shifts) {{{2
template <class T, size_t Index, size_t Total, size_t N>
Vc_INTRINSIC Storage<T, 16 / sizeof(T)> Vc_VDECL extract_part_impl(std::false_type,
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
Vc_INTRINSIC Vc_CONST Storage<T, constexpr_max(16 / sizeof(T), N / Total)> Vc_VDECL
extract_part(Storage<T, N> x)
{
    constexpr size_t NewN = N / Total;
    static_assert(Total > 1, "Total must be greater than 1");
    static_assert(NewN * Total == N, "N must be divisible by Total");
    return extract_part_impl<T>(
        std::integral_constant<bool, (sizeof(T) * NewN >= 16)>(),  // dispatch on whether
                                                                   // the result is a
                                                                   // partial SSE register
                                                                   // or larger
        std::integral_constant<size_t, Index>(), std::integral_constant<size_t, Total>(),
        x);
}

// }}}1

// mask conversions {{{
template <size_t SizeofTo, size_t NTo, size_t SizeofFrom, size_t NFrom, class To,
          bool IsAvx512 = (sizeof(To) < sizeof(__m128))>
struct convert_mask_impl;

#ifdef Vc_HAVE_SSE2
template <size_t Sizeof, size_t N, class To>
struct convert_mask_impl<Sizeof, N, Sizeof, N, To, false> {
    template <class T> Vc_INTRINSIC To operator()(T x) { return intrin_cast<To>(x); }
};

#ifdef Vc_HAVE_AVX512F
template <size_t N, class To> struct convert_mask_impl<1, N, 1, N, To, true> {
    Vc_INTRINSIC To operator()(To x) { return x; }
};

template <size_t NTo, size_t NFrom, class To>
struct convert_mask_impl<1, NTo, 1, NFrom, To, true> {
#ifdef Vc_HAVE_AVX512BW
#ifdef Vc_HAVE_AVX512VL
    Vc_INTRINSIC To operator()(__m128i x) { return _mm_movepi8_mask(x); }
    Vc_INTRINSIC To operator()(__m256i x) { return _mm256_movepi8_mask(x); }
#else   // Vc_HAVE_AVX512VL
    Vc_INTRINSIC To operator()(__m128i x) { return _mm512_movepi8_mask(zeroExtend64(x)); }
    Vc_INTRINSIC To operator()(__m256i x) { return _mm512_movepi8_mask(zeroExtend64(x)); }
#endif  // Vc_HAVE_AVX512VL
#else  // Vc_HAVE_AVX512BW
    Vc_INTRINSIC To operator()(__m128i x) { return _mm_movemask_epi8(x); }
    Vc_INTRINSIC To operator()(__m256i x) { return _mm256_movemask_epi8(x); }
#endif  // Vc_HAVE_AVX512BW
};
template <size_t NTo, size_t NFrom, class To>
struct convert_mask_impl<1, NTo, 2, NFrom, To, true> {
#ifdef Vc_HAVE_AVX512BW
#ifdef Vc_HAVE_AVX512VL
    Vc_INTRINSIC To operator()(__m128i x) { return _mm_movepi16_mask(x); }
    Vc_INTRINSIC To operator()(__m256i x) { return _mm256_movepi16_mask(x); }
#else   // Vc_HAVE_AVX512VL
    Vc_INTRINSIC To operator()(__m128i x) { return _mm512_movepi16_mask(zeroExtend64(x)); }
    Vc_INTRINSIC To operator()(__m256i x) { return _mm512_movepi16_mask(zeroExtend64(x)); }
#endif  // Vc_HAVE_AVX512VL
#else  // Vc_HAVE_AVX512BW
    Vc_INTRINSIC To operator()(__m128i x) { return x86::movemask_epi16(x); }
    Vc_INTRINSIC To operator()(__m256i x) { return x86::movemask_epi16(x); }
#endif  // Vc_HAVE_AVX512BW
};
template <size_t NTo, size_t NFrom, class To>
struct convert_mask_impl<1, NTo, 4, NFrom, To, true> {
#ifdef Vc_HAVE_AVX512BW
#ifdef Vc_HAVE_AVX512VL
    Vc_INTRINSIC To operator()(__m128i x) { return _mm_movepi32_mask(x); }
    Vc_INTRINSIC To operator()(__m256i x) { return _mm256_movepi32_mask(x); }
#else   // Vc_HAVE_AVX512VL
    Vc_INTRINSIC To operator()(__m128i x) { return _mm512_movepi32_mask(zeroExtend64(x)); }
    Vc_INTRINSIC To operator()(__m256i x) { return _mm512_movepi32_mask(zeroExtend64(x)); }
#endif  // Vc_HAVE_AVX512VL
    Vc_INTRINSIC To operator()(__m128 x) { return operator()(_mm_castps_si128(x)); }
    Vc_INTRINSIC To operator()(__m256 x) { return operator()(_mm256_castps_si256(x)); }
#else  // Vc_HAVE_AVX512BW
    Vc_INTRINSIC To operator()(__m128 x) { return _mm_movemask_ps(x); }
    Vc_INTRINSIC To operator()(__m256 x) { return _mm256_movemask_ps(x); }
    Vc_INTRINSIC To operator()(__m128i x) { return operator()(_mm_castsi128_ps(x)); }
    Vc_INTRINSIC To operator()(__m256i x) { return operator()(_mm256_castsi256_ps(x)); }
#endif  // Vc_HAVE_AVX512BW
};
template <size_t NTo, size_t NFrom, class To>
struct convert_mask_impl<1, NTo, 8, NFrom, To, true> {
#ifdef Vc_HAVE_AVX512BW
#ifdef Vc_HAVE_AVX512VL
    Vc_INTRINSIC To operator()(__m128i x) { return _mm_movepi64_mask(x); }
    Vc_INTRINSIC To operator()(__m256i x) { return _mm256_movepi64_mask(x); }
#else   // Vc_HAVE_AVX512VL
    Vc_INTRINSIC To operator()(__m128i x) { return _mm512_movepi64_mask(zeroExtend64(x)); }
    Vc_INTRINSIC To operator()(__m256i x) { return _mm512_movepi64_mask(zeroExtend64(x)); }
#endif  // Vc_HAVE_AVX512VL
    Vc_INTRINSIC To operator()(__m128d x) { return operator()(_mm_castpd_si128(x)); }
    Vc_INTRINSIC To operator()(__m256d x) { return operator()(_mm256_castpd_si256(x)); }
#else  // Vc_HAVE_AVX512BW
    Vc_INTRINSIC To operator()(__m128d x) { return _mm_movemask_pd(x); }
    Vc_INTRINSIC To operator()(__m256d x) { return _mm256_movemask_pd(x); }
    Vc_INTRINSIC To operator()(__m128i x) { return operator()(_mm_castsi128_pd(x)); }
    Vc_INTRINSIC To operator()(__m256i x) { return operator()(_mm256_castsi256_pd(x)); }
#endif  // Vc_HAVE_AVX512BW
};
#endif  // Vc_HAVE_AVX512F

#define Vc_CONVERT_MASK_IMPL_BEGIN(SizeofTo_, NTo_, SizeofFrom_, NFrom_)                 \
    template <class To>                                                                  \
    struct convert_mask_impl<SizeofTo_, NTo_, SizeofFrom_, NFrom_, To, false> {          \
        template <class T> Vc_INTRINSIC To operator()(T x)                               \
        {
#define Vc_CONVERT_MASK_IMPL_END }}

Vc_CONVERT_MASK_IMPL_BEGIN(8, 2, 4, 4)
    return intrin_cast<To>(
        _mm_unpacklo_ps(intrin_cast<__m128>(x), intrin_cast<__m128>(x)));
Vc_CONVERT_MASK_IMPL_END;

Vc_CONVERT_MASK_IMPL_BEGIN(8, 2, 2, 8)
    auto y = _mm_unpacklo_epi16(x, x);
    return intrin_cast<To>(_mm_unpacklo_epi32(y, y));
Vc_CONVERT_MASK_IMPL_END;

Vc_CONVERT_MASK_IMPL_BEGIN(8, 2, 1, 16)
    auto y = _mm_unpacklo_epi8(x, x);
    y = _mm_unpacklo_epi16(y, y);
    return intrin_cast<To>(_mm_unpacklo_epi32(y, y));
Vc_CONVERT_MASK_IMPL_END;

Vc_CONVERT_MASK_IMPL_BEGIN(4, 4, 8, 2)
    auto y = intrin_cast<__m128i>(x);
    return intrin_cast<To>(_mm_packs_epi32(y, y));
Vc_CONVERT_MASK_IMPL_END;

Vc_CONVERT_MASK_IMPL_BEGIN(4, 4, 2, 8)
    return intrin_cast<To>(_mm_unpacklo_epi16(x, x));
Vc_CONVERT_MASK_IMPL_END;

Vc_CONVERT_MASK_IMPL_BEGIN(4, 4, 1, 16)
    auto y = _mm_unpacklo_epi8(x, x);
    return _mm_unpacklo_epi16(y, y);
Vc_CONVERT_MASK_IMPL_END;

Vc_CONVERT_MASK_IMPL_BEGIN(2, 8, 8, 2)
    auto y = _mm_packs_epi32(intrin_cast<__m128i>(x), intrin_cast<__m128i>(x));
    return _mm_packs_epi32(y, y);
Vc_CONVERT_MASK_IMPL_END;

Vc_CONVERT_MASK_IMPL_BEGIN(2, 8, 4, 4)
    return _mm_packs_epi32(intrin_cast<__m128i>(x), intrin_cast<__m128i>(x));
Vc_CONVERT_MASK_IMPL_END;

Vc_CONVERT_MASK_IMPL_BEGIN(2, 8, 1, 16)
    return _mm_unpacklo_epi8(x, x);
Vc_CONVERT_MASK_IMPL_END;

Vc_CONVERT_MASK_IMPL_BEGIN(1, 16, 8, 2)
    auto y = _mm_packs_epi32(intrin_cast<__m128i>(x), intrin_cast<__m128i>(x));
    y = _mm_packs_epi32(y, y);
    return _mm_packs_epi16(y, y);
Vc_CONVERT_MASK_IMPL_END;

Vc_CONVERT_MASK_IMPL_BEGIN(1, 16, 4, 4)
    auto y = _mm_packs_epi32(intrin_cast<__m128i>(x), intrin_cast<__m128i>(x));
    return _mm_packs_epi16(y, y);
Vc_CONVERT_MASK_IMPL_END;

Vc_CONVERT_MASK_IMPL_BEGIN(1, 16, 2, 8)
    return _mm_packs_epi16(x, x);
Vc_CONVERT_MASK_IMPL_END;

#ifdef Vc_HAVE_AVX
Vc_CONVERT_MASK_IMPL_BEGIN(8, 4, 4, 4)
    return intrin_cast<To>(
        concat(_mm_unpacklo_ps(intrin_cast<__m128>(x), intrin_cast<__m128>(x)),
               _mm_unpackhi_ps(intrin_cast<__m128>(x), intrin_cast<__m128>(x))));
Vc_CONVERT_MASK_IMPL_END;

Vc_CONVERT_MASK_IMPL_BEGIN(4, 4, 8, 4)
    auto y = intrin_cast<__m256i>(x);
    return intrin_cast<To>(_mm_packs_epi32(lo128(y), hi128(y)));
Vc_CONVERT_MASK_IMPL_END;

#ifdef Vc_HAVE_AVX2
#endif  // Vc_HAVE_AVX2
#endif  // Vc_HAVE_AVX
#endif  // Vc_HAVE_SSE2

// }}}

}  // namespace x86

template <class To, class T, size_t Size> To convert_mask(Storage<T, Size> x)
{
    return convert_mask_impl<sizeof(typename To::value_type), To::size(), sizeof(T), Size,
                             typename To::VectorType>()(x);
}

}  // namespace detail
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_SIMD_X86_STORAGE_H_
