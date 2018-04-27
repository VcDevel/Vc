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

#ifndef VC_SIMD_X86_ARITHMETICS_H_
#define VC_SIMD_X86_ARITHMETICS_H_

#include "storage.h"
#include "convert.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
namespace x86
{
// plus{{{1
template <class T, size_t N>
constexpr Vc_INTRINSIC Storage<T, N> plus(Storage<T, N> a, Storage<T, N> b)
{
    return a.d + b.d;
}

// minus{{{1
template <class T, size_t N>
constexpr Vc_INTRINSIC Storage<T, N> minus(Storage<T, N> a, Storage<T, N> b)
{
    return a.d - b.d;
}

// multiplies{{{1
template <class T, size_t N>
constexpr Vc_INTRINSIC Storage<T, N> Vc_VDECL multiplies(Storage<T, N> a, Storage<T, N> b)
{
    return a.d * b.d;
}
Vc_INTRINSIC __m128i Vc_VDECL multiplies(x_i08 a, x_i08 b) {
    return or_(
        and_(_mm_mullo_epi16(a, b), srli_epi16<8>(allone<__m128i>())),
        slli_epi16<8>(_mm_mullo_epi16(_mm_srli_si128(a, 1), _mm_srli_si128(b, 1))));
}
Vc_INTRINSIC __m128i Vc_VDECL multiplies(x_u08 a, x_u08 b) { return multiplies(x_i08(a), x_i08(b)); }

#ifdef Vc_HAVE_AVX2
Vc_INTRINSIC __m256i Vc_VDECL multiplies(y_i08 a, y_i08 b) {
    return or_(and_(_mm256_mullo_epi16(a, b), srli_epi16<8>(allone<__m256i>())),
               slli_epi16<8>(
                   _mm256_mullo_epi16(_mm256_srli_si256(a, 1), _mm256_srli_si256(b, 1))));
}
Vc_INTRINSIC __m256i Vc_VDECL multiplies(y_u08 a, y_u08 b) { return multiplies(y_i08(a), y_i08(b)); }
#endif  // Vc_HAVE_AVX2

#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC __m512i Vc_VDECL multiplies(z_i64 a, z_i64 b) {
#ifdef Vc_HAVE_AVX512DQ
    return _mm512_mullo_epi64(a, b);
#else   // Vc_HAVE_AVX512DQ
    const __m512i r0 = _mm512_mul_epu32(a, b);
    const __m512i aShift = _mm512_srli_epi64(a, 32);
    const __m512i bShift = _mm512_srli_epi64(b, 32);
    const __m512i r1 = _mm512_slli_epi64(_mm512_mul_epu32(aShift, b), 32);
    const __m512i r2 = _mm512_slli_epi64(_mm512_mul_epu32(a, bShift), 32);
    return _mm512_add_epi64(_mm512_add_epi64(r0, r1), r2);
#endif  // Vc_HAVE_AVX512DQ
}
Vc_INTRINSIC __m512i Vc_VDECL multiplies(z_u64 a, z_u64 b) { return multiplies(z_i64(a),z_i64(b)); }
#ifdef Vc_HAVE_AVX512BW
Vc_INTRINSIC __m512i Vc_VDECL multiplies(z_i08 a, z_i08 b) {
    return or_(
        and_(_mm512_mullo_epi16(a, b), _mm512_srli_epi16(allone<__m512i>(), 8)),
        _mm512_slli_epi16(_mm512_mullo_epi16(_mm512_srli_epi16(a, 8), _mm512_srli_epi16(b, 8)), 8));
}
Vc_INTRINSIC __m512i Vc_VDECL multiplies(z_u08 a, z_u08 b) { return multiplies(z_i08(a), z_i08(b)); }
#endif  // Vc_HAVE_AVX512BW
#endif  // Vc_HAVE_AVX512F

// divides{{{1
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
// builtin{{{2
template <class T, size_t N>
constexpr Vc_INTRINSIC Storage<T, N> divides(Storage<T, N> a, Storage<T, N> b)
{
    return a.d / b.d;
}
#else   // Vc_USE_BUILTIN_VECTOR_TYPES
// sse{{{2
Vc_INTRINSIC x_i32 Vc_VDECL divides(x_i32 a, x_i32 b) {
#ifdef Vc_HAVE_AVX
    return _mm256_cvttpd_epi32(
        _mm256_div_pd(_mm256_cvtepi32_pd(a), _mm256_cvtepi32_pd(b)));
#else
    return a.d / b.d;
#endif
}

Vc_INTRINSIC x_u32 Vc_VDECL divides(x_u32 a, x_u32 b) {
#ifdef Vc_HAVE_AVX512VL
    return _mm256_cvttpd_epu32(
        _mm256_div_pd(_mm256_cvtepu32_pd(a), _mm256_cvtepu32_pd(b)));
#else
    return a.d / b.d;
#endif
}

Vc_INTRINSIC x_i16 Vc_VDECL divides(x_i16 a, x_i16 b) {
    const __m128i mask = broadcast16(0x0000ffffu);
    const auto ahi = andnot_(mask, a);
    const auto bhi = andnot_(mask, b);
    const auto alo = _mm_slli_epi32(a, 16);
    const auto blo = _mm_slli_epi32(b, 16);
#ifdef Vc_HAVE_AVX
    __m256i c = _mm256_cvttps_epi32(_mm256_div_ps(_mm256_cvtepi32_ps(concat(ahi, alo)),
                                                  _mm256_cvtepi32_ps(concat(bhi, blo))));
    return or_(_mm_slli_epi32(lo128(c), 16), and_(mask, hi128(c)));
#else
    __m128i c[2] = {
        _mm_cvttps_epi32(_mm_div_ps(_mm_cvtepi32_ps(ahi), _mm_cvtepi32_ps(bhi))),
        _mm_cvttps_epi32(_mm_div_ps(_mm_cvtepi32_ps(alo), _mm_cvtepi32_ps(blo)))};
    return or_(_mm_slli_epi32(c[0], 16), and_(mask, c[1]));
#endif
}

Vc_INTRINSIC x_u16 Vc_VDECL divides(x_u16 a, x_u16 b) {
#ifdef Vc_HAVE_AVX
    return convert<x_u16>(_mm256_div_ps(convert<y_f32>(a), convert<y_f32>(b)));
#else
    return or_(
        _mm_slli_epi32(
            _mm_cvttps_epi32(_mm_div_ps(_mm_cvtepi32_ps(_mm_srli_epi32(a, 16)),
                                        _mm_cvtepi32_ps(_mm_srli_epi32(b, 16)))),
            16),
        _mm_cvttps_epi32(_mm_div_ps(_mm_cvtepi32_ps(and_(a, broadcast16(0xffff))),
                                    _mm_cvtepi32_ps(and_(b, broadcast16(0xffff))))));
#endif
}

inline x_i08 Vc_VDECL divides(x_i08 a, x_i08 b) {
#ifdef Vc_HAVE_AVX512F
    return convert<x_i08>(_mm512_div_ps(convert<z_f32>(a), convert<z_f32>(b)));
#else
    const __m128i maskhi = broadcast16(0xff000000u);
    const __m128i masklo = _mm_srli_epi32(maskhi, 24);
    __m128i r[4] = {
        _mm_cvttps_epi32(_mm_div_ps(_mm_cvtepi32_ps(     _mm_slli_epi32(a, 24)         ),
                                    _mm_cvtepi32_ps(     _mm_slli_epi32(b, 24)         ))),
        _mm_cvttps_epi32(_mm_div_ps(_mm_cvtepi32_ps(and_(_mm_slli_epi32(a, 16), maskhi)),
                                    _mm_cvtepi32_ps(and_(_mm_slli_epi32(b, 16), maskhi)))),
        _mm_cvttps_epi32(_mm_div_ps(_mm_cvtepi32_ps(and_(_mm_slli_epi32(a,  8), maskhi)),
                                    _mm_cvtepi32_ps(and_(_mm_slli_epi32(b,  8), maskhi)))),
        _mm_cvttps_epi32(_mm_div_ps(_mm_cvtepi32_ps(and_(               a     , maskhi)),
                                    _mm_cvtepi32_ps(and_(               b     , maskhi))))
    };
    r[0] = and_(r[0], masklo);
    r[1] = _mm_slli_epi32(and_(r[1], masklo), 8);
    r[2] = _mm_slli_epi32(and_(r[2], masklo), 16);
    r[3] = _mm_slli_epi32(r[3], 24);
    return or_(or_(r[0], r[1]), or_(r[2], r[3]));
#endif
}

inline x_u08 Vc_VDECL divides(x_u08 a, x_u08 b) {
#ifdef Vc_HAVE_AVX512F
    return convert<x_u08>(_mm512_div_ps(convert<z_f32>(a), convert<z_f32>(b)));
#else
    const __m128i mask = broadcast16(0xff);
    return or_(
        or_(_mm_cvttps_epi32(_mm_div_ps(_mm_cvtepi32_ps(and_(a, mask)),
                                        _mm_cvtepi32_ps(and_(b, mask)))),
            _mm_slli_epi32(_mm_cvttps_epi32(_mm_div_ps(
                               _mm_cvtepi32_ps(and_(_mm_srli_epi32(a, 8), mask)),
                               _mm_cvtepi32_ps(and_(_mm_srli_epi32(b, 8), mask)))),
                           8)),
        or_(_mm_slli_epi32(_mm_cvttps_epi32(_mm_div_ps(
                               _mm_cvtepi32_ps(and_(_mm_srli_epi32(a, 16), mask)),
                               _mm_cvtepi32_ps(and_(_mm_srli_epi32(b, 16), mask)))),
                           16),
            _mm_slli_epi32(
                _mm_cvttps_epi32(_mm_div_ps(_mm_cvtepi32_ps(_mm_srli_epi32(a, 24)),
                                            _mm_cvtepi32_ps(_mm_srli_epi32(b, 24)))),
                24)));
#endif
}

// avx{{{2
#ifdef Vc_HAVE_AVX2
Vc_INTRINSIC y_i32 Vc_VDECL divides(y_i32 a, y_i32 b) {
#ifdef Vc_HAVE_AVX512F
    return _mm512_cvttpd_epi32(
        _mm512_div_pd(_mm512_cvtepi32_pd(a), _mm512_cvtepi32_pd(b)));
#else
    return concat(_mm256_cvttpd_epi32(_mm256_div_pd(_mm256_cvtepi32_pd(lo128(a)),
                                                    _mm256_cvtepi32_pd(lo128(b)))),
                  _mm256_cvttpd_epi32(_mm256_div_pd(_mm256_cvtepi32_pd(hi128(a)),
                                                    _mm256_cvtepi32_pd(hi128(b)))));
#endif
}
Vc_INTRINSIC y_u32 Vc_VDECL divides(y_u32 a, y_u32 b) {
#ifdef Vc_HAVE_AVX512F
    return _mm512_cvttpd_epu32(
        _mm512_div_pd(_mm512_cvtepu32_pd(a), _mm512_cvtepu32_pd(b)));
#else
    return concat(
        convert<x_u32>(_mm256_div_pd(convert<y_f64>(lo128(a)), convert<y_f64>(lo128(b)))),
        convert<x_u32>(_mm256_div_pd(convert<y_f64>(hi128(a)), convert<y_f64>(hi128(b)))));
#endif
}
Vc_INTRINSIC y_i16 Vc_VDECL divides(y_i16 a, y_i16 b) {
#ifdef Vc_HAVE_AVX512F
    return convert<y_i16>(_mm512_div_ps(convert<z_f32>(a), convert<z_f32>(b)));
#else
    return convert<y_i16>(
        _mm256_div_ps(convert<y_f32>(lo128(a)), convert<y_f32>(lo128(b))),
        _mm256_div_ps(convert<y_f32>(hi128(a)), convert<y_f32>(hi128(b))));
#endif
}
Vc_INTRINSIC y_u16 Vc_VDECL divides(y_u16 a, y_u16 b) {
#ifdef Vc_HAVE_AVX512F
    return convert<y_u16>(_mm512_div_ps(convert<z_f32>(a), convert<z_f32>(b)));
#else
    return convert<y_u16>(
        _mm256_div_ps(convert<y_f32>(lo128(a)), convert<y_f32>(lo128(b))),
        _mm256_div_ps(convert<y_f32>(hi128(a)), convert<y_f32>(hi128(b))));
#endif
}
Vc_INTRINSIC y_i08 Vc_VDECL divides(y_i08 a, y_i08 b) {
#ifdef Vc_HAVE_AVX512F
    return convert<y_i08>(
        _mm512_div_ps(convert<z_f32>(lo128(a)), convert<z_f32>(lo128(b))),
        _mm512_div_ps(convert<z_f32>(hi128(a)), convert<z_f32>(hi128(b))));
#else
    return convert<y_i08>(
        _mm256_div_ps(convert<y_f32>(lo128(a)), convert<y_f32>(lo128(b))),
        _mm256_div_ps(convert<y_f32>(lo128(_mm256_permute4x64_epi64(a, 0x01))),
                      convert<y_f32>(lo128(_mm256_permute4x64_epi64(b, 0x01)))),
        _mm256_div_ps(convert<y_f32>(lo128(_mm256_permute4x64_epi64(a, 0x02))),
                      convert<y_f32>(lo128(_mm256_permute4x64_epi64(b, 0x02)))),
        _mm256_div_ps(convert<y_f32>(lo128(_mm256_permute4x64_epi64(a, 0x03))),
                      convert<y_f32>(lo128(_mm256_permute4x64_epi64(b, 0x03)))));
#endif
}
Vc_INTRINSIC y_u08 Vc_VDECL divides(y_u08 a, y_u08 b) {
#ifdef Vc_HAVE_AVX512F
    return convert<y_u08>(
        _mm512_div_ps(convert<z_f32>(lo128(a)), convert<z_f32>(lo128(b))),
        _mm512_div_ps(convert<z_f32>(hi128(a)), convert<z_f32>(hi128(b))));
#else
    return convert<y_u08>(
        _mm256_div_ps(convert<y_f32>(lo128(a)), convert<y_f32>(lo128(b))),
        _mm256_div_ps(convert<y_f32>(lo128(_mm256_permute4x64_epi64(a, 0x01))),
                      convert<y_f32>(lo128(_mm256_permute4x64_epi64(b, 0x01)))),
        _mm256_div_ps(convert<y_f32>(lo128(_mm256_permute4x64_epi64(a, 0x02))),
                      convert<y_f32>(lo128(_mm256_permute4x64_epi64(b, 0x02)))),
        _mm256_div_ps(convert<y_f32>(lo128(_mm256_permute4x64_epi64(a, 0x03))),
                      convert<y_f32>(lo128(_mm256_permute4x64_epi64(b, 0x03)))));
#endif
}
#endif// Vc_HAVE_AVX2

// avx512{{{2
#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC z_i32 Vc_VDECL divides(z_i32 a, z_i32 b) {
    return concat(_mm512_cvttpd_epi32(_mm512_div_pd(_mm512_cvtepi32_pd(lo256(a)),
                                                    _mm512_cvtepi32_pd(lo256(b)))),
                  _mm512_cvttpd_epi32(_mm512_div_pd(_mm512_cvtepi32_pd(hi256(a)),
                                                    _mm512_cvtepi32_pd(hi256(b)))));
}
Vc_INTRINSIC z_u32 Vc_VDECL divides(z_u32 a, z_u32 b) {
    return concat(_mm512_cvttpd_epu32(_mm512_div_pd(_mm512_cvtepu32_pd(lo256(a)),
                                                    _mm512_cvtepu32_pd(lo256(b)))),
                  _mm512_cvttpd_epu32(_mm512_div_pd(_mm512_cvtepu32_pd(hi256(a)),
                                                    _mm512_cvtepu32_pd(hi256(b)))));
}
Vc_INTRINSIC z_i16 Vc_VDECL divides(z_i16 a, z_i16 b) {
    return convert<z_i16>(
        _mm512_div_ps(convert<z_f32>(lo256(a)), convert<z_f32>(lo256(b))),
        _mm512_div_ps(convert<z_f32>(hi256(a)), convert<z_f32>(hi256(b))));
}
Vc_INTRINSIC z_u16 Vc_VDECL divides(z_u16 a, z_u16 b) {
    return convert<z_u16>(
        _mm512_div_ps(convert<z_f32>(lo256(a)), convert<z_f32>(lo256(b))),
        _mm512_div_ps(convert<z_f32>(hi256(a)), convert<z_f32>(hi256(b))));
}
Vc_INTRINSIC z_i08 Vc_VDECL divides(z_i08 a, z_i08 b) {
    return convert<z_i08>(
        _mm512_div_ps(convert<z_f32>(lo128(a)), convert<z_f32>(lo128(b))),
        _mm512_div_ps(convert<z_f32>(extract128<1>(a)),
                      convert<z_f32>(extract128<1>(b))),
        _mm512_div_ps(convert<z_f32>(extract128<2>(a)),
                      convert<z_f32>(extract128<2>(b))),
        _mm512_div_ps(convert<z_f32>(extract128<3>(a)),
                      convert<z_f32>(extract128<3>(b))));
}
Vc_INTRINSIC z_u08 Vc_VDECL divides(z_u08 a, z_u08 b) {
    return convert<z_u08>(
        _mm512_div_ps(convert<z_f32>(lo128(a)), convert<z_f32>(lo128(b))),
        _mm512_div_ps(convert<z_f32>(extract128<1>(a)),
                      convert<z_f32>(extract128<1>(b))),
        _mm512_div_ps(convert<z_f32>(extract128<2>(a)),
                      convert<z_f32>(extract128<2>(b))),
        _mm512_div_ps(convert<z_f32>(extract128<3>(a)),
                      convert<z_f32>(extract128<3>(b))));
}
#endif// Vc_HAVE_AVX512F
#endif  // Vc_USE_BUILTIN_VECTOR_TYPES

// modulus{{{1
template <class T, size_t N>
constexpr Vc_INTRINSIC Storage<T, N> modulus(Storage<T, N> a, Storage<T, N> b)
{
    static_assert(std::is_integral<T>::value, "modulus is only supported for integral types");
    return a.d % b.d;
}

// bit_and{{{1
template <size_t N>
Vc_INTRINSIC Storage<float, N> bit_and(Storage<float, N> a, Storage<float, N> b)
{
    return and_(a, b);
}
template <size_t N>
Vc_INTRINSIC Storage<double, N> bit_and(Storage<double, N> a, Storage<double, N> b)
{
    return and_(a, b);
}
template <class T, size_t N>
constexpr Vc_INTRINSIC Storage<T, N> bit_and(Storage<T, N> a, Storage<T, N> b)
{
    static_assert(std::is_integral<T>::value,
                  "bit_and is only supported for integral types");
    return a.d & b.d;
}

// bit_or{{{1
template <size_t N>
Vc_INTRINSIC Storage<float, N> bit_or(Storage<float, N> a, Storage<float, N> b)
{
    return or_(a, b);
}
template <size_t N>
Vc_INTRINSIC Storage<double, N> bit_or(Storage<double, N> a, Storage<double, N> b)
{
    return or_(a, b);
}
template <class T, size_t N>
constexpr Vc_INTRINSIC Storage<T, N> bit_or(Storage<T, N> a, Storage<T, N> b)
{
    static_assert(std::is_integral<T>::value,
                  "bit_or is only supported for integral types");
    return a.d | b.d;
}

// bit_xor{{{1
template <size_t N>
Vc_INTRINSIC Storage<float, N> bit_xor(Storage<float, N> a, Storage<float, N> b)
{
    return xor_(a, b);
}
template <size_t N>
Vc_INTRINSIC Storage<double, N> bit_xor(Storage<double, N> a, Storage<double, N> b)
{
    return xor_(a, b);
}
template <class T, size_t N>
constexpr Vc_INTRINSIC Storage<T, N> bit_xor(Storage<T, N> a, Storage<T, N> b)
{
    static_assert(std::is_integral<T>::value,
                  "bit_xor is only supported for integral types");
    return a.d ^ b.d;
}

// bit_shift_left{{{1
template <class T, size_t N>
Vc_INTRINSIC Storage<T, N> constexpr bit_shift_left(Storage<T, N> a, Storage<T, N> b)
{
    static_assert(std::is_integral<T>::value,
                  "bit_shift_left is only supported for integral types");
    return a.d << b.d;
}
template <class T, size_t N>
Vc_INTRINSIC Storage<T, N> constexpr bit_shift_left(Storage<T, N> a, int b)
{
    static_assert(std::is_integral<T>::value, "bit_shift_left is only supported for integral types");
    return a.d << b;
}

#ifdef Vc_GCC
// GCC needs help to optimize better{{{2
// (cf. https://gcc.gnu.org/bugzilla/show_bug.cgi?id=83894)
#define Vc_SHIFT_LEFT_CONSTEXPR_8(v_, n_)                                                \
    if (__builtin_constant_p(n_)) {                                                      \
        if (n_ == 0) {                                                                   \
            return v_;                                                                   \
        } else if (n_ == 1) {                                                            \
            return v_.d + v_.d;                                                          \
        } else if (n_ > 1 && n_ < 8) {                                                   \
            const uchar mask = (0xff << n_) & 0xff;                                      \
            using V = decltype(v_);                                                      \
            using In = typename V::intrin_type; \
            return reinterpret_cast<In>(storage_bitcast<ushort>(v_).d << n_) &                 \
                   V::broadcast(mask).intrin();                                          \
        } else {                                                                         \
            return detail::warn_ub(v_);                                                  \
        }                                                                                \
    }                                                                                    \
    Vc_NOTHING_EXPECTING_SEMICOLON

// SSE 8-bit value_type << int {{{3
Vc_INTRINSIC x_u08 bit_shift_left(x_u08 a, int b)
{
    Vc_SHIFT_LEFT_CONSTEXPR_8(a, b);
#if defined Vc_HAVE_AVX512BW && defined Vc_HAVE_AVX512VL
    return _mm256_cvtepi16_epi8(reinterpret_cast<__m256i>(
        reinterpret_cast<builtin_type_t<ushort, 16>>(_mm256_cvtepi8_epi16(a)) << b));
#else
    using vshort = builtin_type_t<ushort, 8>;
    const auto mask = ((~vshort() >> 8) << b) ^ (~vshort() << 8);
    return detail::to_storage((reinterpret_cast<vshort>(a.d) << b) & mask);
#endif
}
Vc_INTRINSIC x_i08 bit_shift_left(x_i08 a, int b)
{
    return storage_bitcast<schar>(bit_shift_left(storage_bitcast<uchar>(a), b));
}
Vc_INTRINSIC x_chr bit_shift_left(x_chr a, int b)
{
    return storage_bitcast<char>(bit_shift_left(storage_bitcast<uchar>(a), b));
}

// AVX 8-bit value_type << int {{{3
#ifdef Vc_HAVE_AVX2
Vc_INTRINSIC Storage<uchar, 32> bit_shift_left(Storage<uchar, 32> a, int b)
{
    Vc_SHIFT_LEFT_CONSTEXPR_8(a, b);
#if defined Vc_HAVE_AVX512BW
    return _mm512_cvtepi16_epi8(reinterpret_cast<__m512i>(
        reinterpret_cast<builtin_type_t<ushort, 32>>(_mm512_cvtepi8_epi16(a)) << b));
#else
    using vshort = builtin_type_t<ushort, 16>;
    const auto mask = ((~vshort() >> 8) << b) ^ (~vshort() << 8);
    return detail::to_storage((reinterpret_cast<vshort>(a.d) << b) & mask);
#endif
}
Vc_INTRINSIC Storage<schar, 32> bit_shift_left(Storage<schar, 32> a, int b)
{
    return storage_bitcast<schar>(bit_shift_left(storage_bitcast<uchar>(a), b));
}
Vc_INTRINSIC Storage< char, 32> bit_shift_left(Storage< char, 32> a, int b)
{
    return storage_bitcast<char>(bit_shift_left(storage_bitcast<uchar>(a), b));
}
#endif  // Vc_HAVE_AVX2

// AVX512 8-bit value_type << int {{{3
#ifdef Vc_HAVE_AVX512BW
Vc_INTRINSIC z_u08 bit_shift_left(z_u08 a, int b)
{
    Vc_SHIFT_LEFT_CONSTEXPR_8(a, b);
    using vshort = builtin_type_t<ushort, 32>;
    const auto mask = ((~vshort() >> 8) << b) ^ (~vshort() << 8);
    return detail::to_storage((reinterpret_cast<vshort>(a.d) << b) & mask);
}
Vc_INTRINSIC Storage<schar, 64> bit_shift_left(Storage<schar, 64> a, int b)
{
    return storage_bitcast<schar>(bit_shift_left(storage_bitcast<uchar>(a), b));
}
Vc_INTRINSIC Storage< char, 64> bit_shift_left(Storage< char, 64> a, int b)
{
    return storage_bitcast<char>(bit_shift_left(storage_bitcast<uchar>(a), b));
}
#endif  // Vc_HAVE_AVX512BW

#undef Vc_SHIFT_LEFT_CONSTEXPR_8

// simd << simd (improvements/specializations without AVX2) {{{3
#ifndef Vc_HAVE_AVX2
Vc_INTRINSIC x_i16 bit_shift_left(x_i16 a, x_i16 b)
{
    builtin_type_t<int, 4> shift = storage_bitcast<int>(b).d + (0x03f8'03f8 >> 3);
    return multiplies(
        a, x_i16(_mm_cvttps_epi32(reinterpret_cast<__m128>(shift << 23)) |
                 (_mm_cvttps_epi32(reinterpret_cast<__m128>(shift >> 16 << 23)) << 16)));
}
Vc_INTRINSIC x_u16 bit_shift_left(x_u16 a, x_u16 b)
{
    return storage_bitcast<unsigned short>(
        bit_shift_left(storage_bitcast<short>(a), storage_bitcast<short>(b)));
}

Vc_INTRINSIC x_i32 bit_shift_left(x_i32 a, x_i32 b)
{
    return multiplies(
        a, x_i32(_mm_cvttps_epi32(reinterpret_cast<__m128>((b.d << 23) + 0x3f80'0000))));
}
Vc_INTRINSIC x_u32 bit_shift_left(x_u32 a, x_u32 b)
{
    return multiplies(
        a, x_u32(_mm_cvttps_epi32(reinterpret_cast<__m128>((b.d << 23) + 0x3f80'0000))));
}

Vc_INTRINSIC x_i64 bit_shift_left(x_i64 a, x_i64 b)
{
    const auto lo = _mm_sll_epi64(a, b);
    const auto hi = _mm_sll_epi64(a, _mm_unpackhi_epi64(b, b));
#ifdef Vc_HAVE_SSE4_1
    return _mm_blend_epi16(lo, hi, 0xf0);
#else
    //return make_storage<llong>(reinterpret_cast<builtin_type_t<llong, 2>>(lo)[0], reinterpret_cast<builtin_type_t<llong, 2>>(hi)[1]);
    return to_storage(_mm_move_sd(intrin_cast<__m128d>(hi), intrin_cast<__m128d>(lo)));
#endif
}
Vc_INTRINSIC x_u64 bit_shift_left(x_u64 a, x_u64 b)
{
    return storage_bitcast<ullong>(bit_shift_left(storage_bitcast<llong>(a), storage_bitcast<llong>(b)));
}

#endif  // !Vc_HAVE_AVX2

// simd << simd (improvements/specializations with AVX2 and up) {{{3
#ifdef Vc_HAVE_AVX2
Vc_INTRINSIC x_i64 bit_shift_left(x_i64 a, x_i64 b) { return _mm_sllv_epi64(a, b); }
Vc_INTRINSIC x_u64 bit_shift_left(x_u64 a, x_u64 b) { return _mm_sllv_epi64(a, b); }
Vc_INTRINSIC x_i32 bit_shift_left(x_i32 a, x_i32 b) { return _mm_sllv_epi32(a, b); }
Vc_INTRINSIC x_u32 bit_shift_left(x_u32 a, x_u32 b) { return _mm_sllv_epi32(a, b); }

Vc_INTRINSIC y_i64 bit_shift_left(y_i64 a, y_i64 b) { return _mm256_sllv_epi64(a, b); }
Vc_INTRINSIC y_u64 bit_shift_left(y_u64 a, y_u64 b) { return _mm256_sllv_epi64(a, b); }
Vc_INTRINSIC y_i32 bit_shift_left(y_i32 a, y_i32 b) { return _mm256_sllv_epi32(a, b); }
Vc_INTRINSIC y_u32 bit_shift_left(y_u32 a, y_u32 b) { return _mm256_sllv_epi32(a, b); }

#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC z_i64 bit_shift_left(z_i64 a, z_i64 b) { return _mm512_sllv_epi64(a, b); }
Vc_INTRINSIC z_u64 bit_shift_left(z_u64 a, z_u64 b) { return _mm512_sllv_epi64(a, b); }
Vc_INTRINSIC z_i32 bit_shift_left(z_i32 a, z_i32 b) { return _mm512_sllv_epi32(a, b); }
Vc_INTRINSIC z_u32 bit_shift_left(z_u32 a, z_u32 b) { return _mm512_sllv_epi32(a, b); }
#endif  // Vc_HAVE_AVX512F

#ifdef Vc_HAVE_AVX512BW
Vc_INTRINSIC z_i16 bit_shift_left(z_i16 a, z_i16 b) { return _mm512_sllv_epi16(a, b); }
Vc_INTRINSIC z_u16 bit_shift_left(z_u16 a, z_u16 b) { return _mm512_sllv_epi16(a, b); }
Vc_INTRINSIC y_i08 bit_shift_left(y_i08 a, y_i08 b) { return _mm512_cvtepi16_epi8(_mm512_sllv_epi16(_mm512_cvtepi8_epi16(a), _mm512_cvtepi8_epi16(b))); }
Vc_INTRINSIC y_u08 bit_shift_left(y_u08 a, y_u08 b) { return _mm512_cvtepi16_epi8(_mm512_sllv_epi16(_mm512_cvtepi8_epi16(a), _mm512_cvtepi8_epi16(b))); }
Vc_INTRINSIC z_i08 bit_shift_left(z_i08 a, z_i08 b) { return concat(bit_shift_left(lo256(a), lo256(b)), bit_shift_left(hi256(a), hi256(b))); }
Vc_INTRINSIC z_u08 bit_shift_left(z_u08 a, z_u08 b) { return concat(bit_shift_left(lo256(a), lo256(b)), bit_shift_left(hi256(a), hi256(b))); }
#ifdef Vc_HAVE_AVX512VL
Vc_INTRINSIC y_i16 bit_shift_left(y_i16 a, y_i16 b) { return _mm256_sllv_epi16(a, b); }
Vc_INTRINSIC y_u16 bit_shift_left(y_u16 a, y_u16 b) { return _mm256_sllv_epi16(a, b); }
Vc_INTRINSIC x_i16 bit_shift_left(x_i16 a, x_i16 b) { return _mm_sllv_epi16(a, b); }
Vc_INTRINSIC x_u16 bit_shift_left(x_u16 a, x_u16 b) { return _mm_sllv_epi16(a, b); }
Vc_INTRINSIC x_i08 bit_shift_left(x_i08 a, x_i08 b) { return _mm256_cvtepi16_epi8(_mm256_sllv_epi16(_mm256_cvtepi8_epi16(a), _mm256_cvtepi8_epi16(b))); }
Vc_INTRINSIC x_u08 bit_shift_left(x_u08 a, x_u08 b) { return _mm256_cvtepi16_epi8(_mm256_sllv_epi16(_mm256_cvtepi8_epi16(a), _mm256_cvtepi8_epi16(b))); }
#else   // Vc_HAVE_AVX512VL
Vc_INTRINSIC y_i16 bit_shift_left(y_i16 a, y_i16 b) { return lo256(_mm512_sllv_epi16(intrin_cast<__m512i>(a), intrin_cast<__m512i>(b)); }
Vc_INTRINSIC y_u16 bit_shift_left(y_u16 a, y_u16 b) { return lo256(_mm512_sllv_epi16(intrin_cast<__m512i>(a), intrin_cast<__m512i>(b)); }
Vc_INTRINSIC x_i16 bit_shift_left(x_i16 a, x_i16 b) { return lo128(_mm512_sllv_epi16(intrin_cast<__m512i>(a), intrin_cast<__m512i>(b)); }
Vc_INTRINSIC x_u16 bit_shift_left(x_u16 a, x_u16 b) { return lo128(_mm512_sllv_epi16(intrin_cast<__m512i>(a), intrin_cast<__m512i>(b)); }
Vc_INTRINSIC x_i08 bit_shift_left(x_i08 a, x_i08 b) { return lo128(bit_shift_left(y_i08(intrin_cast<__m256i>(a)), y_i08(intrin_cast<__m256i>(b)))); }
Vc_INTRINSIC x_u08 bit_shift_left(x_u08 a, x_u08 b) { return lo128(bit_shift_left(y_u08(intrin_cast<__m256i>(a)), y_u08(intrin_cast<__m256i>(b)))); }
#endif  // Vc_HAVE_AVX512VL
#else   // Vc_HAVE_AVX512BW
Vc_INTRINSIC x_i16 bit_shift_left(x_i16 a, x_i16 b)
{
    // left shift into or over the sign bit is UB => OR suffices
    return _mm_blend_epi16(
        _mm_sllv_epi32(a, and_(b, broadcast(0x0000ffffu))),
        _mm_sllv_epi32(and_(a, broadcast(0xffff0000u)), _mm_srli_epi32(b, 16)), 0xaa);
}
Vc_INTRINSIC x_u16 bit_shift_left(x_u16 a, x_u16 b)
{
    return _mm_blend_epi16(
        _mm_sllv_epi32(a, and_(b, broadcast(0x0000ffffu))),
        _mm_sllv_epi32(and_(a, broadcast(0xffff0000u)), _mm_srli_epi32(b, 16)), 0xaa);
}
Vc_INTRINSIC x_i08 bit_shift_left(x_i08 a, x_i08 b)
{
    // exploit UB: The behavior is undefined if the right operand is [...] greater than or
    // equal to the length in bits of the promoted left operand.
    // left shift into or over the sign bit is UB => never spills into a neighbor
    // => valid input range for each element of b is [0, 7]
    // => only the 3 low bits of b are relevant
    // do a =<< 4 where b[2] is set
    a = _mm_blendv_epi8(a, slli_epi16<4>(a), slli_epi16<5>(b));
    // do a =<< 2 where b[1] is set
    a = _mm_blendv_epi8(a, slli_epi16<2>(a), slli_epi16<6>(b));
    // do a =<< 1 where b[0] is set
    return _mm_blendv_epi8(a, _mm_add_epi8(a, a), slli_epi16<7>(b));
}
Vc_INTRINSIC x_u08 bit_shift_left(x_u08 a, x_u08 b)
{
    // exploit UB: The behavior is undefined if the right operand is [...] greater than or
    // equal to the length in bits of the promoted left operand.
    // => valid input range for each element of b is [0, 7]
    // => only the 3 low bits of b are relevant
    // do a =<< 4 where b[2] is set
    a = _mm_blendv_epi8(a, and_(slli_epi16<4>(a), broadcast(0xf0f0f0f0u)),
                        slli_epi16<5>(b));
    // do a =<< 2 where b[1] is set
    a = _mm_blendv_epi8(a, and_(slli_epi16<2>(a), broadcast(0xfcfcfcfcu)),
                        slli_epi16<6>(b));
    // do a =<< 1 where b[0] is set
    return _mm_blendv_epi8(a, _mm_add_epi8(a, a), slli_epi16<7>(b));
}

Vc_INTRINSIC y_i16 bit_shift_left(y_i16 a, y_i16 b)
{
    // left shift into or over the sign bit is UB => OR suffices
    return _mm256_blend_epi16(
        _mm256_sllv_epi32(a, and_(b, broadcast32(0x0000ffffu))),
        _mm256_sllv_epi32(and_(a, broadcast32(0xffff0000u)), _mm256_srli_epi32(b, 16)),
        0xaa);
}
Vc_INTRINSIC y_u16 bit_shift_left(y_u16 a, y_u16 b)
{
    return _mm256_blend_epi16(
        _mm256_sllv_epi32(a, and_(b, broadcast32(0x0000ffffu))),
        _mm256_sllv_epi32(and_(a, broadcast32(0xffff0000u)), _mm256_srli_epi32(b, 16)),
        0xaa);
}
Vc_INTRINSIC y_i08 bit_shift_left(y_i08 a, y_i08 b)
{
    // exploit UB: The behavior is undefined if the right operand is [...] greater than or
    // equal to the length in bits of the promoted left operand.
    // left shift into or over the sign bit is UB => never spills into a neighbor
    // => valid input range for each element of b is [0, 7]
    // => only the 3 low bits of b are relevant
    // do a =<< 4 where b[2] is set
    a = _mm256_blendv_epi8(a, slli_epi16<4>(a), slli_epi16<5>(b));
    // do a =<< 2 where b[1] is set
    a = _mm256_blendv_epi8(a, slli_epi16<2>(a), slli_epi16<6>(b));
    // do a =<< 1 where b[0] is set
    return _mm256_blendv_epi8(a, _mm256_add_epi8(a, a), slli_epi16<7>(b));
}
Vc_INTRINSIC y_u08 bit_shift_left(y_u08 a, y_u08 b)
{
    // exploit UB: The behavior is undefined if the right operand is [...] greater than or
    // equal to the length in bits of the promoted left operand.
    // => valid input range for each element of b is [0, 7]
    // => only the 3 low bits of b are relevant
    // do a =<< 4 where b[2] is set
    a = _mm256_blendv_epi8(a, and_(slli_epi16<4>(a), broadcast32(0xf0f0f0f0u)),
                           slli_epi16<5>(b));
    // do a =<< 2 where b[1] is set
    a = _mm256_blendv_epi8(a, and_(slli_epi16<2>(a), broadcast32(0xfcfcfcfcu)),
                           slli_epi16<6>(b));
    // do a =<< 1 where b[0] is set
    return _mm256_blendv_epi8(a, _mm256_add_epi8(a, a), slli_epi16<7>(b));
}

/* Not actually part of simd_abi::avx512:
Vc_INTRINSIC z_i16 bit_shift_left(z_i16 a, z_i16 b) { return or_(_mm512_sllv_epi32(and_(a, broadcast64(0x0000ffffu)), and_(b, broadcast64(0x0000ffffu))), _mm512_sllv_epi32(and_(a, broadcast64(0xffff0000u)), _mm512_srli_epi32(b, 16))); }
Vc_INTRINSIC z_u16 bit_shift_left(z_u16 a, z_u16 b) { return or_(_mm512_sllv_epi32(and_(a, broadcast64(0x0000ffffu)), and_(b, broadcast64(0x0000ffffu))), _mm512_sllv_epi32(and_(a, broadcast64(0xffff0000u)), _mm512_srli_epi32(b, 16))); }
Vc_INTRINSIC z_i08 bit_shift_left(z_i08 a, z_i08 b)
{
    return or_(
        or_(_mm512_sllv_epi32(and_(a, broadcast64(0x000000ffu)),
                              and_(b, broadcast64(0x000000ffu))),
            _mm512_sllv_epi32(and_(a, broadcast64(0x0000ff00u)),
                              _mm512_srli_epi32(and_(b, broadcast64(0x0000ff00u)), 8))),
        or_(_mm512_sllv_epi32(and_(a, broadcast64(0x00ff0000u)),
                              _mm512_srli_epi32(and_(b, broadcast64(0x00ff0000u)), 16)),
            _mm512_sllv_epi32(and_(a, broadcast64(0xff000000u)),
                              _mm512_srli_epi32(b, 24))));
}
Vc_INTRINSIC z_u08 bit_shift_left(z_u08 a, z_u08 b) { return z_u08(bit_shift_left(z_i08(a), z_i08(b))); }
*/
#endif  // Vc_HAVE_AVX512BW
#endif  // Vc_HAVE_AVX2
//}}}

Vc_BINARY_OVERLOAD_SAME_VALUE_REP_(bit_shift_left)

#endif  // Vc_GCC

// bit_shift_right{{{1
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
template <class T, size_t N>
Vc_INTRINSIC Storage<T, N> bit_shift_right(Storage<T, N> a, Storage<T, N> b)
{
    static_assert(std::is_integral<T>::value,
                  "bit_shift_right is only supported for integral types");
    return a.d >> b.d;
}
template <class T, size_t N>
Vc_INTRINSIC Storage<T, N> bit_shift_right(Storage<T, N> a, int b)
{
    static_assert(std::is_integral<T>::value,
                  "bit_shift_right is only supported for integral types");
    return a.d >> b;
}
#else   // Vc_USE_BUILTIN_VECTOR_TYPES

// improvements/specializations with newer instruction set extensions
#ifdef Vc_HAVE_AVX2
Vc_INTRINSIC x_i32 bit_shift_right(x_i32 a, x_i32 b) { return _mm_srav_epi32(a, b); }
Vc_INTRINSIC x_u32 bit_shift_right(x_u32 a, x_u32 b) { return _mm_srlv_epi32(a, b); }

Vc_INTRINSIC y_i32 bit_shift_right(y_i32 a, y_i32 b) { return _mm256_srav_epi32(a, b); }
Vc_INTRINSIC y_u32 bit_shift_right(y_u32 a, y_u32 b) { return _mm256_srlv_epi32(a, b); }

#ifdef Vc_HAVE_AVX512VL
Vc_INTRINSIC x_i64 bit_shift_right(x_i64 a, x_i64 b) { return _mm_srav_epi64(a, b); }
Vc_INTRINSIC x_u64 bit_shift_right(x_u64 a, x_u64 b) { return _mm_srlv_epi64(a, b); }

Vc_INTRINSIC y_i64 bit_shift_right(y_i64 a, y_i64 b) { return _mm256_srav_epi64(a, b); }
Vc_INTRINSIC y_u64 bit_shift_right(y_u64 a, y_u64 b) { return _mm256_srlv_epi64(a, b); }
#endif  // Vc_HAVE_AVX512VL

#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC z_i64 bit_shift_right(z_i64 a, z_i64 b) { return _mm512_srav_epi64(a, b); }
Vc_INTRINSIC z_u64 bit_shift_right(z_u64 a, z_u64 b) { return _mm512_srlv_epi64(a, b); }
Vc_INTRINSIC z_i32 bit_shift_right(z_i32 a, z_i32 b) { return _mm512_srav_epi32(a, b); }
Vc_INTRINSIC z_u32 bit_shift_right(z_u32 a, z_u32 b) { return _mm512_srlv_epi32(a, b); }
#endif  // Vc_HAVE_AVX512F

#ifdef Vc_HAVE_AVX512BW
Vc_INTRINSIC z_i16 bit_shift_right(z_i16 a, z_i16 b) { return _mm512_srav_epi16(a, b); }
Vc_INTRINSIC z_u16 bit_shift_right(z_u16 a, z_u16 b) { return _mm512_srlv_epi16(a, b); }
Vc_INTRINSIC y_i08 bit_shift_right(y_i08 a, y_i08 b) { return _mm512_cvtepi16_epi8(_mm512_srav_epi16(_mm512_cvtepi8_epi16(a), _mm512_cvtepi8_epi16(b))); }
Vc_INTRINSIC y_u08 bit_shift_right(y_u08 a, y_u08 b)
{
    return _mm512_cvtepi16_epi8(
        _mm512_srlv_epi16(_mm512_cvtepu8_epi16(a), _mm512_cvtepu8_epi16(b)));
}
Vc_INTRINSIC z_i08 bit_shift_right(z_i08 a, z_i08 b) { return concat(bit_shift_right(lo256(a), lo256(b)), bit_shift_right(hi256(a), hi256(b))); }
Vc_INTRINSIC z_u08 bit_shift_right(z_u08 a, z_u08 b) { return concat(bit_shift_right(lo256(a), lo256(b)), bit_shift_right(hi256(a), hi256(b))); }
#ifdef Vc_HAVE_AVX512VL
Vc_INTRINSIC y_i16 bit_shift_right(y_i16 a, y_i16 b) { return _mm256_srav_epi16(a, b); }
Vc_INTRINSIC y_u16 bit_shift_right(y_u16 a, y_u16 b) { return _mm256_srlv_epi16(a, b); }
#else   // Vc_HAVE_AVX512VL
Vc_INTRINSIC y_i16 bit_shift_right(y_i16 a, y_i16 b) { return lo256(_mm512_srav_epi16(intrin_cast<__m512i>(a), intrin_cast<__m512i>(b)); }
Vc_INTRINSIC y_u16 bit_shift_right(y_u16 a, y_u16 b) { return lo256(_mm512_srlv_epi16(intrin_cast<__m512i>(a), intrin_cast<__m512i>(b)); }
#endif  // Vc_HAVE_AVX512VL
#else   // Vc_HAVE_AVX512BW

Vc_INTRINSIC y_i16 bit_shift_right(y_i16 a, y_i16 b)
{
    auto lo32 = _mm256_srli_epi32(
        _mm256_srav_epi32(_mm256_unpacklo_epi16(_mm256_setzero_si256(), a),
                          _mm256_unpacklo_epi16(b, _mm256_setzero_si256())),
        16);
    auto hi32 = _mm256_srli_epi32(
        _mm256_srav_epi32(_mm256_unpackhi_epi16(_mm256_setzero_si256(), a),
                          _mm256_unpackhi_epi16(b, _mm256_setzero_si256())),
        16);
    return _mm256_packs_epi32(lo32, hi32);
}
Vc_INTRINSIC y_u16 bit_shift_right(y_u16 a, y_u16 b)
{
    return _mm256_blend_epi16(_mm256_srlv_epi32(and_(a, broadcast32(0x0000ffffu)),
                                                and_(b, broadcast32(0x0000ffffu))),
                              _mm256_srlv_epi32(a, _mm256_srli_epi32(b, 16)), 0xaa);
}

Vc_INTRINSIC y_i08 bit_shift_right(y_i08 a, y_i08 b)
{
    // exploit UB: The behavior is undefined if the right operand is [...] greater than or
    // equal to the length in bits of the promoted left operand.
    // => valid input range for each element of b is [0, 7]
    // => only the 3 low bits of b are relevant
    // do a =<< 4 where b[2] is set
    return convert<y_i08>(
        _mm256_srav_epi32(_mm256_cvtepi8_epi32(lo128(a)), _mm256_cvtepi8_epi32(lo128(b))),
        _mm256_srav_epi32(_mm256_cvtepi8_epi32(_mm_unpackhi_epi64(lo128(a), lo128(a))),
                          _mm256_cvtepi8_epi32(_mm_unpackhi_epi64(lo128(b), lo128(b)))),
        _mm256_srav_epi32(_mm256_cvtepi8_epi32(hi128(a)), _mm256_cvtepi8_epi32(hi128(b))),
        _mm256_srav_epi32(_mm256_cvtepi8_epi32(_mm_unpackhi_epi64(hi128(a), hi128(a))),
                          _mm256_cvtepi8_epi32(_mm_unpackhi_epi64(hi128(b), hi128(b)))));
}
Vc_INTRINSIC y_u08 bit_shift_right(y_u08 a, y_u08 b)
{
    // exploit UB: The behavior is undefined if the right operand is [...] greater than or
    // equal to the length in bits of the promoted left operand.
    // => valid input range for each element of b is [0, 7]
    // => only the 3 low bits of b are relevant
    // do a =<< 4 where b[2] is set
    a = _mm256_blendv_epi8(a, and_(srli_epi16<4>(a), broadcast32(0x0f0f0f0fu)),
                           slli_epi16<5>(b));
    // do a =<< 2 where b[1] is set
    a = _mm256_blendv_epi8(a, and_(srli_epi16<2>(a), broadcast32(0x3f3f3f3fu)),
                           slli_epi16<6>(b));
    // do a =<< 1 where b[0] is set
    return _mm256_blendv_epi8(a, and_(srli_epi16<1>(a), broadcast32(0x7f7f7f7f)),
                              slli_epi16<7>(b));
}

/* Not actually part of simd_abi::avx512:
Vc_INTRINSIC z_i16 bit_shift_right(z_i16 a, z_i16 b) { return or_(_mm512_sllv_epi32(and_(a, broadcast64(0x0000ffffu)), and_(b, broadcast64(0x0000ffffu))), _mm512_sllv_epi32(and_(a, broadcast64(0xffff0000u)), _mm512_srli_epi32(b, 16))); }
Vc_INTRINSIC z_u16 bit_shift_right(z_u16 a, z_u16 b) { return or_(_mm512_sllv_epi32(and_(a, broadcast64(0x0000ffffu)), and_(b, broadcast64(0x0000ffffu))), _mm512_sllv_epi32(and_(a, broadcast64(0xffff0000u)), _mm512_srli_epi32(b, 16))); }
Vc_INTRINSIC z_i08 bit_shift_right(z_i08 a, z_i08 b)
{
    return or_(
        or_(_mm512_sllv_epi32(and_(a, broadcast64(0x000000ffu)),
                              and_(b, broadcast64(0x000000ffu))),
            _mm512_sllv_epi32(and_(a, broadcast64(0x0000ff00u)),
                              _mm512_srli_epi32(and_(b, broadcast64(0x0000ff00u)), 8))),
        or_(_mm512_sllv_epi32(and_(a, broadcast64(0x00ff0000u)),
                              _mm512_srli_epi32(and_(b, broadcast64(0x00ff0000u)), 16)),
            _mm512_sllv_epi32(and_(a, broadcast64(0xff000000u)),
                              _mm512_srli_epi32(b, 24))));
}
Vc_INTRINSIC z_u08 bit_shift_right(z_u08 a, z_u08 b) { return z_u08(bit_shift_right(z_i08(a), z_i08(b))); }
*/
#endif  // Vc_HAVE_AVX512BW
#else  // no AVX2

Vc_INTRINSIC x_i32 bit_shift_right(x_i32 a, x_i32 b)
{
    const auto r0 = _mm_sra_epi32(a, _mm_unpacklo_epi32(b, _mm_setzero_si128()));
    const auto r1 = _mm_sra_epi32(a, _mm_srli_epi64(b, 32));
    const auto r2 = _mm_sra_epi32(a, _mm_unpackhi_epi32(b, _mm_setzero_si128()));
    const auto r3 = _mm_sra_epi32(a, _mm_srli_si128(b, 12));
#ifdef Vc_HAVE_SSE4_1
    return _mm_blend_epi16(_mm_blend_epi16(r1, r0, 0x3), _mm_blend_epi16(r3, r2, 0x30),
                           0xf0);
#else   // Vc_HAVE_SSE4_1
    return _mm_unpacklo_epi64(_mm_unpacklo_epi32(r0, _mm_srli_si128(r1, 4)),
                              _mm_unpackhi_epi32(r2, _mm_srli_si128(r3, 4)));
#endif  // Vc_HAVE_SSE4_1
}
Vc_INTRINSIC x_u32 bit_shift_right(x_u32 a, x_u32 b)
{
    const auto r0 = _mm_srl_epi32(a, _mm_unpacklo_epi32(b, _mm_setzero_si128()));
    const auto r1 = _mm_srl_epi32(a, _mm_srli_epi64(b, 32));
    const auto r2 = _mm_srl_epi32(a, _mm_unpackhi_epi32(b, _mm_setzero_si128()));
    const auto r3 = _mm_srl_epi32(a, _mm_srli_si128(b, 12));
#ifdef Vc_HAVE_SSE4_1
    return _mm_blend_epi16(_mm_blend_epi16(r1, r0, 0x3), _mm_blend_epi16(r3, r2, 0x30),
                           0xf0);
#else   // Vc_HAVE_SSE4_1
    return _mm_unpacklo_epi64(_mm_unpacklo_epi32(r0, _mm_srli_si128(r1, 4)),
                              _mm_unpackhi_epi32(r2, _mm_srli_si128(r3, 4)));
#endif  // Vc_HAVE_SSE4_1
}

#endif  // Vc_HAVE_AVX2

// x_i16 and x_u16 {{{2
#if defined Vc_HAVE_AVX512BW && defined Vc_HAVE_AVX512VL

Vc_INTRINSIC x_i16 bit_shift_right(x_i16 a, x_i16 b) { return _mm_srav_epi16(a, b); }
Vc_INTRINSIC x_u16 bit_shift_right(x_u16 a, x_u16 b) { return _mm_srlv_epi16(a, b); }

#elif defined Vc_HAVE_AVX512BW

Vc_INTRINSIC x_i16 bit_shift_right(x_i16 a, x_i16 b)
{
    return lo128(_mm512_srav_epi16(intrin_cast<__m512i>(a), intrin_cast<__m512i>(b));
}
Vc_INTRINSIC x_u16 bit_shift_right(x_u16 a, x_u16 b)
{
    return lo128(_mm512_srlv_epi16(intrin_cast<__m512i>(a), intrin_cast<__m512i>(b));
}

#elif defined Vc_HAVE_AVX2

// _mm256_sr[al]v_epi32 require AVX2
Vc_INTRINSIC x_i16 bit_shift_right(x_i16 a, x_i16 b)
{
    return x86::convert_to<x_i16>(
        y_i32(_mm256_srav_epi32(convert_to<y_i32>(a), convert_to<y_i32>(b))));
}
Vc_INTRINSIC x_u16 bit_shift_right(x_u16 a, x_u16 b)
{
    return x86::convert_to<x_u16>(
        y_u32(_mm256_srlv_epi32(convert_to<y_u32>(a), convert_to<y_u32>(b))));
}

#elif defined Vc_HAVE_SSE4_1

Vc_INTRINSIC x_i16 bit_shift_right(x_i16 a, x_i16 b)
{
    // exploit UB: The behavior is undefined if the right operand is [...] greater than or
    // equal to the length in bits of the promoted left operand.
    // => valid input range for each element of b is [0, 15]
    // => only the 4 low bits of b are relevant
    // shift by 4 and duplicate to high byte
    b = or_(_mm_slli_epi32(b, 4), _mm_slli_epi32(b, 12));
    //b = multiplies(b, broadcast16(0x10101010));
    // do a =>> 8 where b[3] is set
    a = _mm_blendv_epi8(a, _mm_srai_epi16(a, 8), b);
    // do a =>> 4 where b[2] is set
    a = _mm_blendv_epi8(a, _mm_srai_epi16(a, 4), b = _mm_add_epi16(b, b));
    // do a =>> 2 where b[1] is set
    a = _mm_blendv_epi8(a, _mm_srai_epi16(a, 2), b = _mm_add_epi16(b, b));
    // do a =>> 1 where b[0] is set
    return _mm_blendv_epi8(a, _mm_srai_epi16(a, 1), _mm_add_epi16(b, b));
}
Vc_INTRINSIC x_u16 bit_shift_right(x_u16 a, x_u16 b)
{
    // exploit UB: The behavior is undefined if the right operand is [...] greater than or
    // equal to the length in bits of the promoted left operand.
    // => valid input range for each element of b is [0, 15]
    // => only the 4 low bits of b are relevant
    // shift by 4 and duplicate to high byte
    b = or_(slli_epi16<4>(b), slli_epi16<12>(b));
    //b = multiplies(b, broadcast16(0x10101010));
    // do a =>> 8 where b[3] is set
    a = _mm_blendv_epi8(a, srli_epi16<8>(a), b);
    // do a =>> 4 where b[2] is set
    a = _mm_blendv_epi8(a, srli_epi16<4>(a), b = _mm_add_epi16(b, b));
    // do a =>> 2 where b[1] is set
    a = _mm_blendv_epi8(a, srli_epi16<2>(a), b = _mm_add_epi16(b, b));
    // do a =>> 1 where b[0] is set
    return _mm_blendv_epi8(a, srli_epi16<1>(a), _mm_add_epi16(b, b));
}

#else   // no SSE4 (_mm_blendv_epi8)

Vc_INTRINSIC x_i16 bit_shift_right(x_i16 a, x_i16 b)
{
    // exploit UB: The behavior is undefined if the right operand is [...] greater than or
    // equal to the length in bits of the promoted left operand.
    // => valid input range for each element of b is [0, 15]
    // => only the 4 low bits of b are relevant
    // do a =>> 8 where b[3] is set
    a = blendv_epi8(a, _mm_srai_epi16(a, 8), _mm_cmpgt_epi16(b, broadcast16(0x00070007)));
    // do a =>> 4 where b[2] is set
    a = blendv_epi8(
        a, _mm_srai_epi16(a, 4),
        _mm_cmpgt_epi16(and_(b, broadcast16(0x00040004)), _mm_setzero_si128()));
    // do a =>> 2 where b[1] is set
    a = blendv_epi8(
        a, _mm_srai_epi16(a, 2),
        _mm_cmpgt_epi16(and_(b, broadcast16(0x00020002)), _mm_setzero_si128()));
    // do a =>> 1 where b[0] is set
    return blendv_epi8(
        a, _mm_srai_epi16(a, 1),
        _mm_cmpgt_epi16(and_(b, broadcast16(0x00010001)), _mm_setzero_si128()));
}
Vc_INTRINSIC x_u16 bit_shift_right(x_u16 a, x_u16 b)
{
    // exploit UB: The behavior is undefined if the right operand is [...] greater than or
    // equal to the length in bits of the promoted left operand.
    // => valid input range for each element of b is [0, 15]
    // => only the 4 low bits of b are relevant
    // do a =>> 8 where b[3] is set
    a = blendv_epi8(a, srli_epi16<8>(a), _mm_cmpgt_epi16(b, broadcast16(0x00070007)));
    // do a =>> 4 where b[2] is set
    a = blendv_epi8(
        a, srli_epi16<4>(a),
        _mm_cmpgt_epi16(and_(b, broadcast16(0x00040004)), _mm_setzero_si128()));
    // do a =>> 2 where b[1] is set
    a = blendv_epi8(
        a, srli_epi16<2>(a),
        _mm_cmpgt_epi16(and_(b, broadcast16(0x00020002)), _mm_setzero_si128()));
    // do a =>> 1 where b[0] is set
    return blendv_epi8(
        a, srli_epi16<1>(a),
        _mm_cmpgt_epi16(and_(b, broadcast16(0x00010001)), _mm_setzero_si128()));
}

#endif  // Vc_HAVE_AVX512BW

// x_i08 and x_u08 {{{2
#if defined Vc_HAVE_AVX512BW && defined Vc_HAVE_AVX512VL

Vc_INTRINSIC x_i08 bit_shift_right(x_i08 a, x_i08 b)
{
    return _mm256_cvtepi16_epi8(
        _mm256_srav_epi16(_mm256_cvtepi8_epi16(a), _mm256_cvtepi8_epi16(b)));
}
Vc_INTRINSIC x_u08 bit_shift_right(x_u08 a, x_u08 b)
{
    return _mm256_cvtepi16_epi8(
        _mm256_srlv_epi16(_mm256_cvtepu8_epi16(a), _mm256_cvtepu8_epi16(b)));
}

#elif defined Vc_HAVE_AVX512BW

Vc_INTRINSIC x_i08 bit_shift_right(x_i08 a, x_i08 b)
{
    return lo128(
        bit_shift_right(y_i08(intrin_cast<__m256i>(a)), y_i08(intrin_cast<__m256i>(b))));
}
Vc_INTRINSIC x_u08 bit_shift_right(x_u08 a, x_u08 b)
{
    return lo128(
        bit_shift_right(y_u08(intrin_cast<__m256i>(a)), y_u08(intrin_cast<__m256i>(b))));
}

#elif defined Vc_HAVE_SSE4_1

Vc_INTRINSIC x_i08 bit_shift_right(x_i08 a, x_i08 b)
{
    // exploit UB: The behavior is undefined if the right operand is [...] greater than or
    // equal to the length in bits of the promoted left operand.
    // => valid input range for each element of b is [0, 7]
    // => only the 3 low bits of b are relevant
    // do a =<< 4 where b[2] is set
    auto bit7 = and_(a, broadcast16(0x00800080));
    _mm_srai_epi16(slli_epi16<8>(a), 4);
    a = _mm_blendv_epi8(a,
                        _mm_srai_epi16(or_(_mm_sub_epi16(slli_epi16<5>(bit7), bit7),
                                           and_(a, broadcast16(0xf8f8f8f8u))),
                                       4),
                        slli_epi16<5>(b));
    // do a =<< 2 where b[1] is set
    a = _mm_blendv_epi8(a,
                        _mm_srai_epi16(or_(_mm_sub_epi16(slli_epi16<3>(bit7), bit7),
                                           and_(a, broadcast16(0xfcfcfcfcu))),
                                       2),
                        slli_epi16<6>(b));
    // do a =<< 1 where b[0] is set
    return _mm_blendv_epi8(
        a, _mm_srai_epi16(or_(slli_epi16<1>(bit7), and_(a, broadcast16(0xfefefefeu))),
                          1),
        slli_epi16<7>(b));
}
Vc_INTRINSIC x_u08 bit_shift_right(x_u08 a, x_u08 b)
{
    // exploit UB: The behavior is undefined if the right operand is [...] greater than or
    // equal to the length in bits of the promoted left operand.
    // => valid input range for each element of b is [0, 7]
    // => only the 3 low bits of b are relevant
    // do a =<< 4 where b[2] is set
    a = _mm_blendv_epi8(a, and_(srli_epi16<4>(a), broadcast16(0x0f0f0f0fu)),
                        slli_epi16<5>(b));
    // do a =<< 2 where b[1] is set
    a = _mm_blendv_epi8(a, and_(srli_epi16<2>(a), broadcast16(0x3f3f3f3fu)),
                        slli_epi16<6>(b));
    // do a =<< 1 where b[0] is set
    return _mm_blendv_epi8(a, and_(srli_epi16<1>(a), broadcast16(0x7f7f7f7f)),
                           slli_epi16<7>(b));
}

#endif  // Vc_HAVE_AVX512BW

#endif  // Vc_USE_BUILTIN_VECTOR_TYPES

// complement{{{1
template <class T, size_t N>
constexpr Vc_INTRINSIC Storage<T, N> complement(Storage<T, N> v)
{
    return ~v.d;
}

//}}}1
// unary_minus{{{1
// GCC doesn't use the psign instructions, but pxor & psub seem to be just as good a
// choice as pcmpeqd & psign. So meh.
template <class T, size_t N>
constexpr Vc_INTRINSIC Storage<T, N> unary_minus(Storage<T, N> v)
{
    return -v.d;
}

// abs{{{1
template <class T, size_t N>
Vc_NORMAL_MATH constexpr Vc_INTRINSIC Storage<T, N> abs(Storage<T, N> v)
{
    if constexpr (!have_avx512vl && std::is_integral_v<T> && sizeof(T) == 8) {
        // positive value:
        //   negative == 0
        //   a unchanged after xor
        //   a - 0 -> a
        // negative value:
        //   negative == ~0 == -1
        //   a xor ~0    -> -a - 1
        //   -a - 1 - -1 -> -a
        if constexpr(have_sse4_2) {
            const auto negative = reinterpret_cast<builtin_type_t<T, N>>(v.d < 0);
            return (v.d ^ negative) - negative;
        } else {
            // >>63: negative ->  1, positive ->  0
            //  - 1: negative ->  0, positive -> ~0
            //  ~  : negative -> ~0, positive ->  0
            const auto negative = reinterpret_cast<builtin_type_t<T, N>>(
                ~((reinterpret_cast<builtin_type_t<ullong, 2>>(v.d) >> 63) - 1));
            return (v.d ^ negative) - negative;
        }
    } else {
        return v.d < 0 ? -v.d : v.d;
    }
}

// sqrt{{{1
Vc_INTRINSIC x_f32 Vc_VDECL sqrt(x_f32 v) { return _mm_sqrt_ps(v); }
#ifdef Vc_HAVE_SSE2
Vc_INTRINSIC x_f64 Vc_VDECL sqrt(x_f64 v) { return _mm_sqrt_pd(v); }
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
Vc_INTRINSIC y_f32 Vc_VDECL sqrt(y_f32 v) { return _mm256_sqrt_ps(v); }
Vc_INTRINSIC y_f64 Vc_VDECL sqrt(y_f64 v) { return _mm256_sqrt_pd(v); }
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC z_f32 Vc_VDECL sqrt(z_f32 v) { return _mm512_sqrt_ps(v); }
Vc_INTRINSIC z_f64 Vc_VDECL sqrt(z_f64 v) { return _mm512_sqrt_pd(v); }
#endif  // Vc_HAVE_AVX512F

//}}}1

}}  // namespace detail::x86
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_SIMD_X86_ARITHMETICS_H_
