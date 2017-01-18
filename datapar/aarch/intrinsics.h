/*  This file is part of the Vc library. {{{
Copyright © 2017 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_DATAPAR_AARCH_H_
#define VC_DATAPAR_AARCH_H_

#include <limits>
#include "../macros.h"
#include "../detail.h"
#include "../const.h"

namespace Vc_VERSIONED_NAMESPACE::detail
{
namespace aarch
{
using aarch_const = constants<datapar_abi::aarch>;

// builtin_type{{{1
template <typename ValueType, size_t Bytes> struct builtin_type_impl;
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
template <> struct builtin_type_impl<         double   , 16> { typedef          double    type [[gnu::vector_size(16)]]; };
template <> struct builtin_type_impl<         float    , 16> { typedef          float     type [[gnu::vector_size(16)]]; };
template <> struct builtin_type_impl<         long long, 16> { typedef          long long type [[gnu::vector_size(16)]]; };
template <> struct builtin_type_impl<unsigned long long, 16> { typedef unsigned long long type [[gnu::vector_size(16)]]; };
template <> struct builtin_type_impl<         long     , 16> { typedef          long      type [[gnu::vector_size(16)]]; };
template <> struct builtin_type_impl<unsigned long     , 16> { typedef unsigned long      type [[gnu::vector_size(16)]]; };
template <> struct builtin_type_impl<         int      , 16> { typedef          int       type [[gnu::vector_size(16)]]; };
template <> struct builtin_type_impl<unsigned int      , 16> { typedef unsigned int       type [[gnu::vector_size(16)]]; };
template <> struct builtin_type_impl<         short    , 16> { typedef          short     type [[gnu::vector_size(16)]]; };
template <> struct builtin_type_impl<unsigned short    , 16> { typedef unsigned short     type [[gnu::vector_size(16)]]; };
template <> struct builtin_type_impl<         char     , 16> { typedef          char      type [[gnu::vector_size(16)]]; };
template <> struct builtin_type_impl<unsigned char     , 16> { typedef unsigned char      type [[gnu::vector_size(16)]]; };
template <> struct builtin_type_impl<  signed char     , 16> { typedef   signed char      type [[gnu::vector_size(16)]]; };
template <> struct builtin_type_impl<         bool     , 16> { typedef unsigned char      type [[gnu::vector_size(16)]]; };

template <> struct builtin_type_impl<         double   , 32> { typedef          double    type [[gnu::vector_size(32)]]; };
template <> struct builtin_type_impl<         float    , 32> { typedef          float     type [[gnu::vector_size(32)]]; };
template <> struct builtin_type_impl<         long long, 32> { typedef          long long type [[gnu::vector_size(32)]]; };
template <> struct builtin_type_impl<unsigned long long, 32> { typedef unsigned long long type [[gnu::vector_size(32)]]; };
template <> struct builtin_type_impl<         long     , 32> { typedef          long      type [[gnu::vector_size(32)]]; };
template <> struct builtin_type_impl<unsigned long     , 32> { typedef unsigned long      type [[gnu::vector_size(32)]]; };
template <> struct builtin_type_impl<         int      , 32> { typedef          int       type [[gnu::vector_size(32)]]; };
template <> struct builtin_type_impl<unsigned int      , 32> { typedef unsigned int       type [[gnu::vector_size(32)]]; };
template <> struct builtin_type_impl<         short    , 32> { typedef          short     type [[gnu::vector_size(32)]]; };
template <> struct builtin_type_impl<unsigned short    , 32> { typedef unsigned short     type [[gnu::vector_size(32)]]; };
template <> struct builtin_type_impl<         char     , 32> { typedef          char      type [[gnu::vector_size(32)]]; };
template <> struct builtin_type_impl<unsigned char     , 32> { typedef unsigned char      type [[gnu::vector_size(32)]]; };
template <> struct builtin_type_impl<  signed char     , 32> { typedef   signed char      type [[gnu::vector_size(32)]]; };
template <> struct builtin_type_impl<         bool     , 32> { typedef unsigned char      type [[gnu::vector_size(32)]]; };

template <> struct builtin_type_impl<         double   , 64> { typedef          double    type [[gnu::vector_size(64)]]; };
template <> struct builtin_type_impl<         float    , 64> { typedef          float     type [[gnu::vector_size(64)]]; };
template <> struct builtin_type_impl<         long long, 64> { typedef          long long type [[gnu::vector_size(64)]]; };
template <> struct builtin_type_impl<unsigned long long, 64> { typedef unsigned long long type [[gnu::vector_size(64)]]; };
template <> struct builtin_type_impl<         long     , 64> { typedef          long      type [[gnu::vector_size(64)]]; };
template <> struct builtin_type_impl<unsigned long     , 64> { typedef unsigned long      type [[gnu::vector_size(64)]]; };
template <> struct builtin_type_impl<         int      , 64> { typedef          int       type [[gnu::vector_size(64)]]; };
template <> struct builtin_type_impl<unsigned int      , 64> { typedef unsigned int       type [[gnu::vector_size(64)]]; };
template <> struct builtin_type_impl<         short    , 64> { typedef          short     type [[gnu::vector_size(64)]]; };
template <> struct builtin_type_impl<unsigned short    , 64> { typedef unsigned short     type [[gnu::vector_size(64)]]; };
template <> struct builtin_type_impl<         char     , 64> { typedef          char      type [[gnu::vector_size(64)]]; };
template <> struct builtin_type_impl<unsigned char     , 64> { typedef unsigned char      type [[gnu::vector_size(64)]]; };
template <> struct builtin_type_impl<  signed char     , 64> { typedef   signed char      type [[gnu::vector_size(64)]]; };
template <> struct builtin_type_impl<         bool     , 64> { typedef unsigned char      type [[gnu::vector_size(64)]]; };
#endif

template <typename T, size_t Size>
using builtin_type = typename builtin_type_impl<T, Size * sizeof(T)>::type;

// intrinsic_type{{{1
template <typename T, size_t Bytes> struct intrinsic_type_impl {
    static_assert(sizeof(T) == Bytes,
                  "intrinsic_type without SIMD target support may only have Size = 1");
    using type = T;
};

template <> struct intrinsic_type_impl<double, 16> { using type = float64x2_t; };
template <typename T> struct intrinsic_type_impl<T, 16> { using type = int32x4_t; };

template <typename T, size_t Size>
using intrinsic_type = typename intrinsic_type_impl<T, Size * sizeof(T)>::type;

// is_intrinsic{{{1
template <class T> struct is_intrinsic : public std::false_type {};

template <> struct is_intrinsic<float32x4_t> : public std::true_type {};
template <> struct is_intrinsic<float64x2_t> : public std::true_type {};
template <> struct is_intrinsic<int32x4_t> : public std::true_type {};
template <class T> constexpr bool is_intrinsic_v = is_intrinsic<T>::value;

// intrin_cast{{{1
template<typename T> Vc_INTRINSIC_L T intrin_cast(float32x4_t  v) Vc_INTRINSIC_R;
template<typename T> Vc_INTRINSIC_L T intrin_cast(int32x4_t v) Vc_INTRINSIC_R;
template<typename T> Vc_INTRINSIC_L T intrin_cast(float64x2_t v) Vc_INTRINSIC_R;

// 128 -> 128
template<> Vc_INTRINSIC float32x4_t  intrin_cast(float32x4_t  v) { return v; }
template<> Vc_INTRINSIC float32x4_t  intrin_cast(int32x4_t v) { return _mm_castsi128_ps(v); }
template<> Vc_INTRINSIC float32x4_t  intrin_cast(float64x2_t v) { return _mm_castpd_ps(v); }
template<> Vc_INTRINSIC int32x4_t intrin_cast(float32x4_t  v) { return _mm_castps_si128(v); }
template<> Vc_INTRINSIC int32x4_t intrin_cast(int32x4_t v) { return v; }
template<> Vc_INTRINSIC int32x4_t intrin_cast(float64x2_t v) { return _mm_castpd_si128(v); }
template<> Vc_INTRINSIC float64x2_t intrin_cast(float32x4_t  v) { return _mm_castps_pd(v); }
template<> Vc_INTRINSIC float64x2_t intrin_cast(int32x4_t v) { return _mm_castsi128_pd(v); }
template<> Vc_INTRINSIC float64x2_t intrin_cast(float64x2_t v) { return v; }

template <typename V> Vc_INTRINSIC_L Vc_CONST_L V allone() Vc_INTRINSIC_R Vc_CONST_R;
template <> Vc_INTRINSIC Vc_CONST float32x4_t allone<float32x4_t>()
{
    return _mm_load_ps(reinterpret_cast<const float *>(sse_const::AllBitsSet));
}
template <> Vc_INTRINSIC Vc_CONST int32x4_t allone<int32x4_t>()
{
    return _mm_load_si128(reinterpret_cast<const int32x4_t *>(sse_const::AllBitsSet));
}
template <> Vc_INTRINSIC Vc_CONST float64x2_t allone<float64x2_t>()
{
    return _mm_load_pd(reinterpret_cast<const double *>(sse_const::AllBitsSet));
}

// zero{{{1
template <typename V> Vc_INTRINSIC_L Vc_CONST_L V zero() Vc_INTRINSIC_R Vc_CONST_R;
template<> Vc_INTRINSIC Vc_CONST float32x4_t  zero<float32x4_t >() { return _mm_setzero_ps(); }
template<> Vc_INTRINSIC Vc_CONST int32x4_t zero<int32x4_t>() { return _mm_setzero_si128(); }
template<> Vc_INTRINSIC Vc_CONST float64x2_t zero<float64x2_t>() { return _mm_setzero_pd(); }

// one16/32{{{1
Vc_INTRINSIC Vc_CONST float32x4_t  one16( float) { return _mm_load_ps(sse_const::oneFloat); }

Vc_INTRINSIC Vc_CONST float64x2_t one16(double) { return _mm_load_pd(sse_const::oneDouble); }
Vc_INTRINSIC Vc_CONST int32x4_t one16( schar) { return _mm_load_si128(reinterpret_cast<const int32x4_t *>(sse_const::one8)); }
Vc_INTRINSIC Vc_CONST int32x4_t one16( uchar) { return one16(schar()); }
Vc_INTRINSIC Vc_CONST int32x4_t one16( short) { return _mm_load_si128(reinterpret_cast<const int32x4_t *>(sse_const::one16)); }
Vc_INTRINSIC Vc_CONST int32x4_t one16(ushort) { return one16(short()); }
Vc_INTRINSIC Vc_CONST int32x4_t one16(   int) { return _mm_load_si128(reinterpret_cast<const int32x4_t *>(sse_const::one32)); }
Vc_INTRINSIC Vc_CONST int32x4_t one16(  uint) { return one16(int()); }
Vc_INTRINSIC Vc_CONST int32x4_t one16( llong) { return _mm_load_si128(reinterpret_cast<const int32x4_t *>(sse_const::one64)); }
Vc_INTRINSIC Vc_CONST int32x4_t one16(ullong) { return one16(llong()); }
Vc_INTRINSIC Vc_CONST int32x4_t one16(  long) { return one16(equal_int_type_t<long>()); }
Vc_INTRINSIC Vc_CONST int32x4_t one16( ulong) { return one16(equal_int_type_t<ulong>()); }

// signmask{{{1
Vc_INTRINSIC Vc_CONST float64x2_t signmask16(double){ return _mm_load_pd(reinterpret_cast<const double *>(sse_const::signMaskDouble)); }
Vc_INTRINSIC Vc_CONST float32x4_t  signmask16( float){ return _mm_load_ps(reinterpret_cast<const float *>(sse_const::signMaskFloat)); }

// set16/32/64{{{1
Vc_INTRINSIC Vc_CONST float32x4_t set(float x0, float x1, float x2, float x3)
{
    return _mm_set_ps(x3, x2, x1, x0);
}
Vc_INTRINSIC Vc_CONST float64x2_t set(double x0, double x1) { return _mm_set_pd(x1, x0); }

Vc_INTRINSIC Vc_CONST int32x4_t set(int x0, int x1, int x2, int x3)
{
    return _mm_set_epi32(x3, x2, x1, x0);
}
Vc_INTRINSIC Vc_CONST int32x4_t set(uint x0, uint x1, uint x2, uint x3)
{
    return _mm_set_epi32(x3, x2, x1, x0);
}

Vc_INTRINSIC Vc_CONST int32x4_t set(short x0, short x1, short x2, short x3, short x4,
                                  short x5, short x6, short x7)
{
    return _mm_set_epi16(x7, x6, x5, x4, x3, x2, x1, x0);
}
Vc_INTRINSIC Vc_CONST int32x4_t set(ushort x0, ushort x1, ushort x2, ushort x3, ushort x4,
                                  ushort x5, ushort x6, ushort x7)
{
    return _mm_set_epi16(x7, x6, x5, x4, x3, x2, x1, x0);
}

Vc_INTRINSIC Vc_CONST int32x4_t set(schar x0, schar x1, schar x2, schar x3, schar x4,
                                  schar x5, schar x6, schar x7, schar x8, schar x9,
                                  schar x10, schar x11, schar x12, schar x13, schar x14,
                                  schar x15)
{
    return _mm_set_epi8(x15, x14, x13, x12, x11, x10, x9, x8, x7, x6, x5, x4, x3, x2, x1,
                         x0);
}
Vc_INTRINSIC Vc_CONST int32x4_t set(uchar x0, uchar x1, uchar x2, uchar x3, uchar x4,
                                  uchar x5, uchar x6, uchar x7, uchar x8, uchar x9,
                                  uchar x10, uchar x11, uchar x12, uchar x13, uchar x14,
                                  uchar x15)
{
    return _mm_set_epi8(x15, x14, x13, x12, x11, x10, x9, x8, x7, x6, x5, x4, x3, x2, x1,
                         x0);
}

// generic forward for (u)long to (u)int or (u)llong
template <typename... Ts> Vc_INTRINSIC Vc_CONST auto set(Ts... args)
{
    return set(static_cast<equal_int_type_t<Ts>>(args)...);
}

// broadcast16/32/64{{{1
Vc_INTRINSIC float32x4_t  broadcast16( float x) { return _mm_set1_ps(x); }
#ifdef Vc_HAVE_SSE2
Vc_INTRINSIC float64x2_t broadcast16(double x) { return _mm_set1_pd(x); }
Vc_INTRINSIC int32x4_t broadcast16( schar x) { return _mm_set1_epi8(x); }
Vc_INTRINSIC int32x4_t broadcast16( uchar x) { return _mm_set1_epi8(x); }
Vc_INTRINSIC int32x4_t broadcast16( short x) { return _mm_set1_epi16(x); }
Vc_INTRINSIC int32x4_t broadcast16(ushort x) { return _mm_set1_epi16(x); }
Vc_INTRINSIC int32x4_t broadcast16(   int x) { return _mm_set1_epi32(x); }
Vc_INTRINSIC int32x4_t broadcast16(  uint x) { return _mm_set1_epi32(x); }
#endif  // Vc_HAVE_SSE2

// lowest16/32/64{{{1
template <class T>
Vc_INTRINSIC Vc_CONST typename intrinsic_type_impl<T, 16>::type lowest16()
{
    return broadcast16(std::numeric_limits<T>::lowest());
}

#ifdef Vc_HAVE_SSE2
template <> Vc_INTRINSIC Vc_CONST int32x4_t lowest16< short>() { return _mm_load_si128(reinterpret_cast<const int32x4_t *>(sse_const::minShort)); }
template <> Vc_INTRINSIC Vc_CONST int32x4_t lowest16<   int>() { return _mm_load_si128(reinterpret_cast<const int32x4_t *>(sse_const::signMaskFloat)); }
template <> Vc_INTRINSIC Vc_CONST int32x4_t lowest16< llong>() { return _mm_load_si128(reinterpret_cast<const int32x4_t *>(sse_const::signMaskDouble)); }
template <> Vc_INTRINSIC Vc_CONST int32x4_t lowest16<  long>() { return lowest16<equal_int_type_t<long>>(); }

template <> Vc_INTRINSIC Vc_CONST int32x4_t lowest16< uchar>() { return _mm_setzero_si128(); }
template <> Vc_INTRINSIC Vc_CONST int32x4_t lowest16<ushort>() { return _mm_setzero_si128(); }
template <> Vc_INTRINSIC Vc_CONST int32x4_t lowest16<  uint>() { return _mm_setzero_si128(); }
template <> Vc_INTRINSIC Vc_CONST int32x4_t lowest16< ulong>() { return _mm_setzero_si128(); }
template <> Vc_INTRINSIC Vc_CONST int32x4_t lowest16<ullong>() { return _mm_setzero_si128(); }
#endif  // Vc_HAVE_SSE2

// _2_pow_31{{{1
template <class T> inline typename intrinsic_type_impl<T, 16>::type sse_2_pow_31();
template <> Vc_INTRINSIC Vc_CONST float32x4_t  sse_2_pow_31< float>() { return broadcast16( float(1u << 31)); }
#ifdef Vc_HAVE_SSE2
template <> Vc_INTRINSIC Vc_CONST float64x2_t sse_2_pow_31<double>() { return broadcast16(double(1u << 31)); }
template <> Vc_INTRINSIC Vc_CONST int32x4_t sse_2_pow_31<  uint>() { return lowest16<int>(); }
#endif  // Vc_HAVE_SSE2


// SSE intrinsics emulation{{{1
Vc_INTRINSIC int32x4_t setone_epi8 ()  { return _mm_set1_epi8(1); }
Vc_INTRINSIC int32x4_t setone_epu8 ()  { return setone_epi8(); }
Vc_INTRINSIC int32x4_t setone_epi16()  { return _mm_load_si128(reinterpret_cast<const int32x4_t *>(sse_const::one16)); }
Vc_INTRINSIC int32x4_t setone_epu16()  { return setone_epi16(); }
Vc_INTRINSIC int32x4_t setone_epi32()  { return _mm_load_si128(reinterpret_cast<const int32x4_t *>(sse_const::one32)); }
Vc_INTRINSIC int32x4_t setone_epu32()  { return setone_epi32(); }

Vc_INTRINSIC float32x4_t  setone_ps()     { return _mm_load_ps(sse_const::oneFloat); }
Vc_INTRINSIC float64x2_t setone_pd()     { return _mm_load_pd(sse_const::oneDouble); }

Vc_INTRINSIC float64x2_t setabsmask_pd() { return _mm_load_pd(reinterpret_cast<const double *>(sse_const::absMaskDouble)); }
Vc_INTRINSIC float32x4_t  setabsmask_ps() { return _mm_load_ps(reinterpret_cast<const float *>(sse_const::absMaskFloat)); }


#ifdef Vc_HAVE_SSE2
#if defined(Vc_IMPL_XOP)
Vc_INTRINSIC int32x4_t cmplt_epu8 (int32x4_t a, int32x4_t b) { return _mm_comlt_epu8 (a, b); }
Vc_INTRINSIC int32x4_t cmpgt_epu8 (int32x4_t a, int32x4_t b) { return _mm_comgt_epu8 (a, b); }
Vc_INTRINSIC int32x4_t cmplt_epu16(int32x4_t a, int32x4_t b) { return _mm_comlt_epu16(a, b); }
Vc_INTRINSIC int32x4_t cmpgt_epu16(int32x4_t a, int32x4_t b) { return _mm_comgt_epu16(a, b); }
Vc_INTRINSIC int32x4_t cmplt_epu32(int32x4_t a, int32x4_t b) { return _mm_comlt_epu32(a, b); }
Vc_INTRINSIC int32x4_t cmpgt_epu32(int32x4_t a, int32x4_t b) { return _mm_comgt_epu32(a, b); }
Vc_INTRINSIC int32x4_t cmplt_epu64(int32x4_t a, int32x4_t b) { return _mm_comlt_epu64(a, b); }
Vc_INTRINSIC int32x4_t cmpgt_epu64(int32x4_t a, int32x4_t b) { return _mm_comgt_epu64(a, b); }
#else
Vc_INTRINSIC int32x4_t Vc_CONST cmplt_epu8(int32x4_t a, int32x4_t b)
{
    return _mm_cmplt_epi8(_mm_xor_si128(a, lowest16<schar>()),
                          _mm_xor_si128(b, lowest16<schar>()));
}
Vc_INTRINSIC int32x4_t Vc_CONST cmpgt_epu8(int32x4_t a, int32x4_t b)
{
    return _mm_cmpgt_epi8(_mm_xor_si128(a, lowest16<schar>()),
                          _mm_xor_si128(b, lowest16<schar>()));
}
Vc_INTRINSIC int32x4_t Vc_CONST cmplt_epu16(int32x4_t a, int32x4_t b)
{
    return _mm_cmplt_epi16(_mm_xor_si128(a, lowest16<short>()),
                           _mm_xor_si128(b, lowest16<short>()));
}
Vc_INTRINSIC int32x4_t Vc_CONST cmpgt_epu16(int32x4_t a, int32x4_t b)
{
    return _mm_cmpgt_epi16(_mm_xor_si128(a, lowest16<short>()),
                           _mm_xor_si128(b, lowest16<short>()));
}
Vc_INTRINSIC int32x4_t Vc_CONST cmplt_epu32(int32x4_t a, int32x4_t b)
{
    return _mm_cmplt_epi32(_mm_xor_si128(a, lowest16<int>()),
                           _mm_xor_si128(b, lowest16<int>()));
}
Vc_INTRINSIC int32x4_t Vc_CONST cmpgt_epu32(int32x4_t a, int32x4_t b)
{
    return _mm_cmpgt_epi32(_mm_xor_si128(a, lowest16<int>()),
                           _mm_xor_si128(b, lowest16<int>()));
}
Vc_INTRINSIC int32x4_t Vc_CONST cmpgt_epi64(int32x4_t a, int32x4_t b)
{
#ifdef Vc_IMPL_SSE4_2
    return _mm_cmpgt_epi64(a, b);
#else
    const auto aa = _mm_xor_si128(a, _mm_srli_epi64(lowest16<int>(), 32));
    const auto bb = _mm_xor_si128(b, _mm_srli_epi64(lowest16<int>(), 32));
    const auto gt = _mm_cmpgt_epi32(aa, bb);
    const auto eq = _mm_cmpeq_epi32(aa, bb);
    // Algorithm:
    // 1. if the high 32 bits of gt are true, make the full 64 bits true
    // 2. if the high 32 bits of gt are false and the high 32 bits of eq are true,
    //    duplicate the low 32 bits of gt to the high 32 bits (note that this requires
    //    unsigned compare on the lower 32 bits, which is the reason for the xors
    //    above)
    // 3. else make the full 64 bits false

    const auto gt2 =
        _mm_shuffle_epi32(gt, 0xf5);  // dup the high 32 bits to the low 32 bits
    const auto lo = _mm_shuffle_epi32(_mm_and_si128(_mm_srli_epi64(eq, 32), gt), 0xa0);
    return _mm_or_si128(gt2, lo);
#endif
}
Vc_INTRINSIC int32x4_t Vc_CONST cmpgt_epu64(int32x4_t a, int32x4_t b)
{
    return cmpgt_epi64(_mm_xor_si128(a, lowest16<llong>()),
                       _mm_xor_si128(b, lowest16<llong>()));
}
#endif

Vc_INTRINSIC Vc_CONST int32x4_t abs_epi8(int32x4_t a) {
#ifdef Vc_HAVE_SSSE3
    return _mm_abs_epi8(a);
#else
    int32x4_t negative = _mm_cmplt_epi8(a, _mm_setzero_si128());
    return _mm_add_epi8(_mm_xor_si128(a, negative),
                        _mm_and_si128(negative, setone_epi8()));
#endif
}

Vc_INTRINSIC Vc_CONST int32x4_t abs_epi16(int32x4_t a) {
#ifdef Vc_HAVE_SSSE3
    return _mm_abs_epi16(a);
#else
    int32x4_t negative = _mm_cmplt_epi16(a, _mm_setzero_si128());
    return _mm_add_epi16(_mm_xor_si128(a, negative), _mm_srli_epi16(negative, 15));
#endif
}

Vc_INTRINSIC Vc_CONST int32x4_t abs_epi32(int32x4_t a) {
#ifdef Vc_HAVE_SSSE3
    return _mm_abs_epi32(a);
#else
    // positive value:
    //   negative == 0
    //   a unchanged after xor
    //   0 >> 31 -> 0
    //   a + 0 -> a
    // negative value:
    //   negative == -1
    //   a xor -1 -> -a - 1
    //   -1 >> 31 -> 1
    //   -a - 1 + 1 -> -a
    int32x4_t negative = _mm_cmplt_epi32(a, _mm_setzero_si128());
    return _mm_add_epi32(_mm_xor_si128(a, negative), _mm_srli_epi32(negative, 31));
#endif
}

template <int s> Vc_INTRINSIC Vc_CONST int32x4_t alignr(int32x4_t a, int32x4_t b)
{
#ifdef Vc_HAVE_SSSE3
    return _mm_alignr_epi8(a, b, s & 0x1fu);
#else
    switch (s & 0x1fu) {
        case  0: return b;
        case  1: return _mm_or_si128(_mm_slli_si128(a, 15), _mm_srli_si128(b,  1));
        case  2: return _mm_or_si128(_mm_slli_si128(a, 14), _mm_srli_si128(b,  2));
        case  3: return _mm_or_si128(_mm_slli_si128(a, 13), _mm_srli_si128(b,  3));
        case  4: return _mm_or_si128(_mm_slli_si128(a, 12), _mm_srli_si128(b,  4));
        case  5: return _mm_or_si128(_mm_slli_si128(a, 11), _mm_srli_si128(b,  5));
        case  6: return _mm_or_si128(_mm_slli_si128(a, 10), _mm_srli_si128(b,  6));
        case  7: return _mm_or_si128(_mm_slli_si128(a,  9), _mm_srli_si128(b,  7));
        case  8: return _mm_or_si128(_mm_slli_si128(a,  8), _mm_srli_si128(b,  8));
        case  9: return _mm_or_si128(_mm_slli_si128(a,  7), _mm_srli_si128(b,  9));
        case 10: return _mm_or_si128(_mm_slli_si128(a,  6), _mm_srli_si128(b, 10));
        case 11: return _mm_or_si128(_mm_slli_si128(a,  5), _mm_srli_si128(b, 11));
        case 12: return _mm_or_si128(_mm_slli_si128(a,  4), _mm_srli_si128(b, 12));
        case 13: return _mm_or_si128(_mm_slli_si128(a,  3), _mm_srli_si128(b, 13));
        case 14: return _mm_or_si128(_mm_slli_si128(a,  2), _mm_srli_si128(b, 14));
        case 15: return _mm_or_si128(_mm_slli_si128(a,  1), _mm_srli_si128(b, 15));
        case 16: return a;
        case 17: return _mm_srli_si128(a,  1);
        case 18: return _mm_srli_si128(a,  2);
        case 19: return _mm_srli_si128(a,  3);
        case 20: return _mm_srli_si128(a,  4);
        case 21: return _mm_srli_si128(a,  5);
        case 22: return _mm_srli_si128(a,  6);
        case 23: return _mm_srli_si128(a,  7);
        case 24: return _mm_srli_si128(a,  8);
        case 25: return _mm_srli_si128(a,  9);
        case 26: return _mm_srli_si128(a, 10);
        case 27: return _mm_srli_si128(a, 11);
        case 28: return _mm_srli_si128(a, 12);
        case 29: return _mm_srli_si128(a, 13);
        case 30: return _mm_srli_si128(a, 14);
        case 31: return _mm_srli_si128(a, 15);
    }
    return _mm_setzero_si128();
#endif
}
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
template <int shift> Vc_INTRINSIC Vc_CONST __m256i alignr(__m256i s1, __m256i s2)
{
#ifdef Vc_HAVE_AVX2
    return _mm256_alignr_epi8(s1, s2, shift);
#else
    return insert128<1>(
        _mm256_castsi128_si256(_mm_alignr_epi8(_mm256_castsi256_si128(s1),
                                               _mm256_castsi256_si128(s2), shift)),
        _mm_alignr_epi8(extract128<1>(s1), extract128<1>(s2), shift));
#endif  // Vc_HAVE_AVX2
}
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_SSE4_1
Vc_INTRINSIC Vc_CONST int32x4_t cmpeq_epi64(int32x4_t a, int32x4_t b)
{
    return _mm_cmpeq_epi64(a, b);
}
template <int index> Vc_INTRINSIC Vc_CONST int extract_epi32(int32x4_t v)
{
    return _mm_extract_epi32(v, index);
}
Vc_INTRINSIC Vc_CONST float64x2_t blendv_pd(float64x2_t a, float64x2_t b, float64x2_t c)
{
    return _mm_blendv_pd(a, b, c);
}
Vc_INTRINSIC Vc_CONST float32x4_t blendv_ps(float32x4_t a, float32x4_t b, float32x4_t c)
{
    return _mm_blendv_ps(a, b, c);
}
Vc_INTRINSIC Vc_CONST int32x4_t blendv_epi8(int32x4_t a, int32x4_t b, int32x4_t c)
{
    return _mm_blendv_epi8(a, b, c);
}
template <int mask> Vc_INTRINSIC Vc_CONST float64x2_t blend_pd(float64x2_t a, float64x2_t b)
{
    return _mm_blend_pd(a, b, mask);
}
template <int mask> Vc_INTRINSIC Vc_CONST float32x4_t blend_ps(float32x4_t a, float32x4_t b)
{
    return _mm_blend_ps(a, b, mask);
}
template <int mask> Vc_INTRINSIC Vc_CONST int32x4_t blend_epi16(int32x4_t a, int32x4_t b)
{
    return _mm_blend_epi16(a, b, mask);
}
Vc_INTRINSIC Vc_CONST int32x4_t max_epi8(int32x4_t a, int32x4_t b)
{
    return _mm_max_epi8(a, b);
}
Vc_INTRINSIC Vc_CONST int32x4_t max_epi32(int32x4_t a, int32x4_t b)
{
    return _mm_max_epi32(a, b);
}
Vc_INTRINSIC Vc_CONST int32x4_t max_epu16(int32x4_t a, int32x4_t b)
{
    return _mm_max_epu16(a, b);
}
Vc_INTRINSIC Vc_CONST int32x4_t max_epu32(int32x4_t a, int32x4_t b)
{
    return _mm_max_epu32(a, b);
}
Vc_INTRINSIC Vc_CONST int32x4_t min_epu16(int32x4_t a, int32x4_t b)
{
    return _mm_min_epu16(a, b);
}
Vc_INTRINSIC Vc_CONST int32x4_t min_epu32(int32x4_t a, int32x4_t b)
{
    return _mm_min_epu32(a, b);
}
Vc_INTRINSIC Vc_CONST int32x4_t min_epi8(int32x4_t a, int32x4_t b)
{
    return _mm_min_epi8(a, b);
}
Vc_INTRINSIC Vc_CONST int32x4_t min_epi32(int32x4_t a, int32x4_t b)
{
    return _mm_min_epi32(a, b);
}
Vc_INTRINSIC Vc_CONST int32x4_t cvtepu8_epi16(int32x4_t epu8)
{
    return _mm_cvtepu8_epi16(epu8);
}
Vc_INTRINSIC Vc_CONST int32x4_t cvtepi8_epi16(int32x4_t epi8)
{
    return _mm_cvtepi8_epi16(epi8);
}
Vc_INTRINSIC Vc_CONST int32x4_t cvtepu16_epi32(int32x4_t epu16)
{
    return _mm_cvtepu16_epi32(epu16);
}
Vc_INTRINSIC Vc_CONST int32x4_t cvtepi16_epi32(int32x4_t epu16)
{
    return _mm_cvtepi16_epi32(epu16);
}
Vc_INTRINSIC Vc_CONST int32x4_t cvtepu8_epi32(int32x4_t epu8)
{
    return _mm_cvtepu8_epi32(epu8);
}
Vc_INTRINSIC Vc_CONST int32x4_t cvtepi8_epi32(int32x4_t epi8)
{
    return _mm_cvtepi8_epi32(epi8);
}
Vc_INTRINSIC Vc_PURE int32x4_t stream_load_si128(int32x4_t *mem)
{
    return _mm_stream_load_si128(mem);
}
#else  // Vc_HAVE_SSE4_1
Vc_INTRINSIC Vc_CONST float32x4_t  blendv_ps(float32x4_t  a, float32x4_t  b, float32x4_t  c) {
    return _mm_or_ps(_mm_andnot_ps(c, a), _mm_and_ps(c, b));
}
template <int mask> Vc_INTRINSIC Vc_CONST float32x4_t blend_ps(float32x4_t a, float32x4_t b)
{
    int32x4_t c;
    switch (mask) {
    case 0x0:
        return a;
    case 0x1:
        c = _mm_srli_si128(allone<int32x4_t>(), 12);
        break;
    case 0x2:
        c = _mm_slli_si128(_mm_srli_si128(allone<int32x4_t>(), 12), 4);
        break;
    case 0x3:
        c = _mm_srli_si128(allone<int32x4_t>(), 8);
        break;
    case 0x4:
        c = _mm_slli_si128(_mm_srli_si128(allone<int32x4_t>(), 12), 8);
        break;
    case 0x5:
        c = _mm_set_epi32(0, -1, 0, -1);
        break;
    case 0x6:
        c = _mm_slli_si128(_mm_srli_si128(allone<int32x4_t>(), 8), 4);
        break;
    case 0x7:
        c = _mm_srli_si128(allone<int32x4_t>(), 4);
        break;
    case 0x8:
        c = _mm_slli_si128(allone<int32x4_t>(), 12);
        break;
    case 0x9:
        c = _mm_set_epi32(-1, 0, 0, -1);
        break;
    case 0xa:
        c = _mm_set_epi32(-1, 0, -1, 0);
        break;
    case 0xb:
        c = _mm_set_epi32(-1, 0, -1, -1);
        break;
    case 0xc:
        c = _mm_slli_si128(allone<int32x4_t>(), 8);
        break;
    case 0xd:
        c = _mm_set_epi32(-1, -1, 0, -1);
        break;
    case 0xe:
        c = _mm_slli_si128(allone<int32x4_t>(), 4);
        break;
    case 0xf:
        return b;
    default: // may not happen
        abort();
        c = _mm_setzero_si128();
        break;
    }
    float32x4_t _c = _mm_castsi128_ps(c);
    return _mm_or_ps(_mm_andnot_ps(_c, a), _mm_and_ps(_c, b));
}

#ifdef Vc_HAVE_SSE2
Vc_INTRINSIC Vc_CONST float64x2_t blendv_pd(float64x2_t a, float64x2_t b, float64x2_t c) {
    return _mm_or_pd(_mm_andnot_pd(c, a), _mm_and_pd(c, b));
}
Vc_INTRINSIC Vc_CONST int32x4_t blendv_epi8(int32x4_t a, int32x4_t b, int32x4_t c) {
    return _mm_or_si128(_mm_andnot_si128(c, a), _mm_and_si128(c, b));
}

// only use the following blend functions with immediates as mask and, of course, compiling
// with optimization
template <int mask> Vc_INTRINSIC Vc_CONST float64x2_t blend_pd(float64x2_t a, float64x2_t b)
{
    switch (mask) {
    case 0x0:
        return a;
    case 0x1:
        return _mm_shuffle_pd(b, a, 2);
    case 0x2:
        return _mm_shuffle_pd(a, b, 2);
    case 0x3:
        return b;
    default:
        abort();
        return a; // should never be reached, but MSVC needs it else it warns about 'not all control paths return a value'
    }
}
template <int mask> Vc_INTRINSIC Vc_CONST int32x4_t blend_epi16(int32x4_t a, int32x4_t b)
{
    int32x4_t c;
    switch (mask) {
    case 0x00:
        return a;
    case 0x01:
        c = _mm_srli_si128(allone<int32x4_t>(), 14);
        break;
    case 0x03:
        c = _mm_srli_si128(allone<int32x4_t>(), 12);
        break;
    case 0x07:
        c = _mm_srli_si128(allone<int32x4_t>(), 10);
        break;
    case 0x0f:
        return _mm_unpackhi_epi64(_mm_slli_si128(b, 8), a);
    case 0x1f:
        c = _mm_srli_si128(allone<int32x4_t>(), 6);
        break;
    case 0x3f:
        c = _mm_srli_si128(allone<int32x4_t>(), 4);
        break;
    case 0x7f:
        c = _mm_srli_si128(allone<int32x4_t>(), 2);
        break;
    case 0x80:
        c = _mm_slli_si128(allone<int32x4_t>(), 14);
        break;
    case 0xc0:
        c = _mm_slli_si128(allone<int32x4_t>(), 12);
        break;
    case 0xe0:
        c = _mm_slli_si128(allone<int32x4_t>(), 10);
        break;
    case 0xf0:
        c = _mm_slli_si128(allone<int32x4_t>(), 8);
        break;
    case 0xf8:
        c = _mm_slli_si128(allone<int32x4_t>(), 6);
        break;
    case 0xfc:
        c = _mm_slli_si128(allone<int32x4_t>(), 4);
        break;
    case 0xfe:
        c = _mm_slli_si128(allone<int32x4_t>(), 2);
        break;
    case 0xff:
        return b;
    case 0xcc:
        return _mm_unpacklo_epi32(_mm_shuffle_epi32(a, _MM_SHUFFLE(2, 0, 2, 0)), _mm_shuffle_epi32(b, _MM_SHUFFLE(3, 1, 3, 1)));
    case 0x33:
        return _mm_unpacklo_epi32(_mm_shuffle_epi32(b, _MM_SHUFFLE(2, 0, 2, 0)), _mm_shuffle_epi32(a, _MM_SHUFFLE(3, 1, 3, 1)));
    default:
        const int32x4_t shift = _mm_set_epi16(0x0100, 0x0200, 0x0400, 0x0800, 0x1000, 0x2000, 0x4000, -0x7fff);
        c = _mm_srai_epi16(_mm_mullo_epi16(_mm_set1_epi16(mask), shift), 15);
        break;
    }
    return _mm_or_si128(_mm_andnot_si128(c, a), _mm_and_si128(c, b));
}

Vc_INTRINSIC Vc_CONST int32x4_t cmpeq_epi64(int32x4_t a, int32x4_t b) {
    auto tmp = _mm_cmpeq_epi32(a, b);
    return _mm_and_si128(tmp, _mm_shuffle_epi32(tmp, 1*1 + 0*4 + 3*16 + 2*64));
}
template <int index> Vc_INTRINSIC Vc_CONST int extract_epi32(int32x4_t v)
{
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
    typedef int int32v4 __attribute__((__vector_size__(16)));
    return reinterpret_cast<const may_alias<int32v4> &>(v)[index];
#else
    return _mm_cvtsi128_si32(_mm_srli_si128(v, index * 4));
#endif
}

Vc_INTRINSIC Vc_CONST int32x4_t max_epi8 (int32x4_t a, int32x4_t b) {
    return blendv_epi8(b, a, _mm_cmpgt_epi8 (a, b));
}
Vc_INTRINSIC Vc_CONST int32x4_t max_epi32(int32x4_t a, int32x4_t b) {
    return blendv_epi8(b, a, _mm_cmpgt_epi32(a, b));
}
//X         Vc_INTRINSIC Vc_CONST int32x4_t max_epu8 (int32x4_t a, int32x4_t b) {
//X             return _mm_blendv_epi8(b, a, cmpgt_epu8 (a, b));
//X         }
Vc_INTRINSIC Vc_CONST int32x4_t max_epu16(int32x4_t a, int32x4_t b) {
    return blendv_epi8(b, a, cmpgt_epu16(a, b));
}
Vc_INTRINSIC Vc_CONST int32x4_t max_epu32(int32x4_t a, int32x4_t b) {
    return blendv_epi8(b, a, cmpgt_epu32(a, b));
}
//X         Vc_INTRINSIC Vc_CONST int32x4_t _mm_min_epu8 (int32x4_t a, int32x4_t b) {
//X             return _mm_blendv_epi8(a, b, cmpgt_epu8 (a, b));
//X         }
Vc_INTRINSIC Vc_CONST int32x4_t min_epu16(int32x4_t a, int32x4_t b) {
    return blendv_epi8(a, b, cmpgt_epu16(a, b));
}
Vc_INTRINSIC Vc_CONST int32x4_t min_epu32(int32x4_t a, int32x4_t b) {
    return blendv_epi8(a, b, cmpgt_epu32(a, b));
}
Vc_INTRINSIC Vc_CONST int32x4_t min_epi8 (int32x4_t a, int32x4_t b) {
    return blendv_epi8(a, b, _mm_cmpgt_epi8 (a, b));
}
Vc_INTRINSIC Vc_CONST int32x4_t min_epi32(int32x4_t a, int32x4_t b) {
    return blendv_epi8(a, b, _mm_cmpgt_epi32(a, b));
}
Vc_INTRINSIC Vc_CONST int32x4_t cvtepu8_epi16(int32x4_t epu8) {
    return _mm_unpacklo_epi8(epu8, _mm_setzero_si128());
}
Vc_INTRINSIC Vc_CONST int32x4_t cvtepi8_epi16(int32x4_t epi8) {
    return _mm_unpacklo_epi8(epi8, _mm_cmplt_epi8(epi8, _mm_setzero_si128()));
}
Vc_INTRINSIC Vc_CONST int32x4_t cvtepu16_epi32(int32x4_t epu16) {
    return _mm_unpacklo_epi16(epu16, _mm_setzero_si128());
}
Vc_INTRINSIC Vc_CONST int32x4_t cvtepi16_epi32(int32x4_t epu16) {
    return _mm_unpacklo_epi16(epu16, _mm_cmplt_epi16(epu16, _mm_setzero_si128()));
}
Vc_INTRINSIC Vc_CONST int32x4_t cvtepu8_epi32(int32x4_t epu8) {
    return cvtepu16_epi32(cvtepu8_epi16(epu8));
}
Vc_INTRINSIC Vc_CONST int32x4_t cvtepi8_epi32(int32x4_t epi8) {
    const int32x4_t neg = _mm_cmplt_epi8(epi8, _mm_setzero_si128());
    const int32x4_t epi16 = _mm_unpacklo_epi8(epi8, neg);
    return _mm_unpacklo_epi16(epi16, _mm_unpacklo_epi8(neg, neg));
}
Vc_INTRINSIC Vc_PURE int32x4_t stream_load_si128(int32x4_t *mem) {
    return _mm_load_si128(mem);
}
#endif  // Vc_HAVE_SSE2
#endif  // Vc_HAVE_SSE4_1

// testc{{{1
Vc_INTRINSIC Vc_CONST int testc(float32x4_t  a, float32x4_t  b) { return _mm_testc_si128(_mm_castps_si128(a), _mm_castps_si128(b)); }
Vc_INTRINSIC Vc_CONST int testc(float64x2_t a, float64x2_t b) { return _mm_testc_si128(_mm_castpd_si128(a), _mm_castpd_si128(b)); }
Vc_INTRINSIC Vc_CONST int testc(int32x4_t a, int32x4_t b) { return _mm_testc_si128(a, b); }

// testz{{{1
Vc_INTRINSIC Vc_CONST int testz(float32x4_t  a, float32x4_t  b) { return _mm_testz_si128(_mm_castps_si128(a), _mm_castps_si128(b)); }
Vc_INTRINSIC Vc_CONST int testz(float64x2_t a, float64x2_t b) { return _mm_testz_si128(_mm_castpd_si128(a), _mm_castpd_si128(b)); }
Vc_INTRINSIC Vc_CONST int testz(int32x4_t a, int32x4_t b) { return _mm_testz_si128(a, b); }

// testnzc{{{1
Vc_INTRINSIC Vc_CONST int testnzc(float32x4_t  a, float32x4_t  b) { return _mm_testnzc_si128(_mm_castps_si128(a), _mm_castps_si128(b)); }
Vc_INTRINSIC Vc_CONST int testnzc(float64x2_t a, float64x2_t b) { return _mm_testnzc_si128(_mm_castpd_si128(a), _mm_castpd_si128(b)); }
Vc_INTRINSIC Vc_CONST int testnzc(int32x4_t a, int32x4_t b) { return _mm_testnzc_si128(a, b); }

// movemask{{{1
Vc_INTRINSIC Vc_CONST int movemask(float32x4_t  a) { return _mm_movemask_ps(a); }
Vc_INTRINSIC Vc_CONST int movemask(float64x2_t a) { return _mm_movemask_pd(a); }
Vc_INTRINSIC Vc_CONST int movemask(int32x4_t a) { return _mm_movemask_epi8(a); }

// negate{{{1
Vc_ALWAYS_INLINE Vc_CONST float32x4_t negate(float32x4_t v, std::integral_constant<std::size_t, 4>)
{
    return _mm_xor_ps(v, signmask16(float()));
}
Vc_ALWAYS_INLINE Vc_CONST float64x2_t negate(float64x2_t v, std::integral_constant<std::size_t, 8>)
{
    return _mm_xor_pd(v, signmask16(double()));
}
Vc_ALWAYS_INLINE Vc_CONST int32x4_t negate(int32x4_t v, std::integral_constant<std::size_t, 4>)
{
#ifdef Vc_IMPL_SSSE3
    return _mm_sign_epi32(v, allone<int32x4_t>());
#else
    return _mm_sub_epi32(zero<int32x4_t>(), v);
#endif
}
Vc_ALWAYS_INLINE Vc_CONST int32x4_t negate(int32x4_t v, std::integral_constant<std::size_t, 2>)
{
#ifdef Vc_IMPL_SSSE3
    return _mm_sign_epi16(v, allone<int32x4_t>());
#else
    return _mm_sub_epi16(zero<int32x4_t>(), v);
#endif
}

// xor_{{{1
Vc_INTRINSIC float32x4_t  xor_(float32x4_t  a, float32x4_t  b) { return _mm_xor_ps(a, b); }
Vc_INTRINSIC float64x2_t xor_(float64x2_t a, float64x2_t b) { return _mm_xor_pd(a, b); }
Vc_INTRINSIC int32x4_t xor_(int32x4_t a, int32x4_t b) { return _mm_xor_si128(a, b); }

// or_{{{1
Vc_INTRINSIC float32x4_t or_(float32x4_t a, float32x4_t b) { return _mm_or_ps(a, b); }
Vc_INTRINSIC float64x2_t or_(float64x2_t a, float64x2_t b) { return _mm_or_pd(a, b); }
Vc_INTRINSIC int32x4_t or_(int32x4_t a, int32x4_t b) { return _mm_or_si128(a, b); }

// and_{{{1
Vc_INTRINSIC float32x4_t and_(float32x4_t a, float32x4_t b) { return _mm_and_ps(a, b); }
Vc_INTRINSIC float64x2_t and_(float64x2_t a, float64x2_t b) { return _mm_and_pd(a, b); }
Vc_INTRINSIC int32x4_t and_(int32x4_t a, int32x4_t b) { return _mm_and_si128(a, b); }

// andnot_{{{1
Vc_INTRINSIC float32x4_t andnot_(float32x4_t a, float32x4_t b) { return _mm_andnot_ps(a, b); }
Vc_INTRINSIC float64x2_t andnot_(float64x2_t a, float64x2_t b) { return _mm_andnot_pd(a, b); }
Vc_INTRINSIC int32x4_t andnot_(int32x4_t a, int32x4_t b) { return _mm_andnot_si128(a, b); }

// shift_left{{{1
template <int n> Vc_INTRINSIC float32x4_t  shift_left(float32x4_t  v) { return _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(v), n)); }
template <int n> Vc_INTRINSIC float64x2_t shift_left(float64x2_t v) { return _mm_castsi128_pd(_mm_slli_si128(_mm_castpd_si128(v), n)); }
template <int n> Vc_INTRINSIC int32x4_t shift_left(int32x4_t v) { return _mm_slli_si128(v, n); }

// shift_right{{{1
template <int n> Vc_INTRINSIC float32x4_t  shift_right(float32x4_t  v) { return _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(v), n)); }
template <int n> Vc_INTRINSIC float64x2_t shift_right(float64x2_t v) { return _mm_castsi128_pd(_mm_srli_si128(_mm_castpd_si128(v), n)); }
template <int n> Vc_INTRINSIC int32x4_t shift_right(int32x4_t v) { return _mm_srli_si128(v, n); }

// popcnt{{{1
Vc_INTRINSIC Vc_CONST unsigned int popcnt4(unsigned int n)
{
#ifdef Vc_IMPL_POPCNT
    return _mm_popcnt_u32(n);
#else
    n = (n & 0x5U) + ((n >> 1) & 0x5U);
    n = (n & 0x3U) + ((n >> 2) & 0x3U);
    return n;
#endif
}
Vc_INTRINSIC Vc_CONST unsigned int popcnt8(unsigned int n)
{
#ifdef Vc_IMPL_POPCNT
    return _mm_popcnt_u32(n);
#else
    n = (n & 0x55U) + ((n >> 1) & 0x55U);
    n = (n & 0x33U) + ((n >> 2) & 0x33U);
    n = (n & 0x0fU) + ((n >> 4) & 0x0fU);
    return n;
#endif
}
Vc_INTRINSIC Vc_CONST unsigned int popcnt16(unsigned int n)
{
#ifdef Vc_IMPL_POPCNT
    return _mm_popcnt_u32(n);
#else
    n = (n & 0x5555U) + ((n >> 1) & 0x5555U);
    n = (n & 0x3333U) + ((n >> 2) & 0x3333U);
    n = (n & 0x0f0fU) + ((n >> 4) & 0x0f0fU);
    n = (n & 0x00ffU) + ((n >> 8) & 0x00ffU);
    return n;
#endif
}
Vc_INTRINSIC Vc_CONST unsigned int popcnt32(unsigned int n)
{
#ifdef Vc_IMPL_POPCNT
    return _mm_popcnt_u32(n);
#else
    n = (n & 0x55555555U) + ((n >> 1) & 0x55555555U);
    n = (n & 0x33333333U) + ((n >> 2) & 0x33333333U);
    n = (n & 0x0f0f0f0fU) + ((n >> 4) & 0x0f0f0f0fU);
    n = (n & 0x00ff00ffU) + ((n >> 8) & 0x00ff00ffU);
    n = (n & 0x0000ffffU) + ((n >>16) & 0x0000ffffU);
    return n;
#endif
}

// mask_count{{{1
template <size_t Size> int mask_count(float32x4_t );
template <size_t Size> int mask_count(int32x4_t);
template <size_t Size> int mask_count(float64x2_t);
template<> Vc_INTRINSIC Vc_CONST int mask_count<2>(float64x2_t k)
{
    int mask = _mm_movemask_pd(k);
    return (mask & 1) + (mask >> 1);
}
template<> Vc_INTRINSIC Vc_CONST int mask_count<2>(int32x4_t k)
{
    return mask_count<2>(_mm_castsi128_pd(k));
}

template<> Vc_INTRINSIC Vc_CONST int mask_count<4>(float32x4_t  k)
{
#ifdef Vc_IMPL_POPCNT
    return _mm_popcnt_u32(_mm_movemask_ps(k));
#else
    auto x = _mm_srli_epi32(_mm_castps_si128(k), 31);
    x = _mm_add_epi32(x, _mm_shuffle_epi32(x, _MM_SHUFFLE(0, 1, 2, 3)));
    x = _mm_add_epi32(x, _mm_shufflelo_epi16(x, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtsi128_si32(x);
#endif
}
template<> Vc_INTRINSIC Vc_CONST int mask_count<4>(int32x4_t k)
{
    return mask_count<4>(_mm_castsi128_ps(k));
}

template<> Vc_INTRINSIC Vc_CONST int mask_count<8>(int32x4_t k)
{
#ifdef Vc_IMPL_POPCNT
    return _mm_popcnt_u32(_mm_movemask_epi8(k)) / 2;
#else
    auto x = _mm_srli_epi16(k, 15);
    x = _mm_add_epi16(x, _mm_shuffle_epi32(x, _MM_SHUFFLE(0, 1, 2, 3)));
    x = _mm_add_epi16(x, _mm_shufflelo_epi16(x, _MM_SHUFFLE(0, 1, 2, 3)));
    x = _mm_add_epi16(x, _mm_shufflelo_epi16(x, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_extract_epi16(x, 0);
#endif
}

template<> Vc_INTRINSIC Vc_CONST int mask_count<16>(int32x4_t k)
{
    return popcnt16(_mm_movemask_epi8(k));
}

// mask_to_int{{{1
template <size_t Size> inline int mask_to_int(int32x4_t) { static_assert(Size == Size, "Size value not implemented"); return 0; }

template<> Vc_INTRINSIC Vc_CONST int mask_to_int<2>(int32x4_t k)
{
    return _mm_movemask_pd(_mm_castsi128_pd(k));
}
template<> Vc_INTRINSIC Vc_CONST int mask_to_int<4>(int32x4_t k)
{
    return _mm_movemask_ps(_mm_castsi128_ps(k));
}
template<> Vc_INTRINSIC Vc_CONST int mask_to_int<8>(int32x4_t k)
{
    return _mm_movemask_epi8(_mm_packs_epi16(k, _mm_setzero_si128()));
}
template<> Vc_INTRINSIC Vc_CONST int mask_to_int<16>(int32x4_t k)
{
    return _mm_movemask_epi8(k);
}

// is_equal{{{1
template <size_t Size>
Vc_INTRINSIC_L Vc_CONST_L bool is_equal(float32x4_t, float32x4_t) Vc_INTRINSIC_R Vc_CONST_R;
template <size_t Size>
Vc_INTRINSIC_L Vc_CONST_L bool is_not_equal(float32x4_t, float32x4_t) Vc_INTRINSIC_R Vc_CONST_R;
template <> Vc_INTRINSIC Vc_CONST bool is_equal<2>(float32x4_t k1, float32x4_t k2)
{
    return _mm_movemask_pd(_mm_castps_pd(k1)) == _mm_movemask_pd(_mm_castps_pd(k2));
}
template <> Vc_INTRINSIC Vc_CONST bool is_not_equal<2>(float32x4_t k1, float32x4_t k2)
{
    return _mm_movemask_pd(_mm_castps_pd(k1)) != _mm_movemask_pd(_mm_castps_pd(k2));
}

template <> Vc_INTRINSIC Vc_CONST bool is_equal<4>(float32x4_t k1, float32x4_t k2)
{
    return _mm_movemask_ps(k1) == _mm_movemask_ps(k2);
}
template <> Vc_INTRINSIC Vc_CONST bool is_not_equal<4>(float32x4_t k1, float32x4_t k2)
{
    return _mm_movemask_ps(k1) != _mm_movemask_ps(k2);
}

template <> Vc_INTRINSIC Vc_CONST bool is_equal<8>(float32x4_t k1, float32x4_t k2)
{
    return _mm_movemask_epi8(_mm_castps_si128(k1)) ==
           _mm_movemask_epi8(_mm_castps_si128(k2));
}
template <> Vc_INTRINSIC Vc_CONST bool is_not_equal<8>(float32x4_t k1, float32x4_t k2)
{
    return _mm_movemask_epi8(_mm_castps_si128(k1)) !=
           _mm_movemask_epi8(_mm_castps_si128(k2));
}

template <> Vc_INTRINSIC Vc_CONST bool is_equal<16>(float32x4_t k1, float32x4_t k2)
{
    return _mm_movemask_epi8(_mm_castps_si128(k1)) ==
           _mm_movemask_epi8(_mm_castps_si128(k2));
}
template <> Vc_INTRINSIC Vc_CONST bool is_not_equal<16>(float32x4_t k1, float32x4_t k2)
{
    return _mm_movemask_epi8(_mm_castps_si128(k1)) !=
           _mm_movemask_epi8(_mm_castps_si128(k2));
}

// loads{{{1
/**
 * \internal
 * Abstraction for simplifying load operations in the SSE/AVX/AVX512 implementations
 *
 * \note The number in the suffix signifies the number of Bytes
 */
#ifdef Vc_HAVE_SSE
Vc_INTRINSIC float32x4_t load16(const float *mem, when_aligned<16>)
{
    return _mm_load_ps(mem);
}
Vc_INTRINSIC float32x4_t load16(const float *mem, when_unaligned<16>)
{
    return _mm_loadu_ps(mem);
}

Vc_INTRINSIC float64x2_t load16(const double *mem, when_aligned<16>)
{
    return _mm_load_pd(mem);
}

Vc_INTRINSIC float64x2_t load16(const double *mem, when_unaligned<16>)
{
    return _mm_loadu_pd(mem);
}

template <class T> Vc_INTRINSIC int32x4_t load16(const T *mem, when_aligned<16>)
{
    static_assert(std::is_integral<T>::value, "load16<T> is only intended for integral T");
    return _mm_load_si128(reinterpret_cast<const int32x4_t *>(mem));
}

template <class T> Vc_INTRINSIC int32x4_t load16(const T *mem, when_unaligned<16>)
{
    static_assert(std::is_integral<T>::value, "load16<T> is only intended for integral T");
    return _mm_loadu_si128(reinterpret_cast<const int32x4_t *>(mem));
}

// stores{{{1
template <class Flags> Vc_INTRINSIC void store4(float32x4_t v, float *mem, Flags)
{
    *mem = _mm_cvtss_f32(v);
}

template <class Flags> Vc_INTRINSIC void store8(float32x4_t v, float *mem, Flags)
{
    _mm_storel_pi(reinterpret_cast<__m64 *>(mem), v);
}

Vc_INTRINSIC void store16(float32x4_t v, float *mem, when_aligned<16>)
{
    _mm_store_ps(mem, v);
}
Vc_INTRINSIC void store16(float32x4_t v, float *mem, when_unaligned<16>)
{
    _mm_storeu_ps(mem, v);
}

template <class Flags> Vc_INTRINSIC void store8(float64x2_t v, double *mem, Flags)
{
    *mem = _mm_cvtsd_f64(v);
}

Vc_INTRINSIC void store16(float64x2_t v, double *mem, when_aligned<16>)
{
    _mm_store_pd(mem, v);
}
Vc_INTRINSIC void store16(float64x2_t v, double *mem, when_unaligned<16>)
{
    _mm_storeu_pd(mem, v);
}

template <class T, class Flags> Vc_INTRINSIC void store2(int32x4_t v, T *mem, Flags)
{
    static_assert(std::is_integral<T>::value && sizeof(T) <= 2,
                  "store4<T> is only intended for integral T with sizeof(T) <= 2");
    *reinterpret_cast<may_alias<ushort> *>(mem) = uint(_mm_cvtsi128_si32(v));
}

template <class T, class Flags> Vc_INTRINSIC void store4(int32x4_t v, T *mem, Flags)
{
    static_assert(std::is_integral<T>::value && sizeof(T) <= 4,
                  "store4<T> is only intended for integral T with sizeof(T) <= 4");
    *reinterpret_cast<may_alias<int> *>(mem) = _mm_cvtsi128_si32(v);
}

template <class T, class Flags> Vc_INTRINSIC void store8(int32x4_t v, T *mem, Flags)
{
    static_assert(std::is_integral<T>::value, "store8<T> is only intended for integral T");
    _mm_storel_epi64(reinterpret_cast<int32x4_t *>(mem), v);
}

template <class T> Vc_INTRINSIC void store16(int32x4_t v, T *mem, when_aligned<16>)
{
    static_assert(std::is_integral<T>::value, "store16<T> is only intended for integral T");
    _mm_store_si128(reinterpret_cast<int32x4_t *>(mem), v);
}
template <class T> Vc_INTRINSIC void store16(int32x4_t v, T *mem, when_unaligned<16>)
{
    static_assert(std::is_integral<T>::value, "store16<T> is only intended for integral T");
    _mm_storeu_si128(reinterpret_cast<int32x4_t *>(mem), v);
}


// }}}1
}  // namespace aarch
using namespace aarch;
}  // namespace Vc_VERSIONED_NAMESPACE::detail

#endif  // VC_DATAPAR_AARCH_H_
