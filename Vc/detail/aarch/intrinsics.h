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
#include <arm_neon.h>

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
namespace aarch
{
using aarch_const = constants<datapar_abi::neon>;

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

// is_builtin_vector{{{1
template <class T> struct is_builtin_vector : public std::false_type {};
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
template <> struct is_builtin_vector<builtin_type<float, 4>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type<double, 2>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type< llong, 2>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type<ullong, 2>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type<  long, 16 / sizeof( long)>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type< ulong, 16 / sizeof(ulong)>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type<   int, 4>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type<  uint, 4>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type< short, 8>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type<ushort, 8>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type< schar,16>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type< uchar,16>> : public std::true_type {};
#endif
template <class T> constexpr bool is_builtin_vector_v = is_builtin_vector<T>::value;

// intrin_cast{{{1
template<typename T> Vc_INTRINSIC_L T intrin_cast(float32x4_t  v) Vc_INTRINSIC_R;
template<typename T> Vc_INTRINSIC_L T intrin_cast(int32x4_t v) Vc_INTRINSIC_R;
template<typename T> Vc_INTRINSIC_L T intrin_cast(float64x2_t v) Vc_INTRINSIC_R;

// 128 -> 128
// TBD
template<> Vc_INTRINSIC float32x4_t  intrin_cast(float32x4_t  v) { return v; }
template<> Vc_INTRINSIC float32x4_t  intrin_cast(int32x4_t v) { return vcvtq_f32_s32(v); }
template<> Vc_INTRINSIC float32x4_t  intrin_cast(float64x2_t v) { return vreinterpretq_f32_f64(v); }
template<> Vc_INTRINSIC int32x4_t intrin_cast(float32x4_t  v) { return vcvtq_s32_f32(v); }
template<> Vc_INTRINSIC int32x4_t intrin_cast(int32x4_t v) { return v; }
template<> Vc_INTRINSIC int32x4_t intrin_cast(float64x2_t v) { return vreinterpretq_s32_f64(v); }
template<> Vc_INTRINSIC float64x2_t intrin_cast(float32x4_t  v) { return vreinterpretq_f64_f32(v); }
template<> Vc_INTRINSIC float64x2_t intrin_cast(int32x4_t v) { return vreinterpretq_f64_s32(v); }
template<> Vc_INTRINSIC float64x2_t intrin_cast(float64x2_t v) { return v; }

// allone{{{1
template <typename V> Vc_INTRINSIC_L Vc_CONST_L V allone() Vc_INTRINSIC_R Vc_CONST_R;
template <> Vc_INTRINSIC Vc_CONST float32x4_t allone<float32x4_t>()
{
    return vld1q_f32(reinterpret_cast<const float *>(neon_const::AllBitsSet));
}
template <> Vc_INTRINSIC Vc_CONST int32x4_t allone<int32x4_t>()
{
    return vld1q_s32(reinterpret_cast<const int32_t *>(neon_const::AllBitsSet));
}
template <> Vc_INTRINSIC Vc_CONST float64x2_t allone<float64x2_t>()
{
    return vld1q_f64(reinterpret_cast<const double *>(neon_const::AllBitsSet));
}

// zero{{{1
template <typename V> Vc_INTRINSIC_L Vc_CONST_L V zero() Vc_INTRINSIC_R Vc_CONST_R;
template<> Vc_INTRINSIC Vc_CONST float32x4_t  zero<float32x4_t >() { return vdupq_n_f32(0); }
template<> Vc_INTRINSIC Vc_CONST int32x4_t zero<int32x4_t>() { return vdupq_n_s32(0); }
template<> Vc_INTRINSIC Vc_CONST float64x2_t zero<float64x2_t>() { return vdupq_n_f64(0); }

// one16/32{{{1
Vc_INTRINSIC Vc_CONST float32x4_t  one16( float) { return vld1q_f32(neon_const::oneFloat); }
Vc_INTRINSIC Vc_CONST float64x2_t one16(double) { return vld1q_f64(neon_const::oneDouble); }
Vc_INTRINSIC Vc_CONST int32x4_t one16( schar) { return vld1q_s32(reinterpret_cast<const int32_t *>(neon_const::one8)); }
Vc_INTRINSIC Vc_CONST int32x4_t one16( uchar) { return one16(schar()); }
Vc_INTRINSIC Vc_CONST int32x4_t one16( short) { return vld1q_s32(reinterpret_cast<const int32_t *>(neon_const::one16)); }
Vc_INTRINSIC Vc_CONST int32x4_t one16(ushort) { return one16(short()); }
Vc_INTRINSIC Vc_CONST int32x4_t one16(   int) { return vld1q_s32(reinterpret_cast<const int32_t *>(neon_const::one32)); }
Vc_INTRINSIC Vc_CONST int32x4_t one16(  uint) { return one16(int()); }
Vc_INTRINSIC Vc_CONST int32x4_t one16( llong) { return vld1q_s32(reinterpret_cast<const int32_t *>(neon_const::one64)); }
Vc_INTRINSIC Vc_CONST int32x4_t one16(ullong) { return one16(llong()); }
Vc_INTRINSIC Vc_CONST int32x4_t one16(  long) { return one16(equal_int_type_t<long>()); }
Vc_INTRINSIC Vc_CONST int32x4_t one16( ulong) { return one16(equal_int_type_t<ulong>()); }

// signmask{{{1
Vc_INTRINSIC Vc_CONST float64x2_t signmask16(double){ return vld1q_f64(reinterpret_cast<const double *>(neon_const::signMaskDouble)); }
Vc_INTRINSIC Vc_CONST float32x4_t  signmask16( float){ return vld1q_f32(reinterpret_cast<const float *>(neon_const::signMaskFloat)); }

// set16/32/64{{{1
Vc_INTRINSIC Vc_CONST float32x4_t set(float x0, float x1, float x2, float x3)
{
    float __attribute__((aligned(16))) data[4] = { x0, x1, x2, x3 };
	return vld1q_f32(data);
}
Vc_INTRINSIC Vc_CONST float64x2_t set(double x0, double x1)
{
	double __attribute__((aligned(16))) data[2] = { x0, x1 };
	return vld1q_f64(data);
}

Vc_INTRINSIC Vc_CONST int32x4_t set(int x0, int x1, int x2, int x3)
{
    int __attribute__((aligned(16))) data[4] = { x0, x1, x2, x3 };
	return vld1q_s32(data);
}
Vc_INTRINSIC Vc_CONST uint32x4_t set(uint x0, uint x1, uint x2, uint x3)
{
    uint __attribute__((aligned(16))) data[4] = { x0, x1, x2, x3 };
	return vld1q_u32(data);
}

Vc_INTRINSIC Vc_CONST int16x8_t set(short x0, short x1, short x2, short x3, short x4,
                                  short x5, short x6, short x7)
{
    short __attribute__((aligned(16))) data[8] = { x0, x1, x2, x3, x4, x5, x6, x7 };
	return  vld1q_s16(data);
}

Vc_INTRINSIC Vc_CONST uint16x8_t set(ushort x0, ushort x1, ushort x2, ushort x3, ushort x4,
                                  ushort x5, ushort x6, ushort x7)
{
    ushort __attribute__((aligned(16))) data[8] = { x0, x1, x2, x3, x4, x5, x6, x7 };
	return vld1q_u16(data);
}

Vc_INTRINSIC Vc_CONST int8x16_t set(schar x0, schar x1, schar x2, schar x3, schar x4,
                                  schar x5, schar x6, schar x7, schar x8, schar x9,
                                  schar x10, schar x11, schar x12, schar x13, schar x14,
                                  schar x15)
{
    schar __attribute__((aligned(16))) data[16] = { x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15 };
	return vld1q_s8(data);
}
Vc_INTRINSIC Vc_CONST uint8x16_t set(uchar x0, uchar x1, uchar x2, uchar x3, uchar x4,
                                  uchar x5, uchar x6, uchar x7, uchar x8, uchar x9,
                                  uchar x10, uchar x11, uchar x12, uchar x13, uchar x14,
                                  uchar x15)
{
    uchar __attribute__((aligned(16))) data[16] = { x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15 };
    return vld1q_u8(data);
}

// generic forward for (u)long to (u)int or (u)llong
template <typename... Ts> Vc_INTRINSIC Vc_CONST auto set(Ts... args)
{
    return set(static_cast<equal_int_type_t<Ts>>(args)...);
}

// broadcast16/32/64{{{1
Vc_INTRINSIC float32x4_t  broadcast16( float x) { return vdupq_n_f32(x); }
Vc_INTRINSIC float64x2_t broadcast16(double x) { return vdupq_n_f64(x); }
Vc_INTRINSIC int8x16_t broadcast16( schar x) { return vdupq_n_s8(x); }
Vc_INTRINSIC uint8x16_t broadcast16( uchar x) { return vdupq_n_u8(x); }
Vc_INTRINSIC int16x8_t broadcast16( short x) { return vdupq_n_s16(x); }
Vc_INTRINSIC uint16x8_t broadcast16(ushort x) { return vdupq_n_u16(x); }
Vc_INTRINSIC int32x4_t broadcast16(   int x) { return vdupq_n_s32(x); }
Vc_INTRINSIC uint32x4_t broadcast16(  uint x) { return vdupq_n_u32(x); }
//Vc_INTRINSIC int32x4_t broadcast16(  long x) { return sizeof( long) == 4 ? vdupq_n_s32(x) : vdupq_n_s64(x); }
//Vc_INTRINSIC int32x4_t broadcast16( ulong x) { return sizeof(ulong) == 4 ? vdupq_n_u32(x) : vdupq_n_u64(x); }

/*
// lowest16/32/64{{{1
template <class T>
Vc_INTRINSIC Vc_CONST typename intrinsic_type_impl<T, 16>::type lowest16()
{
    return broadcast16(std::numeric_limits<T>::lowest());
}

template <> Vc_INTRINSIC Vc_CONST int16x8_t lowest16< short>() { return  vld1q_s16(reinterpret_cast<const int32_t *>(neon_const::minShort)); }
template <> Vc_INTRINSIC Vc_CONST int32x4_t lowest16<   int>() { return  vld1q_s32(reinterpret_cast<const int32_t *>(neon_const::signMaskFloat)); }
template <> Vc_INTRINSIC Vc_CONST int64x2_t lowest16< llong>() { return  vld1q_s64(reinterpret_cast<const int64_t *>(neon_const::signMaskDouble)); }
template <> Vc_INTRINSIC Vc_CONST int32x4_t lowest16<  long>() { return lowest16<equal_int_type_t<long>>(); }

template <> Vc_INTRINSIC Vc_CONST uint8x16_t lowest16< uchar>() { return vdupq_n_u8(0); }
template <> Vc_INTRINSIC Vc_CONST uint16x8_t lowest16<ushort>() { return vdupq_n_u16(0); }
template <> Vc_INTRINSIC Vc_CONST uint32x4_t lowest16<  uint>() { return vdupq_n_u32(0); }
template <> Vc_INTRINSIC Vc_CONST uint64x2_t lowest16< ulong>() { return vdupq_n_u64(0); }

// _2_pow_31{{{1
template <class T> inline typename intrinsic_type_impl<T, 16>::type neon_2_pow_31();
template <> Vc_INTRINSIC Vc_CONST float32x4_t  neon_2_pow_31< float>() { return broadcast16( float(1u << 31)); }
template <> Vc_INTRINSIC Vc_CONST float64x2_t neon_2_pow_31<double>() { return broadcast16(double(1u << 31)); }
template <> Vc_INTRINSIC Vc_CONST int32x4_t neon_2_pow_31<  uint>() { return lowest16<int>(); }
*/

// blend{{{1
Vc_INTRINSIC Vc_CONST float32x4_t blend(float32x4_t mask, float32x4_t at0, float32x4_t at1){}
Vc_INTRINSIC Vc_CONST float64x2_t blend(float64x2_t mask, float64x4_t at0, float64x2_t at1){}
Vc_INTRINSIC Vc_CONST int32x4_t blend(int32x4_t mask, int32x4_t at0, int32x4_t at1){}

// NEON intrinsics emulation{{{1

// Bit compare{{{1

// movemask{{{1
Vc_INTRINSIC Vc_CONST int movemask_f32(float32x4_t a){}
Vc_INTRINSIC Vc_CONST int movemask_f64(float64x2_t a){}
Vc_INTRINSIC Vc_CONST int movemask_s32(int32x4_t a){}
Vc_INTRINSIC Vc_CONST int movemask_s16(int16x8_t a){}
Vc_INTRINSIC Vc_CONST int movemask_s8(int8x16_t a){}
/*
Vc_INTRINSIC Vc_CONST int movemask_s8(int8x16_t a)
{
    uint8x16_t input = (uint8x16_t)_a;
    const int8_t __attribute__((aligned(16))) xr[8] = { -7, -6, -5, -4, -3, -2, -1, 0 };
    uint8x8_t mask_and = vdup_n_u8(0x80);
    int8x8_t mask_shift = vld1_s8(xr);
    uint8x8_t lo = vget_low_u8(input);
    uint8x8_t hi = vget_high_u8(input);
    lo = vand_u8(lo, mask_and);
    lo = vshl_u8(lo, mask_shift);
    hi = vand_u8(hi, mask_and);
    hi = vshl_u8(hi, mask_shift);
    lo = vpadd_u8(lo, lo);
    lo = vpadd_u8(lo, lo);
    lo = vpadd_u8(lo, lo);
    hi = vpadd_u8(hi, hi);
    hi = vpadd_u8(hi, hi);
    hi = vpadd_u8(hi, hi);
    return ((hi[0] << 8) | (lo[0] & 0xFF));
}
*/
// negate{{{1
Vc_ALWAYS_INLINE Vc_CONST int32x4_t negate(int32x4_t v, std::integral_constant<std::size_t, 4>)
{
	return vnegq_s32(v);
}
Vc_ALWAYS_INLINE Vc_CONST int32x4_t negate(int32x4_t v, std::integral_constant<std::size_t, 2>)
{
	return vnegq_s32(v);
}

// xor_{{{1
Vc_INTRINSIC int64x2_t xor_(int64x2_t a, int64x2_t b) { return veorq_s64(a, b); }
Vc_INTRINSIC int32x4_t xor_(int32x4_t a, int32x4_t b) { return veorq_s32(a, b); }

// or_{{{1
Vc_INTRINSIC int64x2_t or_(int64x2_t a, int64x2_t b) { return vorrq_s64(a, b); }
Vc_INTRINSIC int32x4_t or_(int32x4_t a, int32x4_t b) { return vorrq_s32(a, b); }

// and_{{{1
Vc_INTRINSIC int64x2_t and_(int64x2_t a, int64x2_t b) { return vandq_s64(a, b); }
Vc_INTRINSIC int32x4_t and_(int32x4_t a, int32x4_t b) { return vandq_s32(a, b); }

// andnot_{{{1
Vc_INTRINSIC int64x2_t andnot_(int64x2_t a, int64x2_t b) { return vbicq_s64(a, b); }
Vc_INTRINSIC int32x4_t andnot_(int32x4_t a, int32x4_t b) { return vbicq_s32(a, b); }

// shift_left{{{1
template <int n> Vc_INTRINSIC int64x2_t shift_left(int64x2_t v) { return vshlq_n_s64(v, n); }
template <int n> Vc_INTRINSIC int32x4_t shift_left(int32x4_t v) { return vshlq_n_s32(v, n); }

// shift_right{{{1
template <int n> Vc_INTRINSIC int64x2_t shift_right(int64x2_t v) { return vshrq_n_s64(v, n); }
template <int n> Vc_INTRINSIC int32x4_t shift_right(int32x4_t v) { return vshrq_n_s32(v, n); }

// popcnt{{{1
// Not available for arm NEON?
Vc_INTRINSIC Vc_CONST unsigned int popcnt4(unsigned int n) {}
Vc_INTRINSIC Vc_CONST unsigned int popcnt8(unsigned int n) {}
Vc_INTRINSIC Vc_CONST unsigned int popcnt16(unsigned int n) {}
Vc_INTRINSIC Vc_CONST unsigned int popcnt32(unsigned int n) {}

// mask_count{{{1
template <size_t Size> int mask_count(float32x4_t );
template <size_t Size> int mask_count(int32x4_t);
template <size_t Size> int mask_count(float64x2_t);
template<> Vc_INTRINSIC Vc_CONST int mask_count<2>(float64x2_t k)
{
    int mask = movemask_f64(k);
    return (mask & 1) + (mask >> 1);
}
template<> Vc_INTRINSIC Vc_CONST int mask_count<2>(int32x4_t k) {}
template<> Vc_INTRINSIC Vc_CONST int mask_count<4>(float32x4_t k) {}
template<> Vc_INTRINSIC Vc_CONST int mask_count<4>(int32x4_t k) {}
template<> Vc_INTRINSIC Vc_CONST int mask_count<8>(int32x4_t k) {}
template<> Vc_INTRINSIC Vc_CONST int mask_count<16>(int32x4_t k) {}

// mask_to_int{{{1
template <size_t Size> inline int mask_to_int(int32x4_t) { static_assert(Size == Size, "Size value not implemented"); return 0; }
template<> Vc_INTRINSIC Vc_CONST int mask_to_int<2>(int32x4_t k) {}
template<> Vc_INTRINSIC Vc_CONST int mask_to_int<4>(int32x4_t k) {}
template<> Vc_INTRINSIC Vc_CONST int mask_to_int<8>(int32x4_t k) {}
template<> Vc_INTRINSIC Vc_CONST int mask_to_int<16>(int32x4_t k) {}

// is_equal{{{1
template <size_t Size>
Vc_INTRINSIC_L Vc_CONST_L bool is_equal(float32x4_t, float32x4_t) Vc_INTRINSIC_R Vc_CONST_R;
template <size_t Size>
Vc_INTRINSIC_L Vc_CONST_L bool is_not_equal(float32x4_t, float32x4_t) Vc_INTRINSIC_R Vc_CONST_R;
// TBD: should be changed according new changes from TC
template <> Vc_INTRINSIC Vc_CONST bool is_equal<2>(float32x4_t k1, float32x4_t k2)
{
    //return movemask_f64(*(const float64x2_t *)&(k1)) == movemask_f64(*(const float64x2_t *)&(k2));
}
template <> Vc_INTRINSIC Vc_CONST bool is_not_equal<2>(float32x4_t k1, float32x4_t k2)
{
    //return movemask_f64(*(const float64x2_t *)&(k1)) != movemask_f64(*(const float64x2_t *)&(k2));
}

template <> Vc_INTRINSIC Vc_CONST bool is_equal<4>(float32x4_t k1, float32x4_t k2)
{
    //return movemask_f32(k1) == movemask_f32(k2);
}
template <> Vc_INTRINSIC Vc_CONST bool is_not_equal<4>(float32x4_t k1, float32x4_t k2)
{
    //return movemask_f32(k1) != movemask_f32(k2);
}

template <> Vc_INTRINSIC Vc_CONST bool is_equal<8>(float32x4_t k1, float32x4_t k2)
{
    //return movemask_s16(*(const int16x8_t *)&(k1)) ==
    //       movemask_s16(*(const int16x8_t  *)&(k2));
}
template <> Vc_INTRINSIC Vc_CONST bool is_not_equal<8>(float32x4_t k1, float32x4_t k2)
{
    //return movemask_s16(*(const int16x8_t *)&(k1)) !=
    //       movemask_s16(*(const int16x8t_t *)&(k2));
}

template <> Vc_INTRINSIC Vc_CONST bool is_equal<16>(float32x4_t k1, float32x4_t k2)
{
    //return movemask_s16(*(const int8x16_t *)&(k1)) ==
    //       movemask_s8(*(const int8x16_t *)&(k2));
}
template <> Vc_INTRINSIC Vc_CONST bool is_not_equal<16>(float32x4_t k1, float32x4_t k2)
{
    //return movemask_s8(*(const int8x16_t  *)&(k1)) !=
    //       movemask_s8(*(const int8x16_t  *)&(k2));
}

// loads{{{1
/**
 * \internal
 *
 */
Vc_INTRINSIC float32x4_t load16(const float *mem, when_aligned<16>)
{
    return vld1q_f32(mem);
}
// for neon, alignment doesn't matter
Vc_INTRINSIC float32x4_t load16(const float *mem, when_unaligned<16>)
{
    return vld1q_f32(mem);
}

Vc_INTRINSIC float64x2_t load16(const double *mem, when_aligned<16>)
{
    return vld1q_f64(mem);
}
// for neon, alignment doesn't matter
Vc_INTRINSIC float64x2_t load16(const double *mem, when_unaligned<16>)
{
    return vld1q_f64(mem);
}

template <class T> Vc_INTRINSIC int32x4_t load16(const T *mem, when_aligned<16>)
{
    static_assert(std::is_integral<T>::value, "load16<T> is only intended for integral T");
    return vld1q_s32(reinterpret_cast<const int32_t *>(mem));
}

template <class T> Vc_INTRINSIC int32x4_t load16(const T *mem, when_unaligned<16>)
{
    static_assert(std::is_integral<T>::value, "load16<T> is only intended for integral T");
    return vld1q_s32(reinterpret_cast<const int32_t *>(mem));
}

// stores{{{1
template <class Flags> Vc_INTRINSIC void store4(float32x4_t v, float *mem, Flags)
{
    *mem = vgetq_lane_f32(v, 0);
}
template <class Flags> Vc_INTRINSIC void store8(float32x4_t v, float *mem, Flags) {}
Vc_INTRINSIC void store16(float32x4_t v, float *mem, when_aligned<16>)
{
    vst1q_f32(mem, v);
}
Vc_INTRINSIC void store16(float32x4_t v, float *mem, when_unaligned<16>)
{
    vst1q_f32(mem, v);
}

template <class Flags> Vc_INTRINSIC void store8(float64x2_t v, double *mem, Flags) {}

Vc_INTRINSIC void store16(float64x2_t v, double *mem, when_aligned<16>)
{
    vst1q_f64(mem, v);
}
Vc_INTRINSIC void store16(float64x2_t v, double *mem, when_unaligned<16>)
{
    vst1q_f64(mem, v);
}

template <class T, class Flags> Vc_INTRINSIC void store2(int32x4_t v, T *mem, Flags)
{
    static_assert(std::is_integral<T>::value && sizeof(T) <= 2,
                  "store4<T> is only intended for integral T with sizeof(T) <= 2");
    *reinterpret_cast<may_alias<ushort> *>(mem) = uint(vgetq_lane_s32(v, 0));
}

template <class T, class Flags> Vc_INTRINSIC void store4(int32x4_t v, T *mem, Flags)
{
    static_assert(std::is_integral<T>::value && sizeof(T) <= 4,
                  "store4<T> is only intended for integral T with sizeof(T) <= 4");
    *reinterpret_cast<may_alias<int> *>(mem) = vgetq_lane_s32(v, 0);
}

template <class T, class Flags> Vc_INTRINSIC void store8(int32x4_t v, T *mem, Flags)
{
    static_assert(std::is_integral<T>::value, "store8<T> is only intended for integral T");
    *mem = (int32x4_t)vsetq_lane_s64((int64_t)vget_low_s32(v), *(int64x2_t*)v, 0);
}

template <class T> Vc_INTRINSIC void store16(int32x4_t v, T *mem, when_aligned<16>)
{
    static_assert(std::is_integral<T>::value, "store16<T> is only intended for integral T");
    vst1q_s32(reinterpret_cast<int32_t *>(mem), v);
}
template <class T> Vc_INTRINSIC void store16(int32x4_t v, T *mem, when_unaligned<16>)
{
    static_assert(std::is_integral<T>::value, "store16<T> is only intended for integral T");
    vst1q_s32(reinterpret_cast<int32_t *>(mem), v);
}


// }}}1
}  // namespace aarch
using namespace aarch;
}  // namespace detail
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_DATAPAR_AARCH_H_
