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

#ifndef VC_DATAPAR_AARCH_ARITHMETICS_H_
#define VC_DATAPAR_AARCH_ARITHMETICS_H_

#include "storage.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
namespace aarch
{
// plus{{{1
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
template <class T, size_t N> Vc_INTRINSIC auto plus(Storage<T, N> a, Storage<T, N> b)
{
    return a.builtin() + b.builtin();
}
#else   // Vc_USE_BUILTIN_VECTOR_TYPES
Vc_INTRINSIC float64x2_t Vc_VDECL plus(x_f64 a, x_f64 b) { return vaddq_f64(a, b); }
Vc_INTRINSIC float32x4_t Vc_VDECL plus(x_f32 a, x_f32 b) { return vaddq_f32(a, b); }
Vc_INTRINSIC int8x16_t Vc_VDECL plus(x_s08 a, x_s08 b) { return vaddq_s8(a, b); }
Vc_INTRINSIC uint8x16_t Vc_VDECL plus(x_u08 a, x_u08 b) { return vaddq_u8(a, b); }
Vc_INTRINSIC int16x8_t Vc_VDECL plus(x_s16 a, x_s16 b) { return vaddq_s16(a, b); }
Vc_INTRINSIC uint16x8_t Vc_VDECL plus(x_u16 a, x_u16 b) { return vaddq_u16(a, b); }
Vc_INTRINSIC int32x4_t Vc_VDECL plus(x_s32 a, x_s32 b) { return vaddq_s32(a, b); }
Vc_INTRINSIC uint32x4_t Vc_VDECL plus(x_u32 a, x_u32 b) { return vaddq_u32(a, b); }
#endif  // Vc_USE_BUILTIN_VECTOR_TYPES

// minus{{{1
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
template <class T, size_t N> Vc_INTRINSIC auto minus(Storage<T, N> a, Storage<T, N> b)
{
    return a.builtin() - b.builtin();
}
#else   // Vc_USE_BUILTIN_VECTOR_TYPES
Vc_INTRINSIC float64x2_t Vc_VDECL minus(x_f64 a, x_f64 b) { return vsubq_f64(a, b); }
Vc_INTRINSIC float32x4_t Vc_VDECL minus(x_f32 a, x_f32 b) { return vsubq_f32(a, b); }
Vc_INTRINSIC int8x16_t Vc_VDECL minus(x_s08 a, x_s08 b) { return vsubq_s8(a, b); }
Vc_INTRINSIC uint8x16_t Vc_VDECL minus(x_u08 a, x_u08 b) { return vsubq_u8(a, b); }
Vc_INTRINSIC int16x8_t Vc_VDECL minus(x_s16 a, x_s16 b) { return vsubq_s16(a, b); }
Vc_INTRINSIC uint16x8_t Vc_VDECL minus(x_u16 a, x_u16 b) { return vsubq_u16(a, b); }
Vc_INTRINSIC int32x4_t Vc_VDECL minus(x_s32 a, x_s32 b) { return vsubq_s32(a, b); }
Vc_INTRINSIC uint32x4_t Vc_VDECL minus(x_u32 a, x_u32 b) { return vsubq_u32(a, b); }
#endif  // Vc_USE_BUILTIN_VECTOR_TYPES

// multiplies{{{1
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
template <class T, size_t N> Vc_INTRINSIC auto multiplies(Storage<T, N> a, Storage<T, N> b)
{
    return a.builtin() * b.builtin();
}
#else   // Vc_USE_BUILTIN_VECTOR_TYPES
Vc_INTRINSIC float64x2_t Vc_VDECL multiplies(x_f64 a, x_f64 b) { return vmulq_f64(a, b); }
Vc_INTRINSIC float32x4_t Vc_VDECL multiplies(x_f32 a, x_f32 b) { return vmulq_f32(a, b); }
Vc_INTRINSIC int8x16_t Vc_VDECL multiplies(x_s08 a, x_s08 b) { return vmulq_s8(a, b); }
Vc_INTRINSIC uint8x16_t Vc_VDECL multiplies(x_u08 a, x_u08 b) { return vmulq_u8(a, b); }
Vc_INTRINSIC int16x8_t Vc_VDECL multiplies(x_s16 a, x_s16 b) { return vmulq_s16(a, b); }
Vc_INTRINSIC uint16x8_t Vc_VDECL multiplies(x_u16 a, x_u16 b) { return vmulq_u16(a, b); }
Vc_INTRINSIC uint32x4_t Vc_VDECL multiplies(x_s32 a, x_s32 b) { return vmulq_s32(a, b); }
Vc_INTRINSIC int32x4_t Vc_VDECL multiplies(x_u32 a, x_u32 b) { return vmulq_u32(a, b); }
#endif  // Vc_USE_BUILTIN_VECTOR_TYPES

// divides{{{1
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
template <class T, size_t N> Vc_INTRINSIC auto divides(Storage<T, N> a, Storage<T, N> b)
{
    return a.builtin() / b.builtin();
}
#else   // Vc_USE_BUILTIN_VECTOR_TYPES
Vc_INTRINSIC float64x2_t Vc_VDECL divides(x_f64 a, x_f64 b) { return vmulq_f64(a, b); }
Vc_INTRINSIC float32x4_t Vc_VDECL divides(x_f32 a, x_f32 b) { return vmulq_f32(a, b); }
Vc_INTRINSIC int8x16_t Vc_VDECL divides(x_s08 a, x_s08 b) { }
Vc_INTRINSIC uint8x16_t Vc_VDECL divides(x_u08 a, x_u08 b) { }
Vc_INTRINSIC int16x8_t Vc_VDECL divides(x_s16 a, x_s16 b) { }
Vc_INTRINSIC uint16x8_t Vc_VDECL divides(x_u16 a, x_u16 b) { }
Vc_INTRINSIC int32x4_t Vc_VDECL divides(x_s32 a, x_s32 b) { }
Vc_INTRINSIC uint32x4_t Vc_VDECL divides(x_u32 a, x_u32 b) { }
#endif  // Vc_USE_BUILTIN_VECTOR_TYPES

#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
template <class T, size_t N> Vc_INTRINSIC auto modulus(Storage<T, N> a, Storage<T, N> b)
{
    static_assert(std::is_integral<T>::value, "modulus is only supported for integral types");
    return a.builtin() % b.builtin();
}
#else   // Vc_USE_BUILTIN_VECTOR_TYPES
template <class T, size_t N> Vc_INTRINSIC auto modulus(Storage<T, N> a, Storage<T, N> b)
{
    static_assert(std::is_integral<T>::value, "modulus is only supported for integral types");
    return minus(a, multiplies(divides(a, b), b));
}
#endif  // Vc_USE_BUILTIN_VECTOR_TYPES

// bit_and{{{1
template <class T, size_t N> Vc_INTRINSIC auto bit_and(Storage<T, N> a, Storage<T, N> b)
{
    static_assert(std::is_integral<T>::value, "bit_and is only supported for integral types");
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
    return a.builtin() & b.builtin();
#else   // Vc_USE_BUILTIN_VECTOR_TYPES
    return and_(a, b);
#endif  // Vc_USE_BUILTIN_VECTOR_TYPES
}

// bit_or{{{1
template <class T, size_t N> Vc_INTRINSIC auto bit_or(Storage<T, N> a, Storage<T, N> b)
{
    static_assert(std::is_integral<T>::value, "bit_or is only supported for integral types");
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
    return a.builtin() | b.builtin();
#else   // Vc_USE_BUILTIN_VECTOR_TYPES
    return or_(a, b);
#endif  // Vc_USE_BUILTIN_VECTOR_TYPES
}

// bit_xor{{{1
template <class T, size_t N> Vc_INTRINSIC auto bit_xor(Storage<T, N> a, Storage<T, N> b)
{
    static_assert(std::is_integral<T>::value, "bit_xor is only supported for integral types");
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
    return a.builtin() ^ b.builtin();
#else   // Vc_USE_BUILTIN_VECTOR_TYPES
    return xor_(a, b);
#endif  // Vc_USE_BUILTIN_VECTOR_TYPES
}

// bit_shift_left{{{1
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
template <class T, size_t N> Vc_INTRINSIC auto bit_shift_left(Storage<T, N> a, Storage<T, N> b)
{
    static_assert(std::is_integral<T>::value, "bit_shift_left is only supported for integral types");
    return a.builtin() << b.builtin();
}
#else   // Vc_USE_BUILTIN_VECTOR_TYPES
// generic scalar fallback
template <class T, size_t N> Vc_INTRINSIC auto bit_shift_left(Storage<T, N> a, Storage<T, N> b)
{
    static_assert(std::is_integral<T>::value, "bit_shift_left is only supported for integral types");
    return generate_from_n_evaluations<N, Storage<T, N>>(
        [&](auto i) { return a[i] << b[i]; });
}
#endif

#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
template <class T, size_t N> Vc_INTRINSIC auto bit_shift_right(Storage<T, N> a, Storage<T, N> b)
{
    static_assert(std::is_integral<T>::value, "bit_shift_right is only supported for integral types");
    return a.builtin() >> b.builtin();
}
#else   // Vc_USE_BUILTIN_VECTOR_TYPES
// generic scalar fallback
template <class T, size_t N> Vc_INTRINSIC auto bit_shift_right(Storage<T, N> a, Storage<T, N> b)
{
    static_assert(std::is_integral<T>::value, "bit_shift_right is only supported for integral types");
    return generate_from_n_evaluations<N, Storage<T, N>>(
        [&](auto i) { return a[i] >> b[i]; });
}
#endif

// complement{{{1
template <typename T> Vc_INTRINSIC auto Vc_VDECL complement(T v) {
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
    return ~v.builtin();
#else
    return not_(v);
#endif
}

//}}}1
// unary_minus{{{1
template <typename T> Vc_INTRINSIC auto Vc_VDECL unary_minus(T v) { return minus(T{}, v); }
Vc_INTRINSIC float32x4_t  Vc_VDECL unary_minus(x_f32 v) {}
Vc_INTRINSIC float64x2_t Vc_VDECL unary_minus(x_f64 v) {}
Vc_INTRINSIC int32x4_t Vc_VDECL unary_minus(x_s32 v) {}
Vc_INTRINSIC int32x4_t Vc_VDECL unary_minus(x_u32 v) {}
Vc_INTRINSIC int32x4_t Vc_VDECL unary_minus(x_s16 v) {}
Vc_INTRINSIC int32x4_t Vc_VDECL unary_minus(x_u16 v) {}
Vc_INTRINSIC int32x4_t Vc_VDECL unary_minus(x_s08 v) {}
Vc_INTRINSIC int32x4_t Vc_VDECL unary_minus(x_u08 v) {}

//}}}1
}  // aarch
}  // detail
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_DATAPAR_AARCH_ARITHMETICS_H_
