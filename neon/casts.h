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

#ifndef VC_NEON_CASTS_H_
#define VC_NEON_CASTS_H_

#include "intrinsics.h"
#include "types.h"
#include "macros.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace NEON
{
using uint = unsigned int;
using ushort = unsigned short;
using uchar = unsigned char;
using schar = signed char;

// neon_cast {{{1
template <typename To, typename From> Vc_ALWAYS_INLINE Vc_CONST To neon_cast(From v)
{
    return v;
}
template<> Vc_ALWAYS_INLINE Vc_CONST int32x4_t neon_cast<int32x4_t, float32x4_t>(float32x4_t v) { return vreinterpretq_s32_f32(v); }
template<> Vc_ALWAYS_INLINE Vc_CONST int32x4_t neon_cast<int32x4_t, float64x2_t>(float64x2_t v) { return vreinterpretq_s32_f64(v); }
template<> Vc_ALWAYS_INLINE Vc_CONST float32x4_t  neon_cast<float32x4_t, float64x2_t>(float64x2_t v) { return vreinterpretq_f32_f64(v); }
template<> Vc_ALWAYS_INLINE Vc_CONST float32x4_t  neon_cast<float32x4_t, int32x4_t>(int32x4_t v) { return vreinterpretq_s32_f32(v); }
template<> Vc_ALWAYS_INLINE Vc_CONST float64x2_t neon_cast<float64x2_t, int32x4_t>(int32x4_t v) { return vreinterpretq_f64_s2(v); }
template<> Vc_ALWAYS_INLINE Vc_CONST float64x2_t neon_cast<float64x2_t, float32x4_t >(float32x4_t  v) { return vreinterpretq_f64_f32(v);    }
   
// convert {{{1
template <typename From, typename To> struct ConvertTag
{
};
template <typename From, typename To>
Vc_INTRINSIC typename VectorTraits<To>::VectorType convert(
    typename VectorTraits<From>::VectorType v)
{
    return convert(v, ConvertTag<From, To>());
}

Vc_INTRINSIC int32x4_t convert(float32x4_t  v, ConvertTag<float, int>) { return vcvtq_s32_f32(v); }
Vc_INTRINSIC int32x4_t convert(float64x2_t v, ConvertTag<double, int>) { return ; }
Vc_INTRINSIC int32x4_t convert(int32x4_t v, ConvertTag<int, int>) { return v; }
Vc_INTRINSIC int32x4_t convert(int32x4_t v, ConvertTag<uint, int>) { return v; }
Vc_INTRINSIC int32x4_t convert(int16x8_t v, ConvertTag<short, int>) { return ; }
Vc_INTRINSIC int32x4_t convert(uint16x8_t v, ConvertTag<ushort, int>) { return ; }
Vc_INTRINSIC uint32x4_t convert(float32x4_t v, ConvertTag<float, uint>) { return vcvtq_u32_f32(v); }
Vc_INTRINSIC uint32x4_t convert(float64x2_t v, ConvertTag<double, uint>) { return ; }
Vc_INTRINSIC uint32x4_t convert(int32x4_t v, ConvertTag<int, uint>) { return v; }
Vc_INTRINSIC uint32x4_t convert(uint32x4_t v, ConvertTag<uint, uint>) { return v; }
Vc_INTRINSIC uint32x4_t convert(int16x8_t v, ConvertTag<short, uint>) { return ; }
Vc_INTRINSIC uint32x4_t convert(uint16x8_t v, ConvertTag<ushort, uint>) { return ; }
Vc_INTRINSIC float32x4_t  convert(float32x4_t  v, ConvertTag<float, float>) { return v; }
Vc_INTRINSIC float32x4_t  convert(float64x2_t v, ConvertTag<double, float>) { return ; }
Vc_INTRINSIC float32x4_t  convert(int32x4_t v, ConvertTag<int, float >) { return vcvtq_f32_s32(v); }
Vc_INTRINSIC float32x4_t  convert(uint32x4_t v, ConvertTag<uint, float>) { return vcvtq_f32_u32(v); }
Vc_INTRINSIC float32x4_t  convert(int16x8_t v, ConvertTag<short , float >) { return convert(convert(v, ConvertTag<short, int>()), ConvertTag<int, float>()); }
Vc_INTRINSIC float32x4_t  convert(uint16x8_t v, ConvertTag<ushort, float >) { return convert(convert(v, ConvertTag<ushort, int>()), ConvertTag<int, float>()); }
Vc_INTRINSIC float64x2_t convert(float32x4_t  v, ConvertTag<float , double>) { return ; }
Vc_INTRINSIC float64x2_t convert(float64x2_t v, ConvertTag<double, double>) { return v; }
Vc_INTRINSIC float64x2_t convert(int32x4_t v, ConvertTag<int, double>) { return ; }
Vc_INTRINSIC float64x2_t convert(uint32x4_t v, ConvertTag<uint, double>) { return ; }
Vc_INTRINSIC float64x2_t convert(int16x8_t v, ConvertTag<short, double>) { return convert(convert(v, ConvertTag<short, int>()), ConvertTag<int, double>()); }
Vc_INTRINSIC float64x2_t convert(uint16x8_t v, ConvertTag<ushort, double>) { return convert(convert(v, ConvertTag<ushort, int>()), ConvertTag<int, double>()); }
Vc_INTRINSIC int16x8_t convert(float32x4_t  v, ConvertTag<float , short >) { return ; }
Vc_INTRINSIC int16x8_t convert(int32x4_t v, ConvertTag<int, short >) { return ; }
Vc_INTRINSIC int16x8_t convert(uint32x4_t v, ConvertTag<uint, short >) { return ; }
Vc_INTRINSIC int16x8_t convert(int16x8_t v, ConvertTag<short, short >) { return v; }
Vc_INTRINSIC int16x8_t convert(uint16x8_t v, ConvertTag<ushort, short >) { return v; }
Vc_INTRINSIC int16x8_t convert(float64x2_t v, ConvertTag<double, short >) { return ; }
Vc_INTRINSIC uint16x8_t convert(int32x4_t v, ConvertTag<int   , ushort>) {return ; }
Vc_INTRINSIC uint16x8_t convert(uint32x4_t v, ConvertTag<uint  , ushort>) { return ; }
Vc_INTRINSIC uint16x8_t convert(float32x4_t v, ConvertTag<float , ushort>) { return ; }
Vc_INTRINSIC uint16x8_t convert(int16x8_t v, ConvertTag<short , ushort>) { return v; }
Vc_INTRINSIC uint16x8_t convert(uint16x8_t v, ConvertTag<ushort, ushort>) { return v; }
Vc_INTRINSIC uint16x8_t convert(float64x2_t v, ConvertTag<double, ushort>) { return convert(convert(v, ConvertTag<double, int>()), ConvertTag<int, ushort>()); }

// }}}1 
}  // namespace Neon
Vc_VERSIONED_NAMESPACE_END

#endif // VC_NEON_CASTS_H_

// vim: foldmethod=marker
