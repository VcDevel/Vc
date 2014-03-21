/*  This file is part of the Vc library. {{{
Copyright © 2014 Matthias Kretz <kretz@kde.org>
All rights reserved.

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
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

}}}*/

#ifndef VC_AVX_SIMD_CAST_H
#define VC_AVX_SIMD_CAST_H

#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{

#define Vc_SIMD_CAST_1(from__, to__)                                                               \
    template <typename To>                                                                         \
    Vc_INTRINSIC To simd_cast(from__ x, enable_if<std::is_same<To, to__>::value> = nullarg)

#define Vc_SIMD_CAST_2(from__, to__)                                                               \
    template <typename To>                                                                         \
    Vc_INTRINSIC To                                                                                \
        simd_cast(from__ x0, from__ x1, enable_if<std::is_same<To, to__>::value> = nullarg)

#define Vc_SIMD_CAST_4(from__, to__)                                                               \
    template <typename To>                                                                         \
    Vc_INTRINSIC To simd_cast(from__ x0,                                                           \
                              from__ x1,                                                           \
                              from__ x2,                                                           \
                              from__ x3,                                                           \
                              enable_if<std::is_same<To, to__>::value> = nullarg)

Vc_SIMD_CAST_1(SSE::int_v, Vc_AVX_NAMESPACE::double_v) { return _mm256_cvtepi32_pd(x.data()); }

Vc_SIMD_CAST_2(Vc_AVX_NAMESPACE::double_v, Vc_AVX_NAMESPACE::int_v)
{
    return AVX::concat(_mm256_cvttpd_epi32(x0.data()), _mm256_cvttpd_epi32(x1.data()));
}

Vc_SIMD_CAST_2(SSE::int_v, Vc_AVX_NAMESPACE::float_v)
{
    return _mm256_cvtepi32_ps(AVX::concat(x0.data(), x1.data()));
}

Vc_SIMD_CAST_2(SSE::int_m, Vc_AVX_NAMESPACE::float_m)
{
    return AVX::concat(x0.data(), x1.data());
}

#undef Vc_SIMD_CAST_1
#undef Vc_SIMD_CAST_2
#undef Vc_SIMD_CAST_4

template <typename To, typename From>
inline To simd_cast(From x0, From x1, From x2, From x3, enable_if<sizeof(To) == 32> = nullarg)
{
    const auto y0 = static_cast<To>(x0);
    const auto y1 = static_cast<To>(x1);
    const auto y2 = static_cast<To>(x2);
    const auto y3 = static_cast<To>(x3);
    return AVX::concat(_mm_movelh_ps(y0, y1), _mm_movelh_ps(y2, y3));
}

// simd_cast with offset == 0 is implemented generically in sse/vector.tcc

// From::Size / To::Size is the number of simd_cast required to fully convert all parts of x.
// From::Size / To::Size / 2 thus is the offset where the upper 128bit half of the AVX vector is
// read
template <typename To, int offset, typename From>
inline To simd_cast(
    From x,
    enable_if<offset != 0 && (offset < From::Size / To::Size / 2) && sizeof(From) == 32> = nullarg)
{
    constexpr int shift = sizeof(typename From::VectorEntryType) * offset * To::Size;
    static_assert(shift >= 0 && shift < 16, "");
    return static_cast<To>(From(AVX::avx_cast<typename From::VectorType>(
        _mm_slli_si128(AVX::avx_cast<__m128i>(AVX::lo128(x.data())), shift))));
}

template <typename To, int offset, typename From>
inline To simd_cast(From x, enable_if<(offset >= From::Size / To::Size / 2) && sizeof(From) == 32> = nullarg)
{
    constexpr int shift = sizeof(typename From::VectorEntryType) * offset * To::Size - 16;
    static_assert(shift == 0 || (From::Size / To::Size > 2 && shift > 0 && shift < 16), "");
    if (shift == 0) {
        return static_cast<To>(From(AVX::avx_cast<typename From::VectorType>(AVX::hi128(x.data()))));
    } else {
        return static_cast<To>(From(AVX::avx_cast<typename From::VectorType>(
            _mm_slli_si128(AVX::avx_cast<__m128i>(AVX::hi128(x.data())), shift))));
    }
}

}  // namespace Vc

#include "undomacros.h"

#endif // VC_AVX_SIMD_CAST_H
