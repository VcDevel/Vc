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

#ifndef VC_AVX_VECTOR_H__
#error "Vc/avx/vector.h needs to be included before Vc/avx/simd_cast.h"
#endif
#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{

// helper macros Vc_SIMD_CAST_AVX_[124] & Vc_SIMD_CAST_[124] {{{1
#define Vc_SIMD_CAST_AVX_1(from__, to__)                                                           \
    template <typename To>                                                                         \
    Vc_INTRINSIC Vc_CONST To                                                                       \
        simd_cast(Vc_AVX_NAMESPACE::from__ x,                                                      \
                  enable_if<std::is_same<To, Vc_AVX_NAMESPACE::to__>::value> = nullarg)

#define Vc_SIMD_CAST_AVX_2(from__, to__)                                                           \
    template <typename To>                                                                         \
    Vc_INTRINSIC Vc_CONST To                                                                       \
        simd_cast(Vc_AVX_NAMESPACE::from__ x0,                                                     \
                  Vc_AVX_NAMESPACE::from__ x1,                                                     \
                  enable_if<std::is_same<To, Vc_AVX_NAMESPACE::to__>::value> = nullarg)

#define Vc_SIMD_CAST_AVX_4(from__, to__)                                                           \
    template <typename To>                                                                         \
    Vc_INTRINSIC Vc_CONST To                                                                       \
        simd_cast(Vc_AVX_NAMESPACE::from__ x0,                                                     \
                  Vc_AVX_NAMESPACE::from__ x1,                                                     \
                  Vc_AVX_NAMESPACE::from__ x2,                                                     \
                  Vc_AVX_NAMESPACE::from__ x3,                                                     \
                  enable_if<std::is_same<To, Vc_AVX_NAMESPACE::to__>::value> = nullarg)

#define Vc_SIMD_CAST_1(from__, to__)                                                               \
    template <typename To>                                                                         \
    Vc_INTRINSIC Vc_CONST To                                                                       \
        simd_cast(from__ x, enable_if<std::is_same<To, to__>::value> = nullarg)

#define Vc_SIMD_CAST_2(from__, to__)                                                               \
    template <typename To>                                                                         \
    Vc_INTRINSIC Vc_CONST To                                                                       \
        simd_cast(from__ x0, from__ x1, enable_if<std::is_same<To, to__>::value> = nullarg)

#define Vc_SIMD_CAST_4(from__, to__)                                                               \
    template <typename To>                                                                         \
    Vc_INTRINSIC Vc_CONST To simd_cast(from__ x0,                                                  \
                                       from__ x1,                                                  \
                                       from__ x2,                                                  \
                                       from__ x3,                                                  \
                                       enable_if<std::is_same<To, to__>::value> = nullarg)

// Vector casts without offset {{{1
// AVX::Vector {{{2
Vc_SIMD_CAST_AVX_1( float_v,    int_v) { return _mm256_cvttps_epi32(x.data()); }
Vc_SIMD_CAST_AVX_1(double_v,    int_v) { return AVX::zeroExtend(_mm256_cvttpd_epi32(x.data())); }
Vc_SIMD_CAST_AVX_2(double_v,    int_v) { return AVX::concat(_mm256_cvttpd_epi32(x0.data()), _mm256_cvttpd_epi32(x1.data())); }
Vc_SIMD_CAST_AVX_1(  uint_v,    int_v) { return x.data(); }
Vc_SIMD_CAST_AVX_1( short_v,    int_v) {
    return AVX::concat(_mm_srai_epi32(_mm_unpacklo_epi16(x.data(), x.data()), 16),
                       _mm_srai_epi32(_mm_unpackhi_epi16(x.data(), x.data()), 16));
}
Vc_SIMD_CAST_AVX_1(ushort_v,    int_v) {
    return AVX::concat(_mm_unpacklo_epi16(x.data(), _mm_setzero_si128()),
                       _mm_unpackhi_epi16(x.data(), _mm_setzero_si128()));
}

Vc_SIMD_CAST_AVX_1( float_v,   uint_v) {
    using namespace AvxIntrinsics;
    return _mm256_castps_si256(
        _mm256_blendv_ps(_mm256_castsi256_ps(_mm256_cvttps_epi32(x.data())),
                         _mm256_castsi256_ps(_mm256_add_epi32(
                             _mm256_cvttps_epi32(_mm256_sub_ps(x.data(), _mm256_set1_ps(1u << 31))),
                             _mm256_set1_epi32(1 << 31))),
                         _mm256_cmpge_ps(x.data(), _mm256_set1_ps(1u << 31))));
}
Vc_SIMD_CAST_AVX_1(double_v,   uint_v) {
    return AVX::zeroExtend(
        _mm_add_epi32(_mm256_cvttpd_epi32(_mm256_sub_pd(x.data(), _mm256_set1_pd(0x80000000u))),
                      _mm_set1_epi32(0x80000000u)));
}
Vc_SIMD_CAST_AVX_2(double_v,   uint_v) {
    const auto lo = _mm256_cvttpd_epi32(_mm256_sub_pd(x0.data(), _mm256_set1_pd(0x80000000u)));
    const auto hi = _mm256_cvttpd_epi32(_mm256_sub_pd(x1.data(), _mm256_set1_pd(0x80000000u)));
    const auto offset = _mm_set1_epi32(0x80000000u);
    return AVX::concat(_mm_add_epi32(_mm_unpacklo_epi64(lo, hi), offset),
                       _mm_add_epi32(_mm_unpackhi_epi64(lo, hi), offset));
}
Vc_SIMD_CAST_AVX_1(   int_v,   uint_v) { return x.data(); }
Vc_SIMD_CAST_AVX_1( short_v,   uint_v) {
    // the conversion rule is x mod 2^32
    // and the definition of mod here is the one that yields only positive numbers
    return AVX::concat(_mm_srai_epi32(_mm_unpacklo_epi16(x.data(), x.data()), 16),
                       _mm_srai_epi32(_mm_unpackhi_epi16(x.data(), x.data()), 16));
}
Vc_SIMD_CAST_AVX_1(ushort_v,   uint_v) {
    return AVX::concat(_mm_unpacklo_epi16(x.data(), _mm_setzero_si128()),
                       _mm_unpackhi_epi16(x.data(), _mm_setzero_si128()));
}

Vc_SIMD_CAST_AVX_1(   int_v,  short_v) {
    auto tmp0 =
        _mm_unpacklo_epi16(AVX::lo128(x.data()), AVX::hi128(x.data()));  // 0 4 X X 1 5 X X
    auto tmp1 =
        _mm_unpackhi_epi16(AVX::lo128(x.data()), AVX::hi128(x.data()));  // 2 6 X X 3 7 X X
    auto tmp2 = _mm_unpacklo_epi16(tmp0, tmp1);                            // 0 2 4 6 X X X X
    auto tmp3 = _mm_unpackhi_epi16(tmp0, tmp1);                            // 1 3 5 7 X X X X
    return _mm_unpacklo_epi16(tmp2, tmp3);                                 // 0 1 2 3 4 5 6 7
}
Vc_SIMD_CAST_AVX_1(  uint_v,  short_v) {
    auto tmp0 =
        _mm_unpacklo_epi16(AVX::lo128(x.data()), AVX::hi128(x.data()));  // 0 4 X X 1 5 X X
    auto tmp1 =
        _mm_unpackhi_epi16(AVX::lo128(x.data()), AVX::hi128(x.data()));  // 2 6 X X 3 7 X X
    auto tmp2 = _mm_unpacklo_epi16(tmp0, tmp1);                            // 0 2 4 6 X X X X
    auto tmp3 = _mm_unpackhi_epi16(tmp0, tmp1);                            // 1 3 5 7 X X X X
    return _mm_unpacklo_epi16(tmp2, tmp3);                                 // 0 1 2 3 4 5 6 7
}
Vc_SIMD_CAST_AVX_1(float_v, short_v) {
    const auto tmp = _mm256_cvttps_epi32(x.data());
    return _mm_packs_epi32(AVX::lo128(tmp), AVX::hi128(tmp));
}
Vc_SIMD_CAST_AVX_1(double_v, short_v) {
    const auto tmp = _mm256_cvttpd_epi32(x.data());
    return _mm_packs_epi32(tmp, _mm_setzero_si128());
}
Vc_SIMD_CAST_AVX_2(double_v, short_v) {
    const auto tmp0 = _mm256_cvttpd_epi32(x0.data());
    const auto tmp1 = _mm256_cvttpd_epi32(x1.data());
    return _mm_packs_epi32(tmp0, tmp1);
}
Vc_SIMD_CAST_AVX_1(ushort_v,  short_v) { return x.data(); }

Vc_SIMD_CAST_AVX_1(   int_v, ushort_v) {
    auto tmp0 =
        _mm_unpacklo_epi16(AVX::lo128(x.data()), AVX::hi128(x.data()));  // 0 4 X X 1 5 X X
    auto tmp1 =
        _mm_unpackhi_epi16(AVX::lo128(x.data()), AVX::hi128(x.data()));  // 2 6 X X 3 7 X X
    auto tmp2 = _mm_unpacklo_epi16(tmp0, tmp1);                            // 0 2 4 6 X X X X
    auto tmp3 = _mm_unpackhi_epi16(tmp0, tmp1);                            // 1 3 5 7 X X X X
    return _mm_unpacklo_epi16(tmp2, tmp3);                                 // 0 1 2 3 4 5 6 7
}
Vc_SIMD_CAST_AVX_1(  uint_v, ushort_v) {
    auto tmp0 =
        _mm_unpacklo_epi16(AVX::lo128(x.data()), AVX::hi128(x.data()));  // 0 4 X X 1 5 X X
    auto tmp1 =
        _mm_unpackhi_epi16(AVX::lo128(x.data()), AVX::hi128(x.data()));  // 2 6 X X 3 7 X X
    auto tmp2 = _mm_unpacklo_epi16(tmp0, tmp1);                            // 0 2 4 6 X X X X
    auto tmp3 = _mm_unpackhi_epi16(tmp0, tmp1);                            // 1 3 5 7 X X X X
    return _mm_unpacklo_epi16(tmp2, tmp3);                                 // 0 1 2 3 4 5 6 7
}
Vc_SIMD_CAST_AVX_1(float_v, ushort_v) {
    const auto tmp = _mm256_cvttps_epi32(x.data());
    return _mm_packs_epi32(AVX::lo128(tmp), AVX::hi128(tmp));
}
Vc_SIMD_CAST_AVX_1(double_v, ushort_v) {
    const auto tmp = _mm256_cvttpd_epi32(x.data());
    return _mm_packs_epi32(tmp, _mm_setzero_si128());
}
Vc_SIMD_CAST_AVX_2(double_v, ushort_v) {
    const auto tmp0 = _mm256_cvttpd_epi32(x0.data());
    const auto tmp1 = _mm256_cvttpd_epi32(x1.data());
    return _mm_packs_epi32(tmp0, tmp1);
}
Vc_SIMD_CAST_AVX_1( short_v, ushort_v) { return x.data(); }

Vc_SIMD_CAST_AVX_1(double_v,  float_v) { return AVX::zeroExtend(_mm256_cvtpd_ps(x.data())); }
Vc_SIMD_CAST_AVX_2(double_v,  float_v) { return AVX::concat(_mm256_cvtpd_ps(x0.data()), _mm256_cvtpd_ps(x1.data())); }
Vc_SIMD_CAST_AVX_1(   int_v,  float_v) { return _mm256_cvtepi32_ps(x.data()); }
Vc_SIMD_CAST_AVX_1(  uint_v,  float_v) {
    using namespace AvxIntrinsics;
    return _mm256_blendv_ps(
        _mm256_cvtepi32_ps(x.data()),
        _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(x.data(), _mm256_setmin_epi32())),
                      _mm256_set1_ps(1u << 31)),
        _mm256_castsi256_ps(_mm256_cmplt_epi32(x.data(), _mm256_setzero_si256())));
}
Vc_SIMD_CAST_AVX_1( short_v,  float_v) { return simd_cast<Vc_AVX_NAMESPACE::float_v>(simd_cast<Vc_AVX_NAMESPACE::int_v>(x)); }
Vc_SIMD_CAST_AVX_1(ushort_v,  float_v) { return simd_cast<Vc_AVX_NAMESPACE::float_v>(simd_cast<Vc_AVX_NAMESPACE::int_v>(x)); }

Vc_SIMD_CAST_AVX_1( float_v, double_v) { return _mm256_cvtps_pd(AVX::lo128(x.data())); }
Vc_SIMD_CAST_AVX_1(   int_v, double_v) { return _mm256_cvtepi32_pd(AVX::lo128(x.data())); }
Vc_SIMD_CAST_AVX_1(  uint_v, double_v) {
    using namespace AvxIntrinsics;
    return _mm256_add_pd(_mm256_cvtepi32_pd(_mm_sub_epi32(AVX::lo128(x.data()), _mm_setmin_epi32())),
                      _mm256_set1_pd(1u << 31));
}
Vc_SIMD_CAST_AVX_1( short_v, double_v) { return simd_cast<Vc_AVX_NAMESPACE::double_v>(simd_cast<Vc_AVX_NAMESPACE::int_v>(x)); }
Vc_SIMD_CAST_AVX_1(ushort_v, double_v) { return simd_cast<Vc_AVX_NAMESPACE::double_v>(simd_cast<Vc_AVX_NAMESPACE::int_v>(x)); }

// SSE::Vector to AVX::Vector {{{2
Vc_SIMD_CAST_1(SSE::int_v, Vc_AVX_NAMESPACE:: short_v) { return simd_cast<SSE::short_v>(x).data(); }
Vc_SIMD_CAST_2(SSE::int_v, Vc_AVX_NAMESPACE:: short_v) { return simd_cast<SSE::short_v>(x0, x1).data(); }
Vc_SIMD_CAST_1(SSE::int_v, Vc_AVX_NAMESPACE::ushort_v) { return simd_cast<SSE::ushort_v>(x).data(); }
Vc_SIMD_CAST_2(SSE::int_v, Vc_AVX_NAMESPACE::ushort_v) { return simd_cast<SSE::ushort_v>(x0, x1).data(); }
Vc_SIMD_CAST_1(SSE::int_v, Vc_AVX_NAMESPACE::double_v) { return _mm256_cvtepi32_pd(x.data()); }
Vc_SIMD_CAST_1(SSE::int_v, Vc_AVX_NAMESPACE:: float_v) { return AVX::zeroExtend(_mm_cvtepi32_ps(x.data())); }
Vc_SIMD_CAST_2(SSE::int_v, Vc_AVX_NAMESPACE:: float_v) { return _mm256_cvtepi32_ps(AVX::concat(x0.data(), x1.data())); }

// AVX::Vector to SSE::Vector {{{2
Vc_SIMD_CAST_1(Vc_AVX_NAMESPACE:: float_v, SSE::int_v) { return simd_cast<SSE::int_v>(SSE::float_v{AVX::lo128(x.data())}); }

// AVX::Vector to Scalar::Vector {{{2
#define Vc_SIMD_CAST_AVX_TO_SCALAR(to__)                                                 \
    template <typename To, typename FromT>                                               \
    Vc_INTRINSIC Vc_CONST To                                                             \
        simd_cast(Vc_AVX_NAMESPACE::Vector<FromT> x,                                     \
                  enable_if<std::is_same<To, Scalar::to__>::value> = nullarg)            \
    {                                                                                    \
        return static_cast<To>(x[0]);                                                    \
    }

VC_ALL_VECTOR_TYPES(Vc_SIMD_CAST_AVX_TO_SCALAR)
#undef Vc_SIMD_CAST_AVX_TO_SCALAR

// 1 simdarray to 1 AVX::Vector {{{2
template <typename Return, typename T, std::size_t N, typename V>
Vc_INTRINSIC Vc_CONST Return
    simd_cast(const simdarray<T, N, V, N> &x,
              enable_if<AVX2::is_vector<Return>::value || AVX::is_vector<Return>::value> = nullarg)
{
    return simd_cast<Return>(internal_data(x));
}

template <typename Return, typename T, std::size_t N, typename V>
Vc_INTRINSIC Vc_CONST Return
    simd_cast(const simdarray<T, 2 * N, V, N> &x,
              enable_if<AVX2::is_vector<Return>::value || AVX::is_vector<Return>::value> = nullarg)
{
    return simd_cast<Return>(internal_data(internal_data0(x)), internal_data(internal_data1(x)));
}

template <typename Return, typename T, std::size_t N, typename V>
Vc_INTRINSIC Vc_CONST Return
    simd_cast(const simdarray<T, 4 * N, V, N> &x,
              enable_if<AVX2::is_vector<Return>::value || AVX::is_vector<Return>::value> = nullarg)
{
    return simd_cast<Return>(internal_data(internal_data0(internal_data0(x))),
                             internal_data(internal_data1(internal_data0(x))),
                             internal_data(internal_data0(internal_data1(x))),
                             internal_data(internal_data1(internal_data1(x))));
}

// Mask casts without offset {{{1
// 1 AVX::Mask to 1 AVX::Mask {{{2
template <typename Return, typename T>
Vc_INTRINSIC Vc_CONST Return
    simd_cast(const Vc_AVX_NAMESPACE::Mask<T> &k,
              enable_if<AVX::is_mask<Return>::value || AVX2::is_mask<Return>::value> = nullarg)
{
    return {Vc_AVX_NAMESPACE::internal::mask_cast<Vc_AVX_NAMESPACE::Mask<T>::Size,
                                                  Return::Size,
                                                  typename Return::VectorType>(k.dataI())};
}

// 2 AVX::Mask to 1 AVX::Mask {{{2
Vc_SIMD_CAST_AVX_2(double_m,  float_m) { return AVX::concat(_mm_packs_epi32(AVX::lo128(x0.dataI()), AVX::hi128(x0.dataI())), _mm_packs_epi32(AVX::lo128(x1.dataI()), AVX::hi128(x1.dataI()))); }
Vc_SIMD_CAST_AVX_2(double_m,    int_m) { return AVX::concat(_mm_packs_epi32(AVX::lo128(x0.dataI()), AVX::hi128(x0.dataI())), _mm_packs_epi32(AVX::lo128(x1.dataI()), AVX::hi128(x1.dataI()))); }
Vc_SIMD_CAST_AVX_2(double_m,   uint_m) { return AVX::concat(_mm_packs_epi32(AVX::lo128(x0.dataI()), AVX::hi128(x0.dataI())), _mm_packs_epi32(AVX::lo128(x1.dataI()), AVX::hi128(x1.dataI()))); }
Vc_SIMD_CAST_AVX_2(double_m,  short_m) { return _mm_packs_epi16(_mm_packs_epi32(AVX::lo128(x0.dataI()), AVX::hi128(x0.dataI())), _mm_packs_epi32(AVX::lo128(x1.dataI()), AVX::hi128(x1.dataI()))); }
Vc_SIMD_CAST_AVX_2(double_m, ushort_m) { return _mm_packs_epi16(_mm_packs_epi32(AVX::lo128(x0.dataI()), AVX::hi128(x0.dataI())), _mm_packs_epi32(AVX::lo128(x1.dataI()), AVX::hi128(x1.dataI()))); }

// 1 SSE::Mask to 1 AVX(2)::Mask {{{2
Vc_SIMD_CAST_1(SSE::double_m, Vc_AVX_NAMESPACE::double_m) { return AVX::zeroExtend(x.data()); }
Vc_SIMD_CAST_1(SSE::double_m, Vc_AVX_NAMESPACE:: float_m) { return AVX::zeroExtend(simd_cast<SSE:: float_m>(x).data()); }
Vc_SIMD_CAST_1(SSE::double_m, Vc_AVX_NAMESPACE::   int_m) { return AVX::zeroExtend(simd_cast<SSE::   int_m>(x).data()); }
Vc_SIMD_CAST_1(SSE::double_m, Vc_AVX_NAMESPACE::  uint_m) { return AVX::zeroExtend(simd_cast<SSE::  uint_m>(x).data()); }
Vc_SIMD_CAST_1(SSE::double_m, AVX             :: short_m) { return simd_cast<SSE:: short_m>(x).data(); }
Vc_SIMD_CAST_1(SSE::double_m, AVX             ::ushort_m) { return simd_cast<SSE::ushort_m>(x).data(); }

Vc_SIMD_CAST_1(SSE:: float_m, Vc_AVX_NAMESPACE::double_m) { return AVX::concat(_mm_unpacklo_ps(x.data(), x.data()), _mm_unpackhi_ps(x.data(), x.data())); }
Vc_SIMD_CAST_1(SSE::   int_m, Vc_AVX_NAMESPACE::double_m) { return AVX::concat(_mm_unpacklo_ps(x.data(), x.data()), _mm_unpackhi_ps(x.data(), x.data())); }
Vc_SIMD_CAST_1(SSE::  uint_m, Vc_AVX_NAMESPACE::double_m) { return AVX::concat(_mm_unpacklo_ps(x.data(), x.data()), _mm_unpackhi_ps(x.data(), x.data())); }
Vc_SIMD_CAST_1(SSE:: short_m, Vc_AVX_NAMESPACE::double_m) { auto tmp = _mm_unpacklo_epi16(x.dataI(), x.dataI()); return AVX::concat(_mm_unpacklo_epi32(tmp, tmp), _mm_unpackhi_epi32(tmp, tmp)); }
Vc_SIMD_CAST_1(SSE::ushort_m, Vc_AVX_NAMESPACE::double_m) { auto tmp = _mm_unpacklo_epi16(x.dataI(), x.dataI()); return AVX::concat(_mm_unpacklo_epi32(tmp, tmp), _mm_unpackhi_epi32(tmp, tmp)); }

Vc_SIMD_CAST_1(SSE:: float_m, Vc_AVX_NAMESPACE:: float_m) { return AVX::zeroExtend(x.data()); }
Vc_SIMD_CAST_1(SSE:: float_m, Vc_AVX_NAMESPACE::   int_m) { return AVX::zeroExtend(x.data()); }
Vc_SIMD_CAST_1(SSE:: float_m, Vc_AVX_NAMESPACE::  uint_m) { return AVX::zeroExtend(x.data()); }
Vc_SIMD_CAST_1(SSE::   int_m, Vc_AVX_NAMESPACE:: float_m) { return AVX::zeroExtend(x.data()); }
Vc_SIMD_CAST_1(SSE::   int_m, Vc_AVX_NAMESPACE::   int_m) { return AVX::zeroExtend(x.data()); }
Vc_SIMD_CAST_1(SSE::   int_m, Vc_AVX_NAMESPACE::  uint_m) { return AVX::zeroExtend(x.data()); }
Vc_SIMD_CAST_1(SSE::  uint_m, Vc_AVX_NAMESPACE:: float_m) { return AVX::zeroExtend(x.data()); }
Vc_SIMD_CAST_1(SSE::  uint_m, Vc_AVX_NAMESPACE::   int_m) { return AVX::zeroExtend(x.data()); }
Vc_SIMD_CAST_1(SSE::  uint_m, Vc_AVX_NAMESPACE::  uint_m) { return AVX::zeroExtend(x.data()); }

Vc_SIMD_CAST_1(SSE:: float_m, AVX             :: short_m) { return simd_cast<SSE:: short_m>(x).data(); }
Vc_SIMD_CAST_1(SSE::   int_m, AVX             :: short_m) { return simd_cast<SSE:: short_m>(x).data(); }
Vc_SIMD_CAST_1(SSE::  uint_m, AVX             :: short_m) { return simd_cast<SSE:: short_m>(x).data(); }
Vc_SIMD_CAST_1(SSE:: short_m, AVX             :: short_m) { return simd_cast<SSE:: short_m>(x).data(); }
Vc_SIMD_CAST_1(SSE::ushort_m, AVX             :: short_m) { return simd_cast<SSE:: short_m>(x).data(); }
Vc_SIMD_CAST_1(SSE:: float_m, AVX             ::ushort_m) { return simd_cast<SSE::ushort_m>(x).data(); }
Vc_SIMD_CAST_1(SSE::   int_m, AVX             ::ushort_m) { return simd_cast<SSE::ushort_m>(x).data(); }
Vc_SIMD_CAST_1(SSE::  uint_m, AVX             ::ushort_m) { return simd_cast<SSE::ushort_m>(x).data(); }
Vc_SIMD_CAST_1(SSE:: short_m, AVX             ::ushort_m) { return simd_cast<SSE::ushort_m>(x).data(); }
Vc_SIMD_CAST_1(SSE::ushort_m, AVX             ::ushort_m) { return simd_cast<SSE::ushort_m>(x).data(); }

Vc_SIMD_CAST_1(SSE:: short_m, Vc_AVX_NAMESPACE:: float_m) { return AVX::concat(_mm_unpacklo_epi16(x.dataI(), x.dataI()), _mm_unpackhi_epi16(x.dataI(), x.dataI())); }
Vc_SIMD_CAST_1(SSE:: short_m, Vc_AVX_NAMESPACE::   int_m) { return AVX::concat(_mm_unpacklo_epi16(x.dataI(), x.dataI()), _mm_unpackhi_epi16(x.dataI(), x.dataI())); }
Vc_SIMD_CAST_1(SSE:: short_m, Vc_AVX_NAMESPACE::  uint_m) { return AVX::concat(_mm_unpacklo_epi16(x.dataI(), x.dataI()), _mm_unpackhi_epi16(x.dataI(), x.dataI())); }

Vc_SIMD_CAST_1(SSE::ushort_m, Vc_AVX_NAMESPACE:: float_m) { return AVX::concat(_mm_unpacklo_epi16(x.dataI(), x.dataI()), _mm_unpackhi_epi16(x.dataI(), x.dataI())); }
Vc_SIMD_CAST_1(SSE::ushort_m, Vc_AVX_NAMESPACE::   int_m) { return AVX::concat(_mm_unpacklo_epi16(x.dataI(), x.dataI()), _mm_unpackhi_epi16(x.dataI(), x.dataI())); }
Vc_SIMD_CAST_1(SSE::ushort_m, Vc_AVX_NAMESPACE::  uint_m) { return AVX::concat(_mm_unpacklo_epi16(x.dataI(), x.dataI()), _mm_unpackhi_epi16(x.dataI(), x.dataI())); }

// 2 SSE::Mask to 1 AVX(2)::Mask {{{2
Vc_SIMD_CAST_2(SSE::double_m, Vc_AVX_NAMESPACE::double_m) { return AVX::concat(x0.data(), x1.data()); }
Vc_SIMD_CAST_2(SSE::double_m, Vc_AVX_NAMESPACE:: float_m) { return AVX::zeroExtend(_mm_packs_epi32(x0.dataI(), x1.dataI())); }
Vc_SIMD_CAST_2(SSE::double_m, Vc_AVX_NAMESPACE::   int_m) { return AVX::zeroExtend(_mm_packs_epi32(x0.dataI(), x1.dataI())); }
Vc_SIMD_CAST_2(SSE::double_m, Vc_AVX_NAMESPACE::  uint_m) { return AVX::zeroExtend(_mm_packs_epi32(x0.dataI(), x1.dataI())); }
Vc_SIMD_CAST_2(SSE::double_m, Vc_AVX_NAMESPACE:: short_m) { return _mm_packs_epi16(_mm_packs_epi32(x0.dataI(), x1.dataI()), _mm_setzero_si128()); }
Vc_SIMD_CAST_2(SSE::double_m, Vc_AVX_NAMESPACE::ushort_m) { return _mm_packs_epi16(_mm_packs_epi32(x0.dataI(), x1.dataI()), _mm_setzero_si128()); }

Vc_SIMD_CAST_2(SSE:: float_m, Vc_AVX_NAMESPACE:: float_m) { return AVX::concat(x0.data(), x1.data()); }
Vc_SIMD_CAST_2(SSE:: float_m, Vc_AVX_NAMESPACE::   int_m) { return AVX::concat(x0.data(), x1.data()); }
Vc_SIMD_CAST_2(SSE:: float_m, Vc_AVX_NAMESPACE::  uint_m) { return AVX::concat(x0.data(), x1.data()); }
Vc_SIMD_CAST_2(SSE:: float_m, Vc_AVX_NAMESPACE:: short_m) { return _mm_packs_epi16(x0.dataI(), x1.dataI()); }
Vc_SIMD_CAST_2(SSE:: float_m, Vc_AVX_NAMESPACE::ushort_m) { return _mm_packs_epi16(x0.dataI(), x1.dataI()); }

Vc_SIMD_CAST_2(SSE::   int_m, Vc_AVX_NAMESPACE:: float_m) { return AVX::concat(x0.data(), x1.data()); }
Vc_SIMD_CAST_2(SSE::   int_m, Vc_AVX_NAMESPACE::   int_m) { return AVX::concat(x0.data(), x1.data()); }
Vc_SIMD_CAST_2(SSE::   int_m, Vc_AVX_NAMESPACE::  uint_m) { return AVX::concat(x0.data(), x1.data()); }
Vc_SIMD_CAST_2(SSE::   int_m, Vc_AVX_NAMESPACE:: short_m) { return _mm_packs_epi16(x0.dataI(), x1.dataI()); }
Vc_SIMD_CAST_2(SSE::   int_m, Vc_AVX_NAMESPACE::ushort_m) { return _mm_packs_epi16(x0.dataI(), x1.dataI()); }

Vc_SIMD_CAST_2(SSE::  uint_m, Vc_AVX_NAMESPACE:: float_m) { return AVX::concat(x0.data(), x1.data()); }
Vc_SIMD_CAST_2(SSE::  uint_m, Vc_AVX_NAMESPACE::   int_m) { return AVX::concat(x0.data(), x1.data()); }
Vc_SIMD_CAST_2(SSE::  uint_m, Vc_AVX_NAMESPACE::  uint_m) { return AVX::concat(x0.data(), x1.data()); }
Vc_SIMD_CAST_2(SSE::  uint_m, Vc_AVX_NAMESPACE:: short_m) { return _mm_packs_epi16(x0.dataI(), x1.dataI()); }
Vc_SIMD_CAST_2(SSE::  uint_m, Vc_AVX_NAMESPACE::ushort_m) { return _mm_packs_epi16(x0.dataI(), x1.dataI()); }

// 4 SSE::Mask to 1 AVX(2)::Mask {{{2
Vc_SIMD_CAST_4(SSE::double_m, Vc_AVX_NAMESPACE:: float_m) { return AVX::concat(_mm_packs_epi32(x0.dataI(), x1.dataI()), _mm_packs_epi32(x2.dataI(), x3.dataI())); }
Vc_SIMD_CAST_4(SSE::double_m, Vc_AVX_NAMESPACE::   int_m) { return AVX::concat(_mm_packs_epi32(x0.dataI(), x1.dataI()), _mm_packs_epi32(x2.dataI(), x3.dataI())); }
Vc_SIMD_CAST_4(SSE::double_m, Vc_AVX_NAMESPACE::  uint_m) { return AVX::concat(_mm_packs_epi32(x0.dataI(), x1.dataI()), _mm_packs_epi32(x2.dataI(), x3.dataI())); }
Vc_SIMD_CAST_4(SSE::double_m, Vc_AVX_NAMESPACE:: short_m) { return _mm_packs_epi16(_mm_packs_epi32(x0.dataI(), x1.dataI()), _mm_packs_epi32(x2.dataI(), x3.dataI())); }
Vc_SIMD_CAST_4(SSE::double_m, Vc_AVX_NAMESPACE::ushort_m) { return _mm_packs_epi16(_mm_packs_epi32(x0.dataI(), x1.dataI()), _mm_packs_epi32(x2.dataI(), x3.dataI())); }

// 1 Scalar::Mask to 1 AVX(2)::Mask {{{2
template <typename Return, typename T>
Vc_INTRINSIC Vc_CONST Return
    simd_cast(Scalar::Mask<T> k,
              enable_if<Vc_AVX_NAMESPACE::is_mask<Return>::value> = nullarg)
{
    Return r{false};
    r[0] = k.data();
    return r;
}

// 2 Scalar::Mask to 1 AVX(2)::Mask {{{2
template <typename Return, typename T>
Vc_INTRINSIC Vc_CONST Return
    simd_cast(Scalar::Mask<T> k0,
              Scalar::Mask<T> k1,
              enable_if<Vc_AVX_NAMESPACE::is_mask<Return>::value> = nullarg)
{
    Return r{false};
    r[0] = k0.data();
    r[1] = k1.data();
    return r;
}

// 4 Scalar::Mask to 1 AVX(2)::Mask {{{2
template <typename Return, typename T>
Vc_INTRINSIC Vc_CONST Return simd_cast(
    Scalar::Mask<T> k0,
    Scalar::Mask<T> k1,
    Scalar::Mask<T> k2,
    Scalar::Mask<T> k3,
    enable_if<(Vc_AVX_NAMESPACE::is_mask<Return>::value && Return::Size >= 4)> = nullarg)
{
    Return r{false};
    r[0] = k0.data();
    r[1] = k1.data();
    r[2] = k2.data();
    r[3] = k3.data();
    return r;
}

// 8 Scalar::Mask to 1 AVX(2)::Mask {{{2
template <typename Return, typename T>
Vc_INTRINSIC Vc_CONST Return simd_cast(
    Scalar::Mask<T> k0,
    Scalar::Mask<T> k1,
    Scalar::Mask<T> k2,
    Scalar::Mask<T> k3,
    Scalar::Mask<T> k4,
    Scalar::Mask<T> k5,
    Scalar::Mask<T> k6,
    Scalar::Mask<T> k7,
    enable_if<(Vc_AVX_NAMESPACE::is_mask<Return>::value && Return::Size >= 8)> = nullarg)
{
    Return r{false};
    r[0] = k0.data();
    r[1] = k1.data();
    r[2] = k2.data();
    r[3] = k3.data();
    r[4] = k4.data();
    r[5] = k5.data();
    r[6] = k6.data();
    r[7] = k7.data();
    return r;
}

// 1 AVX::Mask to 1 SSE::Mask {{{2
Vc_SIMD_CAST_1(Vc_AVX_NAMESPACE::double_m, SSE::double_m) { return AVX::lo128(x.data()); }
Vc_SIMD_CAST_1(Vc_AVX_NAMESPACE::double_m, SSE:: float_m) { return _mm_packs_epi32(AVX::lo128(x.dataI()), AVX::hi128(x.dataI())); }
Vc_SIMD_CAST_1(Vc_AVX_NAMESPACE::double_m, SSE::   int_m) { return _mm_packs_epi32(AVX::lo128(x.dataI()), AVX::hi128(x.dataI())); }
Vc_SIMD_CAST_1(Vc_AVX_NAMESPACE::double_m, SSE::  uint_m) { return _mm_packs_epi32(AVX::lo128(x.dataI()), AVX::hi128(x.dataI())); }
Vc_SIMD_CAST_1(Vc_AVX_NAMESPACE::double_m, SSE:: short_m) { return _mm_packs_epi16(_mm_packs_epi32(AVX::lo128(x.dataI()), AVX::hi128(x.dataI())), _mm_setzero_si128()); }
Vc_SIMD_CAST_1(Vc_AVX_NAMESPACE::double_m, SSE::ushort_m) { return _mm_packs_epi16(_mm_packs_epi32(AVX::lo128(x.dataI()), AVX::hi128(x.dataI())), _mm_setzero_si128()); }

Vc_SIMD_CAST_1(Vc_AVX_NAMESPACE:: float_m, SSE::double_m) { return _mm_unpacklo_ps(AVX::lo128(x.data()), AVX::lo128(x.data())); }
Vc_SIMD_CAST_1(Vc_AVX_NAMESPACE:: float_m, SSE:: float_m) { return AVX::lo128(x.data()); }
Vc_SIMD_CAST_1(Vc_AVX_NAMESPACE:: float_m, SSE::   int_m) { return AVX::lo128(x.data()); }
Vc_SIMD_CAST_1(Vc_AVX_NAMESPACE:: float_m, SSE::  uint_m) { return AVX::lo128(x.data()); }
Vc_SIMD_CAST_1(Vc_AVX_NAMESPACE:: float_m, SSE:: short_m) { return _mm_packs_epi16(AVX::lo128(x.dataI()), AVX::hi128(x.dataI())); }
Vc_SIMD_CAST_1(Vc_AVX_NAMESPACE:: float_m, SSE::ushort_m) { return _mm_packs_epi16(AVX::lo128(x.dataI()), AVX::hi128(x.dataI())); }

Vc_SIMD_CAST_1(Vc_AVX_NAMESPACE::   int_m, SSE::double_m) { return _mm_unpacklo_epi32(AVX::lo128(x.dataI()), AVX::lo128(x.dataI())); }
Vc_SIMD_CAST_1(Vc_AVX_NAMESPACE::   int_m, SSE:: float_m) { return AVX::lo128(x.dataI()); }
Vc_SIMD_CAST_1(Vc_AVX_NAMESPACE::   int_m, SSE::   int_m) { return AVX::lo128(x.dataI()); }
Vc_SIMD_CAST_1(Vc_AVX_NAMESPACE::   int_m, SSE::  uint_m) { return AVX::lo128(x.dataI()); }
Vc_SIMD_CAST_1(Vc_AVX_NAMESPACE::   int_m, SSE:: short_m) { return _mm_packs_epi16(AVX::lo128(x.dataI()), AVX::hi128(x.dataI())); }
Vc_SIMD_CAST_1(Vc_AVX_NAMESPACE::   int_m, SSE::ushort_m) { return _mm_packs_epi16(AVX::lo128(x.dataI()), AVX::hi128(x.dataI())); }

Vc_SIMD_CAST_1(Vc_AVX_NAMESPACE::  uint_m, SSE::double_m) { return _mm_unpacklo_epi32(AVX::lo128(x.dataI()), AVX::lo128(x.dataI())); }
Vc_SIMD_CAST_1(Vc_AVX_NAMESPACE::  uint_m, SSE:: float_m) { return AVX::lo128(x.dataI()); }
Vc_SIMD_CAST_1(Vc_AVX_NAMESPACE::  uint_m, SSE::   int_m) { return AVX::lo128(x.dataI()); }
Vc_SIMD_CAST_1(Vc_AVX_NAMESPACE::  uint_m, SSE::  uint_m) { return AVX::lo128(x.dataI()); }
Vc_SIMD_CAST_1(Vc_AVX_NAMESPACE::  uint_m, SSE:: short_m) { return _mm_packs_epi16(AVX::lo128(x.dataI()), AVX::hi128(x.dataI())); }
Vc_SIMD_CAST_1(Vc_AVX_NAMESPACE::  uint_m, SSE::ushort_m) { return _mm_packs_epi16(AVX::lo128(x.dataI()), AVX::hi128(x.dataI())); }

#ifdef VC_IMPL_AVX2
#else
Vc_SIMD_CAST_1(AVX:: short_m, SSE::double_m) { auto tmp = _mm_unpacklo_epi16(x.dataI(), x.dataI()); return _mm_unpacklo_epi32(tmp, tmp); }
Vc_SIMD_CAST_1(AVX:: short_m, SSE:: float_m) { return _mm_unpacklo_epi16(x.dataI(), x.dataI()); }
Vc_SIMD_CAST_1(AVX:: short_m, SSE::   int_m) { return _mm_unpacklo_epi16(x.dataI(), x.dataI()); }
Vc_SIMD_CAST_1(AVX:: short_m, SSE::  uint_m) { return _mm_unpacklo_epi16(x.dataI(), x.dataI()); }
Vc_SIMD_CAST_1(AVX:: short_m, SSE:: short_m) { return x.dataI(); }
Vc_SIMD_CAST_1(AVX:: short_m, SSE::ushort_m) { return x.dataI(); }

Vc_SIMD_CAST_1(AVX::ushort_m, SSE::double_m) { auto tmp = _mm_unpacklo_epi16(x.dataI(), x.dataI()); return _mm_unpacklo_epi32(tmp, tmp); }
Vc_SIMD_CAST_1(AVX::ushort_m, SSE:: float_m) { return _mm_unpacklo_epi16(x.dataI(), x.dataI()); }
Vc_SIMD_CAST_1(AVX::ushort_m, SSE::   int_m) { return _mm_unpacklo_epi16(x.dataI(), x.dataI()); }
Vc_SIMD_CAST_1(AVX::ushort_m, SSE::  uint_m) { return _mm_unpacklo_epi16(x.dataI(), x.dataI()); }
Vc_SIMD_CAST_1(AVX::ushort_m, SSE:: short_m) { return x.dataI(); }
Vc_SIMD_CAST_1(AVX::ushort_m, SSE::ushort_m) { return x.dataI(); }
#endif

// 2 AVX::Mask to 1 SSE::Mask {{{2
Vc_SIMD_CAST_2(Vc_AVX_NAMESPACE::double_m, SSE:: short_m) { return _mm_packs_epi16(_mm_packs_epi32(AVX::lo128(x0.dataI()), AVX::hi128(x0.dataI())), _mm_packs_epi32(AVX::lo128(x1.dataI()), AVX::hi128(x1.dataI()))); }
Vc_SIMD_CAST_2(Vc_AVX_NAMESPACE::double_m, SSE::ushort_m) { return _mm_packs_epi16(_mm_packs_epi32(AVX::lo128(x0.dataI()), AVX::hi128(x0.dataI())), _mm_packs_epi32(AVX::lo128(x1.dataI()), AVX::hi128(x1.dataI()))); }

// 1 AVX::Mask to 1 Scalar::Mask {{{2
#define Vc_SIMD_CAST_AVX_TO_SCALAR(to__)                                                 \
    template <typename To, typename FromT>                                               \
    Vc_INTRINSIC Vc_CONST To                                                             \
        simd_cast(Vc_AVX_NAMESPACE::Mask<FromT> x,                                       \
                  enable_if<std::is_same<To, Scalar::to__>::value> = nullarg)            \
    {                                                                                    \
        return static_cast<To>(x[0]);                                                    \
    }

VC_ALL_VECTOR_TYPES(Vc_SIMD_CAST_AVX_TO_SCALAR)
#undef Vc_SIMD_CAST_AVX_TO_SCALAR

// simd_mask_array to AVX::Mask {{{2
Vc_4_SIMDARRAY_TO_1__(simd_mask_array, Vc_AVX_NAMESPACE::is_mask)
Vc_3_SIMDARRAY_TO_1__(simd_mask_array, Vc_AVX_NAMESPACE::is_mask)
Vc_2_SIMDARRAY_TO_1__(simd_mask_array, Vc_AVX_NAMESPACE::is_mask)
Vc_1_SIMDARRAY_TO_1__(simd_mask_array, Vc_AVX_NAMESPACE::is_mask)

// offset == 0 | convert from AVX(2)::Mask/Vector {{{1
template <typename Return, int offset, typename V>
Vc_INTRINSIC Vc_CONST Return
    simd_cast(V &&x,
              enable_if<offset == 0 &&
                        (Vc_AVX_NAMESPACE::is_vector<Traits::decay<V>>::value ||
                         Vc_AVX_NAMESPACE::is_mask<Traits::decay<V>>::value)> = nullarg)
{
    return simd_cast<Return>(x);
}

// offset == 0 | convert from SSE::Mask/Vector to AVX(2)::Mask/Vector {{{1
template <typename Return, int offset, typename V>
Vc_INTRINSIC Vc_CONST Return simd_cast(
    V &&x,
    enable_if<offset == 0 && ((SSE::is_vector<Traits::decay<V>>::value &&
                               Vc_AVX_NAMESPACE::is_vector<Return>::value) ||
                              (SSE::is_mask<Traits::decay<V>>::value &&
                               Vc_AVX_NAMESPACE::is_mask<Return>::value))> = nullarg)
{
    return simd_cast<Return>(x);
}

// Vector casts with offset {{{1
template <typename Return, int offset, typename T>
Vc_INTRINSIC Vc_CONST Return
    simd_cast(Vc_AVX_NAMESPACE::Vector<T> x,
              enable_if<(offset != 0 && sizeof(Return) <= 32 && sizeof(T) > 2)> = nullarg)
{
    using V = Vc_AVX_NAMESPACE::Vector<T>;
    constexpr int shift = sizeof(T) * offset * Return::Size;
    static_assert(shift > 0 && shift < sizeof(x), "");
    if (shift < 16) {
        return simd_cast<Return>(V{AVX::avx_cast<typename V::VectorType>(
            _mm_srli_si128(AVX::avx_cast<__m128i>(AVX::lo128(x.data())), shift))});
    } else if (shift == 16) {
        return simd_cast<Return>(V{Mem::permute128<X1, Const0>(x.data())});
    } else {
        return simd_cast<Return>(V{AVX::avx_cast<typename V::VectorType>(
            _mm_srli_si128(AVX::avx_cast<__m128i>(AVX::hi128(x.data())), shift - 16))});
    }
}

template <typename Return, int offset, typename T>
Vc_INTRINSIC Vc_CONST Return
    simd_cast(Vc_AVX_NAMESPACE::Vector<T> x,
              enable_if<(offset != 0 && sizeof(Return) <= 32 && sizeof(T) <= 2)> = nullarg)
{
    using V = Vc_AVX_NAMESPACE::Vector<T>;
    constexpr int shift = sizeof(T) * offset * Return::Size;
    static_assert(shift > 0 && shift < sizeof(x), "");
    return simd_cast<Return>(V{AVX::avx_cast<typename V::VectorType>(
        _mm_srli_si128(AVX::avx_cast<__m128i>(x.data()), shift))});
}

// AVX to SSE (Vector<T>)
template <typename Return, int offset, typename V>
Vc_INTRINSIC Vc_CONST Return
    simd_cast(V x,
              enable_if<offset != 0 && (AVX::is_vector<V>::value || AVX2::is_vector<V>::value) &&
                        SSE::is_vector<Return>::value> = nullarg)
{
    constexpr int shift = sizeof(V) / V::Size * offset * Return::Size;
    static_assert(shift > 0, "");
    static_assert(shift < sizeof(V), "");
    using SseVector = SSE::Vector<typename V::EntryType>;
    using Intrin = typename SseVector::VectorType;
    return simd_cast<Return>(SseVector{
        AVX::avx_cast<Intrin>(_mm_alignr_epi8(AVX::lo128(x.data()), AVX::hi128(x.data()), shift))});
}

// Mask casts with offset {{{1
// 1 AVX::Mask to N AVX::Mask {{{2
// It's rather limited: all AVX::Mask types except double_v have Size=8; and double_v::Size=4
// Therefore Return is always double_m and k is either float_m == int_m == uint_m or
// short_m == ushort_m
template <typename Return, int offset, typename T>
Vc_INTRINSIC Vc_CONST Return simd_cast(
    const Vc_AVX_NAMESPACE::Mask<T> &k,
    enable_if<sizeof(k) == 32 && offset == 1 && AVX::is_mask<Return>::value> = nullarg)
{
    const auto tmp = AVX::hi128(k.dataI());
    return AVX::concat(_mm_unpacklo_epi32(tmp, tmp), _mm_unpackhi_epi32(tmp, tmp));
}
template <typename Return, int offset, typename T>
Vc_INTRINSIC Vc_CONST Return simd_cast(
    const Vc_AVX_NAMESPACE::Mask<T> &k,
    enable_if<sizeof(k) == 16 && offset == 1 && AVX::is_mask<Return>::value> = nullarg)
{
    const auto tmp = _mm_unpackhi_epi16(k.dataI(), k.dataI());
    return AVX::concat(_mm_unpacklo_epi32(tmp, tmp), _mm_unpackhi_epi32(tmp, tmp));
}

// 1 SSE::Mask to N AVX(2)::Mask {{{2
#define Vc_SIMD_CAST_OFFSET(from__, to__, offset__)                                      \
    template <typename To, int offset>                                                   \
    Vc_INTRINSIC Vc_CONST To simd_cast(                                                  \
        from__ x,                                                                        \
        enable_if<(offset == offset__ && std::is_same<To, to__>::value)> = nullarg)

Vc_SIMD_CAST_OFFSET(SSE:: short_m, Vc_AVX_NAMESPACE::double_m, 1) { auto tmp = _mm_unpackhi_epi16(x.dataI(), x.dataI()); return AVX::concat(_mm_unpacklo_epi32(tmp, tmp), _mm_unpackhi_epi32(tmp, tmp)); }
Vc_SIMD_CAST_OFFSET(SSE::ushort_m, Vc_AVX_NAMESPACE::double_m, 1) { auto tmp = _mm_unpackhi_epi16(x.dataI(), x.dataI()); return AVX::concat(_mm_unpacklo_epi32(tmp, tmp), _mm_unpackhi_epi32(tmp, tmp)); }

#undef Vc_SIMD_CAST_OFFSET

// AVX to SSE (Mask<T>) {{{2
template <typename Return, int offset, typename M>
Vc_INTRINSIC Vc_CONST Return simd_cast(
    M x,
    enable_if<offset != 0 && (AVX::is_mask<M>::value || AVX2::is_mask<M>::value) &&
              SSE::is_mask<Return>::value> = nullarg)
{
    constexpr int shift = sizeof(M) / M::Size * offset * Return::Size;
    static_assert(shift > 0, "");
    static_assert(shift < sizeof(M), "");
    using SseVector = SSE::Mask<Traits::entry_type_of<typename M::Vector>>;
    using Intrin = typename SseVector::VectorType;
    return simd_cast<Return>(SseVector{AVX::avx_cast<Intrin>(
        _mm_alignr_epi8(AVX::lo128(x.dataI()), AVX::hi128(x.dataI()), shift))});
}

// part of a simd_mask_array to AVX::Mask {{{2
template <typename Return, int offset, typename T, std::size_t N, typename V>
Vc_INTRINSIC Vc_CONST Return simd_cast(const simd_mask_array<T, N, V, N> &x,
                                       enable_if<AVX::is_mask<Return>::value> = nullarg)
{
    return simd_cast<Return, offset>(internal_data(x));
}
template <typename Return,
          int offset,
          typename T,
          std::size_t N,
          typename V,
          std::size_t M>
Vc_INTRINSIC Vc_CONST Return
    simd_cast(const simd_mask_array<T, N, V, M> &x,
              enable_if<(N > M) && (Return::Size * offset <
                                    Traits::decay<decltype(internal_data0(x))>::Size) &&
                        (Return::Size * (offset + 1) >
                         Traits::decay<decltype(internal_data0(x))>::Size) &&
                        AVX::is_mask<Return>::value> = nullarg)
{
    Return r;
    for (std::size_t i = 0; i < Return::Size; ++i) {
        r[i] = x[i + Return::Size * offset];
    }
    return r;
}
template <typename Return,
          int offset,
          typename T,
          std::size_t N,
          typename V,
          std::size_t M>
Vc_INTRINSIC Vc_CONST Return
    simd_cast(const simd_mask_array<T, N, V, M> &x,
              enable_if<(N > M) && (Return::Size * offset <
                                    Traits::decay<decltype(internal_data0(x))>::Size) &&
                        (Return::Size * (offset + 1) <=
                         Traits::decay<decltype(internal_data0(x))>::Size) &&
                        AVX::is_mask<Return>::value> = nullarg)
{
    return simd_cast<Return, offset>(internal_data0(x));
}
template <typename Return,
          int offset,
          typename T,
          std::size_t N,
          typename V,
          std::size_t M>
Vc_INTRINSIC Vc_CONST Return
    simd_cast(const simd_mask_array<T, N, V, M> &x,
              enable_if<(N > M) && (Return::Size * offset >=
                                    Traits::decay<decltype(internal_data0(x))>::Size) &&
                        AVX::is_mask<Return>::value> = nullarg)
{
    return simd_cast<
        Return,
        offset - Traits::decay<decltype(internal_data0(x))>::Size / Return::Size>(
        internal_data1(x));
}

// undef Vc_SIMD_CAST_AVX_[124] & Vc_SIMD_CAST_[124] {{{1
#undef Vc_SIMD_CAST_AVX_1
#undef Vc_SIMD_CAST_AVX_2
#undef Vc_SIMD_CAST_AVX_4

#undef Vc_SIMD_CAST_1
#undef Vc_SIMD_CAST_2
#undef Vc_SIMD_CAST_4
// }}}1

}  // namespace Vc

#include "undomacros.h"

#endif // VC_AVX_SIMD_CAST_H
