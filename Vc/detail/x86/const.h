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

#ifndef VC_DETAIL_X86_CONST_H_
#define VC_DETAIL_X86_CONST_H_

#include "../const.h"
#include "types.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
#ifdef Vc_HAVE_SSE_ABI
template <class X> struct constants<simd_abi::__sse, X> {
    alignas(64) static inline constexpr builtin_type_t<  uint,  4>  absMaskFloat{0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff};
    alignas(16) static inline constexpr builtin_type_t< float,  4> signMaskFloat{-0.f, -0.f, -0.f, -0.f};
    alignas(16) static inline constexpr builtin_type_t<  uint,  4> highMaskFloat{0xfffff000u, 0xfffff000u, 0xfffff000u, 0xfffff000u};
    alignas(16) static inline constexpr builtin_type_t< float,  4>      oneFloat{1.f, 1.f, 1.f, 1.f};

    alignas(16) static inline constexpr builtin_type_t<short ,  8> minShort{-0x8000, -0x8000, -0x8000, -0x8000, -0x8000, -0x8000, -0x8000, -0x8000};
    alignas(16) static inline constexpr builtin_type_t<uchar , 16> one8 {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    alignas(16) static inline constexpr builtin_type_t<ushort,  8> one16{1, 1, 1, 1, 1, 1, 1, 1};
    alignas(16) static inline constexpr builtin_type_t<uint  ,  4> one32{1, 1, 1, 1};

    alignas(16) static inline constexpr builtin_type_t<ullong,  2> one64{1, 1};
    alignas(16) static inline constexpr builtin_type_t<double,  2> oneDouble{1., 1.};
    alignas(16) static inline constexpr builtin_type_t<ullong,  2> highMaskDouble{0xfffffffff8000000ull, 0xfffffffff8000000ull};
    alignas(16) static inline constexpr builtin_type_t<llong ,  2> absMaskDouble{0x7fffffffffffffffll, 0x7fffffffffffffffll};

    alignas(16) static inline constexpr builtin_type_t<double,  2> signMaskDouble{-0., -0.};
    alignas(16) static inline constexpr builtin_type_t<ullong,  2> frexpMask{0xbfefffffffffffffull, 0xbfefffffffffffffull};
    alignas(16) static inline constexpr builtin_type_t<uint  ,  4> IndexesFromZero4{ 0, 1, 2, 3 };
    alignas(16) static inline constexpr builtin_type_t<ushort,  8> IndexesFromZero8{ 0, 1, 2, 3, 4, 5, 6, 7 };

    alignas(16) static inline constexpr builtin_type_t<uchar , 16> IndexesFromZero16{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    alignas(16) static inline constexpr builtin_type_t<uchar , 16> cvti16_i08_shuffle{
        0, 2, 4, 6, 8, 10, 12, 14, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80
    };

#define float_constant(name_, value_) static inline constexpr float name_ = value_
    alignas(64)
    float_constant(f32_pi_4     , (constants<simd_abi::scalar, X>::f32_pi_4));
    float_constant(f32_pi_4_hi  , (constants<simd_abi::scalar, X>::f32_pi_4_hi));
    float_constant(f32_pi_4_rem1, (constants<simd_abi::scalar, X>::f32_pi_4_rem1));
    float_constant(f32_pi_4_rem2, (constants<simd_abi::scalar, X>::f32_pi_4_rem2));

    float_constant(f32_1_16             , 0.0625f);
    float_constant(f32_16               , 16.f);
    float_constant(f32_cos_c0           , 4.166664568298827e-2f);  // ~ 1/4!
    float_constant(f32_cos_c1           , -1.388731625493765e-3f); // ~-1/6!
    float_constant(f32_cos_c2           , 2.443315711809948e-5f);  // ~ 1/8!
    float_constant(f32_sin_c0           , -1.6666654611e-1f); // ~-1/3!
    float_constant(f32_sin_c1           , 8.3321608736e-3f);  // ~ 1/5!
    float_constant(f32_sin_c2           , -1.9515295891e-4f); // ~-1/7!
    float_constant(f32_loss_threshold   , 8192.f); // loss threshold
    float_constant(f32_4_pi             , (float_const< 1, 0x22F983, 0>)); // 1.27323949337005615234375 = 4/π
    float_constant(f32_pi_2             , (float_const< 1, 0x490FDB, 0>)); // π/2
    float_constant(f32_pi               , (float_const< 1, 0x490FDB, 1>)); // π
    float_constant(f32_atan_p0          , 8.05374449538e-2f); // atan P coefficients
    float_constant(f32_atan_p1          , 1.38776856032e-1f); // atan P coefficients
    float_constant(f32_atan_p2          , 1.99777106478e-1f); // atan P coefficients
    float_constant(f32_atan_p3          , 3.33329491539e-1f); // atan P coefficients
    float_constant(f32_atan_threshold_hi, 2.414213562373095f); // tan( 3/8 π )
    float_constant(f32_atan_threshold_lo, 0.414213562373095f); // tan( 1/8 π ) lower threshold for special casing in atan
    float_constant(f32_pi_2_rem         , (float_const<-1, 0x3BBD2E, -25>)); // remainder of pi/2
    float_constant(f32_small_asin_input , 1.e-4f); // small asin input threshold
    float_constant(f32_large_asin_input , 0.f); // padding (for alignment with double)
    float_constant(f32_asin_c0_0        , 4.2163199048e-2f); // asinCoeff0
    float_constant(f32_asin_c0_1        , 2.4181311049e-2f); // asinCoeff0
    float_constant(f32_asin_c0_2        , 4.5470025998e-2f); // asinCoeff0
    float_constant(f32_asin_c0_3        , 7.4953002686e-2f); // asinCoeff0
    float_constant(f32_asin_c0_4        , 1.6666752422e-1f); // asinCoeff0

#define double_constant(name_, value_)                                                   \
    static inline constexpr double name_ = value_
    double_constant(f64_pi_4     , (double_const< 1, 0x921fb54442d18, -1>)); // π/4
    double_constant(f64_pi_4_hi  , (double_const< 1, 0x921fb40000000, -1>)); // π/4 - 30bits precision
    double_constant(f64_pi_4_rem1, (double_const< 1, 0x4442d00000000, -25>)); // π/4 remainder1 - 32bits precision
    double_constant(f64_pi_4_rem2, (double_const< 1, 0x8469898cc5170, -49>)); // π/4 remainder2
    double_constant(f64_1_16     , 0.0625);
    double_constant(f64_16       , 16.);
    double_constant(f64_cos_c0   , (double_const< 1, 0x555555555554b, -5 >)); // ~ 1/4!
    double_constant(f64_cos_c1   , (double_const<-1, 0x6c16c16c14f91, -10>)); // ~-1/6!
    double_constant(f64_cos_c2   , (double_const< 1, 0xa01a019c844f5, -16>)); // ~ 1/8!
    double_constant(f64_cos_c3   , (double_const<-1, 0x27e4f7eac4bc6, -22>)); // ~-1/10!
    double_constant(f64_cos_c4   , (double_const< 1, 0x1ee9d7b4e3f05, -29>)); // ~ 1/12!
    double_constant(f64_cos_c5   , (double_const<-1, 0x8fa49a0861a9b, -37>)); // ~-1/14!
    double_constant(f64_sin_c0   , (double_const<-1, 0x5555555555548, -3 >)); // ~-1/3!
    double_constant(f64_sin_c1   , (double_const< 1, 0x111111110f7d0, -7 >)); // ~ 1/5!
    double_constant(f64_sin_c2   , (double_const<-1, 0xa01a019bfdf03, -13>)); // ~-1/7!
    double_constant(f64_sin_c3   , (double_const< 1, 0x71de3567d48a1, -19>)); // ~ 1/9!
    double_constant(f64_sin_c4   , (double_const<-1, 0xae5e5a9291f5d, -26>)); // ~-1/11!
    double_constant(f64_sin_c5   , (double_const< 1, 0x5d8fd1fd19ccd, -33>)); // ~ 1/13!
    double_constant(f64_4_pi     , (double_const< 1, 0x8BE60DB939105, 0 >)); // 4/π
    double_constant(f64_pi_2     , (double_const< 1, 0x921fb54442d18, 0 >)); // π/2
    double_constant(f64_pi       , (double_const< 1, 0x921fb54442d18, 1 >)); // π
    double_constant(f64_atan_p0  , (double_const<-1, 0xc007fa1f72594, -1>)); // atan P coefficients
    double_constant(f64_atan_p1  , (double_const<-1, 0x028545b6b807a, 4 >)); // atan P coefficients
    double_constant(f64_atan_p2  , (double_const<-1, 0x2c08c36880273, 6 >)); // atan P coefficients
    double_constant(f64_atan_p3  , (double_const<-1, 0xeb8bf2d05ba25, 6 >)); // atan P coefficients
    double_constant(f64_atan_p4  , (double_const<-1, 0x03669fd28ec8e, 6 >)); // atan P coefficients
    double_constant(f64_atan_q0  , (double_const< 1, 0x8dbc45b14603c, 4 >)); // atan Q coefficients
    double_constant(f64_atan_q1  , (double_const< 1, 0x4a0dd43b8fa25, 7 >)); // atan Q coefficients
    double_constant(f64_atan_q2  , (double_const< 1, 0xb0e18d2e2be3b, 8 >)); // atan Q coefficients
    double_constant(f64_atan_q3  , (double_const< 1, 0xe563f13b049ea, 8 >)); // atan Q coefficients
    double_constant(f64_atan_q4  , (double_const< 1, 0x8519efbbd62ec, 7 >)); // atan Q coefficients
    double_constant(f64_atan_threshold_hi, (double_const< 1, 0x3504f333f9de6, 1>)); // tan( 3/8 π )
    double_constant(f64_atan_threshold_lo, 0.66);                                 // lower threshold for special casing in atan
    double_constant(f64_pi_2_rem         , (double_const< 1, 0x1A62633145C07, -54>)); // remainder of pi/2
    double_constant(f64_small_asin_input , 1.e-8); // small asin input threshold
    double_constant(f64_large_asin_input , 0.625); // large asin input threshold
    double_constant(f64_asin_c0_0        , (double_const< 1, 0x84fc3988e9f08, -9>)); // asinCoeff0
    double_constant(f64_asin_c0_1        , (double_const<-1, 0x2079259f9290f, -1>)); // asinCoeff0
    double_constant(f64_asin_c0_2        , (double_const< 1, 0xbdff5baf33e6a, 2 >)); // asinCoeff0
    double_constant(f64_asin_c0_3        , (double_const<-1, 0x991aaac01ab68, 4 >)); // asinCoeff0
    double_constant(f64_asin_c0_4        , (double_const< 1, 0xc896240f3081d, 4 >)); // asinCoeff0
    double_constant(f64_asin_c1_0        , (double_const<-1, 0x5f2a2b6bf5d8c, 4 >)); // asinCoeff1
    double_constant(f64_asin_c1_1        , (double_const< 1, 0x26219af6a7f42, 7 >)); // asinCoeff1
    double_constant(f64_asin_c1_2        , (double_const<-1, 0x7fe08959063ee, 8 >)); // asinCoeff1
    double_constant(f64_asin_c1_3        , (double_const< 1, 0x56709b0b644be, 8 >)); // asinCoeff1
    double_constant(f64_asin_c2_0        , (double_const< 1, 0x16b9b0bd48ad3, -8>)); // asinCoeff2
    double_constant(f64_asin_c2_1        , (double_const<-1, 0x34341333e5c16, -1>)); // asinCoeff2
    double_constant(f64_asin_c2_2        , (double_const< 1, 0x5c74b178a2dd9, 2 >)); // asinCoeff2
    double_constant(f64_asin_c2_3        , (double_const<-1, 0x04331de27907b, 4 >)); // asinCoeff2
    double_constant(f64_asin_c2_4        , (double_const< 1, 0x39007da779259, 4 >)); // asinCoeff2
    double_constant(f64_asin_c2_5        , (double_const<-1, 0x0656c06ceafd5, 3 >)); // asinCoeff2
    double_constant(f64_asin_c3_0        , (double_const<-1, 0xd7b590b5e0eab, 3 >)); // asinCoeff3
    double_constant(f64_asin_c3_1        , (double_const< 1, 0x19fc025fe9054, 6 >)); // asinCoeff3
    double_constant(f64_asin_c3_2        , (double_const<-1, 0x265bb6d3576d7, 7 >)); // asinCoeff3
    double_constant(f64_asin_c3_3        , (double_const< 1, 0x1705684ffbf9d, 7 >)); // asinCoeff3
    double_constant(f64_asin_c3_4        , (double_const<-1, 0x898220a3607ac, 5 >)); // asinCoeff3
};

namespace x86
{
using sse_const = constants<simd_abi::__sse>;
}  // namespace x86
#endif  // Vc_HAVE_SSE_ABI

#ifdef Vc_HAVE_AVX_ABI
template <class X>
struct constants<simd_abi::__avx, X> : public constants<simd_abi::scalar, X> {
    alignas(64) static inline constexpr builtin_type_t<ullong,  4> IndexesFromZero64 = { 0, 1, 2, 3 };
    alignas(32) static inline constexpr builtin_type_t<  uint,  8> IndexesFromZero32 = { 0, 1, 2, 3, 4, 5, 6, 7 };
    alignas(32) static inline constexpr builtin_type_t<ushort, 16> IndexesFromZero16 = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    alignas(32) static inline constexpr builtin_type_t< uchar, 32> IndexesFromZero8  = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 };

    alignas(32) static inline constexpr builtin_type_t< uint ,  8> AllBitsSet = {
        0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU
    };

    static constexpr   uint absMaskFloat[2] = { 0xffffffffu, 0x7fffffffu };
    static constexpr   uint highMaskFloat = 0xfffff000u;
    static constexpr  float oneFloat = 1.f;
    alignas(float) static constexpr ushort one16[2] = { 1, 1 };
    alignas(float) static constexpr  uchar one8[4] = { 1, 1, 1, 1 };
    static constexpr  float _2_pow_31 = 1u << 31;
    static constexpr ullong highMaskDouble = 0xfffffffff8000000ull;
    static constexpr double oneDouble = 1.;
};

namespace x86
{
using avx_const = constants<simd_abi::__avx>;
}  // namespace x86
#endif  // Vc_HAVE_AVX_ABI

}  // namespace detail
Vc_VERSIONED_NAMESPACE_END
#endif  // VC_DETAIL_X86_CONST_H_
