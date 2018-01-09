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

#if defined Vc_MSVC && Vc_MSVC < 191024903
#define Vc_WORK_AROUND_ICE
#endif

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
#ifdef Vc_HAVE_SSE_ABI
#ifdef Vc_WORK_AROUND_ICE
namespace x86
{
namespace sse_const
{
#define constexpr const
#else
template <class X> struct constants<simd_abi::sse, X> {
#endif
    alignas(64) static constexpr int    absMaskFloat[4] = {0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff};
    alignas(16) static constexpr uint   signMaskFloat[4] = {0x80000000, 0x80000000, 0x80000000, 0x80000000};
    alignas(16) static constexpr uint   highMaskFloat[4] = {0xfffff000u, 0xfffff000u, 0xfffff000u, 0xfffff000u};
    alignas(16) static constexpr float  oneFloat[4] = {1.f, 1.f, 1.f, 1.f};

    alignas(16) static constexpr short  minShort[8] = {-0x8000, -0x8000, -0x8000, -0x8000, -0x8000, -0x8000, -0x8000, -0x8000};
    alignas(16) static constexpr uchar  one8[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    alignas(16) static constexpr ushort one16[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    alignas(16) static constexpr uint   one32[4] = {1, 1, 1, 1};

    alignas(16) static constexpr ullong one64[2] = {1, 1};
    alignas(16) static constexpr double oneDouble[2] = {1., 1.};
    alignas(16) static constexpr ullong highMaskDouble[2] = {0xfffffffff8000000ull, 0xfffffffff8000000ull};
    alignas(16) static constexpr llong  absMaskDouble[2] = {0x7fffffffffffffffll, 0x7fffffffffffffffll};

    alignas(16) static constexpr ullong signMaskDouble[2] = {0x8000000000000000ull, 0x8000000000000000ull};
    alignas(16) static constexpr ullong frexpMask[2] = {0xbfefffffffffffffull, 0xbfefffffffffffffull};
    alignas(16) static constexpr uint   IndexesFromZero4[4] = { 0, 1, 2, 3 };
    alignas(16) static constexpr ushort IndexesFromZero8[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };

    alignas(16) static constexpr uchar  IndexesFromZero16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    alignas(16) static constexpr uint   AllBitsSet[4] = { 0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU };
    alignas(16) static constexpr uchar  cvti16_i08_shuffle[16] = {
        0, 2, 4, 6, 8, 10, 12, 14, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80};

    using float4 = float[4];
    using double2 = double[2];
#define Vc_2(x_) {x_, x_}
#define Vc_4(x_) {x_, x_, x_, x_}

    alignas(64) static constexpr float4 f32_pi_4      = Vc_4((constants<simd_abi::scalar, X>::f32_pi_4));
    alignas(16) static constexpr float4 f32_pi_4_hi   = Vc_4((constants<simd_abi::scalar, X>::f32_pi_4_hi));
    alignas(16) static constexpr float4 f32_pi_4_rem1 = Vc_4((constants<simd_abi::scalar, X>::f32_pi_4_rem1));
    alignas(16) static constexpr float4 f32_pi_4_rem2 = Vc_4((constants<simd_abi::scalar, X>::f32_pi_4_rem2));

    alignas(16) static constexpr float4 f32_1_16 = Vc_4(0.0625f);
    alignas(16) static constexpr float4 f32_16 = Vc_4(16.f);
    alignas(16) static constexpr float4 f32_cos_c0 = Vc_4(4.166664568298827e-2f);  // ~ 1/4!
    alignas(16) static constexpr float4 f32_cos_c1 = Vc_4(-1.388731625493765e-3f); // ~-1/6!
    alignas(16) static constexpr float4 f32_cos_c2 = Vc_4(2.443315711809948e-5f);  // ~ 1/8!
    alignas(16) static constexpr float4 f32_sin_c0 = Vc_4(-1.6666654611e-1f); // ~-1/3!
    alignas(16) static constexpr float4 f32_sin_c1 = Vc_4(8.3321608736e-3f);  // ~ 1/5!
    alignas(16) static constexpr float4 f32_sin_c2 = Vc_4(-1.9515295891e-4f); // ~-1/7!
    alignas(16) static constexpr float4 f32_loss_threshold = Vc_4(8192.f); // loss threshold
    alignas(16) static constexpr float4 f32_4_pi = Vc_4((float_const< 1, 0x22F983, 0>)); // 1.27323949337005615234375 = 4/π
    alignas(16) static constexpr float4 f32_pi_2 = Vc_4((float_const< 1, 0x490FDB, 0>)); // π/2
    alignas(16) static constexpr float4 f32_pi = Vc_4((float_const< 1, 0x490FDB, 1>)); // π
    alignas(16) static constexpr float4 f32_atan_p0 = Vc_4(8.05374449538e-2f); // atan P coefficients
    alignas(16) static constexpr float4 f32_atan_p1 = Vc_4(1.38776856032e-1f); // atan P coefficients
    alignas(16) static constexpr float4 f32_atan_p2 = Vc_4(1.99777106478e-1f); // atan P coefficients
    alignas(16) static constexpr float4 f32_atan_p3 = Vc_4(3.33329491539e-1f); // atan P coefficients
    alignas(16) static constexpr float4 f32_atan_threshold_hi = Vc_4(2.414213562373095f); // tan( 3/8 π )
    alignas(16) static constexpr float4 f32_atan_threshold_lo = Vc_4(0.414213562373095f); // tan( 1/8 π ) lower threshold for special casing in atan
    alignas(16) static constexpr float4 f32_pi_2_rem = Vc_4((float_const<-1, 0x3BBD2E, -25>)); // remainder of pi/2
    alignas(16) static constexpr float4 f32_small_asin_input = Vc_4(1.e-4f); // small asin input threshold
    alignas(16) static constexpr float4 f32_large_asin_input = Vc_4(0.f); // padding (for alignment with double)
    alignas(16) static constexpr float4 f32_asin_c0_0 = Vc_4(4.2163199048e-2f); // asinCoeff0
    alignas(16) static constexpr float4 f32_asin_c0_1 = Vc_4(2.4181311049e-2f); // asinCoeff0
    alignas(16) static constexpr float4 f32_asin_c0_2 = Vc_4(4.5470025998e-2f); // asinCoeff0
    alignas(16) static constexpr float4 f32_asin_c0_3 = Vc_4(7.4953002686e-2f); // asinCoeff0
    alignas(16) static constexpr float4 f32_asin_c0_4 = Vc_4(1.6666752422e-1f); // asinCoeff0

    alignas(16) static constexpr double2 f64_pi_4      = Vc_2((double_const< 1, 0x921fb54442d18, -1>)); // π/4
    alignas(16) static constexpr double2 f64_pi_4_hi   = Vc_2((double_const< 1, 0x921fb40000000, -1>)); // π/4 - 30bits precision
    alignas(16) static constexpr double2 f64_pi_4_rem1 = Vc_2((double_const< 1, 0x4442d00000000, -25>)); // π/4 remainder1 - 32bits precision
    alignas(16) static constexpr double2 f64_pi_4_rem2 = Vc_2((double_const< 1, 0x8469898cc5170, -49>)); // π/4 remainder2
    alignas(16) static constexpr double2 f64_1_16 = Vc_2(0.0625);
    alignas(16) static constexpr double2 f64_16 = Vc_2(16.);
    alignas(16) static constexpr double2 f64_cos_c0  = Vc_2((double_const< 1, 0x555555555554b, -5 >)); // ~ 1/4!
    alignas(16) static constexpr double2 f64_cos_c1  = Vc_2((double_const<-1, 0x6c16c16c14f91, -10>)); // ~-1/6!
    alignas(16) static constexpr double2 f64_cos_c2  = Vc_2((double_const< 1, 0xa01a019c844f5, -16>)); // ~ 1/8!
    alignas(16) static constexpr double2 f64_cos_c3  = Vc_2((double_const<-1, 0x27e4f7eac4bc6, -22>)); // ~-1/10!
    alignas(16) static constexpr double2 f64_cos_c4  = Vc_2((double_const< 1, 0x1ee9d7b4e3f05, -29>)); // ~ 1/12!
    alignas(16) static constexpr double2 f64_cos_c5  = Vc_2((double_const<-1, 0x8fa49a0861a9b, -37>)); // ~-1/14!
    alignas(16) static constexpr double2 f64_sin_c0  = Vc_2((double_const<-1, 0x5555555555548, -3 >)); // ~-1/3!
    alignas(16) static constexpr double2 f64_sin_c1  = Vc_2((double_const< 1, 0x111111110f7d0, -7 >)); // ~ 1/5!
    alignas(16) static constexpr double2 f64_sin_c2  = Vc_2((double_const<-1, 0xa01a019bfdf03, -13>)); // ~-1/7!
    alignas(16) static constexpr double2 f64_sin_c3  = Vc_2((double_const< 1, 0x71de3567d48a1, -19>)); // ~ 1/9!
    alignas(16) static constexpr double2 f64_sin_c4  = Vc_2((double_const<-1, 0xae5e5a9291f5d, -26>)); // ~-1/11!
    alignas(16) static constexpr double2 f64_sin_c5  = Vc_2((double_const< 1, 0x5d8fd1fd19ccd, -33>)); // ~ 1/13!
    alignas(16) static constexpr double2 f64_4_pi    = Vc_2((double_const< 1, 0x8BE60DB939105, 0 >)); // 4/π
    alignas(16) static constexpr double2 f64_pi_2    = Vc_2((double_const< 1, 0x921fb54442d18, 0 >)); // π/2
    alignas(16) static constexpr double2 f64_pi      = Vc_2((double_const< 1, 0x921fb54442d18, 1 >)); // π
    alignas(16) static constexpr double2 f64_atan_p0 = Vc_2((double_const<-1, 0xc007fa1f72594, -1>)); // atan P coefficients
    alignas(16) static constexpr double2 f64_atan_p1 = Vc_2((double_const<-1, 0x028545b6b807a, 4 >)); // atan P coefficients
    alignas(16) static constexpr double2 f64_atan_p2 = Vc_2((double_const<-1, 0x2c08c36880273, 6 >)); // atan P coefficients
    alignas(16) static constexpr double2 f64_atan_p3 = Vc_2((double_const<-1, 0xeb8bf2d05ba25, 6 >)); // atan P coefficients
    alignas(16) static constexpr double2 f64_atan_p4 = Vc_2((double_const<-1, 0x03669fd28ec8e, 6 >)); // atan P coefficients
    alignas(16) static constexpr double2 f64_atan_q0 = Vc_2((double_const< 1, 0x8dbc45b14603c, 4 >)); // atan Q coefficients
    alignas(16) static constexpr double2 f64_atan_q1 = Vc_2((double_const< 1, 0x4a0dd43b8fa25, 7 >)); // atan Q coefficients
    alignas(16) static constexpr double2 f64_atan_q2 = Vc_2((double_const< 1, 0xb0e18d2e2be3b, 8 >)); // atan Q coefficients
    alignas(16) static constexpr double2 f64_atan_q3 = Vc_2((double_const< 1, 0xe563f13b049ea, 8 >)); // atan Q coefficients
    alignas(16) static constexpr double2 f64_atan_q4 = Vc_2((double_const< 1, 0x8519efbbd62ec, 7 >)); // atan Q coefficients
    alignas(16) static constexpr double2 f64_atan_threshold_hi = Vc_2((double_const< 1, 0x3504f333f9de6, 1>)); // tan( 3/8 π )
    alignas(16) static constexpr double2 f64_atan_threshold_lo = Vc_2(0.66);                                 // lower threshold for special casing in atan
    alignas(16) static constexpr double2 f64_pi_2_rem = Vc_2((double_const< 1, 0x1A62633145C07, -54>)); // remainder of pi/2
    alignas(16) static constexpr double2 f64_small_asin_input = Vc_2(1.e-8); // small asin input threshold
    alignas(16) static constexpr double2 f64_large_asin_input = Vc_2(0.625); // large asin input threshold
    alignas(16) static constexpr double2 f64_asin_c0_0 = Vc_2((double_const< 1, 0x84fc3988e9f08, -9>)); // asinCoeff0
    alignas(16) static constexpr double2 f64_asin_c0_1 = Vc_2((double_const<-1, 0x2079259f9290f, -1>)); // asinCoeff0
    alignas(16) static constexpr double2 f64_asin_c0_2 = Vc_2((double_const< 1, 0xbdff5baf33e6a, 2 >)); // asinCoeff0
    alignas(16) static constexpr double2 f64_asin_c0_3 = Vc_2((double_const<-1, 0x991aaac01ab68, 4 >)); // asinCoeff0
    alignas(16) static constexpr double2 f64_asin_c0_4 = Vc_2((double_const< 1, 0xc896240f3081d, 4 >)); // asinCoeff0
    alignas(16) static constexpr double2 f64_asin_c1_0 = Vc_2((double_const<-1, 0x5f2a2b6bf5d8c, 4 >)); // asinCoeff1
    alignas(16) static constexpr double2 f64_asin_c1_1 = Vc_2((double_const< 1, 0x26219af6a7f42, 7 >)); // asinCoeff1
    alignas(16) static constexpr double2 f64_asin_c1_2 = Vc_2((double_const<-1, 0x7fe08959063ee, 8 >)); // asinCoeff1
    alignas(16) static constexpr double2 f64_asin_c1_3 = Vc_2((double_const< 1, 0x56709b0b644be, 8 >)); // asinCoeff1
    alignas(16) static constexpr double2 f64_asin_c2_0 = Vc_2((double_const< 1, 0x16b9b0bd48ad3, -8>)); // asinCoeff2
    alignas(16) static constexpr double2 f64_asin_c2_1 = Vc_2((double_const<-1, 0x34341333e5c16, -1>)); // asinCoeff2
    alignas(16) static constexpr double2 f64_asin_c2_2 = Vc_2((double_const< 1, 0x5c74b178a2dd9, 2 >)); // asinCoeff2
    alignas(16) static constexpr double2 f64_asin_c2_3 = Vc_2((double_const<-1, 0x04331de27907b, 4 >)); // asinCoeff2
    alignas(16) static constexpr double2 f64_asin_c2_4 = Vc_2((double_const< 1, 0x39007da779259, 4 >)); // asinCoeff2
    alignas(16) static constexpr double2 f64_asin_c2_5 = Vc_2((double_const<-1, 0x0656c06ceafd5, 3 >)); // asinCoeff2
    alignas(16) static constexpr double2 f64_asin_c3_0 = Vc_2((double_const<-1, 0xd7b590b5e0eab, 3 >)); // asinCoeff3
    alignas(16) static constexpr double2 f64_asin_c3_1 = Vc_2((double_const< 1, 0x19fc025fe9054, 6 >)); // asinCoeff3
    alignas(16) static constexpr double2 f64_asin_c3_2 = Vc_2((double_const<-1, 0x265bb6d3576d7, 7 >)); // asinCoeff3
    alignas(16) static constexpr double2 f64_asin_c3_3 = Vc_2((double_const< 1, 0x1705684ffbf9d, 7 >)); // asinCoeff3
    alignas(16) static constexpr double2 f64_asin_c3_4 = Vc_2((double_const<-1, 0x898220a3607ac, 5 >)); // asinCoeff3

#undef Vc_2
#undef Vc_4

#ifdef Vc_WORK_AROUND_ICE
#undef constexpr
}  // namespace sse_const
}  // namespace x86
#else   // Vc_WORK_AROUND_ICE
};
template <class X> alignas(64) constexpr int    constants<simd_abi::sse, X>::absMaskFloat[4];
template <class X> alignas(16) constexpr uint   constants<simd_abi::sse, X>::signMaskFloat[4];
template <class X> alignas(16) constexpr uint   constants<simd_abi::sse, X>::highMaskFloat[4];
template <class X> alignas(16) constexpr float  constants<simd_abi::sse, X>::oneFloat[4];
template <class X> alignas(16) constexpr short  constants<simd_abi::sse, X>::minShort[8];
template <class X> alignas(16) constexpr uchar  constants<simd_abi::sse, X>::one8[16];
template <class X> alignas(16) constexpr ushort constants<simd_abi::sse, X>::one16[8];
template <class X> alignas(16) constexpr uint   constants<simd_abi::sse, X>::one32[4];
template <class X> alignas(16) constexpr ullong constants<simd_abi::sse, X>::one64[2];
template <class X> alignas(16) constexpr double constants<simd_abi::sse, X>::oneDouble[2];
template <class X> alignas(16) constexpr ullong constants<simd_abi::sse, X>::highMaskDouble[2];
template <class X> alignas(16) constexpr llong  constants<simd_abi::sse, X>::absMaskDouble[2];
template <class X> alignas(16) constexpr ullong constants<simd_abi::sse, X>::signMaskDouble[2];
template <class X> alignas(16) constexpr ullong constants<simd_abi::sse, X>::frexpMask[2];
template <class X> alignas(16) constexpr uint   constants<simd_abi::sse, X>::IndexesFromZero4[4];
template <class X> alignas(16) constexpr ushort constants<simd_abi::sse, X>::IndexesFromZero8[8];
template <class X> alignas(16) constexpr uchar  constants<simd_abi::sse, X>::IndexesFromZero16[16];
template <class X> alignas(16) constexpr uint   constants<simd_abi::sse, X>::AllBitsSet[4];
template <class X> alignas(16) constexpr uchar  constants<simd_abi::sse, X>::cvti16_i08_shuffle[16];

template <class X> alignas(64) constexpr typename constants<simd_abi::sse, X>::float4  constants<simd_abi::sse, X>::f32_pi_4;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::float4  constants<simd_abi::sse, X>::f32_pi_4_hi;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::float4  constants<simd_abi::sse, X>::f32_pi_4_rem1;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::float4  constants<simd_abi::sse, X>::f32_pi_4_rem2;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::float4  constants<simd_abi::sse, X>::f32_1_16;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::float4  constants<simd_abi::sse, X>::f32_16;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::float4  constants<simd_abi::sse, X>::f32_cos_c0;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::float4  constants<simd_abi::sse, X>::f32_cos_c1;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::float4  constants<simd_abi::sse, X>::f32_cos_c2;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::float4  constants<simd_abi::sse, X>::f32_sin_c0;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::float4  constants<simd_abi::sse, X>::f32_sin_c1;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::float4  constants<simd_abi::sse, X>::f32_sin_c2;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::float4  constants<simd_abi::sse, X>::f32_loss_threshold;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::float4  constants<simd_abi::sse, X>::f32_4_pi;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::float4  constants<simd_abi::sse, X>::f32_pi_2;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::float4  constants<simd_abi::sse, X>::f32_pi;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::float4  constants<simd_abi::sse, X>::f32_atan_p0;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::float4  constants<simd_abi::sse, X>::f32_atan_p1;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::float4  constants<simd_abi::sse, X>::f32_atan_p2;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::float4  constants<simd_abi::sse, X>::f32_atan_p3;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::float4  constants<simd_abi::sse, X>::f32_atan_threshold_hi;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::float4  constants<simd_abi::sse, X>::f32_atan_threshold_lo;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::float4  constants<simd_abi::sse, X>::f32_pi_2_rem;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::float4  constants<simd_abi::sse, X>::f32_small_asin_input;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::float4  constants<simd_abi::sse, X>::f32_large_asin_input;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::float4  constants<simd_abi::sse, X>::f32_asin_c0_0;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::float4  constants<simd_abi::sse, X>::f32_asin_c0_1;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::float4  constants<simd_abi::sse, X>::f32_asin_c0_2;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::float4  constants<simd_abi::sse, X>::f32_asin_c0_3;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::float4  constants<simd_abi::sse, X>::f32_asin_c0_4;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_pi_4;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_pi_4_hi;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_pi_4_rem1;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_pi_4_rem2;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_1_16;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_16;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_cos_c0;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_cos_c1;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_cos_c2;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_cos_c3;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_cos_c4;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_cos_c5;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_sin_c0;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_sin_c1;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_sin_c2;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_sin_c3;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_sin_c4;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_sin_c5;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_4_pi;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_pi_2;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_pi;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_atan_p0;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_atan_p1;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_atan_p2;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_atan_p3;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_atan_p4;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_atan_q0;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_atan_q1;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_atan_q2;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_atan_q3;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_atan_q4;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_atan_threshold_hi;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_atan_threshold_lo;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_pi_2_rem;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_small_asin_input;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_large_asin_input;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_asin_c0_0;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_asin_c0_1;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_asin_c0_2;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_asin_c0_3;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_asin_c0_4;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_asin_c1_0;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_asin_c1_1;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_asin_c1_2;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_asin_c1_3;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_asin_c2_0;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_asin_c2_1;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_asin_c2_2;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_asin_c2_3;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_asin_c2_4;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_asin_c2_5;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_asin_c3_0;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_asin_c3_1;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_asin_c3_2;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_asin_c3_3;
template <class X> alignas(16) constexpr typename constants<simd_abi::sse, X>::double2 constants<simd_abi::sse, X>::f64_asin_c3_4;

namespace x86
{
using sse_const = constants<simd_abi::sse>;
}  // namespace x86
#endif  // Vc_WORK_AROUND_ICE
#endif  // Vc_HAVE_SSE_ABI

#ifdef Vc_HAVE_AVX
#ifdef Vc_WORK_AROUND_ICE
namespace x86
{
namespace avx_const
{
#define constexpr const
#else   // Vc_WORK_AROUND_ICE
template <class X>
struct constants<simd_abi::avx, X> : public constants<simd_abi::scalar, X> {
#endif  // Vc_WORK_AROUND_ICE
    alignas(64) static constexpr ullong IndexesFromZero64[ 4] = { 0, 1, 2, 3 };
    alignas(32) static constexpr uint   IndexesFromZero32[ 8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    alignas(32) static constexpr ushort IndexesFromZero16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    alignas(32) static constexpr uchar  IndexesFromZero8 [32] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 };

    alignas(32) static constexpr uint AllBitsSet[8] = {
        0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU
    };

    static constexpr   uint absMaskFloat[2] = { 0xffffffffu, 0x7fffffffu };
    static constexpr   uint signMaskFloat[2] = { 0x0u, 0x80000000u };
    static constexpr   uint highMaskFloat = 0xfffff000u;
    static constexpr  float oneFloat = 1.f;
    alignas(float) static constexpr ushort one16[2] = { 1, 1 };
    alignas(float) static constexpr  uchar one8[4] = { 1, 1, 1, 1 };
    static constexpr  float _2_pow_31 = 1u << 31;
    static constexpr ullong highMaskDouble = 0xfffffffff8000000ull;
    static constexpr double oneDouble = 1.;
#ifdef Vc_WORK_AROUND_ICE
#undef constexpr
#undef Vc_WORK_AROUND_ICE
}  // namespace avx_const
}  // namespace x86
#else   // Vc_WORK_AROUND_ICE
};
template <class X> alignas(64) constexpr ullong constants<simd_abi::avx, X>::IndexesFromZero64[ 4];
template <class X> alignas(32) constexpr uint   constants<simd_abi::avx, X>::IndexesFromZero32[ 8];
template <class X> alignas(32) constexpr ushort constants<simd_abi::avx, X>::IndexesFromZero16[16];
template <class X> alignas(32) constexpr uchar  constants<simd_abi::avx, X>::IndexesFromZero8 [32];
template <class X> alignas(32) constexpr uint   constants<simd_abi::avx, X>::AllBitsSet[8];
template <class X> constexpr   uint constants<simd_abi::avx, X>::absMaskFloat[2];
template <class X> constexpr   uint constants<simd_abi::avx, X>::signMaskFloat[2];
template <class X> constexpr   uint constants<simd_abi::avx, X>::highMaskFloat;
template <class X> constexpr  float constants<simd_abi::avx, X>::oneFloat;
template <class X> alignas(float) constexpr ushort constants<simd_abi::avx, X>::one16[2];
template <class X> alignas(float) constexpr  uchar constants<simd_abi::avx, X>::one8[4];
template <class X> constexpr  float constants<simd_abi::avx, X>::_2_pow_31;
template <class X> constexpr ullong constants<simd_abi::avx, X>::highMaskDouble;
template <class X> constexpr double constants<simd_abi::avx, X>::oneDouble;
namespace x86
{
using avx_const = constants<simd_abi::avx>;
}  // namespace x86
#endif  // Vc_WORK_AROUND_ICE
#endif  // Vc_HAVE_AVX

}  // namespace detail
Vc_VERSIONED_NAMESPACE_END
#endif  // VC_DETAIL_X86_CONST_H_
