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

#ifndef VC_DETAIL_CONST_H_
#define VC_DETAIL_CONST_H_

#include "synopsis.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
template <int exponent, bool = (exponent < 0),
          bool = (exponent < std::numeric_limits<unsigned>::digits)>
struct double_2_pow_impl;

template <> struct double_2_pow_impl<0, false, true> {
    static constexpr double value = 1.;
};
template <int exponent> struct double_2_pow_impl<exponent, true, true> {
    static constexpr double value = 1. / double_2_pow_impl<-exponent>::value;
};
template <int exponent> struct double_2_pow_impl<exponent, false, true> {
    static constexpr double value = (1u << exponent);
};
template <int exponent> struct double_2_pow_impl<exponent, false, false> {
    static constexpr double value =
        (~0u + 1.) *
        double_2_pow_impl<exponent - std::numeric_limits<unsigned>::digits>::value;
};

template <int exponent>
constexpr double double_2_pow = double_2_pow_impl<exponent>::value;

template <int sign, unsigned long long mantissa, int exponent>
constexpr double double_const = (static_cast<double>((mantissa & 0x000fffffffffffffull) |
                                                     0x0010000000000000ull) /
                                 0x0010000000000000ull) *
                                double_2_pow<exponent> * sign;
template <int sign, unsigned int mantissa, int exponent>
constexpr float float_const = (float((mantissa & 0x007fffffu) | 0x00800000u) /
                               0x00800000u) *
                              float(double_2_pow<exponent>) * sign;

template <class T, class = void> struct constants {
    static constexpr float f32_pi_4      = float_const< 1, 0x490FDB, -1>;
    static constexpr float f32_pi_4_hi   = float_const< 1, 0x491000, -1>;
    static constexpr float f32_pi_4_rem1 = float_const<-1, 0x157000, -19>;
    static constexpr float f32_pi_4_rem2 = float_const<-1, 0x6F4B9F, -32>;

    static constexpr float f32_1_16 = 0.0625f;
    static constexpr float f32_16 = 16.f;
    static constexpr float f32_cos_c0 = 4.166664568298827e-2f;  // ~ 1/4!
    static constexpr float f32_cos_c1 = -1.388731625493765e-3f; // ~-1/6!
    static constexpr float f32_cos_c2 = 2.443315711809948e-5f;  // ~ 1/8!
    static constexpr float f32_sin_c0 = -1.6666654611e-1f; // ~-1/3!
    static constexpr float f32_sin_c1 = 8.3321608736e-3f;  // ~ 1/5!
    static constexpr float f32_sin_c2 = -1.9515295891e-4f; // ~-1/7!
    static constexpr float f32_loss_threshold = 8192.f; // loss threshold
    static constexpr float f32_4_pi = float_const< 1, 0x22F983, 0>; // 1.27323949337005615234375 = 4/π
    static constexpr float f32_pi_2 = float_const< 1, 0x490FDB, 0>; // π/2
    static constexpr float f32_pi = float_const< 1, 0x490FDB, 1>; // π
    static constexpr float f32_atan_p0 = 8.05374449538e-2f; // atan P coefficients
    static constexpr float f32_atan_p1 = 1.38776856032e-1f; // atan P coefficients
    static constexpr float f32_atan_p2 = 1.99777106478e-1f; // atan P coefficients
    static constexpr float f32_atan_p3 = 3.33329491539e-1f; // atan P coefficients
    static constexpr float f32_atan_threshold_hi = 2.414213562373095f; // tan( 3/8 π )
    static constexpr float f32_atan_threshold_lo = 0.414213562373095f; // tan( 1/8 π ) lower threshold for special casing in atan
    static constexpr float f32_pi_2_rem = float_const<-1, 0x3BBD2E, -25>; // remainder of pi/2
    static constexpr float f32_small_asin_input = 1.e-4f; // small asin input threshold
    static constexpr float f32_large_asin_input = 0.f; // padding (for alignment with double)
    static constexpr float f32_asin_c0_0 = 4.2163199048e-2f; // asinCoeff0
    static constexpr float f32_asin_c0_1 = 2.4181311049e-2f; // asinCoeff0
    static constexpr float f32_asin_c0_2 = 4.5470025998e-2f; // asinCoeff0
    static constexpr float f32_asin_c0_3 = 7.4953002686e-2f; // asinCoeff0
    static constexpr float f32_asin_c0_4 = 1.6666752422e-1f; // asinCoeff0

    static constexpr double f64_pi_4      = double_const< 1, 0x921fb54442d18, -1>; // π/4
    static constexpr double f64_pi_4_hi   = double_const< 1, 0x921fb40000000, -1>; // π/4 - 30bits precision
    static constexpr double f64_pi_4_rem1 = double_const< 1, 0x4442d00000000, -25>; // π/4 remainder1 - 32bits precision
    static constexpr double f64_pi_4_rem2 = double_const< 1, 0x8469898cc5170, -49>; // π/4 remainder2
    static constexpr double f64_1_16 = 0.0625;
    static constexpr double f64_16 = 16.;
    static constexpr double f64_cos_c0  = double_const< 1, 0x555555555554b, -5 >; // ~ 1/4!
    static constexpr double f64_cos_c1  = double_const<-1, 0x6c16c16c14f91, -10>; // ~-1/6!
    static constexpr double f64_cos_c2  = double_const< 1, 0xa01a019c844f5, -16>; // ~ 1/8!
    static constexpr double f64_cos_c3  = double_const<-1, 0x27e4f7eac4bc6, -22>; // ~-1/10!
    static constexpr double f64_cos_c4  = double_const< 1, 0x1ee9d7b4e3f05, -29>; // ~ 1/12!
    static constexpr double f64_cos_c5  = double_const<-1, 0x8fa49a0861a9b, -37>; // ~-1/14!
    static constexpr double f64_sin_c0  = double_const<-1, 0x5555555555548, -3 >; // ~-1/3!
    static constexpr double f64_sin_c1  = double_const< 1, 0x111111110f7d0, -7 >; // ~ 1/5!
    static constexpr double f64_sin_c2  = double_const<-1, 0xa01a019bfdf03, -13>; // ~-1/7!
    static constexpr double f64_sin_c3  = double_const< 1, 0x71de3567d48a1, -19>; // ~ 1/9!
    static constexpr double f64_sin_c4  = double_const<-1, 0xae5e5a9291f5d, -26>; // ~-1/11!
    static constexpr double f64_sin_c5  = double_const< 1, 0x5d8fd1fd19ccd, -33>; // ~ 1/13!
    static constexpr double f64_4_pi    = double_const< 1, 0x8BE60DB939105, 0 >; // 4/π
    static constexpr double f64_pi_2    = double_const< 1, 0x921fb54442d18, 0 >; // π/2
    static constexpr double f64_pi      = double_const< 1, 0x921fb54442d18, 1 >; // π
    static constexpr double f64_atan_p0 = double_const<-1, 0xc007fa1f72594, -1>; // atan P coefficients
    static constexpr double f64_atan_p1 = double_const<-1, 0x028545b6b807a, 4 >; // atan P coefficients
    static constexpr double f64_atan_p2 = double_const<-1, 0x2c08c36880273, 6 >; // atan P coefficients
    static constexpr double f64_atan_p3 = double_const<-1, 0xeb8bf2d05ba25, 6 >; // atan P coefficients
    static constexpr double f64_atan_p4 = double_const<-1, 0x03669fd28ec8e, 6 >; // atan P coefficients
    static constexpr double f64_atan_q0 = double_const< 1, 0x8dbc45b14603c, 4 >; // atan Q coefficients
    static constexpr double f64_atan_q1 = double_const< 1, 0x4a0dd43b8fa25, 7 >; // atan Q coefficients
    static constexpr double f64_atan_q2 = double_const< 1, 0xb0e18d2e2be3b, 8 >; // atan Q coefficients
    static constexpr double f64_atan_q3 = double_const< 1, 0xe563f13b049ea, 8 >; // atan Q coefficients
    static constexpr double f64_atan_q4 = double_const< 1, 0x8519efbbd62ec, 7 >; // atan Q coefficients
    static constexpr double f64_atan_threshold_hi = double_const< 1, 0x3504f333f9de6, 1>; // tan( 3/8 π )
    static constexpr double f64_atan_threshold_lo = 0.66;                                 // lower threshold for special casing in atan
    static constexpr double f64_pi_2_rem = double_const< 1, 0x1A62633145C07, -54>; // remainder of pi/2
    static constexpr double f64_small_asin_input = 1.e-8; // small asin input threshold
    static constexpr double f64_large_asin_input = 0.625; // large asin input threshold
    static constexpr double f64_asin_c0_0 = double_const< 1, 0x84fc3988e9f08, -9>; // asinCoeff0
    static constexpr double f64_asin_c0_1 = double_const<-1, 0x2079259f9290f, -1>; // asinCoeff0
    static constexpr double f64_asin_c0_2 = double_const< 1, 0xbdff5baf33e6a, 2 >; // asinCoeff0
    static constexpr double f64_asin_c0_3 = double_const<-1, 0x991aaac01ab68, 4 >; // asinCoeff0
    static constexpr double f64_asin_c0_4 = double_const< 1, 0xc896240f3081d, 4 >; // asinCoeff0
    static constexpr double f64_asin_c1_0 = double_const<-1, 0x5f2a2b6bf5d8c, 4 >; // asinCoeff1
    static constexpr double f64_asin_c1_1 = double_const< 1, 0x26219af6a7f42, 7 >; // asinCoeff1
    static constexpr double f64_asin_c1_2 = double_const<-1, 0x7fe08959063ee, 8 >; // asinCoeff1
    static constexpr double f64_asin_c1_3 = double_const< 1, 0x56709b0b644be, 8 >; // asinCoeff1
    static constexpr double f64_asin_c2_0 = double_const< 1, 0x16b9b0bd48ad3, -8>; // asinCoeff2
    static constexpr double f64_asin_c2_1 = double_const<-1, 0x34341333e5c16, -1>; // asinCoeff2
    static constexpr double f64_asin_c2_2 = double_const< 1, 0x5c74b178a2dd9, 2 >; // asinCoeff2
    static constexpr double f64_asin_c2_3 = double_const<-1, 0x04331de27907b, 4 >; // asinCoeff2
    static constexpr double f64_asin_c2_4 = double_const< 1, 0x39007da779259, 4 >; // asinCoeff2
    static constexpr double f64_asin_c2_5 = double_const<-1, 0x0656c06ceafd5, 3 >; // asinCoeff2
    static constexpr double f64_asin_c3_0 = double_const<-1, 0xd7b590b5e0eab, 3 >; // asinCoeff3
    static constexpr double f64_asin_c3_1 = double_const< 1, 0x19fc025fe9054, 6 >; // asinCoeff3
    static constexpr double f64_asin_c3_2 = double_const<-1, 0x265bb6d3576d7, 7 >; // asinCoeff3
    static constexpr double f64_asin_c3_3 = double_const< 1, 0x1705684ffbf9d, 7 >; // asinCoeff3
    static constexpr double f64_asin_c3_4 = double_const<-1, 0x898220a3607ac, 5 >; // asinCoeff3
};

template <class Abi,class T> struct trig;

template <class Abi> struct trig<Abi, float> {
    using C = constants<Abi>;

    static Vc_INTRINSIC Vc_CONST simd<float, Abi> load(const float *arr)
    {
        return {arr, flags::vector_aligned};
    }
    static Vc_INTRINSIC Vc_CONST simd<float, Abi> load(float arr)
    {
        return arr;
    }

    static Vc_INTRINSIC simd<float, Abi> pi_4() { return load(C::f32_pi_4); }
    static Vc_INTRINSIC simd<float, Abi> pi_4_hi() { return load(C::f32_pi_4_hi); }
    static Vc_INTRINSIC simd<float, Abi> pi_4_rem1() { return load(C::f32_pi_4_rem1); }
    static Vc_INTRINSIC simd<float, Abi> pi_4_rem2() { return load(C::f32_pi_4_rem2); }
    static Vc_INTRINSIC simd<float, Abi> _1_16() { return load(C::f32_1_16); }
    static Vc_INTRINSIC simd<float, Abi> _16() { return load(C::f32_16); }
    static Vc_INTRINSIC simd<float, Abi> cos_c0() { return load(C::f32_cos_c0); }
    static Vc_INTRINSIC simd<float, Abi> cos_c1() { return load(C::f32_cos_c1); }
    static Vc_INTRINSIC simd<float, Abi> cos_c2() { return load(C::f32_cos_c2); }
    static Vc_INTRINSIC simd<float, Abi> sin_c0() { return load(C::f32_sin_c0); }
    static Vc_INTRINSIC simd<float, Abi> sin_c1() { return load(C::f32_sin_c1); }
    static Vc_INTRINSIC simd<float, Abi> sin_c2() { return load(C::f32_sin_c2); }
    static Vc_INTRINSIC simd<float, Abi> loss_threshold() { return load(C::f32_loss_threshold); }
    static Vc_INTRINSIC simd<float, Abi> _4_pi() { return load(C::f32_4_pi); }
    static Vc_INTRINSIC simd<float, Abi> pi_2() { return load(C::f32_pi_2); }
    static Vc_INTRINSIC simd<float, Abi> pi() { return load(C::f32_pi); }
    static Vc_INTRINSIC simd<float, Abi> atan_p0() { return load(C::f32_atan_p0); }
    static Vc_INTRINSIC simd<float, Abi> atan_p1() { return load(C::f32_atan_p1); }
    static Vc_INTRINSIC simd<float, Abi> atan_p2() { return load(C::f32_atan_p2); }
    static Vc_INTRINSIC simd<float, Abi> atan_p3() { return load(C::f32_atan_p3); }
    static Vc_INTRINSIC simd<float, Abi> atan_threshold_hi() { return load(C::f32_atan_threshold_hi); }
    static Vc_INTRINSIC simd<float, Abi> atan_threshold_lo() { return load(C::f32_atan_threshold_lo); }
    static Vc_INTRINSIC simd<float, Abi> pi_2_rem() { return load(C::f32_pi_2_rem); }
    static Vc_INTRINSIC simd<float, Abi> small_asin_input() { return load(C::f32_small_asin_input); }
    static Vc_INTRINSIC simd<float, Abi> large_asin_input() { return load(C::f32_large_asin_input); }
    static Vc_INTRINSIC simd<float, Abi> asin_c0_0() { return load(C::f32_asin_c0_0); }
    static Vc_INTRINSIC simd<float, Abi> asin_c0_1() { return load(C::f32_asin_c0_1); }
    static Vc_INTRINSIC simd<float, Abi> asin_c0_2() { return load(C::f32_asin_c0_2); }
    static Vc_INTRINSIC simd<float, Abi> asin_c0_3() { return load(C::f32_asin_c0_3); }
    static Vc_INTRINSIC simd<float, Abi> asin_c0_4() { return load(C::f32_asin_c0_4); }
};

template <class Abi> struct trig<Abi, double> {
    using C = constants<Abi>;

    static Vc_INTRINSIC Vc_CONST simd<double, Abi> load(const double *arr)
    {
        return {arr, flags::vector_aligned};
    }
    static Vc_INTRINSIC Vc_CONST simd<double, Abi> load(double arr)
    {
        return arr;
    }

    static Vc_INTRINSIC simd<double, Abi> pi_4() { return load(C::f64_pi_4); }
    static Vc_INTRINSIC simd<double, Abi> pi_4_hi() { return load(C::f64_pi_4_hi); }
    static Vc_INTRINSIC simd<double, Abi> pi_4_rem1() { return load(C::f64_pi_4_rem1); }
    static Vc_INTRINSIC simd<double, Abi> pi_4_rem2() { return load(C::f64_pi_4_rem2); }
    static Vc_INTRINSIC simd<double, Abi> _1_16() { return load(C::f64_1_16); }
    static Vc_INTRINSIC simd<double, Abi> _16() { return load(C::f64_16); }
    static Vc_INTRINSIC simd<double, Abi> cos_c0() { return load(C::f64_cos_c0); }
    static Vc_INTRINSIC simd<double, Abi> cos_c1() { return load(C::f64_cos_c1); }
    static Vc_INTRINSIC simd<double, Abi> cos_c2() { return load(C::f64_cos_c2); }
    static Vc_INTRINSIC simd<double, Abi> cos_c3() { return load(C::f64_cos_c3); }
    static Vc_INTRINSIC simd<double, Abi> cos_c4() { return load(C::f64_cos_c4); }
    static Vc_INTRINSIC simd<double, Abi> cos_c5() { return load(C::f64_cos_c5); }
    static Vc_INTRINSIC simd<double, Abi> sin_c0() { return load(C::f64_sin_c0); }
    static Vc_INTRINSIC simd<double, Abi> sin_c1() { return load(C::f64_sin_c1); }
    static Vc_INTRINSIC simd<double, Abi> sin_c2() { return load(C::f64_sin_c2); }
    static Vc_INTRINSIC simd<double, Abi> sin_c3() { return load(C::f64_sin_c3); }
    static Vc_INTRINSIC simd<double, Abi> sin_c4() { return load(C::f64_sin_c4); }
    static Vc_INTRINSIC simd<double, Abi> sin_c5() { return load(C::f64_sin_c5); }
    static Vc_INTRINSIC simd<double, Abi> loss_threshold() { return load(C::f64_loss_threshold); }
    static Vc_INTRINSIC simd<double, Abi> _4_pi() { return load(C::f64_4_pi); }
    static Vc_INTRINSIC simd<double, Abi> pi_2() { return load(C::f64_pi_2); }
    static Vc_INTRINSIC simd<double, Abi> pi() { return load(C::f64_pi); }
    static Vc_INTRINSIC simd<double, Abi> atan_p0() { return load(C::f64_atan_p0); }
    static Vc_INTRINSIC simd<double, Abi> atan_p1() { return load(C::f64_atan_p1); }
    static Vc_INTRINSIC simd<double, Abi> atan_p2() { return load(C::f64_atan_p2); }
    static Vc_INTRINSIC simd<double, Abi> atan_p3() { return load(C::f64_atan_p3); }
    static Vc_INTRINSIC simd<double, Abi> atan_threshold_hi() { return load(C::f64_atan_threshold_hi); }
    static Vc_INTRINSIC simd<double, Abi> atan_threshold_lo() { return load(C::f64_atan_threshold_lo); }
    static Vc_INTRINSIC simd<double, Abi> pi_2_rem() { return load(C::f64_pi_2_rem); }
    static Vc_INTRINSIC simd<double, Abi> small_asin_input() { return load(C::f64_small_asin_input); }
    static Vc_INTRINSIC simd<double, Abi> large_asin_input() { return load(C::f64_large_asin_input); }
    static Vc_INTRINSIC simd<double, Abi> asin_c0_0() { return load(C::f64_asin_c0_0); }
    static Vc_INTRINSIC simd<double, Abi> asin_c0_1() { return load(C::f64_asin_c0_1); }
    static Vc_INTRINSIC simd<double, Abi> asin_c0_2() { return load(C::f64_asin_c0_2); }
    static Vc_INTRINSIC simd<double, Abi> asin_c0_3() { return load(C::f64_asin_c0_3); }
    static Vc_INTRINSIC simd<double, Abi> asin_c0_4() { return load(C::f64_asin_c0_4); }
};
}  // namespace detail
Vc_VERSIONED_NAMESPACE_END

#ifdef Vc_HAVE_SSE
#include "x86/const.h"
#endif  // Vc_HAVE_SSE

#ifdef Vc_HAVE_NEON
#include "aarch/const.h"
#endif  // Vc_HAVE_NEON

#endif  // VC_DETAIL_CONST_H_

// vim: foldmethod=marker
