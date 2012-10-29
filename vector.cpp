/*  This file is part of the Vc library.

    Copyright (C) 2009-2012 Matthias Kretz <kretz@kde.org>

    Vc is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    Vc is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with Vc.  If not, see <http://www.gnu.org/licenses/>.

*/

#ifndef V_ALIGN
# ifdef __GNUC__
#  define V_ALIGN(n) __attribute__((aligned(n)))
# else
#  define V_ALIGN(n) __declspec(align(n))
# endif
#endif

#include "avx/const_data.h"
#include "sse/const_data.h"
#include <Vc/version.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "common/macros.h"

namespace Vc
{
namespace AVX
{
    // cacheline 1
    V_ALIGN(64) extern const unsigned int   _IndexesFromZero32[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    V_ALIGN(16) extern const unsigned short _IndexesFromZero16[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    V_ALIGN(16) extern const unsigned char  _IndexesFromZero8 [16]= { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

    // cacheline 2
    template<> const double c_sin<double>::data[8] = {
        0.15915494309189533576888376337251436, // 1 over 2pi
        6.2831853071795864769252867665590058,  // 2pi
        1.5707963267948966192313216916397514, // pi over 2
        3.1415926535897932384626433832795029, // pi
        1.666666666666666574148081281236954964697360992431640625e-01, // 1 over 3!
        8.33333333333333321768510160154619370587170124053955078125e-03, // 1 over 5!
        1.984126984126984125263171154784913596813566982746124267578125e-04, // 1 over 7!
        2.755731922398589251095059327045788677423843182623386383056640625e-06 // 1 over 9!
    };

    // cacheline 3
    template<> const float c_sin<float>::data[8] = {
        1.59154936671257019e-01f, // 1 over 2pi
        6.28318548202514648f,     // 2pi
        1.57079637050628662f,     // pi over 2
        3.14159274101257324f,     // pi
        1.66666671633720398e-01f, // 1 over 3!
        8.33333376795053482e-03f, // 1 over 5!
        1.98412701138295233e-04f, // 1 over 7!
        2.75573188446287531e-06f  // 1 over 9!
    };

    const unsigned       int c_general::absMaskFloat[2] = { 0xffffffffu, 0x7fffffffu };
    const unsigned       int c_general::signMaskFloat[2] = { 0x0u, 0x80000000u };
    const unsigned       int c_general::highMaskFloat = 0xfffff000u;
    const              float c_general::oneFloat = 1.f;
    const unsigned     short c_general::minShort[2] = { 0x8000u, 0x8000u };
    const unsigned     short c_general::one16[2] = { 1, 1 };
    const              float c_general::_2power31 = 1u << 31;

    // cacheline 4
    const unsigned long long c_general::highMaskDouble = 0xfffffffff8000000ull;
    const             double c_general::oneDouble = 1.;
    const unsigned long long c_general::frexpMask = 0xbfefffffffffffffull;

    const unsigned long long c_log<double>::data[21] = {
        0x000003ff000003ffull // bias TODO: remove
      , 0x7ff0000000000000ull // exponentMask (+inf)

      , 0x3f1ab4c293c31bb0ull // P[0]
      , 0x3fdfd6f53f5652f2ull // P[1]
      , 0x4012d2baed926911ull // P[2]
      , 0x402cff72c63eeb2eull // P[3]
      , 0x4031efd6924bc84dull // P[4]
      , 0x401ed5637d7edcf8ull // P[5]

      , 0x40269320ae97ef8eull // Q[0]
      , 0x40469d2c4e19c033ull // Q[1]
      , 0x4054bf33a326bdbdull // Q[2]
      , 0x4051c9e2eb5eae21ull // Q[3]
      , 0x4037200a9e1f25b2ull // Q[4]

      , 0xfff0000000000000ull // -inf
      , 0x0010000000000000ull // min()
      , 0x3fe6a09e667f3bcdull // 1/sqrt(2)
      , 0x3fe6300000000000ull // round(ln(2) * 512) / 512
      , 0xbf2bd0105c610ca8ull // ln(2) - round(ln(2) * 512) / 512
      , 0x3fe0000000000000ull // 0.5
      , 0x3fdbcb7b1526e50eull // log10(e)
      , 0x3ff71547652b82feull // log2(e)
    };

    template<> const unsigned int c_log<float>::data[21] = {
        0x0000007fu // bias TODO: remove
      , 0x7f800000u // exponentMask (+inf)

      , 0x3d9021bbu //  7.0376836292e-2f // P[0]
      , 0xbdebd1b8u // -1.1514610310e-1f // P[1]
      , 0x3def251au //  1.1676998740e-1f // P[2]
      , 0xbdfe5d4fu // -1.2420140846e-1f // P[3]
      , 0x3e11e9bfu //  1.4249322787e-1f // P[4]
      , 0xbe2aae50u // -1.6668057665e-1f // P[5]
      , 0x3e4cceacu //  2.0000714765e-1f // P[6]
      , 0xbe7ffffcu // -2.4999993993e-1f // P[7]
      , 0x3eaaaaaau //  3.3333331174e-1f // P[8]
      , 0           // padding because of c_log<double>
      , 0           // padding because of c_log<double>

      , 0xff800000u // -inf
      , 0x00800000u // min()
      , 0x3f3504f3u // 1/sqrt(2)
      , 0x3f318000u // round(ln(2) * 512) / 512
      , 0xb95e8083u // ln(2) - round(ln(2) * 512) / 512
      , 0x3f000000u // 0.5
      , 0x3ede5bd9u // log10(e)
      , 0x3fb8aa3bu // log2(e)
    };
} // namespace AVX

namespace SSE
{
    // cacheline 1
    V_ALIGN(64) const int c_general::absMaskFloat[4] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
    V_ALIGN(16) const unsigned int c_general::signMaskFloat[4] = { 0x80000000, 0x80000000, 0x80000000, 0x80000000 };
    V_ALIGN(16) const unsigned int c_general::highMaskFloat[4] = { 0xfffff000u, 0xfffff000u, 0xfffff000u, 0xfffff000u };
    V_ALIGN(16) const short c_general::minShort[8] = { -0x8000, -0x8000, -0x8000, -0x8000, -0x8000, -0x8000, -0x8000, -0x8000 };
    V_ALIGN(16) extern const unsigned short _IndexesFromZero8[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };

    // cacheline 2
    V_ALIGN(16) extern const unsigned int   _IndexesFromZero4[4] = { 0, 1, 2, 3 };
    V_ALIGN(16) const unsigned short c_general::one16[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };
    V_ALIGN(16) const unsigned int c_general::one32[4] = { 1, 1, 1, 1 };
    V_ALIGN(16) const float c_general::oneFloat[4] = { 1.f, 1.f, 1.f, 1.f };

    // cacheline 3
    V_ALIGN(16) const unsigned long long c_general::highMaskDouble[2] = { 0xfffffffff8000000ull, 0xfffffffff8000000ull };
    V_ALIGN(16) const double c_general::oneDouble[2] = { 1., 1. };
    V_ALIGN(16) const long long c_general::absMaskDouble[2] = { 0x7fffffffffffffffll, 0x7fffffffffffffffll };
    V_ALIGN(16) const unsigned long long c_general::signMaskDouble[2] = { 0x8000000000000000ull, 0x8000000000000000ull };
    V_ALIGN(16) const unsigned long long c_general::frexpMask[2] = { 0xbfefffffffffffffull, 0xbfefffffffffffffull };

#define _2(x) x, x
    template<> const double c_trig<double>::data[] = {
    // cacheline 4
        _2(Vc_buildDouble(1, 0x921fb54442d18, -1)), // π/4
        _2(Vc_buildDouble(1, 0x921fb40000000, -1)), // π/4 - 30bits precision
        _2(Vc_buildDouble(1, 0x4442d00000000, -25)), // π/4 remainder1 - 32bits precision
        _2(Vc_buildDouble(1, 0x8469898cc5170, -49)), // π/4 remainder2
    // cacheline 5
        _2(0.0625),
        _2(16.),
        _2(0.), // padding
        _2(0.), // padding
    // cacheline 6
        _2(Vc_buildDouble( 1, 0x555555555554b,  -5)), // ~ 1/4!
        _2(Vc_buildDouble(-1, 0x6c16c16c14f91, -10)), // ~-1/6!
        _2(Vc_buildDouble( 1, 0xa01a019c844f5, -16)), // ~ 1/8!
        _2(Vc_buildDouble(-1, 0x27e4f7eac4bc6, -22)), // ~-1/10!
    // cacheline 7
        _2(Vc_buildDouble( 1, 0x1ee9d7b4e3f05, -29)), // ~ 1/12!
        _2(Vc_buildDouble(-1, 0x8fa49a0861a9b, -37)), // ~-1/14!
        _2(Vc_buildDouble(-1, 0x5555555555548,  -3)), // ~-1/3!
        _2(Vc_buildDouble( 1, 0x111111110f7d0,  -7)), // ~ 1/5!
    // cacheline 8
        _2(Vc_buildDouble(-1, 0xa01a019bfdf03, -13)), // ~-1/7!
        _2(Vc_buildDouble( 1, 0x71de3567d48a1, -19)), // ~ 1/9!
        _2(Vc_buildDouble(-1, 0xae5e5a9291f5d, -26)), // ~-1/11!
        _2(Vc_buildDouble( 1, 0x5d8fd1fd19ccd, -33)), // ~ 1/13!
    // cacheline 9
        _2(0.), // padding (for alignment with float)
        _2(Vc_buildDouble(1, 0x8BE60DB939105,  0)), // 4/π
        _2(Vc_buildDouble(1, 0x921fb54442d18,  0)), // π/2
        _2(Vc_buildDouble(1, 0x921fb54442d18,  1)), // π
    // cacheline 10
        _2(Vc_buildDouble(-1, 0xc007fa1f72594, -1)), // atan P coefficients
        _2(Vc_buildDouble(-1, 0x028545b6b807a,  4)), // atan P coefficients
        _2(Vc_buildDouble(-1, 0x2c08c36880273,  6)), // atan P coefficients
        _2(Vc_buildDouble(-1, 0xeb8bf2d05ba25,  6)), // atan P coefficients
    // cacheline 11
        _2(Vc_buildDouble(-1, 0x03669fd28ec8e,  6)), // atan P coefficients
        _2(Vc_buildDouble( 1, 0x8dbc45b14603c,  4)), // atan Q coefficients
        _2(Vc_buildDouble( 1, 0x4a0dd43b8fa25,  7)), // atan Q coefficients
        _2(Vc_buildDouble( 1, 0xb0e18d2e2be3b,  8)), // atan Q coefficients
    // cacheline 12
        _2(Vc_buildDouble( 1, 0xe563f13b049ea,  8)), // atan Q coefficients
        _2(Vc_buildDouble( 1, 0x8519efbbd62ec,  7)), // atan Q coefficients
        _2(Vc_buildDouble( 1, 0x3504f333f9de6,  1)), // tan( 3/8 π )
        _2(0.66),                                    // lower threshold for special casing in atan
    // cacheline 13
        _2(Vc_buildDouble(1, 0x1A62633145C07, -54)), // remainder of pi/2
        _2(1.e-8), // small asin input threshold
        _2(0.625), // large asin input threshold
        _2(0.), // padding
    // cacheline 14
        _2(Vc_buildDouble( 1, 0x84fc3988e9f08, -9)), // asinCoeff0
        _2(Vc_buildDouble(-1, 0x2079259f9290f, -1)), // asinCoeff0
        _2(Vc_buildDouble( 1, 0xbdff5baf33e6a,  2)), // asinCoeff0
        _2(Vc_buildDouble(-1, 0x991aaac01ab68,  4)), // asinCoeff0
    // cacheline 15
        _2(Vc_buildDouble( 1, 0xc896240f3081d,  4)), // asinCoeff0
        _2(Vc_buildDouble(-1, 0x5f2a2b6bf5d8c,  4)), // asinCoeff1
        _2(Vc_buildDouble( 1, 0x26219af6a7f42,  7)), // asinCoeff1
        _2(Vc_buildDouble(-1, 0x7fe08959063ee,  8)), // asinCoeff1
    // cacheline 16
        _2(Vc_buildDouble( 1, 0x56709b0b644be,  8)), // asinCoeff1
        _2(Vc_buildDouble( 1, 0x16b9b0bd48ad3, -8)), // asinCoeff2
        _2(Vc_buildDouble(-1, 0x34341333e5c16, -1)), // asinCoeff2
        _2(Vc_buildDouble( 1, 0x5c74b178a2dd9,  2)), // asinCoeff2
    // cacheline 17
        _2(Vc_buildDouble(-1, 0x04331de27907b,  4)), // asinCoeff2
        _2(Vc_buildDouble( 1, 0x39007da779259,  4)), // asinCoeff2
        _2(Vc_buildDouble(-1, 0x0656c06ceafd5,  3)), // asinCoeff2
        _2(Vc_buildDouble(-1, 0xd7b590b5e0eab,  3)), // asinCoeff3
    // cacheline 18
        _2(Vc_buildDouble( 1, 0x19fc025fe9054,  6)), // asinCoeff3
        _2(Vc_buildDouble(-1, 0x265bb6d3576d7,  7)), // asinCoeff3
        _2(Vc_buildDouble( 1, 0x1705684ffbf9d,  7)), // asinCoeff3
        _2(Vc_buildDouble(-1, 0x898220a3607ac,  5)), // asinCoeff3
    };
#undef _2
#define _4(x) x, x, x, x
    template<> const float c_trig<float>::data[] = {
    // cacheline
        _4(Vc_buildFloat(1, 0x490FDB,  -1)), // π/4
        _4(Vc_buildFloat(1, 0x490000,  -1)), // π/4 - 16 bits precision
        _4(Vc_buildFloat(1, 0x7DA000, -13)), // π/4 remainder1 - 13 bits precision
        _4(Vc_buildFloat(1, 0x222169, -25)), // π/4 remainder2
    // cacheline
        _4(0.0625f),
        _4(16.f),
        _4(0.f), // padding
        _4(0.f), // padding
    // cacheline
        _4(4.166664568298827e-2f),  // ~ 1/4!
        _4(-1.388731625493765e-3f), // ~-1/6!
        _4(2.443315711809948e-5f),  // ~ 1/8!
        _4(0.f), // padding (for alignment with double)
    // cacheline
        _4(0.f), // padding (for alignment with double)
        _4(0.f), // padding (for alignment with double)
        _4(-1.6666654611e-1f), // ~-1/3!
        _4(8.3321608736e-3f),  // ~ 1/5!
    // cacheline
        _4(-1.9515295891e-4f), // ~-1/7!
        _4(0.f), // padding (for alignment with double)
        _4(0.f), // padding (for alignment with double)
        _4(0.f), // padding (for alignment with double)
    // cacheline
        _4(8192.f), // loss threshold
        _4(Vc_buildFloat(1, 0x22F983, 0)), // 1.27323949337005615234375 = 4/π
        _4(Vc_buildFloat(1, 0x490FDB, 0)), // π/2
        _4(Vc_buildFloat(1, 0x490FDB, 1)), // π
    // cacheline
        _4(8.05374449538e-2), // atan P coefficients
        _4(1.38776856032e-1), // atan P coefficients
        _4(1.99777106478e-1), // atan P coefficients
        _4(3.33329491539e-1), // atan P coefficients
    // cacheline
        _4(0.f), // padding (for alignment with double)
        _4(0.f), // padding (for alignment with double)
        _4(0.f), // padding (for alignment with double)
        _4(0.f), // padding (for alignment with double)
    // cacheline
        _4(0.f), // padding (for alignment with double)
        _4(0.f), // padding (for alignment with double)
        _4(2.414213562373095), // tan( 3/8 π )
        _4(0.414213562373095), // tan( 1/8 π ) lower threshold for special casing in atan
    // cacheline
        _4(Vc_buildFloat(-1, 0x3BBD2E, -25)), // remainder of pi/2
        _4(1.e-4), // small asin input threshold
        _4(0.f), // padding (for alignment with double)
        _4(0.f), // padding (for alignment with double)
        _4(4.2163199048e-2),
        _4(2.4181311049e-2),
        _4(4.5470025998e-2),
        _4(7.4953002686e-2),
        _4(1.6666752422e-1),
    };
#undef _4

    // cacheline 8
    V_ALIGN(16) extern const unsigned char _IndexesFromZero16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

    V_ALIGN(64) const unsigned long long c_log<double>::data[21 * 2] = {
      /* 0*/   0x000003ff000003ffull, 0x000003ff000003ffull // bias TODO: remove
      /* 1*/ , 0x7ff0000000000000ull, 0x7ff0000000000000ull // exponentMask (+inf)

      /* 2*/ , 0x3f1ab4c293c31bb0ull, 0x3f1ab4c293c31bb0ull // P[0]
      /* 3*/ , 0x3fdfd6f53f5652f2ull, 0x3fdfd6f53f5652f2ull // P[1]
      /* 4*/ , 0x4012d2baed926911ull, 0x4012d2baed926911ull // P[2]
      /* 5*/ , 0x402cff72c63eeb2eull, 0x402cff72c63eeb2eull // P[3]
      /* 6*/ , 0x4031efd6924bc84dull, 0x4031efd6924bc84dull // P[4]
      /* 7*/ , 0x401ed5637d7edcf8ull, 0x401ed5637d7edcf8ull // P[5]

      /* 8*/ , 0x40269320ae97ef8eull, 0x40269320ae97ef8eull // Q[0]
      /* 9*/ , 0x40469d2c4e19c033ull, 0x40469d2c4e19c033ull // Q[1]
      /*10*/ , 0x4054bf33a326bdbdull, 0x4054bf33a326bdbdull // Q[2]
      /*11*/ , 0x4051c9e2eb5eae21ull, 0x4051c9e2eb5eae21ull // Q[3]
      /*12*/ , 0x4037200a9e1f25b2ull, 0x4037200a9e1f25b2ull // Q[4]

      /*13*/ , 0xfff0000000000000ull, 0xfff0000000000000ull // -inf
      /*14*/ , 0x0010000000000000ull, 0x0010000000000000ull // min()
      /*15*/ , 0x3fe6a09e667f3bcdull, 0x3fe6a09e667f3bcdull // 1/sqrt(2)
      /*16*/ , 0x3fe6300000000000ull, 0x3fe6300000000000ull // round(ln(2) * 512) / 512
      /*17*/ , 0xbf2bd0105c610ca8ull, 0xbf2bd0105c610ca8ull // ln(2) - round(ln(2) * 512) / 512
      /*18*/ , 0x3fe0000000000000ull, 0x3fe0000000000000ull // 0.5
      /*19*/ , 0x3fdbcb7b1526e50eull, 0x3fdbcb7b1526e50eull // log10(e)
      /*20*/ , 0x3ff71547652b82feull, 0x3ff71547652b82feull // log2(e)
    };

    template<> V_ALIGN(64) const unsigned int c_log<float>::data[21 * 4] = {
        0x0000007fu, 0x0000007fu, 0x0000007fu, 0x0000007fu, // bias TODO: remove
        0x7f800000u, 0x7f800000u, 0x7f800000u, 0x7f800000u, // exponentMask (+inf)

        0x3d9021bbu, 0x3d9021bbu, 0x3d9021bbu, 0x3d9021bbu, //  7.0376836292e-2f // P[0]
        0xbdebd1b8u, 0xbdebd1b8u, 0xbdebd1b8u, 0xbdebd1b8u, // -1.1514610310e-1f // P[1]
        0x3def251au, 0x3def251au, 0x3def251au, 0x3def251au, //  1.1676998740e-1f // P[2]
        0xbdfe5d4fu, 0xbdfe5d4fu, 0xbdfe5d4fu, 0xbdfe5d4fu, // -1.2420140846e-1f // P[3]
        0x3e11e9bfu, 0x3e11e9bfu, 0x3e11e9bfu, 0x3e11e9bfu, //  1.4249322787e-1f // P[4]
        0xbe2aae50u, 0xbe2aae50u, 0xbe2aae50u, 0xbe2aae50u, // -1.6668057665e-1f // P[5]
        0x3e4cceacu, 0x3e4cceacu, 0x3e4cceacu, 0x3e4cceacu, //  2.0000714765e-1f // P[6]
        0xbe7ffffcu, 0xbe7ffffcu, 0xbe7ffffcu, 0xbe7ffffcu, // -2.4999993993e-1f // P[7]
        0x3eaaaaaau, 0x3eaaaaaau, 0x3eaaaaaau, 0x3eaaaaaau, //  3.3333331174e-1f // P[8]
        0,           0,           0,           0,           // padding because of c_log<double>
        0,           0,           0,           0,           // padding because of c_log<double>

        0xff800000u, 0xff800000u, 0xff800000u, 0xff800000u, // -inf
        0x00800000u, 0x00800000u, 0x00800000u, 0x00800000u, // min()
        0x3f3504f3u, 0x3f3504f3u, 0x3f3504f3u, 0x3f3504f3u, // 1/sqrt(2)
        // ln(2) = 0x3fe62e42fefa39ef
        // ln(2) = Vc_buildDouble( 1, 0x00062e42fefa39ef, -1)
        //       = Vc_buildFloat( 1, 0x00317217(f7d), -1) + Vc_buildFloat( 1, 0x0077d1cd, -25)
        //       = Vc_buildFloat( 1, 0x00318000(000), -1) + Vc_buildFloat(-1, 0x005e8083, -13)
        0x3f318000u, 0x3f318000u, 0x3f318000u, 0x3f318000u, // round(ln(2) * 512) / 512
        0xb95e8083u, 0xb95e8083u, 0xb95e8083u, 0xb95e8083u, // ln(2) - round(ln(2) * 512) / 512
        0x3f000000u, 0x3f000000u, 0x3f000000u, 0x3f000000u, // 0.5
        0x3ede5bd9u, 0x3ede5bd9u, 0x3ede5bd9u, 0x3ede5bd9u, // log10(e)
        0x3fb8aa3bu, 0x3fb8aa3bu, 0x3fb8aa3bu, 0x3fb8aa3bu, // log2(e)
        // log10(2) = 0x3fd34413509f79ff
        //          = Vc_buildDouble( 1, 0x00034413509f79ff, -2)
        //          = Vc_buildFloat( 1, 0x001a209a(84fbcff8), -2) + Vc_buildFloat( 1, 0x0004fbcff(8), -26)
        //Vc_buildFloat( 1, 0x001a209a, -2), // log10(2)
        //Vc_buildFloat( 1, 0x001a209a, -2), // log10(2)
        //Vc_buildFloat( 1, 0x001a209a, -2), // log10(2)
        //Vc_buildFloat( 1, 0x001a209a, -2), // log10(2)
    };
} // namespace SSE

V_ALIGN(64) unsigned int RandomState[16] = {
    0x5a383a4fu, 0xc68bd45eu, 0x691d6d86u, 0xb367e14fu,
    0xd689dbaau, 0xfde442aau, 0x3d265423u, 0x1a77885cu,
    0x36ed2684u, 0xfb1f049du, 0x19e52f31u, 0x821e4dd7u,
    0x23996d25u, 0x5962725au, 0x6aced4ceu, 0xd4c610f3u
};

// dummy symbol to emit warnings with GCC 4.3
namespace Warnings {
    void _operator_bracket_warning() {}
} // namespace Warnings

const char LIBRARY_VERSION[] = VC_VERSION_STRING;
const unsigned int LIBRARY_VERSION_NUMBER = VC_VERSION_NUMBER;
const unsigned int LIBRARY_ABI_VERSION = VC_LIBRARY_ABI_VERSION;

void checkLibraryAbi(unsigned int compileTimeAbi, unsigned int versionNumber, const char *compileTimeVersion) {
    if (LIBRARY_ABI_VERSION != compileTimeAbi || LIBRARY_VERSION_NUMBER < versionNumber) {
        printf("The versions of libVc.a (%s) and Vc/version.h (%s) are incompatible. Aborting.\n", LIBRARY_VERSION, compileTimeVersion);
        abort();
    }
}

} // namespace Vc

#undef V_ALIGN
