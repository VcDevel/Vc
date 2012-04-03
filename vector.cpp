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

#ifndef M_PI
# define M_PI 3.14159265358979323846
#endif

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
        0.5 / M_PI, // 1 over 2pi
        M_PI * 2.,  // 2pi
        M_PI * 0.5, // pi over 2
        M_PI,       // pi
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
    const              float c_general::oneFloat = 1.f;
    const unsigned     short c_general::minShort[2] = { 0x8000u, 0x8000u };
    const unsigned     short c_general::one16[2] = { 1, 1 };
    const              float c_general::_2power31 = 1u << 31;

    // cacheline 4
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
    V_ALIGN(16) const short c_general::minShort[8] = { -0x8000, -0x8000, -0x8000, -0x8000, -0x8000, -0x8000, -0x8000, -0x8000 };
    V_ALIGN(16) extern const unsigned short _IndexesFromZero8[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };

    // cacheline 2
    V_ALIGN(16) extern const unsigned int   _IndexesFromZero4[4] = { 0, 1, 2, 3 };
    V_ALIGN(16) const unsigned short c_general::one16[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };
    V_ALIGN(16) const unsigned int c_general::one32[4] = { 1, 1, 1, 1 };
    V_ALIGN(16) const float c_general::oneFloat[4] = { 1.f, 1.f, 1.f, 1.f };

    // cacheline 3
    V_ALIGN(16) const double c_general::oneDouble[2] = { 1., 1. };
    V_ALIGN(16) const long long c_general::absMaskDouble[2] = { 0x7fffffffffffffffll, 0x7fffffffffffffffll };
    V_ALIGN(16) const unsigned long long c_general::signMaskDouble[2] = { 0x8000000000000000ull, 0x8000000000000000ull };
    V_ALIGN(16) const unsigned long long c_general::frexpMask[2] = { 0xbfefffffffffffffull, 0xbfefffffffffffffull };

    template<> const float c_sin<float>::data[4 * 8] = {
    // cacheline 4
        // 1 over 2pi
        1.59154936671257019e-01f, 1.59154936671257019e-01f, 1.59154936671257019e-01f, 1.59154936671257019e-01f,
        // 2pi
        6.28318548202514648f, 6.28318548202514648f, 6.28318548202514648f, 6.28318548202514648f,
        // pi over 2
        1.57079637050628662f, 1.57079637050628662f, 1.57079637050628662f, 1.57079637050628662f,
        // pi
        3.14159274101257324f, 3.14159274101257324f, 3.14159274101257324f, 3.14159274101257324f,

    // cacheline 5
        // 1 over 3!
        1.66666671633720398e-01f, 1.66666671633720398e-01f, 1.66666671633720398e-01f, 1.66666671633720398e-01f,
        // 1 over 5!
        8.33333376795053482e-03f, 8.33333376795053482e-03f, 8.33333376795053482e-03f, 8.33333376795053482e-03f,
        // 1 over 7!
        1.98412701138295233e-04f, 1.98412701138295233e-04f, 1.98412701138295233e-04f, 1.98412701138295233e-04f,
        // 1 over 9!
        2.75573188446287531e-06f, 2.75573188446287531e-06f, 2.75573188446287531e-06f, 2.75573188446287531e-06f
    };

    template<> const double c_sin<double>::data[2 * 8] = {
    // cacheline 6
        // 1 over 2pi
        0.5 / M_PI, 0.5 / M_PI,
        // 2pi
        M_PI * 2., M_PI * 2.,
        // pi over 2
        M_PI * 0.5, M_PI * 0.5,
        // pi
        M_PI, M_PI,

    // cacheline 7
        // 1 over 3!
        1.666666666666666574148081281236954964697360992431640625e-01, 1.666666666666666574148081281236954964697360992431640625e-01,
        // 1 over 5!
        8.33333333333333321768510160154619370587170124053955078125e-03, 8.33333333333333321768510160154619370587170124053955078125e-03,
        // 1 over 7!
        1.984126984126984125263171154784913596813566982746124267578125e-04, 1.984126984126984125263171154784913596813566982746124267578125e-04,
        // 1 over 9!
        2.755731922398589251095059327045788677423843182623386383056640625e-06, 2.755731922398589251095059327045788677423843182623386383056640625e-06
    };

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
