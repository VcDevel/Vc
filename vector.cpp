/*  This file is part of the Vc library.

    Copyright (C) 2009 Matthias Kretz <kretz@kde.org>

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

#include <cmath> // for M_PI
#include "sse/const.h"

namespace Vc
{
namespace SSE
{
    // cacheline 1
    V_ALIGN(16) const int c_general::absMaskFloat[4] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
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
    V_ALIGN(16) const long long c_general::absMaskDouble[2] = { 0x7fffffffffffffff, 0x7fffffffffffffff };
    V_ALIGN(16) const unsigned long long c_general::signMaskDouble[2] = { 0x8000000000000000, 0x8000000000000000 };
    V_ALIGN(16) const int _padding00[4] = { 0, 0, 0, 0 };

    template<> const float c_sin<float>::_data[4 * 8] = {
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

    template<> const double c_sin<double>::_data[2 * 8] = {
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
} // namespace SSE

namespace LRBni
{
    // cacheline 8
    V_ALIGN(16) extern const char _IndexesFromZero[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
} // namespace Larrabee
} // namespace Vc

#undef V_ALIGN
