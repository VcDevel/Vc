/*  This file is part of the Vc library. {{{

    Copyright (C) 2013-2014 Matthias Kretz <kretz@kde.org>

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

}}}*/

#ifndef VC_COMMON_MATH_H
#define VC_COMMON_MATH_H

#define VC_COMMON_MATH_H_INTERNAL 1

#include "trigonometric.h"

#include "const.h"
#include "macros.h"

#ifdef VC_IMPL_AVX
#  ifdef VC_IMPL_AVX2
namespace Vc_VERSIONED_NAMESPACE
{
namespace AVX2
{
#  else
namespace Vc_VERSIONED_NAMESPACE
{
namespace AVX
{
#  endif
#include "logarithm.h"
#include "exponential.h"
    inline double_v exp(double_v::AsArg _x) {
        Vector<double> x = _x;
        typedef Vector<double> V;
        typedef V::Mask M;
        typedef Const<double> C;

        const M overflow  = x > Vc::Internal::doubleConstant< 1, 0x0006232bdd7abcd2ull, 9>(); // max log
        const M underflow = x < Vc::Internal::doubleConstant<-1, 0x0006232bdd7abcd2ull, 9>(); // min log

        V px = floor(C::log2_e() * x + 0.5);
        __m128i tmp = _mm256_cvttpd_epi32(px.data());
        const simdarray<int, double_v::Size> n = SSE::int_v{tmp};
        x -= px * C::ln2_large(); //Vc::Internal::doubleConstant<1, 0x00062e4000000000ull, -1>();  // ln2
        x -= px * C::ln2_small(); //Vc::Internal::doubleConstant<1, 0x0007f7d1cf79abcaull, -20>(); // ln2

        const double P[] = {
            Vc::Internal::doubleConstant<1, 0x000089cdd5e44be8ull, -13>(),
            Vc::Internal::doubleConstant<1, 0x000f06d10cca2c7eull,  -6>(),
            Vc::Internal::doubleConstant<1, 0x0000000000000000ull,   0>()
        };
        const double Q[] = {
            Vc::Internal::doubleConstant<1, 0x00092eb6bc365fa0ull, -19>(),
            Vc::Internal::doubleConstant<1, 0x0004ae39b508b6c0ull,  -9>(),
            Vc::Internal::doubleConstant<1, 0x000d17099887e074ull,  -3>(),
            Vc::Internal::doubleConstant<1, 0x0000000000000000ull,   1>()
        };
        const V x2 = x * x;
        px = x * ((P[0] * x2 + P[1]) * x2 + P[2]);
        x =  px / ((((Q[0] * x2 + Q[1]) * x2 + Q[2]) * x2 + Q[3]) - px);
        x = V::One() + 2.0 * x;

        x = ldexp(x, n); // == x * 2ⁿ

        x(overflow) = std::numeric_limits<double>::infinity();
        x.setZero(underflow);

        return x;
    }
}  // namespace AVX(2)
}  // namespace Vc
#endif

#ifdef VC_IMPL_SSE
namespace Vc_VERSIONED_NAMESPACE
{
namespace SSE
{
#include "logarithm.h"
#include "exponential.h"
    inline SSE::double_v exp(SSE::double_v::AsArg _x) {
        SSE::Vector<double> x = _x;
        typedef SSE::Vector<double> V;
        typedef V::Mask M;
        typedef SSE::Const<double> C;

        const M overflow  = x > Vc::Internal::doubleConstant< 1, 0x0006232bdd7abcd2ull, 9>(); // max log
        const M underflow = x < Vc::Internal::doubleConstant<-1, 0x0006232bdd7abcd2ull, 9>(); // min log

        V px = floor(C::log2_e() * x + 0.5);
        simdarray<int, double_v::Size> n;
        _mm_storel_epi64(reinterpret_cast<__m128i *>(&n), _mm_cvttpd_epi32(px.data()));
        x -= px * C::ln2_large(); //Vc::Internal::doubleConstant<1, 0x00062e4000000000ull, -1>();  // ln2
        x -= px * C::ln2_small(); //Vc::Internal::doubleConstant<1, 0x0007f7d1cf79abcaull, -20>(); // ln2

        const double P[] = {
            Vc::Internal::doubleConstant<1, 0x000089cdd5e44be8ull, -13>(),
            Vc::Internal::doubleConstant<1, 0x000f06d10cca2c7eull,  -6>(),
            Vc::Internal::doubleConstant<1, 0x0000000000000000ull,   0>()
        };
        const double Q[] = {
            Vc::Internal::doubleConstant<1, 0x00092eb6bc365fa0ull, -19>(),
            Vc::Internal::doubleConstant<1, 0x0004ae39b508b6c0ull,  -9>(),
            Vc::Internal::doubleConstant<1, 0x000d17099887e074ull,  -3>(),
            Vc::Internal::doubleConstant<1, 0x0000000000000000ull,   1>()
        };
        const V x2 = x * x;
        px = x * ((P[0] * x2 + P[1]) * x2 + P[2]);
        x =  px / ((((Q[0] * x2 + Q[1]) * x2 + Q[2]) * x2 + Q[3]) - px);
        x = V::One() + 2.0 * x;

        x = ldexp(x, n); // == x * 2ⁿ

        x(overflow) = std::numeric_limits<double>::infinity();
        x.setZero(underflow);

        return x;
    }
}  // namespace SSE
}  // namespace Vc
#endif
#include "undomacros.h"

#undef VC_COMMON_MATH_H_INTERNAL

#endif // VC_COMMON_MATH_H
