/*  This file is part of the Vc library. {{{

    Copyright (C) 2013 Matthias Kretz <kretz@kde.org>

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

#include "macros.h"

#ifdef VC_IMPL_AVX
#  ifdef VC_IMPL_AVX2
Vc_NAMESPACE_BEGIN(AVX2)
#  else
Vc_NAMESPACE_BEGIN(AVX)
#  endif
#include "logarithm.h"
#include "exponential.h"
    inline double_v exp(double_v::AsArg _x) {
        Vector<double> x = _x;
        typedef Vector<double> V;
        typedef V::Mask M;
        typedef Const<double> C;

        const M overflow  = x > Vc_buildDouble( 1, 0x0006232bdd7abcd2ull, 9); // max log
        const M underflow = x < Vc_buildDouble(-1, 0x0006232bdd7abcd2ull, 9); // min log

        V px = floor(C::log2_e() * x + 0.5);
        __m128i tmp = _mm256_cvttpd_epi32(px.data());
        Vector<int> n = Vc_IMPL_NAMESPACE::concat(_mm_unpacklo_epi32(tmp, tmp), _mm_unpackhi_epi32(tmp, tmp));
        x -= px * C::ln2_large(); //Vc_buildDouble(1, 0x00062e4000000000ull, -1);  // ln2
        x -= px * C::ln2_small(); //Vc_buildDouble(1, 0x0007f7d1cf79abcaull, -20); // ln2

        const double P[] = {
            Vc_buildDouble(1, 0x000089cdd5e44be8ull, -13),
            Vc_buildDouble(1, 0x000f06d10cca2c7eull,  -6),
            Vc_buildDouble(1, 0x0000000000000000ull,   0)
        };
        const double Q[] = {
            Vc_buildDouble(1, 0x00092eb6bc365fa0ull, -19),
            Vc_buildDouble(1, 0x0004ae39b508b6c0ull,  -9),
            Vc_buildDouble(1, 0x000d17099887e074ull,  -3),
            Vc_buildDouble(1, 0x0000000000000000ull,   1)
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
Vc_NAMESPACE_END
#endif

#ifdef VC_IMPL_SSE
Vc_NAMESPACE_BEGIN(SSE)
#include "logarithm.h"
#include "exponential.h"
    inline SSE::double_v exp(SSE::double_v::AsArg _x) {
        SSE::Vector<double> x = _x;
        typedef SSE::Vector<double> V;
        typedef V::Mask M;
        typedef SSE::Const<double> C;

        const M overflow  = x > Vc_buildDouble( 1, 0x0006232bdd7abcd2ull, 9); // max log
        const M underflow = x < Vc_buildDouble(-1, 0x0006232bdd7abcd2ull, 9); // min log

        V px = floor(C::log2_e() * x + 0.5);
        SSE::Vector<int> n(px);
        n.data() = Mem::permute<X0, X2, X1, X3>(n.data());
        x -= px * C::ln2_large(); //Vc_buildDouble(1, 0x00062e4000000000ull, -1);  // ln2
        x -= px * C::ln2_small(); //Vc_buildDouble(1, 0x0007f7d1cf79abcaull, -20); // ln2

        const double P[] = {
            Vc_buildDouble(1, 0x000089cdd5e44be8ull, -13),
            Vc_buildDouble(1, 0x000f06d10cca2c7eull,  -6),
            Vc_buildDouble(1, 0x0000000000000000ull,   0)
        };
        const double Q[] = {
            Vc_buildDouble(1, 0x00092eb6bc365fa0ull, -19),
            Vc_buildDouble(1, 0x0004ae39b508b6c0ull,  -9),
            Vc_buildDouble(1, 0x000d17099887e074ull,  -3),
            Vc_buildDouble(1, 0x0000000000000000ull,   1)
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
Vc_NAMESPACE_END
#endif
#include "undomacros.h"

#undef VC_COMMON_MATH_H_INTERNAL

#endif // VC_COMMON_MATH_H
