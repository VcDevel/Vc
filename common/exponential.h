/*  This file is part of the Vc library. {{{

    Copyright (C) 2012-2014 Matthias Kretz <kretz@kde.org>

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

    -------------------------------------------------------------------

    The exp implementation is derived from Cephes, which carries the
    following Copyright notice:

    Cephes Math Library Release 2.2:  June, 1992
    Copyright 1984, 1987, 1989 by Stephen L. Moshier
    Direct inquiries to 30 Frost Street, Cambridge, MA 02140

}}}*/

#ifdef VC_COMMON_MATH_H_INTERNAL

constexpr float log2_e = 1.44269504088896341f;
constexpr float MAXLOGF = 88.72283905206835f;
constexpr float MINLOGF = -103.278929903431851103f; /* log(2^-149) */
constexpr float MAXNUMF = 3.4028234663852885981170418348451692544e38f;

    template<typename T> inline Vector<T> exp(VC_ALIGNED_PARAMETER(Vector<T>) _x) {
        typedef Vector<T> V;
        typedef typename V::Mask M;
        typedef Vc_IMPL_NAMESPACE::Const<T> C;

        V x(_x);

        const M overflow  = x > MAXLOGF;
        const M underflow = x < MINLOGF;

        // log₂(eˣ) = x * log₂(e) * log₂(2)
        //          = log₂(2^(x * log₂(e)))
        // => eˣ = 2^(x * log₂(e))
        // => n  = ⌊x * log₂(e) + ½⌋
        // => y  = x - n * ln(2)       | recall that: ln(2) * log₂(e) == 1
        // <=> eˣ = 2ⁿ * eʸ
        V z = floor(C::log2_e() * x + 0.5f);
        const auto n = static_cast<Vc::simdarray<int, V::Size>>(z);
        x -= z * C::ln2_large();
        x -= z * C::ln2_small();

        /* Theoretical peak relative error in [-0.5, +0.5] is 4.2e-9. */
        z = ((((( 1.9875691500E-4f  * x
                + 1.3981999507E-3f) * x
                + 8.3334519073E-3f) * x
                + 4.1665795894E-2f) * x
                + 1.6666665459E-1f) * x
                + 5.0000001201E-1f) * (x * x)
                + x
                + 1.0f;

        x = ldexp(z, n); // == z * 2ⁿ

        x(overflow) = std::numeric_limits<typename V::EntryType>::infinity();
        x.setZero(underflow);

        return x;
    }

#endif // VC_COMMON_MATH_H_INTERNAL
