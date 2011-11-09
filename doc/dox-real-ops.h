/*  This file is part of the Vc library.

    Copyright (C) 2011 Matthias Kretz <kretz@kde.org>

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


/**
 * Copies the sign of \p reference.
 *
 * Returns a value where the sign of the value equals the sign of \p reference. I.e.
 * sign(v.copySign(r)) == sign(r).
 */
inline VECTOR_TYPE copySign(VECTOR_TYPE reference) const;

/**
 * Returns the exponent to base 2.
 *
 * This function provides efficient access to the exponent of the floating point number. The
 * returned value is a fast approximation to the logarithm of base 2. The absolute error of that
 * approximation is between [0, 1[.
 *
 * Examples:
\verbatim
 value | exponent | log2
=======|==========|=======
   1.0 |        0 | 0
   2.0 |        1 | 1
   3.0 |        1 | 1.585
   3.9 |        1 | 1.963
   4.0 |        2 | 2
   4.1 |        2 | 2.036
\endverbatim
 *
 * \warning This function assumes a positive value (non-zero). If the value is negative the sign bit will
 * modify the returned value. If you compile with Vc runtime checks the function will assert
 * values greater than zero.
 *
 * You may use abs to apply this function to negative values:
 * \code
 * abs(v).exponent()
 * \endcode
 */
inline VECTOR_TYPE exponent() const;
