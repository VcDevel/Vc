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


/**
 * \ingroup Math
 *
 * Returns the square root of \p v.
 */
VECTOR_TYPE sqrt(const VECTOR_TYPE &v);

/**
 * \ingroup Math
 *
 * Returns the reciprocal square root of \p v.
 */
VECTOR_TYPE rsqrt(const VECTOR_TYPE &v);

/**
 * \ingroup Math
 *
 * Returns the reciprocal of \p v.
 */
VECTOR_TYPE reciprocal(const VECTOR_TYPE &v);

/**
 * \ingroup Math
 *
 * Returns the absolute value of \p v.
 */
VECTOR_TYPE abs(const VECTOR_TYPE &v);

/**
 * \ingroup Math
 *
 * Returns the closest integer to \p v. 0.5 is rounded to even.
 */
VECTOR_TYPE round(const VECTOR_TYPE &v);

/**
 * \ingroup Math
 *
 * Returns the natural logarithm of \p v.
 */
VECTOR_TYPE log(const VECTOR_TYPE &v);

/**
 * \ingroup Math
 *
 * Returns the sine of \p v.
 */
VECTOR_TYPE sin(const VECTOR_TYPE &v);

/**
 * \ingroup Math
 *
 * Returns the cosine of \p v.
 */
VECTOR_TYPE cos(const VECTOR_TYPE &v);

/**
 * \ingroup Math
 *
 * Returns the arcsine of \p v.
 */
VECTOR_TYPE asin(const VECTOR_TYPE &v);

/**
 * \ingroup Math
 *
 * Returns the arctangent of \p v.
 */
VECTOR_TYPE atan(const VECTOR_TYPE &v);

/**
 * \ingroup Math
 *
 * Returns the arctangent of \p x / \p y.
 */
VECTOR_TYPE atan2(const VECTOR_TYPE &x, const VECTOR_TYPE &y);

/**
 * \ingroup Math
 *
 * Returns the minimum of \p x and \p y.
 */
VECTOR_TYPE min(const VECTOR_TYPE &x, const VECTOR_TYPE &y);

/**
 * \ingroup Math
 *
 * Returns the maximum of \p x and \p y.
 */
VECTOR_TYPE max(const VECTOR_TYPE &x, const VECTOR_TYPE &y);

/**
 * \ingroup Math
 *
 * Returns whether the values in the vector are finite (i.e. not NaN or +/-inf).
 */
MASK_TYPE isfinite(const VECTOR_TYPE &x);

/**
 * \ingroup Math
 *
 * Returns whether the values in the vector are NaN.
 */
MASK_TYPE isnan(const VECTOR_TYPE &x);
