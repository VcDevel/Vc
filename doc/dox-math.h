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
 * Returns the closest integer to \p v; 0.5 is rounded to even.
 */
VECTOR_TYPE round(const VECTOR_TYPE &v);

/**
 * \ingroup Math
 *
 * Returns the natural logarithm of \p v.
 *
 * \note The single-precision implementation has an error of max. 1 ulp (mean 0.020 ulp) in the range ]0, 1000] (including denormals).
 * \note The double-precision implementation has an error of max. 1 ulp (mean 0.020 ulp) in the range ]0, 1000] (including denormals).
 */
VECTOR_TYPE log(const VECTOR_TYPE &v);

/**
 * \ingroup Math
 *
 * Returns the base-2 logarithm of \p v.
 *
 * \note The single-precision implementation has an error of max. 1 ulp (mean 0.016 ulp) in the range ]0, 1000] (including denormals).
 * \note The double-precision implementation has an error of max. 1 ulp (mean 0.016 ulp) in the range ]0, 1000] (including denormals).
 */
VECTOR_TYPE log2(const VECTOR_TYPE &v);

/**
 * \ingroup Math
 *
 * Returns the base-10 logarithm of \p v.
 *
 * \note The single-precision implementation has an error of max. 2 ulp (mean 0.31 ulp) in the range ]0, 1000] (including denormals).
 * \note The double-precision implementation has an error of max. 2 ulp (mean 0.26 ulp) in the range ]0, 1000] (including denormals).
 */
VECTOR_TYPE log10(const VECTOR_TYPE &v);

/**
 * \ingroup Math
 *
 * Returns the exponential of \p v.
 */
VECTOR_TYPE exp(const VECTOR_TYPE &v);

/**
 * \ingroup Math
 *
 * Returns the sine of \p v.
 *
 * \note The single-precision implementation has an error of max. 2 ulp (mean 0.17 ulp) in the range [-8192, 8192].
 * \note The double-precision implementation has an error of max. 8e6 ulp (mean 1040 ulp) in the range [-8192, 8192].
 * \note Vc versions before 0.7 had much larger errors.
 */
VECTOR_TYPE sin(const VECTOR_TYPE &v);

/**
 * \ingroup Math
 *
 * Returns the cosine of \p v.
 *
 * \note The single-precision implementation has an error of max. 2 ulp (mean 0.18 ulp) in the range [-8192, 8192].
 * \note The double-precision implementation has an error of max. 8e6 ulp (mean 1160 ulp) in the range [-8192, 8192].
 * \note Vc versions before 0.7 had much larger errors.
 */
VECTOR_TYPE cos(const VECTOR_TYPE &v);

/**
 * \ingroup Math
 *
 * Returns the arcsine of \p v.
 *
 * \note The single-precision implementation has an error of max. 2 ulp (mean 0.3 ulp).
 * \note The double-precision implementation has an error of max. 36 ulp (mean 0.4 ulp).
 */
VECTOR_TYPE asin(const VECTOR_TYPE &v);

/**
 * \ingroup Math
 *
 * Returns the arctangent of \p v.
 * \note The single-precision implementation has an error of max. 3 ulp (mean 0.4 ulp) in the range [-8192, 8192].
 * \note The double-precision implementation has an error of max. 2 ulp (mean 0.1 ulp) in the range [-8192, 8192].
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
 * Convert floating-point number to fractional and integral components.
 *
 * \param x value to be split into normalized fraction and exponent
 * \param e the exponent to base 2 of \p x
 *
 * \returns the normalized fraction. If \p x is non-zero, the return value is \p x times a power of two, and
 * its absolute value is always in the range [0.5,1).
 *
 * \returns
 * If \p x is zero, then the normalized fraction is zero and zero is stored in \p e.
 *
 * \returns
 * If \p x is a NaN, a NaN is returned, and the value of \p *e is unspecified.
 *
 * \returns
 * If \p x is positive infinity (negative infinity), positive infinity (nega‐
 * tive infinity) is returned, and the value of \p *e is unspecified.
 */
VECTOR_TYPE frexp(const VECTOR_TYPE &x, EXPONENT_TYPE *e);

/**
 * \ingroup Math
 *
 * Multiply floating-point number by integral power of 2
 *
 * \param x value to be multiplied by 2 ^ \p e
 * \param e exponent
 *
 * \returns \p x * 2 ^ \p e
 */
VECTOR_TYPE ldexp(VECTOR_TYPE x, EXPONENT_TYPE e);

/**
 * \ingroup Math
 *
 * Returns a mask that tells whether the values in the vector are finite (i.e.\ not NaN or +/-inf).
 */
MASK_TYPE isfinite(const VECTOR_TYPE &x);

/**
 * \ingroup Math
 *
 * Returns a mask that tells whether the values in the vector are NaN.
 */
MASK_TYPE isnan(const VECTOR_TYPE &x);
