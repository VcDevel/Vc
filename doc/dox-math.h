/*  This file is part of the Vc library. {{{
Copyright © 2009-2016 Matthias Kretz <kretz@kde.org>

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
 * \param v The values to apply the logarithm on.
 * \returns the natural logarithm of \p v.
 *
 * \note The single-precision implementation has an error of max. 1 ulp (mean 0.020 ulp) in the range ]0, 1000] (including denormals).
 * \note The double-precision implementation has an error of max. 1 ulp (mean 0.020 ulp) in the range ]0, 1000] (including denormals).
 */
VECTOR_TYPE log(const VECTOR_TYPE &v);

/**
 * \ingroup Math
 *
 * \param v The values to apply the logarithm on.
 * \returns the base-2 logarithm of \p v.
 *
 * \note The single-precision implementation has an error of max. 1 ulp (mean 0.016 ulp) in the range ]0, 1000] (including denormals).
 * \note The double-precision implementation has an error of max. 1 ulp (mean 0.016 ulp) in the range ]0, 1000] (including denormals).
 */
VECTOR_TYPE log2(const VECTOR_TYPE &v);

/**
 * \ingroup Math
 *
 * \param v The values to apply the logarithm on.
 * \returns the base-10 logarithm of \p v.
 *
 * \note The single-precision implementation has an error of max. 2 ulp (mean 0.31 ulp) in the range ]0, 1000] (including denormals).
 * \note The double-precision implementation has an error of max. 2 ulp (mean 0.26 ulp) in the range ]0, 1000] (including denormals).
 */
VECTOR_TYPE log10(const VECTOR_TYPE &v);

/**
 * \ingroup Math
 *
 * \param v The values to apply the exponential function on.
 * \returns the exponential of \p v.
 */
VECTOR_TYPE exp(const VECTOR_TYPE &v);

/**
 * \ingroup Math
 *
 * \param v The values to apply the sine function on.
 * \returns the sine of \p v.
 *
 * \note The single-precision implementation has an error of max. 2 ulp (mean 0.17 ulp) in the range [-8192, 8192].
 * \note The double-precision implementation has an error of max. 8e6 ulp (mean 1040 ulp) in the range [-8192, 8192].
 * \note Vc versions before 0.7 had much larger errors.
 */
VECTOR_TYPE sin(const VECTOR_TYPE &v);

/**
 * \ingroup Math
 *
 * \param v The values to apply the cosine function on.
 * \returns the cosine of \p v.
 *
 * \note The single-precision implementation has an error of max. 2 ulp (mean 0.18 ulp) in the range [-8192, 8192].
 * \note The double-precision implementation has an error of max. 8e6 ulp (mean 1160 ulp) in the range [-8192, 8192].
 * \note Vc versions before 0.7 had much larger errors.
 */
VECTOR_TYPE cos(const VECTOR_TYPE &v);

/**
 * \ingroup Math
 *
 * \param v The values to apply the arcsine function on.
 * \returns the arcsine of \p v.
 *
 * \note The single-precision implementation has an error of max. 2 ulp (mean 0.3 ulp).
 * \note The double-precision implementation has an error of max. 36 ulp (mean 0.4 ulp).
 */
VECTOR_TYPE asin(const VECTOR_TYPE &v);

/**
 * \ingroup Math
 *
 * \param v The values to apply the arctangent function on.
 * \returns the arctangent of \p v.
 * \note The single-precision implementation has an error of max. 3 ulp (mean 0.4 ulp) in the range [-8192, 8192].
 * \note The double-precision implementation has an error of max. 2 ulp (mean 0.1 ulp) in the range [-8192, 8192].
 */
VECTOR_TYPE atan(const VECTOR_TYPE &v);

/**
 * \ingroup Math
 *
 * Calculates the angle given the lengths of the opposite and adjacent legs in a right
 * triangle.
 * \param y The opposite leg.
 * \param x The adjacent leg.
 * \returns the arctangent of \p y / \p x.
 */
VECTOR_TYPE atan2(const VECTOR_TYPE &y, const VECTOR_TYPE &x);

/**
 * \ingroup Math
 *
 * \param x \VSize{T} values to compare component-wise against \p y.
 * \param y \VSize{T} values to compare component-wise against \p x.
 * \returns the minimum of \p x and \p y.
 */
VECTOR_TYPE min(const VECTOR_TYPE &x, const VECTOR_TYPE &y);

/**
 * \ingroup Math
 *
 * \param x \VSize{T} values to compare component-wise against \p y.
 * \param y \VSize{T} values to compare component-wise against \p x.
 * \returns the maximum of \p x and \p y.
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
 * \param x The \VSize{T} values to check for finite values.
 * \returns a mask that tells whether the values in the vector are finite (i.e.\ not NaN or +/-inf).
 */
MASK_TYPE isfinite(const VECTOR_TYPE &x);

/**
 * \ingroup Math
 *
 * \param x The \VSize{T} values to check for NaN values.
 * \returns a mask that tells whether the values in the vector are NaN.
 */
MASK_TYPE isnan(const VECTOR_TYPE &x);

/**
 * \ingroup Math
 *
 * Multiplies \p a with \p b and then adds \p c, without rounding between the
 * multiplication and the addition.
 *
 * \param a First multiplication factor.
 * \param b Second multiplication factor.
 * \param c Summand that will be added after multiplication.
 * \returns The \VSize{T} values of `a * b + c` with higher precision due to no rounding
 *          between multiplication and addition.
 *
 * \note This operation may have explicit hardware support, in which case it is normally
 *       faster to use the FMA instead of separate multiply and add instructions.
 * \note If the target hardware does not have FMA support this function will be
 *       considerably slower than a normal a * b + c. This is due to the increased
 *       precision fusedMultiplyAdd provides.
 * \note The compiler normally detects opportunities for using the hardware FMA
 *       instructions from normal multiplication and addition/subtraction operators. Use
 *       this function only if you \em require the additional precision.
 */
VECTOR_TYPE fma(VECTOR_TYPE a, VECTOR_TYPE b, VECTOR_TYPE c);
