/*  This file is part of the Vc library. {{{

    Copyright (C) 2012-2013 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_COMMON_IIF_H
#define VC_COMMON_IIF_H

#include <Vc/type_traits>
#include "macros.h"

Vc_PUBLIC_NAMESPACE_BEGIN

namespace
{
    template<typename T> struct assert_for_iif
    {
        typedef T type;
        static_assert(Vc::is_simd_vector<T>::value, "Incorrect use of Vc::iif. If you use a mask as first parameter, the second and third parameters must be of vector type.");
    };
} // anonymous namespace

/**
 * \ingroup Utilities
 *
 * Function to mimic the ternary operator '?:' (inline-if).
 *
 * \param condition  Determines which values are returned. This is analog to the first argument to
 *                   the ternary operator.
 * \param trueValue  The values to return where \p condition is \c true.
 * \param falseValue The values to return where \p condition is \c false.
 * \return A combination of entries from \p trueValue and \p falseValue, according to \p condition.
 *
 * So instead of the scalar variant
 * \code
 * float x = a > 1.f ? b : b + c;
 * \endcode
 * you'd write
 * \code
 * float_v x = Vc::iif (a > 1.f, b, b + c);
 * \endcode
 *
 * Assuming \c a has the values [0, 3, 5, 1], \c b is [1, 1, 1, 1], and \c c is [1, 2, 3, 4], then x
 * will be [2, 2, 3, 5].
 */
template<typename Mask, typename T> Vc_ALWAYS_INLINE
typename std::enable_if<Vc::is_simd_mask<Mask>::value, typename assert_for_iif<T>::type>::type
#ifndef VC_MSVC
iif(Mask condition, T trueValue, T falseValue)
{
#else
iif(const Mask &condition, const T &trueValue, const T &_falseValue)
{
    T falseValue(_falseValue);
#endif
    Vc::where(condition) | falseValue = trueValue;
    return falseValue;
}

/* the following might be a nice shortcut in some cases, but:
 * 1. it fails if there are different vector classes for the same T
 * 2. the semantics are a bit fishy: basically the mask determines how to blow up the scalar values
template<typename Mask, typename T> Vc_ALWAYS_INLINE
typename std::enable_if<Vc::is_simd_mask<Mask>::value && !Vc::is_simd_vector<T>::value, void>::type
#ifndef VC_MSVC
iif(Mask condition, T trueValue, T falseValue)
#else
iif(const Mask &condition, T trueValue, T falseValue)
#endif
{
    Vc::Vector<T> f = falseValue;
    Vc::where(condition) | f = trueValue;
    return f;
}
 */

/**
 * \ingroup Utilities
 *
 * Overload of the above for boolean conditions.
 *
 * This typically results in direct use of the ternary operator. This function makes it easier to
 * switch from a Vc type to a builtin type.
 *
 * \param condition  Determines which value is returned. This is analog to the first argument to
 *                   the ternary operator.
 * \param trueValue  The value to return if \p condition is \c true.
 * \param falseValue The value to return if \p condition is \c false.
 * \return Either \p trueValue or \p falseValue, depending on \p condition.
 */
template<typename T> constexpr T iif (bool condition, const T &trueValue, const T &falseValue)
{
    return condition ? trueValue : falseValue;
}

Vc_NAMESPACE_END

Vc_NAMESPACE_BEGIN(Vc_IMPL_NAMESPACE)
    using Vc::iif;
Vc_NAMESPACE_END

#include "undomacros.h"

#endif // VC_COMMON_IIF_H
