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

#include "macros.h"

/*OUTER_NAMESPACE_BEGIN*/
namespace Vc
{
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
#ifndef VC_MSVC
template<typename T> Vc_ALWAYS_INLINE Vector<T> iif (typename Vector<T>::Mask condition, Vector<T> trueValue, Vector<T> falseValue)
{
#else
template<typename T> Vc_ALWAYS_INLINE Vector<T> iif (const typename Vector<T>::Mask &condition, const Vector<T> &trueValue, const Vector<T> &_falseValue)
{
    Vector<T> falseValue(_falseValue);
#endif
    falseValue(condition) = trueValue;
    return falseValue;
}

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

} // namespace Vc
/*OUTER_NAMESPACE_END*/

#include "undomacros.h"

#endif // VC_COMMON_IIF_H
