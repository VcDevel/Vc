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

#ifndef VC_COMMON_WHERE_H
#define VC_COMMON_WHERE_H

#include "macros.h"

Vc_PUBLIC_NAMESPACE_BEGIN

namespace
{
    template<typename _Mask, typename _LValue> struct MaskedLValue
    {
        typedef _Mask Mask;
        typedef _LValue LValue;

        const Mask &mask;
        LValue &lhs;

        // the ctors must be present, otherwise GCC fails to warn for Vc_WARN_UNUSED_RESULT
        constexpr MaskedLValue(const Mask &m, LValue &l) : mask(m), lhs(l) {}
#ifdef VC_NO_MOVE_CTOR
        constexpr MaskedLValue(const MaskedLValue &) = default;
#else
        MaskedLValue(const MaskedLValue &) = delete;
        constexpr MaskedLValue(MaskedLValue &&) = default;
#endif

        /* It is intentional that the assignment operators return void: When a bool is used for the
         * mask the code might get skipped completely, thus nothing can be returned. This would be
         * like requiring an if statement to return a value.
         */
        template<typename T> Vc_ALWAYS_INLINE void operator  =(T &&rhs) { lhs(mask)   = std::forward<T>(rhs); }
        template<typename T> Vc_ALWAYS_INLINE void operator +=(T &&rhs) { lhs(mask)  += std::forward<T>(rhs); }
        template<typename T> Vc_ALWAYS_INLINE void operator -=(T &&rhs) { lhs(mask)  -= std::forward<T>(rhs); }
        template<typename T> Vc_ALWAYS_INLINE void operator *=(T &&rhs) { lhs(mask)  *= std::forward<T>(rhs); }
        template<typename T> Vc_ALWAYS_INLINE void operator /=(T &&rhs) { lhs(mask)  /= std::forward<T>(rhs); }
        template<typename T> Vc_ALWAYS_INLINE void operator %=(T &&rhs) { lhs(mask)  %= std::forward<T>(rhs); }
        template<typename T> Vc_ALWAYS_INLINE void operator ^=(T &&rhs) { lhs(mask)  ^= std::forward<T>(rhs); }
        template<typename T> Vc_ALWAYS_INLINE void operator &=(T &&rhs) { lhs(mask)  &= std::forward<T>(rhs); }
        template<typename T> Vc_ALWAYS_INLINE void operator |=(T &&rhs) { lhs(mask)  |= std::forward<T>(rhs); }
        template<typename T> Vc_ALWAYS_INLINE void operator<<=(T &&rhs) { lhs(mask) <<= std::forward<T>(rhs); }
        template<typename T> Vc_ALWAYS_INLINE void operator>>=(T &&rhs) { lhs(mask) >>= std::forward<T>(rhs); }
        Vc_ALWAYS_INLINE void operator++()    { ++lhs(mask); }
        Vc_ALWAYS_INLINE void operator++(int) { lhs(mask)++; }
        Vc_ALWAYS_INLINE void operator--()    { --lhs(mask); }
        Vc_ALWAYS_INLINE void operator--(int) { lhs(mask)--; }
    };

    template<typename _LValue> struct MaskedLValue<bool, _LValue>
    {
        typedef bool Mask;
        typedef _LValue LValue;

        const Mask &mask;
        LValue &lhs;

        // the ctors must be present, otherwise GCC fails to warn for Vc_WARN_UNUSED_RESULT
        constexpr MaskedLValue(const Mask &m, LValue &l) : mask(m), lhs(l) {}
#ifdef VC_NO_MOVE_CTOR
        constexpr MaskedLValue(const MaskedLValue &) = default;
#else
        MaskedLValue(const MaskedLValue &) = delete;
        constexpr MaskedLValue(MaskedLValue &&) = default;
#endif

        template<typename T> Vc_ALWAYS_INLINE void operator  =(T &&rhs) { if (mask) lhs   = std::forward<T>(rhs); }
        template<typename T> Vc_ALWAYS_INLINE void operator +=(T &&rhs) { if (mask) lhs  += std::forward<T>(rhs); }
        template<typename T> Vc_ALWAYS_INLINE void operator -=(T &&rhs) { if (mask) lhs  -= std::forward<T>(rhs); }
        template<typename T> Vc_ALWAYS_INLINE void operator *=(T &&rhs) { if (mask) lhs  *= std::forward<T>(rhs); }
        template<typename T> Vc_ALWAYS_INLINE void operator /=(T &&rhs) { if (mask) lhs  /= std::forward<T>(rhs); }
        template<typename T> Vc_ALWAYS_INLINE void operator %=(T &&rhs) { if (mask) lhs  %= std::forward<T>(rhs); }
        template<typename T> Vc_ALWAYS_INLINE void operator ^=(T &&rhs) { if (mask) lhs  ^= std::forward<T>(rhs); }
        template<typename T> Vc_ALWAYS_INLINE void operator &=(T &&rhs) { if (mask) lhs  &= std::forward<T>(rhs); }
        template<typename T> Vc_ALWAYS_INLINE void operator |=(T &&rhs) { if (mask) lhs  |= std::forward<T>(rhs); }
        template<typename T> Vc_ALWAYS_INLINE void operator<<=(T &&rhs) { if (mask) lhs <<= std::forward<T>(rhs); }
        template<typename T> Vc_ALWAYS_INLINE void operator>>=(T &&rhs) { if (mask) lhs >>= std::forward<T>(rhs); }
        Vc_ALWAYS_INLINE void operator++()    { if (mask) ++lhs; }
        Vc_ALWAYS_INLINE void operator++(int) { if (mask) lhs++; }
        Vc_ALWAYS_INLINE void operator--()    { if (mask) --lhs; }
        Vc_ALWAYS_INLINE void operator--(int) { if (mask) lhs--; }
    };

    template<typename _Mask> struct WhereMask
    {
        typedef _Mask Mask;
        const Mask &mask;

        // the ctors must be present, otherwise GCC fails to warn for Vc_WARN_UNUSED_RESULT
        constexpr WhereMask(const Mask &m) : mask(m) {}
        WhereMask(const WhereMask &) = delete;

        template<typename T> constexpr Vc_WARN_UNUSED_RESULT MaskedLValue<Mask, T> operator|(T &&lhs) const
        {
            static_assert(std::is_lvalue_reference<T>::value, "Syntax error: Incorrect use of Vc::where. Maybe operator precedence got you by surprise. Examples of correct usage:\n"
                    "  Vc::where(x < 2) | x += 1;\n"
                    "  (Vc::where(x < 2) | x)++;\n"
                    "  Vc::where(x < 2)(x) += 1;\n"
                    "  Vc::where(x < 2)(x)++;\n"
                    );
            return { mask, lhs };
        }

        template<typename T> constexpr Vc_WARN_UNUSED_RESULT MaskedLValue<Mask, T> operator()(T &&lhs) const
        {
            return operator|(std::forward<T>(lhs));
        }
    };
} // anonymous namespace

/**
 * \ingroup Utilities
 *
 * Conditional assignment.
 *
 * Since compares between SIMD vectors do not return a single boolean, but rather a vector of
 * booleans (mask), one often cannot use if / else statements. Instead, one needs to state
 * that only a subset of entries of a given SIMD vector should be modified. The \c where function
 * can be prepended to any assignment operation to execute a masked assignment.
 *
 * \param mask The mask that selects the entries in the target vector that will be modified.
 *
 * \return This function returns an opaque object that binds to the left operand of an assignment
 * via the binary-or operator or the functor operator. (i.e. either <code>where(mask) | x = y</code>
 * or <code>where(mask)(x) = y</code>)
 *
 * Example:
 * \code
 * template<typename T> void f1(T &x, T &y)
 * {
 *   if (x < 2) {
 *     x *= y;
 *     y += 2;
 *   }
 * }
 * template<typename T> void f2(T &x, T &y)
 * {
 *   where(x < 2) | x *= y;
 *   where(x < 2) | y += 2;
 * }
 * \endcode
 * The block following the if statement in \c f1 will be executed if <code>x &lt; 2</code> evaluates
 * to \c true. If \c T is a scalar type you normally get what you expect. But if \c T is a SIMD
 * vector type, the comparison will use the implicit conversion from a mask to bool, meaning
 * <code>all_of(x &lt; 2)</code>.
 *
 * Most of the time the required operation is a masked assignment as stated in \c f2.
 *
 */
template<typename M> constexpr Vc_WARN_UNUSED_RESULT WhereMask<M> where(const M &mask)
{
    return { mask };
}

template<typename M> constexpr Vc_WARN_UNUSED_RESULT WhereMask<M> _if(const M &m)
{
    return { m };
}

Vc_NAMESPACE_END

Vc_NAMESPACE_BEGIN(Vc_IMPL_NAMESPACE)
    using Vc::where;
Vc_NAMESPACE_END

#include "undomacros.h"

#endif // VC_COMMON_WHERE_H
