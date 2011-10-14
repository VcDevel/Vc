/*  This file is part of the Vc library.

    Copyright (C) 2010 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_COMMON_ALIASINGENTRYHELPER_H
#define VC_COMMON_ALIASINGENTRYHELPER_H

#include "macros.h"

namespace Vc
{
namespace Common
{

template<typename T> class AliasingEntryHelper
{
    private:
        typedef T A MAY_ALIAS;
        A &m_data;

    public:
        template<typename T2>
        AliasingEntryHelper(T2 &d) : m_data(reinterpret_cast<A &>(d)) {}

        AliasingEntryHelper(A &d) : m_data(d) {}

        operator const T() const { return m_data; }

        bool operator==(T x) const { return static_cast<T>(m_data) == x; }
        bool operator!=(T x) const { return static_cast<T>(m_data) != x; }
        bool operator<=(T x) const { return static_cast<T>(m_data) <= x; }
        bool operator>=(T x) const { return static_cast<T>(m_data) >= x; }
        bool operator< (T x) const { return static_cast<T>(m_data) <  x; }
        bool operator> (T x) const { return static_cast<T>(m_data) >  x; }

        T operator-() const { return -static_cast<T>(m_data); }
        T operator~() const { return ~static_cast<T>(m_data); }
        T operator+(T x) const { return static_cast<T>(m_data) + x; }
        T operator-(T x) const { return static_cast<T>(m_data) - x; }
        T operator/(T x) const { return static_cast<T>(m_data) / x; }
        T operator*(T x) const { return static_cast<T>(m_data) * x; }
        T operator|(T x) const { return static_cast<T>(m_data) | x; }
        T operator&(T x) const { return static_cast<T>(m_data) & x; }
        T operator^(T x) const { return static_cast<T>(m_data) ^ x; }
        T operator%(T x) const { return static_cast<T>(m_data) % x; }

        AliasingEntryHelper &operator =(T x) { m_data  = x; return *this; }
        AliasingEntryHelper &operator+=(T x) { m_data += x; return *this; }
        AliasingEntryHelper &operator-=(T x) { m_data -= x; return *this; }
        AliasingEntryHelper &operator/=(T x) { m_data /= x; return *this; }
        AliasingEntryHelper &operator*=(T x) { m_data *= x; return *this; }
        AliasingEntryHelper &operator|=(T x) { m_data |= x; return *this; }
        AliasingEntryHelper &operator&=(T x) { m_data &= x; return *this; }
        AliasingEntryHelper &operator^=(T x) { m_data ^= x; return *this; }
        AliasingEntryHelper &operator%=(T x) { m_data %= x; return *this; }
};

} // namespace Common
} // namespace Vc

#include "undomacros.h"

#endif // VC_COMMON_ALIASINGENTRYHELPER_H
