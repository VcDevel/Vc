/*  This file is part of the Vc library.

    Copyright (C) 2010-2011 Matthias Kretz <kretz@kde.org>

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

template<class StorageType> class AliasingEntryHelper
{
    private:
        typedef typename StorageType::EntryType T;
#ifdef VC_ICC
        StorageType *const m_storage;
        const int m_index;
    public:
        AliasingEntryHelper(StorageType *d, int index) : m_storage(d), m_index(index) {}
        AliasingEntryHelper(const AliasingEntryHelper &rhs) : m_storage(rhs.m_storage), m_index(rhs.m_index) {}
        AliasingEntryHelper &operator=(const AliasingEntryHelper &rhs) {
            m_storage->assign(m_index, rhs);
            return *this;
        }

        AliasingEntryHelper &operator  =(T x) { m_storage->assign(m_index, x); return *this; }
        AliasingEntryHelper &operator +=(T x) { m_storage->assign(m_index, m_storage->m(m_index) + x); return *this; }
        AliasingEntryHelper &operator -=(T x) { m_storage->assign(m_index, m_storage->m(m_index) - x); return *this; }
        AliasingEntryHelper &operator /=(T x) { m_storage->assign(m_index, m_storage->m(m_index) / x); return *this; }
        AliasingEntryHelper &operator *=(T x) { m_storage->assign(m_index, m_storage->m(m_index) * x); return *this; }
        AliasingEntryHelper &operator |=(T x) { m_storage->assign(m_index, m_storage->m(m_index) | x); return *this; }
        AliasingEntryHelper &operator &=(T x) { m_storage->assign(m_index, m_storage->m(m_index) & x); return *this; }
        AliasingEntryHelper &operator ^=(T x) { m_storage->assign(m_index, m_storage->m(m_index) ^ x); return *this; }
        AliasingEntryHelper &operator %=(T x) { m_storage->assign(m_index, m_storage->m(m_index) % x); return *this; }
        AliasingEntryHelper &operator<<=(T x) { m_storage->assign(m_index, m_storage->m(m_index)<< x); return *this; }
        AliasingEntryHelper &operator>>=(T x) { m_storage->assign(m_index, m_storage->m(m_index)>> x); return *this; }
#define m_data m_storage->read(m_index)
#else
        typedef T A MAY_ALIAS;
        A &m_data;
    public:
        template<typename T2>
        AliasingEntryHelper(T2 &d) : m_data(reinterpret_cast<A &>(d)) {}

        AliasingEntryHelper(A &d) : m_data(d) {}
        AliasingEntryHelper &operator=(const AliasingEntryHelper &rhs) {
            m_data = rhs.m_data;
            return *this;
        }

        AliasingEntryHelper &operator =(T x) { m_data  = x; return *this; }
        AliasingEntryHelper &operator+=(T x) { m_data += x; return *this; }
        AliasingEntryHelper &operator-=(T x) { m_data -= x; return *this; }
        AliasingEntryHelper &operator/=(T x) { m_data /= x; return *this; }
        AliasingEntryHelper &operator*=(T x) { m_data *= x; return *this; }
        AliasingEntryHelper &operator|=(T x) { m_data |= x; return *this; }
        AliasingEntryHelper &operator&=(T x) { m_data &= x; return *this; }
        AliasingEntryHelper &operator^=(T x) { m_data ^= x; return *this; }
        AliasingEntryHelper &operator%=(T x) { m_data %= x; return *this; }
        AliasingEntryHelper &operator<<=(T x) { m_data <<= x; return *this; }
        AliasingEntryHelper &operator>>=(T x) { m_data >>= x; return *this; }
#endif

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
        //T operator<<(T x) const { return static_cast<T>(m_data) << x; }
        //T operator>>(T x) const { return static_cast<T>(m_data) >> x; }
#ifdef m_data
#undef m_data
#endif
};

} // namespace Common
} // namespace Vc

#include "undomacros.h"

#endif // VC_COMMON_ALIASINGENTRYHELPER_H
