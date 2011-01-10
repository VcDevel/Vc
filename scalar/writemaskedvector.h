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

#ifndef VC_SCALAR_WRITEMASKEDVECTOR_H
#define VC_SCALAR_WRITEMASKEDVECTOR_H

namespace Vc
{
namespace Scalar
{

template<typename T> class WriteMaskedVector
{
    friend class Vector<T>;
    typedef bool Mask;
    public:
        //prefix
        inline Vector<T> &operator++() { if (mask) ++vec->m_data; return *vec; }
        inline Vector<T> &operator--() { if (mask) --vec->m_data; return *vec; }
        //postfix
        inline Vector<T> operator++(int) { if (mask) return vec->m_data++; return *vec; }
        inline Vector<T> operator--(int) { if (mask) return vec->m_data--; return *vec; }

        inline Vector<T> &operator+=(Vector<T> x) { if (mask) vec->m_data += x.m_data; return *vec; }
        inline Vector<T> &operator-=(Vector<T> x) { if (mask) vec->m_data -= x.m_data; return *vec; }
        inline Vector<T> &operator*=(Vector<T> x) { if (mask) vec->m_data *= x.m_data; return *vec; }
        inline Vector<T> &operator/=(Vector<T> x) { if (mask) vec->m_data /= x.m_data; return *vec; }

        inline Vector<T> &operator=(Vector<T> x) {
            vec->assign(x, mask);
            return *vec;
        }
    private:
        WriteMaskedVector(Vector<T> *v, Mask k) : vec(v), mask(k) {}
        Vector<T> *vec;
        Mask mask;
};

} // namespace Scalar
} // namespace Vc
#endif // VC_SCALAR_WRITEMASKEDVECTOR_H
