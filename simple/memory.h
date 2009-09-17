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

#ifndef VC_SIMPLE_MEMORY_H
#define VC_SIMPLE_MEMORY_H

namespace Vc
{
namespace Simple
{

template<typename T> class _Memory
{
    private:
        T d;
    public:
        inline int size() const { return 1; }
        inline T &operator[](int) { return d; }
        inline T operator[](int) const { return d; }
        inline operator T*() { return &d; }
        inline operator const T*() const { return &d; }

        inline _Memory &operator=(const _Memory &rhs) {
            d = rhs.d;
            return *this;
        }
        inline _Memory &operator=(const Vector<T> &rhs) {
            d = rhs[0];
            return *this;
        }
};

} // namespace Simple
} // namespace Vc

#endif // VC_SIMPLE_MEMORY_H
