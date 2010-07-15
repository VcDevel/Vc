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

namespace Vc
{
namespace LRBni
{

template<typename T> template<typename T2> inline void Vector<T>::store(T2 *mem) const
{
    VectorHelper<T>::store(mem, data.v(), Aligned);
}

template<typename T> template<typename T2> inline void Vector<T>::store(T2 *mem, Mask mask) const
{
    VectorHelper<T>::store(mem, data.v(), mask.data(), Aligned);
}

template<typename T> template<typename T2, typename A> inline void Vector<T>::store(T2 *mem, A align) const
{
    VectorHelper<T>::store(mem, data.v(), align);
}

template<typename T> template<typename T2, typename A> inline void Vector<T>::store(T2 *mem, Mask mask, A align) const
{
    VectorHelper<T>::store(mem, data.v(), mask.data(), align);
}

} // namespace LRBni
} // namespace Vc
