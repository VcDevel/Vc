/*  This file is part of the Vc library.

    Copyright (C) 2011 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_LARRABEE_SORT_TCC
#define VC_LARRABEE_SORT_TCC

namespace Vc
{
namespace LRBni
{

template<> inline Vector<double> Vector<double>::sorted() const
{
    return *this;
}
template<> inline Vector<float> Vector<float>::sorted() const
{
    return *this;
}
template<> inline Vector<int> Vector<int>::sorted() const
{
    return *this;
}
template<> inline Vector<unsigned int> Vector<unsigned int>::sorted() const
{
    return *this;
}
} // namespace LRBni
} // namespace Vc

#endif // VC_LARRABEE_SORT_TCC
