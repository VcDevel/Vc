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

#ifndef VC_SIMPLE_LIMITS_H
#define VC_SIMPLE_LIMITS_H

#include <limits>

namespace Simple
{
    template<typename T> class Vector;
} // namespace Simple

namespace std
{

    template<> inline Simple::Vector<int> numeric_limits<Simple::Vector<int> >::max() throw() { return std::numeric_limits<int>::max(); }
    template<> inline Simple::Vector<int> numeric_limits<Simple::Vector<int> >::min() throw() { return std::numeric_limits<int>::min(); }
    template<> inline Simple::Vector<unsigned int> numeric_limits<Simple::Vector<unsigned int> >::max() throw() { return std::numeric_limits<unsigned int>::max(); }
    template<> inline Simple::Vector<unsigned int> numeric_limits<Simple::Vector<unsigned int> >::min() throw() { return std::numeric_limits<unsigned int>::min(); }
    template<> inline Simple::Vector<short> numeric_limits<Simple::Vector<short> >::max() throw() { return std::numeric_limits<short>::max(); }
    template<> inline Simple::Vector<short> numeric_limits<Simple::Vector<short> >::min() throw() { return std::numeric_limits<short>::min(); }
    template<> inline Simple::Vector<unsigned short> numeric_limits<Simple::Vector<unsigned short> >::max() throw() { return std::numeric_limits<unsigned short>::max(); }
    template<> inline Simple::Vector<unsigned short> numeric_limits<Simple::Vector<unsigned short> >::min() throw() { return std::numeric_limits<unsigned short>::min(); }
    template<> inline Simple::Vector<float> numeric_limits<Simple::Vector<float> >::max() throw() { return std::numeric_limits<float>::max(); }
    template<> inline Simple::Vector<float> numeric_limits<Simple::Vector<float> >::min() throw() { return std::numeric_limits<float>::min(); }
    template<> inline Simple::Vector<double> numeric_limits<Simple::Vector<double> >::max() throw() { return std::numeric_limits<double>::max(); }
    template<> inline Simple::Vector<double> numeric_limits<Simple::Vector<double> >::min() throw() { return std::numeric_limits<double>::min(); }

} // namespace std
#endif // VC_SIMPLE_LIMITS_H
