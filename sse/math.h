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

#ifndef VC_SSE_MATH_H
#define VC_SSE_MATH_H

#include "const.h"

namespace Vc
{
namespace SSE
{
    template<typename T> inline Vector<T> c_sin<T>::_1_2pi()  { return Vector<T>(&_data[0 * Size]); }
    template<typename T> inline Vector<T> c_sin<T>::_2pi()    { return Vector<T>(&_data[1 * Size]); }
    template<typename T> inline Vector<T> c_sin<T>::_pi_2()   { return Vector<T>(&_data[2 * Size]); }
    template<typename T> inline Vector<T> c_sin<T>::_pi()     { return Vector<T>(&_data[3 * Size]); }

    template<typename T> inline Vector<T> c_sin<T>::_1_3fac() { return Vector<T>(&_data[4 * Size]); }
    template<typename T> inline Vector<T> c_sin<T>::_1_5fac() { return Vector<T>(&_data[5 * Size]); }
    template<typename T> inline Vector<T> c_sin<T>::_1_7fac() { return Vector<T>(&_data[6 * Size]); }
    template<typename T> inline Vector<T> c_sin<T>::_1_9fac() { return Vector<T>(&_data[7 * Size]); }

    template<> inline Vector<float8> c_sin<float8>::_1_2pi()  { return Vector<float8>::broadcast4(&c_sin<float>::_data[ 0]); }
    template<> inline Vector<float8> c_sin<float8>::_2pi()    { return Vector<float8>::broadcast4(&c_sin<float>::_data[ 4]); }
    template<> inline Vector<float8> c_sin<float8>::_pi_2()   { return Vector<float8>::broadcast4(&c_sin<float>::_data[ 8]); }
    template<> inline Vector<float8> c_sin<float8>::_pi()     { return Vector<float8>::broadcast4(&c_sin<float>::_data[12]); }

    template<> inline Vector<float8> c_sin<float8>::_1_3fac() { return Vector<float8>::broadcast4(&c_sin<float>::_data[16]); }
    template<> inline Vector<float8> c_sin<float8>::_1_5fac() { return Vector<float8>::broadcast4(&c_sin<float>::_data[20]); }
    template<> inline Vector<float8> c_sin<float8>::_1_7fac() { return Vector<float8>::broadcast4(&c_sin<float>::_data[24]); }
    template<> inline Vector<float8> c_sin<float8>::_1_9fac() { return Vector<float8>::broadcast4(&c_sin<float>::_data[28]); }
} // namespace SSE
} // namespace Vc

#define VC__USE_NAMESPACE SSE
#include "../common/trigonometric.h"
#define VC__USE_NAMESPACE SSE
#include "../common/logarithm.h"

#endif // VC_SSE_MATH_H
