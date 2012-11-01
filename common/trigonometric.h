/*  This file is part of the Vc library.

    Copyright (C) 2009-2012 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_COMMON_TRIGONOMETRIC_H
#define VC_COMMON_TRIGONOMETRIC_H

#ifndef VC__USE_NAMESPACE
#error "Do not include Vc/common/trigonometric.h outside of Vc itself"
#endif
namespace Vc
{
namespace
{
    using Vc::VC__USE_NAMESPACE::Vector;
} // namespace 
template<Implementation Impl> struct Trigonometric
{
    template<typename T> static Vector<T> sin(const Vector<T> &_x);
    template<typename T> static Vector<T> cos(const Vector<T> &_x);
    template<typename T> static void sincos(const Vector<T> &_x, Vector<T> *_sin, Vector<T> *_cos);
    template<typename T> static Vector<T> asin (const Vector<T> &_x);
    template<typename T> static Vector<T> atan (const Vector<T> &_x);
    template<typename T> static Vector<T> atan2(const Vector<T> &y, const Vector<T> &x);
};
namespace VC__USE_NAMESPACE
#undef VC__USE_NAMESPACE
{
    template<typename T> static inline Vector<T> sin(const Vector<T> &_x) {
        return Vc::Trigonometric<VC_IMPL>::sin(_x);
    }
    template<typename T> static inline Vector<T> cos(const Vector<T> &_x) {
        return Vc::Trigonometric<VC_IMPL>::cos(_x);
    }
    template<typename T> static inline void sincos(const Vector<T> &_x, Vector<T> *_sin, Vector<T> *_cos) {
        Vc::Trigonometric<VC_IMPL>::sincos(_x, _sin, _cos);
    }
    template<typename T> static inline Vector<T> asin (const Vector<T> &_x) {
        return Vc::Trigonometric<VC_IMPL>::asin(_x);
    }
    template<typename T> static inline Vector<T> atan (const Vector<T> &_x) {
        return Vc::Trigonometric<VC_IMPL>::atan(_x);
    }
    template<typename T> static inline Vector<T> atan2(const Vector<T> &y, const Vector<T> &x) {
        return Vc::Trigonometric<VC_IMPL>::atan2(y, x);
    }
} // namespace VC__USE_NAMESPACE
} // namespace Vc

#endif // VC_COMMON_TRIGONOMETRIC_H
