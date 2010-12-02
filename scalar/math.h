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

#ifndef VC_SCALAR_MATH_H
#define VC_SCALAR_MATH_H

namespace Vc
{
namespace Scalar
{

template<typename T> static inline Vector<T> min  (const Vector<T> &x, const T &y)
{
    return std::min( x.data(), y );
}

template<typename T> static inline Vector<T> max  (const Vector<T> &x, const T &y)
{
    return std::max( x.data(), y );
}

template<typename T> static inline Vector<T> min  (const T &x, const Vector<T> &y)
{
    return std::min( x, y.data() );
}

template<typename T> static inline Vector<T> max  (const T &x, const Vector<T> &y)
{
    return std::max( x, y.data() );
}

template<typename T> static inline Vector<T> min  (const Vector<T> &x, const Vector<T> &y)
{
    return std::min( x.data(), y.data() );
}

template<typename T> static inline Vector<T> max  (const Vector<T> &x, const Vector<T> &y)
{
    return std::max( x.data(), y.data() );
}

template<typename T> static inline Vector<T> sqrt (const Vector<T> &x)
{
    return std::sqrt( x.data() );
}

template<typename T> static inline Vector<T> rsqrt(const Vector<T> &x)
{
    const T one = 1; return one / std::sqrt( x.data() );
}

template<typename T> static inline Vector<T> abs  (const Vector<T> &x)
{
    return std::abs( x.data() );
}

template<typename T> static inline Vector<T> sin  (const Vector<T> &x)
{
    return std::sin( x.data() );
}

template<typename T> static inline Vector<T> asin (const Vector<T> &x)
{
    return std::asin( x.data() );
}

template<typename T> static inline Vector<T> cos  (const Vector<T> &x)
{
    return std::cos( x.data() );
}

template<typename T> static inline Vector<T> log  (const Vector<T> &x)
{
    return std::log( x.data() );
}

template<typename T> static inline Vector<T> log10(const Vector<T> &x)
{
    return std::log10( x.data() );
}

template<typename T> static inline Vector<T> atan (const Vector<T> &x)
{
    return std::atan( x.data() );
}

template<typename T> static inline Vector<T> atan2(const Vector<T> &x, const Vector<T> &y)
{
    return std::atan2( x.data(), y.data() );
}

template<typename T> static inline Vector<T> round(const Vector<T> &x)
{
    return x;
}

namespace
{
    template<typename T> bool _realIsEvenHalf(T x) {
        const T two = 2;
        const T half = 0.5;
        const T f = std::floor(x * half) * two;
        return (x - f) == half;
    }
} // namespace
template<> inline Vector<float>  round(const Vector<float>  &x)
{
    return std::floor(x.data() + 0.5f) - (_realIsEvenHalf(x.data()) ? 1.f : 0.f);
}

template<> inline Vector<double> round(const Vector<double> &x)
{
    return std::floor(x.data() + 0.5 ) - (_realIsEvenHalf(x.data()) ? 1.  : 0. );
}

template<typename T> static inline Vector<T> reciprocal(const Vector<T> &x)
{
    const T one = 1; return one / x.data();
}

#ifdef isfinite
#undef isfinite
#endif
#ifdef isnan
#undef isnan
#endif
template<typename T> static inline typename Vector<T>::Mask isfinite(const Vector<T> &x)
{
#ifdef _MSC_VER
    return !!_finite(x.data());
#elif defined(__INTEL_COMPILER)
    return ::isfinite(x.data());
#else
    return std::isfinite(x.data());
#endif
}

template<typename T> static inline typename Vector<T>::Mask isnan(const Vector<T> &x)
{
#ifdef _MSC_VER
    return !!_isnan(x.data());
#elif defined(__INTEL_COMPILER)
    return ::isnan(x.data());
#else
    return std::isnan(x.data());
#endif
}

} // namespace Scalar
} // namespace Vc

#endif // VC_SCALAR_MATH_H
