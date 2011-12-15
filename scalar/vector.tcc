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

namespace Vc
{
namespace Scalar
{

// stuff //////////////////////////////////////////////////////////////////////////// {{{1
template<> inline Vector<float> INTRINSIC Vector<float>::copySign(Vector<float> reference) const
{
    union {
        float f;
        unsigned int i;
    } value, sign;
    value.f = data();
    sign.f = reference.data();
    value.i = (sign.i & 0x80000000u) | (value.i & 0x7fffffffu);
    return value.f;
}
template<> inline Vector<double> INTRINSIC Vector<double>::copySign(Vector<double> reference) const
{
    union {
        double f;
        unsigned long long i;
    } value, sign;
    value.f = data();
    sign.f = reference.data();
    value.i = (sign.i & 0x8000000000000000u) | (value.i & 0x7fffffffffffffffu);
    return value.f;
} // }}}1
// bitwise operators {{{1
template<> inline Vector<float> &Vector<float>::operator|=(const Vector<float> &x) {
    typedef unsigned int uinta MAY_ALIAS;
    uinta *left = reinterpret_cast<uinta *>(&m_data);
    const uinta *right = reinterpret_cast<const uinta *>(&x.m_data);
    *left |= *right;
    return *this;
}
template<> inline Vector<float> Vector<float>::operator|(const Vector<float> &x) const {
    Vector<float> ret = *this;
    return ret |= x;
}
template<> inline Vector<float> &Vector<float>::operator&=(const Vector<float> &x) {
    typedef unsigned int uinta MAY_ALIAS;
    uinta *left = reinterpret_cast<uinta *>(&m_data);
    const uinta *right = reinterpret_cast<const uinta *>(&x.m_data);
    *left &= *right;
    return *this;
}
template<> inline Vector<float> Vector<float>::operator&(const Vector<float> &x) const {
    Vector<float> ret = *this;
    return ret &= x;
}
template<> inline Vector<float> &Vector<float>::operator^=(const Vector<float> &x) {
    typedef unsigned int uinta MAY_ALIAS;
    uinta *left = reinterpret_cast<uinta *>(&m_data);
    const uinta *right = reinterpret_cast<const uinta *>(&x.m_data);
    *left ^= *right;
    return *this;
}
template<> inline Vector<float> Vector<float>::operator^(const Vector<float> &x) const {
    Vector<float> ret = *this;
    return ret ^= x;
}

template<> inline Vector<double> &Vector<double>::operator|=(const Vector<double> &x) {
    typedef unsigned long long uinta MAY_ALIAS;
    uinta *left = reinterpret_cast<uinta *>(&m_data);
    const uinta *right = reinterpret_cast<const uinta *>(&x.m_data);
    *left |= *right;
    return *this;
}
template<> inline Vector<double> Vector<double>::operator|(const Vector<double> &x) const {
    Vector<double> ret = *this;
    return ret |= x;
}
template<> inline Vector<double> &Vector<double>::operator&=(const Vector<double> &x) {
    typedef unsigned long long uinta MAY_ALIAS;
    uinta *left = reinterpret_cast<uinta *>(&m_data);
    const uinta *right = reinterpret_cast<const uinta *>(&x.m_data);
    *left &= *right;
    return *this;
}
template<> inline Vector<double> Vector<double>::operator&(const Vector<double> &x) const {
    Vector<double> ret = *this;
    return ret &= x;
}
template<> inline Vector<double> &Vector<double>::operator^=(const Vector<double> &x) {
    typedef unsigned long long uinta MAY_ALIAS;
    uinta *left = reinterpret_cast<uinta *>(&m_data);
    const uinta *right = reinterpret_cast<const uinta *>(&x.m_data);
    *left ^= *right;
    return *this;
}
template<> inline Vector<double> Vector<double>::operator^(const Vector<double> &x) const {
    Vector<double> ret = *this;
    return ret ^= x;
}
// }}}1
// exponent {{{1
template<> inline Vector<float> INTRINSIC Vector<float>::exponent() const
{
    VC_ASSERT(m_data > 0.f);
    union { float f; int i; } value;
    value.f = m_data;
    return static_cast<float>((value.i >> 23) - 0x7f);
}
template<> inline Vector<double> INTRINSIC Vector<double>::exponent() const
{
    VC_ASSERT(m_data > 0.);
    union { double f; long long i; } value;
    value.f = m_data;
    return static_cast<double>((value.i >> 52) - 0x3ff);
}
// }}}1
} // namespace Scalar
} // namespace Vc
// vim: foldmethod=marker
