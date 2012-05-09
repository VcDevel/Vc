/*  This file is part of the Vc library.

    Copyright (C) 2011-2012 Matthias Kretz <kretz@kde.org>

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
ALIGN(64) extern unsigned int RandomState[16];

namespace Scalar
{

// conversion/casts {{{1
template<> template<> inline INTRINSIC short_v &Vector<short>::operator=(const ushort_v &x) {
    data() = static_cast<short>(x.data()); return *this;
}
template<> template<> inline INTRINSIC ushort_v &Vector<unsigned short>::operator=(const short_v &x) {
    data() = static_cast<unsigned short>(x.data()); return *this;
}
template<> template<> inline INTRINSIC int_v &Vector<int>::operator=(const uint_v &x) {
    data() = static_cast<int>(x.data()); return *this;
}
template<> template<> inline INTRINSIC uint_v &Vector<unsigned int>::operator=(const int_v &x) {
    data() = static_cast<unsigned int>(x.data()); return *this;
}

// copySign ///////////////////////////////////////////////////////////////////////// {{{1
template<> inline Vector<float> INTRINSIC Vector<float>::copySign(Vector<float> reference) const
{
    union {
        float f;
        unsigned int i;
    } value, sign;
    value.f = data();
    sign.f = reference.data();
    value.i = (sign.i & 0x80000000u) | (value.i & 0x7fffffffu);
    return float_v(value.f);
}
template<> inline sfloat_v INTRINSIC Vector<sfloat>::copySign(sfloat_v reference) const
{
    return sfloat_v(float_v(m_data).copySign(float_v(reference.data())).data());
}
template<> inline Vector<double> INTRINSIC Vector<double>::copySign(Vector<double> reference) const
{
    union {
        double f;
        unsigned long long i;
    } value, sign;
    value.f = data();
    sign.f = reference.data();
    value.i = (sign.i & 0x8000000000000000ull) | (value.i & 0x7fffffffffffffffull);
    return double_v(value.f);
} // }}}1
// bitwise operators {{{1
#define VC_CAST_OPERATOR_FORWARD(op, IntT, VecT) \
template<> inline VecT &VecT::operator op##=(const VecT &x) { \
    typedef IntT uinta MAY_ALIAS; \
    uinta *left = reinterpret_cast<uinta *>(&m_data); \
    const uinta *right = reinterpret_cast<const uinta *>(&x.m_data); \
    *left op##= *right; \
    return *this; \
} \
template<> inline VecT VecT::operator op(const VecT &x) const { \
    VecT ret = *this; \
    return VecT(ret op##= x); \
}
#define VC_CAST_OPERATOR_FORWARD_FLOAT(op)  VC_CAST_OPERATOR_FORWARD(op, unsigned int, Vector<float>)
#define VC_CAST_OPERATOR_FORWARD_SFLOAT(op) VC_CAST_OPERATOR_FORWARD(op, unsigned int, Vector<sfloat>)
#define VC_CAST_OPERATOR_FORWARD_DOUBLE(op) VC_CAST_OPERATOR_FORWARD(op, unsigned long, Vector<double>)
VC_ALL_BINARY(VC_CAST_OPERATOR_FORWARD_FLOAT)
VC_ALL_BINARY(VC_CAST_OPERATOR_FORWARD_SFLOAT)
VC_ALL_BINARY(VC_CAST_OPERATOR_FORWARD_DOUBLE)
#undef VC_CAST_OPERATOR_FORWARD
#undef VC_CAST_OPERATOR_FORWARD_FLOAT
#undef VC_CAST_OPERATOR_FORWARD_SFLOAT
#undef VC_CAST_OPERATOR_FORWARD_DOUBLE
// }}}1
// operators {{{1
#include "../common/operators.h"
// }}}1
// exponent {{{1
template<> inline Vector<float> INTRINSIC Vector<float>::exponent() const
{
    VC_ASSERT(m_data > 0.f);
    union { float f; int i; } value;
    value.f = m_data;
    return float_v(static_cast<float>((value.i >> 23) - 0x7f));
}
template<> inline sfloat_v INTRINSIC Vector<sfloat>::exponent() const
{
    return sfloat_v(float_v(m_data).exponent().data());
}
template<> inline Vector<double> INTRINSIC Vector<double>::exponent() const
{
    VC_ASSERT(m_data > 0.);
    union { double f; long long i; } value;
    value.f = m_data;
    return double_v(static_cast<double>((value.i >> 52) - 0x3ff));
}
// }}}1
// Random {{{1
static inline ALWAYS_INLINE void _doRandomStep(Vector<unsigned int> &state0,
        Vector<unsigned int> &state1)
{
    state0.load(&Vc::RandomState[0]);
    state1.load(&Vc::RandomState[uint_v::Size]);
    (state1 * 0xdeece66du + 11).store(&Vc::RandomState[uint_v::Size]);
    uint_v((state0 * 0xdeece66du + 11).data() ^ (state1.data() >> 16)).store(&Vc::RandomState[0]);
}

template<typename T> inline INTRINSIC Vector<T> Vector<T>::Random()
{
    Vector<unsigned int> state0, state1;
    _doRandomStep(state0, state1);
    return Vector<T>(static_cast<EntryType>(state0.data()));
}
template<> inline INTRINSIC Vector<float> Vector<float>::Random()
{
    Vector<unsigned int> state0, state1;
    _doRandomStep(state0, state1);
    union { unsigned int i; float f; } x;
    x.i = (state0.data() & 0x0fffffffu) | 0x3f800000u;
    return float_v(x.f - 1.f);
}
template<> inline INTRINSIC sfloat_v Vector<sfloat>::Random()
{
    return sfloat_v(Vector<float>::Random().data());
}
template<> inline INTRINSIC Vector<double> Vector<double>::Random()
{
    typedef unsigned long long uint64 MAY_ALIAS;
    uint64 state0 = *reinterpret_cast<const uint64 *>(&Vc::RandomState[8]);
    state0 = (state0 * 0x5deece66dull + 11) & 0x000fffffffffffffull;
    *reinterpret_cast<uint64 *>(&Vc::RandomState[8]) = state0;
    union { unsigned long long i; double f; } x;
    x.i = state0 | 0x3ff0000000000000ull;
    return double_v(x.f - 1.);
}
// }}}1
} // namespace Scalar
} // namespace Vc
// vim: foldmethod=marker
