/*  This file is part of the Vc library. {{{

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

}}}*/

#ifndef TESTS_ULP_H
#define TESTS_ULP_H

#include <Vc/Vc>
#include <Vc/limits>

#ifdef VC_MSVC
namespace std
{
    static inline bool isnan(float  x) { return _isnan(x); }
    static inline bool isnan(double x) { return _isnan(x); }
} // namespace std
#endif

template <typename T, typename = Vc::enable_if<std::is_floating_point<T>::value>>
static T ulpDiffToReference(T val, T ref)
{
    if (val == ref || (std::isnan(val) && std::isnan(ref))) {
        return 0;
    }
    if (ref == T(0)) {
        return 1 + ulpDiffToReference(std::abs(val), std::numeric_limits<T>::min());
    }
    if (val == T(0)) {
        return 1 + ulpDiffToReference(std::numeric_limits<T>::min(), std::abs(ref));
    }

    int exp;
    /*tmp = */ frexp(ref, &exp); // ref == tmp * 2 ^ exp => tmp == ref * 2 ^ -exp
    // tmp is now in the range [0.5, 1.0[
    // now we want to know how many times we can fit 2^-numeric_limits<T>::digits between tmp and
    // val * 2 ^ -exp
    return ldexp(std::abs(ref - val), std::numeric_limits<T>::digits - exp);
}

template <typename T, typename = Vc::enable_if<std::is_floating_point<T>::value>>
inline T ulpDiffToReferenceSigned(T val, T ref)
{
    return ulpDiffToReference(val, ref) * (val - ref < 0 ? -1 : 1);
}

template<typename T> struct UlpExponentVector_ { typedef Vc::int_v Type; };
template <typename T, std::size_t N> struct UlpExponentVector_<Vc::simdarray<T, N>>
{
    using Type = Vc::simdarray<int, N>;
};

template <typename V, typename = Vc::enable_if<Vc::is_simd_vector<V>::value>>
static V ulpDiffToReference(const V &_val, const V &_ref)
{
    using namespace Vc;
    using T = typename V::EntryType;
    using M = typename V::Mask;

    V val = _val;
    V ref = _ref;

    V diff = V::Zero();

    M zeroMask = ref == V::Zero();
    val  (zeroMask)= abs(val);
    ref  (zeroMask)= std::numeric_limits<V>::min();
    diff (zeroMask)= V::One();
    zeroMask = val == V::Zero();
    ref  (zeroMask)= abs(ref);
    val  (zeroMask)= std::numeric_limits<V>::min();
    diff (zeroMask)+= V::One();

    typename V::IndexType exp;
    frexp(ref, &exp);
    diff += ldexp(abs(ref - val), std::numeric_limits<T>::digits - exp);
    diff.setZero(_val == _ref || (isnan(_val) && isnan(_ref)));
    return diff;
}

template <typename T>
inline Vc::enable_if<Vc::is_simd_vector<T>::value && Vc::is_floating_point<T>::value, T> ulpDiffToReferenceSigned(
    const T &_val, const T &_ref)
{
    return ulpDiffToReference(_val, _ref).copySign(_val - _ref);
}

template <typename T>
inline Vc::enable_if<!Vc::is_floating_point<T>::value, T> ulpDiffToReferenceSigned(
    const T &, const T &)
{
    return 0;
}

#endif // TESTS_ULP_H

// vim: foldmethod=marker
