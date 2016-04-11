/*  This file is part of the Vc library. {{{
Copyright © 2011-2015 Matthias Kretz <kretz@kde.org>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the names of contributing organizations nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

}}}*/

#ifndef TESTS_ULP_H
#define TESTS_ULP_H

#include <Vc/Vc>
#include <Vc/limits>

#ifdef Vc_MSVC
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
template <typename T, std::size_t N> struct UlpExponentVector_<Vc::SimdArray<T, N>>
{
    using Type = Vc::SimdArray<int, N>;
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
    return copysign(ulpDiffToReference(_val, _ref), _val - _ref);
}

template <typename T>
inline Vc::enable_if<!Vc::is_floating_point<T>::value, T> ulpDiffToReferenceSigned(
    const T &, const T &)
{
    return 0;
}

#endif // TESTS_ULP_H

// vim: foldmethod=marker
