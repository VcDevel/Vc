/*  This file is part of the Vc library. {{{

    Copyright (C) 2013 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_COMMON_ITERATORS_H
#define VC_COMMON_ITERATORS_H

#include <array>
#include <Vc/type_traits>
#include "macros.h"

Vc_NAMESPACE_BEGIN(Common)

namespace
{
    template<typename V> class Iterator/*{{{*/
    {
        V &v;
        size_t i;
    public:
        constexpr Iterator(V &_v, size_t _i) : v(_v), i(_i) {}
        constexpr Iterator(const Iterator &) = default;
#ifndef VC_NO_MOVE_CTOR
        constexpr Iterator(Iterator &&) = default;
#endif

        Vc_ALWAYS_INLINE decltype(v[i]) operator->() { return v[i]; }
        Vc_ALWAYS_INLINE decltype(v[i]) operator->() const { return v[i]; }

        Vc_ALWAYS_INLINE decltype(v[i]) operator*() { return v[i]; }
        Vc_ALWAYS_INLINE decltype(v[i]) operator*() const { return v[i]; }

        Vc_ALWAYS_INLINE Iterator &operator++()    { ++i; return *this; }
        Vc_ALWAYS_INLINE Iterator  operator++(int) { Iterator tmp = *this; ++i; return tmp; }

        Vc_ALWAYS_INLINE Iterator &operator--()    { --i; return *this; }
        Vc_ALWAYS_INLINE Iterator  operator--(int) { Iterator tmp = *this; --i; return tmp; }

        // XXX: doesn't check &v == &rhs.v
        Vc_ALWAYS_INLINE bool operator==(const Iterator<V> &rhs) const { return i == rhs.i; }
        Vc_ALWAYS_INLINE bool operator!=(const Iterator<V> &rhs) const { return i != rhs.i; }
        Vc_ALWAYS_INLINE bool operator< (const Iterator<V> &rhs) const { return i <  rhs.i; }
        Vc_ALWAYS_INLINE bool operator<=(const Iterator<V> &rhs) const { return i <= rhs.i; }
        Vc_ALWAYS_INLINE bool operator> (const Iterator<V> &rhs) const { return i >  rhs.i; }
        Vc_ALWAYS_INLINE bool operator>=(const Iterator<V> &rhs) const { return i >= rhs.i; }
    };/*}}}*/

    template<typename V> using ConstIterator = Iterator<const V>;

#ifdef VC_IMPL_MIC
    class BitmaskIterator/*{{{*/
    {
        const int mask;
        int bit;
    public:
        Vc_ALWAYS_INLINE BitmaskIterator(int m) : mask(m), bit(_mm_tzcnt_32(mask)) {}
        Vc_ALWAYS_INLINE BitmaskIterator(const BitmaskIterator &) = default;
#ifndef VC_NO_MOVE_CTOR
        Vc_ALWAYS_INLINE BitmaskIterator(BitmaskIterator &&) = default;
#endif

        Vc_ALWAYS_INLINE size_t operator->() const { return bit; }
        Vc_ALWAYS_INLINE size_t operator*() const { return bit; }

        Vc_ALWAYS_INLINE BitmaskIterator &operator++()    {
            bit = _mm_tzcnti_32(bit, mask);
            return *this;
        }
        Vc_ALWAYS_INLINE BitmaskIterator  operator++(int) {
            BitmaskIterator tmp = *this;
            bit = _mm_tzcnti_32(bit, mask);
            return tmp;
        }

        Vc_ALWAYS_INLINE bool operator==(const BitmaskIterator &rhs) const { return bit == rhs.bit; }
        Vc_ALWAYS_INLINE bool operator!=(const BitmaskIterator &rhs) const { return bit != rhs.bit; }
    };/*}}}*/
#else
    class BitmaskIterator/*{{{*/
    {
        size_t mask;
        size_t bit;

        static size_t nextBit(size_t m)
        {
#ifdef VC_GNU_ASM
            return __builtin_ctzl(m);
#elif defined(_WIN64)
            size_t b;
            _BitScanForward64(&b, m);
            return b;
#elif defined(_WIN32)
            size_t b;
            _BitScanForward(&b, m);
            return b;
#else
#error "Not implemented yet. Please contact vc-devel@compeng.uni-frankfurt.de"
#endif
        }
        static size_t resetBit(size_t m, size_t b)
        {
#ifdef VC_GNU_ASM
            __asm__("btr %1,%0" : "+r"(m) : "r"(b));
#elif defined(_WIN64)
            _bittestandreset64(&m, b);
#elif defined(_WIN32)
            _bittestandreset(&m, b);
#else
#error "Not implemented yet. Please contact vc-devel@compeng.uni-frankfurt.de"
#endif
            return m;
        }
    public:
        BitmaskIterator(size_t m) : mask(m), bit(nextBit(m)) {}
        BitmaskIterator(const BitmaskIterator &) = default;
#ifndef VC_NO_MOVE_CTOR
        BitmaskIterator(BitmaskIterator &&) = default;
#endif

        Vc_ALWAYS_INLINE size_t operator->() const { return bit; }
        Vc_ALWAYS_INLINE size_t operator*() const { return bit; }

        Vc_ALWAYS_INLINE BitmaskIterator &operator++()    { bit = nextBit(mask); return *this; }
        Vc_ALWAYS_INLINE BitmaskIterator  operator++(int) { BitmaskIterator tmp = *this; bit = nextBit(mask); return tmp; }

        Vc_ALWAYS_INLINE bool operator==(const BitmaskIterator &rhs) const { return mask == rhs.mask; }
        Vc_ALWAYS_INLINE bool operator!=(const BitmaskIterator &rhs) const { return mask != rhs.mask; }
    };/*}}}*/
#endif
} // anonymous namespace

template<typename V> constexpr typename std::enable_if<is_simd_mask<V>::value || is_simd_vector<V>::value, Iterator<V>>::type begin(V &v)
{
    return { v, 0 };
}

template<typename V> constexpr typename std::enable_if<is_simd_mask<V>::value || is_simd_vector<V>::value, Iterator<V>>::type end(V &v)
{
    return { v, V::Size };
}

template<typename V> constexpr typename std::enable_if<is_simd_mask<V>::value || is_simd_vector<V>::value, ConstIterator<V>>::type begin(const V &v)
{
    return { v, 0 };
}

template<typename V> constexpr typename std::enable_if<is_simd_mask<V>::value || is_simd_vector<V>::value, ConstIterator<V>>::type end(const V &v)
{
    return { v, V::Size };
}

template<typename V> constexpr typename std::enable_if<is_simd_mask<V>::value || is_simd_vector<V>::value, ConstIterator<V>>::type cbegin(const V &v)
{
    return { v, 0 };
}

template<typename V> constexpr typename std::enable_if<is_simd_mask<V>::value || is_simd_vector<V>::value, ConstIterator<V>>::type cend(const V &v)
{
    return { v, V::Size };
}

template<typename M> Vc_ALWAYS_INLINE BitmaskIterator begin(const WhereMask<M> &w)
{
    return w.mask.toInt();
}

template<typename M> Vc_ALWAYS_INLINE BitmaskIterator end(const WhereMask<M> &)
{
    return 0;
}

Vc_NAMESPACE_END

Vc_NAMESPACE_BEGIN(Vc_IMPL_NAMESPACE)
    using ::Vc::Common::begin;
    using ::Vc::Common::end;
Vc_NAMESPACE_END

#include "undomacros.h"

#endif // VC_COMMON_ITERATORS_H

// vim: foldmethod=marker
