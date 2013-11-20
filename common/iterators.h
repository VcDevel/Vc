/*  This file is part of the Vc library. {{{
Copyright © 2013-2015 Matthias Kretz <kretz@kde.org>
All rights reserved.

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

#ifndef VC_COMMON_ITERATORS_H_
#define VC_COMMON_ITERATORS_H_

#include <array>
#include <iterator>
#include "where.h"
#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace Common
{

template<typename _V, typename Flags> class MemoryVector;
template<typename _V, typename Flags> class MemoryVectorIterator;

// class Iterator {{{
template <typename V>
class Iterator
    : public std::iterator<std::bidirectional_iterator_tag, typename V::EntryType>
    {
        V &v;
        size_t i;
    public:
        constexpr Iterator(V &_v, size_t _i) : v(_v), i(_i) {}
        constexpr Iterator(const Iterator &) = default;
        constexpr Iterator(Iterator &&) = default;

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

#ifdef Vc_IMPL_MIC
    class BitmaskIterator/*{{{*/
    {
        const int mask;
        int bit;
    public:
        Vc_ALWAYS_INLINE BitmaskIterator(int m) : mask(m), bit(_mm_tzcnt_32(mask)) {}
        Vc_ALWAYS_INLINE BitmaskIterator(const BitmaskIterator &) = default;
        Vc_ALWAYS_INLINE BitmaskIterator(BitmaskIterator &&) = default;

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
#ifdef Vc_MSVC
        unsigned long mask;
        unsigned long bit;
#else
        size_t mask;
        size_t bit;
#endif

        void nextBit()
        {
#ifdef Vc_GNU_ASM
            bit = __builtin_ctzl(mask);
#elif defined(Vc_MSVC)
            _BitScanForward(&bit, mask);
#else
#error "Not implemented yet. Please contact vc-devel@compeng.uni-frankfurt.de"
#endif
        }
        void resetLsb()
        {
            // 01100100 - 1 = 01100011
            mask &= (mask - 1);
            /*
#ifdef Vc_GNU_ASM
            __asm__("btr %1,%0" : "+r"(mask) : "r"(bit));
#elif defined(_WIN64)
            _bittestandreset64(&mask, bit);
#elif defined(_WIN32)
            _bittestandreset(&mask, bit);
#else
#error "Not implemented yet. Please contact vc-devel@compeng.uni-frankfurt.de"
#endif
            */
        }
    public:
        BitmaskIterator(decltype(mask) m) : mask(m) { nextBit(); }
        BitmaskIterator(const BitmaskIterator &) = default;
        BitmaskIterator(BitmaskIterator &&) = default;

        Vc_ALWAYS_INLINE size_t operator->() const { return bit; }
        Vc_ALWAYS_INLINE size_t operator*() const { return bit; }

        Vc_ALWAYS_INLINE BitmaskIterator &operator++()    { resetLsb(); nextBit(); return *this; }
        Vc_ALWAYS_INLINE BitmaskIterator  operator++(int) { BitmaskIterator tmp = *this; resetLsb(); nextBit(); return tmp; }

        Vc_ALWAYS_INLINE bool operator==(const BitmaskIterator &rhs) const { return mask == rhs.mask; }
        Vc_ALWAYS_INLINE bool operator!=(const BitmaskIterator &rhs) const { return mask != rhs.mask; }
    };/*}}}*/
#endif

template<typename V> constexpr typename std::enable_if<Traits::is_simd_vector<V>::value, Iterator<V>>::type begin(V &v)
{
    return { v, 0 };
}

template<typename V> constexpr typename std::enable_if<Traits::is_simd_vector<V>::value, Iterator<V>>::type end(V &v)
{
    return { v, V::Size };
}

template<typename V> constexpr typename std::enable_if<Traits::is_simd_mask<V>::value || Traits::is_simd_vector<V>::value, ConstIterator<V>>::type begin(const V &v)
{
    return { v, 0 };
}

template<typename V> constexpr typename std::enable_if<Traits::is_simd_mask<V>::value || Traits::is_simd_vector<V>::value, ConstIterator<V>>::type end(const V &v)
{
    return { v, V::Size };
}

template<typename V> constexpr typename std::enable_if<Traits::is_simd_mask<V>::value || Traits::is_simd_vector<V>::value, ConstIterator<V>>::type cbegin(const V &v)
{
    return { v, 0 };
}

template<typename V> constexpr typename std::enable_if<Traits::is_simd_mask<V>::value || Traits::is_simd_vector<V>::value, ConstIterator<V>>::type cend(const V &v)
{
    return { v, V::Size };
}

template<typename M> Vc_ALWAYS_INLINE BitmaskIterator begin(const WhereImpl::WhereMask<M> &w)
{
    return w.mask.toInt();
}

template<typename M> Vc_ALWAYS_INLINE BitmaskIterator end(const WhereImpl::WhereMask<M> &)
{
    return 0;
}

template<typename V, typename Flags, typename T> Vc_ALWAYS_INLINE MemoryVectorIterator<V, Flags>
    makeIterator(T *mem, Flags)
{
    return new(mem) MemoryVector<V, Flags>;
}

template<typename V, typename Flags, typename T> Vc_ALWAYS_INLINE MemoryVectorIterator<const V, Flags>
    makeIterator(const T *mem, Flags)
{
    return new(const_cast<T *>(mem)) MemoryVector<const V, Flags>;
}

template<typename V, typename Flags, typename FlagsX> Vc_ALWAYS_INLINE MemoryVectorIterator<V, Flags>
    makeIterator(MemoryVector<V, FlagsX> &mv, Flags)
{
    return new(&mv) MemoryVector<V, Flags>;
}

template<typename V, typename Flags, typename FlagsX> Vc_ALWAYS_INLINE MemoryVectorIterator<const V, Flags>
    makeIterator(MemoryVector<const V, FlagsX> &mv, Flags)
{
    return new(&mv) MemoryVector<const V, Flags>;
}

}  // namespace Common

using Common::begin;
using Common::end;
using Common::makeIterator;
}  // namespace Vc

#endif // VC_COMMON_ITERATORS_H_

// vim: foldmethod=marker
