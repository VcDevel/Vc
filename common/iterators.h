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
    template<typename M> class MaskIterator/*{{{*/
    {
        M &originalMask;
        size_t i;
        std::array<bool, M::Size> data;
    public:
        MaskIterator(M &m, size_t _i) : originalMask(m), i(_i), data(m) {}
        MaskIterator(const MaskIterator &) = default;
        MaskIterator(MaskIterator &&) = default;
        MaskIterator &operator=(const MaskIterator &) = default;

        ~MaskIterator()
        {
            originalMask = data;
        }

        Vc_ALWAYS_INLINE bool &operator->() { return data[i]; }
        Vc_ALWAYS_INLINE bool  operator->() const { return data[i]; }

        Vc_ALWAYS_INLINE bool &operator*() { return data[i]; }
        Vc_ALWAYS_INLINE bool  operator*() const { return data[i]; }

        Vc_ALWAYS_INLINE MaskIterator &operator++()    { ++i; return *this; }
        Vc_ALWAYS_INLINE MaskIterator  operator++(int) { MaskIterator tmp = *this; ++i; return tmp; }

        Vc_ALWAYS_INLINE MaskIterator &operator--()    { --i; return *this; }
        Vc_ALWAYS_INLINE MaskIterator  operator--(int) { MaskIterator tmp = *this; --i; return tmp; }

        Vc_ALWAYS_INLINE bool operator==(const MaskIterator &rhs) const { return i == rhs.i; }
        Vc_ALWAYS_INLINE bool operator!=(const MaskIterator &rhs) const { return i != rhs.i; }
        Vc_ALWAYS_INLINE bool operator< (const MaskIterator &rhs) const { return i <  rhs.i; }
        Vc_ALWAYS_INLINE bool operator<=(const MaskIterator &rhs) const { return i <= rhs.i; }
        Vc_ALWAYS_INLINE bool operator> (const MaskIterator &rhs) const { return i >  rhs.i; }
        Vc_ALWAYS_INLINE bool operator>=(const MaskIterator &rhs) const { return i >= rhs.i; }
    };/*}}}*/
    template<typename M> class ConstMaskIterator/*{{{*/
    {
        const M &mask;
        size_t i;
    public:
        ConstMaskIterator(const M &m, size_t _i) : mask(m), i(_i) {}
        ConstMaskIterator(const ConstMaskIterator &) = default;
        ConstMaskIterator(ConstMaskIterator &&) = default;
        ConstMaskIterator &operator=(const ConstMaskIterator &) = default;

        Vc_ALWAYS_INLINE bool  operator->() const { return mask[i]; }

        Vc_ALWAYS_INLINE bool  operator*() const { return mask[i]; }

        Vc_ALWAYS_INLINE ConstMaskIterator &operator++()    { ++i; return *this; }
        Vc_ALWAYS_INLINE ConstMaskIterator  operator++(int) { ConstMaskIterator tmp = *this; ++i; return tmp; }

        Vc_ALWAYS_INLINE ConstMaskIterator &operator--()    { --i; return *this; }
        Vc_ALWAYS_INLINE ConstMaskIterator  operator--(int) { ConstMaskIterator tmp = *this; --i; return tmp; }

        Vc_ALWAYS_INLINE bool operator==(const ConstMaskIterator &rhs) const { return i == rhs.i; }
        Vc_ALWAYS_INLINE bool operator!=(const ConstMaskIterator &rhs) const { return i != rhs.i; }
        Vc_ALWAYS_INLINE bool operator< (const ConstMaskIterator &rhs) const { return i <  rhs.i; }
        Vc_ALWAYS_INLINE bool operator<=(const ConstMaskIterator &rhs) const { return i <= rhs.i; }
        Vc_ALWAYS_INLINE bool operator> (const ConstMaskIterator &rhs) const { return i >  rhs.i; }
        Vc_ALWAYS_INLINE bool operator>=(const ConstMaskIterator &rhs) const { return i >= rhs.i; }
    };/*}}}*/
/*relational operators MaskIterator <-> ConstMaskIterator{{{*/
        template<typename M> Vc_ALWAYS_INLINE bool operator==(const MaskIterator<M> &lhs, const ConstMaskIterator<M> &rhs) { return lhs.i == rhs.i; }
        template<typename M> Vc_ALWAYS_INLINE bool operator!=(const MaskIterator<M> &lhs, const ConstMaskIterator<M> &rhs) { return lhs.i != rhs.i; }
        template<typename M> Vc_ALWAYS_INLINE bool operator< (const MaskIterator<M> &lhs, const ConstMaskIterator<M> &rhs) { return lhs.i <  rhs.i; }
        template<typename M> Vc_ALWAYS_INLINE bool operator<=(const MaskIterator<M> &lhs, const ConstMaskIterator<M> &rhs) { return lhs.i <= rhs.i; }
        template<typename M> Vc_ALWAYS_INLINE bool operator> (const MaskIterator<M> &lhs, const ConstMaskIterator<M> &rhs) { return lhs.i >  rhs.i; }
        template<typename M> Vc_ALWAYS_INLINE bool operator>=(const MaskIterator<M> &lhs, const ConstMaskIterator<M> &rhs) { return lhs.i >= rhs.i; }

        template<typename M> Vc_ALWAYS_INLINE bool operator==(const ConstMaskIterator<M> &lhs, const MaskIterator<M> &rhs) { return lhs.i == rhs.i; }
        template<typename M> Vc_ALWAYS_INLINE bool operator!=(const ConstMaskIterator<M> &lhs, const MaskIterator<M> &rhs) { return lhs.i != rhs.i; }
        template<typename M> Vc_ALWAYS_INLINE bool operator< (const ConstMaskIterator<M> &lhs, const MaskIterator<M> &rhs) { return lhs.i <  rhs.i; }
        template<typename M> Vc_ALWAYS_INLINE bool operator<=(const ConstMaskIterator<M> &lhs, const MaskIterator<M> &rhs) { return lhs.i <= rhs.i; }
        template<typename M> Vc_ALWAYS_INLINE bool operator> (const ConstMaskIterator<M> &lhs, const MaskIterator<M> &rhs) { return lhs.i >  rhs.i; }
        template<typename M> Vc_ALWAYS_INLINE bool operator>=(const ConstMaskIterator<M> &lhs, const MaskIterator<M> &rhs) { return lhs.i >= rhs.i; }
/*}}}*/
    template<typename V> class Iterator/*{{{*/
    {
        V &v;
        size_t i;
    public:
        Iterator(V &_v, size_t _i) : v(_v), i(_i) {}
        Iterator(const Iterator &) = default;
        Iterator(Iterator &&) = default;

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
    template<typename V> class ConstIterator/*{{{*/
    {
        const V &v;
        size_t i;
    public:
        ConstIterator(const V &_v, size_t _i) : v(_v), i(_i) {}
        ConstIterator(const ConstIterator &) = default;
        ConstIterator(ConstIterator &&) = default;

        Vc_ALWAYS_INLINE decltype(v[i]) operator->() const { return v[i]; }
        Vc_ALWAYS_INLINE decltype(v[i]) operator*() const { return v[i]; }

        Vc_ALWAYS_INLINE ConstIterator &operator++()    { ++i; return *this; }
        Vc_ALWAYS_INLINE ConstIterator  operator++(int) { ConstIterator tmp = *this; ++i; return tmp; }

        Vc_ALWAYS_INLINE ConstIterator &operator--()    { --i; return *this; }
        Vc_ALWAYS_INLINE ConstIterator  operator--(int) { ConstIterator tmp = *this; --i; return tmp; }

        // XXX: doesn't check &v == &rhs.v
        Vc_ALWAYS_INLINE bool operator==(const ConstIterator<V> &rhs) const { return i == rhs.i; }
        Vc_ALWAYS_INLINE bool operator!=(const ConstIterator<V> &rhs) const { return i != rhs.i; }
        Vc_ALWAYS_INLINE bool operator< (const ConstIterator<V> &rhs) const { return i <  rhs.i; }
        Vc_ALWAYS_INLINE bool operator<=(const ConstIterator<V> &rhs) const { return i <= rhs.i; }
        Vc_ALWAYS_INLINE bool operator> (const ConstIterator<V> &rhs) const { return i >  rhs.i; }
        Vc_ALWAYS_INLINE bool operator>=(const ConstIterator<V> &rhs) const { return i >= rhs.i; }
    };/*}}}*/

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
        BitmaskIterator(size_t m) : mask(m) { bit = nextBit(mask); }
        BitmaskIterator(const BitmaskIterator &) = default;
        BitmaskIterator(BitmaskIterator &&) = default;

        Vc_ALWAYS_INLINE size_t operator->() const { return bit; }
        Vc_ALWAYS_INLINE size_t operator*() const { return bit; }

        Vc_ALWAYS_INLINE BitmaskIterator &operator++()    { bit = nextBit(mask); return *this; }
        Vc_ALWAYS_INLINE BitmaskIterator  operator++(int) { BitmaskIterator tmp = *this; bit = nextBit(mask); return tmp; }

        Vc_ALWAYS_INLINE bool operator==(const BitmaskIterator &rhs) const { return mask == rhs.mask; }
        Vc_ALWAYS_INLINE bool operator!=(const BitmaskIterator &rhs) const { return mask != rhs.mask; }
    };/*}}}*/
} // anonymous namespace

template<typename V> constexpr typename std::enable_if<is_simd_vector<V>::value, Iterator<V>>::type begin(V &v)
{
    return { v, 0 };
}

template<typename V> constexpr typename std::enable_if<is_simd_vector<V>::value, Iterator<V>>::type end(V &v)
{
    return { v, V::Size };
}

template<typename V> constexpr typename std::enable_if<is_simd_vector<V>::value, ConstIterator<V>>::type begin(const V &v)
{
    return { v, 0 };
}

template<typename V> constexpr typename std::enable_if<is_simd_vector<V>::value, ConstIterator<V>>::type end(const V &v)
{
    return { v, V::Size };
}

template<typename M> constexpr typename std::enable_if<is_simd_mask<M>::value, MaskIterator<M>>::type begin(M &v)
{
    return { v, 0 };
}

template<typename M> constexpr typename std::enable_if<is_simd_mask<M>::value, MaskIterator<M>>::type end(M &v)
{
    return { v, M::Size };
}

template<typename M> constexpr typename std::enable_if<is_simd_mask<M>::value, ConstMaskIterator<M>>::type begin(const M &v)
{
    return { v, 0 };
}

template<typename M> constexpr typename std::enable_if<is_simd_mask<M>::value, ConstMaskIterator<M>>::type end(const M &v)
{
    return { v, M::Size };
}

template<typename M> constexpr BitmaskIterator begin(const WhereMask<M> &w)
{
    return w.mask.toInt();
}

template<typename M> constexpr BitmaskIterator end(const WhereMask<M> &)
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
