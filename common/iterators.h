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
#include <iterator>
#include <Vc/type_traits>
#include "storage.h"
#include "macros.h"

Vc_NAMESPACE_BEGIN(Common)

namespace
{

/**
 * \internal
 *
 * Base class for the case that Vector::operator[] returns an lvalue reference or the iterator is
 * const - in which case the non-const dereference operators don't matter.
 * In that case the iterator implements an RandomAccessIterator.
 */
template <typename V,
          bool = std::is_lvalue_reference<decltype(std::declval<V &>()[0])>::value,
          bool = std::is_const<V>::value>
class IteratorBase
    : public std::iterator<std::random_access_iterator_tag,
                           typename std::decay<decltype(std::declval<V &>()[0])>::type>
{
protected:
    typedef decltype(std::declval<const V &>()[0]) const_reference;

    V &v;
    size_t i;

    constexpr IteratorBase(V &_v, size_t _i) : v(_v), i(_i)
    {
    }

public:
    using reference;

    Vc_ALWAYS_INLINE reference operator->()
    {
        return v[i];
    }
    Vc_ALWAYS_INLINE const_reference operator->() const
    {
        return v[i];
    }

    Vc_ALWAYS_INLINE reference operator*()
    {
        return v[i];
    }
    Vc_ALWAYS_INLINE const_reference operator*() const
    {
        return v[i];
    }
};

/**
 * \internal
 *
 * Implements the storage for N wrapper objects of type T, which wrap vector type V.
 *
 * The struct uses a recursive storage because it needs to call the constructors of the array of
 * wrappers with different arguments. For operator[] access a normal array would suffice.
 */
template <typename V, typename T, size_t N> struct ReferenceArray
{
    T data;
    ReferenceArray<V, T, N - 1> next;

    constexpr ReferenceArray(V *v) : data(*v, N - 1), next(v)
    {
    }

    T &operator[](size_t i)
    {
        return (&data)[N - 1 - i];
    }
    const T &operator[](size_t i) const
    {
        return (&data)[N - 1 - i];
    }
};
/**
 * \internal
 * Specialization of the above to end the recursion.
 */
template <typename V, typename T> struct ReferenceArray<V, T, 1>
{
    T data;

    constexpr ReferenceArray(V *v) : data(*v, 0)
    {
    }

    T &operator[](size_t)
    {
        return data;
    }
    const T &operator[](size_t) const
    {
        return data;
    }
};

/**
 * \internal
 *
 * Base class for the case that Vector::operator[] returns an rvalue. In that case the
 * iterator only implements an OutputIterator.
 *
 * This class is crafted such that the non-const dereference operators return an lvalue reference,
 * though.
 */
template <typename V>
class IteratorBase<V, false, false> : public std::iterator<
                                          std::output_iterator_tag,
                                          AliasedVectorEntry<V,
                                                             decltype(std::declval<const V &>()[0])>
                                          // TODO: Distance == ptrdiff_t?
                                          >
{
    // if copied to V::EntryType, detach
    // if assigned to, write to v[i] immediately
    ReferenceArray<V, value_type, V::Size> lvalues;

protected:
    size_t i;
    constexpr IteratorBase(V &_v, size_t _i) : i(_i), lvalues(&_v)
    {
    }

public:
    Vc_ALWAYS_INLINE reference operator->()
    {
        return lvalues[i];
    }
    Vc_ALWAYS_INLINE value_type operator->() const
    {
        return lvalues[i];
    }

    Vc_ALWAYS_INLINE reference operator*()
    {
        return lvalues[i];
    }
    Vc_ALWAYS_INLINE value_type operator*() const
    {
        return lvalues[i];
    }
};

/**
 * \internal
 */
    template<typename V> class Iterator : public IteratorBase<V>/*{{{*/
    {
        using IteratorBase<V>::i;
    public:
        constexpr Iterator(V &_v, size_t _i) : IteratorBase<V>(_v, _i) {}
        constexpr Iterator(const Iterator &) = default;
#ifndef VC_NO_MOVE_CTOR
        constexpr Iterator(Iterator &&) = default;
#endif

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

        void nextBit()
        {
#ifdef VC_GNU_ASM
            bit = __builtin_ctzl(mask);
#elif defined(_WIN64)
            _BitScanForward64(&bit, mask);
#elif defined(_WIN32)
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
#ifdef VC_GNU_ASM
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
        BitmaskIterator(size_t m) : mask(m) { nextBit(); }
        BitmaskIterator(const BitmaskIterator &) = default;
#ifndef VC_NO_MOVE_CTOR
        BitmaskIterator(BitmaskIterator &&) = default;
#endif

        Vc_ALWAYS_INLINE size_t operator->() const { return bit; }
        Vc_ALWAYS_INLINE size_t operator*() const { return bit; }

        Vc_ALWAYS_INLINE BitmaskIterator &operator++()    { resetLsb(); nextBit(); return *this; }
        Vc_ALWAYS_INLINE BitmaskIterator  operator++(int) { BitmaskIterator tmp = *this; resetLsb(); nextBit(); return tmp; }

        Vc_ALWAYS_INLINE bool operator==(const BitmaskIterator &rhs) const { return mask == rhs.mask; }
        Vc_ALWAYS_INLINE bool operator!=(const BitmaskIterator &rhs) const { return mask != rhs.mask; }
    };/*}}}*/
#endif
} // anonymous namespace

template<typename V> constexpr typename std::enable_if<is_simd_vector<V>::value, Iterator<V>>::type begin(V &v)
{
    return { v, 0 };
}

template<typename V> constexpr typename std::enable_if<is_simd_vector<V>::value, Iterator<V>>::type end(V &v)
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

Vc_NAMESPACE_END

Vc_NAMESPACE_BEGIN(Vc_IMPL_NAMESPACE)
    using ::Vc::Common::begin;
    using ::Vc::Common::end;
    using ::Vc::Common::makeIterator;
Vc_NAMESPACE_END

#include "undomacros.h"

#endif // VC_COMMON_ITERATORS_H

// vim: foldmethod=marker
