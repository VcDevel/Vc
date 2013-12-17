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

#ifndef VC_COMMON_SIMD_ARRAY_DATA_H
#define VC_COMMON_SIMD_ARRAY_DATA_H

#include "macros.h"

Vc_NAMESPACE_BEGIN(Common)

namespace Reductions
{
struct LogicalAnd
{
    template <typename T> T operator()(T l, T r)
    {
        return l && r;
    }
};
struct LogicalOr
{
    template <typename T> T operator()(T l, T r)
    {
        return l || r;
    }
};
} // namespace Reductions

template<typename V, std::size_t N> struct ArrayData;
template<typename M, std::size_t N> struct MaskData;

template<std::size_t N, typename... Typelist> struct select_best_vector_type;

template<std::size_t N, typename T> struct select_best_vector_type<N, T>
{
    using type = T;
};
template<std::size_t N, typename T, typename... Typelist> struct select_best_vector_type<N, T, Typelist...>
{
    using type = typename std::conditional <
                 N<T::Size, typename select_best_vector_type<N, Typelist...>::type, T>::type;
};

template<typename V> struct ArrayData<V, 1>
{
    typedef typename V::EntryType value_type;

    V d;

    V *begin() { return &d; }
    const V *cbegin() const { return &d; }

    V *end() { return &d + 1; }
    const V *cend() { return &d + 1; }

    ArrayData() = default;
    Vc_ALWAYS_INLINE ArrayData(const V &x) : d(x) {}
    Vc_ALWAYS_INLINE ArrayData(const value_type *x) : d(x) {}
    template<typename Flags> Vc_ALWAYS_INLINE ArrayData(const value_type *x, Flags flags)
        : d(x, flags) {}
    template<typename U, typename Flags> Vc_ALWAYS_INLINE ArrayData(const U *x, Flags flags)
        : d(x, flags) {}

    Vc_ALWAYS_INLINE ArrayData(VectorSpecialInitializerIndexesFromZero::IEnum x)
        : d(x)
    {
    }
    Vc_ALWAYS_INLINE ArrayData(VectorSpecialInitializerIndexesFromZero::IEnum x, size_t offset)
        : d(x)
    {
        d += offset;
    }

    template<typename U, typename Flags>
    Vc_ALWAYS_INLINE void load(const U *x, Flags f) {
        d.load(x, f);
    }

    template <typename U, typename Flags> Vc_ALWAYS_INLINE void store(U *x, Flags f)
    {
        d.store(x, f);
    }

    template<typename F, typename... Args>
    inline void call(F function, Args... args) {
        (d.*function)(args...);
    }

#define VC_OPERATOR_IMPL(op)                                                                       \
    Vc_ALWAYS_INLINE void operator op##=(const ArrayData<V, 1> & rhs)                              \
    {                                                                                              \
        d op## = rhs.d;                                                                            \
    }
    VC_ALL_BINARY     (VC_OPERATOR_IMPL)
    VC_ALL_ARITHMETICS(VC_OPERATOR_IMPL)
    VC_ALL_SHIFTS     (VC_OPERATOR_IMPL)
#undef VC_OPERATOR_IMPL
};
template<typename V, std::size_t N> struct ArrayData
{
private:
    static constexpr std::size_t secondOffset = N / 2 * V::Size;
public:
    static_assert(N != 0, "error N must be nonzero!");
    typedef typename V::EntryType value_type;

    ArrayData<V, N / 2> first;
    ArrayData<V, N / 2> second;

    V *begin()
    {
        return first.begin();
    }
    const V *cbegin() const
    {
        return first.cbegin();
    }

    V *end()
    {
        return second.end();
    }
    const V *cend()
    {
        return second.cend();
    }

    ArrayData() = default;

    Vc_ALWAYS_INLINE ArrayData(const V &x) : first(x), second(x)
    {
    }
    Vc_ALWAYS_INLINE ArrayData(const value_type *x) : first(x), second(x + secondOffset)
    {
    }
    template <typename Flags>
    Vc_ALWAYS_INLINE ArrayData(const value_type *x, Flags flags)
        : first(x, flags), second(x + secondOffset, flags)
    {
    }
    template<typename U, typename Flags> Vc_ALWAYS_INLINE ArrayData(const U *x, Flags flags)
        : first(x, flags), second(x + secondOffset, flags) {}

    Vc_ALWAYS_INLINE ArrayData(VectorSpecialInitializerIndexesFromZero::IEnum x)
        : first(x), second(x, secondOffset)
    {
    }
    Vc_ALWAYS_INLINE ArrayData(VectorSpecialInitializerIndexesFromZero::IEnum x, size_t offset)
        : first(x, offset), second(x, offset + secondOffset)
    {
    }

    template<typename U, typename Flags>
    Vc_ALWAYS_INLINE void load(const U *x, Flags f) {
        first.load(x, f);
        second.load(x + secondOffset, f);
    }

    template <typename U, typename Flags> Vc_ALWAYS_INLINE void store(U *x, Flags f)
    {
        first.store(x, f);
        second.store(x + secondOffset, f);
    }

    template<typename F, typename... Args>
    inline void call(F function, Args... args) {
        first.call(function, args...);
        second.call(function, args...);
    }

#define VC_OPERATOR_IMPL(op)                                                                       \
    Vc_ALWAYS_INLINE void operator op##=(const ArrayData<V, N> & rhs)                              \
    {                                                                                              \
        first op## = rhs.first;                                                                            \
        second op## = rhs.second;                                                                      \
    }
    VC_ALL_BINARY     (VC_OPERATOR_IMPL)
    VC_ALL_ARITHMETICS(VC_OPERATOR_IMPL)
    VC_ALL_SHIFTS     (VC_OPERATOR_IMPL)
#undef VC_OPERATOR_IMPL
};

template<typename M, std::size_t N> struct MaskData;
template<typename M> struct MaskData<M, 1>
{
    M *begin() { return &d; }
    const M *cbegin() const { return &d; }

    M *end() { return &d + 1; }
    const M *cend() { return &d + 1; }

    MaskData() = default;
    Vc_ALWAYS_INLINE MaskData(const M &x) : d(x) {}

    Vc_ALWAYS_INLINE Vc_PURE bool isFull() const { return d.isFull(); }
    Vc_ALWAYS_INLINE Vc_PURE bool isEmpty() const { return d.isEmpty(); }

    template<typename V, typename F>
    Vc_ALWAYS_INLINE void assign(const ArrayData<V, 1> &lhs, const ArrayData<V, 1> &rhs, F function) {
        d = (lhs.d.*function)(rhs.d);
    }

    template <typename ReductionFunctor>
    Vc_ALWAYS_INLINE M reduce() const
    {
        return d;
    }

//private:
    M d;
};
template<typename M, std::size_t N> struct MaskData
{
    static_assert(N != 0, "error N must be nonzero!");

    M *begin()
    {
        return first.begin();
    }
    const M *cbegin() const
    {
        return first.cbegin();
    }

    M *end()
    {
        return second.end();
    }
    const M *cend()
    {
        return second.cend();
    }

    MaskData() = default;

    Vc_ALWAYS_INLINE MaskData(const M &x) : first(x), second(x)
    {
    }

    Vc_ALWAYS_INLINE Vc_PURE bool isFull() const
    {
        return reduce<Reductions::LogicalAnd>().isFull();
    }
    Vc_ALWAYS_INLINE Vc_PURE bool isEmpty() const
    {
        return reduce<Reductions::LogicalOr>().isEmpty();
    }

    template <typename V, typename F>
    Vc_ALWAYS_INLINE void assign(const ArrayData<V, N> &lhs, const ArrayData<V, N> &rhs, F function)
    {
        first.assign(lhs.first, rhs.first, function);
        second.assign(lhs.second, rhs.second, function);
    }

    template <typename ReductionFunctor>
    Vc_ALWAYS_INLINE M reduce() const
    {
        return ReductionFunctor()(first.reduce<ReductionFunctor>(), second.reduce<ReductionFunctor>());
    }

//private:
    MaskData<M, N / 2> first;
    MaskData<M, N / 2> second;
};

Vc_NAMESPACE_END

#include "undomacros.h"

#endif // VC_COMMON_SIMD_ARRAY_DATA_H
