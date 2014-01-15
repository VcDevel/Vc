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

namespace Vc_VERSIONED_NAMESPACE
{
namespace Common
{

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

    template <typename... Args>
    Vc_ALWAYS_INLINE void callMember(void (V::*m)(Args...), Args... args)
    {
        (d.*m)(args...);
    }

    template <typename R, typename... Args>
    Vc_ALWAYS_INLINE void callMember(R (V::*m)(Args...), R r, Args... args)
    {
        r = (d.*m)(args...);
    }

    template <typename U, typename Flags> Vc_ALWAYS_INLINE void store(U *x, Flags f)
    {
        d.store(x, f);
    }

    template<typename F, typename... Args>
    inline void call(F function, Args... args) {
        (d.*function)(args...);
    }

    template<typename F, typename... Args>
    inline void assign(F function, Args... args) {
        d = function(args...);
    }

    Vc_ALWAYS_INLINE void assign(const ArrayData &object, V (V::* function)() const) {
        d = (object.d.*function)();
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

    template<typename U> static Vc_ALWAYS_INLINE U split0(U x) { return x; }
    template<typename U> static Vc_ALWAYS_INLINE U split1(U x) { return x; }
    template<typename U> static Vc_ALWAYS_INLINE U* split1(U* ptr) { return ptr + secondOffset; }
    template<typename U> static Vc_ALWAYS_INLINE const ArrayData<U, N/2> &split0 (const ArrayData<U, N> &x) { return x.data0; }
    template<typename U> static Vc_ALWAYS_INLINE const ArrayData<U, N/2> &split1(const ArrayData<U, N> &x) { return x.data1; }
    template<typename M> static Vc_ALWAYS_INLINE const MaskData<M, N/2> &split0 (const MaskData<M, N> &x) { return x.data0; }
    template<typename M> static Vc_ALWAYS_INLINE const MaskData<M, N/2> &split1(const MaskData<M, N> &x) { return x.data1; }
    template<typename U> static Vc_ALWAYS_INLINE ArrayData<U, N/2> &split0 (ArrayData<U, N> &x) { return x.data0; }
    template<typename U> static Vc_ALWAYS_INLINE ArrayData<U, N/2> &split1(ArrayData<U, N> &x) { return x.data1; }
    template<typename M> static Vc_ALWAYS_INLINE MaskData<M, N/2> &split0 (MaskData<M, N> &x) { return x.data0; }
    template<typename M> static Vc_ALWAYS_INLINE MaskData<M, N/2> &split1(MaskData<M, N> &x) { return x.data1; }

public:
    static_assert(N != 0, "error N must be nonzero!");
    typedef typename V::EntryType value_type;

    ArrayData<V, N / 2> data0;
    ArrayData<V, N / 2> data1;

    V *begin()
    {
        return data0.begin();
    }
    const V *cbegin() const
    {
        return data0.cbegin();
    }

    V *end()
    {
        return data1.end();
    }
    const V *cend()
    {
        return data1.cend();
    }

    ArrayData() = default;

    template <typename... Us>
    Vc_ALWAYS_INLINE ArrayData(Us... xs)
        : data0(split0(xs)...), data1(split1(xs)...)
    {
    }

    Vc_ALWAYS_INLINE ArrayData(VectorSpecialInitializerIndexesFromZero::IEnum x)
        : data0(x), data1(x, secondOffset)
    {
    }
    Vc_ALWAYS_INLINE ArrayData(VectorSpecialInitializerIndexesFromZero::IEnum x, size_t offset)
        : data0(x, offset), data1(x, offset + secondOffset)
    {
    }

    template <typename... Args>
    Vc_ALWAYS_INLINE void callMember(void (V::*m)(Args...), Args... args)
    {
        data0.callMember(m, split0(args)...);
        data1.callMember(m, split1(args)...);
    }

    template <typename R, typename... Args>
    Vc_ALWAYS_INLINE void callMember(R (V::*m)(Args...), R r, Args... args)
    {
        data0.callMember(m, split0(r), split0(args)...);
        data1.callMember(m, split1(r), split1(args)...);
    }

    template <typename U, typename Flags>
    Vc_ALWAYS_INLINE void store(U *x, Flags f)
    {
        data0.store(x, f);
        data1.store(x + secondOffset, f);
    }

    template<typename F, typename... Args>
    inline void call(F function, Args... args) {
        data0.call(function, args...);
        data1.call(function, args...);
    }

    template<typename F, typename... Args>
    inline void assign(F function, Args... args) {
        data0.assign(function, args...);
        data1.assign(function, args...);
    }

    inline void assign(const ArrayData &object, V (V::* function)() const) {
        data0.assign(object.data0, function);
        data1.assign(object.data1, function);
    }

#define VC_OPERATOR_IMPL(op)                                                                       \
    Vc_ALWAYS_INLINE void operator op##=(const ArrayData<V, N> & rhs)                              \
    {                                                                                              \
        data0 op## = rhs.data0;                                                                    \
        data1 op## = rhs.data1;                                                                  \
    }
    VC_ALL_BINARY     (VC_OPERATOR_IMPL)
    VC_ALL_ARITHMETICS(VC_OPERATOR_IMPL)
    VC_ALL_SHIFTS     (VC_OPERATOR_IMPL)
#undef VC_OPERATOR_IMPL
};

template<typename M, std::size_t N> struct MaskData;
template<typename M> struct MaskData<M, 1>
{
    using mask_type = M;

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

    template <typename V, typename F>
    Vc_ALWAYS_INLINE void assign(const ArrayData<V, 1> &operand, F function)
    { d = (operand.d.*function)(); }

    template <typename ReductionFunctor>
    Vc_ALWAYS_INLINE M reduce() const
    {
        return d;
    }

    template <typename Reduce, typename F, typename... Masks>
    Vc_ALWAYS_INLINE auto apply(Reduce, F f, Masks... masks)
        const -> decltype(f(std::declval<const mask_type &>(),
                            std::declval<const typename Masks::mask_type &>()...))
    { return f(d, masks.d...); }

//private:
    M d;
};
template<typename M, std::size_t N> struct MaskData
{
    static_assert(N != 0, "error N must be nonzero!");
    using mask_type = M;

    M *begin()
    {
        return data0.begin();
    }
    const M *cbegin() const
    {
        return data0.cbegin();
    }

    M *end()
    {
        return data1.end();
    }
    const M *cend()
    {
        return data1.cend();
    }

    MaskData() = default;

    Vc_ALWAYS_INLINE MaskData(const M &x) : data0(x), data1(x)
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
        data0.assign(lhs.data0, rhs.data0, function);
        data1.assign(lhs.data1, rhs.data1, function);
    }

    template <typename V, typename F>
    Vc_ALWAYS_INLINE void assign(const ArrayData<V, N> &operand, F function)
    {
        data0.assign(operand.data0, function);
        data1.assign(operand.data1, function);
    }

    template <typename ReductionFunctor>
    Vc_ALWAYS_INLINE M reduce() const
    {
        return ReductionFunctor()(data0.reduce<ReductionFunctor>(), data1.reduce<ReductionFunctor>());
    }

    /**
     * \internal
     * Used for comparision operators in simd_mask_array.
     *
     * \param r reduction function that is applied on the return values of f
     * \param f function applied to the actual mask objects in MaskData<M, 1>. The first argument to
     *          f implicitly is the mask object in the leafs of this MaskData object.
     * \param masks zero or more MaskData<M, N> objects that are inputs to f
     */
    template <typename Reduce, typename F, typename... Masks>
    inline auto apply(Reduce r, F f, Masks... masks)
        const -> decltype(f(std::declval<const mask_type &>(),
                            std::declval<const typename Masks::mask_type &>()...))
    { return r(data0.apply(r, f, masks.data0...), data1.apply(r, f, masks.data1...)); }

//private:
    MaskData<M, N / 2> data0;
    MaskData<M, N / 2> data1;
};

}
}

#include "undomacros.h"

#endif // VC_COMMON_SIMD_ARRAY_DATA_H
