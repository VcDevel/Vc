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

#include "subscript.h"
#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace Common
{

template<typename V, std::size_t N> struct ArrayData;
template<typename M, std::size_t N> struct MaskData;

namespace Reductions/*{{{*/
{
struct Defaults
{
    template <typename T> Vc_ALWAYS_INLINE Vc_CONST static T processLeaf(T &&x)
    {
        return std::forward<T>(x);
    }
};
struct LogicalAnd : public Defaults
{
    template <typename T> T operator()(T l, T r)
    {
        return l && r;
    }
};
struct LogicalOr : public Defaults
{
    template <typename T> T operator()(T l, T r)
    {
        return l || r;
    }
};
struct Count : public Defaults
{
    template <typename T> T operator()(T l, T r)
    {
        return l + r;
    }
    template <typename T> Vc_ALWAYS_INLINE Vc_CONST static unsigned int processLeaf(T &&x)
    {
        return std::forward<T>(x).count();
    }
};
} // namespace Reductions}}}
namespace Operations/*{{{*/
{
struct Gather
{
    template <typename V, typename... Args> void operator()(V &v, Args &&... args)
    {
        v.gather(std::forward<Args>(args)...);
    }
};
struct Abs
{
    template <typename V, typename... Args> void operator()(V &v, Args &&... args)
    {
        v = abs(std::forward<Args>(args)...);
    }
};
struct Scatter
{
    template <typename V, typename... Args> void operator()(const V &v, Args &&... args)
    {
        v.scatter(std::forward<Args>(args)...);
    }
};
struct Load
{
    template <typename V, typename... Args> void operator()(V &v, Args &&... args)
    {
        v.load(std::forward<Args>(args)...);
    }
};
struct Store {
    template <typename V, typename... Args> void operator()(const V &v, Args &&... args)
    {
        v.store(std::forward<Args>(args)...);
    }
};
struct Increment {
    template <typename V> void operator()(V &v)
    {
        ++v;
    }
};
struct Decrement {
    template <typename V> void operator()(V &v)
    {
        --v;
    }
};
struct Subscript
{
    template <typename V> decltype(std::declval<V>()[0]) operator()(V &&v, std::size_t i)
    {
        return v[i];
    }
};
struct SetZero
{
    template <typename V, typename... Args> void operator()(V &v, Args &&... args)
    {
        v.setZero(std::forward<Args>(args)...);
    }
};
struct SetZeroInverted
{
    template <typename V, typename... Args> void operator()(V &v, Args &&... args)
    {
        v.setZeroInverted(std::forward<Args>(args)...);
    }
};
struct Assign
{
    template <typename V, typename... Args> void operator()(V &v, Args &&... args)
    {
        v.assign(std::forward<Args>(args)...);
    }
};
}  // namespace Operations }}}
/*select_best_vector_type{{{*/
namespace internal
{
/**
 * \internal
 * AVX::Vector<T> with T int, uint, short, or ushort is either two SSE::Vector<T> or the same as
 * SSE::Vector<T>. Thus we can skip AVX::Vector<T> for integral types altogether.
 */
template <typename T> struct never_best_vector_type : public std::false_type {};

// the AVX namespace only exists in AVX compilations, otherwise it's AVX2 - which is fine
#if defined(VC_IMPL_AVX) && !defined(VC_IMPL_AVX2)
template <typename T> struct never_best_vector_type<AVX::Vector<T>> : public std::is_integral<T> {};
#endif
}  // namespace internal

/**
 * \internal
 * Selects the best SIMD type out of a typelist to store N scalar values.
 */
template<std::size_t N, typename... Typelist> struct select_best_vector_type;

template<std::size_t N, typename T> struct select_best_vector_type<N, T>
{
    using type = T;
};
template<std::size_t N, typename T, typename... Typelist> struct select_best_vector_type<N, T, Typelist...>
{
    using type = typename std::conditional<(N < T::Size || internal::never_best_vector_type<T>::value),
                                           typename select_best_vector_type<N, Typelist...>::type,
                                           T>::type;
};//}}}

/**
 * \internal
 * Helper type to statically communicate segmentation of one vector register into 2^n parts
 * (Pieces).
 */
template <typename T_, std::size_t Pieces_, std::size_t Index_> struct Segment/*{{{*/
{
    static_assert(Index_ < Pieces_, "You found a bug in Vc. Please report.");

    using type = T_;
    using type_decayed = typename std::decay<type>::type;
    static constexpr std::size_t Pieces = Pieces_;
    static constexpr std::size_t Index = Index_;

    type data;

    static constexpr std::size_t EntryOffset = Index * type_decayed::Size / Pieces;

    decltype(std::declval<type>()[0]) operator[](size_t i) { return data[i + EntryOffset]; }
    decltype(std::declval<type>()[0]) operator[](size_t i) const { return data[i + EntryOffset]; }
};/*}}}*/

/**
 * \internal
 * Helper type with static functions to generically adjust arguments for the data0 and data1
 * members.
 */
template <std::size_t secondOffset> struct Split/*{{{*/
{
    template<typename Op = void, typename U> static Vc_ALWAYS_INLINE U lo(U &&x) { return std::forward<U>(x); }
    template<typename Op = void, typename U> static Vc_ALWAYS_INLINE U hi(U &&x) { return std::forward<U>(x); }
    template <typename Op, typename U> static Vc_ALWAYS_INLINE U *hi(U *ptr, typename std::enable_if< std::is_same<Op, Operations::Gather>::value ||  std::is_same<Op, Operations::Scatter>::value>::type = nullptr) { return ptr; }
    template <typename Op, typename U> static Vc_ALWAYS_INLINE U *hi(U *ptr, typename std::enable_if<!std::is_same<Op, Operations::Gather>::value && !std::is_same<Op, Operations::Scatter>::value>::type = nullptr) { return ptr + secondOffset; }
    template <typename Op> static Vc_ALWAYS_INLINE std::size_t hi(std::size_t i, typename std::enable_if< std::is_same<Op, Operations::Subscript>::value>::type = nullptr) { return i + secondOffset; }
    template <typename Op> static Vc_ALWAYS_INLINE std::size_t hi(std::size_t i, typename std::enable_if<!std::is_same<Op, Operations::Subscript>::value>::type = nullptr) { return i; }

    template <typename Op = void, typename U> static Vc_ALWAYS_INLINE Segment<const U &, 2, 0> lo(const ArrayData<U, 1> &x) { return {x.d}; }
    template <typename Op = void, typename U> static Vc_ALWAYS_INLINE Segment<const U &, 2, 1> hi(const ArrayData<U, 1> &x) { return {x.d}; }
    template <typename Op = void, typename U> static Vc_ALWAYS_INLINE Segment<const U &, 2, 0> lo(const  MaskData<U, 1> &x) { return {x.d}; }
    template <typename Op = void, typename U> static Vc_ALWAYS_INLINE Segment<const U &, 2, 1> hi(const  MaskData<U, 1> &x) { return {x.d}; }
    template <typename Op = void, typename U> static Vc_ALWAYS_INLINE Segment<      U &, 2, 0> lo(      ArrayData<U, 1> &x) { return {x.d}; }
    template <typename Op = void, typename U> static Vc_ALWAYS_INLINE Segment<      U &, 2, 1> hi(      ArrayData<U, 1> &x) { return {x.d}; }
    template <typename Op = void, typename U> static Vc_ALWAYS_INLINE Segment<      U &, 2, 0> lo(       MaskData<U, 1> &x) { return {x.d}; }
    template <typename Op = void, typename U> static Vc_ALWAYS_INLINE Segment<      U &, 2, 1> hi(       MaskData<U, 1> &x) { return {x.d}; }
    template <typename Op = void, typename U> static Vc_ALWAYS_INLINE Segment<      U &, 2, 0> lo(      ArrayData<U, 1>&&x) { return {x.d}; }
    template <typename Op = void, typename U> static Vc_ALWAYS_INLINE Segment<      U &, 2, 1> hi(      ArrayData<U, 1>&&x) { return {x.d}; }
    template <typename Op = void, typename U> static Vc_ALWAYS_INLINE Segment<      U &, 2, 0> lo(       MaskData<U, 1>&&x) { return {x.d}; }
    template <typename Op = void, typename U> static Vc_ALWAYS_INLINE Segment<      U &, 2, 1> hi(       MaskData<U, 1>&&x) { return {x.d}; }

    template <typename Op = void, typename U, std::size_t N2, typename = enable_if<(N2 > 1)>> static Vc_ALWAYS_INLINE const ArrayData<U, N2 / 2>  &lo(const ArrayData<U, N2>  &x) { return x.data0; }
    template <typename Op = void, typename U, std::size_t N2, typename = enable_if<(N2 > 1)>> static Vc_ALWAYS_INLINE const ArrayData<U, N2 / 2>  &hi(const ArrayData<U, N2>  &x) { return x.data1; }
    template <typename Op = void, typename M, std::size_t N2, typename = enable_if<(N2 > 1)>> static Vc_ALWAYS_INLINE const  MaskData<M, N2 / 2>  &lo(const  MaskData<M, N2>  &x) { return x.data0; }
    template <typename Op = void, typename M, std::size_t N2, typename = enable_if<(N2 > 1)>> static Vc_ALWAYS_INLINE const  MaskData<M, N2 / 2>  &hi(const  MaskData<M, N2>  &x) { return x.data1; }
    template <typename Op = void, typename U, std::size_t N2, typename = enable_if<(N2 > 1)>> static Vc_ALWAYS_INLINE       ArrayData<U, N2 / 2>  &lo(      ArrayData<U, N2>  &x) { return x.data0; }
    template <typename Op = void, typename U, std::size_t N2, typename = enable_if<(N2 > 1)>> static Vc_ALWAYS_INLINE       ArrayData<U, N2 / 2>  &hi(      ArrayData<U, N2>  &x) { return x.data1; }
    template <typename Op = void, typename M, std::size_t N2, typename = enable_if<(N2 > 1)>> static Vc_ALWAYS_INLINE        MaskData<M, N2 / 2>  &lo(       MaskData<M, N2>  &x) { return x.data0; }
    template <typename Op = void, typename M, std::size_t N2, typename = enable_if<(N2 > 1)>> static Vc_ALWAYS_INLINE        MaskData<M, N2 / 2>  &hi(       MaskData<M, N2>  &x) { return x.data1; }
    template <typename Op = void, typename U, std::size_t N2, typename = enable_if<(N2 > 1)>> static Vc_ALWAYS_INLINE       ArrayData<U, N2 / 2> &&lo(      ArrayData<U, N2> &&x) { return std::move(x.data0); }
    template <typename Op = void, typename U, std::size_t N2, typename = enable_if<(N2 > 1)>> static Vc_ALWAYS_INLINE       ArrayData<U, N2 / 2> &&hi(      ArrayData<U, N2> &&x) { return std::move(x.data1); }
    template <typename Op = void, typename M, std::size_t N2, typename = enable_if<(N2 > 1)>> static Vc_ALWAYS_INLINE        MaskData<M, N2 / 2> &&lo(       MaskData<M, N2> &&x) { return std::move(x.data0); }
    template <typename Op = void, typename M, std::size_t N2, typename = enable_if<(N2 > 1)>> static Vc_ALWAYS_INLINE        MaskData<M, N2 / 2> &&hi(       MaskData<M, N2> &&x) { return std::move(x.data1); }

    template <typename Op = void, typename U, std::size_t Pieces, std::size_t Index> static Vc_ALWAYS_INLINE Segment<const U &, 2 * Pieces, Index * Pieces + 0> lo(const Segment<const U &, Pieces, Index> &x) { return {x.data}; }
    template <typename Op = void, typename U, std::size_t Pieces, std::size_t Index> static Vc_ALWAYS_INLINE Segment<const U &, 2 * Pieces, Index * Pieces + 1> hi(const Segment<const U &, Pieces, Index> &x) { return {x.data}; }
    template <typename Op = void, typename U, std::size_t Pieces, std::size_t Index> static Vc_ALWAYS_INLINE Segment<      U &, 2 * Pieces, Index * Pieces + 0> lo(const Segment<      U &, Pieces, Index> &x) { return {x.data}; }
    template <typename Op = void, typename U, std::size_t Pieces, std::size_t Index> static Vc_ALWAYS_INLINE Segment<      U &, 2 * Pieces, Index * Pieces + 1> hi(const Segment<      U &, Pieces, Index> &x) { return {x.data}; }
    template <typename Op = void, typename U, std::size_t Pieces, std::size_t Index> static Vc_ALWAYS_INLINE Segment<      U &, 2 * Pieces, Index * Pieces + 0> lo(      Segment<      U &, Pieces, Index>&&x) { return {x.data}; }
    template <typename Op = void, typename U, std::size_t Pieces, std::size_t Index> static Vc_ALWAYS_INLINE Segment<      U &, 2 * Pieces, Index * Pieces + 1> hi(      Segment<      U &, Pieces, Index>&&x) { return {x.data}; }
};/*}}}*/

//                                         ArrayData<M, 1>
template<typename V> struct ArrayData<V, 1>/*{{{*/
{
private:
    template <typename U> static Vc_ALWAYS_INLINE const U &actual_value(const ArrayData<U, 1> &x) { return x.d; }
    template <typename M> static Vc_ALWAYS_INLINE const M &actual_value(const MaskData<M, 1> &x) { return x.d; }
    template <typename U> static Vc_ALWAYS_INLINE U &actual_value(ArrayData<U, 1> &x) { return x.d; }
    template <typename M> static Vc_ALWAYS_INLINE M &actual_value(MaskData<M, 1> &x) { return x.d; }
    template <typename U> static Vc_ALWAYS_INLINE U &&actual_value(ArrayData<U, 1> &&x) { return std::move(x.d); }
    template <typename M> static Vc_ALWAYS_INLINE M &&actual_value(MaskData<M, 1> &&x) { return std::move(x.d); }

    template <typename U, std::size_t N, typename = enable_if<(N > 1)>> static Vc_ALWAYS_INLINE const simd_array<typename U::EntryType, U::Size * N, U> actual_value(const ArrayData<U, N> &x) { return x; }
    template <typename U, std::size_t N, typename = enable_if<(N > 1)>> static Vc_ALWAYS_INLINE const simd_array<typename U::EntryType, U::Size * N, U> actual_value(ArrayData<U, N>  &x) { return x; }
    //template <typename U, std::size_t N, typename = enable_if<(N > 1)>> static Vc_ALWAYS_INLINE const simd_mask_array<typename U::EntryType, U::Size * N, U> actual_value(const MaskData<U, N> &x) { return x; }
    //template <typename U, std::size_t N, typename = enable_if<(N > 1)>> static Vc_ALWAYS_INLINE const simd_mask_array<typename U::EntryType, U::Size * N, U> actual_value(MaskData<U, N>  &x) { return x; }

    template <typename T, std::size_t P> static Vc_ALWAYS_INLINE T actual_value(const Segment<T, P, 0> &x) { return x.data; }
    template <typename T, std::size_t P> static Vc_ALWAYS_INLINE T actual_value(Segment<T, P, 0> &&x) { return x.data; }
    template <typename T, std::size_t P, std::size_t I, typename = enable_if<(I > 0)>>
    static Vc_ALWAYS_INLINE typename std::remove_reference<T>::type actual_value(Segment<T, P, I> &&x)
    {
        return x.data.shifted(x.EntryOffset);
    }

    template <typename U> static Vc_ALWAYS_INLINE U actual_value(U &&x) { return std::forward<U>(x); }

public:
    typedef typename V::EntryType value_type;

    V d;

    V *begin() { return &d; }
    const V *cbegin() const { return &d; }

    V *end() { return &d + 1; }
    const V *cend() { return &d + 1; }

    ArrayData() = default;

    template <typename U>
    Vc_ALWAYS_INLINE ArrayData(const ArrayData<U, 2> &x)
        : d(simd_cast<V>(x.data0.d, x.data1.d))
    {
    }

    template <typename U>
    Vc_ALWAYS_INLINE ArrayData(const ArrayData<U, 4> &x)
        : d(simd_cast<V>(x.data0.data0.d, x.data0.data1.d, x.data1.data0.d, x.data1.data1.d))
    {
    }

    template <typename... Us>
    Vc_ALWAYS_INLINE ArrayData(Us &&... xs)
        : d(actual_value(std::forward<Us>(xs))...)
    {
    }

    template <typename... Us, typename = enable_if<!Traits::is_gather_signature<Us...>::value>>
    Vc_ALWAYS_INLINE ArrayData(Us &&... xs)
        : d(actual_value(std::forward<Us>(xs))...)
    {
    }

    template <typename Arg0,
              typename... Args,
              typename = enable_if<Traits::is_gather_signature<Arg0, Args...>::value>>
    Vc_ALWAYS_INLINE ArrayData(Arg0 &&arg0, Args &&... args)
        : d()
    {
        call<Common::Operations::Gather>(std::forward<Arg0>(arg0), std::forward<Args>(args)...);
    }

    Vc_ALWAYS_INLINE ArrayData(VectorSpecialInitializerIndexesFromZero::IEnum x, size_t offset)
        : d(x)
    {
        d += offset;
    }

    template <typename Op, typename... Args> Vc_ALWAYS_INLINE void call(Args... args)
    {
        Op()(d, actual_value(args)...);
    }

    template <typename Op, typename... Args> Vc_ALWAYS_INLINE void call(Args... args) const
    {
        Op()(d, actual_value(args)...);
    }

    template <typename Op, typename R, typename... Args>
    Vc_ALWAYS_INLINE void callReturningMember(R r, Args... args) const
    {
        actual_value(r) = Op()(d, actual_value(args)...);
    }

    template <typename U, typename Flags> Vc_ALWAYS_INLINE void store(U *x, Flags f)
    {
        d.store(x, f);
    }

    template<typename F, typename... Args>
    inline void assign(F function, Args &&... args) {
        d = function(actual_value(std::forward<Args>(args))...);
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

    template<typename W> Vc_ALWAYS_INLINE void cast(const ArrayData<W, 1> &x) {
        d = static_cast<V>(x.d);
    }
    template<typename W> Vc_ALWAYS_INLINE void cast(const ArrayData<W, 2> &x) {
        d = simd_cast<V>(x.data0.d, x.data1.d);
    }
    template<typename W> Vc_ALWAYS_INLINE void cast(const ArrayData<W, 4> &x) {
        d = simd_cast<V>(x.data0.data0.d, x.data0.data1.d, x.data1.data0.d, x.data1.data1.d);
    }
    template<typename W> Vc_ALWAYS_INLINE void cast(const ArrayData<W, 8> &x) {
        d = simd_cast<V>(x.data0.data0.data0.d,
                         x.data0.data0.data1.d,
                         x.data0.data1.data0.d,
                         x.data0.data1.data1.d,
                         x.data1.data0.data0.d,
                         x.data1.data0.data1.d,
                         x.data1.data1.data0.d,
                         x.data1.data1.data1.d);
    }
};/*}}}*/

//                                         ArrayData<M, N>
template<typename V, std::size_t N> struct ArrayData/*{{{*/
{
private:
    static constexpr std::size_t secondOffset = N / 2 * V::Size;
    using Split = Common::Split<secondOffset>;

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

    template <typename... Us, typename = enable_if<!Traits::is_gather_signature<Us...>::value>>
    Vc_ALWAYS_INLINE ArrayData(Us &&... xs)
        : data0(Split::lo(std::forward<Us>(xs))...), data1(Split::hi(std::forward<Us>(xs))...)
    {
    }

    template <typename Arg0,
              typename... Args,
              typename = enable_if<Traits::is_gather_signature<Arg0, Args...>::value>>
    Vc_ALWAYS_INLINE ArrayData(Arg0 &&arg0, Args &&... args)
        : data0(), data1()
    {
        call<Common::Operations::Gather>(std::forward<Arg0>(arg0), std::forward<Args>(args)...);
    }

    Vc_ALWAYS_INLINE ArrayData(VectorSpecialInitializerIndexesFromZero::IEnum x)
        : data0(x), data1(x, secondOffset)
    {
    }
    Vc_ALWAYS_INLINE ArrayData(VectorSpecialInitializerIndexesFromZero::IEnum x, size_t offset)
        : data0(x, offset), data1(x, offset + secondOffset)
    {
    }

    template <typename Op, typename... Args> Vc_ALWAYS_INLINE void call(Args &&... args)
    {
        data0.template call<Op>(Split::template lo<Op>(std::forward<Args>(args))...);
        data1.template call<Op>(Split::template hi<Op>(std::forward<Args>(args))...);
    }
    template <typename Op, typename... Args> Vc_ALWAYS_INLINE void call(Args &&... args) const
    {
        data0.template call<Op>(Split::template lo<Op>(std::forward<Args>(args))...);
        data1.template call<Op>(Split::template hi<Op>(std::forward<Args>(args))...);
    }

    template <typename Op, typename R, typename... Args>
    Vc_ALWAYS_INLINE void callReturningMember(R &&r, Args &&... args) const
    {
        data0.template callReturningMember<Op>(Split::template lo<Op>(std::forward<R>(r)), Split::template lo<Op>(std::forward<Args>(args))...);
        data1.template callReturningMember<Op>(Split::template hi<Op>(std::forward<R>(r)), Split::template hi<Op>(std::forward<Args>(args))...);
    }

    template <typename U, typename Flags>
    Vc_ALWAYS_INLINE void store(U *x, Flags f)
    {
        data0.store(x, f);
        data1.store(x + secondOffset, f);
    }

    template <typename F, typename... Args> inline void assign(F function, Args &&... args)
    {
        data0.assign(function, Split::lo(std::forward<Args>(args))...);
        data1.assign(function, Split::hi(std::forward<Args>(args))...);
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

    template<typename W, std::size_t M> Vc_ALWAYS_INLINE void cast(const ArrayData<W, M> &x) {
        data0.cast(Split::lo(x));
        data1.cast(Split::hi(x));
    }
    template <typename W>
    Vc_ALWAYS_INLINE void cast(
        const ArrayData<W, 1> &x,
        enable_if<std::is_same<W, W>::value &&  // hack to make the enable_if depend on W
                  N == 2> = nullarg)
    {
        data0.d = simd_cast<V, 0>(x.d);
        data1.d = simd_cast<V, 1>(x.d);
    }
    template <typename W>
    Vc_ALWAYS_INLINE void cast(
        const ArrayData<W, 1> &x,
        enable_if<std::is_same<W, W>::value &&  // hack to make the enable_if depend on W
                  N == 4> = nullarg)
    {
        data0.data0.d = simd_cast<V, 0>(x.d);
        data0.data1.d = simd_cast<V, 1>(x.d);
        data1.data0.d = simd_cast<V, 2>(x.d);
        data1.data1.d = simd_cast<V, 3>(x.d);
    }
    template <typename W>
    Vc_ALWAYS_INLINE void cast(
        const ArrayData<W, 1> &x,
        enable_if<std::is_same<W, W>::value &&  // hack to make the enable_if depend on W
                  N == 8> = nullarg)
    {
        data0.data0.data0.d = simd_cast<V, 0>(x.d);
        data0.data0.data1.d = simd_cast<V, 1>(x.d);
        data0.data1.data0.d = simd_cast<V, 2>(x.d);
        data0.data1.data1.d = simd_cast<V, 3>(x.d);
        data1.data0.data0.d = simd_cast<V, 4>(x.d);
        data1.data0.data1.d = simd_cast<V, 5>(x.d);
        data1.data1.data0.d = simd_cast<V, 6>(x.d);
        data1.data1.data1.d = simd_cast<V, 7>(x.d);
    }

    // not implemented - only here for is_gather_signature to find us
    value_type operator[](std::size_t);
    value_type operator[](std::size_t) const;
};/*}}}*/

//                                         MaskData<M, 1>
template<typename M> struct MaskData<M, 1>/*{{{*/
{
    using mask_type = M;

    M *begin() { return &d; }
    const M *cbegin() const { return &d; }

    M *end() { return &d + 1; }
    const M *cend() { return &d + 1; }

    MaskData() = default;
    Vc_ALWAYS_INLINE MaskData(const M &x) : d(x) {}
    Vc_ALWAYS_INLINE MaskData(bool b) : d(b) {}

    template <typename M2>
    Vc_ALWAYS_INLINE MaskData(const M2 &x)
        : d(x)
    {
    }

    template <typename M2>
    Vc_ALWAYS_INLINE MaskData(const MaskData<M2, 2> &x)
        : d(simd_cast<mask_type>(x.data0.d, x.data1.d))
    {
    }

    template <typename M2> Vc_ALWAYS_INLINE MaskData(const MaskData<M2, 1> &x) : d(x.d)
    {
    }

    Vc_ALWAYS_INLINE Vc_PURE bool isFull() const { return d.isFull(); }
    Vc_ALWAYS_INLINE Vc_PURE bool isEmpty() const { return d.isEmpty(); }

    template<typename V, typename F>
    Vc_ALWAYS_INLINE void assign(const ArrayData<V, 1> &lhs, const ArrayData<V, 1> &rhs, F function) {
        d = (lhs.d.*function)(rhs.d);
    }

    template <typename V, typename F>
    Vc_ALWAYS_INLINE void assign(const ArrayData<V, 1> &operand, F function)
    { d = (operand.d.*function)(); }

    template <typename V, typename F>
    Vc_ALWAYS_INLINE void assign(const MaskData<V, 1> &operand, F function)
    { d = (operand.d.*function)(); }

    template <typename ReductionFunctor>
    Vc_ALWAYS_INLINE decltype(ReductionFunctor::processLeaf(std::declval<const M &>()))
        reduce() const
    {
        return ReductionFunctor::processLeaf(d);
    }

    template <typename Reduce, typename F, typename... Masks>
    Vc_ALWAYS_INLINE auto apply(Reduce, F f, Masks... masks)
        const -> decltype(f(std::declval<const mask_type &>(),
                            std::declval<const typename Masks::mask_type &>()...))
    { return f(d, masks.d...); }

//private:
    M d;
};/*}}}*/

//                                         MaskData<M, N>
template<typename M, std::size_t N> struct MaskData/*{{{*/
{
    static_assert(N != 0, "error N must be nonzero!");
    using mask_type = M;

    static constexpr std::size_t secondOffset = N / 2 * mask_type::Size;
    using Split = Common::Split<secondOffset>;

//private:
    MaskData<M, N / 2> data0;
    MaskData<M, N / 2> data1;

//public:
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

    Vc_ALWAYS_INLINE MaskData(bool b) : data0(b), data1(b)
    {
    }

    Vc_ALWAYS_INLINE MaskData(const M &x) : data0(x), data1(x)
    {
    }

    template <typename M2>
    Vc_ALWAYS_INLINE MaskData(const M2 &x)
        : data0(Split::lo(x)), data1(Split::hi(x))
    {
    }

    template <typename M2>
    Vc_ALWAYS_INLINE MaskData(
        const MaskData<M2, 1> &x,
        enable_if<std::is_same<M2, M2>::value &&  // hack to make the enable_if depend on M2
                  N == 2> = nullarg)
        : data0(simd_cast<mask_type, 0>(x.d)), data1(simd_cast<mask_type, 1>(x.d))
    {
    }

    template <typename M2>
    Vc_ALWAYS_INLINE MaskData(
        const MaskData<M2, 1> &x,
        enable_if<std::is_same<M2, M2>::value &&  // hack to make the enable_if depend on M2
                  N == 4> = nullarg)
    {
        data0.data0.d = simd_cast<mask_type, 0>(x.d);
        data0.data1.d = simd_cast<mask_type, 1>(x.d);
        data1.data0.d = simd_cast<mask_type, 2>(x.d);
        data1.data1.d = simd_cast<mask_type, 3>(x.d);
    }

    template<typename M2, std::size_t K>
    Vc_ALWAYS_INLINE MaskData(const MaskData<M2, K> &x) : data0(x.data0), data1(x.data1)
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

    Vc_ALWAYS_INLINE Vc_PURE unsigned int count() const
    {
        return reduce<Reductions::Count>();
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

    template <typename V, typename F>
    Vc_ALWAYS_INLINE void assign(const MaskData<V, N> &operand, F function)
    {
        data0.assign(operand.data0, function);
        data1.assign(operand.data1, function);
    }

    template <typename ReductionFunctor>
    Vc_ALWAYS_INLINE auto reduce() const -> decltype(ReductionFunctor()(
        data0.template reduce<ReductionFunctor>(),
        data1.template reduce<ReductionFunctor>()))
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
};/*}}}*/

}  // namespace Common
}  // namespace Vc

#include "undomacros.h"

#endif // VC_COMMON_SIMD_ARRAY_DATA_H

// vim: foldmethod=marker
