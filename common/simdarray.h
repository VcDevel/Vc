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

#ifndef VC_COMMON_SIMDARRAY_H_
#define VC_COMMON_SIMDARRAY_H_

//#define Vc_DEBUG_SIMD_CAST 1
//#define Vc_DEBUG_SORTED 1
#if defined Vc_DEBUG_SIMD_CAST || defined Vc_DEBUG_SORTED
#include <Vc/IO>
#endif

#include <array>

#include "writemaskedvector.h"
#include "simdarrayhelper.h"
#include "simdmaskarray.h"
#include "utility.h"
#include "interleave.h"
#include "indexsequence.h"
#include "transpose.h"
#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
// internal namespace (min/max helper) {{{1
namespace internal
{
#define Vc_DECLARE_BINARY_FUNCTION__(name__)                                             \
    template <typename T, std::size_t N, typename V, std::size_t M>                      \
    SimdArray<T, N, V, M> Vc_INTRINSIC_L Vc_PURE_L                                       \
        name__(const SimdArray<T, N, V, M> &l, const SimdArray<T, N, V, M> &r)           \
            Vc_INTRINSIC_R Vc_PURE_R;                                                    \
    template <typename T, std::size_t N, typename V>                                     \
    SimdArray<T, N, V, N> Vc_INTRINSIC_L Vc_PURE_L                                       \
        name__(const SimdArray<T, N, V, N> &l, const SimdArray<T, N, V, N> &r)           \
            Vc_INTRINSIC_R Vc_PURE_R;
Vc_DECLARE_BINARY_FUNCTION__(min)
Vc_DECLARE_BINARY_FUNCTION__(max)
#undef Vc_DECLARE_BINARY_FUNCTION__

template <typename T> Vc_INTRINSIC Vc_PURE T min(const T &l, const T &r)
{
    T x = l;
    where(r < l) | x = r;
    return x;
}
template <typename T> Vc_INTRINSIC Vc_PURE T max(const T &l, const T &r)
{
    T x = l;
    where(r > l) | x = r;
    return x;
}
template <typename T> T Vc_INTRINSIC Vc_PURE product_helper__(const T &l, const T &r) { return l * r; }
template <typename T> T Vc_INTRINSIC Vc_PURE sum_helper__(const T &l, const T &r) { return l + r; }
}  // namespace internal

// SimdArray class {{{1
/// \addtogroup SimdArray
/// @{

// atomic SimdArray {{{1
#define Vc_CURRENT_CLASS_NAME SimdArray
/**\internal
 * Specialization of `SimdArray<T, N, VectorType, VectorSize>` for the case where `N ==
 * VectorSize`.
 *
 * This is specialized for implementation purposes: Since the general implementation uses
 * two SimdArray data members it recurses over different SimdArray instantiations. The
 * recursion is ended by this specialization, which has a single \p VectorType_ data
 * member to which all functions are forwarded more or less directly.
 */
template <typename T, std::size_t N, typename VectorType_>
class alignas(
    ((Common::nextPowerOfTwo(N) * (sizeof(VectorType_) / VectorType_::size()) - 1) & 127) +
    1) SimdArray<T, N, VectorType_, N>
{
    static_assert(std::is_same<T, double>::value || std::is_same<T, float>::value ||
                      std::is_same<T, int32_t>::value ||
                      std::is_same<T, uint32_t>::value ||
                      std::is_same<T, int16_t>::value ||
                      std::is_same<T, uint16_t>::value,
                  "SimdArray<T, N> may only be used with T = { double, float, int32_t, uint32_t, "
                  "int16_t, uint16_t }");

public:
    using VectorType = VectorType_;
    using vector_type = VectorType;
    using storage_type = vector_type;
    using vectorentry_type = typename vector_type::VectorEntryType;
    using value_type = T;
    using mask_type = SimdMaskArray<T, N, vector_type>;
    using index_type = SimdArray<int, N>;
    static constexpr std::size_t size() { return N; }
    using Mask = mask_type;
    using MaskType = Mask;
    using MaskArgument = const MaskType &;
    using VectorEntryType = vectorentry_type;
    using EntryType = value_type;
    using IndexType = index_type;
    using AsArg = const SimdArray &;
    static constexpr std::size_t Size = size();
    static constexpr std::size_t MemoryAlignment = storage_type::MemoryAlignment;

    // zero init
    Vc_INTRINSIC SimdArray() = default;

    // default copy ctor/operator
    Vc_INTRINSIC SimdArray(const SimdArray &) = default;
    Vc_INTRINSIC SimdArray(SimdArray &&) = default;
    Vc_INTRINSIC SimdArray &operator=(const SimdArray &) = default;

    // broadcast
    Vc_INTRINSIC SimdArray(const value_type &a) : data(a) {}
    Vc_INTRINSIC SimdArray(value_type &a) : data(a) {}
    Vc_INTRINSIC SimdArray(value_type &&a) : data(a) {}
    template <
        typename U,
        typename = enable_if<std::is_same<U, int>::value && !std::is_same<int, value_type>::value>>
    Vc_INTRINSIC SimdArray(U a)
        : SimdArray(static_cast<value_type>(a))
    {
    }

    // implicit casts
    template <typename U, typename V>
    Vc_INTRINSIC SimdArray(const SimdArray<U, N, V> &x, enable_if<N == V::size()> = nullarg)
        : data(simd_cast<vector_type>(internal_data(x)))
    {
    }
    template <typename U, typename V>
    Vc_INTRINSIC SimdArray(const SimdArray<U, N, V> &x,
                            enable_if<(N > V::size() && N <= 2 * V::size())> = nullarg)
        : data(simd_cast<vector_type>(internal_data(internal_data0(x)), internal_data(internal_data1(x))))
    {
    }
    template <typename U, typename V>
    Vc_INTRINSIC SimdArray(const SimdArray<U, N, V> &x,
                            enable_if<(N > 2 * V::size() && N <= 4 * V::size())> = nullarg)
        : data(simd_cast<vector_type>(internal_data(internal_data0(internal_data0(x))),
                                      internal_data(internal_data1(internal_data0(x))),
                                      internal_data(internal_data0(internal_data1(x))),
                                      internal_data(internal_data1(internal_data1(x)))))
    {
    }

    template <typename V, std::size_t Pieces, std::size_t Index>
    Vc_INTRINSIC SimdArray(Common::Segment<V, Pieces, Index> &&x)
        : data(simd_cast<vector_type, Index>(x.data))
    {
    }

    Vc_INTRINSIC SimdArray(const std::initializer_list<value_type> &init)
        : data(init.begin(), Vc::Unaligned)
    {
#if defined Vc_CXX14 && 0  // doesn't compile yet
        static_assert(init.size() == size(), "The initializer_list argument to "
                                             "SimdArray<T, N> must contain exactly N "
                                             "values.");
#else
        Vc_ASSERT(init.size() == size());
#endif
    }

    // implicit conversion from underlying vector_type
    template <
        typename V,
        typename = enable_if<Traits::is_simd_vector<V>::value && !Traits::isSimdArray<V>::value>>
    explicit Vc_INTRINSIC SimdArray(const V &x)
        : data(simd_cast<vector_type>(x))
    {
    }

    // implicit conversion to Vector<U, AnyAbi> for if Vector<U, AnyAbi>::size() == N and
    // T implicitly convertible to U
    template <typename V,
              typename = enable_if<
                  Traits::is_simd_vector<V>::value && !Traits::isSimdArray<V>::value &&
                  std::is_convertible<T, typename V::EntryType>::value && V::size() == N>>
    Vc_INTRINSIC operator V() const
    {
        return simd_cast<V>(*this);
    }

#include "gatherinterface.h"

    // forward all remaining ctors
    template <typename... Args,
              typename = enable_if<!Traits::is_cast_arguments<Args...>::value &&
                                   !Traits::is_gather_signature<Args...>::value &&
                                   !Traits::is_initializer_list<Args...>::value>>
    explicit Vc_INTRINSIC SimdArray(Args &&... args)
        : data(std::forward<Args>(args)...)
    {
    }

    template <std::size_t Offset>
    explicit Vc_INTRINSIC SimdArray(
        Common::AddOffset<VectorSpecialInitializerIndexesFromZero, Offset>)
        : data(Vc::IndexesFromZero)
    {
        data += value_type(Offset);
    }

    Vc_INTRINSIC void setZero() { data.setZero(); }
    Vc_INTRINSIC void setZero(mask_type k) { data.setZero(internal_data(k)); }
    Vc_INTRINSIC void setZeroInverted() { data.setZeroInverted(); }
    Vc_INTRINSIC void setZeroInverted(mask_type k) { data.setZeroInverted(internal_data(k)); }

    // internal: execute specified Operation
    template <typename Op, typename... Args>
    static Vc_INTRINSIC SimdArray fromOperation(Op op, Args &&... args)
    {
        SimdArray r;
        op(r.data, Common::actual_value(op, std::forward<Args>(args))...);
        return r;
    }

    static Vc_INTRINSIC SimdArray Zero()
    {
        return SimdArray(Vc::Zero);
    }
    static Vc_INTRINSIC SimdArray One()
    {
        return SimdArray(Vc::One);
    }
    static Vc_INTRINSIC SimdArray IndexesFromZero()
    {
        return SimdArray(Vc::IndexesFromZero);
    }
    static Vc_INTRINSIC SimdArray Random()
    {
        return fromOperation(Common::Operations::random());
    }

    template <typename... Args> Vc_INTRINSIC void load(Args &&... args)
    {
        data.load(std::forward<Args>(args)...);
    }

    template <typename... Args> Vc_INTRINSIC void store(Args &&... args) const
    {
        data.store(std::forward<Args>(args)...);
    }

    Vc_INTRINSIC mask_type operator!() const
    {
        return {!data};
    }

    Vc_INTRINSIC SimdArray operator-() const
    {
        return {-data};
    }

    Vc_INTRINSIC SimdArray operator~() const
    {
        return {~data};
    }

    template <typename U,
              typename = enable_if<std::is_integral<T>::value && std::is_integral<U>::value>>
    Vc_INTRINSIC Vc_CONST SimdArray operator<<(U x) const
    {
        return {data << x};
    }
    template <typename U,
              typename = enable_if<std::is_integral<T>::value && std::is_integral<U>::value>>
    Vc_INTRINSIC SimdArray &operator<<=(U x)
    {
        data <<= x;
        return *this;
    }
    template <typename U,
              typename = enable_if<std::is_integral<T>::value && std::is_integral<U>::value>>
    Vc_INTRINSIC Vc_CONST SimdArray operator>>(U x) const
    {
        return {data >> x};
    }
    template <typename U,
              typename = enable_if<std::is_integral<T>::value && std::is_integral<U>::value>>
    Vc_INTRINSIC SimdArray &operator>>=(U x)
    {
        data >>= x;
        return *this;
    }

#define Vc_BINARY_OPERATOR_(op)                                                          \
    Vc_INTRINSIC Vc_CONST SimdArray operator op(const SimdArray &rhs) const              \
    {                                                                                    \
        return {data op rhs.data};                                                       \
    }                                                                                    \
    Vc_INTRINSIC SimdArray &operator op##=(const SimdArray &rhs)                         \
    {                                                                                    \
        data op## = rhs.data;                                                            \
        return *this;                                                                    \
    }
    Vc_ALL_ARITHMETICS(Vc_BINARY_OPERATOR_)
    Vc_ALL_BINARY(Vc_BINARY_OPERATOR_)
    Vc_ALL_SHIFTS(Vc_BINARY_OPERATOR_)
#undef Vc_BINARY_OPERATOR_

#define Vc_COMPARES(op)                                                                  \
    Vc_INTRINSIC mask_type operator op(const SimdArray &rhs) const                       \
    {                                                                                    \
        return {data op rhs.data};                                                       \
    }
    Vc_ALL_COMPARES(Vc_COMPARES)
#undef Vc_COMPARES

    Vc_INTRINSIC decltype(std::declval<vector_type &>()[0]) operator[](std::size_t i)
    {
        return data[i];
    }
    Vc_INTRINSIC value_type operator[](std::size_t i) const { return data[i]; }

    Vc_INTRINSIC Common::WriteMaskedVector<SimdArray, mask_type> operator()(const mask_type &k)
    {
        return {this, k};
    }

    Vc_INTRINSIC void assign(const SimdArray &v, const mask_type &k)
    {
        data.assign(v.data, internal_data(k));
    }

    // reductions ////////////////////////////////////////////////////////
#define Vc_REDUCTION_FUNCTION__(name__)                                                  \
    Vc_INTRINSIC Vc_PURE value_type name__() const { return data.name__(); }             \
                                                                                         \
    Vc_INTRINSIC Vc_PURE value_type name__(mask_type mask) const                         \
    {                                                                                    \
        return data.name__(internal_data(mask));                                         \
    }
    Vc_REDUCTION_FUNCTION__(min)
    Vc_REDUCTION_FUNCTION__(max)
    Vc_REDUCTION_FUNCTION__(product)
    Vc_REDUCTION_FUNCTION__(sum)
#undef Vc_REDUCTION_FUNCTION__
    Vc_INTRINSIC Vc_PURE SimdArray partialSum() const { return data.partialSum(); }

    Vc_INTRINSIC void fusedMultiplyAdd(const SimdArray &factor, const SimdArray &summand)
    {
        data.fusedMultiplyAdd(internal_data(factor), internal_data(summand));
    }

    template <typename F> Vc_INTRINSIC SimdArray apply(F &&f) const
    {
        return {data.apply(std::forward<F>(f))};
    }
    template <typename F> Vc_INTRINSIC SimdArray apply(F &&f, const mask_type &k) const
    {
        return {data.apply(std::forward<F>(f), k)};
    }

    Vc_INTRINSIC SimdArray shifted(int amount) const
    {
        return {data.shifted(amount)};
    }

    template <std::size_t NN>
    Vc_INTRINSIC SimdArray shifted(int amount, const SimdArray<value_type, NN> &shiftIn)
        const
    {
        return {data.shifted(amount, simd_cast<VectorType>(shiftIn))};
    }

    Vc_INTRINSIC SimdArray rotated(int amount) const
    {
        return {data.rotated(amount)};
    }

    Vc_INTRINSIC SimdArray interleaveLow(SimdArray x) const
    {
        return {data.interleaveLow(x.data)};
    }
    Vc_INTRINSIC SimdArray interleaveHigh(SimdArray x) const
    {
        return {data.interleaveHigh(x.data)};
    }

    Vc_INTRINSIC SimdArray reversed() const
    {
        return {data.reversed()};
    }

    Vc_INTRINSIC SimdArray sorted() const
    {
        return {data.sorted()};
    }

    template <typename G> static Vc_INTRINSIC SimdArray generate(const G &gen)
    {
        return {VectorType::generate(gen)};
    }

    friend VectorType &internal_data<>(SimdArray &x);
    friend const VectorType &internal_data<>(const SimdArray &x);

    /// \internal
    Vc_INTRINSIC SimdArray(VectorType &&x) : data(std::move(x)) {}
private:
    storage_type data;
};
template <typename T, std::size_t N, typename VectorType> constexpr std::size_t SimdArray<T, N, VectorType, N>::Size;
template <typename T, std::size_t N, typename VectorType>
constexpr std::size_t SimdArray<T, N, VectorType, N>::MemoryAlignment;
template <typename T, std::size_t N, typename VectorType>
Vc_INTRINSIC VectorType &internal_data(SimdArray<T, N, VectorType, N> &x)
{
    return x.data;
}
template <typename T, std::size_t N, typename VectorType>
Vc_INTRINSIC const VectorType &internal_data(const SimdArray<T, N, VectorType, N> &x)
{
    return x.data;
}

// gatherImplementation {{{2
template <typename T, std::size_t N, typename VectorType>
template <typename MT, typename IT>
inline void SimdArray<T, N, VectorType, N>::gatherImplementation(const MT *mem,
                                                                 IT &&indexes)
{
    data.gather(mem, std::forward<IT>(indexes));
}
template <typename T, std::size_t N, typename VectorType>
template <typename MT, typename IT>
inline void SimdArray<T, N, VectorType, N>::gatherImplementation(const MT *mem,
                                                                 IT &&indexes,
                                                                 MaskArgument mask)
{
    data.gather(mem, std::forward<IT>(indexes), mask);
}

// generic SimdArray {{{1
template <typename T, std::size_t N, typename VectorType, std::size_t>
class alignas(
    ((Common::nextPowerOfTwo(N) * (sizeof(VectorType) / VectorType::size()) - 1) & 127) +
    1) SimdArray
{
    static_assert(std::is_same<T,   double>::value ||
                  std::is_same<T,    float>::value ||
                  std::is_same<T,  int32_t>::value ||
                  std::is_same<T, uint32_t>::value ||
                  std::is_same<T,  int16_t>::value ||
                  std::is_same<T, uint16_t>::value, "SimdArray<T, N> may only be used with T = { double, float, int32_t, uint32_t, int16_t, uint16_t }");

    using my_traits = SimdArrayTraits<T, N>;
    static constexpr std::size_t N0 = my_traits::N0;
    static constexpr std::size_t N1 = my_traits::N1;
    using Split = Common::Split<N0>;

public:
    using storage_type0 = typename my_traits::storage_type0;
    using storage_type1 = typename my_traits::storage_type1;
    static_assert(storage_type0::size() == N0, "");

    using vector_type = VectorType;
    using vectorentry_type = typename storage_type0::vectorentry_type;
    typedef vectorentry_type alias_type Vc_MAY_ALIAS;
    using value_type = T;
    using mask_type = SimdMaskArray<T, N, vector_type>;
    using index_type = SimdArray<int, N>;
    static constexpr std::size_t size() { return N; }
    using Mask = mask_type;
    using MaskType = Mask;
    using MaskArgument = const MaskType &;
    using VectorEntryType = vectorentry_type;
    using EntryType = value_type;
    using IndexType = index_type;
    using AsArg = const SimdArray &;
    static constexpr std::size_t Size = size();
    static constexpr std::size_t MemoryAlignment =
        storage_type0::MemoryAlignment > storage_type1::MemoryAlignment
            ? storage_type0::MemoryAlignment
            : storage_type1::MemoryAlignment;

    //////////////////// constructors //////////////////

    // zero init
    SimdArray() = default;

    // default copy ctor/operator
    SimdArray(const SimdArray &) = default;
    SimdArray(SimdArray &&) = default;
    SimdArray &operator=(const SimdArray &) = default;

    // broadcast
    Vc_INTRINSIC SimdArray(value_type a) : data0(a), data1(a) {}
    template <
        typename U,
        typename = enable_if<std::is_same<U, int>::value && !std::is_same<int, value_type>::value>>
    SimdArray(U a)
        : SimdArray(static_cast<value_type>(a))
    {
    }

    // load ctor
    template <typename U,
              typename Flags = DefaultLoadTag,
              typename = enable_if<Traits::is_load_store_flag<Flags>::value>>
    explicit Vc_INTRINSIC SimdArray(const U *mem, Flags f = Flags())
        : data0(mem, f), data1(mem + storage_type0::size(), f)
    {
    }

    // initializer list
    Vc_INTRINSIC SimdArray(const std::initializer_list<value_type> &init)
        : data0(init.begin(), Vc::Unaligned)
        , data1(init.begin() + storage_type0::size(), Vc::Unaligned)
    {
#if defined Vc_CXX14 && 0  // doesn't compile yet
        static_assert(init.size() == size(), "The initializer_list argument to "
                                             "SimdArray<T, N> must contain exactly N "
                                             "values.");
#else
        Vc_ASSERT(init.size() == size());
#endif
    }

#include "gatherinterface.h"

    // forward all remaining ctors
    template <typename... Args,
              typename = enable_if<!Traits::is_cast_arguments<Args...>::value &&
                                   !Traits::is_initializer_list<Args...>::value &&
                                   !Traits::is_gather_signature<Args...>::value &&
                                   !Traits::is_load_arguments<Args...>::value>>
    explicit Vc_INTRINSIC SimdArray(Args &&... args)
        : data0(Split::lo(args)...)  // no forward here - it could move and thus
                                     // break the next line
        , data1(Split::hi(std::forward<Args>(args))...)
    {
    }

    // explicit casts
    template <typename V>
    Vc_INTRINSIC explicit SimdArray(
        V &&x,
        enable_if<(Traits::is_simd_vector<V>::value && Traits::simd_vector_size<V>::value == N &&
                   !(std::is_convertible<Traits::entry_type_of<V>, T>::value &&
                     Traits::isSimdArray<V>::value))> = nullarg)
        : data0(Split::lo(x)), data1(Split::hi(x))
    {
    }

    // implicit casts
    template <typename V>
    Vc_INTRINSIC SimdArray(
        V &&x,
        enable_if<(Traits::isSimdArray<V>::value && Traits::simd_vector_size<V>::value == N &&
                   std::is_convertible<Traits::entry_type_of<V>, T>::value)> = nullarg)
        : data0(Split::lo(x)), data1(Split::hi(x))
    {
    }

    // implicit conversion to Vector<U, AnyAbi> for if Vector<U, AnyAbi>::size() == N and
    // T implicitly convertible to U
    template <typename V,
              typename = enable_if<
                  Traits::is_simd_vector<V>::value && !Traits::isSimdArray<V>::value &&
                  std::is_convertible<T, typename V::EntryType>::value && V::size() == N>>
    operator V() const
    {
        return simd_cast<V>(*this);
    }

    //////////////////// other functions ///////////////

    Vc_INTRINSIC void setZero()
    {
        data0.setZero();
        data1.setZero();
    }
    Vc_INTRINSIC void setZero(const mask_type &k)
    {
        data0.setZero(Split::lo(k));
        data1.setZero(Split::hi(k));
    }
    Vc_INTRINSIC void setZeroInverted()
    {
        data0.setZeroInverted();
        data1.setZeroInverted();
    }
    Vc_INTRINSIC void setZeroInverted(const mask_type &k)
    {
        data0.setZeroInverted(Split::lo(k));
        data1.setZeroInverted(Split::hi(k));
    }

    // internal: execute specified Operation
    template <typename Op, typename... Args>
    static Vc_INTRINSIC SimdArray fromOperation(Op op, Args &&... args)
    {
        SimdArray r = {
            storage_type0::fromOperation(op, Split::lo(args)...),  // no forward here - it
                                                                   // could move and thus
                                                                   // break the next line
            storage_type1::fromOperation(op, Split::hi(std::forward<Args>(args))...)};
        return r;
    }

    static Vc_INTRINSIC SimdArray Zero()
    {
        return SimdArray(Vc::Zero);
    }
    static Vc_INTRINSIC SimdArray One()
    {
        return SimdArray(Vc::One);
    }
    static Vc_INTRINSIC SimdArray IndexesFromZero()
    {
        return SimdArray(Vc::IndexesFromZero);
    }
    static Vc_INTRINSIC SimdArray Random()
    {
        return fromOperation(Common::Operations::random());
    }

    template <typename U, typename... Args> Vc_INTRINSIC void load(const U *mem, Args &&... args)
    {
        data0.load(mem, Split::lo(args)...);  // no forward here - it could move and thus
                                              // break the next line
        data1.load(mem + storage_type0::size(), Split::hi(std::forward<Args>(args))...);
    }

    template <typename U, typename... Args> Vc_INTRINSIC void store(U *mem, Args &&... args) const
    {
        data0.store(mem, Split::lo(args)...);  // no forward here - it could move and thus
                                               // break the next line
        data1.store(mem + storage_type0::size(), Split::hi(std::forward<Args>(args))...);
    }

    Vc_INTRINSIC mask_type operator!() const
    {
        return {!data0, !data1};
    }

    Vc_INTRINSIC SimdArray operator-() const
    {
        return {-data0, -data1};
    }

    Vc_INTRINSIC SimdArray operator~() const
    {
        return {~data0, ~data1};
    }

    // left/right shift operators {{{2
    template <typename U,
              typename = enable_if<std::is_integral<T>::value && std::is_integral<U>::value>>
    Vc_INTRINSIC Vc_CONST SimdArray operator<<(U x) const
    {
        return {data0 << x, data1 << x};
    }
    template <typename U,
              typename = enable_if<std::is_integral<T>::value && std::is_integral<U>::value>>
    Vc_INTRINSIC SimdArray &operator<<=(U x)
    {
        data0 <<= x;
        data1 <<= x;
        return *this;
    }
    template <typename U,
              typename = enable_if<std::is_integral<T>::value && std::is_integral<U>::value>>
    Vc_INTRINSIC Vc_CONST SimdArray operator>>(U x) const
    {
        return {data0 >> x, data1 >> x};
    }
    template <typename U,
              typename = enable_if<std::is_integral<T>::value && std::is_integral<U>::value>>
    Vc_INTRINSIC SimdArray &operator>>=(U x)
    {
        data0 >>= x;
        data1 >>= x;
        return *this;
    }

    // binary operators {{{2
#define Vc_BINARY_OPERATOR_(op)                                                          \
    Vc_INTRINSIC Vc_CONST SimdArray operator op(const SimdArray &rhs) const              \
    {                                                                                    \
        return {data0 op rhs.data0, data1 op rhs.data1};                                 \
    }                                                                                    \
    Vc_INTRINSIC SimdArray &operator op##=(const SimdArray &rhs)                         \
    {                                                                                    \
        data0 op## = rhs.data0;                                                          \
        data1 op## = rhs.data1;                                                          \
        return *this;                                                                    \
    }
    Vc_ALL_ARITHMETICS(Vc_BINARY_OPERATOR_)
    Vc_ALL_BINARY(Vc_BINARY_OPERATOR_)
    Vc_ALL_SHIFTS(Vc_BINARY_OPERATOR_)
#undef Vc_BINARY_OPERATOR_

#define Vc_COMPARES(op)                                                                  \
    Vc_INTRINSIC mask_type operator op(const SimdArray &rhs) const                       \
    {                                                                                    \
        return {data0 op rhs.data0, data1 op rhs.data1};                                 \
    }
    Vc_ALL_COMPARES(Vc_COMPARES)
#undef Vc_COMPARES

    // operator[] {{{2
    Vc_INTRINSIC value_type operator[](std::size_t i) const
    {
        const auto tmp = reinterpret_cast<const alias_type *>(this);
        return tmp[i];
    }

    Vc_INTRINSIC alias_type &operator[](std::size_t i)
    {
        auto tmp = reinterpret_cast<alias_type *>(this);
        return tmp[i];
    }

    Vc_INTRINSIC Common::WriteMaskedVector<SimdArray, mask_type> operator()(const mask_type &k) //{{{2
    {
        return {this, k};
    }

    Vc_INTRINSIC void assign(const SimdArray &v, const mask_type &k) //{{{2
    {
        data0.assign(v.data0, internal_data0(k));
        data1.assign(v.data1, internal_data1(k));
    }

    // reductions {{{2
#define Vc_REDUCTION_FUNCTION__(name__, binary_fun__)                                    \
    template <typename ForSfinae = void>                                                 \
    Vc_INTRINSIC enable_if<std::is_same<ForSfinae, void>::value &&                       \
                               storage_type0::size() == storage_type1::size(),           \
                           value_type>                                                   \
    name__() const                                                                       \
    {                                                                                    \
        return binary_fun__(data0, data1).name__();                                      \
    }                                                                                    \
                                                                                         \
    template <typename ForSfinae = void>                                                 \
    Vc_INTRINSIC enable_if<std::is_same<ForSfinae, void>::value &&                       \
                               storage_type0::size() != storage_type1::size(),           \
                           value_type>                                                   \
    name__() const                                                                       \
    {                                                                                    \
        return binary_fun__(data0.name__(), data1.name__());                             \
    }                                                                                    \
                                                                                         \
    Vc_INTRINSIC value_type name__(const mask_type &mask) const                          \
    {                                                                                    \
        if (Vc_IS_UNLIKELY(Split::lo(mask).isEmpty())) {                                 \
            return data1.name__(Split::hi(mask));                                        \
        } else if (Vc_IS_UNLIKELY(Split::hi(mask).isEmpty())) {                          \
            return data0.name__(Split::lo(mask));                                        \
        } else {                                                                         \
            return binary_fun__(data0.name__(Split::lo(mask)),                           \
                                data1.name__(Split::hi(mask)));                          \
        }                                                                                \
    }
    Vc_REDUCTION_FUNCTION__(min, Vc::internal::min)
    Vc_REDUCTION_FUNCTION__(max, Vc::internal::max)
    Vc_REDUCTION_FUNCTION__(product, internal::product_helper__)
    Vc_REDUCTION_FUNCTION__(sum, internal::sum_helper__)
#undef Vc_REDUCTION_FUNCTION__
    Vc_INTRINSIC Vc_PURE SimdArray partialSum() const //{{{2
    {
        auto ps0 = data0.partialSum();
        auto tmp = data1;
        tmp[0] += ps0[data0.size() - 1];
        return {std::move(ps0), tmp.partialSum()};
    }

    void fusedMultiplyAdd(const SimdArray &factor, const SimdArray &summand) //{{{2
    {
        data0.fusedMultiplyAdd(Split::lo(factor), Split::lo(summand));
        data1.fusedMultiplyAdd(Split::hi(factor), Split::hi(summand));
    }

    // apply {{{2
    template <typename F> Vc_INTRINSIC SimdArray apply(F &&f) const
    {
        return {data0.apply(f), data1.apply(f)};
    }
    template <typename F> Vc_INTRINSIC SimdArray apply(F &&f, const mask_type &k) const
    {
        return {data0.apply(f, Split::lo(k)), data1.apply(f, Split::hi(k))};
    }

    // shifted {{{2
    inline SimdArray shifted(int amount) const
    {
        constexpr int SSize = Size;
        constexpr int SSize0 = storage_type0::Size;
        constexpr int SSize1 = storage_type1::Size;
        if (amount == 0) {
            return *this;
        }
        if (amount < 0) {
            if (amount > -SSize0) {
                return {data0.shifted(amount), data1.shifted(amount, data0)};
            }
            if (amount == -SSize0) {
                return {storage_type0::Zero(), simd_cast<storage_type1>(data0)};
            }
            if (amount < -SSize0) {
                return {storage_type0::Zero(), simd_cast<storage_type1>(data0.shifted(
                                                   amount + SSize0))};
            }
            return Zero();
        } else {
            if (amount >= SSize) {
                return Zero();
            } else if (amount >= SSize0) {
                return {
                    simd_cast<storage_type0>(data1).shifted(amount - SSize0),
                    storage_type1::Zero()};
            } else if (amount >= SSize1) {
                return {data0.shifted(amount, data1), storage_type1::Zero()};
            } else {
                return {data0.shifted(amount, data1), data1.shifted(amount)};
            }
        }
    }

    template <std::size_t NN>
    inline enable_if<
        !(std::is_same<storage_type0, storage_type1>::value &&  // not bisectable
          N == NN),
        SimdArray>
        shifted(int amount, const SimdArray<value_type, NN> &shiftIn) const
    {
        constexpr int SSize = Size;
        if (amount < 0) {
            return SimdArray::generate([&](int i) -> value_type {
                i += amount;
                if (i >= 0) {
                    return operator[](i);
                } else if (i >= -SSize) {
                    return shiftIn[i + SSize];
                }
                return 0;
            });
        }
        return SimdArray::generate([&](int i) -> value_type {
            i += amount;
            if (i < SSize) {
                return operator[](i);
            } else if (i < 2 * SSize) {
                return shiftIn[i - SSize];
            }
            return 0;
        });
    }

    template <std::size_t NN>
    inline
        enable_if<(std::is_same<storage_type0, storage_type1>::value &&  // bisectable
                   N == NN),
                  SimdArray>
            shifted(int amount, const SimdArray<value_type, NN> &shiftIn) const
    {
        constexpr int SSize = Size;
        if (amount < 0) {
            if (amount > -static_cast<int>(storage_type0::Size)) {
                return {data0.shifted(amount, internal_data1(shiftIn)),
                        data1.shifted(amount, data0)};
            }
            if (amount == -static_cast<int>(storage_type0::Size)) {
                return {storage_type0(internal_data1(shiftIn)), storage_type1(data0)};
            }
            if (amount > -SSize) {
                return {
                    internal_data1(shiftIn)
                        .shifted(amount + static_cast<int>(storage_type0::Size), internal_data0(shiftIn)),
                    data0.shifted(amount + static_cast<int>(storage_type0::Size), internal_data1(shiftIn))};
            }
            if (amount == -SSize) {
                return shiftIn;
            }
            if (amount > -2 * SSize) {
                return shiftIn.shifted(amount + SSize);
            }
        }
        if (amount == 0) {
            return *this;
        }
        if (amount < static_cast<int>(storage_type0::Size)) {
            return {data0.shifted(amount, data1),
                    data1.shifted(amount, internal_data0(shiftIn))};
        }
        if (amount == static_cast<int>(storage_type0::Size)) {
            return {storage_type0(data1), storage_type1(internal_data0(shiftIn))};
        }
        if (amount < SSize) {
            return {data1.shifted(amount - static_cast<int>(storage_type0::Size), internal_data0(shiftIn)),
                    internal_data0(shiftIn)
                        .shifted(amount - static_cast<int>(storage_type0::Size), internal_data1(shiftIn))};
        }
        if (amount == SSize) {
            return shiftIn;
        }
        if (amount < 2 * SSize) {
            return shiftIn.shifted(amount - SSize);
        }
        return Zero();
    }

    // rotated {{{2
    Vc_INTRINSIC SimdArray rotated(int amount) const
    {
        amount %= int(size());
        if (amount == 0) {
            return *this;
        } else if (amount < 0) {
            amount += size();
        }

        auto &&d0cvtd = simd_cast<storage_type1>(data0);
        auto &&d1cvtd = simd_cast<storage_type0>(data1);
        constexpr int size0 = storage_type0::size();
        constexpr int size1 = storage_type1::size();

        if (amount == size0 && std::is_same<storage_type0, storage_type1>::value) {
            return {std::move(d1cvtd), std::move(d0cvtd)};
        } else if (amount < size1) {
            return {data0.shifted(amount, d1cvtd), data1.shifted(amount, d0cvtd)};
        } else if (amount == size1) {
            return {data0.shifted(amount, d1cvtd), std::move(d0cvtd)};
        } else if (int(size()) - amount < size1) {
            return {data0.shifted(amount - int(size()), d1cvtd.shifted(size1 - size0)),
                    data1.shifted(amount - int(size()), data0.shifted(size0 - size1))};
        } else if (int(size()) - amount == size1) {
            return {data0.shifted(-size1, d1cvtd.shifted(size1 - size0)),
                    simd_cast<storage_type1>(data0.shifted(size0 - size1))};
        } else if (amount <= size0) {
            return {data0.shifted(size1, d1cvtd).shifted(amount - size1, data0),
                    simd_cast<storage_type1>(data0.shifted(amount - size1))};
        } else {
            return {data0.shifted(size1, d1cvtd).shifted(amount - size1, data0),
                    simd_cast<storage_type1>(data0.shifted(amount - size1, d1cvtd))};
        }
        return *this;
    }

    // interleaveLow/-High {{{2
    Vc_INTRINSIC SimdArray interleaveLow(const SimdArray &x) const
    {
        // return data0[0], x.data0[0], data0[1], x.data0[1], ...
        return {data0.interleaveLow(x.data0),
                simd_cast<storage_type1>(data0.interleaveHigh(x.data0))};
    }
    Vc_INTRINSIC SimdArray interleaveHigh(const SimdArray &x) const
    {
        return interleaveHighImpl(
            x,
            std::integral_constant<bool, storage_type0::Size == storage_type1::Size>());
    }

private:
    Vc_INTRINSIC SimdArray interleaveHighImpl(const SimdArray &x, std::true_type) const
    {
        return {data1.interleaveLow(x.data1), data1.interleaveHigh(x.data1)};
    }
    inline SimdArray interleaveHighImpl(const SimdArray &x, std::false_type) const
    {
        return {data0.interleaveHigh(x.data0)
                    .shifted(storage_type1::Size,
                             simd_cast<storage_type0>(data1.interleaveLow(x.data1))),
                data1.interleaveHigh(x.data1)};
    }

public:
    inline SimdArray reversed() const //{{{2
    {
        if (std::is_same<storage_type0, storage_type1>::value) {
            return {simd_cast<storage_type0>(data1).reversed(),
                    simd_cast<storage_type1>(data0).reversed()};
        } else {
            return {data0.shifted(storage_type1::Size, data1).reversed(),
                    simd_cast<storage_type1>(data0.reversed().shifted(
                        storage_type0::Size - storage_type1::Size))};
        }
    }
    inline SimdArray sorted() const  //{{{2
    {
        return sortedImpl(
            std::integral_constant<bool, storage_type0::Size == storage_type1::Size>());
    }

    Vc_INTRINSIC SimdArray sortedImpl(std::true_type) const
    {
#ifdef Vc_DEBUG_SORTED
        std::cerr << "-- " << data0 << data1 << '\n';
#endif
        const auto a = data0.sorted();
        const auto b = data1.sorted().reversed();
        const auto lo = internal::min(a, b);
        const auto hi = internal::max(a, b);
        return {lo.sorted(), hi.sorted()};
    }

    Vc_INTRINSIC SimdArray sortedImpl(std::false_type) const
    {
        using SortableArray = SimdArray<value_type, Common::nextPowerOfTwo(size())>;
        auto sortable = simd_cast<SortableArray>(*this);
        for (std::size_t i = Size; i < SortableArray::Size; ++i) {
            using limits = std::numeric_limits<value_type>;
            if (limits::has_infinity) {
                sortable[i] = limits::infinity();
            } else {
                sortable[i] = std::numeric_limits<value_type>::max();
            }
        }
        return simd_cast<SimdArray>(sortable.sorted());

        /* The following implementation appears to be less efficient. But this may need further
         * work.
        const auto a = data0.sorted();
        const auto b = data1.sorted();
#ifdef Vc_DEBUG_SORTED
        std::cerr << "== " << a << b << '\n';
#endif
        auto aIt = Vc::begin(a);
        auto bIt = Vc::begin(b);
        const auto aEnd = Vc::end(a);
        const auto bEnd = Vc::end(b);
        return SimdArray::generate([&](std::size_t) {
            if (aIt == aEnd) {
                return *(bIt++);
            }
            if (bIt == bEnd) {
                return *(aIt++);
            }
            if (*aIt < *bIt) {
                return *(aIt++);
            } else {
                return *(bIt++);
            }
        });
        */
    }

    template <typename G> static Vc_INTRINSIC SimdArray generate(const G &gen) // {{{2
    {
        auto tmp = storage_type0::generate(gen);  // GCC bug: the order of evaluation in
                                                  // an initializer list is well-defined
                                                  // (front to back), but GCC 4.8 doesn't
                                                  // implement this correctly. Therefore
                                                  // we enforce correct order.
        return {std::move(tmp),
                storage_type1::generate([&](std::size_t i) { return gen(i + N0); })};
    }

    // internal_data0/1 {{{2
    friend storage_type0 &internal_data0<>(SimdArray &x);
    friend storage_type1 &internal_data1<>(SimdArray &x);
    friend const storage_type0 &internal_data0<>(const SimdArray &x);
    friend const storage_type1 &internal_data1<>(const SimdArray &x);

    /// \internal
    Vc_INTRINSIC SimdArray(storage_type0 &&x, storage_type1 &&y) //{{{2
        : data0(std::move(x)), data1(std::move(y))
    {
    }
private: //{{{2
    storage_type0 data0;
    storage_type1 data1;
};
#undef Vc_CURRENT_CLASS_NAME
template <typename T, std::size_t N, typename VectorType, std::size_t M> constexpr std::size_t SimdArray<T, N, VectorType, M>::Size;
template <typename T, std::size_t N, typename VectorType, std::size_t M>
constexpr std::size_t SimdArray<T, N, VectorType, M>::MemoryAlignment;

// gatherImplementation {{{2
template <typename T, std::size_t N, typename VectorType, std::size_t M>
template <typename MT, typename IT>
inline void SimdArray<T, N, VectorType, M>::gatherImplementation(const MT *mem,
                                                                 IT &&indexes)
{
    data0.gather(mem, Split::lo(Common::Operations::gather(),
                                indexes));  // don't forward indexes - it could move and
                                            // thus break the next line
    data1.gather(mem, Split::hi(Common::Operations::gather(), std::forward<IT>(indexes)));
}
template <typename T, std::size_t N, typename VectorType, std::size_t M>
template <typename MT, typename IT>
inline void SimdArray<T, N, VectorType, M>::gatherImplementation(const MT *mem,
                                                                 IT &&indexes, MaskArgument mask)
{
    data0.gather(mem, Split::lo(Common::Operations::gather(), indexes),
                 Split::lo(mask));  // don't forward indexes - it could move and
                                    // thus break the next line
    data1.gather(mem, Split::hi(Common::Operations::gather(), std::forward<IT>(indexes)),
                 Split::hi(mask));
}

// internal_data0/1 (SimdArray) {{{1
template <typename T, std::size_t N, typename V, std::size_t M>
Vc_INTRINSIC typename SimdArrayTraits<T, N>::storage_type0 &internal_data0(
    SimdArray<T, N, V, M> &x)
{
    return x.data0;
}
template <typename T, std::size_t N, typename V, std::size_t M>
Vc_INTRINSIC typename SimdArrayTraits<T, N>::storage_type1 &internal_data1(
    SimdArray<T, N, V, M> &x)
{
    return x.data1;
}
template <typename T, std::size_t N, typename V, std::size_t M>
Vc_INTRINSIC const typename SimdArrayTraits<T, N>::storage_type0 &internal_data0(
    const SimdArray<T, N, V, M> &x)
{
    return x.data0;
}
template <typename T, std::size_t N, typename V, std::size_t M>
Vc_INTRINSIC const typename SimdArrayTraits<T, N>::storage_type1 &internal_data1(
    const SimdArray<T, N, V, M> &x)
{
    return x.data1;
}

// binary operators {{{1
namespace result_vector_type_internal
{
template <typename T>
using type = typename std::remove_cv<typename std::remove_reference<T>::type>::type;

template <typename T>
using is_integer_larger_than_int = std::integral_constant<
    bool, std::is_integral<T>::value &&(sizeof(T) > sizeof(int) ||
                                        std::is_same<T, long>::value ||
                                        std::is_same<T, unsigned long>::value)>;

template <
    typename L, typename R, std::size_t N = Traits::isSimdArray<L>::value
                                                ? Traits::simd_vector_size<L>::value
                                                : Traits::simd_vector_size<R>::value,
    bool = (Traits::isSimdArray<L>::value ||
            Traits::isSimdArray<R>::value)  // one of the operands must be a SimdArray
           &&
           !std::is_same<type<L>, type<R>>::value  // if the operands are of the same type
                                                   // use the member function
           &&
           ((std::is_arithmetic<type<L>>::value &&
             !is_integer_larger_than_int<type<L>>::value) ||
            (std::is_arithmetic<type<R>>::value &&
             !is_integer_larger_than_int<
                 type<R>>::value)  // one of the operands is a scalar type
            ||
            (Traits::is_simd_vector<L>::value && !Traits::isSimdArray<L>::value) ||
            (Traits::is_simd_vector<R>::value &&
             !Traits::isSimdArray<R>::value)  // or one of the operands is Vector<T>
            ) > struct evaluate;

template <typename L, typename R, std::size_t N> struct evaluate<L, R, N, true>
{
private:
    using LScalar = Traits::entry_type_of<L>;
    using RScalar = Traits::entry_type_of<R>;

    template <bool B, typename True, typename False>
    using conditional = typename std::conditional<B, True, False>::type;

public:
    // In principle we want the exact same rules for SimdArray<T> ⨉ SimdArray<U> as the standard
    // defines for T ⨉ U. BUT: short ⨉ short returns int (because all integral types smaller than
    // int are promoted to int before any operation). This would imply that SIMD types with integral
    // types smaller than int are more or less useless - and you could use SimdArray<int> from the
    // start. Therefore we special-case those operations where the scalar type of both operands is
    // integral and smaller than int.
    // In addition to that there is no generic support for 64-bit int SIMD types. Therefore
    // promotion to a 64-bit integral type (including `long` because it can potentially have 64
    // bits) also is not done. But if one of the operands is a scalar type that is larger than int
    // then the operator is disabled altogether. We do not want an implicit demotion.
    using type = SimdArray<
        conditional<(std::is_integral<LScalar>::value &&std::is_integral<RScalar>::value &&
                     sizeof(LScalar) < sizeof(int) &&
                     sizeof(RScalar) < sizeof(int)),
                    conditional<(sizeof(LScalar) == sizeof(RScalar)),
                                conditional<std::is_unsigned<LScalar>::value, LScalar, RScalar>,
                                conditional<(sizeof(LScalar) > sizeof(RScalar)), LScalar, RScalar>>,
                    decltype(std::declval<LScalar>() + std::declval<RScalar>())>,
        N>;
};

}  // namespace result_vector_type_internal

template <typename L, typename R>
using result_vector_type = typename result_vector_type_internal::evaluate<L, R>::type;

static_assert(
    std::is_same<result_vector_type<short int, Vc::SimdArray<short unsigned int, 32ul>>,
                 Vc::SimdArray<short unsigned int, 32ul>>::value,
    "result_vector_type does not work");

#define Vc_BINARY_OPERATORS_(op__)                                                       \
    template <typename L, typename R>                                                    \
    Vc_INTRINSIC result_vector_type<L, R> operator op__(L &&lhs, R &&rhs)                \
    {                                                                                    \
        using Return = result_vector_type<L, R>;                                         \
        return Return(std::forward<L>(lhs)) op__ Return(std::forward<R>(rhs));           \
    }
Vc_ALL_ARITHMETICS(Vc_BINARY_OPERATORS_)
Vc_ALL_BINARY(Vc_BINARY_OPERATORS_)
#undef Vc_BINARY_OPERATORS_
#define Vc_BINARY_OPERATORS_(op__)                                                       \
    template <typename L, typename R>                                                    \
    Vc_INTRINSIC typename result_vector_type<L, R>::mask_type operator op__(L &&lhs,     \
                                                                            R &&rhs)     \
    {                                                                                    \
        using Promote = result_vector_type<L, R>;                                        \
        return Promote(std::forward<L>(lhs)) op__ Promote(std::forward<R>(rhs));         \
    }
Vc_ALL_COMPARES(Vc_BINARY_OPERATORS_)
#undef Vc_BINARY_OPERATORS_

// math functions {{{1
template <typename T, std::size_t N> SimdArray<T, N> abs(const SimdArray<T, N> &x)
{
    return SimdArray<T, N>::fromOperation(Common::Operations::Abs(), x);
}
template <typename T, std::size_t N> SimdMaskArray<T, N> isnan(const SimdArray<T, N> &x)
{
    return SimdMaskArray<T, N>::fromOperation(Common::Operations::Isnan(), x);
}
template <typename T, std::size_t N>
SimdArray<T, N> frexp(const SimdArray<T, N> &x, SimdArray<int, N> *e)
{
    return SimdArray<T, N>::fromOperation(Common::Operations::Frexp(), x, e);
}
template <typename T, std::size_t N>
SimdArray<T, N> ldexp(const SimdArray<T, N> &x, const SimdArray<int, N> &e)
{
    return SimdArray<T, N>::fromOperation(Common::Operations::Ldexp(), x, e);
}
template <typename T, std::size_t N>
SimdArray<T, N> sqrt(const SimdArray<T, N> &x)
{
    return SimdArray<T, N>::fromOperation(Common::Operations::Sqrt(), x);
}

// simd_cast {{{1
// simd_cast_impl_smaller_input {{{2
// The following function can be implemented without the sizeof...(From) overload.
// However, ICC has a bug (Premier Issue #6000116338) which leads to an ICE. Splitting the
// function in two works around the issue.
template <typename Return, std::size_t N, typename T, typename... From>
Vc_INTRINSIC Vc_CONST enable_if<sizeof...(From) != 0, Return>
simd_cast_impl_smaller_input(const From &... xs, const T &last)
{
    Return r = simd_cast<Return>(xs...);
    for (size_t i = 0; i < N; ++i) {
        r[i + N * sizeof...(From)] = static_cast<typename Return::EntryType>(last[i]);
    }
    return r;
}
template <typename Return, std::size_t N, typename T>
Vc_INTRINSIC Vc_CONST Return simd_cast_impl_smaller_input(const T &last)
{
    Return r = Return();
    for (size_t i = 0; i < N; ++i) {
        r[i] = static_cast<typename Return::EntryType>(last[i]);
    }
    return r;
}
template <typename Return, std::size_t N, typename T, typename... From>
Vc_INTRINSIC Vc_CONST enable_if<sizeof...(From) != 0, Return> simd_cast_impl_larger_input(
    const From &... xs, const T &last)
{
    Return r = simd_cast<Return>(xs...);
    for (size_t i = N * sizeof...(From); i < Return::Size; ++i) {
        r[i] = static_cast<typename Return::EntryType>(last[i - N * sizeof...(From)]);
    }
    return r;
}
template <typename Return, std::size_t N, typename T>
Vc_INTRINSIC Vc_CONST Return simd_cast_impl_larger_input(const T &last)
{
    Return r = Return();
    for (size_t i = 0; i < Return::size(); ++i) {
        r[i] = static_cast<typename Return::EntryType>(last[i]);
    }
    return r;
}

// simd_cast_without_last (declaration) {{{2
template <typename Return, typename T, typename... From>
Vc_INTRINSIC_L Vc_CONST_L Return
    simd_cast_without_last(const From &... xs, const T &) Vc_INTRINSIC_R Vc_CONST_R;

// are_all_types_equal {{{2
template <typename... Ts> struct are_all_types_equal;
template <typename T>
struct are_all_types_equal<T> : public std::integral_constant<bool, true>
{
};
template <typename T0, typename T1, typename... Ts>
struct are_all_types_equal<T0, T1, Ts...>
    : public std::integral_constant<
          bool, std::is_same<T0, T1>::value && are_all_types_equal<T1, Ts...>::value>
{
};

// simd_cast_interleaved_argument_order (declarations) {{{2
/*! \internal
  The need for simd_cast_interleaved_argument_order stems from a shortcoming in pack
  expansion of variadic templates in C++. For a simd_cast with SimdArray arguments that
  are bisectable (i.e.  \c storage_type0 and \c storage_type1 are equal) the generic
  implementation needs to forward to a simd_cast of the \c internal_data0 and \c
  internal_data1 of the arguments. But the required order of arguments is
  `internal_data0(arg0), internal_data1(arg0), internal_data0(arg1), ...`. This is
  impossible to achieve with pack expansion. It is only possible to write
  `internal_data0(args)..., internal_data1(args)...` and thus have the argument order
  mixed up. The simd_cast_interleaved_argument_order “simply” calls simd_cast with the
  arguments correctly reordered (i.e. interleaved).

  The implementation of simd_cast_interleaved_argument_order is done generically, so that
  it supports any number of arguments. The central idea of the implementation is an
  `extract` function which returns one value of an argument pack determined via an index
  passed as template argument. This index is generated via an index_sequence. The
  `extract` function uses two argument packs (of equal size) to easily return values from
  the front and middle of the argument pack (for doing the deinterleave).
 */
template <typename Return, typename... Ts>
Vc_INTRINSIC Vc_CONST Return
    simd_cast_interleaved_argument_order(const Ts &... a, const Ts &... b);

// simd_cast_with_offset (declarations and one impl) {{{2
// offset == 0 {{{3
template <typename Return, std::size_t offset, typename From, typename... Froms>
Vc_INTRINSIC Vc_CONST
    enable_if<(are_all_types_equal<From, Froms...>::value && offset == 0), Return>
        simd_cast_with_offset(const From &x, const Froms &... xs);
// offset > 0 && offset divisible by Return::Size {{{3
template <typename Return, std::size_t offset, typename From>
Vc_INTRINSIC Vc_CONST
    enable_if<(From::Size > offset && offset > 0 && offset % Return::Size == 0), Return>
        simd_cast_with_offset(const From &x);
// offset > 0 && offset NOT divisible && Return is non-atomic simd(mask)array {{{3
template <typename Return, std::size_t offset, typename From>
Vc_INTRINSIC Vc_CONST
    enable_if<(From::Size > offset && offset > 0 && offset % Return::Size != 0 &&
               ((Traits::isSimdArray<Return>::value &&
                 !Traits::isAtomicSimdArray<Return>::value) ||
                (Traits::isSimdMaskArray<Return>::value &&
                 !Traits::isAtomicSimdMaskArray<Return>::value))),
              Return>
        simd_cast_with_offset(const From &x);
// offset > 0 && offset NOT divisible && Return is atomic simd(mask)array {{{3
template <typename Return, std::size_t offset, typename From>
Vc_INTRINSIC Vc_CONST
    enable_if<(From::Size > offset && offset > 0 && offset % Return::Size != 0 &&
               ((Traits::isSimdArray<Return>::value &&
                 Traits::isAtomicSimdArray<Return>::value) ||
                (Traits::isSimdMaskArray<Return>::value &&
                 Traits::isAtomicSimdMaskArray<Return>::value))),
              Return>
        simd_cast_with_offset(const From &x);
// offset > first argument (drops first arg) {{{3
template <typename Return, std::size_t offset, typename From, typename... Froms>
Vc_INTRINSIC Vc_CONST enable_if<
    (are_all_types_equal<From, Froms...>::value && From::Size <= offset), Return>
    simd_cast_with_offset(const From &, const Froms &... xs)
{
    return simd_cast_with_offset<Return, offset - From::Size>(xs...);
}

// offset > first and only argument (returns Zero) {{{3
template <typename Return, std::size_t offset, typename From>
Vc_INTRINSIC Vc_CONST enable_if<(From::Size <= offset), Return> simd_cast_with_offset(
    const From &)
{
    return Return::Zero();
}

// first_type_of {{{2
template <typename T, typename... Ts> struct first_type_of_impl
{
    using type = T;
};
template <typename... Ts> using first_type_of = typename first_type_of_impl<Ts...>::type;

// simd_cast_drop_arguments (declarations) {{{2
template <typename Return, typename From>
Vc_INTRINSIC Vc_CONST Return simd_cast_drop_arguments(From x);
template <typename Return, typename... Froms>
Vc_INTRINSIC Vc_CONST
    enable_if<(are_all_types_equal<Froms...>::value &&
               sizeof...(Froms) * first_type_of<Froms...>::Size < Return::Size),
              Return>
        simd_cast_drop_arguments(Froms... xs, first_type_of<Froms...> x);
// The following function can be implemented without the sizeof...(From) overload.
// However, ICC has a bug (Premier Issue #6000116338) which leads to an ICE. Splitting the
// function in two works around the issue.
template <typename Return, typename From, typename... Froms>
Vc_INTRINSIC Vc_CONST enable_if<
    (are_all_types_equal<From, Froms...>::value &&
     (1 + sizeof...(Froms)) * From::Size >= Return::Size && sizeof...(Froms) != 0),
    Return>
simd_cast_drop_arguments(Froms... xs, From x, From);
template <typename Return, typename From>
Vc_INTRINSIC Vc_CONST
    enable_if<(are_all_types_equal<From>::value && From::Size >= Return::Size), Return>
    simd_cast_drop_arguments(From x, From);

namespace
{
#ifdef Vc_DEBUG_SIMD_CAST
void debugDoNothing(const std::initializer_list<void *> &) {}
template <typename T0, typename... Ts>
inline void vc_debug_(const char *prefix, const char *suffix, const T0 &arg0,
                      const Ts &... args)
{
    std::cerr << prefix << arg0;
    debugDoNothing({&(std::cerr << ", " << args)...});
    std::cerr << suffix;
}
#else
template <typename T0, typename... Ts>
Vc_INTRINSIC void vc_debug_(const char *, const char *, const T0 &, const Ts &...)
{
}
#endif
}  // unnamed namespace

// simd_cast<T>(xs...) to SimdArray/-mask {{{2
#define Vc_SIMDARRAY_CASTS(SimdArrayType__, trait_name__)                                \
    template <typename Return, typename From, typename... Froms>                         \
    Vc_INTRINSIC Vc_CONST enable_if<(Traits::isAtomic##SimdArrayType__<Return>::value && \
                                     !Traits::is##SimdArrayType__<From>::value &&        \
                                     Traits::is_simd_##trait_name__<From>::value &&      \
                                     From::Size * sizeof...(Froms) < Return::Size &&     \
                                     are_all_types_equal<From, Froms...>::value),        \
                                    Return>                                              \
    simd_cast(From x, Froms... xs)                                                       \
    {                                                                                    \
        vc_debug_("simd_cast{1}(", ")\n", x, xs...);                                     \
        return {simd_cast<typename Return::storage_type>(x, xs...)};                     \
    }                                                                                    \
    template <typename Return, typename From, typename... Froms>                         \
    Vc_INTRINSIC Vc_CONST enable_if<(Traits::isAtomic##SimdArrayType__<Return>::value && \
                                     !Traits::is##SimdArrayType__<From>::value &&        \
                                     Traits::is_simd_##trait_name__<From>::value &&      \
                                     From::Size * sizeof...(Froms) >= Return::Size &&    \
                                     are_all_types_equal<From, Froms...>::value),        \
                                    Return>                                              \
    simd_cast(From x, Froms... xs)                                                       \
    {                                                                                    \
        vc_debug_("simd_cast{2}(", ")\n", x, xs...);                                     \
        return {simd_cast_without_last<Return, From, Froms...>(x, xs...)};               \
    }                                                                                    \
    template <typename Return, typename From, typename... Froms>                         \
    Vc_INTRINSIC Vc_CONST                                                                \
        enable_if<(Traits::is##SimdArrayType__<Return>::value &&                         \
                   !Traits::isAtomic##SimdArrayType__<Return>::value &&                  \
                   !Traits::is##SimdArrayType__<From>::value &&                          \
                   Traits::is_simd_##trait_name__<From>::value &&                        \
                   Common::left_size(Return::Size) <                                     \
                       From::Size * (1 + sizeof...(Froms)) &&                            \
                   are_all_types_equal<From, Froms...>::value),                          \
                  Return>                                                                \
        simd_cast(From x, Froms... xs)                                                   \
    {                                                                                    \
        vc_debug_("simd_cast{3}(", ")\n", x, xs...);                                     \
        using R0 = typename Return::storage_type0;                                       \
        using R1 = typename Return::storage_type1;                                       \
        return {simd_cast_drop_arguments<R0, Froms...>(x, xs...),                        \
                simd_cast_with_offset<R1, R0::Size>(x, xs...)};                          \
    }                                                                                    \
    template <typename Return, typename From, typename... Froms>                         \
    Vc_INTRINSIC Vc_CONST                                                                \
        enable_if<(Traits::is##SimdArrayType__<Return>::value &&                         \
                   !Traits::isAtomic##SimdArrayType__<Return>::value &&                  \
                   !Traits::is##SimdArrayType__<From>::value &&                          \
                   Traits::is_simd_##trait_name__<From>::value &&                        \
                   Common::left_size(Return::Size) >=                                    \
                       From::Size * (1 + sizeof...(Froms)) &&                            \
                   are_all_types_equal<From, Froms...>::value),                          \
                  Return>                                                                \
        simd_cast(From x, Froms... xs)                                                   \
    {                                                                                    \
        vc_debug_("simd_cast{4}(", ")\n", x, xs...);                                     \
        using R0 = typename Return::storage_type0;                                       \
        using R1 = typename Return::storage_type1;                                       \
        return {simd_cast<R0>(x, xs...), R1::Zero()};                                    \
    }
Vc_SIMDARRAY_CASTS(SimdArray, vector)
Vc_SIMDARRAY_CASTS(SimdMaskArray, mask)
#undef Vc_SIMDARRAY_CASTS

// simd_cast<SimdArray/-mask, offset>(V) {{{2
#define Vc_SIMDARRAY_CASTS(SimdArrayType__, trait_name__)                                \
    /* SIMD Vector/Mask to atomic SimdArray/simdmaskarray */                             \
    template <typename Return, int offset, typename From>                                \
    Vc_INTRINSIC Vc_CONST enable_if<(Traits::isAtomic##SimdArrayType__<Return>::value && \
                                     !Traits::is##SimdArrayType__<From>::value &&        \
                                     Traits::is_simd_##trait_name__<From>::value),       \
                                    Return>                                              \
    simd_cast(From x)                                                                    \
    {                                                                                    \
        vc_debug_("simd_cast{offset, atomic}(", ")\n", offset, x);                       \
        return {simd_cast<typename Return::storage_type, offset>(x)};                    \
    }                                                                                    \
    /* both halves of Return array are extracted from argument */                        \
    template <typename Return, int offset, typename From>                                \
    Vc_INTRINSIC Vc_CONST                                                                \
        enable_if<(Traits::is##SimdArrayType__<Return>::value &&                         \
                   !Traits::isAtomic##SimdArrayType__<Return>::value &&                  \
                   !Traits::is##SimdArrayType__<From>::value &&                          \
                   Traits::is_simd_##trait_name__<From>::value &&                        \
                   Return::Size * offset + Common::left_size(Return::Size) <             \
                       From::Size),                                                      \
                  Return>                                                                \
        simd_cast(From x)                                                                \
    {                                                                                    \
        vc_debug_("simd_cast{offset, split Return}(", ")\n", offset, x);                 \
        using R0 = typename Return::storage_type0;                                       \
        constexpr int entries_offset = offset * Return::Size;                            \
        constexpr int entries_offset_right = entries_offset + R0::Size;                  \
        return {                                                                         \
            simd_cast_with_offset<typename Return::storage_type0, entries_offset>(x),    \
            simd_cast_with_offset<typename Return::storage_type1, entries_offset_right>( \
                x)};                                                                     \
    }                                                                                    \
    /* SIMD Vector/Mask to non-atomic SimdArray/simdmaskarray */                         \
    /* right half of Return array is zero */                                             \
    template <typename Return, int offset, typename From>                                \
    Vc_INTRINSIC Vc_CONST                                                                \
        enable_if<(Traits::is##SimdArrayType__<Return>::value &&                         \
                   !Traits::isAtomic##SimdArrayType__<Return>::value &&                  \
                   !Traits::is##SimdArrayType__<From>::value &&                          \
                   Traits::is_simd_##trait_name__<From>::value &&                        \
                   Return::Size * offset + Common::left_size(Return::Size) >=            \
                       From::Size),                                                      \
                  Return>                                                                \
        simd_cast(From x)                                                                \
    {                                                                                    \
        vc_debug_("simd_cast{offset, R1::Zero}(", ")\n", offset, x);                     \
        using R0 = typename Return::storage_type0;                                       \
        using R1 = typename Return::storage_type1;                                       \
        constexpr int entries_offset = offset * Return::Size;                            \
        return {simd_cast_with_offset<R0, entries_offset>(x), R1::Zero()};               \
    }
Vc_SIMDARRAY_CASTS(SimdArray, vector)
Vc_SIMDARRAY_CASTS(SimdMaskArray, mask)
#undef Vc_SIMDARRAY_CASTS

// simd_cast<T>(xs...) from SimdArray/-mask {{{2
#define Vc_SIMDARRAY_CASTS(SimdArrayType__)                                              \
    /* indivisible SimdArrayType__ */                                                    \
    template <typename Return, typename T, std::size_t N, typename V, typename... From>  \
    Vc_INTRINSIC Vc_CONST                                                                \
        enable_if<(are_all_types_equal<SimdArrayType__<T, N, V, N>, From...>::value &&   \
                   (sizeof...(From) == 0 || N * sizeof...(From) < Return::Size) &&       \
                   !std::is_same<Return, SimdArrayType__<T, N, V, N>>::value),           \
                  Return>                                                                \
        simd_cast(const SimdArrayType__<T, N, V, N> &x0, const From &... xs)             \
    {                                                                                    \
        vc_debug_("simd_cast{indivisible}(", ")\n", x0, xs...);                          \
        return simd_cast<Return>(internal_data(x0), internal_data(xs)...);               \
    }                                                                                    \
    /* indivisible SimdArrayType__ && can drop arguments from the end */                 \
    template <typename Return, typename T, std::size_t N, typename V, typename... From>  \
    Vc_INTRINSIC Vc_CONST                                                                \
        enable_if<(are_all_types_equal<SimdArrayType__<T, N, V, N>, From...>::value &&   \
                   (sizeof...(From) > 0 && (N * sizeof...(From) >= Return::Size)) &&     \
                   !std::is_same<Return, SimdArrayType__<T, N, V, N>>::value),           \
                  Return>                                                                \
        simd_cast(const SimdArrayType__<T, N, V, N> &x0, const From &... xs)             \
    {                                                                                    \
        vc_debug_("simd_cast{indivisible2}(", ")\n", x0, xs...);                         \
        return simd_cast_without_last<                                                   \
            Return, typename SimdArrayType__<T, N, V, N>::storage_type,                  \
            typename From::storage_type...>(internal_data(x0), internal_data(xs)...);    \
    }                                                                                    \
    /* bisectable SimdArrayType__ (N = 2^n) && never too large */                        \
    template <typename Return, typename T, std::size_t N, typename V, std::size_t M,     \
              typename... From>                                                          \
    Vc_INTRINSIC Vc_CONST                                                                \
        enable_if<(N != M &&                                                             \
                   are_all_types_equal<SimdArrayType__<T, N, V, M>, From...>::value &&   \
                   N * sizeof...(From) < Return::Size && ((N - 1) & N) == 0),            \
                  Return>                                                                \
        simd_cast(const SimdArrayType__<T, N, V, M> &x0, const From &... xs)             \
    {                                                                                    \
        vc_debug_("simd_cast{bisectable}(", ")\n", x0, xs...);                           \
        return simd_cast_interleaved_argument_order<                                     \
            Return, typename SimdArrayType__<T, N, V, M>::storage_type0,                 \
            typename From::storage_type0...>(internal_data0(x0), internal_data0(xs)...,  \
                                             internal_data1(x0), internal_data1(xs)...); \
    }                                                                                    \
    /* bisectable SimdArrayType__ (N = 2^n) && input so large that at least the last     \
     * input can be dropped */                                                           \
    template <typename Return, typename T, std::size_t N, typename V, std::size_t M,     \
              typename... From>                                                          \
    Vc_INTRINSIC Vc_CONST                                                                \
        enable_if<(N != M &&                                                             \
                   are_all_types_equal<SimdArrayType__<T, N, V, M>, From...>::value &&   \
                   N * sizeof...(From) >= Return::Size && ((N - 1) & N) == 0),           \
                  Return>                                                                \
        simd_cast(const SimdArrayType__<T, N, V, M> &x0, const From &... xs)             \
    {                                                                                    \
        vc_debug_("simd_cast{bisectable2}(", ")\n", x0, xs...);                          \
        return simd_cast_without_last<Return, SimdArrayType__<T, N, V, M>, From...>(     \
            x0, xs...);                                                                  \
    }                                                                                    \
    /* remaining SimdArrayType__ input never larger (N != 2^n) */                        \
    template <typename Return, typename T, std::size_t N, typename V, std::size_t M,     \
              typename... From>                                                          \
    Vc_INTRINSIC Vc_CONST                                                                \
        enable_if<(N != M &&                                                             \
                   are_all_types_equal<SimdArrayType__<T, N, V, M>, From...>::value &&   \
                   N * (1 + sizeof...(From)) <= Return::Size && ((N - 1) & N) != 0),     \
                  Return>                                                                \
        simd_cast(const SimdArrayType__<T, N, V, M> &x0, const From &... xs)             \
    {                                                                                    \
        vc_debug_("simd_cast{remaining}(", ")\n", x0, xs...);                            \
        return simd_cast_impl_smaller_input<Return, N, SimdArrayType__<T, N, V, M>,      \
                                            From...>(x0, xs...);                         \
    }                                                                                    \
    /* remaining SimdArrayType__ input larger (N != 2^n) */                              \
    template <typename Return, typename T, std::size_t N, typename V, std::size_t M,     \
              typename... From>                                                          \
    Vc_INTRINSIC Vc_CONST                                                                \
        enable_if<(N != M &&                                                             \
                   are_all_types_equal<SimdArrayType__<T, N, V, M>, From...>::value &&   \
                   N * (1 + sizeof...(From)) > Return::Size && ((N - 1) & N) != 0),      \
                  Return>                                                                \
        simd_cast(const SimdArrayType__<T, N, V, M> &x0, const From &... xs)             \
    {                                                                                    \
        vc_debug_("simd_cast{remaining2}(", ")\n", x0, xs...);                           \
        return simd_cast_impl_larger_input<Return, N, SimdArrayType__<T, N, V, M>,       \
                                           From...>(x0, xs...);                          \
    }                                                                                    \
    /* a single bisectable SimdArrayType__ (N = 2^n) too large */                        \
    template <typename Return, typename T, std::size_t N, typename V, std::size_t M>     \
    Vc_INTRINSIC Vc_CONST                                                                \
        enable_if<(N != M && N >= 2 * Return::Size && ((N - 1) & N) == 0), Return>       \
        simd_cast(const SimdArrayType__<T, N, V, M> &x)                                  \
    {                                                                                    \
        vc_debug_("simd_cast{single bisectable}(", ")\n", x);                            \
        return simd_cast<Return>(internal_data0(x));                                     \
    }                                                                                    \
    template <typename Return, typename T, std::size_t N, typename V, std::size_t M>     \
    Vc_INTRINSIC Vc_CONST enable_if<(N != M && N > Return::Size &&                       \
                                     N < 2 * Return::Size && ((N - 1) & N) == 0),        \
                                    Return>                                              \
    simd_cast(const SimdArrayType__<T, N, V, M> &x)                                      \
    {                                                                                    \
        vc_debug_("simd_cast{single bisectable2}(", ")\n", x);                           \
        return simd_cast<Return>(internal_data0(x), internal_data1(x));                  \
    }
Vc_SIMDARRAY_CASTS(SimdArray)
Vc_SIMDARRAY_CASTS(SimdMaskArray)
#undef Vc_SIMDARRAY_CASTS

// simd_cast<T, offset>(SimdArray/-mask) {{{2
#define Vc_SIMDARRAY_CASTS(SimdArrayType__)                                              \
    /* offset == 0 is like without offset */                                             \
    template <typename Return, int offset, typename T, std::size_t N, typename V,        \
              std::size_t M>                                                             \
    Vc_INTRINSIC Vc_CONST enable_if<(offset == 0), Return> simd_cast(                    \
        const SimdArrayType__<T, N, V, M> &x)                                            \
    {                                                                                    \
        vc_debug_("simd_cast{offset == 0}(", ")\n", offset, x);                          \
        return simd_cast<Return>(x);                                                     \
    }                                                                                    \
    /* forward to V */                                                                   \
    template <typename Return, int offset, typename T, std::size_t N, typename V>        \
    Vc_INTRINSIC Vc_CONST enable_if<(offset != 0), Return> simd_cast(                    \
        const SimdArrayType__<T, N, V, N> &x)                                            \
    {                                                                                    \
        vc_debug_("simd_cast{offset, forward}(", ")\n", offset, x);                      \
        return simd_cast<Return, offset>(internal_data(x));                              \
    }                                                                                    \
    /* convert from right member of SimdArray */                                         \
    template <typename Return, int offset, typename T, std::size_t N, typename V,        \
              std::size_t M>                                                             \
    Vc_INTRINSIC Vc_CONST                                                                \
        enable_if<(N != M && offset * Return::Size >= Common::left_size(N) &&            \
                   offset != 0 && Common::left_size(N) % Return::Size == 0),             \
                  Return>                                                                \
        simd_cast(const SimdArrayType__<T, N, V, M> &x)                                  \
    {                                                                                    \
        vc_debug_("simd_cast{offset, right}(", ")\n", offset, x);                        \
        return simd_cast<Return, offset - Common::left_size(N) / Return::Size>(          \
            internal_data1(x));                                                          \
    }                                                                                    \
    /* same as above except for odd cases where offset * Return::Size doesn't fit the    \
     * left side of the SimdArray */                                                     \
    template <typename Return, int offset, typename T, std::size_t N, typename V,        \
              std::size_t M>                                                             \
    Vc_INTRINSIC Vc_CONST                                                                \
        enable_if<(N != M && offset * Return::Size >= Common::left_size(N) &&            \
                   offset != 0 && Common::left_size(N) % Return::Size != 0),             \
                  Return>                                                                \
        simd_cast(const SimdArrayType__<T, N, V, M> &x)                                  \
    {                                                                                    \
        vc_debug_("simd_cast{offset, right, nofit}(", ")\n", offset, x);                 \
        return simd_cast_with_offset<Return,                                             \
                                     offset * Return::Size - Common::left_size(N)>(      \
            internal_data1(x));                                                          \
    }                                                                                    \
    /* convert from left member of SimdArray */                                          \
    template <typename Return, int offset, typename T, std::size_t N, typename V,        \
              std::size_t M>                                                             \
    Vc_INTRINSIC Vc_CONST                                                                \
        enable_if<(N != M && /*offset * Return::Size < Common::left_size(N) &&*/         \
                   offset != 0 && (offset + 1) * Return::Size <= Common::left_size(N)),  \
                  Return>                                                                \
        simd_cast(const SimdArrayType__<T, N, V, M> &x)                                  \
    {                                                                                    \
        vc_debug_("simd_cast{offset, left}(", ")\n", offset, x);                         \
        return simd_cast<Return, offset>(internal_data0(x));                             \
    }                                                                                    \
    /* fallback to copying scalars */                                                    \
    template <typename Return, int offset, typename T, std::size_t N, typename V,        \
              std::size_t M>                                                             \
    Vc_INTRINSIC Vc_CONST                                                                \
        enable_if<(N != M && (offset * Return::Size < Common::left_size(N)) &&           \
                   offset != 0 && (offset + 1) * Return::Size > Common::left_size(N)),   \
                  Return>                                                                \
        simd_cast(const SimdArrayType__<T, N, V, M> &x)                                  \
    {                                                                                    \
        vc_debug_("simd_cast{offset, copy scalars}(", ")\n", offset, x);                 \
        using R = typename Return::EntryType;                                            \
        Return r = Return::Zero();                                                       \
        for (std::size_t i = offset * Return::Size;                                      \
             i < std::min(N, (offset + 1) * Return::Size); ++i) {                        \
            r[i - offset * Return::Size] = static_cast<R>(x[i]);                         \
        }                                                                                \
        return r;                                                                        \
    }
Vc_SIMDARRAY_CASTS(SimdArray)
Vc_SIMDARRAY_CASTS(SimdMaskArray)
#undef Vc_SIMDARRAY_CASTS
// simd_cast_drop_arguments (definitions) {{{2
template <typename Return, typename From>
Vc_INTRINSIC Vc_CONST Return simd_cast_drop_arguments(From x)
{
    return simd_cast<Return>(x);
}
template <typename Return, typename... Froms>
Vc_INTRINSIC Vc_CONST
    enable_if<(are_all_types_equal<Froms...>::value &&
               sizeof...(Froms) * first_type_of<Froms...>::Size < Return::Size),
              Return>
        simd_cast_drop_arguments(Froms... xs, first_type_of<Froms...> x)
{
    return simd_cast<Return>(xs..., x);
}
// The following function can be implemented without the sizeof...(From) overload.
// However, ICC has a bug (Premier Issue #6000116338) which leads to an ICE. Splitting the
// function in two works around the issue.
template <typename Return, typename From, typename... Froms>
Vc_INTRINSIC Vc_CONST enable_if<
    (are_all_types_equal<From, Froms...>::value &&
     (1 + sizeof...(Froms)) * From::Size >= Return::Size && sizeof...(Froms) != 0),
    Return>
simd_cast_drop_arguments(Froms... xs, From x, From)
{
    return simd_cast_drop_arguments<Return, Froms...>(xs..., x);
}
template <typename Return, typename From>
Vc_INTRINSIC Vc_CONST
    enable_if<(are_all_types_equal<From>::value && From::Size >= Return::Size), Return>
    simd_cast_drop_arguments(From x, From)
{
    return simd_cast_drop_arguments<Return>(x);
}

// simd_cast_with_offset (definitions) {{{2
    template <typename Return, std::size_t offset, typename From>
    Vc_INTRINSIC Vc_CONST
    enable_if<(From::Size > offset && offset > 0 && offset % Return::Size == 0),
              Return> simd_cast_with_offset(const From &x)
{
    return simd_cast<Return, offset / Return::Size>(x);
}
template <typename Return, std::size_t offset, typename From>
Vc_INTRINSIC Vc_CONST
    enable_if<(From::Size > offset && offset > 0 && offset % Return::Size != 0 &&
               ((Traits::isSimdArray<Return>::value &&
                 !Traits::isAtomicSimdArray<Return>::value) ||
                (Traits::isSimdMaskArray<Return>::value &&
                 !Traits::isAtomicSimdMaskArray<Return>::value))),
              Return>
        simd_cast_with_offset(const From &x)
{
    using R0 = typename Return::storage_type0;
    using R1 = typename Return::storage_type1;
    return {simd_cast_with_offset<R0, offset>(x),
            simd_cast_with_offset<R1, offset + R0::Size>(x)};
}
template <typename Return, std::size_t offset, typename From>
Vc_INTRINSIC Vc_CONST
    enable_if<(From::Size > offset && offset > 0 && offset % Return::Size != 0 &&
               ((Traits::isSimdArray<Return>::value &&
                 Traits::isAtomicSimdArray<Return>::value) ||
                (Traits::isSimdMaskArray<Return>::value &&
                 Traits::isAtomicSimdMaskArray<Return>::value))),
              Return>
        simd_cast_with_offset(const From &x)
{
    return simd_cast<Return, offset / Return::Size>(x.shifted(offset % Return::Size));
}
template <typename Return, std::size_t offset, typename From, typename... Froms>
Vc_INTRINSIC Vc_CONST
    enable_if<(are_all_types_equal<From, Froms...>::value && offset == 0), Return>
        simd_cast_with_offset(const From &x, const Froms &... xs)
{
    return simd_cast<Return>(x, xs...);
}

// simd_cast_without_last (definition) {{{2
template <typename Return, typename T, typename... From>
Vc_INTRINSIC Vc_CONST Return simd_cast_without_last(const From &... xs, const T &)
{
    return simd_cast<Return>(xs...);
}

// simd_cast_interleaved_argument_order (definitions) {{{2

/// \internal returns the first argument
template <std::size_t I, typename T0, typename... Ts>
Vc_INTRINSIC Vc_CONST enable_if<(I == 0), T0> extract_interleaved(const T0 &a0,
                                                                  const Ts &...,
                                                                  const T0 &,
                                                                  const Ts &...)
{
    return a0;
}
/// \internal returns the center argument
template <std::size_t I, typename T0, typename... Ts>
Vc_INTRINSIC Vc_CONST enable_if<(I == 1), T0> extract_interleaved(const T0 &,
                                                                  const Ts &...,
                                                                  const T0 &b0,
                                                                  const Ts &...)
{
    return b0;
}
/// \internal drops the first and center arguments and recurses
template <std::size_t I, typename T0, typename... Ts>
Vc_INTRINSIC Vc_CONST enable_if<(I > 1), T0> extract_interleaved(const T0 &,
                                                                 const Ts &... a,
                                                                 const T0 &,
                                                                 const Ts &... b)
{
    return extract_interleaved<I - 2, Ts...>(a..., b...);
}
/// \internal calls simd_cast with correct argument order thanks to extract_interleaved
/// and extract_back.
template <typename Return, typename... Ts, std::size_t... Indexes>
Vc_INTRINSIC Vc_CONST Return
    simd_cast_interleaved_argument_order_1(index_sequence<Indexes...>, const Ts &... a,
                                           const Ts &... b)
{
    return simd_cast<Return>(extract_interleaved<Indexes, Ts...>(a..., b...)...);
}
/// \internal constructs the necessary index_sequence to pass it to
/// simd_cast_interleaved_argument_order_1
template <typename Return, typename... Ts>
Vc_INTRINSIC Vc_CONST Return
    simd_cast_interleaved_argument_order(const Ts &... a, const Ts &... b)
{
    using seq = make_index_sequence<sizeof...(Ts)*2>;
    return simd_cast_interleaved_argument_order_1<Return, Ts...>(seq(), a..., b...);
}

// binary min/max functions (internal) {{{1
namespace internal
{
#define Vc_BINARY_FUNCTION__(name__)                                                     \
    template <typename T, std::size_t N, typename V, std::size_t M>                      \
    SimdArray<T, N, V, M> Vc_INTRINSIC Vc_PURE                                           \
    name__(const SimdArray<T, N, V, M> &l, const SimdArray<T, N, V, M> &r)               \
    {                                                                                    \
        return {name__(internal_data0(l), internal_data0(r)),                            \
                name__(internal_data1(l), internal_data1(r))};                           \
    }                                                                                    \
    template <typename T, std::size_t N, typename V>                                     \
    SimdArray<T, N, V, N> Vc_INTRINSIC Vc_PURE                                           \
    name__(const SimdArray<T, N, V, N> &l, const SimdArray<T, N, V, N> &r)               \
    {                                                                                    \
        return SimdArray<T, N, V, N>{name__(internal_data(l), internal_data(r))};        \
    }
Vc_BINARY_FUNCTION__(min)
Vc_BINARY_FUNCTION__(max)
#undef Vc_BINARY_FUNCTION__
}  // namespace internal
// conditional_assign {{{1
#define Vc_CONDITIONAL_ASSIGN(name__, op__)                                              \
    template <Operator O, typename T, std::size_t N, typename M, typename U>             \
    Vc_INTRINSIC enable_if<O == Operator::name__, void> conditional_assign(              \
        SimdArray<T, N> &lhs, M &&mask, U &&rhs)                                         \
    {                                                                                    \
        lhs(mask) op__ rhs;                                                              \
    }
Vc_CONDITIONAL_ASSIGN(          Assign,  =)
Vc_CONDITIONAL_ASSIGN(      PlusAssign, +=)
Vc_CONDITIONAL_ASSIGN(     MinusAssign, -=)
Vc_CONDITIONAL_ASSIGN(  MultiplyAssign, *=)
Vc_CONDITIONAL_ASSIGN(    DivideAssign, /=)
Vc_CONDITIONAL_ASSIGN( RemainderAssign, %=)
Vc_CONDITIONAL_ASSIGN(       XorAssign, ^=)
Vc_CONDITIONAL_ASSIGN(       AndAssign, &=)
Vc_CONDITIONAL_ASSIGN(        OrAssign, |=)
Vc_CONDITIONAL_ASSIGN( LeftShiftAssign,<<=)
Vc_CONDITIONAL_ASSIGN(RightShiftAssign,>>=)
#undef Vc_CONDITIONAL_ASSIGN

#define Vc_CONDITIONAL_ASSIGN(name__, expr__)                                            \
    template <Operator O, typename T, std::size_t N, typename M>                         \
    Vc_INTRINSIC enable_if<O == Operator::name__, SimdArray<T, N>> conditional_assign(   \
        SimdArray<T, N> &lhs, M &&mask)                                                  \
    {                                                                                    \
        return expr__;                                                                   \
    }
Vc_CONDITIONAL_ASSIGN(PostIncrement, lhs(mask)++)
Vc_CONDITIONAL_ASSIGN( PreIncrement, ++lhs(mask))
Vc_CONDITIONAL_ASSIGN(PostDecrement, lhs(mask)--)
Vc_CONDITIONAL_ASSIGN( PreDecrement, --lhs(mask))
#undef Vc_CONDITIONAL_ASSIGN
// transpose_impl {{{1
namespace Common
{
    template <int L, typename T, std::size_t N, typename V>
    inline enable_if<L == 4, void> transpose_impl(
        SimdArray<T, N, V, N> * Vc_RESTRICT r[],
        const TransposeProxy<SimdArray<T, N, V, N>, SimdArray<T, N, V, N>,
                             SimdArray<T, N, V, N>, SimdArray<T, N, V, N>> &proxy)
    {
        V *Vc_RESTRICT r2[L] = {&internal_data(*r[0]), &internal_data(*r[1]),
                                &internal_data(*r[2]), &internal_data(*r[3])};
        transpose_impl<L>(
            &r2[0], TransposeProxy<V, V, V, V>{internal_data(std::get<0>(proxy.in)),
                                               internal_data(std::get<1>(proxy.in)),
                                               internal_data(std::get<2>(proxy.in)),
                                               internal_data(std::get<3>(proxy.in))});
    }
    template <int L, typename T, typename V>
    inline enable_if<(L == 2), void> transpose_impl(
        SimdArray<T, 4, V, 1> *Vc_RESTRICT r[],
        const TransposeProxy<SimdArray<T, 2, V, 1>, SimdArray<T, 2, V, 1>,
                             SimdArray<T, 2, V, 1>, SimdArray<T, 2, V, 1>> &proxy)
    {
        auto &lo = *r[0];
        auto &hi = *r[1];
        internal_data0(internal_data0(lo)) = internal_data0(std::get<0>(proxy.in));
        internal_data1(internal_data0(lo)) = internal_data0(std::get<1>(proxy.in));
        internal_data0(internal_data1(lo)) = internal_data0(std::get<2>(proxy.in));
        internal_data1(internal_data1(lo)) = internal_data0(std::get<3>(proxy.in));
        internal_data0(internal_data0(hi)) = internal_data1(std::get<0>(proxy.in));
        internal_data1(internal_data0(hi)) = internal_data1(std::get<1>(proxy.in));
        internal_data0(internal_data1(hi)) = internal_data1(std::get<2>(proxy.in));
        internal_data1(internal_data1(hi)) = internal_data1(std::get<3>(proxy.in));
    }
    template <int L, typename T, std::size_t N, typename V>
    inline enable_if<(L == 4 && N > 1), void> transpose_impl(
        SimdArray<T, N, V, 1> *Vc_RESTRICT r[],
        const TransposeProxy<SimdArray<T, N, V, 1>, SimdArray<T, N, V, 1>,
                             SimdArray<T, N, V, 1>, SimdArray<T, N, V, 1>> &proxy)
    {
        SimdArray<T, N, V, 1> *Vc_RESTRICT r0[L / 2] = {r[0], r[1]};
        SimdArray<T, N, V, 1> *Vc_RESTRICT r1[L / 2] = {r[2], r[3]};
        using H = SimdArray<T, 2>;
        transpose_impl<2>(
            &r0[0], TransposeProxy<H, H, H, H>{internal_data0(std::get<0>(proxy.in)),
                                               internal_data0(std::get<1>(proxy.in)),
                                               internal_data0(std::get<2>(proxy.in)),
                                               internal_data0(std::get<3>(proxy.in))});
        transpose_impl<2>(
            &r1[0], TransposeProxy<H, H, H, H>{internal_data1(std::get<0>(proxy.in)),
                                               internal_data1(std::get<1>(proxy.in)),
                                               internal_data1(std::get<2>(proxy.in)),
                                               internal_data1(std::get<3>(proxy.in))});
    }
    /* TODO:
    template <typename T, std::size_t N, typename V, std::size_t VSize>
    inline enable_if<(N > VSize), void> transpose_impl(
        std::array<SimdArray<T, N, V, VSize> * Vc_RESTRICT, 4> & r,
        const TransposeProxy<SimdArray<T, N, V, VSize>, SimdArray<T, N, V, VSize>,
                             SimdArray<T, N, V, VSize>, SimdArray<T, N, V, VSize>> &proxy)
    {
        typedef SimdArray<T, N, V, VSize> SA;
        std::array<typename SA::storage_type0 * Vc_RESTRICT, 4> r0 = {
            {&internal_data0(*r[0]), &internal_data0(*r[1]), &internal_data0(*r[2]),
             &internal_data0(*r[3])}};
        transpose_impl(
            r0, TransposeProxy<typename SA::storage_type0, typename SA::storage_type0,
                               typename SA::storage_type0, typename SA::storage_type0>{
                    internal_data0(std::get<0>(proxy.in)),
                    internal_data0(std::get<1>(proxy.in)),
                    internal_data0(std::get<2>(proxy.in)),
                    internal_data0(std::get<3>(proxy.in))});

        std::array<typename SA::storage_type1 * Vc_RESTRICT, 4> r1 = {
            {&internal_data1(*r[0]), &internal_data1(*r[1]), &internal_data1(*r[2]),
             &internal_data1(*r[3])}};
        transpose_impl(
            r1, TransposeProxy<typename SA::storage_type1, typename SA::storage_type1,
                               typename SA::storage_type1, typename SA::storage_type1>{
                    internal_data1(std::get<0>(proxy.in)),
                    internal_data1(std::get<1>(proxy.in)),
                    internal_data1(std::get<2>(proxy.in)),
                    internal_data1(std::get<3>(proxy.in))});
    }
    */
}  // namespace Common

// Traits static assertions {{{1
static_assert(Traits::has_no_allocated_data<const volatile Vc::SimdArray<int, 4> &>::value, "");
static_assert(Traits::has_no_allocated_data<const volatile Vc::SimdArray<int, 4>>::value, "");
static_assert(Traits::has_no_allocated_data<volatile Vc::SimdArray<int, 4> &>::value, "");
static_assert(Traits::has_no_allocated_data<volatile Vc::SimdArray<int, 4>>::value, "");
static_assert(Traits::has_no_allocated_data<const Vc::SimdArray<int, 4> &>::value, "");
static_assert(Traits::has_no_allocated_data<const Vc::SimdArray<int, 4>>::value, "");
static_assert(Traits::has_no_allocated_data<Vc::SimdArray<int, 4>>::value, "");
static_assert(Traits::has_no_allocated_data<Vc::SimdArray<int, 4> &&>::value, "");
// }}}1
/// @}

} // namespace Vc_VERSIONED_NAMESPACE

#endif // VC_COMMON_SIMDARRAY_H_

// vim: foldmethod=marker
