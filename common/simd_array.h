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

#ifndef VC_COMMON_SIMD_ARRAY_H
#define VC_COMMON_SIMD_ARRAY_H

#include <type_traits>
#include <array>

#include "writemaskedvector.h"
#include "simd_array_data.h"
#include "simd_mask_array.h"
#include "utility.h"
#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace internal
{
#define Vc_BINARY_FUNCTION__(name__)                                                               \
    template <typename T, std::size_t N, typename V, std::size_t M>                                \
    simd_array<T, N, V, M> Vc_INTRINSIC Vc_PURE                                                    \
        name__(const simd_array<T, N, V, M> &l, const simd_array<T, N, V, M> &r)                   \
    {                                                                                              \
        return {name__(internal_data0(l), internal_data0(r)),                                      \
                name__(internal_data1(l), internal_data1(r))};                                     \
    }                                                                                              \
    template <typename T, std::size_t N, typename V>                                               \
    simd_array<T, N, V, N> Vc_INTRINSIC Vc_PURE                                                    \
        name__(const simd_array<T, N, V, N> &l, const simd_array<T, N, V, N> &r)                   \
    {                                                                                              \
        return {name__(internal_data(l), internal_data(r))};                                       \
    }
Vc_BINARY_FUNCTION__(min)
Vc_BINARY_FUNCTION__(max)
#undef Vc_BINARY_FUNCTION__
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

// === having simd_array<T, N> in the Vc namespace leads to a ABI bug ===
//
// simd_array<double, 4> can be { double[4] }, { __m128d[2] }, or { __m256d } even though the type
// is the same.
// The question is, what should simd_array focus on?
// a) A type that makes interfacing between different implementations possible?
// b) Or a type that makes fixed size SIMD easier and efficient?
//
// a) can be achieved by using a union with T[N] as one member. But this may have more serious
// performance implications than only less efficient parameter passing (because compilers have a
// much harder time wrt. aliasing issues). Also alignment would need to be set to the sizeof in
// order to be compatible with targets with larger alignment requirements.
// But, the in-memory representation of masks is not portable. Thus, at the latest with AVX-512,
// there would be a problem with requiring simd_mask_array<T, N> to be an ABI compatible type.
// AVX-512 uses one bit per boolean, whereas SSE/AVX use sizeof(T) Bytes per boolean. Conversion
// between the two representations is not a trivial operation. Therefore choosing one or the other
// representation will have a considerable impact for the targets that do not use this
// representation. Since the future probably belongs to one bit per boolean representation, I would
// go with that choice.
//
// b) requires that simd_array<T, N> != simd_array<T, N> if
// simd_array<T, N>::vector_type != simd_array<T, N>::vector_type
//
// Therefore use simd_array<T, N, V>, where V follows from the above.
template <typename T,
          std::size_t N,
          typename VectorType = Common::select_best_vector_type<T, N>,
          std::size_t VectorSize = VectorType::size()  // this last parameter is only used for
                                                       // specialization of N == VectorSize
          >
class alignas(N * sizeof(VectorType)) simd_array;

template <typename T, std::size_t N, typename VectorType> class simd_array<T, N, VectorType, N>
{
    static_assert(std::is_same<T, double>::value || std::is_same<T, float>::value ||
                      std::is_same<T, int32_t>::value ||
                      std::is_same<T, uint32_t>::value ||
                      std::is_same<T, int16_t>::value ||
                      std::is_same<T, uint16_t>::value,
                  "simd_array<T, N> may only be used with T = { double, float, int32_t, uint32_t, "
                  "int16_t, uint16_t }");

public:
    using vector_type = VectorType;
    using vectorentry_type = typename vector_type::VectorEntryType;
    using value_type = T;
    using mask_type = simd_mask_array<T, N, vector_type>;
    using index_type = simd_array<int, N>;
    static constexpr std::size_t size() { return N; }
    using Mask = mask_type;
    using VectorEntryType = vectorentry_type;
    using EntryType = value_type;
    using IndexType = index_type;
    static constexpr std::size_t Size = size();

    // zero init
    simd_array() = default;

    // default copy ctor/operator
    simd_array(const simd_array &) = default;
    simd_array(simd_array &&) = default;
    simd_array &operator=(const simd_array &) = default;

    // broadcast
    Vc_INTRINSIC simd_array(value_type a) : data(a) {}
    template <
        typename U,
        typename = enable_if<std::is_same<U, int>::value && !std::is_same<int, value_type>::value>>
    simd_array(U a)
        : simd_array(static_cast<value_type>(a))
    {
    }

    // implicit casts
    template <typename U, typename V>
    Vc_INTRINSIC simd_array(const simd_array<U, N, V> &x, enable_if<N == V::size()> = nullarg)
        : data(simd_cast<vector_type>(internal_data(x)))
    {
    }
    template <typename U, typename V>
    Vc_INTRINSIC simd_array(const simd_array<U, N, V> &x,
                            enable_if<(N > V::size() && N <= 2 * V::size())> = nullarg)
        : data(simd_cast<vector_type>(internal_data(internal_data0(x)), internal_data(internal_data1(x))))
    {
    }
    template <typename U, typename V>
    Vc_INTRINSIC simd_array(const simd_array<U, N, V> &x,
                            enable_if<(N > 2 * V::size() && N <= 4 * V::size())> = nullarg)
        : data(simd_cast<vector_type>(internal_data(internal_data0(internal_data0(x))),
                                      internal_data(internal_data1(internal_data0(x))),
                                      internal_data(internal_data0(internal_data1(x))),
                                      internal_data(internal_data1(internal_data1(x)))))
    {
    }

    template <typename V, std::size_t Pieces, std::size_t Index>
    Vc_INTRINSIC simd_array(Common::Segment<V, Pieces, Index> &&x)
        : data(simd_cast<vector_type, Index>(x.data))
    {
    }

    Vc_INTRINSIC simd_array(const std::initializer_list<value_type> &init)
        : data(init.begin(), Vc::Unaligned)
    {
        // TODO: when initializer_list::size() becomes constexpr (C++14) make this a static_assert
        VC_ASSERT(init.size() == size());
    }

    // forward all remaining ctors
    template <typename... Args,
              typename = enable_if<!Traits::is_cast_arguments<Args...>::value &&
                                   !Traits::is_initializer_list<Args...>::value>>
    explicit Vc_INTRINSIC simd_array(Args &&... args)
        : data(std::forward<Args>(args)...)
    {
    }

    template <std::size_t Offset>
    explicit Vc_INTRINSIC simd_array(
        Common::AddOffset<VectorSpecialInitializerIndexesFromZero::IEnum, Offset>)
        : data(VectorSpecialInitializerIndexesFromZero::IndexesFromZero)
    {
        data += value_type(Offset);
    }

    Vc_INTRINSIC void setZero() { data.setZero(); }
    Vc_INTRINSIC void setZero(mask_type k) { data.setZero(internal_data(k)); }
    Vc_INTRINSIC void setZeroInverted() { data.setZeroInverted(); }
    Vc_INTRINSIC void setZeroInverted(mask_type k) { data.setZeroInverted(internal_data(k)); }

    // internal: execute specified Operation
    template <typename Op, typename... Args>
    static Vc_INTRINSIC simd_array fromOperation(Op op, Args &&... args)
    {
        simd_array r;
        op(r.data, Common::actual_value(op, std::forward<Args>(args))...);
        return r;
    }

    static Vc_INTRINSIC simd_array Zero()
    {
        return simd_array(VectorSpecialInitializerZero::Zero);
    }
    static Vc_INTRINSIC simd_array One()
    {
        return simd_array(VectorSpecialInitializerOne::One);
    }
    static Vc_INTRINSIC simd_array IndexesFromZero()
    {
        return simd_array(VectorSpecialInitializerIndexesFromZero::IndexesFromZero);
    }
    static Vc_INTRINSIC simd_array Random()
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

    Vc_INTRINSIC simd_array operator-() const
    {
        return {-data};
    }

    template <typename U,
              typename = enable_if<std::is_integral<T>::value && std::is_integral<U>::value>>
    Vc_INTRINSIC Vc_CONST simd_array operator<<(U x) const
    {
        return {data << x};
    }
    template <typename U,
              typename = enable_if<std::is_integral<T>::value && std::is_integral<U>::value>>
    Vc_INTRINSIC simd_array &operator<<=(U x)
    {
        data <<= x;
        return *this;
    }
    template <typename U,
              typename = enable_if<std::is_integral<T>::value && std::is_integral<U>::value>>
    Vc_INTRINSIC Vc_CONST simd_array operator>>(U x) const
    {
        return {data >> x};
    }
    template <typename U,
              typename = enable_if<std::is_integral<T>::value && std::is_integral<U>::value>>
    Vc_INTRINSIC simd_array &operator>>=(U x)
    {
        data >>= x;
        return *this;
    }

#define Vc_BINARY_OPERATOR_(op)                                                                    \
    Vc_INTRINSIC Vc_CONST simd_array operator op(const simd_array &rhs) const                      \
    {                                                                                              \
        return {data op rhs.data};                                                                 \
    }                                                                                              \
    Vc_INTRINSIC simd_array &operator op##=(const simd_array & rhs)                                \
    {                                                                                              \
        data op## = rhs.data;                                                                      \
        return *this;                                                                              \
    }
    VC_ALL_ARITHMETICS(Vc_BINARY_OPERATOR_)
    VC_ALL_BINARY(Vc_BINARY_OPERATOR_)
    VC_ALL_SHIFTS(Vc_BINARY_OPERATOR_)
#undef Vc_BINARY_OPERATOR_

#define Vc_COMPARES(op)                                                                            \
    Vc_INTRINSIC mask_type operator op(const simd_array &rhs) const                                \
    {                                                                                              \
        return {data op rhs.data};                                                                 \
    }
    VC_ALL_COMPARES(Vc_COMPARES)
#undef Vc_COMPARES

    Vc_INTRINSIC decltype(std::declval<vector_type &>()[0]) operator[](std::size_t i)
    {
        return data[i];
    }
    Vc_INTRINSIC value_type operator[](std::size_t i) const { return data[i]; }

    Vc_INTRINSIC Common::WriteMaskedVector<simd_array, mask_type> operator()(const mask_type &k)
    {
        return {this, k};
    }

    Vc_INTRINSIC void assign(const simd_array &v, const mask_type &k)
    {
        data.assign(v.data, internal_data(k));
    }

    Vc_INTRINSIC const vectorentry_type *begin() const
    {
        return reinterpret_cast<const vectorentry_type *>(&data.data());
    }

    Vc_INTRINSIC decltype(&std::declval<VectorType &>()[0]) begin()
    {
        return &data[0];
    }

    Vc_INTRINSIC const vectorentry_type *end() const
    {
        return reinterpret_cast<const vectorentry_type *>(&data + 1);
    }

    // reductions ////////////////////////////////////////////////////////
#define Vc_REDUCTION_FUNCTION__(name__)                                                            \
    Vc_INTRINSIC Vc_PURE value_type name__() const { return data.name__(); }                       \
                                                                                                   \
    Vc_INTRINSIC Vc_PURE value_type name__(mask_type mask) const                                   \
    {                                                                                              \
        return data.name__(internal_data(mask));                                                   \
    }
    Vc_REDUCTION_FUNCTION__(min)
    Vc_REDUCTION_FUNCTION__(max)
    Vc_REDUCTION_FUNCTION__(product)
    Vc_REDUCTION_FUNCTION__(sum)
#undef Vc_REDUCTION_FUNCTION__
    Vc_INTRINSIC Vc_PURE simd_array partialSum() const { return data.partialSum(); }

    Vc_INTRINSIC void fusedMultiplyAdd(const simd_array &factor, const simd_array &summand)
    {
        data.fusedMultiplyAdd(internal_data(factor), internal_data(summand));
    }

    template <typename F> Vc_INTRINSIC simd_array apply(F &&f) const
    {
        return {data.apply(std::forward<F>(f))};
    }
    template <typename F> Vc_INTRINSIC simd_array apply(F &&f, const mask_type &k) const
    {
        return {data.apply(std::forward<F>(f), k)};
    }

    friend Vc_INTRINSIC VectorType &internal_data(simd_array &x) { return x.data; }
    friend Vc_INTRINSIC const VectorType &internal_data(const simd_array &x) { return x.data; }

    /// \internal
    Vc_INTRINSIC simd_array(VectorType &&x) : data(std::move(x)) {}
private:
    VectorType data;
};

template <typename T, std::size_t N, typename VectorType, std::size_t> class simd_array
{
    static_assert(std::is_same<T,   double>::value ||
                  std::is_same<T,    float>::value ||
                  std::is_same<T,  int32_t>::value ||
                  std::is_same<T, uint32_t>::value ||
                  std::is_same<T,  int16_t>::value ||
                  std::is_same<T, uint16_t>::value, "simd_array<T, N> may only be used with T = { double, float, int32_t, uint32_t, int16_t, uint16_t }");

    static constexpr std::size_t N0 = Common::nextPowerOfTwo(N - N / 2);

    using storage_type0 = simd_array<T, N0>;
    using storage_type1 = simd_array<T, N - N0>;

    using Split = Common::Split<storage_type0::size()>;

public:
    using vector_type = VectorType;
    using vectorentry_type = typename storage_type0::vectorentry_type;
    typedef vectorentry_type alias_type Vc_MAY_ALIAS;
    using value_type = T;
    using mask_type = simd_mask_array<T, N, vector_type>;
    using index_type = simd_array<int, N>;
    static constexpr std::size_t size() { return N; }
    using Mask = mask_type;
    using VectorEntryType = vectorentry_type;
    using EntryType = value_type;
    using IndexType = index_type;
    static constexpr std::size_t Size = size();

    //////////////////// constructors //////////////////

    // zero init
    simd_array() = default;

    // default copy ctor/operator
    simd_array(const simd_array &) = default;
    simd_array(simd_array &&) = default;
    simd_array &operator=(const simd_array &) = default;

    // broadcast
    Vc_INTRINSIC simd_array(value_type a) : data0(a), data1(a) {}
    template <
        typename U,
        typename = enable_if<std::is_same<U, int>::value && !std::is_same<int, value_type>::value>>
    simd_array(U a)
        : simd_array(static_cast<value_type>(a))
    {
    }

    // load ctor
    template <typename U,
              typename Flags = DefaultLoadTag,
              typename = enable_if<Traits::is_load_store_flag<Flags>::value>>
    explicit Vc_INTRINSIC simd_array(const U *mem, Flags f = Flags())
        : data0(mem, f), data1(mem + storage_type0::size(), f)
    {
    }

    // initializer list
    Vc_INTRINSIC simd_array(const std::initializer_list<value_type> &init)
        : data0(init.begin(), Vc::Unaligned)
        , data1(init.begin() + storage_type0::size(), Vc::Unaligned)
    {
        // TODO: when initializer_list::size() becomes constexpr (C++14) make this a static_assert
        VC_ASSERT(init.size() == size());
    }

    // gather
    template <typename U,
              typename... Args,
              typename = enable_if<Traits::is_gather_signature<U *, Args...>::value>>
    explicit Vc_INTRINSIC simd_array(U *mem, Args &&... args)
        : data0(mem, Split::lo(std::forward<Args>(args))...)
        , data1(mem, Split::hi(std::forward<Args>(args))...)
    {
    }

    // forward all remaining ctors
    template <typename... Args,
              typename = enable_if<!Traits::is_cast_arguments<Args...>::value &&
                                   !Traits::is_initializer_list<Args...>::value &&
                                   !Traits::is_gather_signature<Args...>::value &&
                                   !Traits::is_load_arguments<Args...>::value>>
    explicit Vc_INTRINSIC simd_array(Args &&... args)
        : data0(Split::lo(std::forward<Args>(args))...)
        , data1(Split::hi(std::forward<Args>(args))...)
    {
    }

    // explicit casts
    template <typename V>
    Vc_INTRINSIC explicit simd_array(
        V &&x,
        enable_if<(Traits::is_simd_vector<V>::value && Traits::simd_vector_size<V>::value == N &&
                   !(std::is_convertible<Traits::entry_type_of<V>, T>::value &&
                     Traits::is_simd_array<V>::value))> = nullarg)
        : data0(Split::lo(x)), data1(Split::hi(x))
    {
    }

    // implicit casts
    template <typename V>
    Vc_INTRINSIC simd_array(
        V &&x,
        enable_if<(Traits::is_simd_array<V>::value && Traits::simd_vector_size<V>::value == N &&
                   std::is_convertible<Traits::entry_type_of<V>, T>::value)> = nullarg)
        : data0(Split::lo(x)), data1(Split::hi(x))
    {
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
    static Vc_INTRINSIC simd_array fromOperation(Op op, Args &&... args)
    {
        simd_array r = {storage_type0::fromOperation(op, Split::lo(std::forward<Args>(args))...),
                        storage_type1::fromOperation(op, Split::lo(std::forward<Args>(args))...)};
        return r;
    }

    static Vc_INTRINSIC simd_array Zero()
    {
        return simd_array(VectorSpecialInitializerZero::Zero);
    }
    static Vc_INTRINSIC simd_array One()
    {
        return simd_array(VectorSpecialInitializerOne::One);
    }
    static Vc_INTRINSIC simd_array IndexesFromZero()
    {
        return simd_array(VectorSpecialInitializerIndexesFromZero::IndexesFromZero);
    }
    static Vc_INTRINSIC simd_array Random()
    {
        return fromOperation(Common::Operations::random());
    }

    template <typename U, typename... Args> Vc_INTRINSIC void load(const U *mem, Args &&... args)
    {
        data0.load(mem, Split::lo(std::forward<Args>(args))...);
        data1.load(mem + storage_type0::size(), Split::hi(std::forward<Args>(args))...);
    }

    template <typename U, typename... Args> Vc_INTRINSIC void store(U *mem, Args &&... args) const
    {
        data0.store(mem, Split::lo(std::forward<Args>(args))...);
        data1.store(mem + storage_type0::size(), Split::hi(std::forward<Args>(args))...);
    }

    Vc_INTRINSIC mask_type operator!() const
    {
        return {!data0, !data1};
    }

    Vc_INTRINSIC simd_array operator-() const
    {
        return {-data0, -data1};
    }

    template <typename U,
              typename = enable_if<std::is_integral<T>::value && std::is_integral<U>::value>>
    Vc_INTRINSIC Vc_CONST simd_array operator<<(U x) const
    {
        return {data0 << x, data1 << x};
    }
    template <typename U,
              typename = enable_if<std::is_integral<T>::value && std::is_integral<U>::value>>
    Vc_INTRINSIC simd_array &operator<<=(U x)
    {
        data0 <<= x;
        data1 <<= x;
        return *this;
    }
    template <typename U,
              typename = enable_if<std::is_integral<T>::value && std::is_integral<U>::value>>
    Vc_INTRINSIC Vc_CONST simd_array operator>>(U x) const
    {
        return {data0 >> x, data1 >> x};
    }
    template <typename U,
              typename = enable_if<std::is_integral<T>::value && std::is_integral<U>::value>>
    Vc_INTRINSIC simd_array &operator>>=(U x)
    {
        data0 >>= x;
        data1 >>= x;
        return *this;
    }

#define Vc_BINARY_OPERATOR_(op)                                                                    \
    Vc_INTRINSIC Vc_CONST simd_array operator op(const simd_array &rhs) const                      \
    {                                                                                              \
        return {data0 op rhs.data0, data1 op rhs.data1};                                           \
    }                                                                                              \
    Vc_INTRINSIC simd_array &operator op##=(const simd_array & rhs)                                \
    {                                                                                              \
        data0 op## = rhs.data0;                                                                    \
        data1 op## = rhs.data1;                                                                    \
        return *this;                                                                              \
    }
    VC_ALL_ARITHMETICS(Vc_BINARY_OPERATOR_)
    VC_ALL_BINARY(Vc_BINARY_OPERATOR_)
    VC_ALL_SHIFTS(Vc_BINARY_OPERATOR_)
#undef Vc_BINARY_OPERATOR_

#define Vc_COMPARES(op)                                                                            \
    Vc_INTRINSIC mask_type operator op(const simd_array &rhs) const                                \
    {                                                                                              \
        return {data0 op rhs.data0, data1 op rhs.data1};                                           \
    }
    VC_ALL_COMPARES(Vc_COMPARES)
#undef Vc_COMPARES

    Vc_INTRINSIC value_type operator[](std::size_t i) const
    {
        const auto tmp = reinterpret_cast<const alias_type *>(&data0);
        return tmp[i];
    }

    Vc_INTRINSIC alias_type &operator[](std::size_t i)
    {
        auto tmp = reinterpret_cast<alias_type *>(&data0);
        return tmp[i];
    }

    Vc_INTRINSIC Common::WriteMaskedVector<simd_array, mask_type> operator()(const mask_type &k)
    {
        return {this, k};
    }

    Vc_INTRINSIC void assign(const simd_array &v, const mask_type &k)
    {
        data0.assign(v.data0, internal_data0(k));
        data1.assign(v.data1, internal_data1(k));
    }

    Vc_INTRINSIC const vectorentry_type *begin() const
    {
        return data0.begin();
    }

    Vc_INTRINSIC vectorentry_type *begin()
    {
        return data0.begin();
    }

    Vc_INTRINSIC const vectorentry_type *end() const
    {
        return data0.end();
    }

    // reductions ////////////////////////////////////////////////////////
#define Vc_REDUCTION_FUNCTION__(name__, binary_fun__)                                              \
    template <typename ForSfinae = void>                                                           \
    Vc_INTRINSIC enable_if<                                                                        \
        std::is_same<ForSfinae, void>::value &&storage_type0::size() == storage_type1::size(),     \
        value_type> name__() const                                                                 \
    {                                                                                              \
        return binary_fun__(data0, data1).name__();                                                \
    }                                                                                              \
                                                                                                   \
    template <typename ForSfinae = void>                                                           \
    Vc_INTRINSIC enable_if<                                                                        \
        std::is_same<ForSfinae, void>::value &&storage_type0::size() != storage_type1::size(),     \
        value_type> name__() const                                                                 \
    {                                                                                              \
        return binary_fun__(data0.name__(), data1.name__());                                       \
    }                                                                                              \
                                                                                                   \
    Vc_INTRINSIC value_type name__(const mask_type &mask) const                                    \
    {                                                                                              \
        if (VC_IS_UNLIKELY(Split::lo(mask).isEmpty())) {                                           \
            return data1.name__(Split::hi(mask));                                                  \
        } else if (VC_IS_UNLIKELY(Split::hi(mask).isEmpty())) {                                    \
            return data0.name__(Split::lo(mask));                                                  \
        } else {                                                                                   \
            return binary_fun__(data0.name__(Split::lo(mask)), data1.name__(Split::hi(mask)));     \
        }                                                                                          \
    }
    Vc_REDUCTION_FUNCTION__(min, Vc::internal::min)
    Vc_REDUCTION_FUNCTION__(max, Vc::internal::max)
    Vc_REDUCTION_FUNCTION__(product, internal::product_helper__)
    Vc_REDUCTION_FUNCTION__(sum, internal::sum_helper__)
#undef Vc_REDUCTION_FUNCTION__
    Vc_INTRINSIC Vc_PURE simd_array partialSum() const
    {
        auto ps0 = data0.partialSum();
        auto tmp = data1;
        tmp[0] += ps0[data0.size() - 1];
        return {std::move(ps0), tmp.partialSum()};
    }

    void fusedMultiplyAdd(const simd_array &factor, const simd_array &summand)
    {
        data0.fusedMultiplyAdd(Split::lo(factor), Split::lo(summand));
        data1.fusedMultiplyAdd(Split::hi(factor), Split::hi(summand));
    }

    template <typename F> Vc_INTRINSIC simd_array apply(F &&f) const
    {
        return {data0.apply(f), data1.apply(f)};
    }
    template <typename F> Vc_INTRINSIC simd_array apply(F &&f, const mask_type &k) const
    {
        return {data0.apply(f, Split::lo(k)), data1.apply(f, Split::hi(k))};
    }

    friend Vc_INTRINSIC storage_type0 &internal_data0(simd_array &x) { return x.data0; }
    friend Vc_INTRINSIC storage_type1 &internal_data1(simd_array &x) { return x.data1; }
    friend Vc_INTRINSIC const storage_type0 &internal_data0(const simd_array &x) { return x.data0; }
    friend Vc_INTRINSIC const storage_type1 &internal_data1(const simd_array &x) { return x.data1; }

    /// \internal
    Vc_INTRINSIC simd_array(storage_type0 &&x, storage_type1 &&y)
        : data0(std::move(x)), data1(std::move(y))
    {
    }
private:
    storage_type0 data0;
    storage_type1 data1;
};

// binary operators ////////////////////////////////////////////
namespace result_vector_type_internal
{
template <typename T>
using type = typename std::remove_cv<typename std::remove_reference<T>::type>::type;

template <typename L,
          typename R,
          std::size_t N = Traits::is_simd_array<L>::value ? Traits::simd_vector_size<L>::value
                                                          : Traits::simd_vector_size<R>::value,
          bool = (Traits::is_simd_array<L>::value ||
                  Traits::is_simd_array<R>::value)  // one of the operands must be a simd_array
                 &&
                 !std::is_same<type<L>, type<R>>::value  // if the operands are of the same type
                                                         // use the member function
                 &&
                 (std::is_arithmetic<type<L>>::value ||
                  std::is_arithmetic<type<R>>::value  // one of the operands is a scalar type
                  ||
                  (Traits::is_simd_vector<L>::value && !Traits::is_simd_array<L>::value) ||
                  (Traits::is_simd_vector<R>::value &&
                   !Traits::is_simd_array<R>::value)  // or one of the operands is Vector<T>
                  ) > struct evaluate;

template <typename L, typename R, std::size_t N> struct evaluate<L, R, N, true>
{
private:
    using LScalar = Traits::entry_type_of<L>;
    using RScalar = Traits::entry_type_of<R>;

    template <bool B, typename True, typename False>
    using conditional = typename std::conditional<B, True, False>::type;

public:
    // In principle we want the exact same rules for simd_array<T> ⨉ simd_array<U> as the standard
    // defines for T ⨉ U. BUT: short ⨉ short returns int (because all integral types smaller than
    // int are promoted to int before any operation). This would imply that SIMD types with integral
    // types smaller than int are more or less useless - and you could use simd_array<int> from the
    // start. Therefore we special-case those operations where the scalar type of both operands is
    // integral and smaller than int.
    using type = simd_array<
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
    std::is_same<result_vector_type<short int, Vc_0::simd_array<short unsigned int, 32ul>>,
                 Vc_0::simd_array<short unsigned int, 32ul>>::value,
    "result_vector_type does not work");

#define Vc_BINARY_OPERATORS_(op__)                                                                 \
    template <typename L, typename R>                                                              \
    Vc_INTRINSIC result_vector_type<L, R> operator op__(L &&lhs, R &&rhs)                          \
    {                                                                                              \
        using Return = result_vector_type<L, R>;                                                   \
        return Return(std::forward<L>(lhs)) op__ Return(std::forward<R>(rhs));                     \
    }
VC_ALL_ARITHMETICS(Vc_BINARY_OPERATORS_)
VC_ALL_BINARY(Vc_BINARY_OPERATORS_)
#undef Vc_BINARY_OPERATORS_

// math functions
template <typename T, std::size_t N> simd_array<T, N> abs(const simd_array<T, N> &x)
{
    return simd_array<T, N>::fromOperation(Common::Operations::Abs(), x);
}
template <typename T, std::size_t N> simd_mask_array<T, N> isnan(const simd_array<T, N> &x)
{
    return simd_mask_array<T, N>::fromOperation(Common::Operations::Isnan(), x);
}
template <typename T, std::size_t N>
simd_array<T, N> frexp(const simd_array<T, N> &x, simd_array<int, N> *e)
{
    return simd_array<T, N>::fromOperation(Common::Operations::Frexp(), x, e);
}
template <typename T, std::size_t N>
simd_array<T, N> ldexp(const simd_array<T, N> &x, const simd_array<int, N> &e)
{
    return simd_array<T, N>::fromOperation(Common::Operations::Ldexp(), x, e);
}

} // namespace Vc_VERSIONED_NAMESPACE

#include "undomacros.h"

#endif // VC_COMMON_SIMD_ARRAY_H
