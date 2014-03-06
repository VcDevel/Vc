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
#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{

template <typename T,
          std::size_t N,
          typename VectorType = typename Common::select_best_vector_type<N,
#ifdef VC_IMPL_AVX
                                                                         Vc::Vector<T>,
                                                                         Vc::SSE::Vector<T>,
                                                                         Vc::Scalar::Vector<T>
#elif defined(VC_IMPL_Scalar)
                                                                         Vc::Vector<T>
#else
                                                                         Vc::Vector<T>,
                                                                         Vc::Scalar::Vector<T>
#endif
                                                                         >::type,
          std::size_t VectorSize = VectorType::size()  // this last parameter is only used for
                                                       // specialization of N == VectorSize
          >
class simd_array;

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

    // internal: execute specified Operation
    template <typename Op, typename... Args>
    static Vc_INTRINSIC simd_array fromOperation(Op op, Args &&... args)
    {
        simd_array r;
        op(r.data, std::forward<Args>(args)...);
        return r;
    }

    template <typename... Args> Vc_INTRINSIC void load(Args &&... args)
    {
        data.load(std::forward<Args>(args)...);
    }

#define Vc_ARITHMETIC(op)                                                                          \
    Vc_INTRINSIC simd_array operator op(const simd_array &rhs) const                               \
    {                                                                                              \
        return {data op rhs.data};                                                                 \
    }
    VC_ALL_ARITHMETICS(Vc_ARITHMETIC)
#undef Vc_ARITHMETIC

#define Vc_COMPARES(op)                                                                            \
    Vc_INTRINSIC bool operator op(const simd_array &rhs) const { return data op rhs.data; }
    VC_ALL_COMPARES(Vc_COMPARES)
#undef Vc_COMPARES

    Vc_INTRINSIC value_type operator[](std::size_t i) const
    {
        return data[i];
    }

    Vc_INTRINSIC const vectorentry_type *begin() const
    {
        return reinterpret_cast<const vectorentry_type *>(&data);
    }

    Vc_INTRINSIC const vectorentry_type *end() const
    {
        return reinterpret_cast<const vectorentry_type *>(&data + 1);
    }

private:
    Vc_INTRINSIC simd_array(VectorType &&x) : data(std::move(x)) {}
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

    using storage_type0 = simd_array<T, N / 2>;
    using storage_type1 = simd_array<T, N - N / 2>;

    using Split = Common::Split<storage_type0::size()>;

public:
    using vector_type = VectorType;
    using vectorentry_type = typename storage_type0::vectorentry_type;
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
    Vc_INTRINSIC simd_array(value_type a) : data0(a), data1(a) {}
    template <
        typename U,
        typename = enable_if<std::is_same<U, int>::value && !std::is_same<int, value_type>::value>>
    simd_array(U a)
        : simd_array(static_cast<value_type>(a))
    {
    }

    // forward all remaining ctors
    template <typename... Args,
              typename = enable_if<!Traits::IsCastArguments<Args...>::value &&
                                   !Traits::is_initializer_list<Args...>::value>>
    explicit Vc_INTRINSIC simd_array(Args &&... args)
        : data0(Split::lo(std::forward<Args>(args))...)
        , data1(Split::hi(std::forward<Args>(args))...)
    {
    }

    // implicit casts
    template <typename U, typename V>
    Vc_INTRINSIC simd_array(const simd_array<U, N, V> &x)
        : data0(Split::lo(x)), data1(Split::hi(x))
    {
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

    template <typename... Args> Vc_INTRINSIC void load(Args &&... args)
    {
        data0.load(Split::lo(std::forward<Args>(args))...);
        data1.load(Split::hi(std::forward<Args>(args))...);
    }

    template <typename U>
    using result_vector_type = simd_array<decltype(std::declval<T>() + std::declval<U>()), N>;

    template <typename U>
    Vc_INTRINSIC result_vector_type<U> operator+(const simd_array<U, N> &rhs) const
    {
        return result_vector_type<U>{*this} + result_vector_type<U>{rhs};
    }

#define Vc_ARITHMETIC(op)                                                                          \
    Vc_INTRINSIC simd_array operator op(const simd_array &rhs) const                               \
    {                                                                                              \
        return {data0 op rhs.data0, data1 op rhs.data1};                                           \
    }
    VC_ALL_ARITHMETICS(Vc_ARITHMETIC)
#undef Vc_ARITHMETIC

#define Vc_COMPARES(op)                                                                            \
    Vc_INTRINSIC bool operator op(const simd_array &rhs) const                                     \
    {                                                                                              \
        return data0 op rhs.data0 && data1 op rhs.data1;                                           \
    }
    VC_ALL_COMPARES(Vc_COMPARES)
#undef Vc_COMPARES

    Vc_INTRINSIC value_type operator[](std::size_t i) const
    {
        return data0.begin()[i];
    }

    Vc_INTRINSIC const vectorentry_type *begin() const
    {
        return data0.begin();
    }

    Vc_INTRINSIC const vectorentry_type *end() const
    {
        return data0.end();
    }

private:
    Vc_INTRINSIC simd_array(storage_type0 &&x, storage_type1 &&y)
        : data0(std::move(x)), data1(std::move(y))
    {
    }
    storage_type0 data0;
    storage_type1 data1;
};

#if 0
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
          typename VectorType = typename Common::select_best_vector_type<N,
#ifdef VC_IMPL_AVX
                                                                         Vc::Vector<T>,
                                                                         Vc::SSE::Vector<T>,
                                                                         Vc::Scalar::Vector<T>
#elif defined(VC_IMPL_Scalar)
                                                                         Vc::Vector<T>
#else
                                                                         Vc::Vector<T>,
                                                                         Vc::Scalar::Vector<T>
#endif
                                                                         >::type>
class simd_array
{
    static_assert(std::is_same<T,   double>::value ||
                  std::is_same<T,    float>::value ||
                  std::is_same<T,  int32_t>::value ||
                  std::is_same<T, uint32_t>::value ||
                  std::is_same<T,  int16_t>::value ||
                  std::is_same<T, uint16_t>::value, "simd_array<T, N> may only be used with T = { double, float, int32_t, uint32_t, int16_t, uint16_t }");

    static_assert((N & (N - 1)) == 0, "simd_array<T, N> must be used with a power of two value for N.");

public:
    using vector_type = VectorType;
    typedef T value_type;
    typedef simd_mask_array<T, N, vector_type> mask_type;
    typedef simd_array<int, N> index_type;

    static constexpr std::size_t size() { return N; }
    static constexpr std::size_t register_count = size() > vector_type::Size ? size() / vector_type::Size : 1;

    typedef Common::ArrayData<vector_type, register_count> storage_type;

    // Vc compat:
    typedef mask_type Mask;
    typedef value_type EntryType;
    typedef index_type IndexType;
    static constexpr std::size_t Size = size();

    // zero init
    simd_array() = default;

    // default copy ctor/operator
    simd_array(const simd_array &) = default;
    simd_array(simd_array &&) = default;
    simd_array &operator=(const simd_array &) = default;

    // broadcast
    Vc_ALWAYS_INLINE simd_array(value_type a) : d(a) {}

    // forward all remaining ctors to ArrayData
    template <typename... Args, typename = enable_if<!Traits::IsCastArguments<Args...>::value && !Traits::is_initializer_list<Args...>::value>>
    explicit Vc_ALWAYS_INLINE simd_array(Args &&... args)
        : d(adjustArgument(std::forward<Args>(args))...)
    {
    }

    // implicit casts
    template<typename U, std::size_t M, typename V> Vc_ALWAYS_INLINE simd_array(const simd_array<U, M, V> &x) {
        d.cast(simd_array_data(x));
    }


    explicit Vc_ALWAYS_INLINE simd_array(VectorSpecialInitializerZero::ZEnum x) : d(vector_type(x)) {}
    explicit Vc_ALWAYS_INLINE simd_array(VectorSpecialInitializerOne::OEnum x) : d(vector_type(x)) {}

    static Vc_ALWAYS_INLINE simd_array Zero() { return simd_array(VectorSpecialInitializerZero::Zero); }
    static Vc_ALWAYS_INLINE simd_array One() { return simd_array(VectorSpecialInitializerOne::One); }
    static Vc_ALWAYS_INLINE simd_array IndexesFromZero() { return simd_array(VectorSpecialInitializerIndexesFromZero::IndexesFromZero); }
    static Vc_ALWAYS_INLINE simd_array Random()
    {
        simd_array r;
        r.d.assign(&vector_type::Random);
        return r;
    }

    // initializer_list
    Vc_ALWAYS_INLINE simd_array(std::initializer_list<value_type> x)
    {
        //: d(x.begin(), Vc::Unaligned)  // TODO: it would be nice if there was a way to have the
                                       // compiler understand what it's doing here and thus make
                                       // aligned loads possible
#if __cplusplus > 201400
        static_assert(x.size() == size(), "");
#else
        VC_ASSERT(x.size() == size());
#endif
        d.template call<Common::Operations::Load>(x.begin(), Vc::Unaligned);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // load member functions
    Vc_ALWAYS_INLINE void load(const value_type *x) {
        d.template call<Common::Operations::Load>(x, DefaultLoadTag());
    }
    template<typename Flags>
    Vc_ALWAYS_INLINE void load(const value_type *x, Flags f) {
        d.template call<Common::Operations::Load>(x, f);
    }
    template<typename U, typename Flags = DefaultLoadTag>
    Vc_ALWAYS_INLINE void load(const U *x, Flags f = Flags()) {
        d.template call<Common::Operations::Load>(x, f);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // store member functions
    Vc_ALWAYS_INLINE void store(value_type *x) {
        d.template call<Common::Operations::Store>(x, DefaultStoreTag());
    }
    template <typename Flags> Vc_ALWAYS_INLINE void store(value_type *x, Flags f)
    {
        d.template call<Common::Operations::Store>(x, f);
    }
    template <typename U, typename Flags = DefaultStoreTag>
    Vc_ALWAYS_INLINE void store(U *x, Flags f = Flags())
    {
        d.template call<Common::Operations::Store>(x, f);
    }

#define VC_COMPARE_IMPL(op)                                                                        \
    Vc_ALWAYS_INLINE Vc_PURE mask_type operator op(const simd_array &x) const                      \
    {                                                                                              \
        mask_type r;                                                                               \
        r.d.assign(d, x.d, &vector_type::operator op);                                             \
        return r;                                                                                  \
    }
    VC_ALL_COMPARES(VC_COMPARE_IMPL)
#undef VC_COMPARE_IMPL

#define VC_OPERATOR_IMPL(op)                                                                       \
    Vc_ALWAYS_INLINE simd_array &operator op##=(const simd_array & x)                              \
    {                                                                                              \
        d op## = x.d;                                                                              \
        return *this;                                                                              \
    }                                                                                              \
    inline simd_array operator op(const simd_array &x) const                                       \
    {                                                                                              \
        simd_array r = *this;                                                                      \
        r op## = x;                                                                                \
        return r;                                                                                  \
    }
    VC_ALL_BINARY     (VC_OPERATOR_IMPL)
    VC_ALL_ARITHMETICS(VC_OPERATOR_IMPL)
    VC_ALL_SHIFTS     (VC_OPERATOR_IMPL)
#undef VC_OPERATOR_IMPL

    decltype(std::declval<vector_type &>()[0]) operator[](std::size_t i)
    {
        typedef value_type TT Vc_MAY_ALIAS;
        auto m = reinterpret_cast<TT *>(d.begin());
        return m[i];
    }
    value_type operator[](std::size_t i) const {
        typedef value_type TT Vc_MAY_ALIAS;
        auto m = reinterpret_cast<const TT *>(d.cbegin());
        return m[i];
    }

    //////////////////////
    // unary operators

    //prefix
    Vc_INTRINSIC simd_array &operator++()
    {
        d.template call<Common::Operations::Increment>();
        return *this;
    }
    Vc_INTRINSIC simd_array &operator--()
    {
        d.template call<Common::Operations::Decrement>();
        return *this;
    }
    // postfix
    Vc_INTRINSIC simd_array operator++(int)
    {
        const auto r = *this;
        d.template call<Common::Operations::Increment>();
        return r;
    }
    Vc_INTRINSIC simd_array operator--(int)
    {
        const auto r = *this;
        d.template call<Common::Operations::Decrement>();
        return r;
    }

    Vc_INTRINSIC mask_type operator!() const
    {
        mask_type r;
        r.d.assign(d, &vector_type::operator!);
        return r;
    }

    // TODO: perform integral promotion, simply return simd_array<decltype(-std::declval<T>()), N>
    Vc_INTRINSIC simd_array operator-() const
    {
        simd_array r;
        r.d.assign(d, static_cast<vector_type (vector_type::*)() const>(&vector_type::operator-));
        return r;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // gather / scatter
    template <typename... Args> Vc_INTRINSIC void gather(Args &&... arguments)
    {
        d.template call<Common::Operations::Gather>(adjustArgument(std::forward<Args>(arguments))...);
    }
    template <typename... Args> Vc_INTRINSIC void scatter(Args &&... arguments) const
    {
        d.template call<Common::Operations::Scatter>(adjustArgument(std::forward<Args>(arguments))...);
    }

    Vc_INTRINSIC Common::WriteMaskedVector<simd_array, mask_type> operator()(const Mask &k)
    {
        return {this, k};
    }

    template<typename... Args>
    Vc_INTRINSIC void setZero(Args &&... args)
    {
        d.template call<Common::Operations::SetZero>(adjustArgument(std::forward<Args>(args))...);
    }
    template<typename... Args>
    Vc_INTRINSIC void setZeroInverted(Args &&... args)
    {
        d.template call<Common::Operations::SetZeroInverted>(adjustArgument(std::forward<Args>(args))...);
    }
    template<typename... Args>
    Vc_INTRINSIC void assign(Args &&... args)
    {
        d.template call<Common::Operations::Assign>(adjustArgument(std::forward<Args>(args))...);
    }

// internal:
    simd_array(const storage_type &x) : d(x) {}

private:
    storage_type d;

    friend const decltype(d) & simd_array_data(const simd_array &x) { return x.d; }
    friend decltype(d) & simd_array_data(simd_array &x) { return x.d; }
    friend decltype(std::move(d)) simd_array_data(simd_array &&x) { return std::move(x.d); }

    /*
     * adjustArgument adjusts simd_array and simd_mask_array arguments to pass their data members
     * (ArrayData and MaskData) instead.
     * This function is used to adjust arguments that need to be passed to ArrayData and MaskData.
     *
     * TODO: move to a place where simd_mask_array can also use it.
     */
    template <typename U> static inline U adjustArgument(U &&x)
    {
        return std::forward<U>(x);
    }
    template <typename Container, typename IndexVector>
    static inline storage_type adjustArgument(const Common::SubscriptOperation<Container, IndexVector> &x)
    {
        return static_cast<simd_array>(x).d;
    }
    template <typename U, std::size_t M>
    static inline const typename simd_array<U, M>::storage_type &adjustArgument(
        const simd_array<U, M> &x)
    {
        return simd_array_data(x);
    }
    template <typename U, std::size_t M>
    static inline typename simd_array<U, M>::storage_type &adjustArgument(simd_array<U, M> &x)
    {
        return simd_array_data(x);
    }
    template <typename U, std::size_t M>
    static inline typename simd_array<U, M>::storage_type &&adjustArgument(simd_array<U, M> &&x)
    {
        return std::move(simd_array_data(x));
    }
    template <typename U, std::size_t M>
    static inline const typename simd_mask_array<U, M>::storage_type &adjustArgument(
        const simd_mask_array<U, M> &x)
    {
        return simd_mask_array_data(x);
    }
    template <typename U, std::size_t M>
    static inline typename simd_mask_array<U, M>::storage_type &adjustArgument(
        simd_mask_array<U, M> &x)
    {
        return simd_mask_array_data(x);
    }
    template <typename U, std::size_t M>
    static inline typename simd_mask_array<U, M>::storage_type &&adjustArgument(
        simd_mask_array<U, M> &&x)
    {
        return std::move(simd_mask_array_data(x));
    }

    template <typename U, typename A> static inline const U *adjustArgument(const std::vector<U, A> &x)
    {
        VC_ASSERT(x.size() >= size());
        return &x[0];
    }

    template <typename I>
    static inline typename simd_array<I, size()>::storage_type adjustArgument(
        const std::initializer_list<I> &x)
    {
        return simd_array_data(simd_array<I, size()>{x});
    }
};

template <typename T, std::size_t N> simd_array<T, N> abs(simd_array<T, N> x)
{
    simd_array<T, N> r;
    simd_array_data(r).template call<Common::Operations::Abs>(simd_array_data(x));
    //using V = typename simd_array<T, N>::vector_type;
    //simd_array_data(r).assign(static_cast<V(&)(const V &)>(abs), simd_array_data(x));
    return r;
}
#endif

} // namespace Vc_VERSIONED_NAMESPACE

#include "undomacros.h"

#endif // VC_COMMON_SIMD_ARRAY_H
