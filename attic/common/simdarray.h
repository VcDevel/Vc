/*  This file is part of the Vc library. {{{
Copyright © 2013-2015 Matthias Kretz <kretz@kde.org>

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

Vc_VERSIONED_NAMESPACE_BEGIN
// internal namespace (product & sum helper) {{{1
namespace internal
{
template <typename T> T Vc_INTRINSIC Vc_PURE product_helper_(const T &l, const T &r) { return l * r; }
template <typename T> T Vc_INTRINSIC Vc_PURE sum_helper_(const T &l, const T &r) { return l + r; }
}  // namespace internal

// min & max declarations {{{1
template <typename T, std::size_t N, typename V, std::size_t M>
inline SimdArray<T, N, V, M> min(const SimdArray<T, N, V, M> &x,
                                 const SimdArray<T, N, V, M> &y);
template <typename T, std::size_t N, typename V, std::size_t M>
inline SimdArray<T, N, V, M> max(const SimdArray<T, N, V, M> &x,
                                 const SimdArray<T, N, V, M> &y);

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
class SimdArray<T, N, VectorType_, N>
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
    using reference = Detail::ElementReference<SimdArray>;
    static constexpr std::size_t Size = size();
    static constexpr std::size_t MemoryAlignment = storage_type::MemoryAlignment;

    // zero init
#ifndef Vc_MSVC  // bogus error C2580
    Vc_INTRINSIC SimdArray() = default;
#endif

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
    Vc_INTRINSIC SimdArray(const SimdArray<U, N, V> &x, enable_if<N == V::Size> = nullarg)
        : data(simd_cast<vector_type>(internal_data(x)))
    {
    }
    template <typename U, typename V>
    Vc_INTRINSIC SimdArray(const SimdArray<U, N, V> &x,
                            enable_if<(N > V::Size && N <= 2 * V::Size)> = nullarg)
        : data(simd_cast<vector_type>(internal_data(internal_data0(x)), internal_data(internal_data1(x))))
    {
    }
    template <typename U, typename V>
    Vc_INTRINSIC SimdArray(const SimdArray<U, N, V> &x,
                            enable_if<(N > 2 * V::Size && N <= 4 * V::Size)> = nullarg)
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
    template <
        typename U, typename A,
        typename = enable_if<std::is_convertible<T, U>::value && Vector<U, A>::Size == N>>
    Vc_INTRINSIC operator Vector<U, A>() const
    {
        return simd_cast<Vector<U, A>>(data);
    }

#include "gatherinterface.h"
#include "scatterinterface.h"

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

    Vc_INTRINSIC void setQnan() { data.setQnan(); }
    Vc_INTRINSIC void setQnan(mask_type m) { data.setQnan(internal_data(m)); }

    // internal: execute specified Operation
    template <typename Op, typename... Args>
    static Vc_INTRINSIC SimdArray fromOperation(Op op, Args &&... args)
    {
        SimdArray r;
        Common::unpackArgumentsAuto(op, r.data, std::forward<Args>(args)...);
        return r;
    }

    template <typename Op, typename... Args>
    static Vc_INTRINSIC void callOperation(Op op, Args &&... args)
    {
        Common::unpackArgumentsAuto(op, nullptr, std::forward<Args>(args)...);
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

    /// Returns a copy of itself
    Vc_INTRINSIC SimdArray operator+() const { return *this; }

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
    Vc_ALL_ARITHMETICS(Vc_BINARY_OPERATOR_);
    Vc_ALL_BINARY(Vc_BINARY_OPERATOR_);
    Vc_ALL_SHIFTS(Vc_BINARY_OPERATOR_);
#undef Vc_BINARY_OPERATOR_

#define Vc_COMPARES(op)                                                                  \
    Vc_INTRINSIC mask_type operator op(const SimdArray &rhs) const                       \
    {                                                                                    \
        return {data op rhs.data};                                                       \
    }
    Vc_ALL_COMPARES(Vc_COMPARES);
#undef Vc_COMPARES

    /// \copydoc Vector::isNegative
    Vc_DEPRECATED("use isnegative(x) instead") Vc_INTRINSIC MaskType isNegative() const
    {
        return {isnegative(data)};
    }

private:
    friend reference;
    Vc_INTRINSIC static value_type get(const SimdArray &o, int i) noexcept
    {
        return o.data[i];
    }
    template <typename U>
    Vc_INTRINSIC static void set(SimdArray &o, int i, U &&v) noexcept(
        noexcept(std::declval<value_type &>() = v))
    {
        o.data[i] = v;
    }

public:
    Vc_INTRINSIC reference operator[](size_t i) noexcept
    {
        static_assert(noexcept(reference{std::declval<SimdArray &>(), int()}), "");
        return {*this, int(i)};
    }
    Vc_INTRINSIC value_type operator[](size_t i) const noexcept
    {
        return get(*this, int(i));
    }

    Vc_INTRINSIC Common::WriteMaskedVector<SimdArray, mask_type> operator()(const mask_type &k)
    {
        return {*this, k};
    }

    Vc_INTRINSIC void assign(const SimdArray &v, const mask_type &k)
    {
        data.assign(v.data, internal_data(k));
    }

    // reductions ////////////////////////////////////////////////////////
#define Vc_REDUCTION_FUNCTION_(name_)                                                    \
    Vc_INTRINSIC Vc_PURE value_type name_() const { return data.name_(); }               \
    Vc_INTRINSIC Vc_PURE value_type name_(mask_type mask) const                          \
    {                                                                                    \
        return data.name_(internal_data(mask));                                          \
    }                                                                                    \
    Vc_NOTHING_EXPECTING_SEMICOLON
    Vc_REDUCTION_FUNCTION_(min);
    Vc_REDUCTION_FUNCTION_(max);
    Vc_REDUCTION_FUNCTION_(product);
    Vc_REDUCTION_FUNCTION_(sum);
#undef Vc_REDUCTION_FUNCTION_
    Vc_INTRINSIC Vc_PURE SimdArray partialSum() const { return data.partialSum(); }

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

    /// \copydoc Vector::exponent
    Vc_DEPRECATED("use exponent(x) instead") Vc_INTRINSIC SimdArray exponent() const
    {
        return {exponent(data)};
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

    Vc_DEPRECATED("use copysign(x, y) instead") Vc_INTRINSIC SimdArray
        copySign(const SimdArray &reference) const
    {
        return {Vc::copysign(data, reference.data)};
    }

    friend VectorType &internal_data<>(SimdArray &x);
    friend const VectorType &internal_data<>(const SimdArray &x);

    /// \internal
    Vc_INTRINSIC SimdArray(VectorType &&x) : data(std::move(x)) {}

    Vc_FREE_STORE_OPERATORS_ALIGNED(alignof(storage_type));

private:
    // The alignas attribute attached to the class declaration above is ignored by ICC
    // 17.0.0 (at least). So just move the alignas attribute down here where it works for
    // all compilers.
    alignas(static_cast<std::size_t>(
        Common::BoundedAlignment<Common::NextPowerOfTwo<N>::value * sizeof(VectorType_) /
                                 VectorType_::size()>::value)) storage_type data;
};
template <typename T, std::size_t N, typename VectorType> constexpr std::size_t SimdArray<T, N, VectorType, N>::Size;
template <typename T, std::size_t N, typename VectorType>
constexpr std::size_t SimdArray<T, N, VectorType, N>::MemoryAlignment;
template <typename T, std::size_t N, typename VectorType>
#ifndef Vc_MSVC
Vc_INTRINSIC
#endif
VectorType &internal_data(SimdArray<T, N, VectorType, N> &x)
{
    return x.data;
}
template <typename T, std::size_t N, typename VectorType>
#ifndef Vc_MSVC
Vc_INTRINSIC
#endif
const VectorType &internal_data(const SimdArray<T, N, VectorType, N> &x)
{
    return x.data;
}

// unpackIfSegment {{{2
template <typename T> T unpackIfSegment(T &&x) { return std::forward<T>(x); }
template <typename T, size_t Pieces, size_t Index>
auto unpackIfSegment(Common::Segment<T, Pieces, Index> &&x) -> decltype(x.asSimdArray())
{
    return x.asSimdArray();
}

// gatherImplementation {{{2
template <typename T, std::size_t N, typename VectorType>
template <typename MT, typename IT>
inline void SimdArray<T, N, VectorType, N>::gatherImplementation(const MT *mem,
                                                                 IT &&indexes)
{
    data.gather(mem, unpackIfSegment(std::forward<IT>(indexes)));
}
template <typename T, std::size_t N, typename VectorType>
template <typename MT, typename IT>
inline void SimdArray<T, N, VectorType, N>::gatherImplementation(const MT *mem,
                                                                 IT &&indexes,
                                                                 MaskArgument mask)
{
    data.gather(mem, unpackIfSegment(std::forward<IT>(indexes)), mask);
}

// scatterImplementation {{{2
template <typename T, std::size_t N, typename VectorType>
template <typename MT, typename IT>
inline void SimdArray<T, N, VectorType, N>::scatterImplementation(MT *mem,
                                                                 IT &&indexes) const
{
    data.scatter(mem, unpackIfSegment(std::forward<IT>(indexes)));
}
template <typename T, std::size_t N, typename VectorType>
template <typename MT, typename IT>
inline void SimdArray<T, N, VectorType, N>::scatterImplementation(MT *mem,
                                                                 IT &&indexes,
                                                                 MaskArgument mask) const
{
    data.scatter(mem, unpackIfSegment(std::forward<IT>(indexes)), mask);
}

// generic SimdArray {{{1
/**
 * Data-parallel arithmetic type with user-defined number of elements.
 *
 * \tparam T The type of the vector's elements. The supported types currently are limited
 *           to the types supported by Vc::Vector<T>.
 *
 * \tparam N The number of elements to store and process concurrently. You can choose an
 *           arbitrary number, though not every number is a good idea.
 *           Generally, a power of two value or the sum of two power of two values might
 *           work efficiently, though this depends a lot on the target system.
 *
 * \tparam V Don't change the default value unless you really know what you are doing.
 *           This type is set to the underlying native Vc::Vector type used in the
 *           implementation of the type.
 *           Having it as part of the type name guards against some cases of ODR
 *           violations (i.e. linking incompatible translation units / libraries).
 *
 * \tparam Wt Don't ever change the default value.
 *           This parameter is an unfortunate implementation detail shining through.
 *
 * \warning Choosing \p N too large (what “too large” means depends on the target) will
 *          result in excessive compilation times and high (or too high) register
 *          pressure, thus potentially negating the improvement from concurrent execution.
 *          As a rule of thumb, keep \p N less or equal to `2 * float_v::size()`.
 *
 * \warning A special portability concern arises from a current limitation in the MIC
 *          implementation (Intel Knights Corner), where SimdArray types with \p T = \p
 *          (u)short require an \p N either less than short_v::size() or a multiple of
 *          short_v::size().
 *
 * \headerfile simdarray.h <Vc/SimdArray>
 */
template <typename T, size_t N, typename V, size_t Wt> class SimdArray
{
    static_assert(std::is_same<T,   double>::value ||
                  std::is_same<T,    float>::value ||
                  std::is_same<T,  int32_t>::value ||
                  std::is_same<T, uint32_t>::value ||
                  std::is_same<T,  int16_t>::value ||
                  std::is_same<T, uint16_t>::value, "SimdArray<T, N> may only be used with T = { double, float, int32_t, uint32_t, int16_t, uint16_t }");
    static_assert(
        // either the EntryType and VectorEntryType of the main V are equal
        std::is_same<typename V::EntryType, typename V::VectorEntryType>::value ||
            // or N is a multiple of V::size()
            (N % V::size() == 0),
        "SimdArray<(un)signed short, N> on MIC only works correctly for N = k * "
        "MIC::(u)short_v::size(), i.e. k * 16.");

    using my_traits = SimdArrayTraits<T, N>;
    static constexpr std::size_t N0 = my_traits::N0;
    static constexpr std::size_t N1 = my_traits::N1;
    using Split = Common::Split<N0>;
    template <typename U, std::size_t K> using CArray = U[K];

public:
    using storage_type0 = typename my_traits::storage_type0;
    using storage_type1 = typename my_traits::storage_type1;
    static_assert(storage_type0::size() == N0, "");

    /**\internal
     * This type reveals the implementation-specific type used for the data member.
     */
    using vector_type = V;
    using vectorentry_type = typename storage_type0::vectorentry_type;
    typedef vectorentry_type alias_type Vc_MAY_ALIAS;

    /// The type of the elements (i.e.\ \p T)
    using value_type = T;

    /// The type of the mask used for masked operations and returned from comparisons.
    using mask_type = SimdMaskArray<T, N, vector_type>;

    /// The type of the vector used for indexes in gather and scatter operations.
    using index_type = SimdArray<int, N>;

    /**
     * Returns \p N, the number of scalar components in an object of this type.
     *
     * The size of the SimdArray, i.e. the number of scalar elements in the vector. In
     * contrast to Vector::size() you have control over this value via the \p N template
     * parameter of the SimdArray class template.
     *
     * \returns The number of scalar values stored and manipulated concurrently by objects
     * of this type.
     */
    static constexpr std::size_t size() { return N; }

    /// \copydoc mask_type
    using Mask = mask_type;
    /// \copydoc mask_type
    using MaskType = Mask;
    using MaskArgument = const MaskType &;
    using VectorEntryType = vectorentry_type;
    /// \copydoc value_type
    using EntryType = value_type;
    /// \copydoc index_type
    using IndexType = index_type;
    using AsArg = const SimdArray &;

    using reference = Detail::ElementReference<SimdArray>;

    ///\copydoc Vector::MemoryAlignment
    static constexpr std::size_t MemoryAlignment =
        storage_type0::MemoryAlignment > storage_type1::MemoryAlignment
            ? storage_type0::MemoryAlignment
            : storage_type1::MemoryAlignment;

    /// \name Generators
    ///@{

    ///\copybrief Vector::Zero
    static Vc_INTRINSIC SimdArray Zero()
    {
        return SimdArray(Vc::Zero);
    }

    ///\copybrief Vector::One
    static Vc_INTRINSIC SimdArray One()
    {
        return SimdArray(Vc::One);
    }

    ///\copybrief Vector::IndexesFromZero
    static Vc_INTRINSIC SimdArray IndexesFromZero()
    {
        return SimdArray(Vc::IndexesFromZero);
    }

    ///\copydoc Vector::Random
    static Vc_INTRINSIC SimdArray Random()
    {
        return fromOperation(Common::Operations::random());
    }

    ///\copybrief Vector::generate
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
    ///@}

    /// \name Compile-Time Constant Initialization
    ///@{

    ///\copydoc Vector::Vector()
#ifndef Vc_MSVC  // bogus error C2580
    SimdArray() = default;
#endif
    ///@}

    /// \name Conversion/Broadcast Constructors
    ///@{

    ///\copydoc Vector::Vector(EntryType)
    Vc_INTRINSIC SimdArray(value_type a) : data0(a), data1(a) {}
    template <
        typename U,
        typename = enable_if<std::is_same<U, int>::value && !std::is_same<int, value_type>::value>>
    SimdArray(U a)
        : SimdArray(static_cast<value_type>(a))
    {
    }
    ///@}

    // default copy ctor/operator
    SimdArray(const SimdArray &) = default;
    SimdArray(SimdArray &&) = default;
    SimdArray &operator=(const SimdArray &) = default;

    // load ctor
    template <typename U,
              typename Flags = DefaultLoadTag,
              typename = enable_if<Traits::is_load_store_flag<Flags>::value>>
    explicit Vc_INTRINSIC SimdArray(const U *mem, Flags f = Flags())
        : data0(mem, f), data1(mem + storage_type0::size(), f)
    {
    }

// MSVC does overload resolution differently and takes the const U *mem overload (I hope)
#ifndef Vc_MSVC
    /**\internal
     * Load from a C-array. This is basically the same function as the load constructor
     * above, except that the forwarding reference overload would steal the deal and the
     * constructor above doesn't get called. This overload is required to enable loads
     * from C-arrays.
     */
    template <typename U, std::size_t Extent, typename Flags = DefaultLoadTag,
              typename = enable_if<Traits::is_load_store_flag<Flags>::value>>
    explicit Vc_INTRINSIC SimdArray(CArray<U, Extent> &mem, Flags f = Flags())
        : data0(&mem[0], f), data1(&mem[storage_type0::size()], f)
    {
    }
    /**\internal
     * Const overload of the above.
     */
    template <typename U, std::size_t Extent, typename Flags = DefaultLoadTag,
              typename = enable_if<Traits::is_load_store_flag<Flags>::value>>
    explicit Vc_INTRINSIC SimdArray(const CArray<U, Extent> &mem, Flags f = Flags())
        : data0(&mem[0], f), data1(&mem[storage_type0::size()], f)
    {
    }
#endif

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
#include "scatterinterface.h"

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
    template <typename W>
    Vc_INTRINSIC explicit SimdArray(
        W &&x,
        enable_if<(Traits::is_simd_vector<W>::value && Traits::simd_vector_size<W>::value == N &&
                   !(std::is_convertible<Traits::entry_type_of<W>, T>::value &&
                     Traits::isSimdArray<W>::value))> = nullarg)
        : data0(Split::lo(x)), data1(Split::hi(x))
    {
    }

    // implicit casts
    template <typename W>
    Vc_INTRINSIC SimdArray(
        W &&x,
        enable_if<(Traits::isSimdArray<W>::value && Traits::simd_vector_size<W>::value == N &&
                   std::is_convertible<Traits::entry_type_of<W>, T>::value)> = nullarg)
        : data0(Split::lo(x)), data1(Split::hi(x))
    {
    }

    // implicit conversion to Vector<U, AnyAbi> for if Vector<U, AnyAbi>::size() == N and
    // T implicitly convertible to U
    template <
        typename U, typename A,
        typename = enable_if<std::is_convertible<T, U>::value && Vector<U, A>::Size == N>>
    operator Vector<U, A>() const
    {
        return simd_cast<Vector<U, A>>(data0, data1);
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


    Vc_INTRINSIC void setQnan() {
        data0.setQnan();
        data1.setQnan();
    }
    Vc_INTRINSIC void setQnan(const mask_type &m) {
        data0.setQnan(Split::lo(m));
        data1.setQnan(Split::hi(m));
    }

    ///\internal execute specified Operation
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

    ///\internal
    template <typename Op, typename... Args>
    static Vc_INTRINSIC void callOperation(Op op, Args &&... args)
    {
        storage_type0::callOperation(op, Split::lo(args)...);
        storage_type1::callOperation(op, Split::hi(std::forward<Args>(args))...);
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

    /// Returns a copy of itself
    Vc_INTRINSIC SimdArray operator+() const { return *this; }

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
    Vc_ALL_ARITHMETICS(Vc_BINARY_OPERATOR_);
    Vc_ALL_BINARY(Vc_BINARY_OPERATOR_);
    Vc_ALL_SHIFTS(Vc_BINARY_OPERATOR_);
#undef Vc_BINARY_OPERATOR_

#define Vc_COMPARES(op)                                                                  \
    Vc_INTRINSIC mask_type operator op(const SimdArray &rhs) const                       \
    {                                                                                    \
        return {data0 op rhs.data0, data1 op rhs.data1};                                 \
    }
    Vc_ALL_COMPARES(Vc_COMPARES);
#undef Vc_COMPARES

    // operator[] {{{2
    /// \name Scalar Subscript Operators
    ///@{

private:
    friend reference;
    Vc_INTRINSIC static value_type get(const SimdArray &o, int i) noexcept
    {
        return reinterpret_cast<const alias_type *>(&o)[i];
    }
    template <typename U>
    Vc_INTRINSIC static void set(SimdArray &o, int i, U &&v) noexcept(
        noexcept(std::declval<value_type &>() = v))
    {
        reinterpret_cast<alias_type *>(&o)[i] = v;
    }

public:
    ///\copydoc Vector::operator[](size_t)
    Vc_INTRINSIC reference operator[](size_t i) noexcept
    {
        static_assert(noexcept(reference{std::declval<SimdArray &>(), int()}), "");
        return {*this, int(i)};
    }

    ///\copydoc Vector::operator[](size_t) const
    Vc_INTRINSIC value_type operator[](size_t index) const noexcept
    {
        return get(*this, int(index));
    }
    ///@}

    // operator(){{{2
    ///\copydoc Vector::operator()(MaskType)
    Vc_INTRINSIC Common::WriteMaskedVector<SimdArray, mask_type> operator()(
        const mask_type &mask)
    {
        return {*this, mask};
    }

    ///\internal
    Vc_INTRINSIC void assign(const SimdArray &v, const mask_type &k) //{{{2
    {
        data0.assign(v.data0, internal_data0(k));
        data1.assign(v.data1, internal_data1(k));
    }

    // reductions {{{2
#define Vc_REDUCTION_FUNCTION_(name_, binary_fun_, scalar_fun_)                          \
private:                                                                                 \
    template <typename ForSfinae = void>                                                 \
    Vc_INTRINSIC enable_if<std::is_same<ForSfinae, void>::value &&                       \
                               storage_type0::Size == storage_type1::Size,           \
                           value_type> name_##_impl() const                              \
    {                                                                                    \
        return binary_fun_(data0, data1).name_();                                        \
    }                                                                                    \
                                                                                         \
    template <typename ForSfinae = void>                                                 \
    Vc_INTRINSIC enable_if<std::is_same<ForSfinae, void>::value &&                       \
                               storage_type0::Size != storage_type1::Size,           \
                           value_type> name_##_impl() const                              \
    {                                                                                    \
        return scalar_fun_(data0.name_(), data1.name_());                                \
    }                                                                                    \
                                                                                         \
public:                                                                                  \
    /**\copybrief Vector::##name_ */                                                     \
    Vc_INTRINSIC value_type name_() const { return name_##_impl(); }                     \
    /**\copybrief Vector::##name_ */                                                     \
    Vc_INTRINSIC value_type name_(const mask_type &mask) const                           \
    {                                                                                    \
        if (Vc_IS_UNLIKELY(Split::lo(mask).isEmpty())) {                                 \
            return data1.name_(Split::hi(mask));                                         \
        } else if (Vc_IS_UNLIKELY(Split::hi(mask).isEmpty())) {                          \
            return data0.name_(Split::lo(mask));                                         \
        } else {                                                                         \
            return scalar_fun_(data0.name_(Split::lo(mask)),                             \
                               data1.name_(Split::hi(mask)));                            \
        }                                                                                \
    }                                                                                    \
    Vc_NOTHING_EXPECTING_SEMICOLON
    Vc_REDUCTION_FUNCTION_(min, Vc::min, std::min);
    Vc_REDUCTION_FUNCTION_(max, Vc::max, std::max);
    Vc_REDUCTION_FUNCTION_(product, internal::product_helper_, internal::product_helper_);
    Vc_REDUCTION_FUNCTION_(sum, internal::sum_helper_, internal::sum_helper_);
#undef Vc_REDUCTION_FUNCTION_
    ///\copybrief Vector::partialSum
    Vc_INTRINSIC Vc_PURE SimdArray partialSum() const //{{{2
    {
        auto ps0 = data0.partialSum();
        auto tmp = data1;
        tmp[0] += ps0[data0.size() - 1];
        return {std::move(ps0), tmp.partialSum()};
    }

    // apply {{{2
    ///\copybrief Vector::apply(F &&) const
    template <typename F> inline SimdArray apply(F &&f) const
    {
        return {data0.apply(f), data1.apply(f)};
    }
    ///\copybrief Vector::apply(F &&, MaskType) const
    template <typename F> inline SimdArray apply(F &&f, const mask_type &k) const
    {
        return {data0.apply(f, Split::lo(k)), data1.apply(f, Split::hi(k))};
    }

    // shifted {{{2
    ///\copybrief Vector::shifted(int) const
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

private:
    // workaround for MSVC not understanding the simpler and shorter expression of the boolean
    // expression directly in the enable_if below
    template <std::size_t NN> struct bisectable_shift
        : public std::integral_constant<bool,
                                        std::is_same<storage_type0, storage_type1>::value &&  // bisectable
                                        N == NN>
    {
    };

public:
    template <std::size_t NN>
    inline SimdArray shifted(enable_if<bisectable_shift<NN>::value, int> amount,
            const SimdArray<value_type, NN> &shiftIn) const
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
    ///\copybrief Vector::rotated
    Vc_INTRINSIC SimdArray rotated(int amount) const
    {
        amount %= int(size());
        if (amount == 0) {
            return *this;
        } else if (amount < 0) {
            amount += size();
        }

#ifdef Vc_MSVC
        // MSVC fails to find a SimdArray::shifted function with 2 arguments. So use store
        // ->
        // load to implement the function instead.
        alignas(MemoryAlignment) T tmp[N + data0.size()];
        data0.store(&tmp[0], Vc::Aligned);
        data1.store(&tmp[data0.size()], Vc::Aligned);
        data0.store(&tmp[N], Vc::Unaligned);
        SimdArray r;
        r.data0.load(&tmp[amount], Vc::Unaligned);
        r.data1.load(&tmp[(amount + data0.size()) % size()], Vc::Unaligned);
        return r;
#else
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
#endif
    }

    // interleaveLow/-High {{{2
    ///\internal \copydoc Vector::interleaveLow
    Vc_INTRINSIC SimdArray interleaveLow(const SimdArray &x) const
    {
        // return data0[0], x.data0[0], data0[1], x.data0[1], ...
        return {data0.interleaveLow(x.data0),
                simd_cast<storage_type1>(data0.interleaveHigh(x.data0))};
    }
    ///\internal \copydoc Vector::interleaveHigh
    Vc_INTRINSIC SimdArray interleaveHigh(const SimdArray &x) const
    {
        return interleaveHighImpl(
            x,
            std::integral_constant<bool, storage_type0::Size == storage_type1::Size>());
    }

private:
    ///\internal
    Vc_INTRINSIC SimdArray interleaveHighImpl(const SimdArray &x, std::true_type) const
    {
        return {data1.interleaveLow(x.data1), data1.interleaveHigh(x.data1)};
    }
    ///\internal
    inline SimdArray interleaveHighImpl(const SimdArray &x, std::false_type) const
    {
        return {data0.interleaveHigh(x.data0)
                    .shifted(storage_type1::Size,
                             simd_cast<storage_type0>(data1.interleaveLow(x.data1))),
                data1.interleaveHigh(x.data1)};
    }

public:
    ///\copybrief Vector::reversed
    inline SimdArray reversed() const //{{{2
    {
        if (std::is_same<storage_type0, storage_type1>::value) {
            return {simd_cast<storage_type0>(data1).reversed(),
                    simd_cast<storage_type1>(data0).reversed()};
        } else {
#ifdef Vc_MSVC
            // MSVC fails to find a SimdArray::shifted function with 2 arguments. So use
            // store
            // -> load to implement the function instead.
            alignas(MemoryAlignment) T tmp[N];
            data1.reversed().store(&tmp[0], Vc::Aligned);
            data0.reversed().store(&tmp[data1.size()], Vc::Unaligned);
            return SimdArray{&tmp[0], Vc::Aligned};
#else
            return {data0.shifted(storage_type1::Size, data1).reversed(),
                    simd_cast<storage_type1>(data0.reversed().shifted(
                        storage_type0::Size - storage_type1::Size))};
#endif
        }
    }
    ///\copydoc Vector::sorted
    inline SimdArray sorted() const  //{{{2
    {
        return sortedImpl(
            std::integral_constant<bool, storage_type0::Size == storage_type1::Size>());
    }

    ///\internal
    Vc_INTRINSIC SimdArray sortedImpl(std::true_type) const
    {
#ifdef Vc_DEBUG_SORTED
        std::cerr << "-- " << data0 << data1 << '\n';
#endif
        const auto a = data0.sorted();
        const auto b = data1.sorted().reversed();
        const auto lo = Vc::min(a, b);
        const auto hi = Vc::max(a, b);
        return {lo.sorted(), hi.sorted()};
    }

    ///\internal
    Vc_INTRINSIC SimdArray sortedImpl(std::false_type) const
    {
        using SortableArray =
            SimdArray<value_type, Common::NextPowerOfTwo<size()>::value>;
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

    /// \name Deprecated Members
    ///@{

    ///\copydoc size
    ///\deprecated Use size() instead.
    static constexpr std::size_t Size = size();

    /// \copydoc Vector::exponent
    Vc_DEPRECATED("use exponent(x) instead") Vc_INTRINSIC SimdArray exponent() const
    {
        return {exponent(data0), exponent(data1)};
    }

    /// \copydoc Vector::isNegative
    Vc_DEPRECATED("use isnegative(x) instead") Vc_INTRINSIC MaskType isNegative() const
    {
        return {isnegative(data0), isnegative(data1)};
    }

    ///\copydoc Vector::copySign
    Vc_DEPRECATED("use copysign(x, y) instead") Vc_INTRINSIC SimdArray
        copySign(const SimdArray &reference) const
    {
        return {Vc::copysign(data0, reference.data0),
                Vc::copysign(data1, reference.data1)};
    }
    ///@}

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

    Vc_FREE_STORE_OPERATORS_ALIGNED(alignof(storage_type0));

private: //{{{2
    // The alignas attribute attached to the class declaration above is ignored by ICC
    // 17.0.0 (at least). So just move the alignas attribute down here where it works for
    // all compilers.
    alignas(static_cast<std::size_t>(
        Common::BoundedAlignment<Common::NextPowerOfTwo<N>::value * sizeof(V) /
                                 V::size()>::value)) storage_type0 data0;
    storage_type1 data1;
};
#undef Vc_CURRENT_CLASS_NAME
template <typename T, std::size_t N, typename V, std::size_t M>
constexpr std::size_t SimdArray<T, N, V, M>::Size;
template <typename T, std::size_t N, typename V, std::size_t M>
constexpr std::size_t SimdArray<T, N, V, M>::MemoryAlignment;

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

// scatterImplementation {{{2
template <typename T, std::size_t N, typename VectorType, std::size_t M>
template <typename MT, typename IT>
inline void SimdArray<T, N, VectorType, M>::scatterImplementation(MT *mem,
                                                                 IT &&indexes) const
{
    data0.scatter(mem, Split::lo(Common::Operations::gather(),
                                indexes));  // don't forward indexes - it could move and
                                            // thus break the next line
    data1.scatter(mem, Split::hi(Common::Operations::gather(), std::forward<IT>(indexes)));
}
template <typename T, std::size_t N, typename VectorType, std::size_t M>
template <typename MT, typename IT>
inline void SimdArray<T, N, VectorType, M>::scatterImplementation(MT *mem,
                                                                 IT &&indexes, MaskArgument mask) const
{
    data0.scatter(mem, Split::lo(Common::Operations::gather(), indexes),
                 Split::lo(mask));  // don't forward indexes - it could move and
                                    // thus break the next line
    data1.scatter(mem, Split::hi(Common::Operations::gather(), std::forward<IT>(indexes)),
                 Split::hi(mask));
}

// internal_data0/1 (SimdArray) {{{1
///\internal Returns the first data member of a generic SimdArray
template <typename T, std::size_t N, typename V, std::size_t M>
#ifndef Vc_MSVC
Vc_INTRINSIC
#endif
typename SimdArrayTraits<T, N>::storage_type0 &internal_data0(
    SimdArray<T, N, V, M> &x)
{
    return x.data0;
}
///\internal Returns the second data member of a generic SimdArray
template <typename T, std::size_t N, typename V, std::size_t M>
#ifndef Vc_MSVC
Vc_INTRINSIC
#endif
typename SimdArrayTraits<T, N>::storage_type1 &internal_data1(
    SimdArray<T, N, V, M> &x)
{
    return x.data1;
}
///\internal Returns the first data member of a generic SimdArray (const overload)
template <typename T, std::size_t N, typename V, std::size_t M>
#ifndef Vc_MSVC
Vc_INTRINSIC
#endif
const typename SimdArrayTraits<T, N>::storage_type0 &internal_data0(
    const SimdArray<T, N, V, M> &x)
{
    return x.data0;
}
///\internal Returns the second data member of a generic SimdArray (const overload)
template <typename T, std::size_t N, typename V, std::size_t M>
#ifndef Vc_MSVC
Vc_INTRINSIC
#endif
const typename SimdArrayTraits<T, N>::storage_type1 &internal_data1(
    const SimdArray<T, N, V, M> &x)
{
    return x.data1;
}

// MSVC workaround for SimdArray(storage_type0, storage_type1) ctor{{{1
// MSVC sometimes stores x to data1. By first broadcasting 0 and then assigning y
// in the body the bug is supressed.
#if defined Vc_MSVC && defined Vc_IMPL_SSE
template <>
Vc_INTRINSIC SimdArray<double, 8, SSE::Vector<double>, 2>::SimdArray(
    SimdArray<double, 4> &&x, SimdArray<double, 4> &&y)
    : data0(x), data1(0)
{
    data1 = y;
}
#endif

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
    typename L, typename R,
    std::size_t N = Traits::isSimdArray<L>::value ? Traits::simd_vector_size<L>::value
                                                  : Traits::simd_vector_size<R>::value,
    bool =
        (Traits::isSimdArray<L>::value ||
         Traits::isSimdArray<R>::value)  // one of the operands must be a SimdArray
        && !std::is_same<type<L>, type<R>>::value  // if the operands are of the same type
                                                   // use the member function
        &&
        ((std::is_arithmetic<type<L>>::value &&
          !is_integer_larger_than_int<type<L>>::value) ||
         (std::is_arithmetic<type<R>>::value &&
          !is_integer_larger_than_int<type<R>>::value)  // one of the operands is a scalar
                                                        // type
         ||
         (  // or one of the operands is Vector<T> with Vector<T>::size() ==
            // SimdArray::size()
             Traits::simd_vector_size<L>::value == Traits::simd_vector_size<R>::value &&
             ((Traits::is_simd_vector<L>::value && !Traits::isSimdArray<L>::value) ||
              (Traits::is_simd_vector<R>::value && !Traits::isSimdArray<R>::value))))>
struct evaluate;

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

#define Vc_BINARY_OPERATORS_(op_)                                                        \
    /*!\brief Applies op_ component-wise and concurrently.  */                           \
    template <typename L, typename R>                                                    \
    Vc_INTRINSIC result_vector_type<L, R> operator op_(L &&lhs, R &&rhs)                 \
    {                                                                                    \
        using Return = result_vector_type<L, R>;                                         \
        return Return(std::forward<L>(lhs)) op_ Return(std::forward<R>(rhs));            \
    }
/**
 * \name Arithmetic and Bitwise Operators
 *
 * Applies the operator component-wise and concurrently on \p lhs and \p rhs and returns
 * a new SimdArray object containing the result values.
 *
 * This operator only participates in overload resolution if:
 * \li At least one of the template parameters \p L or \p R is a SimdArray type.
 * \li Either \p L or \p R is a fundamental arithmetic type but not an integral type
 *     larger than \c int \n
 *     or \n
 *     \p L or \p R is a Vc::Vector type with equal number of elements (Vector::size() ==
 *     SimdArray::size()).
 *
 * The return type of the operator is a SimdArray type using the more precise EntryType of
 * \p L or \p R and the same number of elements as the SimdArray argument(s).
 */
///@{
Vc_ALL_ARITHMETICS(Vc_BINARY_OPERATORS_);
Vc_ALL_BINARY(Vc_BINARY_OPERATORS_);
///@}
#undef Vc_BINARY_OPERATORS_
#define Vc_BINARY_OPERATORS_(op_)                                                        \
    /*!\brief Applies op_ component-wise and concurrently.  */                           \
    template <typename L, typename R>                                                    \
    Vc_INTRINSIC typename result_vector_type<L, R>::mask_type operator op_(L &&lhs,      \
                                                                           R &&rhs)      \
    {                                                                                    \
        using Promote = result_vector_type<L, R>;                                        \
        return Promote(std::forward<L>(lhs)) op_ Promote(std::forward<R>(rhs));          \
    }
/**
 * \name Compare Operators
 *
 * Applies the operator component-wise and concurrently on \p lhs and \p rhs and returns
 * a new SimdMaskArray object containing the result values.
 *
 * This operator only participates in overload resolution if (same rules as above):
 * \li At least one of the template parameters \p L or \p R is a SimdArray type.
 * \li Either \p L or \p R is a fundamental arithmetic type but not an integral type
 *     larger than \c int \n
 *     or \n
 *     \p L or \p R is a Vc::Vector type with equal number of elements (Vector::size() ==
 *     SimdArray::size()).
 *
 * The return type of the operator is a SimdMaskArray type using the more precise EntryType of
 * \p L or \p R and the same number of elements as the SimdArray argument(s).
 */
///@{
Vc_ALL_COMPARES(Vc_BINARY_OPERATORS_);
///@}
#undef Vc_BINARY_OPERATORS_

// math functions {{{1
#define Vc_FORWARD_UNARY_OPERATOR(name_)                                                 \
    /*!\brief Applies the std::name_ function component-wise and concurrently. */        \
    template <typename T, std::size_t N, typename V, std::size_t M>                      \
    inline SimdArray<T, N, V, M> name_(const SimdArray<T, N, V, M> &x)                   \
    {                                                                                    \
        return SimdArray<T, N, V, M>::fromOperation(                                     \
            Common::Operations::Forward_##name_(), x);                                   \
    }                                                                                    \
    Vc_NOTHING_EXPECTING_SEMICOLON

#define Vc_FORWARD_UNARY_BOOL_OPERATOR(name_)                                            \
    /*!\brief Applies the std::name_ function component-wise and concurrently. */        \
    template <typename T, std::size_t N, typename V, std::size_t M>                      \
    inline SimdMaskArray<T, N, V, M> name_(const SimdArray<T, N, V, M> &x)               \
    {                                                                                    \
        return SimdMaskArray<T, N, V, M>::fromOperation(                                 \
            Common::Operations::Forward_##name_(), x);                                   \
    }                                                                                    \
    Vc_NOTHING_EXPECTING_SEMICOLON

#define Vc_FORWARD_BINARY_OPERATOR(name_)                                                \
    /*!\brief Applies the std::name_ function component-wise and concurrently. */        \
    template <typename T, std::size_t N, typename V, std::size_t M>                      \
    inline SimdArray<T, N, V, M> name_(const SimdArray<T, N, V, M> &x,                   \
                                       const SimdArray<T, N, V, M> &y)                   \
    {                                                                                    \
        return SimdArray<T, N, V, M>::fromOperation(                                     \
            Common::Operations::Forward_##name_(), x, y);                                \
    }                                                                                    \
    Vc_NOTHING_EXPECTING_SEMICOLON

/**
 * \name Math functions
 * These functions evaluate the 
 */
///@{
Vc_FORWARD_UNARY_OPERATOR(abs);
Vc_FORWARD_UNARY_OPERATOR(asin);
Vc_FORWARD_UNARY_OPERATOR(atan);
Vc_FORWARD_BINARY_OPERATOR(atan2);
Vc_FORWARD_UNARY_OPERATOR(ceil);
Vc_FORWARD_BINARY_OPERATOR(copysign);
Vc_FORWARD_UNARY_OPERATOR(cos);
Vc_FORWARD_UNARY_OPERATOR(exp);
Vc_FORWARD_UNARY_OPERATOR(exponent);
Vc_FORWARD_UNARY_OPERATOR(floor);
/// Applies the std::fma function component-wise and concurrently.
template <typename T, std::size_t N>
inline SimdArray<T, N> fma(const SimdArray<T, N> &a, const SimdArray<T, N> &b,
                           const SimdArray<T, N> &c)
{
    return SimdArray<T, N>::fromOperation(Common::Operations::Forward_fma(), a, b, c);
}
Vc_FORWARD_UNARY_BOOL_OPERATOR(isfinite);
Vc_FORWARD_UNARY_BOOL_OPERATOR(isinf);
Vc_FORWARD_UNARY_BOOL_OPERATOR(isnan);
#if defined Vc_MSVC && defined Vc_IMPL_SSE
inline SimdMaskArray<double, 8, SSE::Vector<double>, 2> isnan(
    const SimdArray<double, 8, SSE::Vector<double>, 2> &x)
{
    using V = SSE::Vector<double>;
    const SimdArray<double, 4, V, 2> &x0 = internal_data0(x);
    const SimdArray<double, 4, V, 2> &x1 = internal_data1(x);
    SimdMaskArray<double, 4, V, 2> r0;
    SimdMaskArray<double, 4, V, 2> r1;
    internal_data(internal_data0(r0)) = isnan(internal_data(internal_data0(x0)));
    internal_data(internal_data1(r0)) = isnan(internal_data(internal_data1(x0)));
    internal_data(internal_data0(r1)) = isnan(internal_data(internal_data0(x1)));
    internal_data(internal_data1(r1)) = isnan(internal_data(internal_data1(x1)));
    return {std::move(r0), std::move(r1)};
}
#endif
Vc_FORWARD_UNARY_BOOL_OPERATOR(isnegative);
/// Applies the std::frexp function component-wise and concurrently.
template <typename T, std::size_t N>
inline SimdArray<T, N> frexp(const SimdArray<T, N> &x, SimdArray<int, N> *e)
{
    return SimdArray<T, N>::fromOperation(Common::Operations::Forward_frexp(), x, e);
}
/// Applies the std::ldexp function component-wise and concurrently.
template <typename T, std::size_t N>
inline SimdArray<T, N> ldexp(const SimdArray<T, N> &x, const SimdArray<int, N> &e)
{
    return SimdArray<T, N>::fromOperation(Common::Operations::Forward_ldexp(), x, e);
}
Vc_FORWARD_UNARY_OPERATOR(log);
Vc_FORWARD_UNARY_OPERATOR(log10);
Vc_FORWARD_UNARY_OPERATOR(log2);
Vc_FORWARD_UNARY_OPERATOR(reciprocal);
Vc_FORWARD_UNARY_OPERATOR(round);
Vc_FORWARD_UNARY_OPERATOR(rsqrt);
Vc_FORWARD_UNARY_OPERATOR(sin);
/// Determines sine and cosine concurrently and component-wise on \p x.
template <typename T, std::size_t N>
void sincos(const SimdArray<T, N> &x, SimdArray<T, N> *sin, SimdArray<T, N> *cos)
{
    SimdArray<T, N>::callOperation(Common::Operations::Forward_sincos(), x, sin, cos);
}
Vc_FORWARD_UNARY_OPERATOR(sqrt);
Vc_FORWARD_UNARY_OPERATOR(trunc);
Vc_FORWARD_BINARY_OPERATOR(min);
Vc_FORWARD_BINARY_OPERATOR(max);
///@}
#undef Vc_FORWARD_UNARY_OPERATOR
#undef Vc_FORWARD_UNARY_BOOL_OPERATOR
#undef Vc_FORWARD_BINARY_OPERATOR

// simd_cast {{{1
#ifdef Vc_MSVC
#define Vc_DUMMY_ARG0 , int = 0
#define Vc_DUMMY_ARG1 , long = 0
#define Vc_DUMMY_ARG2 , short = 0
#define Vc_DUMMY_ARG3 , char = '0'
#define Vc_DUMMY_ARG4 , unsigned = 0u
#define Vc_DUMMY_ARG5 , unsigned short = 0u
#else
#define Vc_DUMMY_ARG0
#define Vc_DUMMY_ARG1
#define Vc_DUMMY_ARG2
#define Vc_DUMMY_ARG3
#define Vc_DUMMY_ARG4
#define Vc_DUMMY_ARG5
#endif  // Vc_MSVC

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

// is_less trait{{{2
template <size_t A, size_t B>
struct is_less : public std::integral_constant<bool, (A < B)> {
};

// is_power_of_2 trait{{{2
template <size_t N>
struct is_power_of_2 : public std::integral_constant<bool, ((N - 1) & N) == 0> {
};

// simd_cast<T>(xs...) to SimdArray/-mask {{{2
#define Vc_SIMDARRAY_CASTS(SimdArrayType_, NativeType_)                                  \
    template <typename Return, typename T, typename A, typename... Froms>                \
    Vc_INTRINSIC Vc_CONST enable_if<                                                     \
        (Traits::isAtomic##SimdArrayType_<Return>::value &&                              \
         is_less<NativeType_<T, A>::Size * sizeof...(Froms), Return::Size>::value &&     \
         are_all_types_equal<NativeType_<T, A>, Froms...>::value),                       \
        Return>                                                                          \
    simd_cast(NativeType_<T, A> x, Froms... xs)                                          \
    {                                                                                    \
        vc_debug_("simd_cast{1}(", ")\n", x, xs...);                                     \
        return {simd_cast<typename Return::storage_type>(x, xs...)};                     \
    }                                                                                    \
    template <typename Return, typename T, typename A, typename... Froms>                \
    Vc_INTRINSIC Vc_CONST enable_if<                                                     \
        (Traits::isAtomic##SimdArrayType_<Return>::value &&                              \
         !is_less<NativeType_<T, A>::Size * sizeof...(Froms), Return::Size>::value &&    \
         are_all_types_equal<NativeType_<T, A>, Froms...>::value),                       \
        Return>                                                                          \
    simd_cast(NativeType_<T, A> x, Froms... xs)                                          \
    {                                                                                    \
        vc_debug_("simd_cast{2}(", ")\n", x, xs...);                                     \
        return {simd_cast_without_last<Return, NativeType_<T, A>, Froms...>(x, xs...)};  \
    }                                                                                    \
    template <typename Return, typename T, typename A, typename... Froms>                \
    Vc_INTRINSIC Vc_CONST                                                                \
        enable_if<(Traits::is##SimdArrayType_<Return>::value &&                          \
                   !Traits::isAtomic##SimdArrayType_<Return>::value &&                   \
                   is_less<Common::left_size<Return::Size>(),                            \
                           NativeType_<T, A>::Size *(1 + sizeof...(Froms))>::value &&    \
                   are_all_types_equal<NativeType_<T, A>, Froms...>::value),             \
                  Return>                                                                \
        simd_cast(NativeType_<T, A> x, Froms... xs)                                      \
    {                                                                                    \
        vc_debug_("simd_cast{3}(", ")\n", x, xs...);                                     \
        using R0 = typename Return::storage_type0;                                       \
        using R1 = typename Return::storage_type1;                                       \
        return {simd_cast_drop_arguments<R0, Froms...>(x, xs...),                        \
                simd_cast_with_offset<R1, R0::Size>(x, xs...)};                          \
    }                                                                                    \
    template <typename Return, typename T, typename A, typename... Froms>                \
    Vc_INTRINSIC Vc_CONST                                                                \
        enable_if<(Traits::is##SimdArrayType_<Return>::value &&                          \
                   !Traits::isAtomic##SimdArrayType_<Return>::value &&                   \
                   !is_less<Common::left_size<Return::Size>(),                           \
                            NativeType_<T, A>::Size *(1 + sizeof...(Froms))>::value &&   \
                   are_all_types_equal<NativeType_<T, A>, Froms...>::value),             \
                  Return>                                                                \
        simd_cast(NativeType_<T, A> x, Froms... xs)                                      \
    {                                                                                    \
        vc_debug_("simd_cast{4}(", ")\n", x, xs...);                                     \
        using R0 = typename Return::storage_type0;                                       \
        using R1 = typename Return::storage_type1;                                       \
        return {simd_cast<R0>(x, xs...), R1::Zero()};                                    \
    }                                                                                    \
    Vc_NOTHING_EXPECTING_SEMICOLON

Vc_SIMDARRAY_CASTS(SimdArray, Vc::Vector);
Vc_SIMDARRAY_CASTS(SimdMaskArray, Vc::Mask);
#undef Vc_SIMDARRAY_CASTS

// simd_cast<SimdArray/-mask, offset>(V) {{{2
#define Vc_SIMDARRAY_CASTS(SimdArrayType_, NativeType_)                                  \
    /* SIMD Vector/Mask to atomic SimdArray/simdmaskarray */                             \
    template <typename Return, int offset, typename T, typename A>                       \
    Vc_INTRINSIC Vc_CONST                                                                \
        enable_if<Traits::isAtomic##SimdArrayType_<Return>::value, Return>               \
        simd_cast(NativeType_<T, A> x Vc_DUMMY_ARG0)                                     \
    {                                                                                    \
        vc_debug_("simd_cast{offset, atomic}(", ")\n", offset, x);                       \
        return {simd_cast<typename Return::storage_type, offset>(x)};                    \
    }                                                                                    \
    /* both halves of Return array are extracted from argument */                        \
    template <typename Return, int offset, typename T, typename A>                       \
    Vc_INTRINSIC Vc_CONST                                                                \
        enable_if<(Traits::is##SimdArrayType_<Return>::value &&                          \
                   !Traits::isAtomic##SimdArrayType_<Return>::value &&                   \
                   Return::Size * offset + Common::left_size<Return::Size>() <           \
                       NativeType_<T, A>::Size),                                         \
                  Return>                                                                \
        simd_cast(NativeType_<T, A> x Vc_DUMMY_ARG1)                                     \
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
    template <typename Return, int offset, typename T, typename A>                       \
    Vc_INTRINSIC Vc_CONST                                                                \
        enable_if<(Traits::is##SimdArrayType_<Return>::value &&                          \
                   !Traits::isAtomic##SimdArrayType_<Return>::value &&                   \
                   Return::Size * offset + Common::left_size<Return::Size>() >=          \
                       NativeType_<T, A>::Size),                                         \
                  Return>                                                                \
        simd_cast(NativeType_<T, A> x Vc_DUMMY_ARG2)                                     \
    {                                                                                    \
        vc_debug_("simd_cast{offset, R1::Zero}(", ")\n", offset, x);                     \
        using R0 = typename Return::storage_type0;                                       \
        using R1 = typename Return::storage_type1;                                       \
        constexpr int entries_offset = offset * Return::Size;                            \
        return {simd_cast_with_offset<R0, entries_offset>(x), R1::Zero()};               \
    }                                                                                    \
    Vc_NOTHING_EXPECTING_SEMICOLON

Vc_SIMDARRAY_CASTS(SimdArray, Vc::Vector);
Vc_SIMDARRAY_CASTS(SimdMaskArray, Vc::Mask);
#undef Vc_SIMDARRAY_CASTS

// simd_cast<T>(xs...) from SimdArray/-mask {{{2
#define Vc_SIMDARRAY_CASTS(SimdArrayType_)                                               \
    /* indivisible SimdArrayType_ */                                                     \
    template <typename Return, typename T, std::size_t N, typename V, typename... From>  \
    Vc_INTRINSIC Vc_CONST                                                                \
        enable_if<(are_all_types_equal<SimdArrayType_<T, N, V, N>, From...>::value &&    \
                   (sizeof...(From) == 0 || N * sizeof...(From) < Return::Size) &&       \
                   !std::is_same<Return, SimdArrayType_<T, N, V, N>>::value),            \
                  Return>                                                                \
        simd_cast(const SimdArrayType_<T, N, V, N> &x0, const From &... xs)              \
    {                                                                                    \
        vc_debug_("simd_cast{indivisible}(", ")\n", x0, xs...);                          \
        return simd_cast<Return>(internal_data(x0), internal_data(xs)...);               \
    }                                                                                    \
    /* indivisible SimdArrayType_ && can drop arguments from the end */                  \
    template <typename Return, typename T, std::size_t N, typename V, typename... From>  \
    Vc_INTRINSIC Vc_CONST                                                                \
        enable_if<(are_all_types_equal<SimdArrayType_<T, N, V, N>, From...>::value &&    \
                   (sizeof...(From) > 0 && (N * sizeof...(From) >= Return::Size)) &&     \
                   !std::is_same<Return, SimdArrayType_<T, N, V, N>>::value),            \
                  Return>                                                                \
        simd_cast(const SimdArrayType_<T, N, V, N> &x0, const From &... xs)              \
    {                                                                                    \
        vc_debug_("simd_cast{indivisible2}(", ")\n", x0, xs...);                         \
        return simd_cast_without_last<Return,                                            \
                                      typename SimdArrayType_<T, N, V, N>::storage_type, \
                                      typename From::storage_type...>(                   \
            internal_data(x0), internal_data(xs)...);                                    \
    }                                                                                    \
    /* bisectable SimdArrayType_ (N = 2^n) && never too large */                         \
    template <typename Return, typename T, std::size_t N, typename V, std::size_t M,     \
              typename... From>                                                          \
    Vc_INTRINSIC Vc_CONST enable_if<                                                     \
        (N != M && are_all_types_equal<SimdArrayType_<T, N, V, M>, From...>::value &&    \
         !std::is_same<Return, SimdArrayType_<T, N, V, M>>::value &&                     \
         is_less<N * sizeof...(From), Return::Size>::value && is_power_of_2<N>::value),  \
        Return>                                                                          \
    simd_cast(const SimdArrayType_<T, N, V, M> &x0, const From &... xs)                  \
    {                                                                                    \
        vc_debug_("simd_cast{bisectable}(", ")\n", x0, xs...);                           \
        return simd_cast_interleaved_argument_order<                                     \
            Return, typename SimdArrayType_<T, N, V, M>::storage_type0,                  \
            typename From::storage_type0...>(internal_data0(x0), internal_data0(xs)...,  \
                                             internal_data1(x0), internal_data1(xs)...); \
    }                                                                                    \
    /* bisectable SimdArrayType_ (N = 2^n) && input so large that at least the last      \
     * input can be dropped */                                                           \
    template <typename Return, typename T, std::size_t N, typename V, std::size_t M,     \
              typename... From>                                                          \
    Vc_INTRINSIC Vc_CONST enable_if<                                                     \
        (N != M && are_all_types_equal<SimdArrayType_<T, N, V, M>, From...>::value &&    \
         !is_less<N * sizeof...(From), Return::Size>::value && is_power_of_2<N>::value), \
        Return>                                                                          \
    simd_cast(const SimdArrayType_<T, N, V, M> &x0, const From &... xs)                  \
    {                                                                                    \
        vc_debug_("simd_cast{bisectable2}(", ")\n", x0, xs...);                          \
        return simd_cast_without_last<Return, SimdArrayType_<T, N, V, M>, From...>(      \
            x0, xs...);                                                                  \
    }                                                                                    \
    /* remaining SimdArrayType_ input never larger (N != 2^n) */                         \
    template <typename Return, typename T, std::size_t N, typename V, std::size_t M,     \
              typename... From>                                                          \
    Vc_INTRINSIC Vc_CONST enable_if<                                                     \
        (N != M && are_all_types_equal<SimdArrayType_<T, N, V, M>, From...>::value &&    \
         N * (1 + sizeof...(From)) <= Return::Size && !is_power_of_2<N>::value),         \
        Return>                                                                          \
    simd_cast(const SimdArrayType_<T, N, V, M> &x0, const From &... xs)                  \
    {                                                                                    \
        vc_debug_("simd_cast{remaining}(", ")\n", x0, xs...);                            \
        return simd_cast_impl_smaller_input<Return, N, SimdArrayType_<T, N, V, M>,       \
                                            From...>(x0, xs...);                         \
    }                                                                                    \
    /* remaining SimdArrayType_ input larger (N != 2^n) */                               \
    template <typename Return, typename T, std::size_t N, typename V, std::size_t M,     \
              typename... From>                                                          \
    Vc_INTRINSIC Vc_CONST enable_if<                                                     \
        (N != M && are_all_types_equal<SimdArrayType_<T, N, V, M>, From...>::value &&    \
         N * (1 + sizeof...(From)) > Return::Size && !is_power_of_2<N>::value),          \
        Return>                                                                          \
    simd_cast(const SimdArrayType_<T, N, V, M> &x0, const From &... xs)                  \
    {                                                                                    \
        vc_debug_("simd_cast{remaining2}(", ")\n", x0, xs...);                           \
        return simd_cast_impl_larger_input<Return, N, SimdArrayType_<T, N, V, M>,        \
                                           From...>(x0, xs...);                          \
    }                                                                                    \
    /* a single bisectable SimdArrayType_ (N = 2^n) too large */                         \
    template <typename Return, typename T, std::size_t N, typename V, std::size_t M>     \
    Vc_INTRINSIC Vc_CONST                                                                \
        enable_if<(N != M && N >= 2 * Return::Size && is_power_of_2<N>::value), Return>  \
        simd_cast(const SimdArrayType_<T, N, V, M> &x)                                   \
    {                                                                                    \
        vc_debug_("simd_cast{single bisectable}(", ")\n", x);                            \
        return simd_cast<Return>(internal_data0(x));                                     \
    }                                                                                    \
    template <typename Return, typename T, std::size_t N, typename V, std::size_t M>     \
    Vc_INTRINSIC Vc_CONST enable_if<(N != M && N > Return::Size &&                       \
                                     N < 2 * Return::Size && is_power_of_2<N>::value),   \
                                    Return>                                              \
    simd_cast(const SimdArrayType_<T, N, V, M> &x)                                       \
    {                                                                                    \
        vc_debug_("simd_cast{single bisectable2}(", ")\n", x);                           \
        return simd_cast<Return>(internal_data0(x), internal_data1(x));                  \
    }                                                                                    \
    Vc_NOTHING_EXPECTING_SEMICOLON

Vc_SIMDARRAY_CASTS(SimdArray);
Vc_SIMDARRAY_CASTS(SimdMaskArray);
#undef Vc_SIMDARRAY_CASTS

// simd_cast<T, offset>(SimdArray/-mask) {{{2
#define Vc_SIMDARRAY_CASTS(SimdArrayType_)                                               \
    /* offset == 0 is like without offset */                                             \
    template <typename Return, int offset, typename T, std::size_t N, typename V,        \
              std::size_t M>                                                             \
    Vc_INTRINSIC Vc_CONST enable_if<(offset == 0), Return> simd_cast(                    \
        const SimdArrayType_<T, N, V, M> &x Vc_DUMMY_ARG0)                               \
    {                                                                                    \
        vc_debug_("simd_cast{offset == 0}(", ")\n", offset, x);                          \
        return simd_cast<Return>(x);                                                     \
    }                                                                                    \
    /* forward to V */                                                                   \
    template <typename Return, int offset, typename T, std::size_t N, typename V>        \
    Vc_INTRINSIC Vc_CONST enable_if<(offset != 0), Return> simd_cast(                    \
        const SimdArrayType_<T, N, V, N> &x Vc_DUMMY_ARG1)                               \
    {                                                                                    \
        vc_debug_("simd_cast{offset, forward}(", ")\n", offset, x);                      \
        return simd_cast<Return, offset>(internal_data(x));                              \
    }                                                                                    \
    /* convert from right member of SimdArray */                                         \
    template <typename Return, int offset, typename T, std::size_t N, typename V,        \
              std::size_t M>                                                             \
    Vc_INTRINSIC Vc_CONST                                                                \
        enable_if<(N != M && offset * Return::Size >= Common::left_size<N>() &&          \
                   offset != 0 && Common::left_size<N>() % Return::Size == 0),           \
                  Return>                                                                \
        simd_cast(const SimdArrayType_<T, N, V, M> &x Vc_DUMMY_ARG2)                     \
    {                                                                                    \
        vc_debug_("simd_cast{offset, right}(", ")\n", offset, x);                        \
        return simd_cast<Return, offset - Common::left_size<N>() / Return::Size>(        \
            internal_data1(x));                                                          \
    }                                                                                    \
    /* same as above except for odd cases where offset * Return::Size doesn't fit the    \
     * left side of the SimdArray */                                                     \
    template <typename Return, int offset, typename T, std::size_t N, typename V,        \
              std::size_t M>                                                             \
    Vc_INTRINSIC Vc_CONST                                                                \
        enable_if<(N != M && offset * Return::Size >= Common::left_size<N>() &&          \
                   offset != 0 && Common::left_size<N>() % Return::Size != 0),           \
                  Return>                                                                \
        simd_cast(const SimdArrayType_<T, N, V, M> &x Vc_DUMMY_ARG3)                     \
    {                                                                                    \
        vc_debug_("simd_cast{offset, right, nofit}(", ")\n", offset, x);                 \
        return simd_cast_with_offset<Return,                                             \
                                     offset * Return::Size - Common::left_size<N>()>(    \
            internal_data1(x));                                                          \
    }                                                                                    \
    /* convert from left member of SimdArray */                                          \
    template <typename Return, int offset, typename T, std::size_t N, typename V,        \
              std::size_t M>                                                             \
    Vc_INTRINSIC Vc_CONST enable_if<                                                     \
        (N != M && /*offset * Return::Size < Common::left_size<N>() &&*/                 \
         offset != 0 && (offset + 1) * Return::Size <= Common::left_size<N>()),          \
        Return>                                                                          \
    simd_cast(const SimdArrayType_<T, N, V, M> &x Vc_DUMMY_ARG4)                         \
    {                                                                                    \
        vc_debug_("simd_cast{offset, left}(", ")\n", offset, x);                         \
        return simd_cast<Return, offset>(internal_data0(x));                             \
    }                                                                                    \
    /* fallback to copying scalars */                                                    \
    template <typename Return, int offset, typename T, std::size_t N, typename V,        \
              std::size_t M>                                                             \
    Vc_INTRINSIC Vc_CONST                                                                \
        enable_if<(N != M && (offset * Return::Size < Common::left_size<N>()) &&         \
                   offset != 0 && (offset + 1) * Return::Size > Common::left_size<N>()), \
                  Return>                                                                \
        simd_cast(const SimdArrayType_<T, N, V, M> &x Vc_DUMMY_ARG5)                     \
    {                                                                                    \
        vc_debug_("simd_cast{offset, copy scalars}(", ")\n", offset, x);                 \
        using R = typename Return::EntryType;                                            \
        Return r = Return::Zero();                                                       \
        for (std::size_t i = offset * Return::Size;                                      \
             i < std::min(N, (offset + 1) * Return::Size); ++i) {                        \
            r[i - offset * Return::Size] = static_cast<R>(x[i]);                         \
        }                                                                                \
        return r;                                                                        \
    }                                                                                    \
    Vc_NOTHING_EXPECTING_SEMICOLON
Vc_SIMDARRAY_CASTS(SimdArray);
Vc_SIMDARRAY_CASTS(SimdMaskArray);
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

#ifdef Vc_MSVC
// MSVC doesn't see that the Ts pack below can be empty and thus complains when extract_interleaved
// is called with only 2 arguments. These overloads here are *INCORRECT standard C++*, but they make
// MSVC do the right thing.
template <std::size_t I, typename T0>
Vc_INTRINSIC Vc_CONST enable_if<(I == 0), T0> extract_interleaved(const T0 &a0, const T0 &)
{
    return a0;
}
template <std::size_t I, typename T0>
Vc_INTRINSIC Vc_CONST enable_if<(I == 1), T0> extract_interleaved(const T0 &, const T0 &b0)
{
    return b0;
}
#endif  // Vc_MSVC

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

// conditional_assign {{{1
#define Vc_CONDITIONAL_ASSIGN(name_, op_)                                                \
    template <Operator O, typename T, std::size_t N, typename V, size_t VN, typename M,  \
              typename U>                                                                \
    Vc_INTRINSIC enable_if<O == Operator::name_, void> conditional_assign(               \
        SimdArray<T, N, V, VN> &lhs, M &&mask, U &&rhs)                                  \
    {                                                                                    \
        lhs(mask) op_ rhs;                                                               \
    }                                                                                    \
    Vc_NOTHING_EXPECTING_SEMICOLON
Vc_CONDITIONAL_ASSIGN(          Assign,  =);
Vc_CONDITIONAL_ASSIGN(      PlusAssign, +=);
Vc_CONDITIONAL_ASSIGN(     MinusAssign, -=);
Vc_CONDITIONAL_ASSIGN(  MultiplyAssign, *=);
Vc_CONDITIONAL_ASSIGN(    DivideAssign, /=);
Vc_CONDITIONAL_ASSIGN( RemainderAssign, %=);
Vc_CONDITIONAL_ASSIGN(       XorAssign, ^=);
Vc_CONDITIONAL_ASSIGN(       AndAssign, &=);
Vc_CONDITIONAL_ASSIGN(        OrAssign, |=);
Vc_CONDITIONAL_ASSIGN( LeftShiftAssign,<<=);
Vc_CONDITIONAL_ASSIGN(RightShiftAssign,>>=);
#undef Vc_CONDITIONAL_ASSIGN

#define Vc_CONDITIONAL_ASSIGN(name_, expr_)                                              \
    template <Operator O, typename T, std::size_t N, typename V, size_t VN, typename M>  \
    Vc_INTRINSIC enable_if<O == Operator::name_, SimdArray<T, N, V, VN>>                 \
    conditional_assign(SimdArray<T, N, V, VN> &lhs, M &&mask)                            \
    {                                                                                    \
        return expr_;                                                                    \
    }                                                                                    \
    Vc_NOTHING_EXPECTING_SEMICOLON
Vc_CONDITIONAL_ASSIGN(PostIncrement, lhs(mask)++);
Vc_CONDITIONAL_ASSIGN( PreIncrement, ++lhs(mask));
Vc_CONDITIONAL_ASSIGN(PostDecrement, lhs(mask)--);
Vc_CONDITIONAL_ASSIGN( PreDecrement, --lhs(mask));
#undef Vc_CONDITIONAL_ASSIGN
// transpose_impl {{{1
namespace Common
{
template <typename T, size_t N, typename V>
inline void transpose_impl(
    TransposeTag<4, 4>, SimdArray<T, N, V, N> *Vc_RESTRICT r[],
    const TransposeProxy<SimdArray<T, N, V, N>, SimdArray<T, N, V, N>,
                         SimdArray<T, N, V, N>, SimdArray<T, N, V, N>> &proxy)
{
    V *Vc_RESTRICT r2[4] = {&internal_data(*r[0]), &internal_data(*r[1]),
                            &internal_data(*r[2]), &internal_data(*r[3])};
    transpose_impl(TransposeTag<4, 4>(), &r2[0],
                   TransposeProxy<V, V, V, V>{internal_data(std::get<0>(proxy.in)),
                                              internal_data(std::get<1>(proxy.in)),
                                              internal_data(std::get<2>(proxy.in)),
                                              internal_data(std::get<3>(proxy.in))});
}

template <typename T, typename V>
inline void transpose_impl(
    TransposeTag<2, 4>, SimdArray<T, 4, V, 1> *Vc_RESTRICT r[],
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

template <typename T, typename V>
inline void transpose_impl(
    TransposeTag<4, 4>, SimdArray<T, 1, V, 1> *Vc_RESTRICT r[],
    const TransposeProxy<SimdArray<T, 1, V, 1>, SimdArray<T, 1, V, 1>,
                         SimdArray<T, 1, V, 1>, SimdArray<T, 1, V, 1>> &proxy)
{
    V *Vc_RESTRICT r2[4] = {&internal_data(*r[0]), &internal_data(*r[1]),
                            &internal_data(*r[2]), &internal_data(*r[3])};
    transpose_impl(TransposeTag<4, 4>(), &r2[0],
                   TransposeProxy<V, V, V, V>{internal_data(std::get<0>(proxy.in)),
                                              internal_data(std::get<1>(proxy.in)),
                                              internal_data(std::get<2>(proxy.in)),
                                              internal_data(std::get<3>(proxy.in))});
}

template <typename T, size_t N, typename V>
inline void transpose_impl(
    TransposeTag<4, 4>, SimdArray<T, N, V, 1> *Vc_RESTRICT r[],
    const TransposeProxy<SimdArray<T, N, V, 1>, SimdArray<T, N, V, 1>,
                         SimdArray<T, N, V, 1>, SimdArray<T, N, V, 1>> &proxy)
{
    SimdArray<T, N, V, 1> *Vc_RESTRICT r0[4 / 2] = {r[0], r[1]};
    SimdArray<T, N, V, 1> *Vc_RESTRICT r1[4 / 2] = {r[2], r[3]};
    using H = SimdArray<T, 2>;
    transpose_impl(TransposeTag<2, 4>(), &r0[0],
                   TransposeProxy<H, H, H, H>{internal_data0(std::get<0>(proxy.in)),
                                              internal_data0(std::get<1>(proxy.in)),
                                              internal_data0(std::get<2>(proxy.in)),
                                              internal_data0(std::get<3>(proxy.in))});
    transpose_impl(TransposeTag<2, 4>(), &r1[0],
                   TransposeProxy<H, H, H, H>{internal_data1(std::get<0>(proxy.in)),
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

Vc_VERSIONED_NAMESPACE_END

// numeric_limits {{{1
namespace std
{
template <typename T, size_t N, typename V, size_t VN>
struct numeric_limits<Vc::SimdArray<T, N, V, VN>> : public numeric_limits<T> {
private:
    using R = Vc::SimdArray<T, N, V, VN>;

public:
    static Vc_ALWAYS_INLINE Vc_CONST R max() noexcept { return numeric_limits<T>::max(); }
    static Vc_ALWAYS_INLINE Vc_CONST R min() noexcept { return numeric_limits<T>::min(); }
    static Vc_ALWAYS_INLINE Vc_CONST R lowest() noexcept
    {
        return numeric_limits<T>::lowest();
    }
    static Vc_ALWAYS_INLINE Vc_CONST R epsilon() noexcept
    {
        return numeric_limits<T>::epsilon();
    }
    static Vc_ALWAYS_INLINE Vc_CONST R round_error() noexcept
    {
        return numeric_limits<T>::round_error();
    }
    static Vc_ALWAYS_INLINE Vc_CONST R infinity() noexcept
    {
        return numeric_limits<T>::infinity();
    }
    static Vc_ALWAYS_INLINE Vc_CONST R quiet_NaN() noexcept
    {
        return numeric_limits<T>::quiet_NaN();
    }
    static Vc_ALWAYS_INLINE Vc_CONST R signaling_NaN() noexcept
    {
        return numeric_limits<T>::signaling_NaN();
    }
    static Vc_ALWAYS_INLINE Vc_CONST R denorm_min() noexcept
    {
        return numeric_limits<T>::denorm_min();
    }
};
}  // namespace std
//}}}1

#endif // VC_COMMON_SIMDARRAY_H_

// vim: foldmethod=marker
