/*  This file is part of the Vc library. {{{
Copyright © 2009-2015 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_MIC_VECTOR_H_
#define VC_MIC_VECTOR_H_

#ifdef CAN_OFFLOAD
#pragma offload_attribute(push, target(mic))
#endif

#include "types.h"
#include "intrinsics.h"
#include "casts.h"
#include "../common/storage.h"
#include "mask.h"
#include "storemixin.h"
//#include "vectormultiplication.h"
#include "writemaskedvector.h"
#include "sorthelper.h"
#include "../common/where.h"
#include "macros.h"

#ifdef isfinite
#undef isfinite
#endif
#ifdef isnan
#undef isnan
#endif

#define Vc_HAVE_FMA

namespace Vc_VERSIONED_NAMESPACE
{
#define Vc_CURRENT_CLASS_NAME Vector
template <typename T>
class Vector<T, VectorAbi::Mic> : public MIC::StoreMixin<MIC::Vector<T>, T>
{
    static_assert(std::is_arithmetic<T>::value,
                  "Vector<T> only accepts arithmetic builtin types as template parameter T.");

public:
    using abi = VectorAbi::Mic;

private:
    //friend class VectorMultiplication<T>;
    friend class MIC::WriteMaskedVector<T>;
    //friend Vector<T> operator+<>(const Vector<T> &x, const VectorMultiplication<T> &y);
    //friend Vector<T> operator-<>(const Vector<T> &x, const VectorMultiplication<T> &y);
    friend class Vector<float, abi>;
    friend class Vector<double, abi>;
    friend class Vector<int, abi>;
    friend class Vector<unsigned int, abi>;
    friend class MIC::StoreMixin<Vector, T>;

public:
    Vc_FREE_STORE_OPERATORS_ALIGNED(64);
    typedef typename MIC::VectorTypeHelper<T>::Type VectorType;
    using vector_type = VectorType;
    Vc_ALIGNED_TYPEDEF(sizeof(T), T, EntryType);
    using value_type = EntryType;
    typedef typename MIC::DetermineVectorEntryType<T>::Type VectorEntryType;
    static constexpr size_t Size = sizeof(VectorType) / sizeof(VectorEntryType);
    static constexpr size_t MemoryAlignment = sizeof(EntryType) * Size;
    enum Constants {
        HasVectorDivision = true
    };
    typedef MIC::Mask<T> Mask;
    using MaskType = Mask;
    using mask_type = Mask;
    typedef typename Mask::AsArg MaskArgument;
    typedef Vector<T> AsArg; // for now only ICC can compile this code and it is not broken :)
    typedef VectorType VectorTypeArg;
    using reference = Detail::ElementReference<Vector>;

    inline const VectorType &data() const { return d.v(); }
    inline VectorType &data() { return d.v(); }

protected:
    typedef Common::VectorMemoryUnion<VectorType, VectorEntryType> StorageType;
    StorageType d;
    Vc_DEPRECATED("renamed to data()") inline const VectorType vdata() const { return d.v(); }

    template <typename V> static Vc_INTRINSIC VectorType _cast(V x)
    {
        return MIC::mic_cast<VectorType>(x);
    }

public:
    template <typename MemType>
    using UpDownC =
        MIC::UpDownConversion<VectorEntryType, typename std::decay<MemType>::type>;

    /**
     * Reinterpret some array of T as a vector of T. You may only do this if the pointer is
     * aligned correctly and the content of the memory isn't changed from somewhere else because
     * the load operation will happen implicitly at some later point(s).
     */
    static inline Vector fromMemory(T *mem) {
        assert(0 == (mem & (VectorAlignment - 1)));
        return reinterpret_cast<Vector<T> >(mem);
    }

#include "../common/generalinterface.h"
    using IndexType = SimdArray<int, Size>;

    static Vector Random();

    ///////////////////////////////////////////////////////////////////////////////////////////
    // internal: required to enable returning objects of VectorType
    Vc_INTRINSIC Vector(VectorType x) : d(x) {}

    // implict conversion from compatible Vector<U>
    template <typename U>
    Vc_INTRINSIC Vector(
        Vector<U> x,
        typename std::enable_if<Traits::is_implicit_cast_allowed<U, T>::value,
                                void *>::type = nullptr)
        : d(MIC::convert<U, T>(x.data()))
    {
    }

#if Vc_IS_VERSION_1
    // static_cast from the remaining Vector<U>
    template <typename U>
    Vc_DEPRECATED("use simd_cast instead of explicit type casting to "
                  "convert between vector types") Vc_INTRINSIC
        explicit Vector(
            Vector<U> x,
            typename std::enable_if<!Traits::is_implicit_cast_allowed<U, T>::value,
                                    void *>::type = nullptr)
        : d(MIC::convert<U, T>(x.data()))
    {
    }
#endif

    ///////////////////////////////////////////////////////////////////////////////////////////
    // broadcast
    Vc_INTRINSIC Vector(EntryType a) : d(MIC::_set1(a)) {}
    template <typename U>
    Vc_INTRINSIC Vector(
        U a,
        typename std::enable_if<std::is_same<U, int>::value && !std::is_same<U, EntryType>::value,
                                void *>::type = nullptr)
        : Vector(static_cast<EntryType>(a))
        {
        }

#include "../common/loadinterface.h"

    ///////////////////////////////////////////////////////////////////////////////////////////
    // zeroing
    inline void setZero();
    inline void setZero(MaskArgument k);
    inline void setZeroInverted(MaskArgument k);

    inline void setQnan();
    inline void setQnan(MaskArgument k);

    ///////////////////////////////////////////////////////////////////////////////////////////
    // stores in StoreMixin

#include "../common/gatherinterface.h"
#include "../common/scatterinterface.h"

    ///////////////////////////////////////////////////////////////////////////////////////////
    //prefix
    Vc_ALWAYS_INLINE Vector &operator++()
    {
        d.v() = MIC::_add<VectorEntryType>(d.v(), Detail::one(EntryType()));
        return *this;
    }
    //postfix
    Vc_ALWAYS_INLINE Vector operator++(int)
    {
        const Vector<T> r = *this;
        d.v() = MIC::_add<VectorEntryType>(d.v(), Detail::one(EntryType()));
        return r;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // aliasing scalar access
private:
    friend reference;
    Vc_INTRINSIC static value_type get(const Vector &o, int i) noexcept
    {
        return o.d.m(i);
    }
    template <typename U>
    Vc_INTRINSIC static void set(Vector &o, int i, U &&v) noexcept(
        noexcept(std::declval<value_type &>() = v))
    {
        return o.d.set(i, v);
    }

public:
    Vc_ALWAYS_INLINE reference operator[](size_t index) noexcept
    {
        static_assert(noexcept(reference{std::declval<Vector &>(), int()}), "");
        return {*this, int(index)};
    }
    Vc_ALWAYS_INLINE value_type operator[](size_t index) const noexcept
    {
        return d.m(index);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // unary operators
    Vc_INTRINSIC Vc_PURE Mask operator!() const
    {
        return *this == Zero();
    }
    Vc_PURE Vc_ALWAYS_INLINE Vc_FLATTEN Vector operator~() const
    {
#ifndef Vc_ENABLE_FLOAT_BIT_OPERATORS
        static_assert(std::is_integral<T>::value,
                      "bit-complement can only be used with Vectors of integral type");
#endif
        return MIC::_andnot(d.v(), MIC::allone<VectorType>());
    }
    Vc_PURE_L Vc_ALWAYS_INLINE_L Vector operator-() const Vc_PURE_R Vc_ALWAYS_INLINE_R;
    Vc_PURE Vc_ALWAYS_INLINE Vc_FLATTEN Vector operator+() const { return *this; }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // binary operators
    Vc_ALWAYS_INLINE_L Vector &operator<<=(AsArg x) Vc_ALWAYS_INLINE_R;
    Vc_ALWAYS_INLINE_L Vector &operator>>=(AsArg x) Vc_ALWAYS_INLINE_R;
    Vc_ALWAYS_INLINE_L Vector  operator<< (AsArg x) const Vc_ALWAYS_INLINE_R;
    Vc_ALWAYS_INLINE_L Vector  operator>> (AsArg x) const Vc_ALWAYS_INLINE_R;

    Vc_ALWAYS_INLINE_L Vector &operator<<=(unsigned int x) Vc_ALWAYS_INLINE_R;
    Vc_ALWAYS_INLINE_L Vector &operator>>=(unsigned int x) Vc_ALWAYS_INLINE_R;
    Vc_ALWAYS_INLINE_L Vector  operator<< (unsigned int x) const Vc_ALWAYS_INLINE_R;
    Vc_ALWAYS_INLINE_L Vector  operator>> (unsigned int x) const Vc_ALWAYS_INLINE_R;

    Vc_DEPRECATED("use isnegative(x) instead") Vc_INTRINSIC Vc_PURE Mask
        isNegative() const
    {
        return Vc::isnegative(*this);
    }

    Vc_INTRINSIC_L void assign(Vector<T> v, Mask mask) Vc_INTRINSIC_R;

    template <typename V2>
    Vc_DEPRECATED("Use simd_cast instead of Vector::staticCast") Vc_INTRINSIC V2
        staticCast() const
    {
        return V2(*this);
    }
    template <typename V2>
    Vc_DEPRECATED("use reinterpret_components_cast instead") Vc_INTRINSIC V2
        reinterpretCast() const
    {
        return MIC::mic_cast<typename V2::VectorType>(d.v());
    }

    Vc_ALWAYS_INLINE MIC::WriteMaskedVector<T> operator()(MaskArgument k)
    {
        return MIC::WriteMaskedVector<T>(this, k);
    }

    Vc_ALWAYS_INLINE EntryType min() const { return Detail::min(d.v(), EntryType()); }
    Vc_ALWAYS_INLINE EntryType max() const { return Detail::max(d.v(), EntryType()); }
    Vc_ALWAYS_INLINE EntryType product() const { return Detail::mul(d.v(), EntryType()); }
    Vc_ALWAYS_INLINE EntryType sum() const { return Detail::add(d.v(), EntryType()); }
    Vc_ALWAYS_INLINE_L Vector partialSum() const Vc_ALWAYS_INLINE_R;
    inline EntryType min(MaskArgument m) const;
    inline EntryType max(MaskArgument m) const;
    inline EntryType product(MaskArgument m) const;
    inline EntryType sum(MaskArgument m) const;

    Vc_INTRINSIC_L Vector shifted(int amount, Vector shiftIn) const Vc_INTRINSIC_R;
    Vc_INTRINSIC_L Vector shifted(int amount) const Vc_INTRINSIC_R;
    Vc_INTRINSIC_L Vector rotated(int amount) const Vc_INTRINSIC_R;
    Vc_INTRINSIC_L Vc_PURE_L Vector reversed() const Vc_INTRINSIC_R Vc_PURE_R;
    Vc_INTRINSIC Vector sorted() const { return MIC::SortHelper<T>::sort(d.v()); }

    template<typename F> void callWithValuesSorted(F &f) {
        EntryType value = d.m(0);
        f(value);
        for (int i = 1; i < Size; ++i) {
            if (EntryType(d.m(i)) != value) {
                value = d.m(i);
                f(value);
            }
        }
    }

    template<typename F> Vc_INTRINSIC void call(F &&f) const {
        Common::for_all_vector_entries<Size>([&](size_t i) { f(EntryType(d.m(i))); });
    }

    template<typename F> Vc_INTRINSIC void call(F &&f, const Mask &mask) const {
        for (size_t i : where(mask)) {
            f(EntryType(d.m(i)));
        }
    }

    template<typename F> Vc_INTRINSIC Vector<T> apply(F &&f) const {
        Vector<T> r;
        Common::for_all_vector_entries<Size>(
            [&](size_t i) { r.d.set(i, f(EntryType(d.m(i)))); });
        return r;
    }

    template<typename F> Vc_INTRINSIC Vector<T> apply(F &&f, const Mask &mask) const {
        Vector<T> r(*this);
        for (size_t i : where(mask)) {
            r.d.set(i, f(EntryType(r.d.m(i))));
        }
        return r;
    }

    template<typename IndexT> Vc_INTRINSIC void fill(EntryType (&f)(IndexT)) {
        Common::for_all_vector_entries<Size>([&](size_t i) { d.set(i, f(i)); });
    }
    Vc_INTRINSIC void fill(EntryType (&f)()) {
        Common::for_all_vector_entries<Size>([&](size_t i) { d.set(i, f()); });
    }

    template <typename G> static Vc_INTRINSIC Vector generate(G gen)
    {
        Vector r;
        Common::unrolled_loop<std::size_t, 0, Size>([&](std::size_t i) {
            r.d.set(i, gen(i));
        });
        return r;
    }

    Vc_DEPRECATED("use copysign(x, y) instead") Vc_INTRINSIC Vector
        copySign(AsArg reference) const
    {
        return Vc::copysign(*this, reference);
    }
    Vc_DEPRECATED("use exponent(x) instead") Vc_INTRINSIC Vector exponent() const
    {
        return Vc::exponent(*this);
    }

    Vc_INTRINSIC_L Vector interleaveLow(Vector x) const Vc_INTRINSIC_R;
    Vc_INTRINSIC_L Vector interleaveHigh(Vector x) const Vc_INTRINSIC_R;
};
#undef Vc_CURRENT_CLASS_NAME
template <typename T> constexpr size_t Vector<T, VectorAbi::Mic>::Size;
template <typename T> constexpr size_t Vector<T, VectorAbi::Mic>::MemoryAlignment;

Vc_INTRINSIC MIC::int_v    min(const MIC::int_v    &x, const MIC::int_v    &y) { return _mm512_min_epi32(x.data(), y.data()); }
Vc_INTRINSIC MIC::uint_v   min(const MIC::uint_v   &x, const MIC::uint_v   &y) { return _mm512_min_epu32(x.data(), y.data()); }
Vc_INTRINSIC MIC::short_v  min(const MIC::short_v  &x, const MIC::short_v  &y) { return _mm512_min_epi32(x.data(), y.data()); }
Vc_INTRINSIC MIC::ushort_v min(const MIC::ushort_v &x, const MIC::ushort_v &y) { return _mm512_min_epu32(x.data(), y.data()); }
Vc_INTRINSIC MIC::float_v  min(const MIC::float_v  &x, const MIC::float_v  &y) { return _mm512_min_ps   (x.data(), y.data()); }
Vc_INTRINSIC MIC::double_v min(const MIC::double_v &x, const MIC::double_v &y) { return _mm512_min_pd   (x.data(), y.data()); }
Vc_INTRINSIC MIC::int_v    max(const MIC::int_v    &x, const MIC::int_v    &y) { return _mm512_max_epi32(x.data(), y.data()); }
Vc_INTRINSIC MIC::uint_v   max(const MIC::uint_v   &x, const MIC::uint_v   &y) { return _mm512_max_epu32(x.data(), y.data()); }
Vc_INTRINSIC MIC::short_v  max(const MIC::short_v  &x, const MIC::short_v  &y) { return _mm512_max_epi32(x.data(), y.data()); }
Vc_INTRINSIC MIC::ushort_v max(const MIC::ushort_v &x, const MIC::ushort_v &y) { return _mm512_max_epu32(x.data(), y.data()); }
Vc_INTRINSIC MIC::float_v  max(const MIC::float_v  &x, const MIC::float_v  &y) { return _mm512_max_ps   (x.data(), y.data()); }
Vc_INTRINSIC MIC::double_v max(const MIC::double_v &x, const MIC::double_v &y) { return _mm512_max_pd   (x.data(), y.data()); }

Vc_ALWAYS_INLINE MIC::double_v atan2(MIC::double_v x, MIC::double_v y)
{
    return _mm512_atan2_pd(x.data(), y.data());
}
Vc_ALWAYS_INLINE MIC::float_v atan2(MIC::float_v x, MIC::float_v y)
{
    return _mm512_atan2_ps(x.data(), y.data());
}

template <typename T>
Vc_ALWAYS_INLINE Vc_CONST enable_if<std::is_signed<T>::value, MIC::Vector<T>> abs(
    MIC::Vector<T> x)
{
    return Detail::abs(x.data(), T());
}

#define Vc_MATH_OP1(name_, call_)                                                        \
    Vc_ALWAYS_INLINE MIC::double_v name_(MIC::double_v x)                                \
    {                                                                                    \
        return _mm512_##call_##_pd(x.data());                                            \
    }                                                                                    \
    Vc_ALWAYS_INLINE MIC::float_v name_(MIC::float_v x)                                  \
    {                                                                                    \
        return _mm512_##call_##_ps(x.data());                                            \
    }                                                                                    \
    Vc_NOTHING_EXPECTING_SEMICOLON
    Vc_MATH_OP1(sqrt, sqrt);
    Vc_MATH_OP1(rsqrt, rsqrt);
    Vc_MATH_OP1(sin, sin);
    Vc_MATH_OP1(cos, cos);
    Vc_MATH_OP1(log, log);
    Vc_MATH_OP1(log2, log2);
    Vc_MATH_OP1(log10, log10);
    Vc_MATH_OP1(atan, atan);
    Vc_MATH_OP1(reciprocal, recip);
    Vc_MATH_OP1(asin, asin);
    Vc_MATH_OP1(floor, floor);
    Vc_MATH_OP1(ceil, ceil);
    Vc_MATH_OP1(exp, exp);
#undef Vc_MATH_OP1
    Vc_ALWAYS_INLINE MIC::double_v round(MIC::double_v x)
    {
        return _mm512_roundfxpnt_adjust_pd(x.data(), _MM_FROUND_TO_NEAREST_INT,
                                           _MM_EXPADJ_NONE);
    }
    Vc_ALWAYS_INLINE MIC::float_v round(MIC::float_v x)
    {
        return _mm512_round_ps(x.data(), _MM_FROUND_TO_NEAREST_INT, _MM_EXPADJ_NONE);
    }
    template <typename T> Vc_ALWAYS_INLINE MIC::Vector<T> round(MIC::Vector<T> x)
    {
        return x;
    }

    template<typename T> static inline void sincos(const Vector<T> &x, Vector<T> *s, Vector<T> *c) {
        *s = sin(x);
        *c = cos(x);
    }

#define Vc_CONDITIONAL_ASSIGN(name_, op_)                                                \
    template <Operator O, typename T, typename M, typename U>                            \
    Vc_INTRINSIC enable_if<O == Operator::name_, void> conditional_assign(               \
        Vector<T> &lhs, M &&mask, U &&rhs)                                               \
    {                                                                                    \
        lhs(mask) op_ rhs;                                                               \
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

#define Vc_CONDITIONAL_ASSIGN(name_, expr_)                                              \
    template <Operator O, typename T, typename M>                                        \
    Vc_INTRINSIC enable_if<O == Operator::name_, Vector<T>> conditional_assign(          \
        Vector<T> &lhs, M &&mask)                                                        \
    {                                                                                    \
        return expr_;                                                                    \
    }
Vc_CONDITIONAL_ASSIGN(PostIncrement, lhs(mask)++)
Vc_CONDITIONAL_ASSIGN( PreIncrement, ++lhs(mask))
Vc_CONDITIONAL_ASSIGN(PostDecrement, lhs(mask)--)
Vc_CONDITIONAL_ASSIGN( PreDecrement, --lhs(mask))
#undef Vc_CONDITIONAL_ASSIGN

} // namespace Vc

#include "vector.tcc"
#include "simd_cast.h"

#ifdef CAN_OFFLOAD
#pragma offload_attribute(pop)
#endif

#endif // VC_MIC_VECTOR_H_
