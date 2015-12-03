/*  This file is part of the Vc library. {{{
Copyright © 2009-2015 Matthias Kretz <kretz@kde.org>
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
#include "vectorhelper.h"
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
    Vc_FREE_STORE_OPERATORS_ALIGNED(64)
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

    inline const VectorType &data() const { return d.v(); }
    inline VectorType &data() { return d.v(); }

protected:
    // helper that specializes on VectorType
    typedef MIC::VectorHelper<VectorType> HV;

    // helper that specializes on T
    typedef MIC::VectorHelper<VectorEntryType> HT;

    typedef Common::VectorMemoryUnion<VectorType, VectorEntryType> StorageType;
    StorageType d;
    Vc_DEPRECATED("renamed to data()") inline const VectorType vdata() const { return d.v(); }

    template<typename V> static Vc_INTRINSIC VectorType _cast(Vc_ALIGNED_PARAMETER(V) x) { return MIC::mic_cast<VectorType>(x); }

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
        Vc_ALIGNED_PARAMETER(Vector<U>) x,
        typename std::enable_if<Traits::is_implicit_cast_allowed<U, T>::value,
                                void *>::type = nullptr)
        : d(MIC::convert<U, T>(x.data()))
    {
    }

    // static_cast from the remaining Vector<U>
    template <typename U>
    Vc_INTRINSIC explicit Vector(
        Vc_ALIGNED_PARAMETER(Vector<U>) x,
        typename std::enable_if<!Traits::is_implicit_cast_allowed<U, T>::value,
                                void *>::type = nullptr)
        : d(MIC::convert<U, T>(x.data()))
    {
    }

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
    Vc_ALWAYS_INLINE Vector &operator++() { d.v() = MIC::_add<VectorEntryType>(d.v(), HV::one()); return *this; }
    //postfix
    Vc_ALWAYS_INLINE Vector operator++(int) { const Vector<T> r = *this; d.v() = MIC::_add<VectorEntryType>(d.v(), HV::one()); return r; }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // aliasing scalar access
    Vc_INTRINSIC_L auto operator[](size_t index)Vc_INTRINSIC_R -> decltype(d.ref(0)) &;
    Vc_INTRINSIC_L EntryType operator[](size_t index) const Vc_INTRINSIC_R;

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
#ifdef Vc_ENABLE_FLOAT_BIT_OPERATORS
#define Vc_ASSERT_
#else
#define Vc_ASSERT_                                                                       \
    static_assert(std::is_integral<T>::value,                                            \
                  "bitwise-operators can only be used with Vectors of integral type")
#endif
#define Vc_OP(symbol, fun)                                                               \
    Vc_ALWAYS_INLINE Vector &operator symbol##=(AsArg x)                                 \
    {                                                                                    \
        Vc_ASSERT_;                                                                      \
        d.v() = fun(d.v(), x.d.v());                                                     \
        return *this;                                                                    \
    }                                                                                    \
    Vc_ALWAYS_INLINE Vector operator symbol(AsArg x) const                               \
    {                                                                                    \
        Vc_ASSERT_;                                                                      \
        return Vector<T>(fun(d.v(), x.d.v()));                                           \
    }

    Vc_OP(%, MIC::mod_<EntryType>)
    Vc_OP(*, MIC::_mul<VectorEntryType>)
    Vc_OP(+, MIC::_add<VectorEntryType>)
    Vc_OP(-, MIC::_sub<VectorEntryType>)
    Vc_OP(/, MIC::_div<VectorEntryType>) // ushort_v::operator/ is specialized in vector.tcc
    Vc_OP(|, MIC::_or)
    Vc_OP(&, MIC::_and)
    Vc_OP(^, MIC::_xor)
#undef Vc_OP

    Vc_ALWAYS_INLINE_L Vector &operator<<=(AsArg x) Vc_ALWAYS_INLINE_R;
    Vc_ALWAYS_INLINE_L Vector &operator>>=(AsArg x) Vc_ALWAYS_INLINE_R;
    Vc_ALWAYS_INLINE_L Vector  operator<< (AsArg x) const Vc_ALWAYS_INLINE_R;
    Vc_ALWAYS_INLINE_L Vector  operator>> (AsArg x) const Vc_ALWAYS_INLINE_R;

    Vc_ALWAYS_INLINE_L Vector &operator<<=(unsigned int x) Vc_ALWAYS_INLINE_R;
    Vc_ALWAYS_INLINE_L Vector &operator>>=(unsigned int x) Vc_ALWAYS_INLINE_R;
    Vc_ALWAYS_INLINE_L Vector  operator<< (unsigned int x) const Vc_ALWAYS_INLINE_R;
    Vc_ALWAYS_INLINE_L Vector  operator>> (unsigned int x) const Vc_ALWAYS_INLINE_R;

#define Vc_OPcmp(symbol, fun) \
    Vc_ALWAYS_INLINE Mask operator symbol(AsArg x) const { return HT::fun(d.v(), x.d.v()); }

    // ushort_v specializations are in vector.tcc!
    Vc_OPcmp(==, cmpeq)
    Vc_OPcmp(!=, cmpneq)
    Vc_OPcmp(>=, cmpnlt)
    Vc_OPcmp(>, cmpnle)
    Vc_OPcmp(<, cmplt)
    Vc_OPcmp(<=, cmple)
#undef Vc_OPcmp

    Vc_INTRINSIC Vc_PURE Vc_DEPRECATED("use isnegative(x) instead") Mask
        isNegative() const
    {
        return Vc::isnegative(*this);
    }

    Vc_INTRINSIC_L void assign(Vector<T> v, Mask mask) Vc_INTRINSIC_R;

    template <typename V2>
    Vc_INTRINSIC Vc_DEPRECATED("Use simd_cast instead of Vector::staticCast") V2
        staticCast() const
    {
        return V2(*this);
    }
    template<typename V2> Vc_INTRINSIC V2 reinterpretCast() const { return MIC::mic_cast<typename V2::VectorType>(d.v()); }

    Vc_ALWAYS_INLINE MIC::WriteMaskedVector<T> operator()(MaskArgument k)
    {
        return MIC::WriteMaskedVector<T>(this, k);
    }

    inline EntryType min() const { return HT::reduce_min(d.v()); }
    inline EntryType max() const { return HT::reduce_max(d.v()); }
    inline EntryType product() const { return HT::reduce_mul(d.v()); }
    inline EntryType sum() const { return HT::reduce_add(d.v()); }
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

    Vc_INTRINSIC_L Vector copySign(AsArg reference) const Vc_INTRINSIC_R;
    Vc_INTRINSIC Vc_DEPRECATED("use exponent(x) instead") Vector exponent() const
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
template <typename T>
static inline MIC::Vector<T> atan2(MIC::Vector<T> x, MIC::Vector<T> y)
{
    return MIC::VectorHelper<typename MIC::Vector<T>::VectorEntryType>::atan2(x.data(),
                                                                              y.data());
}

template <typename T,
          typename =
              enable_if<std::is_same<T, double>::value || std::is_same<T, float>::value ||
                        std::is_same<T, short>::value || std::is_same<T, int>::value>>
Vc_ALWAYS_INLINE Vc_PURE MIC::Vector<T> abs(MIC::Vector<T> x)
{
    return MIC::VectorHelper<typename MIC::Vector<T>::VectorEntryType>::abs(x.data());
}

#define Vc_MATH_OP1(name, call)                                                          \
    template <typename T> static Vc_ALWAYS_INLINE Vector<T> name(const Vector<T> &x)     \
    {                                                                                    \
        typedef MIC::VectorHelper<typename Vector<T>::VectorEntryType> HT;               \
        return HT::call(x.data());                                                       \
    }
    Vc_MATH_OP1(sqrt, sqrt)
    Vc_MATH_OP1(rsqrt, rsqrt)
    Vc_MATH_OP1(sin, sin)
    Vc_MATH_OP1(cos, cos)
    Vc_MATH_OP1(log, log)
    Vc_MATH_OP1(log2, log2)
    Vc_MATH_OP1(log10, log10)
    Vc_MATH_OP1(atan, atan)
    Vc_MATH_OP1(reciprocal, recip)
    Vc_MATH_OP1(round, round)
    Vc_MATH_OP1(asin, asin)
    Vc_MATH_OP1(floor, floor)
    Vc_MATH_OP1(ceil, ceil)
    Vc_MATH_OP1(exp, exp)
#undef Vc_MATH_OP1

    template<typename T> static inline void sincos(const Vector<T> &x, Vector<T> *sin, Vector<T> *cos) {
        typedef MIC::VectorHelper<typename Vector<T>::VectorEntryType> HT; \
        *sin = HT::sin(x.data());
        *cos = HT::cos(x.data());
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
