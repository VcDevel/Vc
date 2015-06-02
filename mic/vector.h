/*  This file is part of the Vc library. {{{
Copyright © 2009-2014 Matthias Kretz <kretz@kde.org>
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

#ifndef VC_MIC_VECTOR_H
#define VC_MIC_VECTOR_H

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
#include "iterators.h"
#include "where.h"
#include "macros.h"

#ifdef isfinite
#undef isfinite
#endif
#ifdef isnan
#undef isnan
#endif

#define VC_HAVE_FMA

namespace Vc_VERSIONED_NAMESPACE
{
namespace Vc_IMPL_NAMESPACE
{

#define VC_CURRENT_CLASS_NAME Vector
template<typename T> class Vector : public StoreMixin<Vector<T>, T>
{
    static_assert(std::is_arithmetic<T>::value,
                  "Vector<T> only accepts arithmetic builtin types as template parameter T.");

    //friend class VectorMultiplication<T>;
    friend class WriteMaskedVector<T>;
    //friend Vector<T> operator+<>(const Vector<T> &x, const VectorMultiplication<T> &y);
    //friend Vector<T> operator-<>(const Vector<T> &x, const VectorMultiplication<T> &y);
    friend class Vector<float>;
    friend class Vector<double>;
    friend class Vector<int>;
    friend class Vector<unsigned int>;
    friend class StoreMixin<Vector<T>, T>;

public:
    FREE_STORE_OPERATORS_ALIGNED(64)
    typedef typename VectorTypeHelper<T>::Type VectorType;
    using vector_type = VectorType;
    typedef typename DetermineEntryType<T>::Type EntryType;
    using value_type = EntryType;
    typedef typename DetermineVectorEntryType<T>::Type VectorEntryType;
    static constexpr size_t Size = sizeof(VectorType) / sizeof(VectorEntryType);
    typedef simdarray<int, 16, MIC::int_v, 16> IndexType;
    enum Constants {
        MemoryAlignment = sizeof(EntryType) * Size,
        HasVectorDivision = true
    };
    typedef Vc_IMPL_NAMESPACE::Mask<T> Mask;
    using MaskType = Mask;
    using mask_type = Mask;
    typedef typename Mask::AsArg MaskArgument;
    typedef Vector<T> AsArg; // for now only ICC can compile this code and it is not broken :)
    typedef VectorType VectorTypeArg;

    inline const VectorType data() const { return d.v(); }
    inline VectorType &data() { return d.v(); }

protected:
    // helper that specializes on VectorType
    typedef VectorHelper<VectorType> HV;

    // helper that specializes on T
    typedef VectorHelper<VectorEntryType> HT;

    typedef Common::VectorMemoryUnion<VectorType, VectorEntryType> StorageType;
    StorageType d;
    VC_DEPRECATED("renamed to data()") inline const VectorType vdata() const { return d.v(); }

    template<typename V> static Vc_INTRINSIC VectorType _cast(VC_ALIGNED_PARAMETER(V) x) { return mic_cast<VectorType>(x); }

public:
    template<typename MemType> using UpDownC = UpDownConversion<VectorEntryType, typename std::decay<MemType>::type>;

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

    static Vector Random();

    ///////////////////////////////////////////////////////////////////////////////////////////
    // internal: required to enable returning objects of VectorType
    Vc_INTRINSIC Vector(VectorType x) : d(x) {}

    // implict conversion from compatible Vector<U>
    template <typename U>
    Vc_INTRINSIC Vector(
        VC_ALIGNED_PARAMETER(Vector<U>) x,
        typename std::enable_if<is_implicit_cast_allowed<U, T>::value, void *>::type = nullptr)
        : d(StaticCastHelper<U, T>::cast(x.data()))
    {
    }

    // static_cast from the remaining Vector<U>
    template <typename U>
    Vc_INTRINSIC explicit Vector(
        VC_ALIGNED_PARAMETER(Vector<U>) x,
        typename std::enable_if<!is_implicit_cast_allowed<U, T>::value, void *>::type = nullptr)
        : d(StaticCastHelper<U, T>::cast(x.data()))
    {
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // broadcast
    Vc_INTRINSIC Vector(EntryType a) : d(_set1(a)) {}
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

    ///////////////////////////////////////////////////////////////////////////////////////////
    // swizzles
    Vc_INTRINSIC_L Vc_CONST_L const Vector<T> &abcd() const Vc_INTRINSIC_R Vc_CONST_R;
    Vc_INTRINSIC_L Vc_CONST_L       Vector<T>  cdab() const Vc_INTRINSIC_R Vc_CONST_R;
    Vc_INTRINSIC_L Vc_CONST_L       Vector<T>  badc() const Vc_INTRINSIC_R Vc_CONST_R;
    Vc_INTRINSIC_L Vc_CONST_L       Vector<T>  aaaa() const Vc_INTRINSIC_R Vc_CONST_R;
    Vc_INTRINSIC_L Vc_CONST_L       Vector<T>  bbbb() const Vc_INTRINSIC_R Vc_CONST_R;
    Vc_INTRINSIC_L Vc_CONST_L       Vector<T>  cccc() const Vc_INTRINSIC_R Vc_CONST_R;
    Vc_INTRINSIC_L Vc_CONST_L       Vector<T>  dddd() const Vc_INTRINSIC_R Vc_CONST_R;
    Vc_INTRINSIC_L Vc_CONST_L       Vector<T>  bcad() const Vc_INTRINSIC_R Vc_CONST_R;
    Vc_INTRINSIC_L Vc_CONST_L       Vector<T>  bcda() const Vc_INTRINSIC_R Vc_CONST_R;
    Vc_INTRINSIC_L Vc_CONST_L       Vector<T>  dabc() const Vc_INTRINSIC_R Vc_CONST_R;
    Vc_INTRINSIC_L Vc_CONST_L       Vector<T>  acbd() const Vc_INTRINSIC_R Vc_CONST_R;
    Vc_INTRINSIC_L Vc_CONST_L       Vector<T>  dbca() const Vc_INTRINSIC_R Vc_CONST_R;
    Vc_INTRINSIC_L Vc_CONST_L       Vector<T>  dcba() const Vc_INTRINSIC_R Vc_CONST_R;

#include "../common/gatherinterface.h"
#include "../common/scatterinterface.h"

    ///////////////////////////////////////////////////////////////////////////////////////////
    //prefix
    Vc_ALWAYS_INLINE Vector &operator++() { d.v() = _add<VectorEntryType>(d.v(), HV::one()); return *this; }
    //postfix
    Vc_ALWAYS_INLINE Vector operator++(int) { const Vector<T> r = *this; d.v() = _add<VectorEntryType>(d.v(), HV::one()); return r; }

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
#ifndef VC_ENABLE_FLOAT_BIT_OPERATORS
        static_assert(std::is_integral<T>::value,
                      "bit-complement can only be used with Vectors of integral type");
#endif
        return _andnot(d.v(), _setallone<VectorType>());
    }
    Vc_PURE_L Vc_ALWAYS_INLINE_L Vector operator-() const Vc_PURE_R Vc_ALWAYS_INLINE_R;
    Vc_PURE Vc_ALWAYS_INLINE Vc_FLATTEN Vector operator+() const { return *this; }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // binary operators
#ifdef VC_ENABLE_FLOAT_BIT_OPERATORS
#define VC_ASSERT__
#else
#define VC_ASSERT__                                                                                \
    static_assert(std::is_integral<T>::value,                                                      \
                  "bitwise-operators can only be used with Vectors of integral type")
#endif
#define Vc_OP(symbol, fun)                                                                         \
    Vc_ALWAYS_INLINE Vector &operator symbol##=(AsArg x)                                           \
    {                                                                                              \
        VC_ASSERT__;                                                                               \
        d.v() = fun(d.v(), x.d.v());                                                               \
        return *this;                                                                              \
    }                                                                                              \
    Vc_ALWAYS_INLINE Vector operator symbol(AsArg x) const                                         \
    {                                                                                              \
        VC_ASSERT__;                                                                               \
        return Vector<T>(fun(d.v(), x.d.v()));                                                     \
    }

    Vc_OP(%, mod_<VectorEntryType>)
    Vc_OP(*, _mul<VectorEntryType>)
    Vc_OP(+, _add<VectorEntryType>)
    Vc_OP(-, _sub<VectorEntryType>)
    Vc_OP(/, _div<VectorEntryType>) // ushort_v::operator/ is specialized in vector.tcc
    Vc_OP(|, _or)
    Vc_OP(&, _and)
    Vc_OP(^, _xor)
#undef Vc_OP

    Vc_ALWAYS_INLINE_L Vector &operator<<=(AsArg x) Vc_ALWAYS_INLINE_R;
    Vc_ALWAYS_INLINE_L Vector &operator>>=(AsArg x) Vc_ALWAYS_INLINE_R;
    Vc_ALWAYS_INLINE_L Vector  operator<< (AsArg x) const Vc_ALWAYS_INLINE_R;
    Vc_ALWAYS_INLINE_L Vector  operator>> (AsArg x) const Vc_ALWAYS_INLINE_R;

    Vc_ALWAYS_INLINE_L Vector &operator<<=(unsigned int x) Vc_ALWAYS_INLINE_R;
    Vc_ALWAYS_INLINE_L Vector &operator>>=(unsigned int x) Vc_ALWAYS_INLINE_R;
    Vc_ALWAYS_INLINE_L Vector  operator<< (unsigned int x) const Vc_ALWAYS_INLINE_R;
    Vc_ALWAYS_INLINE_L Vector  operator>> (unsigned int x) const Vc_ALWAYS_INLINE_R;

#define OPcmp(symbol, fun) \
    Vc_ALWAYS_INLINE Mask operator symbol(AsArg x) const { return HT::fun(d.v(), x.d.v()); }

    // ushort_v specializations are in vector.tcc!
    OPcmp(==, cmpeq)
    OPcmp(!=, cmpneq)
    OPcmp(>=, cmpnlt)
    OPcmp(>, cmpnle)
    OPcmp(<, cmplt)
    OPcmp(<=, cmple)
#undef OPcmp
    Vc_INTRINSIC_L Vc_PURE_L Mask isNegative() const Vc_INTRINSIC_R Vc_PURE_R;

    Vc_INTRINSIC void fusedMultiplyAdd(const Vector<T> &factor, const Vector<T> &summand) {
        d.v() = HT::multiplyAndAdd(d.v(), factor.data(), summand.data());
    }

    Vc_INTRINSIC_L void assign(Vector<T> v, Mask mask) Vc_INTRINSIC_R;

    template<typename V2> Vc_INTRINSIC V2 staticCast() const { return V2(*this); }
    template<typename V2> Vc_INTRINSIC V2 reinterpretCast() const { return mic_cast<typename V2::VectorType>(d.v()); }

    Vc_ALWAYS_INLINE WriteMaskedVector<T> operator()(MaskArgument k) { return WriteMaskedVector<T>(this, k); }

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
    Vc_INTRINSIC Vector sorted() const { return SortHelper<T>::sort(d.v()); }

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
        for_all_vector_entries(i,
                f(EntryType(d.m(i)));
                );
    }

    template<typename F> Vc_INTRINSIC void call(F &&f, const Mask &mask) const {
        Vc_foreach_bit(size_t i, mask) {
            f(EntryType(d.m(i)));
        }
    }

    template<typename F> Vc_INTRINSIC Vector<T> apply(F &&f) const {
        Vector<T> r;
        for_all_vector_entries(i,
                r.d.set(i, f(EntryType(d.m(i))));
                );
        return r;
    }

    template<typename F> Vc_INTRINSIC Vector<T> apply(F &&f, const Mask &mask) const {
        Vector<T> r(*this);
        Vc_foreach_bit (size_t i, mask) {
            r.d.set(i, f(EntryType(r.d.m(i))));
        }
        return r;
    }

    template<typename IndexT> Vc_INTRINSIC void fill(EntryType (&f)(IndexT)) {
        for_all_vector_entries(i,
                d.set(i, f(i));
                );
    }
    Vc_INTRINSIC void fill(EntryType (&f)()) {
        for_all_vector_entries(i,
                d.set(i, f());
                );
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
    Vc_INTRINSIC_L Vector exponent() const Vc_INTRINSIC_R;

    Vc_INTRINSIC_L Vector interleaveLow(Vector x) const Vc_INTRINSIC_R;
    Vc_INTRINSIC_L Vector interleaveHigh(Vector x) const Vc_INTRINSIC_R;
};
#undef VC_CURRENT_CLASS_NAME
template<typename T> constexpr size_t Vector<T>::Size;

template<typename T> struct SwizzledVector
{
    Vector<T> v;
    unsigned int s;
};

#define MATH_OP2(name, call) \
    template<typename T> static inline Vector<T> name(Vector<T> x, Vector<T> y) \
    { \
        typedef VectorHelper<typename Vector<T>::VectorEntryType> HT; \
        return HT::call(x.data(), y.data()); \
    }
    MATH_OP2(min, min)
    MATH_OP2(max, max)
    MATH_OP2(atan2, atan2)
#undef MATH_OP2

template <typename T,
          typename = enable_if<std::is_same<T, double>::value || std::is_same<T, float>::value ||
                               std::is_same<T, short>::value ||
                               std::is_same<T, int>::value>>
Vc_ALWAYS_INLINE Vc_PURE Vector<T> abs(Vector<T> x)
{
    return VectorHelper<typename Vector<T>::VectorEntryType>::abs(x.data());
}

#define MATH_OP1(name, call) \
    template<typename T> static Vc_ALWAYS_INLINE Vector<T> name(const Vector<T> &x) \
    { \
        typedef VectorHelper<typename Vector<T>::VectorEntryType> HT; \
        return HT::call(x.data()); \
    }
    MATH_OP1(sqrt, sqrt)
    MATH_OP1(rsqrt, rsqrt)
    MATH_OP1(sin, sin)
    MATH_OP1(cos, cos)
    MATH_OP1(log, log)
    MATH_OP1(log2, log2)
    MATH_OP1(log10, log10)
    MATH_OP1(atan, atan)
    MATH_OP1(reciprocal, recip)
    MATH_OP1(round, round)
    MATH_OP1(asin, asin)
    MATH_OP1(floor, floor)
    MATH_OP1(ceil, ceil)
    MATH_OP1(exp, exp)
#undef MATH_OP1

    template<typename T> static inline void sincos(const Vector<T> &x, Vector<T> *sin, Vector<T> *cos) {
        typedef VectorHelper<typename Vector<T>::VectorEntryType> HT; \
        *sin = HT::sin(x.data());
        *cos = HT::cos(x.data());
    }

#include "forcetoregisters.tcc"

#define Vc_CONDITIONAL_ASSIGN(name__, op__)                                              \
    template <Operator O, typename T, typename M, typename U>                            \
    Vc_INTRINSIC enable_if<O == Operator::name__, void> conditional_assign(              \
        Vector<T> &lhs, M &&mask, U &&rhs)                                               \
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
    template <Operator O, typename T, typename M>                                        \
    Vc_INTRINSIC enable_if<O == Operator::name__, Vector<T>> conditional_assign(         \
        Vector<T> &lhs, M &&mask)                                                        \
    {                                                                                    \
        return expr__;                                                                   \
    }
Vc_CONDITIONAL_ASSIGN(PostIncrement, lhs(mask)++)
Vc_CONDITIONAL_ASSIGN( PreIncrement, ++lhs(mask))
Vc_CONDITIONAL_ASSIGN(PostDecrement, lhs(mask)--)
Vc_CONDITIONAL_ASSIGN( PreDecrement, --lhs(mask))
#undef Vc_CONDITIONAL_ASSIGN

} // namespace MIC
using MIC::conditional_assign;
} // namespace Vc

#include "vector.tcc"
#include "undomacros.h"

#ifdef CAN_OFFLOAD
#pragma offload_attribute(pop)
#endif

#endif // VC_MIC_VECTOR_H
