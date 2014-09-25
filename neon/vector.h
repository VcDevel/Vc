/*{{{
Copyright © 2014 Matthias Kretz <kretz@kde.org>
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
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

}}}*/

#ifndef VC_NEON_VECTOR_H_
#define VC_NEON_VECTOR_H_

#include "intrinsics.h"
#include "mask.h"
#include "../common/storage.h"
#include "../common/writemaskedvector.h"
#include "../traits/type_traits.h"
#include "macros.h"

#ifdef isfinite
#undef isfinite
#endif
#ifdef isnan
#undef isnan
#endif

namespace Vc_VERSIONED_NAMESPACE
{
namespace NEON
{
template <typename T> class Vector
{
    static_assert(std::is_arithmetic<T>::value,
                  "Vector<T> only accepts arithmetic builtin types as template parameter T.");

public:
    FREE_STORE_OPERATORS_ALIGNED(16)  // TODO: uses _mm_malloc / _mm_free. Needs a replacement

    using VectorType = typename VectorTraits<T>::Type;
    using EntryType = T;
    using VectorEntryType = EntryType;

    static constexpr size_t Size = sizeof(VectorType) / sizeof(EntryType);
    static constexpr size_t MemoryAlignment = alignof(VectorType);

    using IndexType = simdarray<int, Size>;
    using MaskType = NEON::Mask<T>;
    using Mask = NEON::Mask<T>;
    using MaskArg = const Mask;
    using MaskArgument = const Mask;
    using AsArg = const Vector;

    // STL style member types:
    using vector_type = VectorType;
    using value_type = EntryType;
    using index_type = IndexType;
    using mask_type = MaskType;

private:
    using StorageType = Common::VectorMemoryUnion<VectorType, EntryType>;
    StorageType d;

public:
#include "../common/generalinterface.h"

    static Vc_INTRINSIC_L Vector Random() Vc_INTRINSIC_R;

    ///////////////////////////////////////////////////////////////////////////////////////////
    // internal: required to enable returning objects of VectorType from functions with return
    // type Vector<T>
    template <typename U,
              typename = enable_if<std::is_convertible<U, VectorType>::value &&
                                   !std::is_same<VectorType, EntryType>::
                                        value>  // we have a problem with double_v where
                                                // EntryType == VectorType == double. In
                                                // that case we need to disable this
                                                // constructor overload to resolve the
                                                // otherwise resulting ambiguity
              >
    Vc_INTRINSIC Vector(U x)
        : d(x)
    {
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // copy
    Vc_INTRINSIC Vector(const Vector &x) = default;
    Vc_INTRINSIC Vector &operator=(const Vector &v)
    {
        d.v() = v.d.v();
        return *this;
    }

#include "../common/vector/casts.h"
#include "../common/loadinterface.h"
#include "../common/storeinterface.h"

    ///////////////////////////////////////////////////////////////////////////////////////////
    // zeroing
    Vc_INTRINSIC_L void setZero() Vc_INTRINSIC_R;
    Vc_INTRINSIC_L void setZero(const Mask &k) Vc_INTRINSIC_R;
    Vc_INTRINSIC_L void setZeroInverted(const Mask &k) Vc_INTRINSIC_R;

    Vc_INTRINSIC_L void setQnan() Vc_INTRINSIC_R;
    Vc_INTRINSIC_L void setQnan(MaskArg k) Vc_INTRINSIC_R;

    ///////////////////////////////////////////////////////////////////////////////////////////
    // swizzles
    Vc_INTRINSIC_L Vc_PURE_L const Vector<T> &abcd() const Vc_INTRINSIC_R Vc_PURE_R;
    Vc_INTRINSIC_L Vc_PURE_L const Vector<T> cdab() const Vc_INTRINSIC_R Vc_PURE_R;
    Vc_INTRINSIC_L Vc_PURE_L const Vector<T> badc() const Vc_INTRINSIC_R Vc_PURE_R;
    Vc_INTRINSIC_L Vc_PURE_L const Vector<T> aaaa() const Vc_INTRINSIC_R Vc_PURE_R;
    Vc_INTRINSIC_L Vc_PURE_L const Vector<T> bbbb() const Vc_INTRINSIC_R Vc_PURE_R;
    Vc_INTRINSIC_L Vc_PURE_L const Vector<T> cccc() const Vc_INTRINSIC_R Vc_PURE_R;
    Vc_INTRINSIC_L Vc_PURE_L const Vector<T> dddd() const Vc_INTRINSIC_R Vc_PURE_R;
    Vc_INTRINSIC_L Vc_PURE_L const Vector<T> bcad() const Vc_INTRINSIC_R Vc_PURE_R;
    Vc_INTRINSIC_L Vc_PURE_L const Vector<T> bcda() const Vc_INTRINSIC_R Vc_PURE_R;
    Vc_INTRINSIC_L Vc_PURE_L const Vector<T> dabc() const Vc_INTRINSIC_R Vc_PURE_R;
    Vc_INTRINSIC_L Vc_PURE_L const Vector<T> acbd() const Vc_INTRINSIC_R Vc_PURE_R;
    Vc_INTRINSIC_L Vc_PURE_L const Vector<T> dbca() const Vc_INTRINSIC_R Vc_PURE_R;
    Vc_INTRINSIC_L Vc_PURE_L const Vector<T> dcba() const Vc_INTRINSIC_R Vc_PURE_R;

#include "../common/gatherinterface.h"
#include "../common/scatterinterface.h"

    ///////////////////////////////////////////////////////////////////////////////////////////
    // prefix
    Vc_ALWAYS_INLINE Vector &operator++();
    Vc_ALWAYS_INLINE Vector &operator--();
    // postfix
    Vc_ALWAYS_INLINE Vector operator++(int);
    Vc_ALWAYS_INLINE Vector operator--(int);

    Vc_INTRINSIC decltype(d.m(0)) operator[](size_t index) { return d.m(index); }
    Vc_ALWAYS_INLINE EntryType operator[](size_t index) const { return d.m(index); }

    Vc_INTRINSIC Vc_PURE Mask operator!() const { return *this == Zero(); }
    Vc_ALWAYS_INLINE Vector operator~() const;
    Vc_ALWAYS_INLINE_L Vc_PURE_L Vector operator-() const Vc_ALWAYS_INLINE_R Vc_PURE_R;
    Vc_INTRINSIC Vc_PURE Vector operator+() const { return *this; }

    Vc_ALWAYS_INLINE Vector &operator%=(const Vector &x)
    {
        *this = *this % x;
        return *this;
    }
    inline Vc_PURE Vector operator%(const Vector &x) const;

#define OP(symbol)                                                                            \
    Vc_ALWAYS_INLINE Vector &operator symbol##=(const Vector &x);                                  \
    Vc_ALWAYS_INLINE Vc_PURE Vector operator symbol(const Vector &x) const;

    OP(+)
    OP(-)
    OP(*)
#undef OP
    inline Vector &operator/=(EntryType x);
    inline Vector &operator/=(Vector x);
    inline Vc_PURE_L Vector operator/(Vector x) const Vc_PURE_R;

// bitwise ops
#define OP_VEC(op)                                                                                 \
    Vc_INTRINSIC Vector &operator op##=(AsArg x)                                                   \
    {                                                                                              \
        static_assert(std::is_integral<T>::value,                                                  \
                      "bitwise-operators can only be used with Vectors of integral type");         \
    }                                                                                              \
    Vc_INTRINSIC Vc_PURE Vector operator op(AsArg x) const                                         \
    {                                                                                              \
        static_assert(std::is_integral<T>::value,                                                  \
                      "bitwise-operators can only be used with Vectors of integral type");         \
    }
    VC_ALL_BINARY(OP_VEC)
    VC_ALL_SHIFTS(OP_VEC)
#undef OP_VEC

    Vc_ALWAYS_INLINE_L Vector<T> &operator>>=(int x) Vc_ALWAYS_INLINE_R;
    Vc_ALWAYS_INLINE_L Vector<T> &operator<<=(int x) Vc_ALWAYS_INLINE_R;
    Vc_ALWAYS_INLINE_L Vector<T> operator>>(int x) const Vc_ALWAYS_INLINE_R;
    Vc_ALWAYS_INLINE_L Vector<T> operator<<(int x) const Vc_ALWAYS_INLINE_R;

#define OPcmp(symbol, fun)                                                                         \
    Vc_ALWAYS_INLINE Vc_PURE Mask operator symbol(const Vector &x) const;

    OPcmp(==, cmpeq)
    OPcmp(!=, cmpneq)
    OPcmp(>=, cmpnlt)
    OPcmp(>, cmpnle)
    OPcmp(<, cmplt)
    OPcmp(<=, cmple)
#undef OPcmp
    Vc_INTRINSIC_L Vc_PURE_L Mask isNegative() const Vc_PURE_R Vc_INTRINSIC_R;

    Vc_ALWAYS_INLINE void fusedMultiplyAdd(const Vector<T> &factor, const Vector<T> &summand);

    Vc_ALWAYS_INLINE void assign(const Vector<T> &v, const Mask &mask);

    template <typename V2> Vc_ALWAYS_INLINE V2 staticCast() const { return V2(*this); }
    template <typename V2> Vc_ALWAYS_INLINE V2 reinterpretCast() const;

    /*
    Vc_ALWAYS_INLINE Common::WriteMaskedVector<T> operator()(const Mask &k)
    {
        return Common::WriteMaskedVector<T>(this, k);
    }
    */

    /**
     * \return \p true  This vector was completely filled. m2 might be 0 or != 0. You still have
     *                  to test this.
     *         \p false This vector was not completely filled. m2 is all 0.
     */
    // inline bool pack(Mask &m1, Vector<T> &v2, Mask &m2) {
    // return VectorHelper<T>::pack(data(), m1.data, v2.data(), m2.data);
    //}

    Vc_ALWAYS_INLINE VectorType &data() { return d.v(); }
    Vc_ALWAYS_INLINE const VectorType &data() const { return d.v(); }

    Vc_ALWAYS_INLINE EntryType min() const;
    Vc_ALWAYS_INLINE EntryType max() const;
    Vc_ALWAYS_INLINE EntryType product() const;
    Vc_ALWAYS_INLINE EntryType sum() const;
    Vc_ALWAYS_INLINE_L Vector partialSum() const Vc_ALWAYS_INLINE_R;
    // template<typename BinaryOperation> Vc_ALWAYS_INLINE_L Vector partialSum(BinaryOperation op)
    // const Vc_ALWAYS_INLINE_R;
    Vc_ALWAYS_INLINE_L EntryType min(MaskArg m) const Vc_ALWAYS_INLINE_R;
    Vc_ALWAYS_INLINE_L EntryType max(MaskArg m) const Vc_ALWAYS_INLINE_R;
    Vc_ALWAYS_INLINE_L EntryType product(MaskArg m) const Vc_ALWAYS_INLINE_R;
    Vc_ALWAYS_INLINE_L EntryType sum(MaskArg m) const Vc_ALWAYS_INLINE_R;

    Vc_INTRINSIC_L Vector shifted(int amount, Vector shiftIn) const Vc_INTRINSIC_R;
    Vc_INTRINSIC_L Vector shifted(int amount) const Vc_INTRINSIC_R;
    Vc_INTRINSIC_L Vector rotated(int amount) const Vc_INTRINSIC_R;
    Vc_ALWAYS_INLINE Vector sorted() const;

    template <typename F> void callWithValuesSorted(F &&f)
    {
        EntryType value = d.m(0);
        f(value);
        for (size_t i = 1; i < Size; ++i) {
            if (d.m(i) != value) {
                value = d.m(i);
                f(value);
            }
        }
    }

    template <typename F> Vc_INTRINSIC void call(F &&f) const
    {
        for_all_vector_entries(i, f(EntryType(d.m(i))););
    }

    template <typename F> Vc_INTRINSIC void call(F &&f, const Mask &mask) const
    {
        for (size_t i : where(mask)) {
            f(EntryType(d.m(i)));
        }
    }

    template <typename F> Vc_INTRINSIC Vector<T> apply(F &&f) const
    {
        Vector<T> r;
        for_all_vector_entries(i, r.d.m(i) = f(EntryType(d.m(i))););
        return r;
    }

    template <typename F> Vc_INTRINSIC Vector<T> apply(F &&f, const Mask &mask) const
    {
        Vector<T> r(*this);
        for (size_t i : where(mask)) {
            r.d.m(i) = f(EntryType(r.d.m(i)));
        }
        return r;
    }

    template <typename IndexT> Vc_INTRINSIC void fill(EntryType (&f)(IndexT))
    {
        for_all_vector_entries(i, d.m(i) = f(i););
    }
    Vc_INTRINSIC void fill(EntryType (&f)()) { for_all_vector_entries(i, d.m(i) = f();); }

    Vc_INTRINSIC_L Vector copySign(AsArg reference) const Vc_INTRINSIC_R;
    Vc_INTRINSIC_L Vector exponent() const Vc_INTRINSIC_R;
};
template <typename T> constexpr size_t Vector<T>::Size;

static_assert(Traits::is_simd_vector<double_v>::value, "is_simd_vector<double_v>::value");
static_assert(Traits::is_simd_vector<float_v>::value, "is_simd_vector< float_v>::value");
static_assert(Traits::is_simd_vector<int_v>::value, "is_simd_vector<   int_v>::value");
static_assert(Traits::is_simd_vector<uint_v>::value, "is_simd_vector<  uint_v>::value");
static_assert(Traits::is_simd_vector<short_v>::value, "is_simd_vector< short_v>::value");
static_assert(Traits::is_simd_vector<ushort_v>::value, "is_simd_vector<ushort_v>::value");
static_assert(Traits::is_simd_mask<double_m>::value, "is_simd_mask  <double_m>::value");
static_assert(Traits::is_simd_mask<float_m>::value, "is_simd_mask  < float_m>::value");
static_assert(Traits::is_simd_mask<int_m>::value, "is_simd_mask  <   int_m>::value");
static_assert(Traits::is_simd_mask<uint_m>::value, "is_simd_mask  <  uint_m>::value");
static_assert(Traits::is_simd_mask<short_m>::value, "is_simd_mask  < short_m>::value");
static_assert(Traits::is_simd_mask<ushort_m>::value, "is_simd_mask  <ushort_m>::value");

static_assert(!std::is_convertible<float *, short_v>::value,
              "A float* should never implicitly convert to short_v. Something is broken.");
static_assert(!std::is_convertible<int *, short_v>::value,
              "An int* should never implicitly convert to short_v. Something is broken.");
static_assert(!std::is_convertible<short *, short_v>::value,
              "A short* should never implicitly convert to short_v. Something is broken.");

}
}

#include "vector.tcc"
#include "undomacros.h"

#endif  // VC_NEON_VECTOR_H_
