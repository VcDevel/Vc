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

    static constexpr size_t Size = sizeof(VectorType) / sizeof(EntryType);
    static constexpr size_t MemoryAlignment = alignof(VectorType);

    using IndexType = simd_array<int, Size>;
    using MaskType = NEON::Mask<T>;

    // STL style member types:
    using vector_type = VectorType;
    using value_type = EntryType;
    using index_type = IndexType;
    using mask_type = MaskType;
    // STL container interface:
    static constexpr size_t size() { return Size; }

private:
    using StorageType = Common::VectorMemoryUnion<VectorType, EntryType>;
    StorageType d;

public:
#include "../common/generalinterface.h"

    static Vc_INTRINSIC_L Vector Random() Vc_INTRINSIC_R;

    ///////////////////////////////////////////////////////////////////////////////////////////
    // internal: required to enable returning objects of VectorType from functions with return
    // type Vector<T>
    Vc_INTRINSIC Vector(VectorType x) : d(x) {}

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
    // expand/merge 1 float_v <=> 2 double_v          XXX rationale? remove it for release? XXX
    explicit inline Vector(const Vector<typename HT::ConcatType> *a);
    inline void expand(Vector<typename HT::ConcatType> *x) const;

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
    Vc_ALWAYS_INLINE Vector &operator++()
    {
        data() = VectorHelper<T>::add(data(), VectorHelper<T>::one());
        return *this;
    }
    Vc_ALWAYS_INLINE Vector &operator--()
    {
        data() = VectorHelper<T>::sub(data(), VectorHelper<T>::one());
        return *this;
    }
    // postfix
    Vc_ALWAYS_INLINE Vector operator++(int)
    {
        const Vector<T> r = *this;
        data() = VectorHelper<T>::add(data(), VectorHelper<T>::one());
        return r;
    }
    Vc_ALWAYS_INLINE Vector operator--(int)
    {
        const Vector<T> r = *this;
        data() = VectorHelper<T>::sub(data(), VectorHelper<T>::one());
        return r;
    }

    Vc_INTRINSIC decltype(d.m(0)) operator[](size_t index) { return d.m(index); }
    Vc_ALWAYS_INLINE EntryType operator[](size_t index) const { return d.m(index); }

    Vc_INTRINSIC Vc_PURE Mask operator!() const { return *this == Zero(); }
    Vc_ALWAYS_INLINE Vector operator~() const
    {
#ifndef VC_ENABLE_FLOAT_BIT_OPERATORS
        static_assert(std::is_integral<T>::value,
                      "bit-complement can only be used with Vectors of integral type");
#endif
        return VectorHelper<VectorType>::andnot_(data(), VectorHelper<VectorType>::allone());
    }
    Vc_ALWAYS_INLINE_L Vc_PURE_L Vector operator-() const Vc_ALWAYS_INLINE_R Vc_PURE_R;
    Vc_INTRINSIC Vc_PURE Vector operator+() const { return *this; }

    Vc_ALWAYS_INLINE Vector &operator%=(const Vector &x)
    {
        *this = *this % x;
        return *this;
    }
    inline Vc_PURE Vector operator%(const Vector &x) const;

#define OP(symbol, fun)                                                                            \
    Vc_ALWAYS_INLINE Vector &operator symbol##=(const Vector &x)                                   \
    {                                                                                              \
        data() = VectorHelper<T>::fun(data(), x.data());                                           \
        return *this;                                                                              \
    }                                                                                              \
    Vc_ALWAYS_INLINE Vc_PURE Vector operator symbol(const Vector &x) const                         \
    {                                                                                              \
        return Vector<T>(VectorHelper<T>::fun(data(), x.data()));                                  \
    }

    OP(+, add)
    OP(-, sub)
    OP(*, mul)
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
    Vc_ALWAYS_INLINE Vc_PURE Mask operator symbol(const Vector &x) const                           \
    {                                                                                              \
        return HT::fun(data(), x.data());                                                          \
    }

    OPcmp(==, cmpeq) OPcmp(!=, cmpneq) OPcmp(>=, cmpnlt) OPcmp(>, cmpnle) OPcmp(<, cmplt)
        OPcmp(<=, cmple)
#undef OPcmp
        Vc_INTRINSIC_L Vc_PURE_L Mask isNegative() const Vc_PURE_R Vc_INTRINSIC_R;

    Vc_ALWAYS_INLINE void fusedMultiplyAdd(const Vector<T> &factor, const Vector<T> &summand)
    {
        VectorHelper<T>::fma(data(), factor.data(), summand.data());
    }

    Vc_ALWAYS_INLINE void assign(const Vector<T> &v, const Mask &mask)
    {
        const VectorType k = avx_cast<VectorType>(mask.data());
        data() = VectorHelper<VectorType>::blend(data(), v.data(), k);
    }

    template <typename V2> Vc_ALWAYS_INLINE V2 staticCast() const { return V2(*this); }
    template <typename V2> Vc_ALWAYS_INLINE V2 reinterpretCast() const
    {
        return avx_cast<typename V2::VectorType>(data());
    }

    Vc_ALWAYS_INLINE WriteMaskedVector<T> operator()(const Mask &k)
    {
        return WriteMaskedVector<T>(this, k);
    }

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

    Vc_ALWAYS_INLINE EntryType min() const { return VectorHelper<T>::min(data()); }
    Vc_ALWAYS_INLINE EntryType max() const { return VectorHelper<T>::max(data()); }
    Vc_ALWAYS_INLINE EntryType product() const { return VectorHelper<T>::mul(data()); }
    Vc_ALWAYS_INLINE EntryType sum() const { return VectorHelper<T>::add(data()); }
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
    Vc_ALWAYS_INLINE Vector sorted() const { return SortHelper<T>::sort(data()); }

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

template <typename T> class SwizzledVector : public Vector<T>
{
};

static Vc_ALWAYS_INLINE int_v min(const int_v &x, const int_v &y)
{
    return _mm256_min_epi32(x.data(), y.data());
}
static Vc_ALWAYS_INLINE uint_v min(const uint_v &x, const uint_v &y)
{
    return _mm256_min_epu32(x.data(), y.data());
}
static Vc_ALWAYS_INLINE short_v min(const short_v &x, const short_v &y)
{
    return _mm_min_epi16(x.data(), y.data());
}
static Vc_ALWAYS_INLINE ushort_v min(const ushort_v &x, const ushort_v &y)
{
    return _mm_min_epu16(x.data(), y.data());
}
static Vc_ALWAYS_INLINE float_v min(const float_v &x, const float_v &y)
{
    return _mm256_min_ps(x.data(), y.data());
}
static Vc_ALWAYS_INLINE double_v min(const double_v &x, const double_v &y)
{
    return _mm256_min_pd(x.data(), y.data());
}
static Vc_ALWAYS_INLINE int_v max(const int_v &x, const int_v &y)
{
    return _mm256_max_epi32(x.data(), y.data());
}
static Vc_ALWAYS_INLINE uint_v max(const uint_v &x, const uint_v &y)
{
    return _mm256_max_epu32(x.data(), y.data());
}
static Vc_ALWAYS_INLINE short_v max(const short_v &x, const short_v &y)
{
    return _mm_max_epi16(x.data(), y.data());
}
static Vc_ALWAYS_INLINE ushort_v max(const ushort_v &x, const ushort_v &y)
{
    return _mm_max_epu16(x.data(), y.data());
}
static Vc_ALWAYS_INLINE float_v max(const float_v &x, const float_v &y)
{
    return _mm256_max_ps(x.data(), y.data());
}
static Vc_ALWAYS_INLINE double_v max(const double_v &x, const double_v &y)
{
    return _mm256_max_pd(x.data(), y.data());
}

template <typename T,
          typename = enable_if<std::is_same<T, double>::value || std::is_same<T, float>::value ||
                               std::is_same<T, short>::value ||
                               std::is_same<T, int>::value>>
Vc_ALWAYS_INLINE Vc_PURE Vector<T> abs(Vector<T> x)
{
    return VectorHelper<T>::abs(x.data());
}

template <typename T> Vc_ALWAYS_INLINE Vc_PURE Vector<T> sqrt(const Vector<T> &x)
{
    return VectorHelper<T>::sqrt(x.data());
}
template <typename T> Vc_ALWAYS_INLINE Vc_PURE Vector<T> rsqrt(const Vector<T> &x)
{
    return VectorHelper<T>::rsqrt(x.data());
}
template <typename T> Vc_ALWAYS_INLINE Vc_PURE Vector<T> reciprocal(const Vector<T> &x)
{
    return VectorHelper<T>::reciprocal(x.data());
}
template <typename T> Vc_ALWAYS_INLINE Vc_PURE Vector<T> round(const Vector<T> &x)
{
    return VectorHelper<T>::round(x.data());
}

template <typename T> Vc_ALWAYS_INLINE Vc_PURE typename Vector<T>::Mask isfinite(const Vector<T> &x)
{
    return VectorHelper<T>::isFinite(x.data());
}
template <typename T> Vc_ALWAYS_INLINE Vc_PURE typename Vector<T>::Mask isinf(const Vector<T> &x)
{
    return VectorHelper<T>::isInfinite(x.data());
}
template <typename T> Vc_ALWAYS_INLINE Vc_PURE typename Vector<T>::Mask isnan(const Vector<T> &x)
{
    return VectorHelper<T>::isNaN(x.data());
}

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
