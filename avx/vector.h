/*  This file is part of the Vc library.

    Copyright (C) 2009-2012 Matthias Kretz <kretz@kde.org>

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

*/

#ifndef AVX_VECTOR_H
#define AVX_VECTOR_H

#include "intrinsics.h"
#include "vectorhelper.h"
#include "mask.h"
#include "writemaskedvector.h"
#include "sorthelper.h"
#include <algorithm>
#include <cmath>
#include "../common/aliasingentryhelper.h"
#include "../common/memoryfwd.h"
#include "macros.h"

#ifdef isfinite
#undef isfinite
#endif
#ifdef isnan
#undef isnan
#endif

namespace Vc
{
namespace AVX
{
enum VectorAlignmentEnum { VectorAlignment = 32 };

template<typename T> class Vector
{
    public:
        FREE_STORE_OPERATORS_ALIGNED(32)

        typedef typename VectorTypeHelper<T>::Type VectorType;
        typedef typename DetermineEntryType<T>::Type EntryType;
        enum Constants {
            Size = sizeof(VectorType) / sizeof(EntryType),
            HasVectorDivision = HasVectorDivisionHelper<T>::Value
        };
        typedef Vector<typename IndexTypeHelper<T>::Type> IndexType;
        typedef typename Vc::AVX::Mask<Size, sizeof(VectorType)> Mask;
        typedef typename Mask::AsArg MaskArg;
        typedef Vc::Memory<Vector<T>, Size> Memory;
#ifdef VC_PASSING_VECTOR_BY_VALUE_IS_BROKEN
        typedef const Vector<T> &AsArg;
#else
        typedef Vector<T> AsArg;
#endif

    protected:
        // helper that specializes on VectorType
        typedef VectorHelper<VectorType> HV;

        // helper that specializes on T
        typedef VectorHelper<T> HT;

        // cast any m256/m128 to VectorType
        static inline VectorType INTRINSIC _cast(__m128  v) { return avx_cast<VectorType>(v); }
        static inline VectorType INTRINSIC _cast(__m128i v) { return avx_cast<VectorType>(v); }
        static inline VectorType INTRINSIC _cast(__m128d v) { return avx_cast<VectorType>(v); }
        static inline VectorType INTRINSIC _cast(__m256  v) { return avx_cast<VectorType>(v); }
        static inline VectorType INTRINSIC _cast(__m256i v) { return avx_cast<VectorType>(v); }
        static inline VectorType INTRINSIC _cast(__m256d v) { return avx_cast<VectorType>(v); }

        typedef Common::VectorMemoryUnion<VectorType, EntryType> StorageType;
        StorageType d;

    public:
        ///////////////////////////////////////////////////////////////////////////////////////////
        // uninitialized
        inline Vector() {}

        ///////////////////////////////////////////////////////////////////////////////////////////
        // constants
        explicit Vector(VectorSpecialInitializerZero::ZEnum);
        explicit Vector(VectorSpecialInitializerOne::OEnum);
        explicit Vector(VectorSpecialInitializerIndexesFromZero::IEnum);
        static Vector Zero();
        static Vector One();
        static Vector IndexesFromZero();
        static Vector Random();

        ///////////////////////////////////////////////////////////////////////////////////////////
        // internal: required to enable returning objects of VectorType
        inline Vector(const VectorType &x) : d(x) {}

        ///////////////////////////////////////////////////////////////////////////////////////////
        // static_cast / copy ctor
        template<typename T2> explicit Vector(Vector<T2> x);

        // implicit cast
        template<typename OtherT> inline INTRINSIC_L Vector &operator=(const Vector<OtherT> &x) INTRINSIC_R;

        // copy assignment
        inline Vector &operator=(AsArg v) { d.v() = v.d.v(); return *this; }

        ///////////////////////////////////////////////////////////////////////////////////////////
        // broadcast
        explicit Vector(EntryType a);
        template<typename TT> inline INTRINSIC Vector(TT x, VC_EXACT_TYPE(TT, EntryType, void *) = 0) : d(HT::set(x)) {}
        inline Vector &operator=(EntryType a) { d.v() = HT::set(a); return *this; }

        ///////////////////////////////////////////////////////////////////////////////////////////
        // load ctors
        explicit inline INTRINSIC_L
            Vector(const EntryType *x) INTRINSIC_R;
        template<typename Alignment> inline INTRINSIC_L
            Vector(const EntryType *x, Alignment align) INTRINSIC_R;
        template<typename OtherT> explicit inline INTRINSIC_L
            Vector(const OtherT    *x) INTRINSIC_R;
        template<typename OtherT, typename Alignment> inline INTRINSIC_L
            Vector(const OtherT    *x, Alignment align) INTRINSIC_R;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // load member functions
        inline INTRINSIC_L
            void load(const EntryType *mem) INTRINSIC_R;
        template<typename Alignment> inline INTRINSIC_L
            void load(const EntryType *mem, Alignment align) INTRINSIC_R;
        template<typename OtherT> inline INTRINSIC_L
            void load(const OtherT    *mem) INTRINSIC_R;
        template<typename OtherT, typename Alignment> inline INTRINSIC_L
            void load(const OtherT    *mem, Alignment align) INTRINSIC_R;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // expand/merge 1 float_v <=> 2 double_v          XXX rationale? remove it for release? XXX
        explicit inline Vector(const Vector<typename HT::ConcatType> *a);
        inline void expand(Vector<typename HT::ConcatType> *x) const;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // zeroing
        inline void setZero();
        inline void setZero(const Mask &k);

        void setQnan();
        void setQnan(MaskArg k);

        ///////////////////////////////////////////////////////////////////////////////////////////
        // stores
        inline void store(EntryType *mem) const;
        inline void store(EntryType *mem, const Mask &mask) const;
        template<typename A> inline void store(EntryType *mem, A align) const;
        template<typename A> inline void store(EntryType *mem, const Mask &mask, A align) const;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // swizzles
        inline const Vector<T> INTRINSIC_L CONST_L &abcd() const INTRINSIC_R CONST_R;
        inline const Vector<T> INTRINSIC_L CONST_L  cdab() const INTRINSIC_R CONST_R;
        inline const Vector<T> INTRINSIC_L CONST_L  badc() const INTRINSIC_R CONST_R;
        inline const Vector<T> INTRINSIC_L CONST_L  aaaa() const INTRINSIC_R CONST_R;
        inline const Vector<T> INTRINSIC_L CONST_L  bbbb() const INTRINSIC_R CONST_R;
        inline const Vector<T> INTRINSIC_L CONST_L  cccc() const INTRINSIC_R CONST_R;
        inline const Vector<T> INTRINSIC_L CONST_L  dddd() const INTRINSIC_R CONST_R;
        inline const Vector<T> INTRINSIC_L CONST_L  bcad() const INTRINSIC_R CONST_R;
        inline const Vector<T> INTRINSIC_L CONST_L  bcda() const INTRINSIC_R CONST_R;
        inline const Vector<T> INTRINSIC_L CONST_L  dabc() const INTRINSIC_R CONST_R;
        inline const Vector<T> INTRINSIC_L CONST_L  acbd() const INTRINSIC_R CONST_R;
        inline const Vector<T> INTRINSIC_L CONST_L  dbca() const INTRINSIC_R CONST_R;
        inline const Vector<T> INTRINSIC_L CONST_L  dcba() const INTRINSIC_R CONST_R;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // gathers
        template<typename IndexT> Vector(const EntryType *mem, const IndexT *indexes);
        template<typename IndexT> Vector(const EntryType *mem, Vector<IndexT> indexes);
        template<typename IndexT> Vector(const EntryType *mem, const IndexT *indexes, MaskArg mask);
        template<typename IndexT> Vector(const EntryType *mem, Vector<IndexT> indexes, MaskArg mask);
        template<typename S1, typename IT> Vector(const S1 *array, const EntryType S1::* member1, IT indexes);
        template<typename S1, typename IT> Vector(const S1 *array, const EntryType S1::* member1, IT indexes, MaskArg mask);
        template<typename S1, typename S2, typename IT> Vector(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes);
        template<typename S1, typename S2, typename IT> Vector(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes, MaskArg mask);
        template<typename S1, typename IT1, typename IT2> Vector(const S1 *array, const EntryType *const S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes);
        template<typename S1, typename IT1, typename IT2> Vector(const S1 *array, const EntryType *const S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes, MaskArg mask);
        template<typename Index> void gather(const EntryType *mem, Index indexes);
        template<typename Index> void gather(const EntryType *mem, Index indexes, MaskArg mask);
#ifdef VC_USE_SET_GATHERS
        template<typename IT> void gather(const EntryType *mem, Vector<IT> indexes, MaskArg mask);
#endif
        template<typename S1, typename IT> void gather(const S1 *array, const EntryType S1::* member1, IT indexes);
        template<typename S1, typename IT> void gather(const S1 *array, const EntryType S1::* member1, IT indexes, MaskArg mask);
        template<typename S1, typename S2, typename IT> void gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes);
        template<typename S1, typename S2, typename IT> void gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes, MaskArg mask);
        template<typename S1, typename IT1, typename IT2> void gather(const S1 *array, const EntryType *const S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes);
        template<typename S1, typename IT1, typename IT2> void gather(const S1 *array, const EntryType *const S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes, MaskArg mask);

        ///////////////////////////////////////////////////////////////////////////////////////////
        // scatters
        template<typename Index> void scatter(EntryType *mem, Index indexes) const;
        template<typename Index> void scatter(EntryType *mem, Index indexes, MaskArg mask) const;
        template<typename S1, typename IT> void scatter(S1 *array, EntryType S1::* member1, IT indexes) const;
        template<typename S1, typename IT> void scatter(S1 *array, EntryType S1::* member1, IT indexes, MaskArg mask) const;
        template<typename S1, typename S2, typename IT> void scatter(S1 *array, S2 S1::* member1, EntryType S2::* member2, IT indexes) const;
        template<typename S1, typename S2, typename IT> void scatter(S1 *array, S2 S1::* member1, EntryType S2::* member2, IT indexes, MaskArg mask) const;
        template<typename S1, typename IT1, typename IT2> void scatter(S1 *array, EntryType *S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes) const;
        template<typename S1, typename IT1, typename IT2> void scatter(S1 *array, EntryType *S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes, MaskArg mask) const;

        ///////////////////////////////////////////////////////////////////////////////////////////
        //prefix
        inline Vector ALWAYS_INLINE &operator++() { data() = VectorHelper<T>::add(data(), VectorHelper<T>::one()); return *this; }
        //postfix
        inline Vector ALWAYS_INLINE operator++(int) { const Vector<T> r = *this; data() = VectorHelper<T>::add(data(), VectorHelper<T>::one()); return r; }

        inline Common::AliasingEntryHelper<StorageType> INTRINSIC operator[](int index) {
#if defined(VC_GCC) && VC_GCC >= 0x40300 && VC_GCC < 0x40400
            ::Vc::Warnings::_operator_bracket_warning();
#endif
            return d.m(index);
        }
        inline EntryType ALWAYS_INLINE operator[](int index) const {
            return d.m(index);
        }

        inline Vector ALWAYS_INLINE operator~() const { return VectorHelper<VectorType>::andnot_(data(), VectorHelper<VectorType>::allone()); }
        inline Vector<typename NegateTypeHelper<T>::Type> operator-() const;

#define OP1(fun) \
        inline Vector fun() const { return Vector<T>(VectorHelper<T>::fun(data())); } \
        inline Vector &fun##_eq() { data() = VectorHelper<T>::fun(data()); return *this; }
        OP1(sqrt)
        OP1(abs)
#undef OP1

#define OP(symbol, fun) \
        inline Vector ALWAYS_INLINE &operator symbol##=(const Vector<T> &x) { data() = VectorHelper<T>::fun(data(), x.data()); return *this; } \
        inline Vector ALWAYS_INLINE &operator symbol##=(EntryType x) { return operator symbol##=(Vector(x)); } \
        inline Vector ALWAYS_INLINE operator symbol(const Vector<T> &x) const { return Vector<T>(VectorHelper<T>::fun(data(), x.data())); } \
        template<typename TT> inline VC_EXACT_TYPE(TT, EntryType, Vector) ALWAYS_INLINE operator symbol(TT x) const { return operator symbol(Vector(x)); }

        OP(+, add)
        OP(-, sub)
        OP(*, mul)
#undef OP
        inline Vector &operator/=(EntryType x);
        template<typename TT> inline PURE_L VC_EXACT_TYPE(TT, EntryType, Vector) operator/(TT x) const PURE_R;
        inline Vector &operator/=(const Vector<T> &x);
        inline Vector  operator/ (const Vector<T> &x) const;

        // bitwise ops
#define OP_VEC(op) \
        inline Vector<T> ALWAYS_INLINE_L &operator op##=(AsArg x) ALWAYS_INLINE_R; \
        inline Vector<T> ALWAYS_INLINE_L  operator op   (AsArg x) const ALWAYS_INLINE_R;
#define OP_ENTRY(op) \
        inline ALWAYS_INLINE Vector<T> &operator op##=(EntryType x) { return operator op##=(Vector(x)); } \
        template<typename TT> inline ALWAYS_INLINE VC_EXACT_TYPE(TT, EntryType, Vector) operator op(TT x) const { return operator op(Vector(x)); }
        VC_ALL_BINARY(OP_VEC)
        VC_ALL_BINARY(OP_ENTRY)
        VC_ALL_SHIFTS(OP_VEC)
#undef OP_VEC
#undef OP_ENTRY

        inline Vector<T> ALWAYS_INLINE_L &operator>>=(int x) ALWAYS_INLINE_R;
        inline Vector<T> ALWAYS_INLINE_L &operator<<=(int x) ALWAYS_INLINE_R;
        inline Vector<T> ALWAYS_INLINE_L operator>>(int x) const ALWAYS_INLINE_R;
        inline Vector<T> ALWAYS_INLINE_L operator<<(int x) const ALWAYS_INLINE_R;

#define OPcmp(symbol, fun) \
        inline Mask ALWAYS_INLINE operator symbol(AsArg x) const { return VectorHelper<T>::fun(data(), x.data()); } \
        template<typename TT> inline VC_EXACT_TYPE(TT, EntryType, Mask) ALWAYS_INLINE operator symbol(TT x) const { return operator symbol(Vector(x)); }

        OPcmp(==, cmpeq)
        OPcmp(!=, cmpneq)
        OPcmp(>=, cmpnlt)
        OPcmp(>, cmpnle)
        OPcmp(<, cmplt)
        OPcmp(<=, cmple)
#undef OPcmp

        inline void multiplyAndAdd(const Vector<T> &factor, const Vector<T> &summand) {
            VectorHelper<T>::multiplyAndAdd(data(), factor, summand);
        }

        inline void assign( const Vector<T> &v, const Mask &mask ) {
            const VectorType k = avx_cast<VectorType>(mask.data());
            data() = VectorHelper<VectorType>::blend(data(), v.data(), k);
        }

        template<typename V2> inline V2 staticCast() const { return V2(*this); }
        template<typename V2> inline V2 reinterpretCast() const { return avx_cast<typename V2::VectorType>(data()); }

        inline WriteMaskedVector<T> ALWAYS_INLINE operator()(const Mask &k) { return WriteMaskedVector<T>(this, k); }

        /**
         * \return \p true  This vector was completely filled. m2 might be 0 or != 0. You still have
         *                  to test this.
         *         \p false This vector was not completely filled. m2 is all 0.
         */
        //inline bool pack(Mask &m1, Vector<T> &v2, Mask &m2) {
            //return VectorHelper<T>::pack(data(), m1.data, v2.data(), m2.data);
        //}

        inline VectorType &data() { return d.v(); }
        inline const VectorType &data() const { return d.v(); }

        inline EntryType min() const { return VectorHelper<T>::min(data()); }
        inline EntryType max() const { return VectorHelper<T>::max(data()); }
        inline EntryType product() const { return VectorHelper<T>::mul(data()); }
        inline EntryType sum() const { return VectorHelper<T>::add(data()); }
        inline EntryType min(MaskArg m) const;
        inline EntryType max(MaskArg m) const;
        inline EntryType product(MaskArg m) const;
        inline EntryType sum(MaskArg m) const;

        inline Vector sorted() const { return SortHelper<T>::sort(data()); }

        template<typename F> void callWithValuesSorted(F &f) {
            EntryType value = d.m(0);
            f(value);
            for (int i = 1; i < Size; ++i) {
                if (d.m(i) != value) {
                    value = d.m(i);
                    f(value);
                }
            }
        }

        template<typename F> inline void INTRINSIC call(const F &f) const {
            for_all_vector_entries(i,
                    f(EntryType(d.m(i)));
                    );
        }
        template<typename F> inline void INTRINSIC call(F &f) const {
            for_all_vector_entries(i,
                    f(EntryType(d.m(i)));
                    );
        }

        template<typename F> inline void INTRINSIC call(const F &f, const Mask &mask) const {
            Vc_foreach_bit(size_t i, mask) {
                f(EntryType(d.m(i)));
            }
        }
        template<typename F> inline void INTRINSIC call(F &f, const Mask &mask) const {
            Vc_foreach_bit(size_t i, mask) {
                f(EntryType(d.m(i)));
            }
        }

        template<typename F> inline Vector<T> INTRINSIC apply(const F &f) const {
            Vector<T> r;
            for_all_vector_entries(i,
                    r.d.m(i) = f(EntryType(d.m(i)));
                    );
            return r;
        }
        template<typename F> inline Vector<T> INTRINSIC apply(F &f) const {
            Vector<T> r;
            for_all_vector_entries(i,
                    r.d.m(i) = f(EntryType(d.m(i)));
                    );
            return r;
        }

        template<typename F> inline Vector<T> INTRINSIC apply(const F &f, const Mask &mask) const {
            Vector<T> r(*this);
            Vc_foreach_bit (size_t i, mask) {
                r.d.m(i) = f(EntryType(r.d.m(i)));
            }
            return r;
        }
        template<typename F> inline Vector<T> INTRINSIC apply(F &f, const Mask &mask) const {
            Vector<T> r(*this);
            Vc_foreach_bit (size_t i, mask) {
                r.d.m(i) = f(EntryType(r.d.m(i)));
            }
            return r;
        }

        template<typename IndexT> inline void INTRINSIC fill(EntryType (&f)(IndexT)) {
            for_all_vector_entries(i,
                    d.m(i) = f(i);
                    );
        }
        inline void INTRINSIC fill(EntryType (&f)()) {
            for_all_vector_entries(i,
                    d.m(i) = f();
                    );
        }

        inline INTRINSIC_L Vector copySign(AsArg reference) const INTRINSIC_R;
        inline INTRINSIC_L Vector exponent() const INTRINSIC_R;
};

typedef Vector<double>         double_v;
typedef Vector<float>          float_v;
typedef Vector<sfloat>         sfloat_v;
typedef Vector<int>            int_v;
typedef Vector<unsigned int>   uint_v;
typedef Vector<short>          short_v;
typedef Vector<unsigned short> ushort_v;
typedef double_v::Mask double_m;
typedef  float_v::Mask float_m;
typedef sfloat_v::Mask sfloat_m;
typedef    int_v::Mask int_m;
typedef   uint_v::Mask uint_m;
typedef  short_v::Mask short_m;
typedef ushort_v::Mask ushort_m;

template<typename T> class SwizzledVector : public Vector<T> {};

static inline int_v    min(const int_v    &x, const int_v    &y) { return _mm256_min_epi32(x.data(), y.data()); }
static inline uint_v   min(const uint_v   &x, const uint_v   &y) { return _mm256_min_epu32(x.data(), y.data()); }
static inline short_v  min(const short_v  &x, const short_v  &y) { return _mm_min_epi16(x.data(), y.data()); }
static inline ushort_v min(const ushort_v &x, const ushort_v &y) { return _mm_min_epu16(x.data(), y.data()); }
static inline float_v  min(const float_v  &x, const float_v  &y) { return _mm256_min_ps(x.data(), y.data()); }
static inline sfloat_v min(const sfloat_v &x, const sfloat_v &y) { return _mm256_min_ps(x.data(), y.data()); }
static inline double_v min(const double_v &x, const double_v &y) { return _mm256_min_pd(x.data(), y.data()); }
static inline int_v    max(const int_v    &x, const int_v    &y) { return _mm256_max_epi32(x.data(), y.data()); }
static inline uint_v   max(const uint_v   &x, const uint_v   &y) { return _mm256_max_epu32(x.data(), y.data()); }
static inline short_v  max(const short_v  &x, const short_v  &y) { return _mm_max_epi16(x.data(), y.data()); }
static inline ushort_v max(const ushort_v &x, const ushort_v &y) { return _mm_max_epu16(x.data(), y.data()); }
static inline float_v  max(const float_v  &x, const float_v  &y) { return _mm256_max_ps(x.data(), y.data()); }
static inline sfloat_v max(const sfloat_v &x, const sfloat_v &y) { return _mm256_max_ps(x.data(), y.data()); }
static inline double_v max(const double_v &x, const double_v &y) { return _mm256_max_pd(x.data(), y.data()); }

  template<typename T> static inline Vector<T> sqrt (const Vector<T> &x) { return VectorHelper<T>::sqrt(x.data()); }
  template<typename T> static inline Vector<T> rsqrt(const Vector<T> &x) { return VectorHelper<T>::rsqrt(x.data()); }
  template<typename T> static inline Vector<T> abs  (const Vector<T> &x) { return VectorHelper<T>::abs(x.data()); }
  template<typename T> static inline Vector<T> reciprocal(const Vector<T> &x) { return VectorHelper<T>::reciprocal(x.data()); }
  template<typename T> static inline Vector<T> round(const Vector<T> &x) { return VectorHelper<T>::round(x.data()); }

  template<typename T> static inline typename Vector<T>::Mask isfinite(const Vector<T> &x) { return VectorHelper<T>::isFinite(x.data()); }
  template<typename T> static inline typename Vector<T>::Mask isnan(const Vector<T> &x) { return VectorHelper<T>::isNaN(x.data()); }

#include "forceToRegisters.tcc"
} // namespace AVX
} // namespace Vc

#include "vector.tcc"
#include "math.h"
#include "undomacros.h"

#endif // AVX_VECTOR_H
