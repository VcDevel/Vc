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

#ifndef SSE_VECTOR_H
#define SSE_VECTOR_H

#include "intrinsics.h"
#include "types.h"
#include "vectorhelper.h"
#include "mask.h"
#include "../common/aliasingentryhelper.h"
#include "../common/memoryfwd.h"
#include <algorithm>
#include <cmath>

#include "macros.h"

#ifdef isfinite
#undef isfinite
#endif
#ifdef isnan
#undef isnan
#endif

namespace Vc
{
namespace SSE
{
template<typename T>
class WriteMaskedVector
{
    friend class Vector<T>;
    typedef typename VectorTraits<T>::MaskType Mask;
    typedef typename Vector<T>::EntryType EntryType;
    public:
        FREE_STORE_OPERATORS_ALIGNED(16)
        //prefix
        inline INTRINSIC Vector<T> &operator++() {
            vec->data() = VectorHelper<T>::add(vec->data(),
                    VectorHelper<T>::notMaskedToZero(VectorHelper<T>::one(), mask.data())
                    );
            return *vec;
        }
        inline INTRINSIC Vector<T> &operator--() {
            vec->data() = VectorHelper<T>::sub(vec->data(),
                    VectorHelper<T>::notMaskedToZero(VectorHelper<T>::one(), mask.data())
                    );
            return *vec;
        }
        //postfix
        inline INTRINSIC Vector<T> operator++(int) {
            Vector<T> ret(*vec);
            vec->data() = VectorHelper<T>::add(vec->data(),
                    VectorHelper<T>::notMaskedToZero(VectorHelper<T>::one(), mask.data())
                    );
            return ret;
        }
        inline INTRINSIC Vector<T> operator--(int) {
            Vector<T> ret(*vec);
            vec->data() = VectorHelper<T>::sub(vec->data(),
                    VectorHelper<T>::notMaskedToZero(VectorHelper<T>::one(), mask.data())
                    );
            return ret;
        }

        inline INTRINSIC Vector<T> &operator+=(const Vector<T> &x) {
            vec->data() = VectorHelper<T>::add(vec->data(), VectorHelper<T>::notMaskedToZero(x.data(), mask.data()));
            return *vec;
        }
        inline INTRINSIC Vector<T> &operator-=(const Vector<T> &x) {
            vec->data() = VectorHelper<T>::sub(vec->data(), VectorHelper<T>::notMaskedToZero(x.data(), mask.data()));
            return *vec;
        }
        inline INTRINSIC Vector<T> &operator*=(const Vector<T> &x) {
            vec->data() = VectorHelper<T>::mul(vec->data(), x.data(), mask.data());
            return *vec;
        }
        inline INTRINSIC CONST Vector<T> &operator/=(const Vector<T> &x);

        inline INTRINSIC Vector<T> &operator+=(EntryType x) {
            return operator+=(Vector<T>(x));
        }
        inline INTRINSIC Vector<T> &operator-=(EntryType x) {
            return operator-=(Vector<T>(x));
        }
        inline INTRINSIC Vector<T> &operator*=(EntryType x) {
            return operator*=(Vector<T>(x));
        }
        inline INTRINSIC Vector<T> &operator/=(EntryType x) {
            return operator/=(Vector<T>(x));
        }

        inline INTRINSIC Vector<T> &operator=(const Vector<T> &x) {
            vec->assign(x, mask);
            return *vec;
        }

        inline INTRINSIC Vector<T> &operator=(EntryType x) {
            vec->assign(Vector<T>(x), mask);
            return *vec;
        }

        template<typename F> inline void INTRINSIC call(const F &f) const {
            return vec->call(f, mask);
        }
        template<typename F> inline void INTRINSIC call(F &f) const {
            return vec->call(f, mask);
        }
        template<typename F> inline Vector<T> INTRINSIC apply(const F &f) const {
            return vec->apply(f, mask);
        }
        template<typename F> inline Vector<T> INTRINSIC apply(F &f) const {
            return vec->apply(f, mask);
        }

    private:
        WriteMaskedVector(Vector<T> *v, const Mask &k) : vec(v), mask(k) {}
        Vector<T> *const vec;
        Mask mask;
};

template<typename T> class Vector
{
    friend class WriteMaskedVector<T>;
    protected:
        typedef typename VectorTraits<T>::StorageType StorageType;
        StorageType d;
        typedef typename VectorTraits<T>::GatherMaskType GatherMask;
        typedef VectorHelper<typename VectorTraits<T>::VectorType> HV;
        typedef VectorHelper<T> HT;
    public:
        FREE_STORE_OPERATORS_ALIGNED(16)

        enum Constants { Size = VectorTraits<T>::Size };
        typedef typename VectorTraits<T>::VectorType VectorType;
        typedef typename VectorTraits<T>::EntryType EntryType;
        typedef typename VectorTraits<T>::IndexType IndexType;
        typedef typename VectorTraits<T>::MaskType Mask;
        typedef typename Mask::Argument MaskArg;
        typedef Vc::Memory<Vector<T>, Size> Memory;
#ifdef VC_PASSING_VECTOR_BY_VALUE_IS_BROKEN
        typedef const Vector<T> &AsArg;
#else
        typedef const Vector<T> AsArg;
#endif

        typedef T _T;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // uninitialized
        inline Vector() {}

        ///////////////////////////////////////////////////////////////////////////////////////////
        // constants
        explicit inline INTRINSIC_L Vector(VectorSpecialInitializerZero::ZEnum) INTRINSIC_R;
        explicit inline INTRINSIC_L Vector(VectorSpecialInitializerOne::OEnum) INTRINSIC_R;
        explicit inline INTRINSIC_L Vector(VectorSpecialInitializerIndexesFromZero::IEnum) INTRINSIC_R;
        static inline INTRINSIC_L Vector Zero() INTRINSIC_R;
        static inline INTRINSIC_L Vector One() INTRINSIC_R;
        static inline INTRINSIC_L Vector IndexesFromZero() INTRINSIC_R;
        static inline INTRINSIC_L Vector Random() INTRINSIC_R;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // internal: required to enable returning objects of VectorType
        inline Vector(const VectorType &x) : d(x) {}

        ///////////////////////////////////////////////////////////////////////////////////////////
        // static_cast / copy ctor
        template<typename OtherT> explicit inline INTRINSIC_L Vector(const Vector<OtherT> &x) INTRINSIC_R;

        // implicit cast
        template<typename OtherT> inline INTRINSIC_L Vector &operator=(const Vector<OtherT> &x) INTRINSIC_R;

        // copy assignment
        inline Vector &operator=(AsArg v) { d.v() = v.d.v(); return *this; }

        ///////////////////////////////////////////////////////////////////////////////////////////
        // broadcast
        explicit Vector(EntryType a);
        template<typename TT> inline INTRINSIC Vector(TT x, VC_EXACT_TYPE(TT, EntryType, void *) = 0) : d(HT::set(x)) {}
        static inline Vector INTRINSIC broadcast4(const EntryType *x) { return Vector<T>(x); }
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
        // expand 1 float_v to 2 double_v                 XXX rationale? remove it for release? XXX
        explicit inline INTRINSIC_L Vector(const Vector<typename CtorTypeHelper<T>::Type> *a) INTRINSIC_R;
        void expand(Vector<typename ExpandTypeHelper<T>::Type> *x) const;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // zeroing
        inline void INTRINSIC_L setZero() INTRINSIC_R;
        inline void INTRINSIC_L setZero(const Mask &k) INTRINSIC_R;

        inline void INTRINSIC_L setQnan() INTRINSIC_R;
        inline void INTRINSIC_L setQnan(typename Mask::Argument k) INTRINSIC_R;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // stores
        inline void INTRINSIC_L store(EntryType *mem) const INTRINSIC_R;
        inline void INTRINSIC_L store(EntryType *mem, const Mask &mask) const INTRINSIC_R;
        template<typename A> inline void INTRINSIC_L store(EntryType *mem, A align) const INTRINSIC_R;
        template<typename A> inline void INTRINSIC_L store(EntryType *mem, const Mask &mask, A align) const INTRINSIC_R;

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
        template<typename IndexT> Vector(const EntryType *mem, const Vector<IndexT> indexes);
        template<typename IndexT> Vector(const EntryType *mem, const IndexT *indexes, MaskArg mask);
        template<typename IndexT> Vector(const EntryType *mem, const Vector<IndexT> indexes, MaskArg mask);
        template<typename S1, typename IT> Vector(const S1 *array, const EntryType S1::* member1, const IT indexes);
        template<typename S1, typename IT> Vector(const S1 *array, const EntryType S1::* member1, const IT indexes, MaskArg mask);
        template<typename S1, typename S2, typename IT> Vector(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, const IT indexes);
        template<typename S1, typename S2, typename IT> Vector(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, const IT indexes, MaskArg mask);
        template<typename S1, typename IT1, typename IT2> Vector(const S1 *array, const EntryType *const S1::* ptrMember1, const IT1 outerIndexes, const IT2 innerIndexes);
        template<typename S1, typename IT1, typename IT2> Vector(const S1 *array, const EntryType *const S1::* ptrMember1, const IT1 outerIndexes, const IT2 innerIndexes, MaskArg mask);
        template<typename Index> void gather(const EntryType *mem, const Index indexes);
        template<typename Index> void gather(const EntryType *mem, const Index indexes, MaskArg mask);
#ifdef VC_USE_SET_GATHERS
        template<typename IT> void gather(const EntryType *mem, Vector<IT> indexes, MaskArg mask);
#endif
        template<typename S1, typename IT> void gather(const S1 *array, const EntryType S1::* member1, const IT indexes);
        template<typename S1, typename IT> void gather(const S1 *array, const EntryType S1::* member1, const IT indexes, MaskArg mask);
        template<typename S1, typename S2, typename IT> void gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, const IT indexes);
        template<typename S1, typename S2, typename IT> void gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, const IT indexes, MaskArg mask);
        template<typename S1, typename IT1, typename IT2> void gather(const S1 *array, const EntryType *const S1::* ptrMember1, const IT1 outerIndexes, const IT2 innerIndexes);
        template<typename S1, typename IT1, typename IT2> void gather(const S1 *array, const EntryType *const S1::* ptrMember1, const IT1 outerIndexes, const IT2 innerIndexes, MaskArg mask);

        ///////////////////////////////////////////////////////////////////////////////////////////
        // scatters
        template<typename Index> void scatter(EntryType *mem, const Index indexes) const;
        template<typename Index> void scatter(EntryType *mem, const Index indexes, MaskArg mask) const;
        template<typename S1, typename IT> void scatter(S1 *array, EntryType S1::* member1, const IT indexes) const;
        template<typename S1, typename IT> void scatter(S1 *array, EntryType S1::* member1, const IT indexes, MaskArg mask) const;
        template<typename S1, typename S2, typename IT> void scatter(S1 *array, S2 S1::* member1, EntryType S2::* member2, const IT indexes) const;
        template<typename S1, typename S2, typename IT> void scatter(S1 *array, S2 S1::* member1, EntryType S2::* member2, const IT indexes, MaskArg mask) const;
        template<typename S1, typename IT1, typename IT2> void scatter(S1 *array, EntryType *S1::* ptrMember1, const IT1 outerIndexes, const IT2 innerIndexes) const;
        template<typename S1, typename IT1, typename IT2> void scatter(S1 *array, EntryType *S1::* ptrMember1, const IT1 outerIndexes, const IT2 innerIndexes, MaskArg mask) const;

        //prefix
        inline Vector INTRINSIC &operator++() { data() = VectorHelper<T>::add(data(), VectorHelper<T>::one()); return *this; }
        //postfix
        inline Vector INTRINSIC operator++(int) { const Vector<T> r = *this; data() = VectorHelper<T>::add(data(), VectorHelper<T>::one()); return r; }

        inline Common::AliasingEntryHelper<StorageType> INTRINSIC operator[](size_t index) {
#if defined(VC_GCC) && VC_GCC >= 0x40300 && VC_GCC < 0x40400
            ::Vc::Warnings::_operator_bracket_warning();
#endif
            return d.m(index);
        }
        inline EntryType INTRINSIC_L operator[](size_t index) const PURE INTRINSIC_R;

        inline Vector PURE INTRINSIC operator~() const { return VectorHelper<VectorType>::andnot_(data(), VectorHelper<VectorType>::allone()); }
        inline Vector<typename NegateTypeHelper<T>::Type> operator-() const;

#define OP(symbol, fun) \
        inline Vector INTRINSIC &operator symbol##=(const Vector<T> &x) { data() = VectorHelper<T>::fun(data(), x.data()); return *this; } \
        inline Vector INTRINSIC &operator symbol##=(EntryType x) { return operator symbol##=(Vector<T>(x)); } \
        inline Vector PURE INTRINSIC operator symbol(const Vector<T> &x) const { return HT::fun(data(), x.data()); } \
        template<typename TT> inline VC_EXACT_TYPE(TT, EntryType, Vector) PURE INTRINSIC operator symbol(TT x) const { return operator symbol(Vector(x)); }

        OP(+, add)
        OP(-, sub)
        OP(*, mul)
#undef OP

        inline INTRINSIC_L Vector &operator<<=(AsArg shift)       INTRINSIC_R;
        inline INTRINSIC_L Vector  operator<< (AsArg shift) const INTRINSIC_R;
        inline INTRINSIC_L Vector &operator<<=(  int shift)       INTRINSIC_R;
        inline INTRINSIC_L Vector  operator<< (  int shift) const INTRINSIC_R;
        inline INTRINSIC_L Vector &operator>>=(AsArg shift)       INTRINSIC_R;
        inline INTRINSIC_L Vector  operator>> (AsArg shift) const INTRINSIC_R;
        inline INTRINSIC_L Vector &operator>>=(  int shift)       INTRINSIC_R;
        inline INTRINSIC_L Vector  operator>> (  int shift) const INTRINSIC_R;

        inline INTRINSIC_L Vector &operator/=(const Vector<T> &x) INTRINSIC_R;
        inline INTRINSIC_L Vector  operator/ (const Vector<T> &x) const PURE INTRINSIC_R;
        inline INTRINSIC_L Vector &operator/=(EntryType x) INTRINSIC_R;
        template<typename TT> inline INTRINSIC_L VC_EXACT_TYPE(TT, typename DetermineEntryType<T>::Type, Vector<T>) operator/(TT x) const PURE INTRINSIC_R;

#define OP(symbol, fun) \
        inline Vector INTRINSIC_L &operator symbol##=(const Vector<T> &x) INTRINSIC_R; \
        inline Vector INTRINSIC_L operator symbol(const Vector<T> &x) const PURE INTRINSIC_R; \
        inline Vector INTRINSIC &operator symbol##=(EntryType x) { return operator symbol##=(Vector(x)); } \
        template<typename TT> inline VC_EXACT_TYPE(TT, EntryType, Vector) PURE INTRINSIC operator symbol(TT x) const { return operator symbol(Vector(x)); }
        OP(|, or_)
        OP(&, and_)
        OP(^, xor_)
#undef OP
#define OPcmp(symbol, fun) \
        inline Mask PURE INTRINSIC operator symbol(const Vector<T> &x) const { return VectorHelper<T>::fun(data(), x.data()); } \
        template<typename TT> inline VC_EXACT_TYPE(TT, EntryType, Mask) PURE INTRINSIC operator symbol(TT x) const { return operator symbol(Vector(x)); }

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
            const VectorType k = mm128_reinterpret_cast<VectorType>(mask.data());
            data() = VectorHelper<VectorType>::blend(data(), v.data(), k);
        }

        template<typename V2> inline V2 staticCast() const { return StaticCastHelper<T, typename V2::_T>::cast(data()); }
        template<typename V2> inline V2 reinterpretCast() const { return mm128_reinterpret_cast<typename V2::VectorType>(data()); }

        inline WriteMaskedVector<T> INTRINSIC operator()(const Mask &k) { return WriteMaskedVector<T>(this, k); }

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

        inline EntryType INTRINSIC min() const { return VectorHelper<T>::min(data()); }
        inline EntryType INTRINSIC max() const { return VectorHelper<T>::max(data()); }
        inline EntryType INTRINSIC product() const { return VectorHelper<T>::mul(data()); }
        inline EntryType INTRINSIC sum() const { return VectorHelper<T>::add(data()); }
        inline INTRINSIC_L EntryType min(MaskArg m) const INTRINSIC_R;
        inline INTRINSIC_L EntryType max(MaskArg m) const INTRINSIC_R;
        inline INTRINSIC_L EntryType product(MaskArg m) const INTRINSIC_R;
        inline INTRINSIC_L EntryType sum(MaskArg m) const INTRINSIC_R;

        inline Vector sorted() const { return SortHelper<VectorType, Size>::sort(data()); }

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

        inline INTRINSIC_L Vector copySign(typename Vector::AsArg reference) const INTRINSIC_R;
        inline INTRINSIC_L Vector exponent() const INTRINSIC_R;
};

typedef Vector<double>         double_v;
typedef Vector<float>          float_v;
typedef Vector<float8>         sfloat_v;
typedef Vector<int>            int_v;
typedef Vector<unsigned int>   uint_v;
typedef Vector<short>          short_v;
typedef Vector<unsigned short> ushort_v;
typedef double_v::Mask double_m;
typedef float_v::Mask float_m;
typedef sfloat_v::Mask sfloat_m;
typedef int_v::Mask int_m;
typedef uint_v::Mask uint_m;
typedef short_v::Mask short_m;
typedef ushort_v::Mask ushort_m;

template<> inline Vector<float8> Vector<float8>::broadcast4(const float *x) {
    const _M128 &v = VectorHelper<_M128>::load(x, Aligned);
    return Vector<float8>(M256::create(v, v));
}

template<typename T> class SwizzledVector : public Vector<T> {};

static inline int_v    min(const int_v    &x, const int_v    &y) { return _mm_min_epi32(x.data(), y.data()); }
static inline uint_v   min(const uint_v   &x, const uint_v   &y) { return _mm_min_epu32(x.data(), y.data()); }
static inline short_v  min(const short_v  &x, const short_v  &y) { return _mm_min_epi16(x.data(), y.data()); }
static inline ushort_v min(const ushort_v &x, const ushort_v &y) { return _mm_min_epu16(x.data(), y.data()); }
static inline float_v  min(const float_v  &x, const float_v  &y) { return _mm_min_ps(x.data(), y.data()); }
static inline double_v min(const double_v &x, const double_v &y) { return _mm_min_pd(x.data(), y.data()); }
static inline int_v    max(const int_v    &x, const int_v    &y) { return _mm_max_epi32(x.data(), y.data()); }
static inline uint_v   max(const uint_v   &x, const uint_v   &y) { return _mm_max_epu32(x.data(), y.data()); }
static inline short_v  max(const short_v  &x, const short_v  &y) { return _mm_max_epi16(x.data(), y.data()); }
static inline ushort_v max(const ushort_v &x, const ushort_v &y) { return _mm_max_epu16(x.data(), y.data()); }
static inline float_v  max(const float_v  &x, const float_v  &y) { return _mm_max_ps(x.data(), y.data()); }
static inline double_v max(const double_v &x, const double_v &y) { return _mm_max_pd(x.data(), y.data()); }

static inline sfloat_v min(const sfloat_v &x, const sfloat_v &y) {
    return M256::create(_mm_min_ps(x.data()[0], y.data()[0]), _mm_min_ps(x.data()[1], y.data()[1]));
}
static inline sfloat_v max(const sfloat_v &x, const sfloat_v &y) {
    return M256::create(_mm_max_ps(x.data()[0], y.data()[0]), _mm_max_ps(x.data()[1], y.data()[1]));
}

  template<typename T> static inline Vector<T> sqrt (const Vector<T> &x) { return VectorHelper<T>::sqrt(x.data()); }
  template<typename T> static inline Vector<T> rsqrt(const Vector<T> &x) { return VectorHelper<T>::rsqrt(x.data()); }
  template<typename T> static inline Vector<T> abs  (const Vector<T> &x) { return VectorHelper<T>::abs(x.data()); }
  template<typename T> static inline Vector<T> reciprocal(const Vector<T> &x) { return VectorHelper<T>::reciprocal(x.data()); }
  template<typename T> static inline Vector<T> round(const Vector<T> &x) { return VectorHelper<T>::round(x.data()); }

  template<typename T> static inline typename Vector<T>::Mask isfinite(const Vector<T> &x) { return VectorHelper<T>::isFinite(x.data()); }
  template<typename T> static inline typename Vector<T>::Mask isnan(const Vector<T> &x) { return VectorHelper<T>::isNaN(x.data()); }

#include "forceToRegisters.tcc"
#ifdef VC_GNU_ASM
template<>
inline void ALWAYS_INLINE forceToRegisters(const Vector<float8> &x1) {
  __asm__ __volatile__(""::"x"(x1.data()[0]), "x"(x1.data()[1]));
}
#elif defined(VC_MSVC)
#pragma optimize("g", off)
template<>
inline void ALWAYS_INLINE forceToRegisters(const Vector<float8> &/*x1*/) {
}
#endif
} // namespace SSE
} // namespace Vc

#include "undomacros.h"
#include "vector.tcc"
#include "math.h"
#endif // SSE_VECTOR_H
