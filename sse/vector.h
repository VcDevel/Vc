/*  This file is part of the Vc library.

    Copyright (C) 2009 Matthias Kretz <kretz@kde.org>

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
#include "vectorbase.h"
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
    typedef typename VectorBase<T>::MaskType Mask;
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
        inline INTRINSIC Vector<T> &operator/=(const Vector<T> &x) {
            vec->data() = VectorHelper<T>::div(vec->data(), x.data(), mask.data());
            return *vec;
        }

        inline INTRINSIC Vector<T> &operator=(const Vector<T> &x) {
            vec->assign(x, mask);
            return *vec;
        }
    private:
        WriteMaskedVector(Vector<T> *v, const Mask &k) : vec(v), mask(k) {}
        Vector<T> *vec;
        Mask mask;
};

template<typename T>
class Vector : public VectorBase<T>
{
    protected:
        typedef VectorBase<T> Base;
        using Base::d;
        typedef typename Base::GatherMaskType GatherMask;
        typedef VectorHelper<typename Base::VectorType> HV;
        typedef VectorHelper<T> HT;
    public:
        FREE_STORE_OPERATORS_ALIGNED(16)

        enum { Size = Base::Size };
        typedef typename Base::VectorType VectorType;
        typedef typename Base::EntryType  EntryType;
        typedef Vector<typename IndexTypeHelper<Size>::Type> IndexType;
        typedef typename Base::MaskType Mask;
	typedef typename Mask::Argument MaskArg;
        typedef Vc::Memory<Vector<T>, Size> Memory;

	typedef typename Base::AsArg AsArg;

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
        static inline INTRINSIC_L Vector IndexesFromZero() INTRINSIC_R;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // internal: required to enable returning objects of VectorType
        inline Vector(const VectorType &x) : Base(x) {}

        ///////////////////////////////////////////////////////////////////////////////////////////
        // static_cast / copy ctor
        template<typename OtherT> explicit inline INTRINSIC_L Vector(const Vector<OtherT> &x) INTRINSIC_R;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // broadcast
        Vector(EntryType a);
        static inline Vector INTRINSIC broadcast4(const EntryType *x) { return Vector<T>(x); }

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

        ///////////////////////////////////////////////////////////////////////////////////////////
        // stores
        inline void INTRINSIC_L store(EntryType *mem) const INTRINSIC_R;
        inline void INTRINSIC_L store(EntryType *mem, const Mask &mask) const INTRINSIC_R;
        template<typename A> inline void INTRINSIC_L store(EntryType *mem, A align) const INTRINSIC_R;
        template<typename A> inline void INTRINSIC_L store(EntryType *mem, const Mask &mask, A align) const INTRINSIC_R;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // swizzles
        inline const Vector<T> INTRINSIC &dcba() const { return *this; }
        inline const Vector<T> INTRINSIC cdab() const { return reinterpret_cast<VectorType>(_mm_shuffle_epi32(data(), _MM_SHUFFLE(2, 3, 0, 1))); }
        inline const Vector<T> INTRINSIC badc() const { return reinterpret_cast<VectorType>(_mm_shuffle_epi32(data(), _MM_SHUFFLE(1, 0, 3, 2))); }
        inline const Vector<T> INTRINSIC aaaa() const { return reinterpret_cast<VectorType>(_mm_shuffle_epi32(data(), _MM_SHUFFLE(0, 0, 0, 0))); }
        inline const Vector<T> INTRINSIC bbbb() const { return reinterpret_cast<VectorType>(_mm_shuffle_epi32(data(), _MM_SHUFFLE(1, 1, 1, 1))); }
        inline const Vector<T> INTRINSIC cccc() const { return reinterpret_cast<VectorType>(_mm_shuffle_epi32(data(), _MM_SHUFFLE(2, 2, 2, 2))); }
        inline const Vector<T> INTRINSIC dddd() const { return reinterpret_cast<VectorType>(_mm_shuffle_epi32(data(), _MM_SHUFFLE(3, 3, 3, 3))); }
        inline const Vector<T> INTRINSIC dacb() const { return reinterpret_cast<VectorType>(_mm_shuffle_epi32(data(), _MM_SHUFFLE(3, 0, 2, 1))); }

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

        inline Common::AliasingEntryHelper<EntryType> INTRINSIC operator[](size_t index) {
#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ == 3
            ::Vc::Warnings::_operator_bracket_warning();
#endif
            return Base::d.m(index);
        }
        inline EntryType INTRINSIC_L operator[](size_t index) const PURE INTRINSIC_R;

        inline Vector PURE INTRINSIC operator~() const { return VectorHelper<VectorType>::andnot_(data(), VectorHelper<VectorType>::allone()); }
        inline Vector<typename NegateTypeHelper<T>::Type> operator-() const;

#define OP(symbol, fun) \
        inline Vector INTRINSIC &operator symbol##=(const Vector<T> &x) { data() = VectorHelper<T>::fun(data(), x.data()); return *this; } \
        inline Vector PURE INTRINSIC operator symbol(const Vector<T> &x) const { return Vector<T>(VectorHelper<T>::fun(data(), x.data())); }

        OP(+, add)
        OP(-, sub)
        OP(*, mul)
#undef OP

        inline INTRINSIC_L Vector &operator/=(const Vector<T> &x) INTRINSIC_R;
        inline INTRINSIC_L Vector  operator/ (const Vector<T> &x) const PURE INTRINSIC_R;
        inline INTRINSIC_L Vector &operator/=(EntryType x) INTRINSIC_R;
        inline INTRINSIC_L Vector  operator/ (EntryType x) const PURE INTRINSIC_R;

#define OP(symbol, fun) \
        inline Vector INTRINSIC &operator symbol##=(const Vector<T> &x) { data() = VectorHelper<VectorType>::fun(data(), x.data()); return *this; } \
        inline Vector PURE INTRINSIC operator symbol(const Vector<T> &x) const { return Vector<T>(VectorHelper<VectorType>::fun(data(), x.data())); }
        OP(|, or_)
        OP(&, and_)
        OP(^, xor_)
#undef OP
#define OPcmp(symbol, fun) \
        inline Mask PURE INTRINSIC operator symbol(const Vector<T> &x) const { return VectorHelper<T>::fun(data(), x.data()); }

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

        inline VectorType &data() { return Base::data(); }
        inline const VectorType &data() const { return Base::data(); }

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
            EntryType value = Base::d.m(0);
            f(value);
            for (int i = 1; i < Size; ++i) {
                if (Base::d.m(i) != value) {
                    value = Base::d.m(i);
                    f(value);
                }
            }
        }
};

typedef Vector<double>         double_v;
typedef Vector<float>          float_v;
typedef Vector<float8>         sfloat_v;
typedef Vector<int>            int_v;
typedef Vector<unsigned int>   uint_v;
typedef Vector<short>          short_v;
typedef Vector<unsigned short> ushort_v;

template<> inline Vector<float8> Vector<float8>::broadcast4(const float *x) {
    const _M128 &v = VectorHelper<_M128>::load(x, Aligned);
    return Vector<float8>(M256::create(v, v));
}

template<typename T> class SwizzledVector : public Vector<T> {};

template<typename T> inline Vector<T> INTRINSIC operator+(const typename Vector<T>::EntryType &x, const Vector<T> &v) { return v.operator+(x); }
template<typename T> inline Vector<T> INTRINSIC operator*(const typename Vector<T>::EntryType &x, const Vector<T> &v) { return v.operator*(x); }
template<typename T> inline Vector<T> INTRINSIC operator-(const typename Vector<T>::EntryType &x, const Vector<T> &v) { return Vector<T>(x) - v; }
template<typename T> inline Vector<T> INTRINSIC operator/(const typename Vector<T>::EntryType &x, const Vector<T> &v) { return Vector<T>(x) / v; }
template<typename T> inline typename Vector<T>::Mask INTRINSIC operator< (const typename Vector<T>::EntryType &x, const Vector<T> &v) { return Vector<T>(x) <  v; }
template<typename T> inline typename Vector<T>::Mask INTRINSIC operator<=(const typename Vector<T>::EntryType &x, const Vector<T> &v) { return Vector<T>(x) <= v; }
template<typename T> inline typename Vector<T>::Mask INTRINSIC operator> (const typename Vector<T>::EntryType &x, const Vector<T> &v) { return Vector<T>(x) >  v; }
template<typename T> inline typename Vector<T>::Mask INTRINSIC operator>=(const typename Vector<T>::EntryType &x, const Vector<T> &v) { return Vector<T>(x) >= v; }
template<typename T> inline typename Vector<T>::Mask INTRINSIC operator==(const typename Vector<T>::EntryType &x, const Vector<T> &v) { return Vector<T>(x) == v; }
template<typename T> inline typename Vector<T>::Mask INTRINSIC operator!=(const typename Vector<T>::EntryType &x, const Vector<T> &v) { return Vector<T>(x) != v; }

  template<typename T> static inline Vector<T> min  (const Vector<T> &x, const Vector<T> &y) { return VectorHelper<T>::min(x.data(), y.data()); }
  template<typename T> static inline Vector<T> max  (const Vector<T> &x, const Vector<T> &y) { return VectorHelper<T>::max(x.data(), y.data()); }
  template<typename T> static inline Vector<T> min  (const Vector<T> &x, const typename Vector<T>::EntryType &y) { return min(x.data(), Vector<T>(y).data()); }
  template<typename T> static inline Vector<T> max  (const Vector<T> &x, const typename Vector<T>::EntryType &y) { return max(x.data(), Vector<T>(y).data()); }
  template<typename T> static inline Vector<T> min  (const typename Vector<T>::EntryType &x, const Vector<T> &y) { return min(Vector<T>(x).data(), y.data()); }
  template<typename T> static inline Vector<T> max  (const typename Vector<T>::EntryType &x, const Vector<T> &y) { return max(Vector<T>(x).data(), y.data()); }
  template<typename T> static inline Vector<T> sqrt (const Vector<T> &x) { return VectorHelper<T>::sqrt(x.data()); }
  template<typename T> static inline Vector<T> rsqrt(const Vector<T> &x) { return VectorHelper<T>::rsqrt(x.data()); }
  template<typename T> static inline Vector<T> abs  (const Vector<T> &x) { return VectorHelper<T>::abs(x.data()); }
  template<typename T> static inline Vector<T> log  (const Vector<T> &x) { return VectorHelper<T>::log(x.data()); }
  template<typename T> static inline Vector<T> log10(const Vector<T> &x) { return VectorHelper<T>::log10(x.data()); }
  template<typename T> static inline Vector<T> reciprocal(const Vector<T> &x) { return VectorHelper<T>::reciprocal(x.data()); }
  template<typename T> static inline Vector<T> round(const Vector<T> &x) { return VectorHelper<T>::round(x.data()); }

  template<typename T> static inline typename Vector<T>::Mask isfinite(const Vector<T> &x) { return VectorHelper<T>::isFinite(x.data()); }
  template<typename T> static inline typename Vector<T>::Mask isnan(const Vector<T> &x) { return VectorHelper<T>::isNaN(x.data()); }

#include "forceToRegisters.tcc"
#ifdef __GNUC__
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

#undef STORE_VECTOR
} // namespace SSE
} // namespace Vc

#include "math.h"
#include "undomacros.h"
#include "vector.tcc"
#endif // SSE_VECTOR_H
