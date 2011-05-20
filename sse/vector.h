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
    template<typename V, unsigned int Size> class Memory;

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
        inline Vector<T> &operator++() INTRINSIC {
            vec->data() = VectorHelper<T>::add(vec->data(),
                    VectorHelper<T>::notMaskedToZero(VectorHelper<T>::one(), mask.data())
                    );
            return *vec;
        }
        inline Vector<T> &operator--() INTRINSIC {
            vec->data() = VectorHelper<T>::sub(vec->data(),
                    VectorHelper<T>::notMaskedToZero(VectorHelper<T>::one(), mask.data())
                    );
            return *vec;
        }
        //postfix
        inline Vector<T> operator++(int) INTRINSIC {
            Vector<T> ret(*vec);
            vec->data() = VectorHelper<T>::add(vec->data(),
                    VectorHelper<T>::notMaskedToZero(VectorHelper<T>::one(), mask.data())
                    );
            return ret;
        }
        inline Vector<T> operator--(int) INTRINSIC {
            Vector<T> ret(*vec);
            vec->data() = VectorHelper<T>::sub(vec->data(),
                    VectorHelper<T>::notMaskedToZero(VectorHelper<T>::one(), mask.data())
                    );
            return ret;
        }

        inline Vector<T> &operator+=(const Vector<T> &x) INTRINSIC {
            vec->data() = VectorHelper<T>::add(vec->data(), VectorHelper<T>::notMaskedToZero(x.data(), mask.data()));
            return *vec;
        }
        inline Vector<T> &operator-=(const Vector<T> &x) INTRINSIC {
            vec->data() = VectorHelper<T>::sub(vec->data(), VectorHelper<T>::notMaskedToZero(x.data(), mask.data()));
            return *vec;
        }
        inline Vector<T> &operator*=(const Vector<T> &x) INTRINSIC {
            vec->data() = VectorHelper<T>::mul(vec->data(), x.data(), mask.data());
            return *vec;
        }
        inline Vector<T> &operator/=(const Vector<T> &x) INTRINSIC {
            vec->data() = VectorHelper<T>::div(vec->data(), x.data(), mask.data());
            return *vec;
        }

        inline Vector<T> &operator=(const Vector<T> &x) INTRINSIC {
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
        typedef Vc::Memory<Vector<T>, Size> Memory;

        typedef T _T;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // uninitialized
        inline Vector() {}

        ///////////////////////////////////////////////////////////////////////////////////////////
        // constants
        explicit inline Vector(VectorSpecialInitializerZero::ZEnum) INTRINSIC;
        explicit inline Vector(VectorSpecialInitializerOne::OEnum) INTRINSIC;
        explicit inline Vector(VectorSpecialInitializerIndexesFromZero::IEnum) INTRINSIC;
        static inline Vector Zero() INTRINSIC;
        static inline Vector IndexesFromZero() INTRINSIC;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // internal: required to enable returning objects of VectorType
        inline Vector(const VectorType &x) : Base(x) {}

        ///////////////////////////////////////////////////////////////////////////////////////////
        // static_cast / copy ctor
        template<typename OtherT> explicit inline Vector(const Vector<OtherT> &x) INTRINSIC;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // broadcast
        Vector(EntryType a);
        static inline Vector broadcast4(const EntryType *x) INTRINSIC { return Vector<T>(x); }

        ///////////////////////////////////////////////////////////////////////////////////////////
        // load ctors
        explicit inline Vector(const EntryType *x) INTRINSIC;
        template<typename Alignment> inline Vector(const EntryType *x, Alignment align) INTRINSIC;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // expand 1 float_v to 2 double_v                 XXX rationale? remove it for release? XXX
        explicit inline Vector(const Vector<typename CtorTypeHelper<T>::Type> *a) INTRINSIC;
        void expand(Vector<typename ExpandTypeHelper<T>::Type> *x) const;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // load member functions
        inline void load(const EntryType *mem) INTRINSIC;
        template<typename Alignment> inline void load(const EntryType *mem, Alignment align) INTRINSIC;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // zeroing
        inline void setZero() INTRINSIC;
        inline void setZero(const Mask &k) INTRINSIC;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // stores
        inline void store(EntryType *mem) const INTRINSIC;
        inline void store(EntryType *mem, const Mask &mask) const INTRINSIC;
        template<typename A> inline void store(EntryType *mem, A align) const INTRINSIC;
        template<typename A> inline void store(EntryType *mem, const Mask &mask, A align) const INTRINSIC;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // swizzles
        inline const Vector<T> &dcba() const INTRINSIC { return *this; }
        inline const Vector<T> cdab() const INTRINSIC { return reinterpret_cast<VectorType>(_mm_shuffle_epi32(data(), _MM_SHUFFLE(2, 3, 0, 1))); }
        inline const Vector<T> badc() const INTRINSIC { return reinterpret_cast<VectorType>(_mm_shuffle_epi32(data(), _MM_SHUFFLE(1, 0, 3, 2))); }
        inline const Vector<T> aaaa() const INTRINSIC { return reinterpret_cast<VectorType>(_mm_shuffle_epi32(data(), _MM_SHUFFLE(0, 0, 0, 0))); }
        inline const Vector<T> bbbb() const INTRINSIC { return reinterpret_cast<VectorType>(_mm_shuffle_epi32(data(), _MM_SHUFFLE(1, 1, 1, 1))); }
        inline const Vector<T> cccc() const INTRINSIC { return reinterpret_cast<VectorType>(_mm_shuffle_epi32(data(), _MM_SHUFFLE(2, 2, 2, 2))); }
        inline const Vector<T> dddd() const INTRINSIC { return reinterpret_cast<VectorType>(_mm_shuffle_epi32(data(), _MM_SHUFFLE(3, 3, 3, 3))); }
        inline const Vector<T> dacb() const INTRINSIC { return reinterpret_cast<VectorType>(_mm_shuffle_epi32(data(), _MM_SHUFFLE(3, 0, 2, 1))); }

        ///////////////////////////////////////////////////////////////////////////////////////////
        // gathers
        template<typename IndexT> Vector(const EntryType *mem, const IndexT *indexes);
        template<typename IndexT> Vector(const EntryType *mem, Vector<IndexT> indexes);
        template<typename IndexT> Vector(const EntryType *mem, const IndexT *indexes, Mask mask);
        template<typename IndexT> Vector(const EntryType *mem, Vector<IndexT> indexes, Mask mask);
        template<typename S1, typename IT> Vector(const S1 *array, const EntryType S1::* member1, IT indexes);
        template<typename S1, typename IT> Vector(const S1 *array, const EntryType S1::* member1, IT indexes, Mask mask);
        template<typename S1, typename S2, typename IT> Vector(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes);
        template<typename S1, typename S2, typename IT> Vector(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes, Mask mask);
        template<typename S1, typename IT1, typename IT2> Vector(const S1 *array, const EntryType *const S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes);
        template<typename S1, typename IT1, typename IT2> Vector(const S1 *array, const EntryType *const S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes, Mask mask);
        template<typename Index> void gather(const EntryType *mem, Index indexes);
        template<typename Index> void gather(const EntryType *mem, Index indexes, Mask mask);
#ifdef VC_USE_SET_GATHERS
        template<typename IT> void gather(const EntryType *mem, Vector<IT> indexes, Mask mask);
#endif
        template<typename S1, typename IT> void gather(const S1 *array, const EntryType S1::* member1, IT indexes);
        template<typename S1, typename IT> void gather(const S1 *array, const EntryType S1::* member1, IT indexes, Mask mask);
        template<typename S1, typename S2, typename IT> void gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes);
        template<typename S1, typename S2, typename IT> void gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes, Mask mask);
        template<typename S1, typename IT1, typename IT2> void gather(const S1 *array, const EntryType *const S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes);
        template<typename S1, typename IT1, typename IT2> void gather(const S1 *array, const EntryType *const S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes, Mask mask);

        ///////////////////////////////////////////////////////////////////////////////////////////
        // scatters
        template<typename Index> void scatter(EntryType *mem, Index indexes) const;
        template<typename Index> void scatter(EntryType *mem, Index indexes, Mask mask) const;
        template<typename S1, typename IT> void scatter(S1 *array, EntryType S1::* member1, IT indexes) const;
        template<typename S1, typename IT> void scatter(S1 *array, EntryType S1::* member1, IT indexes, Mask mask) const;
        template<typename S1, typename S2, typename IT> void scatter(S1 *array, S2 S1::* member1, EntryType S2::* member2, IT indexes) const;
        template<typename S1, typename S2, typename IT> void scatter(S1 *array, S2 S1::* member1, EntryType S2::* member2, IT indexes, Mask mask) const;
        template<typename S1, typename IT1, typename IT2> void scatter(S1 *array, EntryType *S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes) const;
        template<typename S1, typename IT1, typename IT2> void scatter(S1 *array, EntryType *S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes, Mask mask) const;

        //prefix
        inline Vector &operator++() INTRINSIC { data() = VectorHelper<T>::add(data(), VectorHelper<T>::one()); return *this; }
        //postfix
        inline Vector operator++(int) INTRINSIC { const Vector<T> r = *this; data() = VectorHelper<T>::add(data(), VectorHelper<T>::one()); return r; }

        inline Common::AliasingEntryHelper<EntryType> operator[](int index) INTRINSIC {
#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ == 3
            ::Vc::Warnings::_operator_bracket_warning();
#endif
            return Base::d.m(index);
        }
        inline EntryType operator[](int index) const PURE INTRINSIC {
            return Base::d.m(index);
        }

        inline Vector operator~() const PURE INTRINSIC { return VectorHelper<VectorType>::andnot_(data(), VectorHelper<VectorType>::allone()); }
        inline Vector<typename NegateTypeHelper<T>::Type> operator-() const;

#define OP(symbol, fun) \
        inline Vector &operator symbol##=(const Vector<T> &x) INTRINSIC { data() = VectorHelper<T>::fun(data(), x.data()); return *this; } \
        inline Vector operator symbol(const Vector<T> &x) const PURE INTRINSIC { return Vector<T>(VectorHelper<T>::fun(data(), x.data())); }

        OP(+, add)
        OP(-, sub)
        OP(*, mul)
#undef OP

        inline Vector &operator/=(const Vector<T> &x) INTRINSIC;
        inline Vector  operator/ (const Vector<T> &x) const PURE INTRINSIC;
        inline Vector &operator/=(EntryType x) INTRINSIC;
        inline Vector  operator/ (EntryType x) const PURE INTRINSIC;

#define OP(symbol, fun) \
        inline Vector &operator symbol##=(const Vector<T> &x) INTRINSIC { data() = VectorHelper<VectorType>::fun(data(), x.data()); return *this; } \
        inline Vector operator symbol(const Vector<T> &x) const PURE INTRINSIC { return Vector<T>(VectorHelper<VectorType>::fun(data(), x.data())); }
        OP(|, or_)
        OP(&, and_)
        OP(^, xor_)
#undef OP
#define OPcmp(symbol, fun) \
        inline Mask operator symbol(const Vector<T> &x) const PURE INTRINSIC { return VectorHelper<T>::fun(data(), x.data()); }

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

        inline WriteMaskedVector<T> operator()(const Mask &k) INTRINSIC { return WriteMaskedVector<T>(this, k); }

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

        inline EntryType min() const INTRINSIC { return VectorHelper<T>::min(data()); }
        inline EntryType max() const INTRINSIC { return VectorHelper<T>::max(data()); }
        inline EntryType product() const INTRINSIC { return VectorHelper<T>::mul(data()); }
        inline EntryType sum() const INTRINSIC { return VectorHelper<T>::add(data()); }
        inline EntryType min(Mask m) const INTRINSIC;
        inline EntryType max(Mask m) const INTRINSIC;
        inline EntryType product(Mask m) const INTRINSIC;
        inline EntryType sum(Mask m) const INTRINSIC;

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

template<> inline Vector<float8> Vector<float8>::broadcast4(const float *x) {
    const _M128 &v = VectorHelper<_M128>::load(x, Aligned);
    return Vector<float8>(M256::create(v, v));
}

template<typename T> class SwizzledVector : public Vector<T> {};

template<typename T> inline Vector<T> operator+(const typename Vector<T>::EntryType &x, const Vector<T> &v) INTRINSIC;
template<typename T> inline Vector<T> operator*(const typename Vector<T>::EntryType &x, const Vector<T> &v) INTRINSIC;
template<typename T> inline Vector<T> operator-(const typename Vector<T>::EntryType &x, const Vector<T> &v) INTRINSIC;
template<typename T> inline Vector<T> operator/(const typename Vector<T>::EntryType &x, const Vector<T> &v) INTRINSIC;
template<typename T> inline typename Vector<T>::Mask  operator< (const typename Vector<T>::EntryType &x, const Vector<T> &v) INTRINSIC;
template<typename T> inline typename Vector<T>::Mask  operator<=(const typename Vector<T>::EntryType &x, const Vector<T> &v) INTRINSIC;
template<typename T> inline typename Vector<T>::Mask  operator> (const typename Vector<T>::EntryType &x, const Vector<T> &v) INTRINSIC;
template<typename T> inline typename Vector<T>::Mask  operator>=(const typename Vector<T>::EntryType &x, const Vector<T> &v) INTRINSIC;
template<typename T> inline typename Vector<T>::Mask  operator==(const typename Vector<T>::EntryType &x, const Vector<T> &v) INTRINSIC;
template<typename T> inline typename Vector<T>::Mask  operator!=(const typename Vector<T>::EntryType &x, const Vector<T> &v) INTRINSIC;
template<typename T> inline Vector<T> operator+(const typename Vector<T>::EntryType &x, const Vector<T> &v) { return v.operator+(x); }
template<typename T> inline Vector<T> operator*(const typename Vector<T>::EntryType &x, const Vector<T> &v) { return v.operator*(x); }
template<typename T> inline Vector<T> operator-(const typename Vector<T>::EntryType &x, const Vector<T> &v) { return Vector<T>(x) - v; }
template<typename T> inline Vector<T> operator/(const typename Vector<T>::EntryType &x, const Vector<T> &v) { return Vector<T>(x) / v; }
template<typename T> inline typename Vector<T>::Mask  operator< (const typename Vector<T>::EntryType &x, const Vector<T> &v) { return Vector<T>(x) <  v; }
template<typename T> inline typename Vector<T>::Mask  operator<=(const typename Vector<T>::EntryType &x, const Vector<T> &v) { return Vector<T>(x) <= v; }
template<typename T> inline typename Vector<T>::Mask  operator> (const typename Vector<T>::EntryType &x, const Vector<T> &v) { return Vector<T>(x) >  v; }
template<typename T> inline typename Vector<T>::Mask  operator>=(const typename Vector<T>::EntryType &x, const Vector<T> &v) { return Vector<T>(x) >= v; }
template<typename T> inline typename Vector<T>::Mask  operator==(const typename Vector<T>::EntryType &x, const Vector<T> &v) { return Vector<T>(x) == v; }
template<typename T> inline typename Vector<T>::Mask  operator!=(const typename Vector<T>::EntryType &x, const Vector<T> &v) { return Vector<T>(x) != v; }

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

#ifdef _MSC_VER
#define VC_FTR_EMPTY
#include "forceToRegisters.def"
#undef VC_FTR_EMPTY
#else
#include "forceToRegisters.def"
template<> inline void forceToRegisters(const Vector<float8> &x) { __asm__ __volatile__(""::"x"(x.data()[0]), "x"(x.data()[1])); }
#endif

#undef STORE_VECTOR
} // namespace SSE
} // namespace Vc

#include "math.h"
#include "undomacros.h"
#include "vector.tcc"
#endif // SSE_VECTOR_H
