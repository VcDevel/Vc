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
#include <algorithm>
#include <cmath>

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
    enum { VectorAlignment = 16 };

template<typename T>
class WriteMaskedVector
{
    friend class Vector<T>;
    typedef typename VectorBase<T>::MaskType Mask;
    public:
        FREE_STORE_OPERATORS_ALIGNED(16)
        //prefix
        inline Vector<T> &operator++() ALWAYS_INLINE {
            vec->data() = VectorHelper<T>::add(vec->data(),
                    VectorHelper<T>::notMaskedToZero(VectorHelper<T>::one(), mask.data())
                    );
            return *vec;
        }
        inline Vector<T> &operator--() ALWAYS_INLINE {
            vec->data() = VectorHelper<T>::sub(vec->data(),
                    VectorHelper<T>::notMaskedToZero(VectorHelper<T>::one(), mask.data())
                    );
            return *vec;
        }
        //postfix
        inline Vector<T> operator++(int) ALWAYS_INLINE {
            Vector<T> ret(*vec);
            vec->data() = VectorHelper<T>::add(vec->data(),
                    VectorHelper<T>::notMaskedToZero(VectorHelper<T>::one(), mask.data())
                    );
            return ret;
        }
        inline Vector<T> operator--(int) ALWAYS_INLINE {
            Vector<T> ret(*vec);
            vec->data() = VectorHelper<T>::sub(vec->data(),
                    VectorHelper<T>::notMaskedToZero(VectorHelper<T>::one(), mask.data())
                    );
            return ret;
        }

        inline Vector<T> &operator+=(const Vector<T> &x) ALWAYS_INLINE {
            vec->data() = VectorHelper<T>::add(vec->data(), VectorHelper<T>::notMaskedToZero(x.data(), mask.data()));
            return *vec;
        }
        inline Vector<T> &operator-=(const Vector<T> &x) ALWAYS_INLINE {
            vec->data() = VectorHelper<T>::sub(vec->data(), VectorHelper<T>::notMaskedToZero(x.data(), mask.data()));
            return *vec;
        }
        inline Vector<T> &operator*=(const Vector<T> &x) ALWAYS_INLINE {
            vec->data() = VectorHelper<T>::mul(vec->data(), x.data(), mask.data());
            return *vec;
        }
        inline Vector<T> &operator/=(const Vector<T> &x) ALWAYS_INLINE {
            vec->data() = VectorHelper<T>::div(vec->data(), x.data(), mask.data());
            return *vec;
        }

        inline Vector<T> &operator=(const Vector<T> &x) ALWAYS_INLINE {
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
        typedef typename Base::StorageType::AliasingEntryType AliasingEntryType;
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
        explicit Vector(VectorSpecialInitializerZero::ZEnum);
        explicit Vector(VectorSpecialInitializerOne::OEnum);
        explicit Vector(VectorSpecialInitializerIndexesFromZero::IEnum);
        static Vector Zero();
        static Vector IndexesFromZero();

        ///////////////////////////////////////////////////////////////////////////////////////////
        // internal: required to enable returning objects of VectorType
        inline Vector(const VectorType &x) : Base(x) {}

        ///////////////////////////////////////////////////////////////////////////////////////////
        // static_cast / copy ctor
        template<typename OtherT> explicit Vector(const Vector<OtherT> &x);

        ///////////////////////////////////////////////////////////////////////////////////////////
        // broadcast
        Vector(EntryType a);
        static inline Vector broadcast4(const EntryType *x) { return Vector<T>(x); }

        ///////////////////////////////////////////////////////////////////////////////////////////
        // load ctors
        explicit Vector(const EntryType *x);
        template<typename Alignment> Vector(const EntryType *x, Alignment align);
        explicit Vector(const Vector<typename CtorTypeHelper<T>::Type> *a);

        ///////////////////////////////////////////////////////////////////////////////////////////
        // expand 1 short_v to 2 int_v                    XXX rationale? remove it for release? XXX
        void expand(Vector<typename ExpandTypeHelper<T>::Type> *x) const;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // load member functions
        void load(const EntryType *mem);
        template<typename Alignment> void load(const EntryType *mem, Alignment align);

        ///////////////////////////////////////////////////////////////////////////////////////////
        // zeroing
        void makeZero();
        void makeZero(const Mask &k);

        ///////////////////////////////////////////////////////////////////////////////////////////
        // stores
        void store(EntryType *mem) const;
        void store(EntryType *mem, const Mask &mask) const;
        template<typename A> void store(EntryType *mem, A align) const;
        template<typename A> void store(EntryType *mem, const Mask &mask, A align) const;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // swizzles
        inline const Vector<T> &dcba() const { return *this; }
        inline const Vector<T> cdab() const { return reinterpret_cast<VectorType>(_mm_shuffle_epi32(data(), _MM_SHUFFLE(2, 3, 0, 1))); }
        inline const Vector<T> badc() const { return reinterpret_cast<VectorType>(_mm_shuffle_epi32(data(), _MM_SHUFFLE(1, 0, 3, 2))); }
        inline const Vector<T> aaaa() const { return reinterpret_cast<VectorType>(_mm_shuffle_epi32(data(), _MM_SHUFFLE(0, 0, 0, 0))); }
        inline const Vector<T> bbbb() const { return reinterpret_cast<VectorType>(_mm_shuffle_epi32(data(), _MM_SHUFFLE(1, 1, 1, 1))); }
        inline const Vector<T> cccc() const { return reinterpret_cast<VectorType>(_mm_shuffle_epi32(data(), _MM_SHUFFLE(2, 2, 2, 2))); }
        inline const Vector<T> dddd() const { return reinterpret_cast<VectorType>(_mm_shuffle_epi32(data(), _MM_SHUFFLE(3, 3, 3, 3))); }
        inline const Vector<T> dacb() const { return reinterpret_cast<VectorType>(_mm_shuffle_epi32(data(), _MM_SHUFFLE(3, 0, 2, 1))); }

        inline Vector(const EntryType *array, const unsigned int *indexes) {
            GatherHelper<T>::gather(*this, indexes, array);
        }
        inline Vector(const EntryType *array, const IndexType &indexes) {
            GatherHelper<T>::gather(*this, indexes, array);
        }
        inline Vector(const EntryType *array, const IndexType &indexes, const GatherMask &mask) {
#ifdef VC_GATHER_SET
            typedef typename IndexType::VectorType IType;
            const IType k = mm128_reinterpret_cast<IType>(mask.dataIndex());
            GatherHelper<T>::gather(*this, VectorHelper<IType>::and_(k, indexes.data()), array);
#else
            GeneralHelpers::maskedGatherHelper(*this, indexes, mask.toInt(), array);
#endif
        }
        inline Vector(const EntryType *array, const IndexType &indexes, const GatherMask &mask, VectorSpecialInitializerZero::ZEnum)
#ifndef VC_GATHER_SET
            : Base(VectorHelper<VectorType>::zero())
#endif
        {
#ifdef VC_GATHER_SET
            typedef typename IndexType::VectorType IType;
            const IType k = mm128_reinterpret_cast<IType>(mask.dataIndex());
            GatherHelper<T>::gather(*this, VectorHelper<IType>::and_(k, indexes.data()), array);
            data() = VectorHelper<VectorType>::and_(mm128_reinterpret_cast<VectorType>(mask.data()), data());
#else
            GeneralHelpers::maskedGatherHelper(*this, indexes, mask.toInt(), array);
#endif
        }
        inline Vector(const EntryType *array, const IndexType &indexes, const GatherMask &mask, EntryType def)
            : Base(VectorHelper<T>::set(def))
        {
#ifdef VC_GATHER_SET
            typedef typename IndexType::VectorType IType;
            const IType k = mm128_reinterpret_cast<IType>(mask.dataIndex());
            Vector<T> tmp;
            GatherHelper<T>::gather(tmp, VectorHelper<IType>::and_(indexes.data(), k), array);
            assign(tmp, mask.data());
#else
            GeneralHelpers::maskedGatherHelper(*this, indexes, mask.toInt(), array);
#endif
        }

        inline void gather(const EntryType *array, const IndexType &indexes) {
            GatherHelper<T>::gather(*this, indexes, array);
        }
        inline void gather(const EntryType *array, const IndexType &indexes, const GatherMask &mask) {
#ifdef VC_GATHER_SET
            typedef typename IndexType::VectorType IType;
            const IType k = mm128_reinterpret_cast<IType>(mask.dataIndex());
            Vector<T> tmp;
            GatherHelper<T>::gather(tmp, VectorHelper<IType>::and_(k, indexes.data()), array);
            assign(tmp, mask.data());
#else
            GeneralHelpers::maskedGatherHelper(*this, indexes, mask.toInt(), array);
#endif
        }

        inline void scatter(EntryType *array, const IndexType &indexes) const {
            ScatterHelper<T>::scatter(*this, indexes, array);
        }
        inline void scatter(EntryType *array, const IndexType &indexes, const GatherMask &mask) const {
            ScatterHelper<T>::scatter(*this, indexes, mask.toInt(), array);
        }

        /**
         * \param array An array of objects where one member should be gathered
         * \param member1 A member pointer to the member of the class/struct that should be gathered
         * \param indexes The indexes in the array. The correct offsets are calculated
         *                automatically.
         * \param mask Optional mask to select only parts of the vector that should be gathered
         */
        template<typename S1> inline Vector(const S1 *array, const EntryType S1::* member1, const IndexType &indexes) {
            GatherHelper<T>::gather(*this, indexes, array, member1);
        }
        template<typename S1> inline Vector(const S1 *array, const EntryType S1::* member1,
                const IndexType &indexes, const GatherMask &mask) {
            GeneralHelpers::maskedGatherStructHelper<sizeof(S1)>(*this, indexes, mask.toInt(), &(array[0].*(member1)));
        }
        template<typename S1, typename S2> inline Vector(const S1 *array, const S2 S1::* member1,
                const EntryType S2::* member2, const IndexType &indexes) {
            GatherHelper<T>::gather(*this, indexes, array, member1, member2);
        }
        template<typename S1, typename S2> inline Vector(const S1 *array, const S2 S1::* member1,
                const EntryType S2::* member2, const IndexType &indexes, const GatherMask &mask) {
            GeneralHelpers::maskedGatherStructHelper<sizeof(S1)>(*this, indexes, mask.toInt(), &(array[0].*(member1).*(member2)));
        }

        template<typename S1> inline Vector(const S1 *array, const EntryType *const S1::* ptrMember1,
                const IndexType &outerIndex, const IndexType &innerIndex, const GatherMask &mask) {
            GeneralHelpers::maskedDoubleGatherHelper<sizeof(S1)>(*this, outerIndex, innerIndex, mask.toInt(), &(array[0].*(ptrMember1)));
        }
        template<typename S1> inline void gather(const S1 *array, const EntryType *const S1::* ptrMember1,
                const IndexType &outerIndex, const IndexType &innerIndex, const GatherMask &mask) {
            GeneralHelpers::maskedDoubleGatherHelper<sizeof(S1)>(*this, outerIndex, innerIndex, mask.toInt(), &(array[0].*(ptrMember1)));
        }

        template<typename S1> inline void gather(const S1 *array, const EntryType S1::* member1,
                const IndexType &indexes) {
            GatherHelper<T>::gather(*this, indexes, array, member1);
        }
        template<typename S1> inline void gather(const S1 *array, const EntryType S1::* member1,
                const IndexType &indexes, const GatherMask &mask) {
            GeneralHelpers::maskedGatherStructHelper<sizeof(S1)>(*this, indexes, mask.toInt(), &(array[0].*(member1)));
        }
        template<typename S1, typename S2> inline void gather(const S1 *array, const S2 S1::* member1,
                const EntryType S2::* member2, const IndexType &indexes) {
            GatherHelper<T>::gather(*this, indexes, array, member1, member2);
        }
        template<typename S1, typename S2> inline void gather(const S1 *array, const S2 S1::* member1,
                const EntryType S2::* member2, const IndexType &indexes, const GatherMask &mask) {
            GeneralHelpers::maskedGatherStructHelper<sizeof(S1)>(*this, indexes, mask.toInt(), &(array[0].*(member1).*(member2)));
        }

        template<typename S1> inline void scatter(S1 *array, EntryType S1::* member1,
                const IndexType &indexes) const {
            ScatterHelper<T>::scatter(*this, indexes, array, member1);
        }
        template<typename S1> inline void scatter(S1 *array, EntryType S1::* member1,
                const IndexType &indexes, const GatherMask &mask) const {
            ScatterHelper<T>::scatter(*this, indexes, mask.toInt(), array, member1);
        }
        template<typename S1, typename S2> inline void scatter(S1 *array, S2 S1::* member1,
                EntryType S2::* member2, const IndexType &indexes) const {
            ScatterHelper<T>::scatter(*this, indexes, array, member1, member2);
        }
        template<typename S1, typename S2> inline void scatter(S1 *array, S2 S1::* member1,
                EntryType S2::* member2, const IndexType &indexes, const GatherMask &mask) const {
            ScatterHelper<T>::scatter(*this, indexes, mask.toInt(), array, member1, member2);
        }

        //prefix
        inline Vector &operator++() ALWAYS_INLINE { data() = VectorHelper<T>::add(data(), VectorHelper<T>::one()); return *this; }
        //postfix
        inline Vector operator++(int) ALWAYS_INLINE { const Vector<T> r = *this; data() = VectorHelper<T>::add(data(), VectorHelper<T>::one()); return r; }

        inline AliasingEntryType &operator[](int index) ALWAYS_INLINE {
            return Base::d.m(index);
        }
        inline EntryType operator[](int index) const ALWAYS_INLINE {
            return Base::d.m(index);
        }

        inline Vector operator~() const ALWAYS_INLINE { return VectorHelper<VectorType>::andnot_(data(), VectorHelper<VectorType>::allone()); }

#define OP1(fun) \
        inline Vector fun() const { return Vector<T>(VectorHelper<T>::fun(data())); } \
        inline Vector &fun##_eq() { data() = VectorHelper<T>::fun(data()); return *this; }
        OP1(sqrt)
        OP1(abs)
#undef OP1

        inline Vector operator-() const ALWAYS_INLINE { return VectorHelper<T>::negate(data()); }

#define OP(symbol, fun) \
        inline Vector &operator symbol##=(const Vector<T> &x) ALWAYS_INLINE { data() = VectorHelper<T>::fun(data(), x.data()); return *this; } \
        inline Vector operator symbol(const Vector<T> &x) const ALWAYS_INLINE { return Vector<T>(VectorHelper<T>::fun(data(), x.data())); }

        OP(+, add)
        OP(-, sub)
        OP(*, mul)
        OP(/, div)
#undef OP

#define OP(symbol, fun) \
        inline Vector &operator symbol##=(const Vector<T> &x) ALWAYS_INLINE { data() = VectorHelper<VectorType>::fun(data(), x.data()); return *this; } \
        inline Vector operator symbol(const Vector<T> &x) const ALWAYS_INLINE { return Vector<T>(VectorHelper<VectorType>::fun(data(), x.data())); }
        OP(|, or_)
        OP(&, and_)
        OP(^, xor_)
#undef OP
#define OPcmp(symbol, fun) \
        inline Mask operator symbol(const Vector<T> &x) const ALWAYS_INLINE { return VectorHelper<T>::fun(data(), x.data()); }

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

        inline WriteMaskedVector<T> operator()(const Mask &k) ALWAYS_INLINE { return WriteMaskedVector<T>(this, k); }

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

        inline EntryType min() const { return VectorHelper<T>::min(data()); }
        inline EntryType max() const { return VectorHelper<T>::max(data()); }
        inline EntryType product() const { return VectorHelper<T>::mul(data()); }
        inline EntryType sum() const { return VectorHelper<T>::add(data()); }

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

template<typename T> inline Vector<T> operator+(const typename Vector<T>::EntryType &x, const Vector<T> &v) ALWAYS_INLINE;
template<typename T> inline Vector<T> operator*(const typename Vector<T>::EntryType &x, const Vector<T> &v) ALWAYS_INLINE;
template<typename T> inline Vector<T> operator-(const typename Vector<T>::EntryType &x, const Vector<T> &v) ALWAYS_INLINE;
template<typename T> inline Vector<T> operator/(const typename Vector<T>::EntryType &x, const Vector<T> &v) ALWAYS_INLINE;
template<typename T> inline typename Vector<T>::Mask  operator< (const typename Vector<T>::EntryType &x, const Vector<T> &v) ALWAYS_INLINE;
template<typename T> inline typename Vector<T>::Mask  operator<=(const typename Vector<T>::EntryType &x, const Vector<T> &v) ALWAYS_INLINE;
template<typename T> inline typename Vector<T>::Mask  operator> (const typename Vector<T>::EntryType &x, const Vector<T> &v) ALWAYS_INLINE;
template<typename T> inline typename Vector<T>::Mask  operator>=(const typename Vector<T>::EntryType &x, const Vector<T> &v) ALWAYS_INLINE;
template<typename T> inline typename Vector<T>::Mask  operator==(const typename Vector<T>::EntryType &x, const Vector<T> &v) ALWAYS_INLINE;
template<typename T> inline typename Vector<T>::Mask  operator!=(const typename Vector<T>::EntryType &x, const Vector<T> &v) ALWAYS_INLINE;
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

#define OP_IMPL(T, symbol, fun) \
  template<> inline Vector<T> &VectorBase<T>::operator symbol##=(const VectorBase<T> &x) { d.v() = VectorHelper<T>::fun(d.v(), x.d.v()); return *static_cast<Vector<T> *>(this); } \
  template<> inline Vector<T>  VectorBase<T>::operator symbol(const VectorBase<T> &x) const { return Vector<T>(VectorHelper<T>::fun(d.v(), x.d.v())); }
  OP_IMPL(int, &, and_)
  OP_IMPL(int, |, or_)
  OP_IMPL(int, ^, xor_)
  OP_IMPL(int, <<, sll)
  OP_IMPL(int, >>, srl)
  OP_IMPL(unsigned int, &, and_)
  OP_IMPL(unsigned int, |, or_)
  OP_IMPL(unsigned int, ^, xor_)
  OP_IMPL(unsigned int, <<, sll)
  OP_IMPL(unsigned int, >>, srl)
  OP_IMPL(short, &, and_)
  OP_IMPL(short, |, or_)
  OP_IMPL(short, ^, xor_)
  OP_IMPL(short, <<, sll)
  OP_IMPL(short, >>, srl)
  OP_IMPL(unsigned short, &, and_)
  OP_IMPL(unsigned short, |, or_)
  OP_IMPL(unsigned short, ^, xor_)
  OP_IMPL(unsigned short, <<, sll)
  OP_IMPL(unsigned short, >>, srl)
#undef OP_IMPL
#define OP_IMPL(T, symbol, fun) \
  template<> inline Vector<T> &VectorBase<T>::operator symbol##=(int x) { d.v() = VectorHelper<T>::fun(d.v(), x); return *static_cast<Vector<T> *>(this); } \
  template<> inline Vector<T>  VectorBase<T>::operator symbol(int x) const { return Vector<T>(VectorHelper<T>::fun(d.v(), x)); }
  OP_IMPL(int, <<, slli)
  OP_IMPL(int, >>, srli)
  OP_IMPL(unsigned int, <<, slli)
  OP_IMPL(unsigned int, >>, srli)
  OP_IMPL(short, <<, slli)
  OP_IMPL(short, >>, srli)
  OP_IMPL(unsigned short, <<, slli)
  OP_IMPL(unsigned short, >>, srli)
#undef OP_IMPL

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
