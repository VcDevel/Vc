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
#include "memory.h"
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
    public:
        FREE_STORE_OPERATORS_ALIGNED(16)

        enum { Size = Base::Size };
        typedef typename Base::VectorType VectorType;
        typedef typename Base::EntryType  EntryType;
        typedef Vector<typename IndexTypeHelper<Size>::Type> IndexType;
        typedef typename Base::MaskType Mask;
        typedef _Memory<T> Memory;

        /**
         * uninitialized
         */
        inline Vector() {}

        /**
         * initialized to 0 in all 128 bits
         */
        inline explicit Vector(VectorSpecialInitializerZero::ZEnum) : Base(VectorHelper<VectorType>::zero()) {}

        /**
         * initialized to 1 for all entries in the vector
         */
        inline explicit Vector(VectorSpecialInitializerOne::OEnum) : Base(VectorHelper<T>::one()) {}

        /**
         * initialized to 0, 1 (, 2, 3 (, 4, 5, 6, 7))
         */
        inline explicit Vector(VectorSpecialInitializerIndexesFromZero::IEnum) : Base(VectorHelper<VectorType>::load(Base::_IndexesFromZero())) {}

        /**
         * initialize with given _M128 vector
         */
        inline Vector(const VectorType &x) : Base(x) {}

        template<typename OtherT>
        explicit inline Vector(const Vector<OtherT> &x) : Base(StaticCastHelper<OtherT, T>::cast(x.data())) {}

        /**
         * initialize all values with the given value
         */
        inline Vector(EntryType a) : Base(VectorHelper<T>::set(a)) {}

        /**
         * Initialize the vector with the given data. \param x must point to 64 byte aligned 512
         * byte data.
         */
        inline explicit Vector(const EntryType *x) : Base(VectorHelper<VectorType>::load(x)) {}

        inline explicit Vector(const Vector<typename CtorTypeHelper<T>::Type> *a)
            : Base(VectorHelper<T>::concat(a[0].data(), a[1].data()))
        {}

        inline void expand(Vector<typename ExpandTypeHelper<T>::Type> *x) const
        {
            if (Size == 8u) {
                x[0].data() = VectorHelper<T>::expand0(data());
                x[1].data() = VectorHelper<T>::expand1(data());
            }
        }

        static inline Vector broadcast4(const EntryType *x) { return Vector<T>(x); }

        inline void load(const EntryType *mem) { data() = VectorHelper<VectorType>::load(mem); }

        static inline Vector loadUnaligned(const EntryType *mem) { return VectorHelper<VectorType>::loadUnaligned(mem); }

        inline void makeZero() { data() = VectorHelper<VectorType>::zero(); }

        /**
         * Set all entries to zero where the mask is set. I.e. a 4-vector with a mask of 0111 would
         * set the last three entries to 0.
         */
        inline void makeZero(const Mask &k) { data() = VectorHelper<VectorType>::andnot_(mm128_reinterpret_cast<VectorType>(k.data()), data()); }

        /**
         * Store the vector data to the given memory. The memory must be 64 byte aligned and of 512
         * bytes size.
         */
        inline void store(EntryType *mem) const { VectorHelper<VectorType>::store(mem, data()); }
        inline void store(EntryType *mem, const Mask &mask) const {
            const VectorType &old = VectorHelper<VectorType>::load(mem);
            VectorHelper<VectorType>::store(mem, VectorHelper<VectorType>::blend(old, data(), mm128_reinterpret_cast<VectorType>(mask.data())));
        }

        /**
         * Non-temporal store variant. Writes to the memory without polluting the cache.
         */
        inline void storeStreaming(EntryType *mem) const { VectorHelper<VectorType>::storeStreaming(mem, data()); }

        inline const Vector<T> &dcba() const { return *this; }
        inline const Vector<T> cdab() const { return reinterpret_cast<VectorType>(_mm_shuffle_epi32(data(), _MM_SHUFFLE(2, 3, 0, 1))); }
        inline const Vector<T> badc() const { return reinterpret_cast<VectorType>(_mm_shuffle_epi32(data(), _MM_SHUFFLE(1, 0, 3, 2))); }
        inline const Vector<T> aaaa() const { return reinterpret_cast<VectorType>(_mm_shuffle_epi32(data(), _MM_SHUFFLE(0, 0, 0, 0))); }
        inline const Vector<T> bbbb() const { return reinterpret_cast<VectorType>(_mm_shuffle_epi32(data(), _MM_SHUFFLE(1, 1, 1, 1))); }
        inline const Vector<T> cccc() const { return reinterpret_cast<VectorType>(_mm_shuffle_epi32(data(), _MM_SHUFFLE(2, 2, 2, 2))); }
        inline const Vector<T> dddd() const { return reinterpret_cast<VectorType>(_mm_shuffle_epi32(data(), _MM_SHUFFLE(3, 3, 3, 3))); }
        inline const Vector<T> dacb() const { return reinterpret_cast<VectorType>(_mm_shuffle_epi32(data(), _MM_SHUFFLE(3, 0, 2, 1))); }

        inline Vector(const EntryType *array, const IndexType &indexes) {
            GatherHelper<T>::gather(*this, indexes, array);
        }
        inline Vector(const EntryType *array, const IndexType &indexes, const GatherMask &mask) {
            GeneralHelpers::maskedGatherHelper(*this, indexes, mask.toInt(), array);
        }
        inline Vector(const EntryType *array, const IndexType &indexes, const GatherMask &mask, VectorSpecialInitializerZero::ZEnum)
            : Base(VectorHelper<VectorType>::zero())
        {
            GeneralHelpers::maskedGatherHelper(*this, indexes, mask.toInt(), array);
        }
        inline Vector(const EntryType *array, const IndexType &indexes, const GatherMask &mask, EntryType def)
            : Base(VectorHelper<T>::set(def))
        {
            GeneralHelpers::maskedGatherHelper(*this, indexes, mask.toInt(), array);
        }

        inline void gather(const EntryType *array, const IndexType &indexes) {
            GatherHelper<T>::gather(*this, indexes, array);
        }
        inline void gather(const EntryType *array, const IndexType &indexes, const GatherMask &mask) {
            GeneralHelpers::maskedGatherHelper(*this, indexes, mask.toInt(), array);
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

        template<typename T2> inline Vector<T2> staticCast() const { return StaticCastHelper<T, T2>::cast(data()); }
        template<typename T2> inline Vector<T2> reinterpretCast() const { return ReinterpretCastHelper<T, T2>::cast(data()); }

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
    const _M128 &v = VectorHelper<_M128>::load(x);
    return Vector<float8>(M256(v, v));
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
  template<> inline Vector<T> &VectorBase<T>::operator symbol##=(const Vector<T> &x) { d.v() = VectorHelper<T>::fun(d.v(), x.d.v()); return *static_cast<Vector<T> *>(this); } \
  template<> inline Vector<T>  VectorBase<T>::operator symbol(const Vector<T> &x) const { return Vector<T>(VectorHelper<T>::fun(d.v(), x.d.v())); }
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

#endif // SSE_VECTOR_H
