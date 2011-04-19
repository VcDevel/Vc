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

namespace Warnings
{
    void _operator_bracket_warning()
#if defined(__GNUC__) && !defined(__INTEL_COMPILER) && __GNUC__ >= 4 && __GNUC_MINOR__ >= 3
        __attribute__((warning("\n\tUse of Vc::SSE::Vector::operator[] to modify scalar entries is known to miscompile with GCC 4.3.x.\n\tPlease upgrade to a more recent GCC or avoid operator[] altogether.\n\t(This warning adds an unnecessary function call to operator[] which should work around the problem at a little extra the cost.)")))
#endif
        ;
} // namespace Warnings

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
        explicit inline Vector(const Vector<typename CtorTypeHelper<T>::Type> *a) INTRINSIC;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // expand 1 short_v to 2 int_v                    XXX rationale? remove it for release? XXX
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
        inline Vector operator-() const PURE INTRINSIC { return VectorHelper<T>::negate(data()); }

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

        inline void assign( const Vector<T> &v, const Mask &mask ) {
            const VectorType k = mm128_reinterpret_cast<VectorType>(mask.data());
            data() = VectorHelper<VectorType>::blend(data(), v.data(), k);
        }

        template<typename V2> inline V2 staticCast() const { return StaticCastHelper<T, typename V2::_T>::cast(data()); }
        template<typename V2> inline V2 reinterpretCast() const { return mm128_reinterpret_cast<typename V2::VectorType>(data()); }

        inline WriteMaskedVector<T> operator()(const Mask &k) INTRINSIC { return WriteMaskedVector<T>(this, k); }

        inline VectorType &data() { return Base::data(); }
        inline const VectorType &data() const { return Base::data(); }

        inline EntryType min() const INTRINSIC { return VectorHelper<T>::min(data()); }
        inline EntryType max() const INTRINSIC { return VectorHelper<T>::max(data()); }
        inline EntryType product() const INTRINSIC { return VectorHelper<T>::mul(data()); }
        inline EntryType sum() const INTRINSIC { return VectorHelper<T>::add(data()); }

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
