/*  This file is part of the Vc library.

    Copyright (C) 2009-2010 Matthias Kretz <kretz@kde.org>

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
enum { VectorAlignment = 32 };

template<typename T> class Vector
{
    public:
        FREE_STORE_OPERATORS_ALIGNED(32)

        typedef typename VectorTypeHelper<T>::Type VectorType;
        typedef T EntryType;
        enum { Size = sizeof(VectorType) / sizeof(EntryType),
            HasVectorDivision = HasVectorDivisionHelper<T>::Value
        };
        typedef Vector<typename IndexTypeHelper<T>::Type> IndexType;
        typedef typename Vc::AVX::Mask<Size, sizeof(VectorType)> Mask;
        typedef Vc::Memory<Vector<T>, Size> Memory;

    protected:
        // helper that specializes on VectorType
        typedef VectorHelper<VectorType> HV;

        // helper that specializes on T
        typedef VectorHelper<T> HT;

        // cast any m256/m128 to VectorType
        static inline VectorType _cast(__m128  v) INTRINSIC { return avx_cast<VectorType>(v); }
        static inline VectorType _cast(__m128i v) INTRINSIC { return avx_cast<VectorType>(v); }
        static inline VectorType _cast(__m128d v) INTRINSIC { return avx_cast<VectorType>(v); }
        static inline VectorType _cast(__m256  v) INTRINSIC { return avx_cast<VectorType>(v); }
        static inline VectorType _cast(__m256i v) INTRINSIC { return avx_cast<VectorType>(v); }
        static inline VectorType _cast(__m256d v) INTRINSIC { return avx_cast<VectorType>(v); }

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

        ///////////////////////////////////////////////////////////////////////////////////////////
        // broadcast
        Vector(EntryType a);

        ///////////////////////////////////////////////////////////////////////////////////////////
        // load ctors
        explicit inline
            Vector(const EntryType *x) INTRINSIC;
        template<typename Alignment> inline
            Vector(const EntryType *x, Alignment align) INTRINSIC;
        template<typename OtherT> explicit inline
            Vector(const OtherT    *x) INTRINSIC;
        template<typename OtherT, typename Alignment> inline
            Vector(const OtherT    *x, Alignment align) INTRINSIC;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // load member functions
        inline
            void load(const EntryType *mem) INTRINSIC;
        template<typename Alignment> inline
            void load(const EntryType *mem, Alignment align) INTRINSIC;
        template<typename OtherT> inline
            void load(const OtherT    *mem) INTRINSIC;
        template<typename OtherT, typename Alignment> inline
            void load(const OtherT    *mem, Alignment align) INTRINSIC;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // expand/merge 1 float_v <=> 2 double_v          XXX rationale? remove it for release? XXX
        explicit inline Vector(const Vector<typename HT::ConcatType> *a);
        inline void expand(Vector<typename HT::ConcatType> *x) const;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // zeroing
        inline void setZero();
        inline void setZero(const Mask &k);

        void setQnan();
        void setQnan(Mask k);

        ///////////////////////////////////////////////////////////////////////////////////////////
        // stores
        inline void store(EntryType *mem) const;
        inline void store(EntryType *mem, const Mask &mask) const;
        template<typename A> inline void store(EntryType *mem, A align) const;
        template<typename A> inline void store(EntryType *mem, const Mask &mask, A align) const;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // swizzles
        inline const Vector<T> &dcba() const { return *this; }
        inline const Vector<T> INTRINSIC CONST cdab() const { return HV::cdab(d.v()); }
        inline const Vector<T> INTRINSIC CONST badc() const { return HV::badc(d.v()); }
        inline const Vector<T> INTRINSIC CONST aaaa() const { return HV::aaaa(d.v()); }
        inline const Vector<T> INTRINSIC CONST bbbb() const { return HV::bbbb(d.v()); }
        inline const Vector<T> INTRINSIC CONST cccc() const { return HV::cccc(d.v()); }
        inline const Vector<T> INTRINSIC CONST dddd() const { return HV::dddd(d.v()); }
        inline const Vector<T> INTRINSIC CONST dacb() const { return HV::dacb(d.v()); }

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

        ///////////////////////////////////////////////////////////////////////////////////////////
        //prefix
        inline Vector &operator++() ALWAYS_INLINE { data() = VectorHelper<T>::add(data(), VectorHelper<T>::one()); return *this; }
        //postfix
        inline Vector operator++(int) ALWAYS_INLINE { const Vector<T> r = *this; data() = VectorHelper<T>::add(data(), VectorHelper<T>::one()); return r; }

        inline Common::AliasingEntryHelper<EntryType> INTRINSIC operator[](int index) {
#if defined(VC_GCC) && VC_GCC >= 0x40300 && VC_GCC < 0x40400
            ::Vc::Warnings::_operator_bracket_warning();
#endif
            return d.m(index);
        }
        inline EntryType operator[](int index) const ALWAYS_INLINE {
            return d.m(index);
        }

        inline Vector operator~() const ALWAYS_INLINE { return VectorHelper<VectorType>::andnot_(data(), VectorHelper<VectorType>::allone()); }
        inline Vector<typename NegateTypeHelper<T>::Type> operator-() const;

#define OP1(fun) \
        inline Vector fun() const { return Vector<T>(VectorHelper<T>::fun(data())); } \
        inline Vector &fun##_eq() { data() = VectorHelper<T>::fun(data()); return *this; }
        OP1(sqrt)
        OP1(abs)
#undef OP1

#define OP(symbol, fun) \
        inline Vector &operator symbol##=(const Vector<T> &x) ALWAYS_INLINE { data() = VectorHelper<T>::fun(data(), x.data()); return *this; } \
        inline Vector operator symbol(const Vector<T> &x) const ALWAYS_INLINE { return Vector<T>(VectorHelper<T>::fun(data(), x.data())); }

        OP(+, add)
        OP(-, sub)
        OP(*, mul)
#undef OP
        inline Vector &operator/=(EntryType x);
        inline Vector  operator/ (EntryType x) const;
        inline Vector &operator/=(const Vector<T> &x);
        inline Vector  operator/ (const Vector<T> &x) const;

        // bitwise ops
        inline Vector<T> &operator|= (Vector<T> x) ALWAYS_INLINE;
        inline Vector<T> &operator&= (Vector<T> x) ALWAYS_INLINE;
        inline Vector<T> &operator^= (Vector<T> x) ALWAYS_INLINE;
        inline Vector<T> &operator>>=(Vector<T> x) ALWAYS_INLINE;
        inline Vector<T> &operator<<=(Vector<T> x) ALWAYS_INLINE;
        inline Vector<T> &operator>>=(int x) ALWAYS_INLINE;
        inline Vector<T> &operator<<=(int x) ALWAYS_INLINE;

        inline Vector<T> operator| (Vector<T> x) const ALWAYS_INLINE;
        inline Vector<T> operator& (Vector<T> x) const ALWAYS_INLINE;
        inline Vector<T> operator^ (Vector<T> x) const ALWAYS_INLINE;
        inline Vector<T> operator>>(Vector<T> x) const ALWAYS_INLINE;
        inline Vector<T> operator<<(Vector<T> x) const ALWAYS_INLINE;
        inline Vector<T> operator>>(int x) const ALWAYS_INLINE;
        inline Vector<T> operator<<(int x) const ALWAYS_INLINE;

#define OPcmp(symbol, fun) \
        inline Mask operator symbol(Vector<T> x) const ALWAYS_INLINE { return VectorHelper<T>::fun(data(), x.data()); }

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

        inline WriteMaskedVector<T> operator()(const Mask &k) ALWAYS_INLINE { return WriteMaskedVector<T>(this, k); }

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
        inline EntryType min(Mask m) const;
        inline EntryType max(Mask m) const;
        inline EntryType product(Mask m) const;
        inline EntryType sum(Mask m) const;

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

        template<typename F> inline void INTRINSIC call(F &f) const {
            for_all_vector_entries(i,
                    f(EntryType(d.m(i)));
                    );
        }

        template<typename F> inline void INTRINSIC call(F &f, const Mask &mask) const {
            Vc_foreach_bit(size_t i, mask) {
                f(EntryType(d.m(i)));
            }
        }

        template<typename F> inline Vector<T> INTRINSIC apply(F &f) const {
            Vector<T> r;
            for_all_vector_entries(i,
                    r.d.m(i) = f(EntryType(d.m(i)));
                    );
            return r;
        }

        template<typename F> inline Vector<T> INTRINSIC apply(F &f, const Mask &mask) const {
            Vector<T> r(*this);
            Vc_foreach_bit (size_t i, mask) {
                r.d.m(i) = f(EntryType(r.d.m(i)));
            }
            return r;
        }

        inline INTRINSIC_L Vector copySign(Vector reference) const INTRINSIC_R;
        inline INTRINSIC_L Vector exponent() const INTRINSIC_R;
};

typedef Vector<double>         double_v;
typedef Vector<float>          float_v;
typedef Vector<float>          sfloat_v;
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

  template<typename T> static inline Vector<T> min  (const Vector<T> &x, const Vector<T> &y) { return VectorHelper<T>::min(x.data(), y.data()); }
  template<typename T> static inline Vector<T> max  (const Vector<T> &x, const Vector<T> &y) { return VectorHelper<T>::max(x.data(), y.data()); }
  template<typename T> static inline Vector<T> min  (const Vector<T> &x, const typename Vector<T>::EntryType &y) { return min(x.data(), Vector<T>(y).data()); }
  template<typename T> static inline Vector<T> max  (const Vector<T> &x, const typename Vector<T>::EntryType &y) { return max(x.data(), Vector<T>(y).data()); }
  template<typename T> static inline Vector<T> min  (const typename Vector<T>::EntryType &x, const Vector<T> &y) { return min(Vector<T>(x).data(), y.data()); }
  template<typename T> static inline Vector<T> max  (const typename Vector<T>::EntryType &x, const Vector<T> &y) { return max(Vector<T>(x).data(), y.data()); }
  template<typename T> static inline Vector<T> sqrt (const Vector<T> &x) { return VectorHelper<T>::sqrt(x.data()); }
  template<typename T> static inline Vector<T> rsqrt(const Vector<T> &x) { return VectorHelper<T>::rsqrt(x.data()); }
  template<typename T> static inline Vector<T> abs  (const Vector<T> &x) { return VectorHelper<T>::abs(x.data()); }
  template<typename T> static inline Vector<T> reciprocal(const Vector<T> &x) { return VectorHelper<T>::reciprocal(x.data()); }
  template<typename T> static inline Vector<T> round(const Vector<T> &x) { return VectorHelper<T>::round(x.data()); }

  template<typename T> static inline typename Vector<T>::Mask isfinite(const Vector<T> &x) { return VectorHelper<T>::isFinite(x.data()); }
  template<typename T> static inline typename Vector<T>::Mask isnan(const Vector<T> &x) { return VectorHelper<T>::isNaN(x.data()); }

#include "forceToRegisters.tcc"

#undef STORE_VECTOR
} // namespace AVX
} // namespace Vc

#include "vector.tcc"
#include "math.h"
#include "undomacros.h"

#endif // AVX_VECTOR_H
