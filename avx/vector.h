/*  This file is part of the Vc library.

    Copyright (C) 2009-2013 Matthias Kretz <kretz@kde.org>

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

Vc_NAMESPACE_BEGIN(Vc_IMPL_NAMESPACE)
enum VectorAlignmentEnum { VectorAlignment = 32 };

template<typename T> class Vector
{
    public:
        FREE_STORE_OPERATORS_ALIGNED(32)

        typedef typename VectorTypeHelper<T>::Type VectorType;
        typedef typename DetermineEntryType<T>::Type EntryType;
        static constexpr size_t Size = sizeof(VectorType) / sizeof(EntryType);
        enum Constants {
            MemoryAlignment = alignof(VectorType),
            HasVectorDivision = HasVectorDivisionHelper<T>::Value
        };
        typedef Vector<typename IndexTypeHelper<T>::Type> IndexType;
        typedef Vc_IMPL_NAMESPACE::Mask<T> Mask;
        typedef typename Mask::AsArg MaskArg;
        typedef Vc::Memory<Vector<T>, Size> Memory;
#ifdef VC_PASSING_VECTOR_BY_VALUE_IS_BROKEN
        typedef const Vector<T> &AsArg;
        typedef const VectorType &VectorTypeArg;
#else
        typedef Vector<T> AsArg;
        typedef VectorType VectorTypeArg;
#endif

    protected:
        // helper that specializes on VectorType
        typedef VectorHelper<VectorType> HV;

        // helper that specializes on T
        typedef VectorHelper<T> HT;

        // cast any m256/m128 to VectorType
        template<typename V> static Vc_INTRINSIC VectorType _cast(VC_ALIGNED_PARAMETER(V) v) { return avx_cast<VectorType>(v); }

        typedef Common::VectorMemoryUnion<VectorType, EntryType> StorageType;
        StorageType d;

        using WidthT = Common::WidthT<VectorType>;
        // ICC can't compile this:
        // static constexpr WidthT Width = WidthT();

    public:
        ///////////////////////////////////////////////////////////////////////////////////////////
        // uninitialized
        Vc_ALWAYS_INLINE Vector() {}

        ///////////////////////////////////////////////////////////////////////////////////////////
        // constants
        explicit Vc_ALWAYS_INLINE_L Vector(VectorSpecialInitializerZero::ZEnum) Vc_ALWAYS_INLINE_R;
        explicit Vc_ALWAYS_INLINE_L Vector(VectorSpecialInitializerOne::OEnum) Vc_ALWAYS_INLINE_R;
        explicit Vc_ALWAYS_INLINE_L Vector(VectorSpecialInitializerIndexesFromZero::IEnum) Vc_ALWAYS_INLINE_R;
        static Vc_INTRINSIC_L Vc_CONST_L Vector Zero() Vc_INTRINSIC_R Vc_CONST_R;
        static Vc_INTRINSIC_L Vc_CONST_L Vector One() Vc_INTRINSIC_R Vc_CONST_R;
        static Vc_INTRINSIC_L Vc_CONST_L Vector IndexesFromZero() Vc_INTRINSIC_R Vc_CONST_R;
        static Vc_ALWAYS_INLINE_L Vector Random() Vc_ALWAYS_INLINE_R;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // internal: required to enable returning objects of VectorType
        Vc_ALWAYS_INLINE Vector(VectorTypeArg x) : d(x) {}
#ifdef VC_UNCONDITIONAL_AVX2_INTRINSICS
        Vc_ALWAYS_INLINE Vector(typename VectorType::Base x) : d(x) {}
#endif

        ///////////////////////////////////////////////////////////////////////////////////////////
        // copy
        Vc_INTRINSIC Vector(const Vector &x) = default;
        Vc_INTRINSIC Vector &operator=(const Vector &v) { d.v() = v.d.v(); return *this; }

        // implict conversion from compatible Vector<U>
        template<typename U> Vc_INTRINSIC Vector(VC_ALIGNED_PARAMETER(Vector<U>) x,
                typename std::enable_if<is_implicit_cast_allowed<U, T>::value, void *>::type = nullptr)
            : d(StaticCastHelper<U, T>::cast(x.data())) {}

        // static_cast from the remaining Vector<U>
        template<typename U> Vc_INTRINSIC explicit Vector(VC_ALIGNED_PARAMETER(Vector<U>) x,
                typename std::enable_if<!is_implicit_cast_allowed<U, T>::value, void *>::type = nullptr)
            : d(StaticCastHelper<U, T>::cast(x.data())) {}

        ///////////////////////////////////////////////////////////////////////////////////////////
        // broadcast
        Vc_INTRINSIC Vector(EntryType a) : d(HT::set(a)) {}
        template <typename U>
        Vc_INTRINSIC Vector(U a,
                            typename std::enable_if<std::is_same<U, int>::value &&
                                                        !std::is_same<U, EntryType>::value,
                                                    void *>::type = nullptr)
            : Vector(static_cast<EntryType>(a))
        {
        }

        ///////////////////////////////////////////////////////////////////////////////////////////
        // load ctors
        explicit Vc_INTRINSIC Vector(const EntryType * x) { load(x); }
        template<typename Flags = AlignedT> explicit Vc_INTRINSIC Vector(const EntryType * x, Flags flags = Flags())
        {
            load(x, flags);
        }
        template<typename OtherT, typename Flags = AlignedT> explicit Vc_INTRINSIC Vector(const OtherT *x, Flags flags = Flags())
        {
            load(x, flags);
        }

        ///////////////////////////////////////////////////////////////////////////////////////////
        // load member functions
        Vc_INTRINSIC void load(const EntryType *mem) { load<AlignedT>(mem, Aligned); }
        template<typename Flags = AlignedT> Vc_INTRINSIC_L
            void load(const EntryType *mem, Flags) Vc_INTRINSIC_R;
        template<typename OtherT, typename Flags = AlignedT> Vc_INTRINSIC_L
            void load(const OtherT    *mem, Flags = Flags()) Vc_INTRINSIC_R;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // expand/merge 1 float_v <=> 2 double_v          XXX rationale? remove it for release? XXX
        explicit inline Vector(const Vector<typename HT::ConcatType> *a);
        inline void expand(Vector<typename HT::ConcatType> *x) const;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // zeroing
        Vc_INTRINSIC_L void setZero() Vc_INTRINSIC_R;
        Vc_INTRINSIC_L void setZero(const Mask &k) Vc_INTRINSIC_R;

        Vc_INTRINSIC_L void setQnan() Vc_INTRINSIC_R;
        Vc_INTRINSIC_L void setQnan(MaskArg k) Vc_INTRINSIC_R;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // stores
        template<typename T2, typename Flags = AlignedT> Vc_INTRINSIC_L void store(T2 *mem, Flags = Flags()) const Vc_INTRINSIC_R;
        template<typename T2, typename Flags = AlignedT> Vc_INTRINSIC_L void store(T2 *mem, Mask mask, Flags = Flags()) const Vc_INTRINSIC_R;
        // the following store overloads are here to support classes that have a cast operator to EntryType.
        // Without this overload GCC complains about not finding a matching store function.
        Vc_INTRINSIC void store(EntryType *mem) const { store<EntryType, AlignedT>(mem); }
        template<typename Flags = AlignedT> Vc_INTRINSIC void store(EntryType *mem, Flags flags) const { store<EntryType, Flags>(mem, flags); }
        Vc_INTRINSIC void store(EntryType *mem, Mask mask) const { store<EntryType, AlignedT>(mem, mask); }
        template<typename Flags = AlignedT> Vc_INTRINSIC void store(EntryType *mem, Mask mask, Flags flags) const { store<EntryType, Flags>(mem, mask, flags); }

        ///////////////////////////////////////////////////////////////////////////////////////////
        // swizzles
        Vc_INTRINSIC_L Vc_PURE_L const Vector<T> &abcd() const Vc_INTRINSIC_R Vc_PURE_R;
        Vc_INTRINSIC_L Vc_PURE_L const Vector<T>  cdab() const Vc_INTRINSIC_R Vc_PURE_R;
        Vc_INTRINSIC_L Vc_PURE_L const Vector<T>  badc() const Vc_INTRINSIC_R Vc_PURE_R;
        Vc_INTRINSIC_L Vc_PURE_L const Vector<T>  aaaa() const Vc_INTRINSIC_R Vc_PURE_R;
        Vc_INTRINSIC_L Vc_PURE_L const Vector<T>  bbbb() const Vc_INTRINSIC_R Vc_PURE_R;
        Vc_INTRINSIC_L Vc_PURE_L const Vector<T>  cccc() const Vc_INTRINSIC_R Vc_PURE_R;
        Vc_INTRINSIC_L Vc_PURE_L const Vector<T>  dddd() const Vc_INTRINSIC_R Vc_PURE_R;
        Vc_INTRINSIC_L Vc_PURE_L const Vector<T>  bcad() const Vc_INTRINSIC_R Vc_PURE_R;
        Vc_INTRINSIC_L Vc_PURE_L const Vector<T>  bcda() const Vc_INTRINSIC_R Vc_PURE_R;
        Vc_INTRINSIC_L Vc_PURE_L const Vector<T>  dabc() const Vc_INTRINSIC_R Vc_PURE_R;
        Vc_INTRINSIC_L Vc_PURE_L const Vector<T>  acbd() const Vc_INTRINSIC_R Vc_PURE_R;
        Vc_INTRINSIC_L Vc_PURE_L const Vector<T>  dbca() const Vc_INTRINSIC_R Vc_PURE_R;
        Vc_INTRINSIC_L Vc_PURE_L const Vector<T>  dcba() const Vc_INTRINSIC_R Vc_PURE_R;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // gathers
        template<typename IndexT> Vector(const EntryType *mem, const IndexT *indexes);
        template<typename IndexT> Vector(const EntryType *mem, VC_ALIGNED_PARAMETER(Vector<IndexT>) indexes);
        template<typename IndexT> Vector(const EntryType *mem, const IndexT *indexes, MaskArg mask);
        template<typename IndexT> Vector(const EntryType *mem, VC_ALIGNED_PARAMETER(Vector<IndexT>) indexes, MaskArg mask);
        template<typename S1, typename IT> Vector(const S1 *array, const EntryType S1::* member1, VC_ALIGNED_PARAMETER(IT) indexes);
        template<typename S1, typename IT> Vector(const S1 *array, const EntryType S1::* member1, VC_ALIGNED_PARAMETER(IT) indexes, MaskArg mask);
        template<typename S1, typename S2, typename IT> Vector(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, VC_ALIGNED_PARAMETER(IT) indexes);
        template<typename S1, typename S2, typename IT> Vector(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, VC_ALIGNED_PARAMETER(IT) indexes, MaskArg mask);
        template<typename S1, typename IT1, typename IT2> Vector(const S1 *array, const EntryType *const S1::* ptrMember1, VC_ALIGNED_PARAMETER(IT1) outerIndexes, VC_ALIGNED_PARAMETER(IT2) innerIndexes);
        template<typename S1, typename IT1, typename IT2> Vector(const S1 *array, const EntryType *const S1::* ptrMember1, VC_ALIGNED_PARAMETER(IT1) outerIndexes, VC_ALIGNED_PARAMETER(IT2) innerIndexes, MaskArg mask);
        template<typename Index> Vc_ALWAYS_INLINE_L void gather(const EntryType *mem, VC_ALIGNED_PARAMETER(Index) indexes) Vc_ALWAYS_INLINE_R;
        template<typename Index> Vc_ALWAYS_INLINE_L void gather(const EntryType *mem, VC_ALIGNED_PARAMETER(Index) indexes, MaskArg mask) Vc_ALWAYS_INLINE_R;
#ifdef VC_USE_SET_GATHERS
        template<typename IT> Vc_ALWAYS_INLINE_L void gather(const EntryType *mem, VC_ALIGNED_PARAMETER(Vector<IT>) indexes, MaskArg mask) Vc_ALWAYS_INLINE_R;
#endif
        template<typename S1, typename IT> Vc_ALWAYS_INLINE_L void gather(const S1 *array, const EntryType S1::* member1, VC_ALIGNED_PARAMETER(IT) indexes) Vc_ALWAYS_INLINE_R;
        template<typename S1, typename IT> Vc_ALWAYS_INLINE_L void gather(const S1 *array, const EntryType S1::* member1, VC_ALIGNED_PARAMETER(IT) indexes, MaskArg mask) Vc_ALWAYS_INLINE_R;
        template<typename S1, typename S2, typename IT> Vc_ALWAYS_INLINE_L void gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, VC_ALIGNED_PARAMETER(IT) indexes) Vc_ALWAYS_INLINE_R;
        template<typename S1, typename S2, typename IT> Vc_ALWAYS_INLINE_L void gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, VC_ALIGNED_PARAMETER(IT) indexes, MaskArg mask) Vc_ALWAYS_INLINE_R;
        template<typename S1, typename IT1, typename IT2> Vc_ALWAYS_INLINE_L void gather(const S1 *array, const EntryType *const S1::* ptrMember1, VC_ALIGNED_PARAMETER(IT1) outerIndexes, VC_ALIGNED_PARAMETER(IT2) innerIndexes) Vc_ALWAYS_INLINE_R;
        template<typename S1, typename IT1, typename IT2> Vc_ALWAYS_INLINE_L void gather(const S1 *array, const EntryType *const S1::* ptrMember1, VC_ALIGNED_PARAMETER(IT1) outerIndexes, VC_ALIGNED_PARAMETER(IT2) innerIndexes, MaskArg mask) Vc_ALWAYS_INLINE_R;

        ///////////////////////////////////////////////////////////////////////////////////////////
        // scatters
        template<typename Index> void scatter(EntryType *mem, VC_ALIGNED_PARAMETER(Index) indexes) const;
        template<typename Index> void scatter(EntryType *mem, VC_ALIGNED_PARAMETER(Index) indexes, MaskArg mask) const;
        template<typename S1, typename IT> void scatter(S1 *array, EntryType S1::* member1, VC_ALIGNED_PARAMETER(IT) indexes) const;
        template<typename S1, typename IT> void scatter(S1 *array, EntryType S1::* member1, VC_ALIGNED_PARAMETER(IT) indexes, MaskArg mask) const;
        template<typename S1, typename S2, typename IT> void scatter(S1 *array, S2 S1::* member1, EntryType S2::* member2, VC_ALIGNED_PARAMETER(IT) indexes) const;
        template<typename S1, typename S2, typename IT> void scatter(S1 *array, S2 S1::* member1, EntryType S2::* member2, VC_ALIGNED_PARAMETER(IT) indexes, MaskArg mask) const;
        template<typename S1, typename IT1, typename IT2> void scatter(S1 *array, EntryType *S1::* ptrMember1, VC_ALIGNED_PARAMETER(IT1) outerIndexes, VC_ALIGNED_PARAMETER(IT2) innerIndexes) const;
        template<typename S1, typename IT1, typename IT2> void scatter(S1 *array, EntryType *S1::* ptrMember1, VC_ALIGNED_PARAMETER(IT1) outerIndexes, VC_ALIGNED_PARAMETER(IT2) innerIndexes, MaskArg mask) const;

        ///////////////////////////////////////////////////////////////////////////////////////////
        //prefix
        Vc_ALWAYS_INLINE Vector &operator++() { data() = VectorHelper<T>::add(data(), VectorHelper<T>::one()); return *this; }
        Vc_ALWAYS_INLINE Vector &operator--() { data() = VectorHelper<T>::sub(data(), VectorHelper<T>::one()); return *this; }
        //postfix
        Vc_ALWAYS_INLINE Vector operator++(int) { const Vector<T> r = *this; data() = VectorHelper<T>::add(data(), VectorHelper<T>::one()); return r; }
        Vc_ALWAYS_INLINE Vector operator--(int) { const Vector<T> r = *this; data() = VectorHelper<T>::sub(data(), VectorHelper<T>::one()); return r; }

        Vc_INTRINSIC decltype(d.m(0)) &operator[](size_t index) {
            return d.m(index);
        }
        Vc_ALWAYS_INLINE EntryType operator[](size_t index) const {
            return d.m(index);
        }

        Vc_ALWAYS_INLINE Vector operator~() const { return VectorHelper<VectorType>::andnot_(data(), VectorHelper<VectorType>::allone()); }
        Vc_ALWAYS_INLINE_L Vc_PURE_L Vector<typename NegateTypeHelper<T>::Type> operator-() const Vc_ALWAYS_INLINE_R Vc_PURE_R;
        Vc_INTRINSIC Vc_PURE Vector operator+() const { return *this; }

#define OP(symbol, fun) \
        Vc_ALWAYS_INLINE Vector &operator symbol##=(const Vector &x) { data() = VectorHelper<T>::fun(data(), x.data()); return *this; } \
        Vc_ALWAYS_INLINE Vc_PURE Vector operator symbol(const Vector &x) const { return Vector<T>(VectorHelper<T>::fun(data(), x.data())); }

        OP(+, add)
        OP(-, sub)
        OP(*, mul)
#undef OP
        inline Vector &operator/=(EntryType x);
        inline Vector &operator/=(VC_ALIGNED_PARAMETER(Vector) x);
        inline Vc_PURE_L Vector operator/ (VC_ALIGNED_PARAMETER(Vector) x) const Vc_PURE_R;

        // bitwise ops
#define OP_VEC(op) \
        Vc_ALWAYS_INLINE_L Vector<T> &operator op##=(AsArg x) Vc_ALWAYS_INLINE_R; \
        Vc_ALWAYS_INLINE_L Vc_PURE_L Vector<T>  operator op   (AsArg x) const Vc_ALWAYS_INLINE_R Vc_PURE_R;
        VC_ALL_BINARY(OP_VEC)
        VC_ALL_SHIFTS(OP_VEC)
#undef OP_VEC

        Vc_ALWAYS_INLINE_L Vector<T> &operator>>=(int x) Vc_ALWAYS_INLINE_R;
        Vc_ALWAYS_INLINE_L Vector<T> &operator<<=(int x) Vc_ALWAYS_INLINE_R;
        Vc_ALWAYS_INLINE_L Vector<T> operator>>(int x) const Vc_ALWAYS_INLINE_R;
        Vc_ALWAYS_INLINE_L Vector<T> operator<<(int x) const Vc_ALWAYS_INLINE_R;

#define OPcmp(symbol, fun) \
        Vc_ALWAYS_INLINE Vc_PURE Mask operator symbol(const Vector &x) const { return HT::fun(data(), x.data()); }

        OPcmp(==, cmpeq)
        OPcmp(!=, cmpneq)
        OPcmp(>=, cmpnlt)
        OPcmp(>, cmpnle)
        OPcmp(<, cmplt)
        OPcmp(<=, cmple)
#undef OPcmp
        Vc_INTRINSIC_L Vc_PURE_L Mask isNegative() const Vc_PURE_R Vc_INTRINSIC_R;

        Vc_ALWAYS_INLINE void fusedMultiplyAdd(const Vector<T> &factor, const Vector<T> &summand) {
            VectorHelper<T>::fma(data(), factor.data(), summand.data());
        }

        Vc_ALWAYS_INLINE void assign( const Vector<T> &v, const Mask &mask ) {
            const VectorType k = avx_cast<VectorType>(mask.data());
            data() = VectorHelper<VectorType>::blend(data(), v.data(), k);
        }

        template<typename V2> Vc_ALWAYS_INLINE V2 staticCast() const { return V2(*this); }
        template<typename V2> Vc_ALWAYS_INLINE V2 reinterpretCast() const { return avx_cast<typename V2::VectorType>(data()); }

        Vc_ALWAYS_INLINE WriteMaskedVector<T> operator()(const Mask &k) { return WriteMaskedVector<T>(this, k); }

        /**
         * \return \p true  This vector was completely filled. m2 might be 0 or != 0. You still have
         *                  to test this.
         *         \p false This vector was not completely filled. m2 is all 0.
         */
        //inline bool pack(Mask &m1, Vector<T> &v2, Mask &m2) {
            //return VectorHelper<T>::pack(data(), m1.data, v2.data(), m2.data);
        //}

        Vc_ALWAYS_INLINE VectorType &data() { return d.v(); }
        Vc_ALWAYS_INLINE const VectorType data() const { return d.v(); }

        Vc_ALWAYS_INLINE EntryType min() const { return VectorHelper<T>::min(data()); }
        Vc_ALWAYS_INLINE EntryType max() const { return VectorHelper<T>::max(data()); }
        Vc_ALWAYS_INLINE EntryType product() const { return VectorHelper<T>::mul(data()); }
        Vc_ALWAYS_INLINE EntryType sum() const { return VectorHelper<T>::add(data()); }
        Vc_ALWAYS_INLINE_L Vector partialSum() const Vc_ALWAYS_INLINE_R;
        //template<typename BinaryOperation> Vc_ALWAYS_INLINE_L Vector partialSum(BinaryOperation op) const Vc_ALWAYS_INLINE_R;
        Vc_ALWAYS_INLINE_L EntryType min(MaskArg m) const Vc_ALWAYS_INLINE_R;
        Vc_ALWAYS_INLINE_L EntryType max(MaskArg m) const Vc_ALWAYS_INLINE_R;
        Vc_ALWAYS_INLINE_L EntryType product(MaskArg m) const Vc_ALWAYS_INLINE_R;
        Vc_ALWAYS_INLINE_L EntryType sum(MaskArg m) const Vc_ALWAYS_INLINE_R;

        Vc_INTRINSIC_L Vector shifted(int amount, Vector shiftIn) const Vc_INTRINSIC_R;
        Vc_INTRINSIC_L Vector shifted(int amount) const Vc_INTRINSIC_R;
        Vc_INTRINSIC_L Vector rotated(int amount) const Vc_INTRINSIC_R;
        Vc_ALWAYS_INLINE Vector sorted() const { return SortHelper<T>::sort(data()); }

#ifdef VC_NO_MOVE_CTOR
        template<typename F> Vc_INTRINSIC void call(const F &f) const {
            for_all_vector_entries(i,
                    f(EntryType(d.m(i)));
                    );
        }
        template<typename F> Vc_INTRINSIC void call(const F &f, const Mask &mask) const {
            Vc_foreach_bit(size_t i, mask) {
                f(EntryType(d.m(i)));
            }
        }
        template<typename F> Vc_INTRINSIC Vector<T> apply(const F &f) const {
            Vector<T> r;
            for_all_vector_entries(i,
                    r.d.m(i) = f(EntryType(d.m(i)));
                    );
            return r;
        }
        template<typename F> Vc_INTRINSIC Vector<T> apply(const F &f, const Mask &mask) const {
            Vector<T> r(*this);
            Vc_foreach_bit (size_t i, mask) {
                r.d.m(i) = f(EntryType(r.d.m(i)));
            }
            return r;
        }
#endif

        template<typename F> void callWithValuesSorted(F VC_RR_ f) {
            EntryType value = d.m(0);
            f(value);
            for (size_t i = 1; i < Size; ++i) {
                if (d.m(i) != value) {
                    value = d.m(i);
                    f(value);
                }
            }
        }

        template<typename F> Vc_INTRINSIC void call(F VC_RR_ f) const {
            for_all_vector_entries(i,
                    f(EntryType(d.m(i)));
                    );
        }

        template<typename F> Vc_INTRINSIC void call(F VC_RR_ f, const Mask &mask) const {
            Vc_foreach_bit(size_t i, mask) {
                f(EntryType(d.m(i)));
            }
        }

        template<typename F> Vc_INTRINSIC Vector<T> apply(F VC_RR_ f) const {
            Vector<T> r;
            for_all_vector_entries(i,
                    r.d.m(i) = f(EntryType(d.m(i)));
                    );
            return r;
        }

        template<typename F> Vc_INTRINSIC Vector<T> apply(F VC_RR_ f, const Mask &mask) const {
            Vector<T> r(*this);
            Vc_foreach_bit (size_t i, mask) {
                r.d.m(i) = f(EntryType(r.d.m(i)));
            }
            return r;
        }

        template<typename IndexT> Vc_INTRINSIC void fill(EntryType (&f)(IndexT)) {
            for_all_vector_entries(i,
                    d.m(i) = f(i);
                    );
        }
        Vc_INTRINSIC void fill(EntryType (&f)()) {
            for_all_vector_entries(i,
                    d.m(i) = f();
                    );
        }

        Vc_INTRINSIC_L Vector copySign(AsArg reference) const Vc_INTRINSIC_R;
        Vc_INTRINSIC_L Vector exponent() const Vc_INTRINSIC_R;
};
template<typename T> constexpr size_t Vector<T>::Size;

typedef Vector<double>         double_v;
typedef Vector<float>          float_v;
typedef Vector<int>            int_v;
typedef Vector<unsigned int>   uint_v;
typedef Vector<short>          short_v;
typedef Vector<unsigned short> ushort_v;
typedef double_v::Mask double_m;
typedef  float_v::Mask float_m;
typedef    int_v::Mask int_m;
typedef   uint_v::Mask uint_m;
typedef  short_v::Mask short_m;
typedef ushort_v::Mask ushort_m;

template<typename T> class SwizzledVector : public Vector<T> {};

static Vc_ALWAYS_INLINE int_v    min(const int_v    &x, const int_v    &y) { return _mm256_min_epi32(x.data(), y.data()); }
static Vc_ALWAYS_INLINE uint_v   min(const uint_v   &x, const uint_v   &y) { return _mm256_min_epu32(x.data(), y.data()); }
static Vc_ALWAYS_INLINE short_v  min(const short_v  &x, const short_v  &y) { return _mm_min_epi16(x.data(), y.data()); }
static Vc_ALWAYS_INLINE ushort_v min(const ushort_v &x, const ushort_v &y) { return _mm_min_epu16(x.data(), y.data()); }
static Vc_ALWAYS_INLINE float_v  min(const float_v  &x, const float_v  &y) { return _mm256_min_ps(x.data(), y.data()); }
static Vc_ALWAYS_INLINE double_v min(const double_v &x, const double_v &y) { return _mm256_min_pd(x.data(), y.data()); }
static Vc_ALWAYS_INLINE int_v    max(const int_v    &x, const int_v    &y) { return _mm256_max_epi32(x.data(), y.data()); }
static Vc_ALWAYS_INLINE uint_v   max(const uint_v   &x, const uint_v   &y) { return _mm256_max_epu32(x.data(), y.data()); }
static Vc_ALWAYS_INLINE short_v  max(const short_v  &x, const short_v  &y) { return _mm_max_epi16(x.data(), y.data()); }
static Vc_ALWAYS_INLINE ushort_v max(const ushort_v &x, const ushort_v &y) { return _mm_max_epu16(x.data(), y.data()); }
static Vc_ALWAYS_INLINE float_v  max(const float_v  &x, const float_v  &y) { return _mm256_max_ps(x.data(), y.data()); }
static Vc_ALWAYS_INLINE double_v max(const double_v &x, const double_v &y) { return _mm256_max_pd(x.data(), y.data()); }

  template<typename T> static Vc_ALWAYS_INLINE Vector<T> sqrt (const Vector<T> &x) { return VectorHelper<T>::sqrt(x.data()); }
  template<typename T> static Vc_ALWAYS_INLINE Vector<T> rsqrt(const Vector<T> &x) { return VectorHelper<T>::rsqrt(x.data()); }
  template<typename T> static Vc_ALWAYS_INLINE Vector<T> abs  (const Vector<T> &x) { return VectorHelper<T>::abs(x.data()); }
  template<typename T> static Vc_ALWAYS_INLINE Vector<T> reciprocal(const Vector<T> &x) { return VectorHelper<T>::reciprocal(x.data()); }
  template<typename T> static Vc_ALWAYS_INLINE Vector<T> round(const Vector<T> &x) { return VectorHelper<T>::round(x.data()); }

  template<typename T> static Vc_ALWAYS_INLINE typename Vector<T>::Mask isfinite(const Vector<T> &x) { return VectorHelper<T>::isFinite(x.data()); }
  template<typename T> static Vc_ALWAYS_INLINE typename Vector<T>::Mask isinf(const Vector<T> &x) { return VectorHelper<T>::isInfinite(x.data()); }
  template<typename T> static Vc_ALWAYS_INLINE typename Vector<T>::Mask isnan(const Vector<T> &x) { return VectorHelper<T>::isNaN(x.data()); }

static_assert(!std::is_convertible<float *, short_v>::value, "A float* should never implicitly convert to short_v. Something is broken.");
static_assert(!std::is_convertible<int *  , short_v>::value, "An int* should never implicitly convert to short_v. Something is broken.");
static_assert(!std::is_convertible<short *, short_v>::value, "A short* should never implicitly convert to short_v. Something is broken.");

#include "forceToRegisters.tcc"
Vc_IMPL_NAMESPACE_END

#include "vector.tcc"
#include "math.h"
#include "undomacros.h"

#endif // AVX_VECTOR_H
