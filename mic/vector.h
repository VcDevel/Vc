/*  This file is part of the Vc library. {{{

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

}}}*/

#ifndef VC_MIC_VECTOR_H
#define VC_MIC_VECTOR_H

#ifdef CAN_OFFLOAD
#pragma offload_attribute(push, target(mic))
#endif

#include "types.h"
#include "intrinsics.h"
#include "casts.h"
#include "../common/storage.h"
#include "mask.h"
#include "vectorhelper.h"
#include "storemixin.h"
//#include "vectormultiplication.h"
#include "writemaskedvector.h"
#include "sorthelper.h"
#include "macros.h"

#ifdef isfinite
#undef isfinite
#endif
#ifdef isnan
#undef isnan
#endif

#define VC_HAVE_FMA

Vc_NAMESPACE_BEGIN(Vc_IMPL_NAMESPACE)

template<typename T> class Vector : public StoreMixin<Vector<T>, T>
{
    //friend class VectorMultiplication<T>;
    friend class WriteMaskedVector<T>;
    //friend Vector<T> operator+<>(const Vector<T> &x, const VectorMultiplication<T> &y);
    //friend Vector<T> operator-<>(const Vector<T> &x, const VectorMultiplication<T> &y);
    friend class Vector<float>;
    friend class Vector<double>;
    friend class Vector<int>;
    friend class Vector<unsigned int>;
    friend class StoreMixin<Vector<T>, T>;

public:
    FREE_STORE_OPERATORS_ALIGNED(64)
    typedef typename VectorTypeHelper<T>::Type VectorType;
    typedef typename DetermineEntryType<T>::Type EntryType;
    typedef typename DetermineVectorEntryType<T>::Type VectorEntryType;
    typedef Vector<unsigned int> IndexType;
    enum Constants {
        Size = sizeof(VectorType) / sizeof(VectorEntryType),
        MemoryAlignment = sizeof(EntryType) * Size,
        HasVectorDivision = true
    };
    typedef Vc_IMPL_NAMESPACE::Mask<Size> Mask;
    typedef typename Mask::AsArg MaskArg;
    typedef Vc::Memory<Vector<T>, Size> Memory;
    typedef Vector<T> AsArg; // for now only ICC can compile this code and it is not broken :)
    typedef VectorType VectorTypeArg;

    inline const VectorType data() const { return d.v(); }
    inline VectorType &data() { return d.v(); }

protected:
    // helper that specializes on VectorType
    typedef VectorHelper<VectorType> HV;

    // helper that specializes on T
    typedef VectorHelper<VectorEntryType> HT;

    typedef Common::VectorMemoryUnion<VectorType, VectorEntryType> StorageType;
    StorageType d;
    VC_DEPRECATED("renamed to data()") inline const VectorType vdata() const { return d.v(); }

    template<typename MemType> using UpDownC = UpDownConversion<VectorEntryType, typename std::remove_cv<MemType>::type>;

    template<typename V> static Vc_INTRINSIC VectorType _cast(VC_ALIGNED_PARAMETER(V) x) { return mic_cast<VectorType>(x); }

public:

    /**
     * Reinterpret some array of T as a vector of T. You may only do this if the pointer is
     * aligned correctly and the content of the memory isn't changed from somewhere else because
     * the load operation will happen implicitly at some later point(s).
     */
    static inline Vector fromMemory(T *mem) {
        assert(0 == (mem & (VectorAlignment - 1)));
        return reinterpret_cast<Vector<T> >(mem);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // uninitialized
    inline Vector() {}

    ///////////////////////////////////////////////////////////////////////////////////////////
    // constants
    inline explicit Vector(VectorSpecialInitializerZero::ZEnum);
    inline explicit Vector(VectorSpecialInitializerOne::OEnum);
    inline explicit Vector(VectorSpecialInitializerIndexesFromZero::IEnum);
    static inline Vector Zero();
    static inline Vector One();
    static inline Vector IndexesFromZero();
    static Vector Random();

    ///////////////////////////////////////////////////////////////////////////////////////////
    // internal: required to enable returning objects of VectorType
    inline Vector(VectorType x) : d(x) {}

    ///////////////////////////////////////////////////////////////////////////////////////////
    // static_cast / copy ctor
    template<typename OtherT> explicit inline Vector(Vector<OtherT> x);
    //template<typename OtherT> explicit inline Vector(VectorMultiplication<OtherT> x);

    // implicit cast
    template<typename OtherT> Vc_INTRINSIC_L Vector &operator=(const Vector<OtherT> &x) Vc_INTRINSIC_R;

    // copy assignment
    inline Vector &operator=(AsArg v) { d.v() = v.d.v(); return *this; }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // broadcast
    explicit Vector(EntryType a);
    template<typename TT> Vc_INTRINSIC Vector(TT x, VC_EXACT_TYPE(TT, EntryType, void *) = 0) : d(HT::set(x)) {}
    inline Vector &operator=(EntryType a) { d.v() = HT::set(a); return *this; }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // load ctors
    explicit Vc_INTRINSIC_L
        Vector(const VectorEntryType *x) Vc_INTRINSIC_R;
    template<typename Alignment> Vc_INTRINSIC_L
        Vector(const VectorEntryType *x, Alignment align) Vc_INTRINSIC_R;
    template<typename OtherT> explicit Vc_INTRINSIC_L
        Vector(const OtherT    *x) Vc_INTRINSIC_R;
    template<typename OtherT, typename Alignment> Vc_INTRINSIC_L
        Vector(const OtherT    *x, Alignment align) Vc_INTRINSIC_R;

    ///////////////////////////////////////////////////////////////////////////////////////////
    // load member functions
    Vc_INTRINSIC_L
        void load(const VectorEntryType *mem) Vc_INTRINSIC_R;
    template<typename Alignment> Vc_INTRINSIC_L
        void load(const VectorEntryType *mem, Alignment align) Vc_INTRINSIC_R;
    template<typename OtherT> Vc_INTRINSIC_L
        void load(const OtherT    *mem) Vc_INTRINSIC_R;
    template<typename OtherT, typename Alignment> Vc_INTRINSIC_L
        void load(const OtherT    *mem, Alignment align) Vc_INTRINSIC_R;

    ///////////////////////////////////////////////////////////////////////////////////////////
    // expand 1 float_v to 2 double_v                 XXX rationale? remove it for release? XXX
    // TODO: handle 8 <-> 16 conversions
    explicit Vc_ALWAYS_INLINE_L Vc_FLATTEN Vector(const Vector<typename ConcatTypeHelper<T>::Type> *a) Vc_ALWAYS_INLINE_R;
    Vc_ALWAYS_INLINE_R Vc_FLATTEN void expand(Vector<typename ConcatTypeHelper<T>::Type> *x) const Vc_ALWAYS_INLINE_R;

    ///////////////////////////////////////////////////////////////////////////////////////////
    // zeroing
    inline void setZero();
    inline void setZero(MaskArg k);

    inline void setQnan();
    inline void setQnan(MaskArg k);

    ///////////////////////////////////////////////////////////////////////////////////////////
    // stores in StoreMixin

    ///////////////////////////////////////////////////////////////////////////////////////////
    // swizzles
    Vc_INTRINSIC_L Vc_CONST_L const Vector<T> &abcd() const Vc_INTRINSIC_R Vc_CONST_R;
    Vc_INTRINSIC_L Vc_CONST_L const Vector<T>  cdab() const Vc_INTRINSIC_R Vc_CONST_R;
    Vc_INTRINSIC_L Vc_CONST_L const Vector<T>  badc() const Vc_INTRINSIC_R Vc_CONST_R;
    Vc_INTRINSIC_L Vc_CONST_L const Vector<T>  aaaa() const Vc_INTRINSIC_R Vc_CONST_R;
    Vc_INTRINSIC_L Vc_CONST_L const Vector<T>  bbbb() const Vc_INTRINSIC_R Vc_CONST_R;
    Vc_INTRINSIC_L Vc_CONST_L const Vector<T>  cccc() const Vc_INTRINSIC_R Vc_CONST_R;
    Vc_INTRINSIC_L Vc_CONST_L const Vector<T>  dddd() const Vc_INTRINSIC_R Vc_CONST_R;
    Vc_INTRINSIC_L Vc_CONST_L const Vector<T>  bcad() const Vc_INTRINSIC_R Vc_CONST_R;
    Vc_INTRINSIC_L Vc_CONST_L const Vector<T>  bcda() const Vc_INTRINSIC_R Vc_CONST_R;
    Vc_INTRINSIC_L Vc_CONST_L const Vector<T>  dabc() const Vc_INTRINSIC_R Vc_CONST_R;
    Vc_INTRINSIC_L Vc_CONST_L const Vector<T>  acbd() const Vc_INTRINSIC_R Vc_CONST_R;
    Vc_INTRINSIC_L Vc_CONST_L const Vector<T>  dbca() const Vc_INTRINSIC_R Vc_CONST_R;
    Vc_INTRINSIC_L Vc_CONST_L const Vector<T>  dcba() const Vc_INTRINSIC_R Vc_CONST_R;

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
    Vc_ALWAYS_INLINE Vector &operator++() { d.v() = _add(d.v(), HV::one()); return *this; }
    //postfix
    Vc_ALWAYS_INLINE Vector operator++(int) { const Vector<T> r = *this; d.v() = _add(d.v(), HV::one()); return r; }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // aliasing scalar access
    Vc_INTRINSIC decltype(d.m(0)) &operator[](int index) {
        return d.m(index);
    }
    Vc_ALWAYS_INLINE EntryType operator[](int index) const {
        return d.m(index);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // unary operators
    Vc_PURE Vc_ALWAYS_INLINE Vc_FLATTEN Vector operator~() const { return _andnot(d.v(), _setallone<VectorType>()); }
    Vc_PURE Vc_ALWAYS_INLINE Vc_FLATTEN Vector<typename NegateTypeHelper<T>::Type> operator-() const;
    Vc_PURE Vc_ALWAYS_INLINE Vc_FLATTEN Vector operator+() const { return *this; }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // binary operators
#define Vc_OP(symbol, fun) \
    Vc_ALWAYS_INLINE Vector &operator symbol##=(AsArg x) { d.v() = fun(d.v(), x.d.v()); return *this; } \
    Vc_ALWAYS_INLINE Vector &operator symbol##=(EntryType x) { return operator symbol##=(Vector(x)); } \
    Vc_ALWAYS_INLINE Vector operator symbol(AsArg x) const { return Vector<T>(fun(d.v(), x.d.v())); } \
    template<typename TT> Vc_ALWAYS_INLINE VC_EXACT_TYPE(TT, EntryType, Vector) operator symbol(TT x) const { return operator symbol(Vector(x)); }

    Vc_OP(*, _mul)
    Vc_OP(+, _add)
    Vc_OP(-, _sub)
    Vc_OP(/, _div)
    Vc_OP(|, _or)
    Vc_OP(&, _and)
    Vc_OP(^, _xor)
#undef Vc_OP

    Vc_ALWAYS_INLINE_L Vector &operator<<=(AsArg x) Vc_ALWAYS_INLINE_R;
    Vc_ALWAYS_INLINE_L Vector &operator>>=(AsArg x) Vc_ALWAYS_INLINE_R;
    Vc_ALWAYS_INLINE_L Vector  operator<< (AsArg x) const Vc_ALWAYS_INLINE_R;
    Vc_ALWAYS_INLINE_L Vector  operator>> (AsArg x) const Vc_ALWAYS_INLINE_R;

    Vc_ALWAYS_INLINE_L Vector &operator<<=(unsigned int x) Vc_ALWAYS_INLINE_R;
    Vc_ALWAYS_INLINE_L Vector &operator>>=(unsigned int x) Vc_ALWAYS_INLINE_R;
    Vc_ALWAYS_INLINE_L Vector  operator<< (unsigned int x) const Vc_ALWAYS_INLINE_R;
    Vc_ALWAYS_INLINE_L Vector  operator>> (unsigned int x) const Vc_ALWAYS_INLINE_R;

#define OPcmp(symbol, fun) \
    Vc_ALWAYS_INLINE Mask operator symbol(AsArg x) const { return HT::fun(d.v(), x.d.v()); } \
    template<typename TT> Vc_ALWAYS_INLINE VC_EXACT_TYPE(TT, EntryType, Mask) operator symbol(TT x) const { return operator symbol(Vector(x)); }

    // ushort_v specializations are in vector.tcc!
    OPcmp(==, cmpeq)
    OPcmp(!=, cmpneq)
    OPcmp(>=, cmpnlt)
    OPcmp(>, cmpnle)
    OPcmp(<, cmplt)
    OPcmp(<=, cmple)
#undef OPcmp

    Vc_INTRINSIC void fusedMultiplyAdd(const Vector<T> &factor, const Vector<T> &summand) {
        d.v() = HT::multiplyAndAdd(d.v(), factor.data(), summand.data());
    }

    Vc_INTRINSIC_L void assign(Vector<T> v, Mask mask) Vc_INTRINSIC_R;

    template<typename V2> Vc_INTRINSIC V2 staticCast() const { return V2(*this); }
    template<typename V2> Vc_INTRINSIC V2 reinterpretCast() const { return mic_cast<typename V2::VectorType>(d.v()); }

    Vc_ALWAYS_INLINE WriteMaskedVector<T> operator()(MaskArg k) { return WriteMaskedVector<T>(this, k); }

    inline EntryType min() const { return HT::reduce_min(d.v()); }
    inline EntryType max() const { return HT::reduce_max(d.v()); }
    inline EntryType product() const { return HT::reduce_mul(d.v()); }
    inline EntryType sum() const { return HT::reduce_add(d.v()); }
    Vc_ALWAYS_INLINE_L Vector partialSum() const Vc_ALWAYS_INLINE_R;
    inline EntryType min(MaskArg m) const;
    inline EntryType max(MaskArg m) const;
    inline EntryType product(MaskArg m) const;
    inline EntryType sum(MaskArg m) const;

    Vc_INTRINSIC_L Vector shifted(int amount) const Vc_INTRINSIC_R;
    Vc_INTRINSIC_L Vector rotated(int amount) const Vc_INTRINSIC_R;
    Vc_INTRINSIC Vector sorted() const { return SortHelper<T>::sort(d.v()); }

    template<typename F> void callWithValuesSorted(F &f) {
        EntryType value = d.m(0);
        f(value);
        for (int i = 1; i < Size; ++i) {
            if (EntryType(d.m(i)) != value) {
                value = d.m(i);
                f(value);
            }
        }
    }

    template<typename F> Vc_INTRINSIC void call(F &&f) const {
        for_all_vector_entries(i,
                f(EntryType(d.m(i)));
                );
    }

    template<typename F> Vc_INTRINSIC void call(F &&f, const Mask &mask) const {
        Vc_foreach_bit(size_t i, mask) {
            f(EntryType(d.m(i)));
        }
    }

    template<typename F> Vc_INTRINSIC Vector<T> apply(F &&f) const {
        Vector<T> r;
        for_all_vector_entries(i,
                r.d.m(i) = f(EntryType(d.m(i)));
                );
        return r;
    }

    template<typename F> Vc_INTRINSIC Vector<T> apply(F &&f, const Mask &mask) const {
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

template<typename T> struct SwizzledVector
{
    Vector<T> v;
    unsigned int s;
};

#define MATH_OP2(name, call) \
    template<typename T> static inline Vector<T> name(Vector<T> x, Vector<T> y) \
    { return HT::call(x.data(), y.data()); } \
    /*template<typename T> static inline Vector<T> name(Vector<T> x, VectorMultiplication<T> y) \
    { return HT::call(x.data(), y.data()); } \
    template<typename T> static inline Vector<T> name(VectorMultiplication<T> x, Vector<T> y) \
    { return HT::call(x.data(), y.data()); } \
    template<typename T> static inline Vector<T> name(VectorMultiplication<T> x, VectorMultiplication<T> y) \
    { return HT::call(x.data(), y.data()); }*/
    MATH_OP2(min, min)
    MATH_OP2(max, max)
    MATH_OP2(atan2, atan2)
#undef MATH_OP2

#define MATH_OP1(name, call) \
    template<typename T> static inline Vector<T> name(const Vector<T> &x)               { return VectorHelper<T>::call(x.data()); } \
    /*template<typename T> static inline Vector<T> name(const VectorMultiplication<T> &x) { return VectorHelper<T>::call(x.data()); }*/
    MATH_OP1(sqrt, sqrt)
    MATH_OP1(rsqrt, rsqrt)
    MATH_OP1(abs, abs)
    MATH_OP1(sin, sin)
    MATH_OP1(cos, cos)
    MATH_OP1(log, log)
    MATH_OP1(log2, log2)
    MATH_OP1(log10, log10)
    MATH_OP1(atan, atan)
    MATH_OP1(reciprocal, recip)
    MATH_OP1(round, round)
    MATH_OP1(asin, asin)
    MATH_OP1(floor, floor)
    MATH_OP1(ceil, ceil)
    MATH_OP1(exp, exp)
#undef MATH_OP1

    // TODO: implement the following:
    inline double_v frexp(double_v::AsArg v, int_v *e) { return v; }
    inline float_v frexp(float_v::AsArg v, int_v *e) { return v; }
    inline sfloat_v frexp(sfloat_v::AsArg v, short_v *e) { return v; }
    inline double_v ldexp(double_v::AsArg v, int_v::AsArg _e) { return v; }
    inline float_v ldexp(float_v::AsArg v, int_v::AsArg _e) { return v; }
    inline sfloat_v ldexp(sfloat_v::AsArg v, short_v::AsArg _e) { return v; }
    template<typename T> static inline void sincos(const Vector<T> &x, Vector<T> *sin, Vector<T> *cos) {
        VectorHelper<T>::sincos(x.data(), sin->data(), cos->data());
    }

    template<typename T> static inline Mask<Vector<T>::Size> isfinite(const Vector<T> &x) { return VectorHelper<T>::isFinite(x.data()); }
    //template<typename T> static inline Mask<Vector<T>::Size> isfinite(const VectorMultiplication<T> &x) { return VectorHelper<T>::isFinite(x.data()); }
    template<typename T> static inline Mask<Vector<T>::Size> isnan(const Vector<T> &x) { return VectorHelper<T>::isNaN(x.data()); }
    //template<typename T> static inline Mask<Vector<T>::Size> isnan(const VectorMultiplication<T> &x) { return VectorHelper<T>::isNaN(x.data()); }

#include "forcetoregisters.tcc"

Vc_NAMESPACE_END

#include "vector.tcc"
#include "undomacros.h"
#include "helperimpl.h"
#include "math.h"

#ifdef CAN_OFFLOAD
#pragma offload_attribute(pop)
#endif

#endif // VC_MIC_VECTOR_H
