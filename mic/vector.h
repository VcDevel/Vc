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
    static constexpr size_t Size = sizeof(VectorType) / sizeof(VectorEntryType);
    enum Constants {
        MemoryAlignment = sizeof(EntryType) * Size,
        HasVectorDivision = true
    };
    typedef Vc_IMPL_NAMESPACE::Mask<T> Mask;
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
    Vc_INTRINSIC Vector(VectorType x) : d(x) {}

    ///////////////////////////////////////////////////////////////////////////////////////////
    // copy
    Vc_INTRINSIC Vector(const Vector &x) = default; //: d(x.data()) {}
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
    Vc_INTRINSIC Vector(EntryType a) : d(_set1(a)) {}
    template <typename U>
    Vc_INTRINSIC Vector(
        U a,
        typename std::enable_if<std::is_same<U, int>::value && !std::is_same<U, EntryType>::value,
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
    // expand 1 float_v to 2 double_v                 XXX rationale? remove it for release? XXX
    // TODO: handle 8 <-> 16 conversions
    explicit Vc_ALWAYS_INLINE_L Vc_FLATTEN Vector(const Vector<typename ConcatTypeHelper<T>::Type> *a) Vc_ALWAYS_INLINE_R;
    Vc_ALWAYS_INLINE_L Vc_FLATTEN void expand(Vector<typename ConcatTypeHelper<T>::Type> *x) const Vc_ALWAYS_INLINE_R;

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
    Vc_INTRINSIC_L Vc_CONST_L       Vector<T>  cdab() const Vc_INTRINSIC_R Vc_CONST_R;
    Vc_INTRINSIC_L Vc_CONST_L       Vector<T>  badc() const Vc_INTRINSIC_R Vc_CONST_R;
    Vc_INTRINSIC_L Vc_CONST_L       Vector<T>  aaaa() const Vc_INTRINSIC_R Vc_CONST_R;
    Vc_INTRINSIC_L Vc_CONST_L       Vector<T>  bbbb() const Vc_INTRINSIC_R Vc_CONST_R;
    Vc_INTRINSIC_L Vc_CONST_L       Vector<T>  cccc() const Vc_INTRINSIC_R Vc_CONST_R;
    Vc_INTRINSIC_L Vc_CONST_L       Vector<T>  dddd() const Vc_INTRINSIC_R Vc_CONST_R;
    Vc_INTRINSIC_L Vc_CONST_L       Vector<T>  bcad() const Vc_INTRINSIC_R Vc_CONST_R;
    Vc_INTRINSIC_L Vc_CONST_L       Vector<T>  bcda() const Vc_INTRINSIC_R Vc_CONST_R;
    Vc_INTRINSIC_L Vc_CONST_L       Vector<T>  dabc() const Vc_INTRINSIC_R Vc_CONST_R;
    Vc_INTRINSIC_L Vc_CONST_L       Vector<T>  acbd() const Vc_INTRINSIC_R Vc_CONST_R;
    Vc_INTRINSIC_L Vc_CONST_L       Vector<T>  dbca() const Vc_INTRINSIC_R Vc_CONST_R;
    Vc_INTRINSIC_L Vc_CONST_L       Vector<T>  dcba() const Vc_INTRINSIC_R Vc_CONST_R;

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
    Vc_ALWAYS_INLINE Vector &operator++() { d.v() = _add<VectorEntryType>(d.v(), HV::one()); return *this; }
    //postfix
    Vc_ALWAYS_INLINE Vector operator++(int) { const Vector<T> r = *this; d.v() = _add<VectorEntryType>(d.v(), HV::one()); return r; }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // aliasing scalar access
    Vc_INTRINSIC decltype(d.m(0)) &operator[](size_t index) {
        return d.m(index);
    }
    Vc_ALWAYS_INLINE EntryType operator[](size_t index) const {
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
    Vc_ALWAYS_INLINE Vector operator symbol(AsArg x) const { return Vector<T>(fun(d.v(), x.d.v())); }

    Vc_OP(*, _mul<VectorEntryType>)
    Vc_OP(+, _add<VectorEntryType>)
    Vc_OP(-, _sub<VectorEntryType>)
    Vc_OP(/, _div<VectorEntryType>) // ushort_v::operator/ is specialized in vector.tcc
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
    Vc_ALWAYS_INLINE Mask operator symbol(AsArg x) const { return HT::fun(d.v(), x.d.v()); }

    // ushort_v specializations are in vector.tcc!
    OPcmp(==, cmpeq)
    OPcmp(!=, cmpneq)
    OPcmp(>=, cmpnlt)
    OPcmp(>, cmpnle)
    OPcmp(<, cmplt)
    OPcmp(<=, cmple)
#undef OPcmp
    Vc_INTRINSIC_L Vc_PURE_L Mask isNegative() const Vc_INTRINSIC_R Vc_PURE_R;

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

    Vc_INTRINSIC_L Vector shifted(int amount, Vector shiftIn) const Vc_INTRINSIC_R;
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
    template<typename F> Vc_INTRINSIC Vector<T> apply(const F &&f, const Mask &mask) const {
        Vector<T> r(*this);
        Vc_foreach_bit (size_t i, mask) {
            r.d.m(i) = f(EntryType(r.d.m(i)));
        }
        return r;
    }

    template<typename F> Vc_INTRINSIC void call(F &f) const {
        for_all_vector_entries(i,
                f(EntryType(d.m(i)));
                );
    }
    template<typename F> Vc_INTRINSIC void call(F &f, const Mask &mask) const {
        Vc_foreach_bit(size_t i, mask) {
            f(EntryType(d.m(i)));
        }
    }
    template<typename F> Vc_INTRINSIC Vector<T> apply(F &f) const {
        Vector<T> r;
        for_all_vector_entries(i,
                r.d.m(i) = f(EntryType(d.m(i)));
                );
        return r;
    }
    template<typename F> Vc_INTRINSIC Vector<T> apply(F &f, const Mask &mask) const {
        Vector<T> r(*this);
        Vc_foreach_bit (size_t i, mask) {
            r.d.m(i) = f(EntryType(r.d.m(i)));
        }
        return r;
    }
#else
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
#endif

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

template<typename T> struct SwizzledVector
{
    Vector<T> v;
    unsigned int s;
};

#define MATH_OP2(name, call) \
    template<typename T> static inline Vector<T> name(Vector<T> x, Vector<T> y) \
    { \
        typedef VectorHelper<typename Vector<T>::VectorEntryType> HT; \
        return HT::call(x.data(), y.data()); \
    }
    MATH_OP2(min, min)
    MATH_OP2(max, max)
    MATH_OP2(atan2, atan2)
#undef MATH_OP2

#define MATH_OP1(name, call) \
    template<typename T> static Vc_ALWAYS_INLINE Vector<T> name(const Vector<T> &x) \
    { \
        typedef VectorHelper<typename Vector<T>::VectorEntryType> HT; \
        return HT::call(x.data()); \
    }
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

    template<typename T> static inline void sincos(const Vector<T> &x, Vector<T> *sin, Vector<T> *cos) {
        typedef VectorHelper<typename Vector<T>::VectorEntryType> HT; \
        *sin = HT::sin(x.data());
        *cos = HT::cos(x.data());
    }

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
