/*  This file is part of the Vc library.

    Copyright (C) 2010-2011 Matthias Kretz <kretz@kde.org>

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

#include "limits.h"
#include "../common/bitscanintrinsics.h"
#include "macros.h"

namespace Vc
{
namespace SSE
{

///////////////////////////////////////////////////////////////////////////////////////////
// constants {{{1
template<typename T> inline Vector<T>::Vector(VectorSpecialInitializerZero::ZEnum)
    : Base(VectorHelper<VectorType>::zero())
{
}

template<typename T> inline Vector<T>::Vector(VectorSpecialInitializerOne::OEnum)
    : Base(VectorHelper<T>::one())
{
}

template<typename T> inline Vector<T>::Vector(VectorSpecialInitializerIndexesFromZero::IEnum)
    : Base(VectorHelper<VectorType>::load(Base::_IndexesFromZero(), Aligned))
{
}

template<typename T> inline Vector<T> Vector<T>::Zero()
{
    return VectorHelper<VectorType>::zero();
}

template<typename T> inline Vector<T> Vector<T>::IndexesFromZero()
{
    return VectorHelper<VectorType>::load(Base::_IndexesFromZero(), Aligned);
}

template<typename T> template<typename OtherT> inline Vector<T>::Vector(const Vector<OtherT> &x)
    : Base(StaticCastHelper<OtherT, T>::cast(x.data()))
{
}

template<typename T> inline Vector<T>::Vector(EntryType a)
    : Base(VectorHelper<T>::set(a))
{
}

///////////////////////////////////////////////////////////////////////////////////////////
// load ctors {{{1
template<typename T> inline Vector<T>::Vector(const EntryType *x)
    : Base(VectorHelper<VectorType>::load(x, Aligned))
{
}

template<typename T> template<typename A> inline Vector<T>::Vector(const EntryType *x, A align)
    : Base(VectorHelper<VectorType>::load(x, align))
{
}

template<typename T> inline Vector<T>::Vector(const Vector<typename CtorTypeHelper<T>::Type> *a)
    : Base(VectorHelper<T>::concat(a[0].data(), a[1].data()))
{
}

template<typename T> inline void Vector<T>::expand(Vector<typename ExpandTypeHelper<T>::Type> *x) const
{
    if (Size == 8u) {
        x[0].data() = VectorHelper<T>::expand0(data());
        x[1].data() = VectorHelper<T>::expand1(data());
    }
}

///////////////////////////////////////////////////////////////////////////////////////////
// load member functions {{{1
template<typename T> inline void Vector<T>::load(const EntryType *mem)
{
    data() = VectorHelper<VectorType>::load(mem, Aligned);
}

template<typename T> template<typename A> inline void Vector<T>::load(const EntryType *mem, A align)
{
    data() = VectorHelper<VectorType>::load(mem, align);
}

///////////////////////////////////////////////////////////////////////////////////////////
// zeroing {{{1
template<typename T> inline void Vector<T>::setZero()
{
    data() = VectorHelper<VectorType>::zero();
}

template<typename T> inline void Vector<T>::setZero(const Mask &k)
{
    data() = VectorHelper<VectorType>::andnot_(mm128_reinterpret_cast<VectorType>(k.data()), data());
}

///////////////////////////////////////////////////////////////////////////////////////////
// stores {{{1
template<typename T> inline void Vector<T>::store(EntryType *mem) const
{
    VectorHelper<VectorType>::store(mem, data(), Aligned);
}

template<typename T> inline void Vector<T>::store(EntryType *mem, const Mask &mask) const
{
    VectorHelper<VectorType>::store(mem, data(), mm128_reinterpret_cast<VectorType>(mask.data()), Aligned);
}

template<typename T> template<typename A> inline void Vector<T>::store(EntryType *mem, A align) const
{
    VectorHelper<VectorType>::store(mem, data(), align);
}

template<typename T> template<typename A> inline void Vector<T>::store(EntryType *mem, const Mask &mask, A align) const
{
    HV::store(mem, data(), mm128_reinterpret_cast<VectorType>(mask.data()), align);
}

///////////////////////////////////////////////////////////////////////////////////////////
// division {{{1
template<typename T> inline Vector<T> &Vector<T>::operator/=(EntryType x)
{
    if (Base::HasVectorDivision) {
        return operator/=(Vector<T>(x));
    }
    for_all_vector_entries(i,
            d.m(i) /= x;
            );
    return *this;
}

template<typename T> inline Vector<T> Vector<T>::operator/(EntryType x) const
{
    if (Base::HasVectorDivision) {
        return operator/(Vector<T>(x));
    }
    Vector<T> r;
    for_all_vector_entries(i,
            r.d.m(i) = d.m(i) / x;
            );
    return r;
}

template<typename T> inline Vector<T> &Vector<T>::operator/=(const Vector<T> &x)
{
    for_all_vector_entries(i,
            d.m(i) /= x.d.m(i);
            );
    return *this;
}

template<typename T> inline Vector<T> Vector<T>::operator/(const Vector<T> &x) const
{
    Vector<T> r;
    for_all_vector_entries(i,
            r.d.m(i) = d.m(i) / x.d.m(i);
            );
    return r;
}

template<> inline Vector<short> &Vector<short>::operator/=(const Vector<short> &x)
{
    __m128 lo = _mm_cvtepi32_ps(VectorHelper<short>::expand0(d.v()));
    __m128 hi = _mm_cvtepi32_ps(VectorHelper<short>::expand1(d.v()));
    lo = _mm_div_ps(lo, _mm_cvtepi32_ps(VectorHelper<short>::expand0(x.d.v())));
    hi = _mm_div_ps(hi, _mm_cvtepi32_ps(VectorHelper<short>::expand1(x.d.v())));
    d.v() = _mm_packs_epi32(_mm_cvtps_epi32(lo), _mm_cvtps_epi32(hi));
    return *this;
}

template<> inline Vector<short> ALWAYS_INLINE Vector<short>::operator/(const Vector<short> &x) const
{
    __m128 lo = _mm_cvtepi32_ps(VectorHelper<short>::expand0(d.v()));
    __m128 hi = _mm_cvtepi32_ps(VectorHelper<short>::expand1(d.v()));
    lo = _mm_div_ps(lo, _mm_cvtepi32_ps(VectorHelper<short>::expand0(x.d.v())));
    hi = _mm_div_ps(hi, _mm_cvtepi32_ps(VectorHelper<short>::expand1(x.d.v())));
    return _mm_packs_epi32(_mm_cvtps_epi32(lo), _mm_cvtps_epi32(hi));
}

template<> inline Vector<unsigned short> &Vector<unsigned short>::operator/=(const Vector<unsigned short> &x)
{
    __m128 lo = _mm_cvtepi32_ps(VectorHelper<short>::expand0(d.v()));
    __m128 hi = _mm_cvtepi32_ps(VectorHelper<short>::expand1(d.v()));
    lo = _mm_div_ps(lo, _mm_cvtepi32_ps(VectorHelper<short>::expand0(x.d.v())));
    hi = _mm_div_ps(hi, _mm_cvtepi32_ps(VectorHelper<short>::expand1(x.d.v())));
    d.v() = _mm_packs_epi32(_mm_cvtps_epi32(lo), _mm_cvtps_epi32(hi));
    return *this;
}

template<> inline Vector<unsigned short> ALWAYS_INLINE Vector<unsigned short>::operator/(const Vector<unsigned short> &x) const
{
    __m128 lo = _mm_cvtepi32_ps(VectorHelper<short>::expand0(d.v()));
    __m128 hi = _mm_cvtepi32_ps(VectorHelper<short>::expand1(d.v()));
    lo = _mm_div_ps(lo, _mm_cvtepi32_ps(VectorHelper<short>::expand0(x.d.v())));
    hi = _mm_div_ps(hi, _mm_cvtepi32_ps(VectorHelper<short>::expand1(x.d.v())));
    return _mm_packs_epi32(_mm_cvtps_epi32(lo), _mm_cvtps_epi32(hi));
}

template<> inline Vector<float> &Vector<float>::operator/=(const Vector<float> &x)
{
    d.v() = _mm_div_ps(d.v(), x.d.v());
    return *this;
}

template<> inline Vector<float> Vector<float>::operator/(const Vector<float> &x) const
{
    return _mm_div_ps(d.v(), x.d.v());
}

template<> inline Vector<float8> &Vector<float8>::operator/=(const Vector<float8> &x)
{
    d.v()[0] = _mm_div_ps(d.v()[0], x.d.v()[0]);
    d.v()[1] = _mm_div_ps(d.v()[1], x.d.v()[1]);
    return *this;
}

template<> inline Vector<float8> Vector<float8>::operator/(const Vector<float8> &x) const
{
    Vector<float8> r;
    r.d.v()[0] = _mm_div_ps(d.v()[0], x.d.v()[0]);
    r.d.v()[1] = _mm_div_ps(d.v()[1], x.d.v()[1]);
    return r;
}

template<> inline Vector<double> &Vector<double>::operator/=(const Vector<double> &x)
{
    d.v() = _mm_div_pd(d.v(), x.d.v());
    return *this;
}

template<> inline Vector<double> Vector<double>::operator/(const Vector<double> &x) const
{
    return _mm_div_pd(d.v(), x.d.v());
}

///////////////////////////////////////////////////////////////////////////////////////////
// integer ops {{{1
#define OP_IMPL(T, symbol, fun) \
template<> inline Vector<T> &VectorBase<T>::operator symbol##=(const VectorBase<T> &x) \
{ \
    d.v() = VectorHelper<T>::fun(d.v(), x.d.v()); \
    return *static_cast<Vector<T> *>(this); \
} \
template<> inline Vector<T>  VectorBase<T>::operator symbol(const VectorBase<T> &x) const \
{ \
    return Vector<T>(VectorHelper<T>::fun(d.v(), x.d.v())); \
}
OP_IMPL(int, &, and_)
OP_IMPL(int, |, or_)
OP_IMPL(int, ^, xor_)
OP_IMPL(unsigned int, &, and_)
OP_IMPL(unsigned int, |, or_)
OP_IMPL(unsigned int, ^, xor_)
OP_IMPL(short, &, and_)
OP_IMPL(short, |, or_)
OP_IMPL(short, ^, xor_)
OP_IMPL(unsigned short, &, and_)
OP_IMPL(unsigned short, |, or_)
OP_IMPL(unsigned short, ^, xor_)
#undef OP_IMPL

#if defined(__GNUC__) && !defined(__INTEL_COMPILER) && __GNUC__ == 4 && __GNUC_MINOR__ == 6 && __GNUC_PATCHLEVEL__ == 0 && __XOP__
#define VC_WORKAROUND_IN
#define VC_WORKAROUND __attribute__((optimize("no-tree-vectorize"),weak))
#else
#define VC_WORKAROUND_IN inline
#define VC_WORKAROUND INTRINSIC
#endif

#define OP_IMPL(T, symbol) \
template<> VC_WORKAROUND_IN Vector<T> &VC_WORKAROUND VectorBase<T>::operator symbol##=(const VectorBase<T> &x) \
{ \
    for_all_vector_entries(i, \
            d.m(i) symbol##= x.d.m(i); \
            ); \
    return *static_cast<Vector<T> *>(this); \
} \
template<> inline Vector<T>  VectorBase<T>::operator symbol(const VectorBase<T> &x) const \
{ \
    Vector<T> r; \
    for_all_vector_entries(i, \
            r.d.m(i) = d.m(i) symbol x.d.m(i); \
            ); \
    return r; \
}
OP_IMPL(int, <<)
OP_IMPL(int, >>)
OP_IMPL(unsigned int, <<)
OP_IMPL(unsigned int, >>)
OP_IMPL(short, <<)
OP_IMPL(short, >>)
OP_IMPL(unsigned short, <<)
OP_IMPL(unsigned short, >>)
#undef OP_IMPL
#undef VC_WORKAROUND
#undef VC_WORKAROUND_IN

#define OP_IMPL(T, SUFFIX) \
template<> inline Vector<T> &VectorBase<T>::operator<<=(int x) \
{ \
    d.v() = CAT(_mm_slli_epi, SUFFIX)(d.v(), x); \
    return *static_cast<Vector<T> *>(this); \
} \
template<> inline Vector<T> &VectorBase<T>::operator>>=(int x) \
{ \
    d.v() = CAT(_mm_srli_epi, SUFFIX)(d.v(), x); \
    return *static_cast<Vector<T> *>(this); \
} \
template<> inline Vector<T> VectorBase<T>::operator<<(int x) const \
{ \
    return CAT(_mm_slli_epi, SUFFIX)(d.v(), x); \
} \
template<> inline Vector<T> VectorBase<T>::operator>>(int x) const \
{ \
    return CAT(_mm_srli_epi, SUFFIX)(d.v(), x); \
}
OP_IMPL(int, 32)
OP_IMPL(unsigned int, 32)
OP_IMPL(short, 16)
OP_IMPL(unsigned short, 16)
#undef OP_IMPL

///////////////////////////////////////////////////////////////////////////////////////////
// gathers {{{1
template<typename T> template<typename IndexT> inline ALWAYS_INLINE Vector<T>::Vector(const EntryType *mem, const IndexT *indexes)
{
    gather(mem, indexes);
}
template<typename T> template<typename IndexT> inline ALWAYS_INLINE Vector<T>::Vector(const EntryType *mem, Vector<IndexT> indexes)
{
    gather(mem, indexes);
}

template<typename T> template<typename IndexT> inline ALWAYS_INLINE Vector<T>::Vector(const EntryType *mem, const IndexT *indexes, Mask mask)
    : Base(HT::zero())
{
    gather(mem, indexes, mask);
}

template<typename T> template<typename IndexT> inline ALWAYS_INLINE Vector<T>::Vector(const EntryType *mem, Vector<IndexT> indexes, Mask mask)
    : Base(HT::zero())
{
    gather(mem, indexes, mask);
}

template<typename T> template<typename S1, typename IT> inline ALWAYS_INLINE Vector<T>::Vector(const S1 *array, const EntryType S1::* member1, IT indexes)
{
    gather(array, member1, indexes);
}
template<typename T> template<typename S1, typename IT> inline ALWAYS_INLINE Vector<T>::Vector(const S1 *array, const EntryType S1::* member1, IT indexes, Mask mask)
    : Base(HT::zero())
{
    gather(array, member1, indexes, mask);
}
template<typename T> template<typename S1, typename S2, typename IT> inline ALWAYS_INLINE Vector<T>::Vector(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes)
{
    gather(array, member1, member2, indexes);
}
template<typename T> template<typename S1, typename S2, typename IT> inline ALWAYS_INLINE Vector<T>::Vector(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes, Mask mask)
    : Base(HT::zero())
{
    gather(array, member1, member2, indexes, mask);
}
template<typename T> template<typename S1, typename IT1, typename IT2> inline ALWAYS_INLINE Vector<T>::Vector(const S1 *array, const EntryType *const S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes)
{
    gather(array, ptrMember1, outerIndexes, innerIndexes);
}
template<typename T> template<typename S1, typename IT1, typename IT2> inline ALWAYS_INLINE Vector<T>::Vector(const S1 *array, const EntryType *const S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes, Mask mask)
    : Base(HT::zero())
{
    gather(array, ptrMember1, outerIndexes, innerIndexes, mask);
}

template<typename T, size_t Size> struct IndexSizeChecker { static void check() {} };
template<typename T, size_t Size> struct IndexSizeChecker<Vector<T>, Size>
{
    static void check() {
        VC_STATIC_ASSERT(Vector<T>::Size >= Size, IndexVector_must_have_greater_or_equal_number_of_entries);
    }
};
template<> template<typename Index> inline void ALWAYS_INLINE FLATTEN Vector<double>::gather(const EntryType *mem, Index indexes)
{
    IndexSizeChecker<Index, Size>::check();
    d.v() = _mm_setr_pd(mem[indexes[0]], mem[indexes[1]]);
}
template<> template<typename Index> inline void ALWAYS_INLINE FLATTEN Vector<float>::gather(const EntryType *mem, Index indexes)
{
    IndexSizeChecker<Index, Size>::check();
    d.v() = _mm_setr_ps(mem[indexes[0]], mem[indexes[1]], mem[indexes[2]], mem[indexes[3]]);
}
template<> template<typename Index> inline void ALWAYS_INLINE FLATTEN Vector<float8>::gather(const EntryType *mem, Index indexes)
{
    IndexSizeChecker<Index, Size>::check();
    d.v()[0] = _mm_setr_ps(mem[indexes[0]], mem[indexes[1]], mem[indexes[2]], mem[indexes[3]]);
    d.v()[1] = _mm_setr_ps(mem[indexes[4]], mem[indexes[5]], mem[indexes[6]], mem[indexes[7]]);
}
template<> template<typename Index> inline void ALWAYS_INLINE FLATTEN Vector<int>::gather(const EntryType *mem, Index indexes)
{
    IndexSizeChecker<Index, Size>::check();
    d.v() = _mm_setr_epi32(mem[indexes[0]], mem[indexes[1]], mem[indexes[2]], mem[indexes[3]]);
}
template<> template<typename Index> inline void ALWAYS_INLINE FLATTEN Vector<unsigned int>::gather(const EntryType *mem, Index indexes)
{
    IndexSizeChecker<Index, Size>::check();
    d.v() = _mm_setr_epi32(mem[indexes[0]], mem[indexes[1]], mem[indexes[2]], mem[indexes[3]]);
}
template<> template<typename Index> inline void ALWAYS_INLINE FLATTEN Vector<short>::gather(const EntryType *mem, Index indexes)
{
    IndexSizeChecker<Index, Size>::check();
    d.v() = _mm_setr_epi16(mem[indexes[0]], mem[indexes[1]], mem[indexes[2]], mem[indexes[3]],
            mem[indexes[4]], mem[indexes[5]], mem[indexes[6]], mem[indexes[7]]);
}
template<> template<typename Index> inline void ALWAYS_INLINE FLATTEN Vector<unsigned short>::gather(const EntryType *mem, Index indexes)
{
    IndexSizeChecker<Index, Size>::check();
    d.v() = _mm_setr_epi16(mem[indexes[0]], mem[indexes[1]], mem[indexes[2]], mem[indexes[3]],
                mem[indexes[4]], mem[indexes[5]], mem[indexes[6]], mem[indexes[7]]);
}

#ifdef VC_USE_SET_GATHERS
template<typename T> template<typename IT> inline void ALWAYS_INLINE Vector<T>::gather(const EntryType *mem, Vector<IT> indexes, Mask mask)
{
    IndexSizeChecker<Vector<IT>, Size>::check();
    indexes.setZero(!static_cast<typename Vector<IT>::Mask>(mask));
    (*this)(mask) = Vector<T>(mem, indexes);
}
#endif

#ifdef VC_USE_BSF_GATHERS
#define VC_MASKED_GATHER                        \
    int bits = mask.toInt();                    \
    while (bits) {                              \
        const int i = _bit_scan_forward(bits);  \
        bits &= ~(1 << i); /* btr? */           \
        d.m(i) = ith_value(i);                  \
    }
#elif defined(VC_USE_POPCNT_BSF_GATHERS)
#define VC_MASKED_GATHER                        \
    unsigned int bits = mask.toInt();           \
    unsigned int low, high = 0;                 \
    switch (_mm_popcnt_u32(bits)) {             \
    case 8:                                     \
        high = _bit_scan_reverse(bits);         \
        d.m(high) = ith_value(high);            \
        high = (1 << high);                     \
    case 7:                                     \
        low = _bit_scan_forward(bits);          \
        bits ^= high | (1 << low);              \
        d.m(low) = ith_value(low);              \
    case 6:                                     \
        high = _bit_scan_reverse(bits);         \
        d.m(high) = ith_value(high);            \
        high = (1 << high);                     \
    case 5:                                     \
        low = _bit_scan_forward(bits);          \
        bits ^= high | (1 << low);              \
        d.m(low) = ith_value(low);              \
    case 4:                                     \
        high = _bit_scan_reverse(bits);         \
        d.m(high) = ith_value(high);            \
        high = (1 << high);                     \
    case 3:                                     \
        low = _bit_scan_forward(bits);          \
        bits ^= high | (1 << low);              \
        d.m(low) = ith_value(low);              \
    case 2:                                     \
        high = _bit_scan_reverse(bits);         \
        d.m(high) = ith_value(high);            \
    case 1:                                     \
        low = _bit_scan_forward(bits);          \
        d.m(low) = ith_value(low);              \
    case 0:                                     \
        break;                                  \
    }
#else
#define VC_MASKED_GATHER                        \
    if (mask.isEmpty()) {                       \
        return;                                 \
    }                                           \
    for_all_vector_entries(i,                   \
            if (mask[i]) d.m(i) = ith_value(i); \
            );
#endif

template<typename T> template<typename Index>
inline void INTRINSIC Vector<T>::gather(const EntryType *mem, Index indexes, Mask mask)
{
    IndexSizeChecker<Index, Size>::check();
#define ith_value(_i_) (mem[indexes[_i_]])
    VC_MASKED_GATHER
#undef ith_value
}

template<> template<typename S1, typename IT>
inline void ALWAYS_INLINE FLATTEN Vector<double>::gather(const S1 *array, const EntryType S1::* member1, IT indexes)
{
    IndexSizeChecker<IT, Size>::check();
    d.v() = _mm_setr_pd(array[indexes[0]].*(member1), array[indexes[1]].*(member1));
}
template<> template<typename S1, typename IT>
inline void ALWAYS_INLINE FLATTEN Vector<float>::gather(const S1 *array, const EntryType S1::* member1, IT indexes)
{
    IndexSizeChecker<IT, Size>::check();
    d.v() = _mm_setr_ps(array[indexes[0]].*(member1), array[indexes[1]].*(member1), array[indexes[2]].*(member1),
            array[indexes[3]].*(member1));
}
template<> template<typename S1, typename IT>
inline void ALWAYS_INLINE FLATTEN Vector<float8>::gather(const S1 *array, const EntryType S1::* member1, IT indexes)
{
    IndexSizeChecker<IT, Size>::check();
    d.v()[0] = _mm_setr_ps(array[indexes[0]].*(member1), array[indexes[1]].*(member1), array[indexes[2]].*(member1),
            array[indexes[3]].*(member1));
    d.v()[1] = _mm_setr_ps(array[indexes[4]].*(member1), array[indexes[5]].*(member1), array[indexes[6]].*(member1),
            array[indexes[7]].*(member1));
}
template<> template<typename S1, typename IT>
inline void ALWAYS_INLINE FLATTEN Vector<int>::gather(const S1 *array, const EntryType S1::* member1, IT indexes)
{
    IndexSizeChecker<IT, Size>::check();
    d.v() = _mm_setr_epi32(array[indexes[0]].*(member1), array[indexes[1]].*(member1), array[indexes[2]].*(member1),
            array[indexes[3]].*(member1));
}
template<> template<typename S1, typename IT>
inline void ALWAYS_INLINE FLATTEN Vector<unsigned int>::gather(const S1 *array, const EntryType S1::* member1, IT indexes)
{
    IndexSizeChecker<IT, Size>::check();
    d.v() = _mm_setr_epi32(array[indexes[0]].*(member1), array[indexes[1]].*(member1), array[indexes[2]].*(member1),
            array[indexes[3]].*(member1));
}
template<> template<typename S1, typename IT>
inline void ALWAYS_INLINE FLATTEN Vector<short>::gather(const S1 *array, const EntryType S1::* member1, IT indexes)
{
    IndexSizeChecker<IT, Size>::check();
    d.v() = _mm_setr_epi16(array[indexes[0]].*(member1), array[indexes[1]].*(member1), array[indexes[2]].*(member1),
            array[indexes[3]].*(member1), array[indexes[4]].*(member1), array[indexes[5]].*(member1),
            array[indexes[6]].*(member1), array[indexes[7]].*(member1));
}
template<> template<typename S1, typename IT>
inline void ALWAYS_INLINE FLATTEN Vector<unsigned short>::gather(const S1 *array, const EntryType S1::* member1, IT indexes)
{
    IndexSizeChecker<IT, Size>::check();
    d.v() = _mm_setr_epi16(array[indexes[0]].*(member1), array[indexes[1]].*(member1), array[indexes[2]].*(member1),
            array[indexes[3]].*(member1), array[indexes[4]].*(member1), array[indexes[5]].*(member1),
            array[indexes[6]].*(member1), array[indexes[7]].*(member1));
}
template<typename T> template<typename S1, typename IT>
inline void ALWAYS_INLINE FLATTEN Vector<T>::gather(const S1 *array, const EntryType S1::* member1, IT indexes, Mask mask)
{
    IndexSizeChecker<IT, Size>::check();
#define ith_value(_i_) (array[indexes[_i_]].*(member1))
    VC_MASKED_GATHER
#undef ith_value
}
template<> template<typename S1, typename S2, typename IT>
inline void ALWAYS_INLINE FLATTEN Vector<double>::gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes)
{
    IndexSizeChecker<IT, Size>::check();
    d.v() = _mm_setr_pd(array[indexes[0]].*(member1).*(member2), array[indexes[1]].*(member1).*(member2));
}
template<> template<typename S1, typename S2, typename IT>
inline void ALWAYS_INLINE FLATTEN Vector<float>::gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes)
{
    IndexSizeChecker<IT, Size>::check();
    d.v() = _mm_setr_ps(array[indexes[0]].*(member1).*(member2), array[indexes[1]].*(member1).*(member2), array[indexes[2]].*(member1).*(member2),
            array[indexes[3]].*(member1).*(member2));
}
template<> template<typename S1, typename S2, typename IT>
inline void ALWAYS_INLINE FLATTEN Vector<float8>::gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes)
{
    IndexSizeChecker<IT, Size>::check();
    d.v()[0] = _mm_setr_ps(array[indexes[0]].*(member1).*(member2), array[indexes[1]].*(member1).*(member2),
            array[indexes[2]].*(member1).*(member2), array[indexes[3]].*(member1).*(member2));
    d.v()[1] = _mm_setr_ps(array[indexes[4]].*(member1).*(member2), array[indexes[5]].*(member1).*(member2),
            array[indexes[6]].*(member1).*(member2), array[indexes[7]].*(member1).*(member2));
}
template<> template<typename S1, typename S2, typename IT>
inline void ALWAYS_INLINE FLATTEN Vector<int>::gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes)
{
    IndexSizeChecker<IT, Size>::check();
    d.v() = _mm_setr_epi32(array[indexes[0]].*(member1).*(member2), array[indexes[1]].*(member1).*(member2),
            array[indexes[2]].*(member1).*(member2), array[indexes[3]].*(member1).*(member2));
}
template<> template<typename S1, typename S2, typename IT>
inline void ALWAYS_INLINE FLATTEN Vector<unsigned int>::gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes)
{
    IndexSizeChecker<IT, Size>::check();
    d.v() = _mm_setr_epi32(array[indexes[0]].*(member1).*(member2), array[indexes[1]].*(member1).*(member2),
            array[indexes[2]].*(member1).*(member2), array[indexes[3]].*(member1).*(member2));
}
template<> template<typename S1, typename S2, typename IT>
inline void ALWAYS_INLINE FLATTEN Vector<short>::gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes)
{
    IndexSizeChecker<IT, Size>::check();
    d.v() = _mm_setr_epi16(array[indexes[0]].*(member1).*(member2), array[indexes[1]].*(member1).*(member2), array[indexes[2]].*(member1).*(member2),
            array[indexes[3]].*(member1).*(member2), array[indexes[4]].*(member1).*(member2), array[indexes[5]].*(member1).*(member2),
            array[indexes[6]].*(member1).*(member2), array[indexes[7]].*(member1).*(member2));
}
template<> template<typename S1, typename S2, typename IT>
inline void ALWAYS_INLINE FLATTEN Vector<unsigned short>::gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes)
{
    IndexSizeChecker<IT, Size>::check();
    d.v() = _mm_setr_epi16(array[indexes[0]].*(member1).*(member2), array[indexes[1]].*(member1).*(member2), array[indexes[2]].*(member1).*(member2),
            array[indexes[3]].*(member1).*(member2), array[indexes[4]].*(member1).*(member2), array[indexes[5]].*(member1).*(member2),
            array[indexes[6]].*(member1).*(member2), array[indexes[7]].*(member1).*(member2));
}
template<typename T> template<typename S1, typename S2, typename IT>
inline void ALWAYS_INLINE FLATTEN Vector<T>::gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes, Mask mask)
{
    IndexSizeChecker<IT, Size>::check();
#define ith_value(_i_) (array[indexes[_i_]].*(member1).*(member2))
    VC_MASKED_GATHER
#undef ith_value
}
template<> template<typename S1, typename IT1, typename IT2>
inline void ALWAYS_INLINE FLATTEN Vector<double>::gather(const S1 *array, const EntryType *const S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes)
{
    IndexSizeChecker<IT1, Size>::check();
    IndexSizeChecker<IT2, Size>::check();
    d.v() = _mm_setr_pd((array[outerIndexes[0]].*(ptrMember1))[innerIndexes[0]],
            (array[outerIndexes[1]].*(ptrMember1))[innerIndexes[1]]);
}
template<> template<typename S1, typename IT1, typename IT2>
inline void ALWAYS_INLINE FLATTEN Vector<float>::gather(const S1 *array, const EntryType *const S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes)
{
    IndexSizeChecker<IT1, Size>::check();
    IndexSizeChecker<IT2, Size>::check();
    d.v() = _mm_setr_ps((array[outerIndexes[0]].*(ptrMember1))[innerIndexes[0]], (array[outerIndexes[1]].*(ptrMember1))[innerIndexes[1]],
            (array[outerIndexes[2]].*(ptrMember1))[innerIndexes[2]], (array[outerIndexes[3]].*(ptrMember1))[innerIndexes[3]]);
}
template<> template<typename S1, typename IT1, typename IT2>
inline void ALWAYS_INLINE FLATTEN Vector<float8>::gather(const S1 *array, const EntryType *const S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes)
{
    IndexSizeChecker<IT1, Size>::check();
    IndexSizeChecker<IT2, Size>::check();
    d.v()[0] = _mm_setr_ps((array[outerIndexes[0]].*(ptrMember1))[innerIndexes[0]], (array[outerIndexes[1]].*(ptrMember1))[innerIndexes[1]],
            (array[outerIndexes[2]].*(ptrMember1))[innerIndexes[2]], (array[outerIndexes[3]].*(ptrMember1))[innerIndexes[3]]);
    d.v()[1] = _mm_setr_ps((array[outerIndexes[4]].*(ptrMember1))[innerIndexes[4]], (array[outerIndexes[5]].*(ptrMember1))[innerIndexes[5]],
            (array[outerIndexes[6]].*(ptrMember1))[innerIndexes[6]], (array[outerIndexes[7]].*(ptrMember1))[innerIndexes[7]]);
}
template<> template<typename S1, typename IT1, typename IT2>
inline void ALWAYS_INLINE FLATTEN Vector<int>::gather(const S1 *array, const EntryType *const S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes)
{
    IndexSizeChecker<IT1, Size>::check();
    IndexSizeChecker<IT2, Size>::check();
    d.v() = _mm_setr_epi32((array[outerIndexes[0]].*(ptrMember1))[innerIndexes[0]], (array[outerIndexes[1]].*(ptrMember1))[innerIndexes[1]],
            (array[outerIndexes[2]].*(ptrMember1))[innerIndexes[2]], (array[outerIndexes[3]].*(ptrMember1))[innerIndexes[3]]);
}
template<> template<typename S1, typename IT1, typename IT2>
inline void ALWAYS_INLINE FLATTEN Vector<unsigned int>::gather(const S1 *array, const EntryType *const S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes)
{
    IndexSizeChecker<IT1, Size>::check();
    IndexSizeChecker<IT2, Size>::check();
    d.v() = _mm_setr_epi32((array[outerIndexes[0]].*(ptrMember1))[innerIndexes[0]], (array[outerIndexes[1]].*(ptrMember1))[innerIndexes[1]],
            (array[outerIndexes[2]].*(ptrMember1))[innerIndexes[2]], (array[outerIndexes[3]].*(ptrMember1))[innerIndexes[3]]);
}
template<> template<typename S1, typename IT1, typename IT2>
inline void ALWAYS_INLINE FLATTEN Vector<short>::gather(const S1 *array, const EntryType *const S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes)
{
    IndexSizeChecker<IT1, Size>::check();
    IndexSizeChecker<IT2, Size>::check();
    d.v() = _mm_setr_epi16((array[outerIndexes[0]].*(ptrMember1))[innerIndexes[0]], (array[outerIndexes[1]].*(ptrMember1))[innerIndexes[1]],
            (array[outerIndexes[2]].*(ptrMember1))[innerIndexes[2]], (array[outerIndexes[3]].*(ptrMember1))[innerIndexes[3]],
            (array[outerIndexes[4]].*(ptrMember1))[innerIndexes[4]], (array[outerIndexes[5]].*(ptrMember1))[innerIndexes[5]],
            (array[outerIndexes[6]].*(ptrMember1))[innerIndexes[6]], (array[outerIndexes[7]].*(ptrMember1))[innerIndexes[7]]);
}
template<> template<typename S1, typename IT1, typename IT2>
inline void ALWAYS_INLINE FLATTEN Vector<unsigned short>::gather(const S1 *array, const EntryType *const S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes)
{
    IndexSizeChecker<IT1, Size>::check();
    IndexSizeChecker<IT2, Size>::check();
    d.v() = _mm_setr_epi16((array[outerIndexes[0]].*(ptrMember1))[innerIndexes[0]], (array[outerIndexes[1]].*(ptrMember1))[innerIndexes[1]],
            (array[outerIndexes[2]].*(ptrMember1))[innerIndexes[2]], (array[outerIndexes[3]].*(ptrMember1))[innerIndexes[3]],
            (array[outerIndexes[4]].*(ptrMember1))[innerIndexes[4]], (array[outerIndexes[5]].*(ptrMember1))[innerIndexes[5]],
            (array[outerIndexes[6]].*(ptrMember1))[innerIndexes[6]], (array[outerIndexes[7]].*(ptrMember1))[innerIndexes[7]]);
}
template<typename T> template<typename S1, typename IT1, typename IT2>
inline void ALWAYS_INLINE FLATTEN Vector<T>::gather(const S1 *array, const EntryType *const S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes, Mask mask)
{
    IndexSizeChecker<IT1, Size>::check();
    IndexSizeChecker<IT2, Size>::check();
#define ith_value(_i_) (array[outerIndexes[_i_]].*(ptrMember1))[innerIndexes[_i_]]
    VC_MASKED_GATHER
#undef ith_value
}

#undef VC_MASKED_GATHER
#ifdef VC_USE_BSF_SCATTERS
#define VC_MASKED_SCATTER                       \
    int bits = mask.toInt();                    \
    while (bits) {                              \
        const int i = _bit_scan_forward(bits);  \
        bits ^= (1 << i); /* btr? */            \
        ith_value(i) = d.m(i);                  \
    }
#elif defined(VC_USE_POPCNT_BSF_SCATTERS)
#define VC_MASKED_SCATTER                       \
    unsigned int bits = mask.toInt();           \
    unsigned int low, high = 0;                 \
    switch (_mm_popcnt_u32(bits)) {             \
    case 8:                                     \
        high = _bit_scan_reverse(bits);         \
        ith_value(high) = d.m(high);            \
        high = (1 << high);                     \
    case 7:                                     \
        low = _bit_scan_forward(bits);          \
        bits ^= high | (1 << low);              \
        ith_value(low) = d.m(low);              \
    case 6:                                     \
        high = _bit_scan_reverse(bits);         \
        ith_value(high) = d.m(high);            \
        high = (1 << high);                     \
    case 5:                                     \
        low = _bit_scan_forward(bits);          \
        bits ^= high | (1 << low);              \
        ith_value(low) = d.m(low);              \
    case 4:                                     \
        high = _bit_scan_reverse(bits);         \
        ith_value(high) = d.m(high);            \
        high = (1 << high);                     \
    case 3:                                     \
        low = _bit_scan_forward(bits);          \
        bits ^= high | (1 << low);              \
        ith_value(low) = d.m(low);              \
    case 2:                                     \
        high = _bit_scan_reverse(bits);         \
        ith_value(high) = d.m(high);            \
    case 1:                                     \
        low = _bit_scan_forward(bits);          \
        ith_value(low) = d.m(low);              \
    case 0:                                     \
        break;                                  \
    }
#else
#define VC_MASKED_SCATTER                       \
    if (mask.isEmpty()) {                       \
        return;                                 \
    }                                           \
    for_all_vector_entries(i,                   \
            if (mask[i]) ith_value(i) = d.m(i); \
            );
#endif

template<typename T> template<typename Index> inline void ALWAYS_INLINE FLATTEN Vector<T>::scatter(EntryType *mem, Index indexes) const
{
    for_all_vector_entries(i,
            mem[indexes[i]] = d.m(i);
            );
}
template<typename T> template<typename Index> inline void ALWAYS_INLINE FLATTEN Vector<T>::scatter(EntryType *mem, Index indexes, Mask mask) const
{
#define ith_value(_i_) mem[indexes[_i_]]
    VC_MASKED_SCATTER
#undef ith_value
}
template<typename T> template<typename S1, typename IT> inline void ALWAYS_INLINE FLATTEN Vector<T>::scatter(S1 *array, EntryType S1::* member1, IT indexes) const
{
    for_all_vector_entries(i,
            array[indexes[i]].*(member1) = d.m(i);
            );
}
template<typename T> template<typename S1, typename IT> inline void ALWAYS_INLINE FLATTEN Vector<T>::scatter(S1 *array, EntryType S1::* member1, IT indexes, Mask mask) const
{
#define ith_value(_i_) array[indexes[_i_]].*(member1)
    VC_MASKED_SCATTER
#undef ith_value
}
template<typename T> template<typename S1, typename S2, typename IT> inline void ALWAYS_INLINE FLATTEN Vector<T>::scatter(S1 *array, S2 S1::* member1, EntryType S2::* member2, IT indexes) const
{
    for_all_vector_entries(i,
            array[indexes[i]].*(member1).*(member2) = d.m(i);
            );
}
template<typename T> template<typename S1, typename S2, typename IT> inline void ALWAYS_INLINE FLATTEN Vector<T>::scatter(S1 *array, S2 S1::* member1, EntryType S2::* member2, IT indexes, Mask mask) const
{
#define ith_value(_i_) array[indexes[_i_]].*(member1).*(member2)
    VC_MASKED_SCATTER
#undef ith_value
}
template<typename T> template<typename S1, typename IT1, typename IT2> inline void ALWAYS_INLINE FLATTEN Vector<T>::scatter(S1 *array, EntryType *S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes) const
{
    for_all_vector_entries(i,
            (array[innerIndexes[i]].*(ptrMember1))[outerIndexes[i]] = d.m(i);
            );
}
template<typename T> template<typename S1, typename IT1, typename IT2> inline void ALWAYS_INLINE FLATTEN Vector<T>::scatter(S1 *array, EntryType *S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes, Mask mask) const
{
#define ith_value(_i_) (array[outerIndexes[_i_]].*(ptrMember1))[innerIndexes[_i_]]
    VC_MASKED_SCATTER
#undef ith_value
}

///////////////////////////////////////////////////////////////////////////////////////////
// operator- {{{1
template<> inline Vector<double> PURE ALWAYS_INLINE FLATTEN Vector<double>::operator-() const
{
    return _mm_xor_pd(d.v(), _mm_setsignmask_pd());
}
template<> inline Vector<float> PURE ALWAYS_INLINE FLATTEN Vector<float>::operator-() const
{
    return _mm_xor_ps(d.v(), _mm_setsignmask_ps());
}
template<> inline Vector<float8> PURE ALWAYS_INLINE FLATTEN Vector<float8>::operator-() const
{
    return M256::create(
            _mm_xor_ps(d.v()[0], _mm_setsignmask_ps()),
            _mm_xor_ps(d.v()[1], _mm_setsignmask_ps()));
}
template<> inline Vector<int> PURE ALWAYS_INLINE FLATTEN Vector<int>::operator-() const
{
#ifdef VC_IMPL_SSSE3
    return _mm_sign_epi32(d.v(), _mm_setallone_si128());
#else
    return _mm_add_epi32(_mm_xor_si128(d.v(), _mm_setallone_si128()), _mm_setone_epi32());
#endif
}
template<> inline Vector<int> PURE ALWAYS_INLINE FLATTEN Vector<unsigned int>::operator-() const
{
#ifdef VC_IMPL_SSSE3
    return _mm_sign_epi32(d.v(), _mm_setallone_si128());
#else
    return _mm_add_epi32(_mm_xor_si128(d.v(), _mm_setallone_si128()), _mm_setone_epi32());
#endif
}
template<> inline Vector<short> PURE ALWAYS_INLINE FLATTEN Vector<short>::operator-() const
{
#ifdef VC_IMPL_SSSE3
    return _mm_sign_epi16(d.v(), _mm_setallone_si128());
#else
    return _mm_mullo_epi16(d.v(), _mm_setallone_si128());
#endif
}
template<> inline Vector<short> PURE ALWAYS_INLINE FLATTEN Vector<unsigned short>::operator-() const
{
#ifdef VC_IMPL_SSSE3
    return _mm_sign_epi16(d.v(), _mm_setallone_si128());
#else
    return _mm_mullo_epi16(d.v(), _mm_setallone_si128());
#endif
}

///////////////////////////////////////////////////////////////////////////////////////////
// horizontal ops {{{1
template<typename T> inline typename Vector<T>::EntryType Vector<T>::min(Mask m) const
{
    Vector<T> tmp = std::numeric_limits<Vector<T> >::max();
    tmp(m) = *this;
    return tmp.min();
}
template<typename T> inline typename Vector<T>::EntryType Vector<T>::max(Mask m) const
{
    Vector<T> tmp = std::numeric_limits<Vector<T> >::min();
    tmp(m) = *this;
    return tmp.max();
}
template<typename T> inline typename Vector<T>::EntryType Vector<T>::product(Mask m) const
{
    Vector<T> tmp(VectorSpecialInitializerOne::One);
    tmp(m) = *this;
    return tmp.product();
}
template<typename T> inline typename Vector<T>::EntryType Vector<T>::sum(Mask m) const
{
    Vector<T> tmp(VectorSpecialInitializerZero::Zero);
    tmp(m) = *this;
    return tmp.sum();
}

} // namespace SSE
} // namespace Vc

#include "undomacros.h"

// vim: foldmethod=marker
