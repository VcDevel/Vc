/*  This file is part of the Vc library.

    Copyright (C) 2011 Matthias Kretz <kretz@kde.org>

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

#include "macros.h"

namespace Vc
{
namespace AVX
{

///////////////////////////////////////////////////////////////////////////////////////////
template<typename T> inline ALWAYS_INLINE Vector<T>::Vector(VectorSpecialInitializerZero::ZEnum) : d(HT::zero()) {}
template<typename T> inline ALWAYS_INLINE Vector<T>::Vector(VectorSpecialInitializerOne::OEnum) : d(HT::one()) {}
template<typename T> inline ALWAYS_INLINE Vector<T>::Vector(VectorSpecialInitializerIndexesFromZero::IEnum)
    : d(HV::load(IndexesFromZeroData<T>::address(), Aligned)) {}

template<typename T> inline Vector<T> INTRINSIC CONST Vector<T>::Zero() { return HT::zero(); }
template<typename T> inline Vector<T> INTRINSIC CONST Vector<T>::One() { return HT::one(); }
template<typename T> inline Vector<T> INTRINSIC CONST Vector<T>::IndexesFromZero() { return HV::load(IndexesFromZeroData<T>::address(), Aligned); }

template<typename T> template<typename T2> inline ALWAYS_INLINE Vector<T>::Vector(Vector<T2> x)
    : d(StaticCastHelper<T2, T>::cast(x.data())) {}

template<typename T> inline ALWAYS_INLINE Vector<T>::Vector(EntryType x) : d(HT::set(x)) {}
template<> inline ALWAYS_INLINE Vector<double>::Vector(EntryType x) : d(_mm256_set1_pd(x)) {}


///////////////////////////////////////////////////////////////////////////////////////////
// load ctors
template<typename T> inline ALWAYS_INLINE Vector<T>::Vector(const EntryType *x)
    : d(HV::load(x, Aligned)) {}

template<typename T> template<typename A> inline ALWAYS_INLINE Vector<T>::Vector(const EntryType *x, A align)
    : d(HV::load(x, align)) {}

///////////////////////////////////////////////////////////////////////////////////////////
// load member functions
template<typename T> inline void INTRINSIC Vector<T>::load(const EntryType *mem)
{
    data() = HV::load(mem, Aligned);
}
template<typename T> template<typename A> inline void INTRINSIC Vector<T>::load(const EntryType *mem, A align)
{
    data() = HV::load(mem, align);
}

///////////////////////////////////////////////////////////////////////////////////////////
// zeroing
template<typename T> inline void INTRINSIC Vector<T>::setZero()
{
    data() = HV::zero();
}
template<typename T> inline void INTRINSIC Vector<T>::setZero(const Mask &k)
{
    data() = HV::andnot_(avx_cast<VectorType>(k.data()), data());
}

template<> inline void INTRINSIC Vector<double>::setQnan()
{
    data() = _mm256_setallone_pd();
}
template<> inline void INTRINSIC Vector<double>::setQnan(Mask k)
{
    data() = _mm256_or_pd(data(), k.dataD());
}
template<> inline void INTRINSIC Vector<float>::setQnan()
{
    data() = _mm256_setallone_ps();
}
template<> inline void INTRINSIC Vector<float>::setQnan(Mask k)
{
    data() = _mm256_or_ps(data(), k.data());
}

///////////////////////////////////////////////////////////////////////////////////////////
// stores
template<typename T> inline void INTRINSIC Vector<T>::store(EntryType *mem) const
{
    HV::store(mem, data(), Aligned);
}
template<typename T> inline void INTRINSIC Vector<T>::store(EntryType *mem, const Mask &mask) const
{
    HV::store(mem, data(), avx_cast<VectorType>(mask.data()), Aligned);
}
template<typename T> template<typename A> inline void INTRINSIC Vector<T>::store(EntryType *mem, A align) const
{
    HV::store(mem, data(), align);
}
template<typename T> template<typename A> inline void INTRINSIC Vector<T>::store(EntryType *mem, const Mask &mask, A align) const
{
    HV::store(mem, data(), avx_cast<VectorType>(mask.data()), align);
}

///////////////////////////////////////////////////////////////////////////////////////////
// expand/merge 1 float_v <=> 2 double_v          XXX rationale? remove it for release? XXX
template<typename T> inline ALWAYS_INLINE FLATTEN Vector<T>::Vector(const Vector<typename HT::ConcatType> *a)
    : d(a[0])
{
}
template<> inline ALWAYS_INLINE FLATTEN Vector<float>::Vector(const Vector<HT::ConcatType> *a)
    : d(concat(_mm256_cvtpd_ps(a[0].data()), _mm256_cvtpd_ps(a[1].data())))
{
}
template<> inline ALWAYS_INLINE FLATTEN Vector<short>::Vector(const Vector<HT::ConcatType> *a)
    : d(_mm_packs_epi32(lo128(a->data()), hi128(a->data())))
{
}
template<> inline ALWAYS_INLINE FLATTEN Vector<unsigned short>::Vector(const Vector<HT::ConcatType> *a)
    : d(_mm_packus_epi32(lo128(a->data()), hi128(a->data())))
{
}
template<typename T> inline void ALWAYS_INLINE FLATTEN Vector<T>::expand(Vector<typename HT::ConcatType> *x) const
{
    x[0] = *this;
}
template<> inline void ALWAYS_INLINE FLATTEN Vector<float>::expand(Vector<HT::ConcatType> *x) const
{
    x[0].data() = _mm256_cvtps_pd(lo128(d.v()));
    x[1].data() = _mm256_cvtps_pd(hi128(d.v()));
}
template<> inline void ALWAYS_INLINE FLATTEN Vector<short>::expand(Vector<HT::ConcatType> *x) const
{
    x[0].data() = concat(_mm_cvtepi16_epi32(d.v()),
            _mm_cvtepi16_epi32(_mm_unpackhi_epi64(d.v(), d.v())));
}
template<> inline void ALWAYS_INLINE FLATTEN Vector<unsigned short>::expand(Vector<HT::ConcatType> *x) const
{
    x[0].data() = concat(_mm_cvtepu16_epi32(d.v()),
            _mm_cvtepu16_epi32(_mm_unpackhi_epi64(d.v(), d.v())));
}

///////////////////////////////////////////////////////////////////////////////////////////
// swizzles
template<> inline const Vector<double> INTRINSIC Vector<double>::aaaa() const { const double &tmp = d.m(0); return _mm256_broadcast_sd(&tmp); }
template<> inline const Vector<double> INTRINSIC Vector<double>::bbbb() const { const double &tmp = d.m(1); return _mm256_broadcast_sd(&tmp); }
template<> inline const Vector<double> INTRINSIC Vector<double>::cccc() const { const double &tmp = d.m(2); return _mm256_broadcast_sd(&tmp); }
template<> inline const Vector<double> INTRINSIC Vector<double>::dddd() const { const double &tmp = d.m(3); return _mm256_broadcast_sd(&tmp); }

///////////////////////////////////////////////////////////////////////////////////////////
// operators
///////////////////////////////////////////////////////////////////////////////////////////
//// division
template<typename T> inline Vector<T> &Vector<T>::operator/=(EntryType x)
{
    if (HasVectorDivision) {
        return operator/=(Vector<T>(x));
    }
    for_all_vector_entries(i,
            d.m(i) /= x;
            );
    return *this;
}
template<typename T> inline PURE Vector<T> Vector<T>::operator/(EntryType x) const
{
    if (HasVectorDivision) {
        return operator/(Vector<T>(x));
    }
    Vector<T> r;
    for_all_vector_entries(i,
            r.d.m(i) = d.m(i) / x;
            );
    return r;
}
// per default fall back to scalar division
template<typename T> inline Vector<T> &Vector<T>::operator/=(const Vector<T> &x)
{
    for_all_vector_entries(i,
            d.m(i) /= x.d.m(i);
            );
    return *this;
}

template<typename T> inline Vector<T> PURE Vector<T>::operator/(const Vector<T> &x) const
{
    Vector<T> r;
    for_all_vector_entries(i,
            r.d.m(i) = d.m(i) / x.d.m(i);
            );
    return r;
}
// specialize division on type
static inline __m256i INTRINSIC CONST divInt(__m256i a, __m256i b) {
    const __m256d lo1 = _mm256_cvtepi32_pd(lo128(a));
    const __m256d lo2 = _mm256_cvtepi32_pd(lo128(b));
    const __m256d hi1 = _mm256_cvtepi32_pd(hi128(a));
    const __m256d hi2 = _mm256_cvtepi32_pd(hi128(b));
    return concat(
            _mm256_cvttpd_epi32(_mm256_div_pd(lo1, lo2)),
            _mm256_cvttpd_epi32(_mm256_div_pd(hi1, hi2))
            );
}
template<> inline Vector<int> &Vector<int>::operator/=(const Vector<int> &x)
{
    d.v() = divInt(d.v(), x.d.v());
    return *this;
}
template<> inline Vector<int> PURE Vector<int>::operator/(const Vector<int> &x) const
{
    return divInt(d.v(), x.d.v());
}
static inline __m256i CONST divUInt(__m256i a, __m256i b) {
    __m256d loa = _mm256_cvtepi32_pd(lo128(a));
    __m256d hia = _mm256_cvtepi32_pd(hi128(a));
    __m256d lob = _mm256_cvtepi32_pd(lo128(b));
    __m256d hib = _mm256_cvtepi32_pd(hi128(b));
    // if a >= 2^31 then after conversion to double it will contain a negative number (i.e. a-2^32)
    // to get the right number back we have to add 2^32 where a >= 2^31
    loa = _mm256_add_pd(loa, _mm256_and_pd(_mm256_cmp_pd(loa, _mm256_setzero_pd(), _CMP_LT_OS), _mm256_set1_pd(4294967296.)));
    hia = _mm256_add_pd(hia, _mm256_and_pd(_mm256_cmp_pd(hia, _mm256_setzero_pd(), _CMP_LT_OS), _mm256_set1_pd(4294967296.)));
    // we don't do the same for b because division by b >= 2^31 should be a seldom corner case and
    // we rather want the standard stuff fast
    //
    // there is one remaining problem: a >= 2^31 and b == 1
    // in that case the return value would be 2^31
    return avx_cast<__m256i>(_mm256_blendv_ps(avx_cast<__m256>(concat(
                        _mm256_cvttpd_epi32(_mm256_div_pd(loa, lob)),
                        _mm256_cvttpd_epi32(_mm256_div_pd(hia, hib))
                        )), avx_cast<__m256>(a), avx_cast<__m256>(concat(
                            _mm_cmpeq_epi32(lo128(b), _mm_setone_epi32()),
                            _mm_cmpeq_epi32(hi128(b), _mm_setone_epi32())))));
}
template<> inline Vector<unsigned int> &ALWAYS_INLINE Vector<unsigned int>::operator/=(const Vector<unsigned int> &x)
{
    d.v() = divUInt(d.v(), x.d.v());
    return *this;
}
template<> inline Vector<unsigned int> ALWAYS_INLINE PURE Vector<unsigned int>::operator/(const Vector<unsigned int> &x) const
{
    return divUInt(d.v(), x.d.v());
}
template<typename T> static inline __m128i CONST divShort(__m128i a, __m128i b)
{
    const __m256 r = _mm256_div_ps(StaticCastHelper<T, float>::cast(a),
            StaticCastHelper<T, float>::cast(b));
    return StaticCastHelper<float, T>::cast(r);
}
template<> inline Vector<short> &ALWAYS_INLINE Vector<short>::operator/=(const Vector<short> &x)
{
    d.v() = divShort<short>(d.v(), x.d.v());
    return *this;
}
template<> inline Vector<short> ALWAYS_INLINE PURE Vector<short>::operator/(const Vector<short> &x) const
{
    return divShort<short>(d.v(), x.d.v());
}
template<> inline Vector<unsigned short> &ALWAYS_INLINE Vector<unsigned short>::operator/=(const Vector<unsigned short> &x)
{
    d.v() = divShort<unsigned short>(d.v(), x.d.v());
    return *this;
}
template<> inline Vector<unsigned short> ALWAYS_INLINE PURE Vector<unsigned short>::operator/(const Vector<unsigned short> &x) const
{
    return divShort<unsigned short>(d.v(), x.d.v());
}
template<> inline Vector<float> &INTRINSIC Vector<float>::operator/=(const Vector<float> &x)
{
    d.v() = _mm256_div_ps(d.v(), x.d.v());
    return *this;
}
template<> inline Vector<float> INTRINSIC PURE Vector<float>::operator/(const Vector<float> &x) const
{
    return _mm256_div_ps(d.v(), x.d.v());
}
template<> inline Vector<double> &INTRINSIC Vector<double>::operator/=(const Vector<double> &x)
{
    d.v() = _mm256_div_pd(d.v(), x.d.v());
    return *this;
}
template<> inline Vector<double> INTRINSIC PURE Vector<double>::operator/(const Vector<double> &x) const
{
    return _mm256_div_pd(d.v(), x.d.v());
}

///////////////////////////////////////////////////////////////////////////////////////////
//// integer ops
#define OP_IMPL(T, symbol) \
template<> inline Vector<T> &Vector<T>::operator symbol##=(Vector<T> x) \
{ \
    for_all_vector_entries(i, d.m(i) symbol##= x.d.m(i); ); \
    return *this; \
} \
template<> inline Vector<T>  Vector<T>::operator symbol(Vector<T> x) const \
{ \
    Vector<T> r; \
    for_all_vector_entries(i, r.d.m(i) = d.m(i) symbol x.d.m(i); ); \
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

#define OP_IMPL(T, PREFIX, SUFFIX) \
template<> inline Vector<T> & INTRINSIC CONST Vector<T>::operator<<=(int x) \
{ \
    d.v() = CAT3(PREFIX, _slli_epi, SUFFIX)(d.v(), x); \
    return *this; \
} \
template<> inline Vector<T> & INTRINSIC CONST Vector<T>::operator>>=(int x) \
{ \
    d.v() = CAT3(PREFIX, _srli_epi, SUFFIX)(d.v(), x); \
    return *this; \
} \
template<> inline Vector<T> INTRINSIC CONST Vector<T>::operator<<(int x) const \
{ \
    return CAT3(PREFIX, _slli_epi, SUFFIX)(d.v(), x); \
} \
template<> inline Vector<T> INTRINSIC CONST Vector<T>::operator>>(int x) const \
{ \
    return CAT3(PREFIX, _srli_epi, SUFFIX)(d.v(), x); \
}
OP_IMPL(int, _mm256, 32)
OP_IMPL(unsigned int, _mm256, 32)
OP_IMPL(short, _mm, 16)
OP_IMPL(unsigned short, _mm, 16)
#undef OP_IMPL

#define OP_IMPL(T, symbol, fun) \
  template<> inline Vector<T> &Vector<T>::operator symbol##=(Vector<T> x) { d.v() = VectorHelper<T>::fun(d.v(), x.d.v()); return *this; } \
  template<> inline Vector<T>  Vector<T>::operator symbol(Vector<T> x) const { return Vector<T>(VectorHelper<T>::fun(d.v(), x.d.v())); }
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

///////////////////////////////////////////////////////////////////////////////////////////
//// gathers
// Better implementation (hopefully) with _mm256_set_
//X template<typename T> template<typename Index> Vector<T>::Vector(const EntryType *mem, const Index *indexes)
//X {
//X     for_all_vector_entries(int i,
//X             d.m(i) = mem[indexes[i]];
//X             );
//X }
template<typename T> template<typename IndexT> inline ALWAYS_INLINE Vector<T>::Vector(const EntryType *mem, const IndexT *indexes)
{
    gather(mem, indexes);
}
template<typename T> template<typename IndexT> inline ALWAYS_INLINE Vector<T>::Vector(const EntryType *mem, Vector<IndexT> indexes)
{
    gather(mem, indexes);
}

template<typename T> template<typename IndexT> inline ALWAYS_INLINE Vector<T>::Vector(const EntryType *mem, const IndexT *indexes, Mask mask)
    : d(HT::zero())
{
    gather(mem, indexes, mask);
}

template<typename T> template<typename IndexT> inline ALWAYS_INLINE Vector<T>::Vector(const EntryType *mem, Vector<IndexT> indexes, Mask mask)
    : d(HT::zero())
{
    gather(mem, indexes, mask);
}

template<typename T> template<typename S1, typename IT> inline ALWAYS_INLINE Vector<T>::Vector(const S1 *array, const EntryType S1::* member1, IT indexes)
{
    gather(array, member1, indexes);
}
template<typename T> template<typename S1, typename IT> inline ALWAYS_INLINE Vector<T>::Vector(const S1 *array, const EntryType S1::* member1, IT indexes, Mask mask)
    : d(HT::zero())
{
    gather(array, member1, indexes, mask);
}
template<typename T> template<typename S1, typename S2, typename IT> inline ALWAYS_INLINE Vector<T>::Vector(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes)
{
    gather(array, member1, member2, indexes);
}
template<typename T> template<typename S1, typename S2, typename IT> inline ALWAYS_INLINE Vector<T>::Vector(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes, Mask mask)
    : d(HT::zero())
{
    gather(array, member1, member2, indexes, mask);
}
template<typename T> template<typename S1, typename IT1, typename IT2> inline ALWAYS_INLINE Vector<T>::Vector(const S1 *array, const EntryType *const S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes)
{
    gather(array, ptrMember1, outerIndexes, innerIndexes);
}
template<typename T> template<typename S1, typename IT1, typename IT2> inline ALWAYS_INLINE Vector<T>::Vector(const S1 *array, const EntryType *const S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes, Mask mask)
    : d(HT::zero())
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
    d.v() = _mm256_setr_pd(mem[indexes[0]], mem[indexes[1]], mem[indexes[2]], mem[indexes[3]]);
}
template<> template<typename Index> inline void ALWAYS_INLINE FLATTEN Vector<float>::gather(const EntryType *mem, Index indexes)
{
    IndexSizeChecker<Index, Size>::check();
    d.v() = _mm256_setr_ps(mem[indexes[0]], mem[indexes[1]], mem[indexes[2]], mem[indexes[3]],
            mem[indexes[4]], mem[indexes[5]], mem[indexes[6]], mem[indexes[7]]);
}
template<> template<typename Index> inline void ALWAYS_INLINE FLATTEN Vector<int>::gather(const EntryType *mem, Index indexes)
{
    IndexSizeChecker<Index, Size>::check();
    d.v() = _mm256_setr_epi32(mem[indexes[0]], mem[indexes[1]], mem[indexes[2]], mem[indexes[3]],
            mem[indexes[4]], mem[indexes[5]], mem[indexes[6]], mem[indexes[7]]);
}
template<> template<typename Index> inline void ALWAYS_INLINE FLATTEN Vector<unsigned int>::gather(const EntryType *mem, Index indexes)
{
    IndexSizeChecker<Index, Size>::check();
    d.v() = _mm256_setr_epi32(mem[indexes[0]], mem[indexes[1]], mem[indexes[2]], mem[indexes[3]],
            mem[indexes[4]], mem[indexes[5]], mem[indexes[6]], mem[indexes[7]]);
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

#ifdef VC_GATHER_SET
template<typename T> template<typename IT> inline void ALWAYS_INLINE Vector<T>::gather(const EntryType *mem, Vector<IT> indexes, Mask mask)
{
    IndexSizeChecker<Vector<IT>, Size>::check();
    indexes.setZero(!mask);
    gather(mem, indexes);
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
    case 7:                                     \
        low = _bit_scan_forward(bits);          \
        bits ^= (1 << high) | (1 << low);       \
        d.m(low) = ith_value(low);              \
    case 6:                                     \
        high = _bit_scan_reverse(bits);         \
        d.m(high) = ith_value(high);            \
    case 5:                                     \
        low = _bit_scan_forward(bits);          \
        bits ^= (1 << high) | (1 << low);       \
        d.m(low) = ith_value(low);              \
    case 4:                                     \
        high = _bit_scan_reverse(bits);         \
        d.m(high) = ith_value(high);            \
    case 3:                                     \
        low = _bit_scan_forward(bits);          \
        bits ^= (1 << high) | (1 << low);       \
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
    d.v() = _mm256_setr_pd(array[indexes[0]].*(member1), array[indexes[1]].*(member1),
            array[indexes[2]].*(member1), array[indexes[3]].*(member1));
}
template<> template<typename S1, typename IT>
inline void ALWAYS_INLINE FLATTEN Vector<float>::gather(const S1 *array, const EntryType S1::* member1, IT indexes)
{
    IndexSizeChecker<IT, Size>::check();
    d.v() = _mm256_setr_ps(array[indexes[0]].*(member1), array[indexes[1]].*(member1), array[indexes[2]].*(member1),
            array[indexes[3]].*(member1), array[indexes[4]].*(member1), array[indexes[5]].*(member1),
            array[indexes[6]].*(member1), array[indexes[7]].*(member1));
}
template<> template<typename S1, typename IT>
inline void ALWAYS_INLINE FLATTEN Vector<int>::gather(const S1 *array, const EntryType S1::* member1, IT indexes)
{
    IndexSizeChecker<IT, Size>::check();
    d.v() = _mm256_setr_epi32(array[indexes[0]].*(member1), array[indexes[1]].*(member1), array[indexes[2]].*(member1),
            array[indexes[3]].*(member1), array[indexes[4]].*(member1), array[indexes[5]].*(member1),
            array[indexes[6]].*(member1), array[indexes[7]].*(member1));
}
template<> template<typename S1, typename IT>
inline void ALWAYS_INLINE FLATTEN Vector<unsigned int>::gather(const S1 *array, const EntryType S1::* member1, IT indexes)
{
    IndexSizeChecker<IT, Size>::check();
    d.v() = _mm256_setr_epi32(array[indexes[0]].*(member1), array[indexes[1]].*(member1), array[indexes[2]].*(member1),
            array[indexes[3]].*(member1), array[indexes[4]].*(member1), array[indexes[5]].*(member1),
            array[indexes[6]].*(member1), array[indexes[7]].*(member1));
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
    d.v() = _mm256_setr_pd(array[indexes[0]].*(member1).*(member2), array[indexes[1]].*(member1).*(member2),
            array[indexes[2]].*(member1).*(member2), array[indexes[3]].*(member1).*(member2));
}
template<> template<typename S1, typename S2, typename IT>
inline void ALWAYS_INLINE FLATTEN Vector<float>::gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes)
{
    IndexSizeChecker<IT, Size>::check();
    d.v() = _mm256_setr_ps(array[indexes[0]].*(member1).*(member2), array[indexes[1]].*(member1).*(member2), array[indexes[2]].*(member1).*(member2),
            array[indexes[3]].*(member1).*(member2), array[indexes[4]].*(member1).*(member2), array[indexes[5]].*(member1).*(member2),
            array[indexes[6]].*(member1).*(member2), array[indexes[7]].*(member1).*(member2));
}
template<> template<typename S1, typename S2, typename IT>
inline void ALWAYS_INLINE FLATTEN Vector<int>::gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes)
{
    IndexSizeChecker<IT, Size>::check();
    d.v() = _mm256_setr_epi32(array[indexes[0]].*(member1).*(member2), array[indexes[1]].*(member1).*(member2), array[indexes[2]].*(member1).*(member2),
            array[indexes[3]].*(member1).*(member2), array[indexes[4]].*(member1).*(member2), array[indexes[5]].*(member1).*(member2),
            array[indexes[6]].*(member1).*(member2), array[indexes[7]].*(member1).*(member2));
}
template<> template<typename S1, typename S2, typename IT>
inline void ALWAYS_INLINE FLATTEN Vector<unsigned int>::gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes)
{
    IndexSizeChecker<IT, Size>::check();
    d.v() = _mm256_setr_epi32(array[indexes[0]].*(member1).*(member2), array[indexes[1]].*(member1).*(member2), array[indexes[2]].*(member1).*(member2),
            array[indexes[3]].*(member1).*(member2), array[indexes[4]].*(member1).*(member2), array[indexes[5]].*(member1).*(member2),
            array[indexes[6]].*(member1).*(member2), array[indexes[7]].*(member1).*(member2));
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
    d.v() = _mm256_setr_pd((array[outerIndexes[0]].*(ptrMember1))[innerIndexes[0]], (array[outerIndexes[1]].*(ptrMember1))[innerIndexes[1]],
            (array[outerIndexes[2]].*(ptrMember1))[innerIndexes[2]], (array[outerIndexes[3]].*(ptrMember1))[innerIndexes[3]]);
}
template<> template<typename S1, typename IT1, typename IT2>
inline void ALWAYS_INLINE FLATTEN Vector<float>::gather(const S1 *array, const EntryType *const S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes)
{
    IndexSizeChecker<IT1, Size>::check();
    IndexSizeChecker<IT2, Size>::check();
    d.v() = _mm256_setr_ps((array[outerIndexes[0]].*(ptrMember1))[innerIndexes[0]], (array[outerIndexes[1]].*(ptrMember1))[innerIndexes[1]],
            (array[outerIndexes[2]].*(ptrMember1))[innerIndexes[2]], (array[outerIndexes[3]].*(ptrMember1))[innerIndexes[3]],
            (array[outerIndexes[4]].*(ptrMember1))[innerIndexes[4]], (array[outerIndexes[5]].*(ptrMember1))[innerIndexes[5]],
            (array[outerIndexes[6]].*(ptrMember1))[innerIndexes[6]], (array[outerIndexes[7]].*(ptrMember1))[innerIndexes[7]]);
}
template<> template<typename S1, typename IT1, typename IT2>
inline void ALWAYS_INLINE FLATTEN Vector<int>::gather(const S1 *array, const EntryType *const S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes)
{
    IndexSizeChecker<IT1, Size>::check();
    IndexSizeChecker<IT2, Size>::check();
    d.v() = _mm256_setr_epi32((array[outerIndexes[0]].*(ptrMember1))[innerIndexes[0]], (array[outerIndexes[1]].*(ptrMember1))[innerIndexes[1]],
            (array[outerIndexes[2]].*(ptrMember1))[innerIndexes[2]], (array[outerIndexes[3]].*(ptrMember1))[innerIndexes[3]],
            (array[outerIndexes[4]].*(ptrMember1))[innerIndexes[4]], (array[outerIndexes[5]].*(ptrMember1))[innerIndexes[5]],
            (array[outerIndexes[6]].*(ptrMember1))[innerIndexes[6]], (array[outerIndexes[7]].*(ptrMember1))[innerIndexes[7]]);
}
template<> template<typename S1, typename IT1, typename IT2>
inline void ALWAYS_INLINE FLATTEN Vector<unsigned int>::gather(const S1 *array, const EntryType *const S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes)
{
    IndexSizeChecker<IT1, Size>::check();
    IndexSizeChecker<IT2, Size>::check();
    d.v() = _mm256_setr_epi32((array[outerIndexes[0]].*(ptrMember1))[innerIndexes[0]], (array[outerIndexes[1]].*(ptrMember1))[innerIndexes[1]],
            (array[outerIndexes[2]].*(ptrMember1))[innerIndexes[2]], (array[outerIndexes[3]].*(ptrMember1))[innerIndexes[3]],
            (array[outerIndexes[4]].*(ptrMember1))[innerIndexes[4]], (array[outerIndexes[5]].*(ptrMember1))[innerIndexes[5]],
            (array[outerIndexes[6]].*(ptrMember1))[innerIndexes[6]], (array[outerIndexes[7]].*(ptrMember1))[innerIndexes[7]]);
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
    case 7:                                     \
        low = _bit_scan_forward(bits);          \
        bits ^= (1 << high) | (1 << low);       \
        ith_value(low) = d.m(low);              \
    case 6:                                     \
        high = _bit_scan_reverse(bits);         \
        ith_value(high) = d.m(high);            \
    case 5:                                     \
        low = _bit_scan_forward(bits);          \
        bits ^= (1 << high) | (1 << low);       \
        ith_value(low) = d.m(low);              \
    case 4:                                     \
        high = _bit_scan_reverse(bits);         \
        ith_value(high) = d.m(high);            \
    case 3:                                     \
        low = _bit_scan_forward(bits);          \
        bits ^= (1 << high) | (1 << low);       \
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

template<typename T> template<typename Index> inline void ALWAYS_INLINE FLATTEN Vector<T>::scatter(EntryType *mem, Index indexes)
{
    for_all_vector_entries(i,
            mem[indexes[i]] = d.m(i);
            );
}
template<typename T> template<typename Index> inline void ALWAYS_INLINE FLATTEN Vector<T>::scatter(EntryType *mem, Index indexes, Mask mask)
{
#define ith_value(_i_) mem[indexes[_i_]]
    VC_MASKED_SCATTER
#undef ith_value
}
template<typename T> template<typename S1, typename IT> inline void ALWAYS_INLINE FLATTEN Vector<T>::scatter(S1 *array, EntryType S1::* member1, IT indexes)
{
    for_all_vector_entries(i,
            array[indexes[i]].*(member1) = d.m(i);
            );
}
template<typename T> template<typename S1, typename IT> inline void ALWAYS_INLINE FLATTEN Vector<T>::scatter(S1 *array, EntryType S1::* member1, IT indexes, Mask mask)
{
#define ith_value(_i_) array[indexes[_i_]].*(member1)
    VC_MASKED_SCATTER
#undef ith_value
}
template<typename T> template<typename S1, typename S2, typename IT> inline void ALWAYS_INLINE FLATTEN Vector<T>::scatter(S1 *array, S2 S1::* member1, EntryType S2::* member2, IT indexes)
{
    for_all_vector_entries(i,
            array[indexes[i]].*(member1).*(member2) = d.m(i);
            );
}
template<typename T> template<typename S1, typename S2, typename IT> inline void ALWAYS_INLINE FLATTEN Vector<T>::scatter(S1 *array, S2 S1::* member1, EntryType S2::* member2, IT indexes, Mask mask)
{
#define ith_value(_i_) array[indexes[_i_]].*(member1).*(member2)
    VC_MASKED_SCATTER
#undef ith_value
}
template<typename T> template<typename S1, typename IT1, typename IT2> inline void ALWAYS_INLINE FLATTEN Vector<T>::scatter(S1 *array, EntryType *S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes)
{
    for_all_vector_entries(i,
            (array[innerIndexes[i]].*(ptrMember1))[outerIndexes[i]] = d.m(i);
            );
}
template<typename T> template<typename S1, typename IT1, typename IT2> inline void ALWAYS_INLINE FLATTEN Vector<T>::scatter(S1 *array, EntryType *S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes, Mask mask)
{
#define ith_value(_i_) (array[outerIndexes[_i_]].*(ptrMember1))[innerIndexes[_i_]]
    VC_MASKED_SCATTER
#undef ith_value
}

template<> inline Vector<double> PURE ALWAYS_INLINE FLATTEN Vector<double>::operator-() const
{
    return _mm256_xor_pd(d.v(), _mm256_setsignmask_pd());
}
template<> inline Vector<float> PURE ALWAYS_INLINE FLATTEN Vector<float>::operator-() const
{
    return _mm256_xor_ps(d.v(), _mm256_setsignmask_ps());
}
template<> inline Vector<int> PURE ALWAYS_INLINE FLATTEN Vector<int>::operator-() const
{
    return _mm256_sign_epi32(d.v(), _mm256_setallone_si256());
}
template<> inline Vector<int> PURE ALWAYS_INLINE FLATTEN Vector<unsigned int>::operator-() const
{
    return _mm256_sign_epi32(d.v(), _mm256_setallone_si256());
}
template<> inline Vector<short> PURE ALWAYS_INLINE FLATTEN Vector<short>::operator-() const
{
    return _mm_sign_epi16(d.v(), _mm_setallone_si128());
}
template<> inline Vector<short> PURE ALWAYS_INLINE FLATTEN Vector<unsigned short>::operator-() const
{
    return _mm_sign_epi16(d.v(), _mm_setallone_si128());
}
} // namespace AVX
} // namespace Vc

#include "undomacros.h"
