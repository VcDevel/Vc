/*  This file is part of the Vc library.

    Copyright (C) 2010 Matthias Kretz <kretz@kde.org>

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

#include "debug.h"

namespace Vc
{
namespace LRBni
{

template<typename Parent, typename T> template<typename T2> inline void StoreMixin<Parent, T>::store(T2 *mem) const
{
    VectorHelper<T>::store(mem, static_cast<const Parent *>(this)->vdata(), Aligned);
}

template<typename Parent, typename T> template<typename T2> inline void StoreMixin<Parent, T>::store(T2 *mem, Mask mask) const
{
    VectorHelper<T>::store(mem, static_cast<const Parent *>(this)->vdata(), mask.data(), Aligned);
}

template<typename Parent, typename T> template<typename T2, typename A> inline void StoreMixin<Parent, T>::store(T2 *mem, A align) const
{
    VectorHelper<T>::store(mem, static_cast<const Parent *>(this)->vdata(), align);
}

template<typename Parent, typename T> template<typename T2, typename A> inline void StoreMixin<Parent, T>::store(T2 *mem, Mask mask, A align) const
{
    VectorHelper<T>::store(mem, static_cast<const Parent *>(this)->vdata(), mask.data(), align);
}

template<> inline Vector<double> INTRINSIC Vector<double>::operator-() const
{
    return lrb_cast<__m512d>(_mm512_xor_pi(lrb_cast<__m512i>(d.v()), _mm512_set_1to8_pq(0x8000000000000000ull)));
}
template<> inline Vector<float> INTRINSIC Vector<float>::operator-() const
{
    return lrb_cast<__m512>(_mm512_xor_pi(lrb_cast<__m512i>(d.v()), _mm512_set_1to16_pi(0x80000000u)));
}
template<> inline Vector<int> INTRINSIC Vector<int>::operator-() const
{
    return (~(*this)) + 1;
}
template<> inline Vector<int> INTRINSIC Vector<unsigned int>::operator-() const
{
    return Vector<int>(~(*this)) + 1;
}

template<typename T> inline typename Vector<T>::EntryType Vector<T>::min(Mask m) const
{
    return _mm512_mask_reduce_min_pi(m.data(), vdata());
}
template<> inline float Vector<float>::min(Mask m) const
{
    return _mm512_mask_reduce_min_ps(m.data(), vdata());
}
template<> inline double Vector<double>::min(Mask m) const
{
    return _mm512_mask_reduce_min_pd(m.data(), vdata());
}
template<typename T> inline typename Vector<T>::EntryType Vector<T>::max(Mask m) const
{
    return _mm512_mask_reduce_max_pi(m.data(), vdata());
}
template<> inline float Vector<float>::max(Mask m) const
{
    return _mm512_mask_reduce_max_ps(m.data(), vdata());
}
template<> inline double Vector<double>::max(Mask m) const
{
    return _mm512_mask_reduce_max_pd(m.data(), vdata());
}
template<typename T> inline typename Vector<T>::EntryType Vector<T>::product(Mask m) const
{
    return _mm512_mask_reduce_mul_pi(m.data(), vdata());
}
template<> inline float Vector<float>::product(Mask m) const
{
    return _mm512_mask_reduce_mul_ps(m.data(), vdata());
}
template<> inline double Vector<double>::product(Mask m) const
{
    return _mm512_mask_reduce_mul_pd(m.data(), vdata());
}
template<typename T> inline typename Vector<T>::EntryType Vector<T>::sum(Mask m) const
{
    return _mm512_mask_reduce_add_pi(m.data(), vdata());
}
template<> inline float Vector<float>::sum(Mask m) const
{
    return _mm512_mask_reduce_add_ps(m.data(), vdata());
}
template<> inline double Vector<double>::sum(Mask m) const
{
    return _mm512_mask_reduce_add_pd(m.data(), vdata());
}

// gathers {{{1
template<typename T> template<typename OtherT, typename IndexT> inline ALWAYS_INLINE Vector<T>::Vector(const OtherT *mem, const IndexT *indexes)
{
    gather(mem, indexes);
}
template<typename T> template<typename OtherT, typename IndexT> inline ALWAYS_INLINE Vector<T>::Vector(const OtherT *mem, Vector<IndexT> indexes)
{
    gather(mem, indexes);
}

template<typename T> template<typename OtherT, typename IndexT> inline ALWAYS_INLINE Vector<T>::Vector(const OtherT *mem, const IndexT *indexes, Mask mask)
    : d(lrb_cast<VectorType>(_mm512_setzero_ps()))
{
    gather(mem, indexes, mask);
}

template<typename T> template<typename OtherT, typename IndexT> inline ALWAYS_INLINE Vector<T>::Vector(const OtherT *mem, Vector<IndexT> indexes, Mask mask)
    : d(lrb_cast<VectorType>(_mm512_setzero_ps()))
{
    gather(mem, indexes, mask);
}

template<typename T> template<typename OtherT, typename S1, typename IT> inline ALWAYS_INLINE Vector<T>::Vector(const S1 *array, const OtherT S1::* member1, IT indexes)
{
    gather(array, member1, indexes);
}
template<typename T> template<typename OtherT, typename S1, typename IT> inline ALWAYS_INLINE Vector<T>::Vector(const S1 *array, const OtherT S1::* member1, IT indexes, Mask mask)
    : d(lrb_cast<VectorType>(_mm512_setzero_ps()))
{
    gather(array, member1, indexes, mask);
}
template<typename T> template<typename OtherT, typename S1, typename S2, typename IT> inline ALWAYS_INLINE Vector<T>::Vector(const S1 *array, const S2 S1::* member1, const OtherT S2::* member2, IT indexes)
{
    gather(array, member1, member2, indexes);
}
template<typename T> template<typename OtherT, typename S1, typename S2, typename IT> inline ALWAYS_INLINE Vector<T>::Vector(const S1 *array, const S2 S1::* member1, const OtherT S2::* member2, IT indexes, Mask mask)
    : d(lrb_cast<VectorType>(_mm512_setzero_ps()))
{
    gather(array, member1, member2, indexes, mask);
}
template<typename T> template<typename OtherT, typename S1, typename IT1, typename IT2> inline ALWAYS_INLINE Vector<T>::Vector(const S1 *array, const OtherT *const S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes)
{
    gather(array, ptrMember1, outerIndexes, innerIndexes);
}
template<typename T> template<typename OtherT, typename S1, typename IT1, typename IT2> inline ALWAYS_INLINE Vector<T>::Vector(const S1 *array, const OtherT *const S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes, Mask mask)
    : d(lrb_cast<VectorType>(_mm512_setzero_ps()))
{
    gather(array, ptrMember1, outerIndexes, innerIndexes, mask);
}

template<typename T, size_t Size> struct IndexSizeChecker { static void check() { IndexSizeChecker<Vector<T>, Size>::check(); } };
template<typename T, size_t Size> struct IndexSizeChecker<Vector<T>, Size>
{
    static void check() {
        VC_STATIC_ASSERT(Vector<T>::Size >= Size, IndexVector_must_have_greater_or_equal_number_of_entries);
    }
};
template<size_t Scale>
static inline void prepareDoubleGatherIndexes(_M512I &indexes) {
    indexes = lrb_cast<_M512I>(_mm512_mask_movq(
                _mm512_shuf128x32(lrb_cast<_M512>(indexes), _MM_PERM_BBAA, _MM_PERM_DDCC),
                0x33,
                _mm512_shuf128x32(lrb_cast<_M512>(indexes), _MM_PERM_BBAA, _MM_PERM_BBAA)
                ));
    indexes = _mm512_madd231_pi(_mm512_set_4to16_pi(0, 1, 0, 1), indexes, _mm512_set_1to16_pi(Scale));
}
template<> template<typename OtherT, typename Index> inline void ALWAYS_INLINE FLATTEN Vector<double>::gather(const OtherT *mem, const Index *indexes)
{
    // FIXME: this will read twice as much from memory as required
    gather(mem, Vector<Index>(indexes, Vc::Unaligned));
}
template<> template<typename OtherT, typename Index> inline void ALWAYS_INLINE FLATTEN Vector<double>::gather(const OtherT *mem, Vector<Index> indexes)
{
    IndexSizeChecker<Index, Size>::check();
    _M512I i = indexes.data();
    prepareDoubleGatherIndexes<2>(i);
    d.v() = lrb_cast<VectorType>(_mm512_gatherd(i, const_cast<OtherT *>(mem), _MM_FULLUPC_NONE, _MM_SCALE_4, _MM_HINT_NONE));
}
template<> template<typename OtherT, typename Index> inline void ALWAYS_INLINE FLATTEN Vector<float>::gather(const OtherT *mem, const Index *indexes)
{
    gather(mem, Vector<Index>(indexes, Vc::Unaligned));
}
template<> template<typename OtherT, typename Index> inline void ALWAYS_INLINE FLATTEN Vector<float>::gather(const OtherT *mem, Vector<Index> indexes)
{
    IndexSizeChecker<Index, Size>::check();
    d.v() = lrb_cast<VectorType>(_mm512_gatherd(indexes.data(), const_cast<OtherT *>(mem), _MM_FULLUPC_NONE, _MM_SCALE_4, _MM_HINT_NONE));
}
template<> template<typename OtherT, typename Index> inline void ALWAYS_INLINE FLATTEN Vector<int>::gather(const OtherT *mem, const Index *indexes)
{
    gather(mem, Vector<Index>(indexes, Vc::Unaligned));
}
template<> template<typename OtherT, typename Index> inline void ALWAYS_INLINE FLATTEN Vector<int>::gather(const OtherT *mem, Vector<Index> indexes)
{
    IndexSizeChecker<Index, Size>::check();
    d.v() = lrb_cast<VectorType>(_mm512_gatherd(indexes.data(), const_cast<OtherT *>(mem), _MM_FULLUPC_NONE, _MM_SCALE_4, _MM_HINT_NONE));
}
template<typename T> template<typename OtherT, typename Index> inline void ALWAYS_INLINE FLATTEN Vector<T>::gather(const OtherT *mem, const Index *indexes)
{
    gather(mem, Vector<Index>(indexes, Vc::Unaligned));
}
template<typename T> template<typename OtherT, typename Index> inline void ALWAYS_INLINE FLATTEN Vector<T>::gather(const OtherT *mem, Vector<Index> indexes)
{
    IndexSizeChecker<Index, Size>::check();
    d.v() = lrb_cast<VectorType>(_mm512_gatherd(indexes.data(), const_cast<OtherT *>(mem), _MM_FULLUPC_NONE, _MM_SCALE_4, _MM_HINT_NONE));
}

template<> template<typename OtherT, typename Index> inline void ALWAYS_INLINE FLATTEN Vector<double>::gather(const OtherT *mem, const Index *indexes, Mask mask)
{
    // FIXME: this will read twice as much from memory as required
    gather(mem, Vector<Index>(indexes, Vc::Unaligned), mask);
}
template<> template<typename OtherT, typename Index> inline void ALWAYS_INLINE FLATTEN Vector<double>::gather(const OtherT *mem,
        Vector<Index> indexes, Mask mask)
{
    IndexSizeChecker<Index, Size>::check();
    _M512I i = indexes.data();
    prepareDoubleGatherIndexes<2>(i);
    d.v() = lrb_cast<VectorType>(_mm512_mask_gatherd(lrb_cast<__m512>(d.v()), _mm_bitinterleave11_16(mask.data(), mask.data()), i, const_cast<OtherT *>(mem), _MM_FULLUPC_NONE, _MM_SCALE_4, _MM_HINT_NONE));
}
template<typename T> template<typename OtherT, typename Index> inline void ALWAYS_INLINE FLATTEN Vector<T>::gather(const OtherT *mem, const Index *indexes, Mask mask)
{
    gather(mem, Vector<Index>(indexes, Vc::Unaligned), mask);
}
template<typename T> template<typename OtherT, typename Index> inline void ALWAYS_INLINE FLATTEN Vector<T>::gather(const OtherT *mem, Vector<Index> indexes, Mask mask)
{
    IndexSizeChecker<Index, Size>::check();
    d.v() = lrb_cast<VectorType>(_mm512_mask_gatherd(lrb_cast<__m512>(data()), mask.data(), indexes.data(), const_cast<OtherT *>(mem), _MM_FULLUPC_NONE, _MM_SCALE_4, _MM_HINT_NONE));
}

template<> template<typename OtherT, typename S1, typename IT>
inline void ALWAYS_INLINE FLATTEN Vector<double>::gather(const S1 *array, const OtherT S1::* member1, const IT *indexes)
{
    // FIXME: this will read twice as much from memory as required
    gather(array, member1, Vector<IT>(indexes, Vc::Unaligned));
}
template<> template<typename OtherT, typename S1, typename IT>
inline void ALWAYS_INLINE FLATTEN Vector<double>::gather(const S1 *array, const OtherT S1::* member1, Vector<IT> indexes)
{
    IndexSizeChecker<IT, Size>::check();
    enum { Scale = sizeof(S1) / 4 };
    VC_STATIC_ASSERT((sizeof(S1) % Scale) == 0, Struct_size_needs_to_be_a_multiple_of_4);
    _M512I i = indexes.data();
    prepareDoubleGatherIndexes<Scale>(i);
    d.v() = lrb_cast<VectorType>(_mm512_gatherd(i, const_cast<OtherT *>(&(array->*member1)), _MM_FULLUPC_NONE, _MM_SCALE_4, _MM_HINT_NONE));
}
template<typename T> template<typename OtherT, typename S1, typename IT>
inline void ALWAYS_INLINE FLATTEN Vector<T>::gather(const S1 *array, const OtherT S1::* member1, const IT *indexes)
{
    gather(array, member1, Vector<IT>(indexes, Vc::Unaligned));
}
template<typename T> template<typename OtherT, typename S1, typename IT>
inline void ALWAYS_INLINE FLATTEN Vector<T>::gather(const S1 *array, const OtherT S1::* member1, Vector<IT> indexes)
{
    IndexSizeChecker<IT, Size>::check();
    enum { Scale = sizeof(S1) / sizeof(OtherT) };
    VC_STATIC_ASSERT((sizeof(S1) % Scale) == 0, Struct_size_needs_to_be_a_multiple_of_the_gathered_member_size);
    d.v() = lrb_cast<VectorType>(_mm512_gatherd((indexes * Scale).vdata(), const_cast<OtherT *>(&(array->*member1)),
                _MM_FULLUPC_NONE, IndexScale<OtherT>::value(), _MM_HINT_NONE));
}

template<> template<typename OtherT, typename S1, typename IT>
inline void ALWAYS_INLINE FLATTEN Vector<double>::gather(const S1 *array, const OtherT S1::* member1, const IT *indexes, Mask mask)
{
    // FIXME: this will read twice as much from memory as required
    gather(array, member1, Vector<IT>(indexes, Vc::Unaligned), mask);
}
template<> template<typename OtherT, typename S1, typename IT>
inline void ALWAYS_INLINE FLATTEN Vector<double>::gather(const S1 *array, const OtherT S1::* member1, Vector<IT> indexes, Mask mask)
{
    IndexSizeChecker<IT, Size>::check();
    enum { Scale = sizeof(S1) / 4 };
    VC_STATIC_ASSERT((sizeof(S1) % Scale) == 0, Struct_size_needs_to_be_a_multiple_of_4);
    _M512I i = indexes.data();
    prepareDoubleGatherIndexes<Scale>(i);
    d.v() = lrb_cast<VectorType>(_mm512_mask_gatherd(lrb_cast<__m512>(d.v()), _mm_bitinterleave11_16(mask.data(), mask.data()), i, const_cast<OtherT *>(&(array->*member1)), _MM_FULLUPC_NONE,
                _MM_SCALE_4, _MM_HINT_NONE));
}
template<typename T> template<typename OtherT, typename S1, typename IT>
inline void ALWAYS_INLINE FLATTEN Vector<T>::gather(const S1 *array, const OtherT S1::* member1, const IT *indexes, Mask mask)
{
    gather(array, member1, Vector<IT>(indexes, Vc::Unaligned), mask);
}
template<typename T> template<typename OtherT, typename S1, typename IT>
inline void ALWAYS_INLINE FLATTEN Vector<T>::gather(const S1 *array, const OtherT S1::* member1, Vector<IT> indexes, Mask mask)
{
    IndexSizeChecker<IT, Size>::check();
    enum { Scale = sizeof(S1) / sizeof(OtherT) };
    VC_STATIC_ASSERT((sizeof(S1) % Scale) == 0, Struct_size_needs_to_be_a_multiple_of_the_gathered_member_size);
    d.v() = lrb_cast<VectorType>(_mm512_mask_gatherd(lrb_cast<__m512>(d.v()), mask.data(), (indexes * Scale).vdata(), const_cast<OtherT *>(&(array->*member1)), _MM_FULLUPC_NONE,
                IndexScale<OtherT>::value(), _MM_HINT_NONE));
}


template<> template<typename OtherT, typename S1, typename S2, typename IT>
inline void ALWAYS_INLINE FLATTEN Vector<double>::gather(const S1 *array, const S2 S1::* member1, const OtherT S2::* member2,
        const IT *indexes)
{
    // FIXME: this will read twice as much from memory as required
    gather(array, member1, Vector<IT>(indexes, Vc::Unaligned));
}
template<> template<typename OtherT, typename S1, typename S2, typename IT>
inline void ALWAYS_INLINE FLATTEN Vector<double>::gather(const S1 *array, const S2 S1::* member1, const OtherT S2::* member2,
        Vector<IT> indexes)
{
    IndexSizeChecker<IT, Size>::check();
    enum { Scale = sizeof(S1) / 4 };
    VC_STATIC_ASSERT((sizeof(S1) % Scale) == 0, Struct_size_needs_to_be_a_multiple_of_4);
    _M512I i = indexes.data();
    prepareDoubleGatherIndexes<Scale>(i);
    d.v() = lrb_cast<VectorType>(_mm512_gatherd(i, const_cast<OtherT *>(&(array->*(member1).*(member2))), _MM_FULLUPC_NONE, _MM_SCALE_4, _MM_HINT_NONE));
}
template<typename T> template<typename OtherT, typename S1, typename S2, typename IT>
inline void ALWAYS_INLINE FLATTEN Vector<T>::gather(const S1 *array, const S2 S1::* member1, const OtherT S2::* member2,
        const IT *indexes)
{
    gather(array, member1, Vector<IT>(indexes, Vc::Unaligned));
}
template<typename T> template<typename OtherT, typename S1, typename S2, typename IT>
inline void ALWAYS_INLINE FLATTEN Vector<T>::gather(const S1 *array, const S2 S1::* member1, const OtherT S2::* member2,
        Vector<IT> indexes)
{
    IndexSizeChecker<IT, Size>::check();
    enum { Scale = sizeof(S1) / sizeof(OtherT) };
    VC_STATIC_ASSERT((sizeof(S1) % Scale) == 0, Struct_size_needs_to_be_a_multiple_of_the_gathered_member_size);
    d.v() = lrb_cast<VectorType>(_mm512_gatherd((indexes * Scale).data(), const_cast<OtherT *>(&(array->*member1.*member2)), _MM_FULLUPC_NONE,
                IndexScale<OtherT>::value(), _MM_HINT_NONE));
}

template<> template<typename OtherT, typename S1, typename S2, typename IT>
inline void ALWAYS_INLINE FLATTEN Vector<double>::gather(const S1 *array, const S2 S1::* member1, const OtherT S2::* member2,
        const IT *indexes, Mask mask)
{
    // FIXME: this will read twice as much from memory as required
    gather(array, member1, Vector<IT>(indexes, Vc::Unaligned), mask);
}
template<> template<typename OtherT, typename S1, typename S2, typename IT>
inline void ALWAYS_INLINE FLATTEN Vector<double>::gather(const S1 *array, const S2 S1::* member1, const OtherT S2::* member2,
        Vector<IT> indexes, Mask mask)
{
    IndexSizeChecker<IT, Size>::check();
    enum { Scale = sizeof(S1) / 4 };
    VC_STATIC_ASSERT((sizeof(S1) % Scale) == 0, Struct_size_needs_to_be_a_multiple_of_4);
    _M512I i = indexes.data();
    prepareDoubleGatherIndexes<Scale>(i);
    d.v() = lrb_cast<VectorType>(_mm512_mask_gatherd(lrb_cast<__m512>(d.v()), _mm_bitinterleave11_16(mask.data(), mask.data()), i, const_cast<OtherT *>(&(array->*(member1).*(member2))), _MM_FULLUPC_NONE,
                _MM_SCALE_4, _MM_HINT_NONE));
}
template<typename T> template<typename OtherT, typename S1, typename S2, typename IT>
inline void ALWAYS_INLINE FLATTEN Vector<T>::gather(const S1 *array, const S2 S1::* member1, const OtherT S2::* member2,
        const IT *indexes, Mask mask)
{
    gather(array, member1, Vector<IT>(indexes, Vc::Unaligned), mask);
}
template<typename T> template<typename OtherT, typename S1, typename S2, typename IT>
inline void ALWAYS_INLINE FLATTEN Vector<T>::gather(const S1 *array, const S2 S1::* member1, const OtherT S2::* member2,
        Vector<IT> indexes, Mask mask)
{
    IndexSizeChecker<IT, Size>::check();
    enum { Scale = sizeof(S1) / sizeof(OtherT) };
    VC_STATIC_ASSERT((sizeof(S1) % Scale) == 0, Struct_size_needs_to_be_a_multiple_of_the_gathered_member_size);
    d.v() = lrb_cast<VectorType>(_mm512_mask_gatherd(lrb_cast<__m512>(d.v()), mask.data(), (indexes * Scale).vdata(), const_cast<OtherT *>(&(array->*member1.*member2)),
                _MM_FULLUPC_NONE, IndexScale<OtherT>::value(), _MM_HINT_NONE));
}

template<> template<typename OtherT, typename S1, typename IT1, typename IT2>
inline void ALWAYS_INLINE FLATTEN Vector<double>::gather(const S1 *array, const OtherT *const S1::* ptrMember1,
        const IT1 *outerIndexes, const IT2 *innerIndexes)
{
    gather(array, ptrMember1, Vector<IT1>(outerIndexes), Vector<IT2>(innerIndexes));
}
template<> template<typename OtherT, typename S1, typename IT1, typename IT2>
inline void ALWAYS_INLINE FLATTEN Vector<double>::gather(const S1 *array, const OtherT *const S1::* ptrMember1,
        Vector<IT1> outerIndexes, Vector<IT2> innerIndexes)
{
    IndexSizeChecker<IT1, Size>::check();
    IndexSizeChecker<IT2, Size>::check();
    enum { Scale = sizeof(S1) / 4 };
    VC_STATIC_ASSERT((sizeof(S1) % Scale) == 0, Struct_size_needs_to_be_a_multiple_of_4);

    // offsets = innerIndexes + (array[outerIndexes]->ptrMember1 - array[0]->ptrMember1) / sizeof(double)
    const _M512I oi = _mm512_mask_mull_pi(_mm512_setzero_pi(), 0x00ff, outerIndexes.data(), _mm512_set_1to16_pi(Scale));
    _M512I offsets = lrb_cast<_M512I>(_mm512_gatherd(oi, const_cast<OtherT const **const>(&(array->*ptrMember1)), _MM_FULLUPC_NONE, _MM_SCALE_4, _MM_HINT_NONE));
    offsets = _mm512_sub_pi(offsets, _mm512_set_1to16_pi(reinterpret_cast<const int &>(array->*ptrMember1)));
    offsets = _mm512_srl_pi(offsets, _mm512_set_1to16_pi(3)); // divide by sizeof(double)
    offsets = _mm512_add_pi(offsets, innerIndexes.data());

    prepareDoubleGatherIndexes<2>(offsets);
    d.v() = lrb_cast<VectorType>(_mm512_gatherd(offsets, const_cast<OtherT *>(array->*ptrMember1), _MM_FULLUPC_NONE,
                _MM_SCALE_4, _MM_HINT_NONE));
}
template<typename T> template<typename OtherT, typename S1, typename IT1, typename IT2>
inline void ALWAYS_INLINE FLATTEN Vector<T>::gather(const S1 *array, const OtherT *const S1::* ptrMember1,
        const IT1 *outerIndexes, const IT2 *innerIndexes)
{
    gather(array, ptrMember1, Vector<IT1>(outerIndexes), Vector<IT2>(innerIndexes));
}
template<typename T> template<typename OtherT, typename S1, typename IT1, typename IT2>
inline void ALWAYS_INLINE FLATTEN Vector<T>::gather(const S1 *array, const OtherT *const S1::* ptrMember1,
        Vector<IT1> outerIndexes, Vector<IT2> innerIndexes)
{
    IndexSizeChecker<IT1, Size>::check();
    IndexSizeChecker<IT2, Size>::check();
    enum { Scale = sizeof(S1) / 4 };
    VC_STATIC_ASSERT((sizeof(S1) % Scale) == 0, Struct_size_needs_to_be_a_multiple_of_4);

    // offsets = innerIndexes + (array[outerIndexes]->ptrMember1 - array[0]->ptrMember1) / sizeof(OtherT)
    const _M512I oi = _mm512_mull_pi(outerIndexes.data(), _mm512_set_1to16_pi(Scale));
    _M512I offsets = lrb_cast<_M512I>(_mm512_gatherd(oi, const_cast<OtherT const **const>(&(array->*ptrMember1)), _MM_FULLUPC_NONE, _MM_SCALE_4, _MM_HINT_NONE));
    offsets = _mm512_sub_pi(offsets, _mm512_set_1to16_pi(*reinterpret_cast<const int *>(&(array->*ptrMember1))));
    offsets = _mm512_srl_pi(offsets, _mm512_set_1to16_pi(sizeof(OtherT) == 4 ? 2 : (sizeof(OtherT) == 2) ? 1 : 0)); // divide by sizeof(OtherT)
    offsets = _mm512_add_pi(offsets, innerIndexes.data());

    d.v() = lrb_cast<VectorType>(_mm512_gatherd(offsets, const_cast<OtherT *>(array->*ptrMember1), _MM_FULLUPC_NONE,
                IndexScale<OtherT>::value(), _MM_HINT_NONE));
}

template<> template<typename OtherT, typename S1, typename IT1, typename IT2>
inline void ALWAYS_INLINE FLATTEN Vector<double>::gather(const S1 *array, const OtherT *const S1::* ptrMember1,
        const IT1 *outerIndexes, const IT2 *innerIndexes, Mask mask)
{
    gather(array, ptrMember1, Vector<IT1>(outerIndexes), Vector<IT2>(innerIndexes), mask);
}
template<> template<typename OtherT, typename S1, typename IT1, typename IT2>
inline void ALWAYS_INLINE FLATTEN Vector<double>::gather(const S1 *array, const OtherT *const S1::* ptrMember1,
        Vector<IT1> outerIndexes, Vector<IT2> innerIndexes, Mask mask)
{
    IndexSizeChecker<IT1, Size>::check();
    IndexSizeChecker<IT2, Size>::check();
    enum { Scale = sizeof(S1) / 4 };
    VC_STATIC_ASSERT((sizeof(S1) % Scale) == 0, Struct_size_needs_to_be_a_multiple_of_4);

    // offsets = innerIndexes + (array[outerIndexes]->ptrMember1 - array[0]->ptrMember1) / sizeof(double)
    const _M512I oi = _mm512_mask_mull_pi(_mm512_setzero_pi(), mask.data(), outerIndexes.data(), _mm512_set_1to16_pi(Scale));
    _M512I offsets = lrb_cast<_M512I>(_mm512_mask_gatherd(_mm512_setzero_ps(), mask.data(), oi,
                const_cast<OtherT const **const>(&(array->*ptrMember1)), _MM_FULLUPC_NONE, _MM_SCALE_4, _MM_HINT_NONE));
    offsets = _mm512_mask_sub_pi(_mm512_setzero_pi(), mask.data(), offsets,
            _mm512_set_1to16_pi(reinterpret_cast<const int &>(array->*ptrMember1)));
    offsets = _mm512_mask_srl_pi(_mm512_setzero_pi(), mask.data(), offsets, _mm512_set_1to16_pi(3)); // divide by sizeof(double)
    offsets = _mm512_mask_add_pi(_mm512_setzero_pi(), mask.data(), offsets, innerIndexes.data());

    prepareDoubleGatherIndexes<2>(offsets);
    d.v() = lrb_cast<VectorType>(_mm512_mask_gatherd(lrb_cast<__m512>(d.v()), _mm_bitinterleave11_16(mask.data(), mask.data()), offsets,
                const_cast<OtherT *>(array->*ptrMember1), _MM_FULLUPC_NONE, _MM_SCALE_4, _MM_HINT_NONE));
}
template<typename T> template<typename OtherT, typename S1, typename IT1, typename IT2>
inline void ALWAYS_INLINE FLATTEN Vector<T>::gather(const S1 *array, const OtherT *const S1::* ptrMember1,
        const IT1 *outerIndexes, const IT2 *innerIndexes, Mask mask)
{
    gather(array, ptrMember1, Vector<IT1>(outerIndexes), Vector<IT2>(innerIndexes), mask);
}
template<typename T> template<typename OtherT, typename S1, typename IT1, typename IT2>
inline void ALWAYS_INLINE FLATTEN Vector<T>::gather(const S1 *array, const OtherT *const S1::* ptrMember1,
        Vector<IT1> outerIndexes, Vector<IT2> innerIndexes, Mask mask)
{
    IndexSizeChecker<IT1, Size>::check();
    IndexSizeChecker<IT2, Size>::check();
    enum { Scale = sizeof(S1) / 4 };
    VC_STATIC_ASSERT((sizeof(S1) % Scale) == 0, Struct_size_needs_to_be_a_multiple_of_4);

    // offsets = innerIndexes + (array[outerIndexes]->ptrMember1 - array[0]->ptrMember1) / sizeof(OtherT)
    const _M512I oi = _mm512_mask_mull_pi(_mm512_setzero_pi(), mask.data(), outerIndexes.data(), _mm512_set_1to16_pi(Scale));
    VC_DEBUG << oi;
    _M512I offsets = lrb_cast<_M512I>(_mm512_mask_gatherd(_mm512_setzero_ps(), mask.data(), oi,
               const_cast<OtherT const **const>(&(array->*ptrMember1)),
                _MM_FULLUPC_NONE, _MM_SCALE_4, _MM_HINT_NONE));
    VC_DEBUG << offsets;
    offsets = _mm512_mask_sub_pi(_mm512_setzero_pi(), mask.data(), offsets,
            _mm512_set_1to16_pi(*reinterpret_cast<const int *>(&(array->*ptrMember1))));
    VC_DEBUG << offsets;
    offsets = _mm512_mask_srl_pi(_mm512_setzero_pi(), mask.data(), offsets,
            _mm512_set_1to16_pi(sizeof(OtherT) == 4 ? 2 : (sizeof(OtherT) == 2) ? 1 : 0)); // divide by sizeof(OtherT)
    VC_DEBUG << offsets;
    offsets = _mm512_mask_add_pi(_mm512_setzero_pi(), mask.data(), offsets, innerIndexes.data());
    VC_DEBUG << offsets;

    d.v() = lrb_cast<VectorType>(_mm512_mask_gatherd(lrb_cast<__m512>(d.v()), mask.data(), offsets, const_cast<OtherT *>(array->*ptrMember1), _MM_FULLUPC_NONE,
                IndexScale<OtherT>::value(), _MM_HINT_NONE));
}

// scatters {{{1
template<typename T> template<typename OtherT, typename Index> inline void ALWAYS_INLINE FLATTEN Vector<T>::scatter(
        OtherT *mem, const Index *indexes) const
{
    scatter(mem, Vector<Index>(indexes, Vc::Unaligned));
}
template<typename T> template<typename OtherT, typename Index> inline void ALWAYS_INLINE FLATTEN Vector<T>::scatter(
        OtherT *mem, Vector<Index> indexes) const
{
    IndexSizeChecker<Index, Size>::check();
    _mm512_scatterd(mem, indexes.data(), lrb_cast<__m512>(d.v()), _MM_DOWNC_NONE, IndexScale<OtherT>::value(), _MM_HINT_NONE);
}
template<> template<typename OtherT, typename Index> inline void ALWAYS_INLINE FLATTEN Vector<double>::scatter(
        OtherT *mem, Vector<Index> indexes) const
{
    IndexSizeChecker<Index, Size>::check();
    _M512I i = indexes.data();
    prepareDoubleGatherIndexes<2>(i);
    _mm512_scatterd(mem, i, lrb_cast<__m512>(d.v()), _MM_DOWNC_NONE, _MM_SCALE_4, _MM_HINT_NONE);
}

template<typename T> template<typename OtherT, typename Index> inline void ALWAYS_INLINE FLATTEN Vector<T>::scatter(
        OtherT *mem, const Index *indexes, Mask mask) const
{
    scatter(mem, Vector<Index>(indexes, Vc::Unaligned), mask);
}
template<typename T> template<typename OtherT, typename Index> inline void ALWAYS_INLINE FLATTEN Vector<T>::scatter(
        OtherT *mem, Vector<Index> indexes, Mask mask) const
{
    IndexSizeChecker<Index, Size>::check();
    _mm512_mask_scatterd(mem, mask.data(), indexes.data(), lrb_cast<__m512>(d.v()), _MM_DOWNC_NONE, IndexScale<OtherT>::value(), _MM_HINT_NONE);
}
template<> template<typename OtherT, typename Index> inline void ALWAYS_INLINE FLATTEN Vector<double>::scatter(
        OtherT *mem, Vector<Index> indexes, Mask mask) const
{
    IndexSizeChecker<Index, Size>::check();
    _M512I i = indexes.data();
    prepareDoubleGatherIndexes<2>(i);
    _mm512_mask_scatterd(mem, _mm_bitinterleave11_16(mask.data(), mask.data()), i, lrb_cast<__m512>(d.v()), _MM_DOWNC_NONE, _MM_SCALE_4, _MM_HINT_NONE);
}

template<typename T> template<typename OtherT, typename S1, typename IT> inline void ALWAYS_INLINE FLATTEN Vector<T>::scatter(
        S1 *array, OtherT S1::* member1, const IT *indexes) const
{
    scatter(array, member1, Vector<IT>(indexes, Vc::Unaligned));
}
template<typename T> template<typename OtherT, typename S1, typename IT> inline void ALWAYS_INLINE FLATTEN Vector<T>::scatter(
        S1 *array, OtherT S1::* member1, Vector<IT> indexes) const
{
    IndexSizeChecker<IT, Size>::check();
    enum { Scale = sizeof(S1) / sizeof(OtherT) };
    VC_STATIC_ASSERT((sizeof(S1) % Scale) == 0, Struct_size_needs_to_be_a_multiple_of_the_gathered_member_size);
    _mm512_scatterd(&(array->*member1), (indexes * Scale).vdata(), lrb_cast<__m512>(d.v()), _MM_DOWNC_NONE,
            IndexScale<OtherT>::value(), _MM_HINT_NONE);
}
template<> template<typename OtherT, typename S1, typename IT> inline void ALWAYS_INLINE FLATTEN Vector<double>::scatter(
        S1 *array, OtherT S1::* member1, Vector<IT> indexes) const
{
    IndexSizeChecker<IT, Size>::check();
    enum { Scale = sizeof(S1) / 4 };
    VC_STATIC_ASSERT((sizeof(S1) % Scale) == 0, Struct_size_needs_to_be_a_multiple_of_4);
    _M512I i = indexes.data();
    prepareDoubleGatherIndexes<Scale>(i);
    _mm512_scatterd(&(array->*member1), i, lrb_cast<__m512>(d.v()), _MM_DOWNC_NONE, _MM_SCALE_4, _MM_HINT_NONE);
}

template<typename T> template<typename OtherT, typename S1, typename IT> inline void ALWAYS_INLINE FLATTEN Vector<T>::scatter(
        S1 *array, OtherT S1::* member1, const IT *indexes, Mask mask) const
{
    scatter(array, member1, Vector<IT>(indexes, Vc::Unaligned), mask);
}
template<typename T> template<typename OtherT, typename S1, typename IT> inline void ALWAYS_INLINE FLATTEN Vector<T>::scatter(
        S1 *array, OtherT S1::* member1, Vector<IT> indexes, Mask mask) const
{
    IndexSizeChecker<IT, Size>::check();
    enum { Scale = sizeof(S1) / sizeof(OtherT) };
    VC_STATIC_ASSERT((sizeof(S1) % Scale) == 0, Struct_size_needs_to_be_a_multiple_of_the_gathered_member_size);
    _mm512_mask_scatterd(&(array->*member1), mask.data(), (indexes * Scale).vdata(), lrb_cast<__m512>(d.v()), _MM_DOWNC_NONE,
            IndexScale<OtherT>::value(), _MM_HINT_NONE);
}
template<> template<typename OtherT, typename S1, typename IT> inline void ALWAYS_INLINE FLATTEN Vector<double>::scatter(
        S1 *array, OtherT S1::* member1, Vector<IT> indexes, Mask mask) const
{
    IndexSizeChecker<IT, Size>::check();
    enum { Scale = sizeof(S1) / 4 };
    VC_STATIC_ASSERT((sizeof(S1) % Scale) == 0, Struct_size_needs_to_be_a_multiple_of_4);
    _M512I i = indexes.data();
    prepareDoubleGatherIndexes<Scale>(i);
    _mm512_mask_scatterd(&(array->*member1), _mm_bitinterleave11_16(mask.data(), mask.data()), i, lrb_cast<__m512>(d.v()), _MM_DOWNC_NONE,
            _MM_SCALE_4, _MM_HINT_NONE);
}

template<typename T> template<typename OtherT, typename S1, typename S2, typename IT> inline void ALWAYS_INLINE FLATTEN Vector<T>::scatter(
        S1 *array, S2 S1::* member1, OtherT S2::* member2, const IT *indexes) const
{
    scatter(array, member1, member2, Vector<IT>(indexes, Vc::Unaligned));
}
template<typename T> template<typename OtherT, typename S1, typename S2, typename IT> inline void ALWAYS_INLINE FLATTEN Vector<T>::scatter(
        S1 *array, S2 S1::* member1, OtherT S2::* member2, Vector<IT> indexes) const
{
    IndexSizeChecker<IT, Size>::check();
    enum { Scale = sizeof(S1) / sizeof(OtherT) };
    VC_STATIC_ASSERT((sizeof(S1) % Scale) == 0, Struct_size_needs_to_be_a_multiple_of_the_gathered_member_size);
    _mm512_scatterd(&(array->*member1.*member2), (indexes * Scale).vdata(), lrb_cast<__m512>(d.v()), _MM_DOWNC_NONE,
            IndexScale<OtherT>::value(), _MM_HINT_NONE);
}
template<> template<typename OtherT, typename S1, typename S2, typename IT> inline void ALWAYS_INLINE FLATTEN Vector<double>::scatter(
        S1 *array, S2 S1::* member1, OtherT S2::* member2, Vector<IT> indexes) const
{
    IndexSizeChecker<IT, Size>::check();
    enum { Scale = sizeof(S1) / 4 };
    VC_STATIC_ASSERT((sizeof(S1) % Scale) == 0, Struct_size_needs_to_be_a_multiple_of_4);
    _M512I i = indexes.data();
    prepareDoubleGatherIndexes<Scale>(i);
    _mm512_scatterd(&(array->*member1.*member2), i, lrb_cast<__m512>(d.v()), _MM_DOWNC_NONE, _MM_SCALE_4, _MM_HINT_NONE);
}

template<typename T> template<typename OtherT, typename S1, typename S2, typename IT> inline void ALWAYS_INLINE FLATTEN Vector<T>::scatter(
        S1 *array, S2 S1::* member1, OtherT S2::* member2, const IT *indexes, Mask mask) const
{
    scatter(array, member1, member2, Vector<IT>(indexes, Vc::Unaligned), mask);
}
template<typename T> template<typename OtherT, typename S1, typename S2, typename IT> inline void ALWAYS_INLINE FLATTEN Vector<T>::scatter(
        S1 *array, S2 S1::* member1, OtherT S2::* member2, Vector<IT> indexes, Mask mask) const
{
    IndexSizeChecker<IT, Size>::check();
    enum { Scale = sizeof(S1) / sizeof(OtherT) };
    VC_STATIC_ASSERT((sizeof(S1) % Scale) == 0, Struct_size_needs_to_be_a_multiple_of_the_gathered_member_size);
    _mm512_mask_scatterd(&(array->*member1.*member2), mask.data(), (indexes * Scale).vdata(), lrb_cast<__m512>(d.v()), _MM_DOWNC_NONE,
            IndexScale<OtherT>::value(), _MM_HINT_NONE);
}
template<> template<typename OtherT, typename S1, typename S2, typename IT> inline void ALWAYS_INLINE FLATTEN Vector<double>::scatter(
        S1 *array, S2 S1::* member1, OtherT S2::* member2, Vector<IT> indexes, Mask mask) const
{
    IndexSizeChecker<IT, Size>::check();
    enum { Scale = sizeof(S1) / 4 };
    VC_STATIC_ASSERT((sizeof(S1) % Scale) == 0, Struct_size_needs_to_be_a_multiple_of_4);
    _M512I i = indexes.data();
    prepareDoubleGatherIndexes<Scale>(i);
    _mm512_mask_scatterd(&(array->*member1.*member2), _mm_bitinterleave11_16(mask.data(), mask.data()), i, lrb_cast<__m512>(d.v()),
            _MM_DOWNC_NONE, _MM_SCALE_4, _MM_HINT_NONE);
}

} // namespace LRBni
} // namespace Vc

// vim: foldmethod=marker
