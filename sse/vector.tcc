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

#include "macros.h"

namespace Vc
{
namespace SSE
{

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

template<typename T> inline void Vector<T>::load(const EntryType *mem)
{
    data() = VectorHelper<VectorType>::load(mem, Aligned);
}

template<typename T> template<typename A> inline void Vector<T>::load(const EntryType *mem, A align)
{
    data() = VectorHelper<VectorType>::load(mem, align);
}

template<typename T> inline void Vector<T>::setZero()
{
    data() = VectorHelper<VectorType>::zero();
}

template<typename T> inline void Vector<T>::setZero(const Mask &k)
{
    data() = VectorHelper<VectorType>::andnot_(mm128_reinterpret_cast<VectorType>(k.data()), data());
}

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

template<typename T> template<typename S1, typename IT1, typename IT2> inline ALWAYS_INLINE Vector<T>::Vector(const S1 *array, const EntryType *const S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes)
{
    gather(array, ptrMember1, outerIndexes, innerIndexes);
}

template<typename T, size_t Size> struct IndexSizeChecker { static void check() {} };
template<typename T, size_t Size> struct IndexSizeChecker<Vector<T>, Size>
{
    static void check() {
        VC_STATIC_ASSERT(Vector<T>::Size >= Size, IndexVector_must_have_greater_or_equal_number_of_entries);
    }
};
template<> template<typename S1, typename IT1, typename IT2>
inline void ALWAYS_INLINE FLATTEN Vector<double>::gather(const S1 *array, const EntryType *const S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes)
{
    IndexSizeChecker<IT1, Size>::check();
    IndexSizeChecker<IT2, Size>::check();
    d.v() = _mm_setr_pd((array[outerIndexes[0]].*(ptrMember1))[innerIndexes[0]], (array[outerIndexes[1]].*(ptrMember1))[innerIndexes[1]]);
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

} // namespace SSE
} // namespace Vc

#include "undomacros.h"
