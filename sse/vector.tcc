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

template<typename T> inline void Vector<T>::makeZero()
{
    data() = VectorHelper<VectorType>::zero();
}

template<typename T> inline void Vector<T>::makeZero(const Mask &k)
{
    data() = VectorHelper<VectorType>::andnot_(mm128_reinterpret_cast<VectorType>(k.data()), data());
}

template<typename T> inline void Vector<T>::store(EntryType *mem) const
{
    VectorHelper<VectorType>::store(mem, data(), Aligned);
}

template<typename T> inline void Vector<T>::store(EntryType *mem, const Mask &mask) const
{
    // this executes an unaligned store because SSE does not implement aligned masked stores
    VectorHelper<VectorType>::store(mem, data(), mm128_reinterpret_cast<VectorType>(mask.data()));
}

template<typename T> template<typename A> inline void Vector<T>::store(EntryType *mem, A align) const
{
    VectorHelper<VectorType>::store(mem, data(), align);
}

template<typename T> template<typename A> inline void Vector<T>::store(EntryType *mem, const Mask &mask, A) const
{
    store(mem, mask);
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
    __m128 lo = _mm_cvtepi32_ps(_mm_unpacklo_epi16(d.v(), _mm_setzero_si128()));
    __m128 hi = _mm_cvtepi32_ps(_mm_unpackhi_epi16(d.v(), _mm_setzero_si128()));
    lo = _mm_div_ps(lo, _mm_cvtepi32_ps(_mm_unpacklo_epi16(x.d.v(), _mm_setzero_si128())));
    hi = _mm_div_ps(hi, _mm_cvtepi32_ps(_mm_unpackhi_epi16(x.d.v(), _mm_setzero_si128())));
    d.v() = _mm_packs_epi32(_mm_cvtps_epi32(lo), _mm_cvtps_epi32(hi));
    return *this;
}

template<> inline Vector<short> Vector<short>::operator/(const Vector<short> &x) const
{
    __m128 lo = _mm_cvtepi32_ps(_mm_unpacklo_epi16(d.v(), _mm_setzero_si128()));
    __m128 hi = _mm_cvtepi32_ps(_mm_unpackhi_epi16(d.v(), _mm_setzero_si128()));
    lo = _mm_div_ps(lo, _mm_cvtepi32_ps(_mm_unpacklo_epi16(x.d.v(), _mm_setzero_si128())));
    hi = _mm_div_ps(hi, _mm_cvtepi32_ps(_mm_unpackhi_epi16(x.d.v(), _mm_setzero_si128())));
    return _mm_packs_epi32(_mm_cvtps_epi32(lo), _mm_cvtps_epi32(hi));
}

template<> inline Vector<unsigned short> &Vector<unsigned short>::operator/=(const Vector<unsigned short> &x)
{
    __m128 lo = _mm_cvtepi32_ps(_mm_unpacklo_epi16(d.v(), _mm_setzero_si128()));
    __m128 hi = _mm_cvtepi32_ps(_mm_unpackhi_epi16(d.v(), _mm_setzero_si128()));
    lo = _mm_div_ps(lo, _mm_cvtepi32_ps(_mm_unpacklo_epi16(x.d.v(), _mm_setzero_si128())));
    hi = _mm_div_ps(hi, _mm_cvtepi32_ps(_mm_unpackhi_epi16(x.d.v(), _mm_setzero_si128())));
    d.v() = _mm_packs_epi32(_mm_cvtps_epi32(lo), _mm_cvtps_epi32(hi));
    return *this;
}

template<> inline Vector<unsigned short> Vector<unsigned short>::operator/(const Vector<unsigned short> &x) const
{
    __m128 lo = _mm_cvtepi32_ps(_mm_unpacklo_epi16(d.v(), _mm_setzero_si128()));
    __m128 hi = _mm_cvtepi32_ps(_mm_unpackhi_epi16(d.v(), _mm_setzero_si128()));
    lo = _mm_div_ps(lo, _mm_cvtepi32_ps(_mm_unpacklo_epi16(x.d.v(), _mm_setzero_si128())));
    hi = _mm_div_ps(hi, _mm_cvtepi32_ps(_mm_unpackhi_epi16(x.d.v(), _mm_setzero_si128())));
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

#define OP_IMPL(T, symbol) \
template<> inline Vector<T> &VectorBase<T>::operator symbol##=(const VectorBase<T> &x) \
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

} // namespace SSE
} // namespace Vc

#include "undomacros.h"
