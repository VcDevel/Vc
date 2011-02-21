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
template<typename T> inline Vector<T>::Vector(VectorSpecialInitializerZero::ZEnum) : Base(HT::zero()) {}
template<typename T> inline Vector<T>::Vector(VectorSpecialInitializerOne::OEnum) : Base(HT::one()) {}
template<typename T> inline Vector<T>::Vector(VectorSpecialInitializerIndexesFromZero::IEnum)
    : Base(HV::load(Base::_IndexesFromZero(), Aligned)) {}

template<typename T> inline Vector<T> Vector<T>::Zero() { return HT::zero(); }
template<typename T> inline Vector<T> Vector<T>::One() { return HT::one(); }
template<typename T> inline Vector<T> Vector<T>::IndexesFromZero() { return HV::load(Base::_IndexesFromZero(), Aligned); }

template<typename T> template<typename T2> inline Vector<T>::Vector(Vector<T2> x)
    : Base(StaticCastHelper<T2, T>::cast(x.data())) {}

template<typename T> inline Vector<T>::Vector(EntryType x) : Base(HT::set(x)) {}


///////////////////////////////////////////////////////////////////////////////////////////
// load ctors
template<typename T> inline Vector<T>::Vector(const EntryType *x)
    : Base(HV::load(x, Aligned)) {}

template<typename T> template<typename A> inline Vector<T>::Vector(const EntryType *x, A align)
    : Base(HV::load(x, align)) {}

template<typename T> inline Vector<T>::Vector(const Vector<typename HT::ConcatType> *x)
    : Base(HT::concat(x[0].data(), x[1].data())) {}

///////////////////////////////////////////////////////////////////////////////////////////
// load member functions
template<typename T> inline void Vector<T>::load(const EntryType *mem)
{
    data() = HV::load(mem, Aligned);
}
template<typename T> template<typename A> inline void Vector<T>::load(const EntryType *mem, A align)
{
    data() = HV::load(mem, align);
}

///////////////////////////////////////////////////////////////////////////////////////////
// zeroing
template<typename T> inline void Vector<T>::setZero()
{
    data() = HV::zero();
}
template<typename T> inline void Vector<T>::setZero(const Mask &k)
{
    data() = HV::andnot_(avx_cast<VectorType>(k.data()), data());
}

///////////////////////////////////////////////////////////////////////////////////////////
// stores
template<typename T> inline void Vector<T>::store(EntryType *mem) const
{
    HV::store(mem, data(), Aligned);
}
template<typename T> inline void Vector<T>::store(EntryType *mem, const Mask &mask) const
{
    HV::store(mem, data(), avx_cast<VectorType>(mask.data()), Aligned);
}
template<typename T> template<typename A> inline void Vector<T>::store(EntryType *mem, A align) const
{
    HV::store(mem, data(), align);
}
template<typename T> template<typename A> inline void Vector<T>::store(EntryType *mem, const Mask &mask, A align) const
{
    HV::store(mem, data(), avx_cast<VectorType>(mask.data()), align);
}

///////////////////////////////////////////////////////////////////////////////////////////
// swizzles
template<> inline const Vector<double> INTRINSIC CONST Vector<double>::aaaa() const { const double &tmp = d.m(0); return _mm256_broadcast_sd(&tmp); }
template<> inline const Vector<double> INTRINSIC CONST Vector<double>::bbbb() const { const double &tmp = d.m(1); return _mm256_broadcast_sd(&tmp); }
template<> inline const Vector<double> INTRINSIC CONST Vector<double>::cccc() const { const double &tmp = d.m(2); return _mm256_broadcast_sd(&tmp); }
template<> inline const Vector<double> INTRINSIC CONST Vector<double>::dddd() const { const double &tmp = d.m(3); return _mm256_broadcast_sd(&tmp); }

///////////////////////////////////////////////////////////////////////////////////////////
// operators
///////////////////////////////////////////////////////////////////////////////////////////
//// division
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
// per default fall back to scalar division
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
// specialize division on type
static inline __m256i INTRINSIC divInt(__m256i a, __m256i b) {
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
template<> inline Vector<int> Vector<int>::operator/(const Vector<int> &x) const
{
    return divInt(d.v(), x.d.v());
}
static inline __m256i INTRINSIC divUInt(__m256i a, __m256i b) {
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
    return concat(
            _mm256_cvttpd_epi32(_mm256_div_pd(loa, lob)),
            _mm256_cvttpd_epi32(_mm256_div_pd(hia, hib))
            );
}
template<> inline Vector<unsigned int> &Vector<unsigned int>::operator/=(const Vector<unsigned int> &x)
{
    d.v() = divUInt(d.v(), x.d.v());
    return *this;
}
template<> inline Vector<unsigned int> Vector<unsigned int>::operator/(const Vector<unsigned int> &x) const
{
    return divUInt(d.v(), x.d.v());
}
template<typename T> static inline __m128i INTRINSIC divShort(__m128i a, __m128i b)
{
    const __m256 r = _mm256_div_ps(StaticCastHelper<T, float>::cast(a),
            StaticCastHelper<T, float>::cast(b));
    return StaticCastHelper<float, T>::cast(r);
}
template<> inline Vector<short> &Vector<short>::operator/=(const Vector<short> &x)
{
    d.v() = divShort<short>(d.v(), x.d.v());
    return *this;
}
template<> inline Vector<short> Vector<short>::operator/(const Vector<short> &x) const
{
    return divShort<short>(d.v(), x.d.v());
}
template<> inline Vector<unsigned short> &Vector<unsigned short>::operator/=(const Vector<unsigned short> &x)
{
    d.v() = divShort<unsigned short>(d.v(), x.d.v());
    return *this;
}
template<> inline Vector<unsigned short> Vector<unsigned short>::operator/(const Vector<unsigned short> &x) const
{
    return divShort<unsigned short>(d.v(), x.d.v());
}
template<> inline Vector<float> &Vector<float>::operator/=(const Vector<float> &x)
{
    d.v() = _mm256_div_ps(d.v(), x.d.v());
    return *this;
}
template<> inline Vector<float> Vector<float>::operator/(const Vector<float> &x) const
{
    return _mm256_div_ps(d.v(), x.d.v());
}
template<> inline Vector<double> &Vector<double>::operator/=(const Vector<double> &x)
{
    d.v() = _mm256_div_pd(d.v(), x.d.v());
    return *this;
}
template<> inline Vector<double> Vector<double>::operator/(const Vector<double> &x) const
{
    return _mm256_div_pd(d.v(), x.d.v());
}

///////////////////////////////////////////////////////////////////////////////////////////
//// gathers
// Better implementation (hopefully) with _mm256_set_
//X template<typename T> template<typename Index> Vector<T>::Vector(const EntryType *mem, const Index *indexes)
//X {
//X     for_all_vector_entries(int i,
//X             d.m(i) = mem[indexes[i]];
//X             );
//X }
template<> template<typename Index> Vector<double>::Vector(const EntryType *mem, const Index *indexes)
    : Base(_mm256_set_pd(mem[indexes[0]], mem[indexes[1]], mem[indexes[2]], mem[indexes[3]])) {}

template<> template<typename Index> Vector<float>::Vector(const EntryType *mem, const Index *indexes)
    : Base(_mm256_set_ps(mem[indexes[0]], mem[indexes[1]], mem[indexes[2]], mem[indexes[3]],
                mem[indexes[4]], mem[indexes[5]], mem[indexes[6]], mem[indexes[7]])) {}

template<> template<typename Index> Vector<int>::Vector(const EntryType *mem, const Index *indexes)
    : Base(_mm256_set_epi32(mem[indexes[0]], mem[indexes[1]], mem[indexes[2]], mem[indexes[3]],
                mem[indexes[4]], mem[indexes[5]], mem[indexes[6]], mem[indexes[7]])) {}

template<> template<typename Index> Vector<unsigned int>::Vector(const EntryType *mem, const Index *indexes)
    : Base(_mm256_set_epi32(mem[indexes[0]], mem[indexes[1]], mem[indexes[2]], mem[indexes[3]],
                mem[indexes[4]], mem[indexes[5]], mem[indexes[6]], mem[indexes[7]])) {}

template<> template<typename Index> Vector<short>::Vector(const EntryType *mem, const Index *indexes)
    : Base(_mm_set_epi16(mem[indexes[0]], mem[indexes[1]], mem[indexes[2]], mem[indexes[3]],
                mem[indexes[4]], mem[indexes[5]], mem[indexes[6]], mem[indexes[7]])) {}

template<> template<typename Index> Vector<unsigned short>::Vector(const EntryType *mem, const Index *indexes)
    : Base(_mm_set_epi16(mem[indexes[0]], mem[indexes[1]], mem[indexes[2]], mem[indexes[3]],
                mem[indexes[4]], mem[indexes[5]], mem[indexes[6]], mem[indexes[7]])) {}

template<typename T> template<typename Index> Vector<T>::Vector(const EntryType *mem, const Index *indexes, Mask mask)
{
    d.v() = HT::zero();
    for_all_vector_entries(i,
            if (mask[i]) d.m(i) = mem[indexes[i]];
            );
}

} // namespace AVX
} // namespace Vc

#include "undomacros.h"
