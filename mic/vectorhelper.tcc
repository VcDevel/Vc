/*  This file is part of the Vc library. {{{

    Copyright (C) 2010-2012 Matthias Kretz <kretz@kde.org>

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

Vc_NAMESPACE_BEGIN(Vc_IMPL_NAMESPACE)
// zero, one {{{1
template<> inline __m512  VectorHelper<__m512 >::zero() { return _mm512_setzero_ps(); }
template<> inline __m512d VectorHelper<__m512d>::zero() { return _mm512_setzero_pd(); }
template<> inline __m512i VectorHelper<__m512i>::zero() { return _mm512_setzero_epi32(); }

template<> inline __m512  VectorHelper<__m512 >::one() { return _mm512_set1_ps(1.f); }
template<> inline __m512d VectorHelper<__m512d>::one() { return _mm512_set1_pd(1.); }
template<> inline __m512i VectorHelper<__m512i>::one() { return _mm512_set1_epi32(1); }

// load and store {{{1
template<> inline __m512d VectorHelper<double>::load(const double *x, AlignedFlag)
{
    return _mm512_load_pd(x);
}

template<> inline __m512d VectorHelper<double>::load(const double *x, UnalignedFlag)
{
    return _mm512_loadu_pd(x);
}

template<> inline __m512d VectorHelper<double>::load(const double *x, StreamingAndAlignedFlag)
{
    return _mm512_extload_pd(x, _MM_UPCONV_PD_NONE, _MM_BROADCAST_8X8, _MM_HINT_NT);
}

template<> inline __m512d VectorHelper<double>::load(const double *x, StreamingAndUnalignedFlag)
{
    return _mm512_loadu_pd(x, _MM_UPCONV_PD_NONE, _MM_HINT_NT);
}

template<> inline void VectorHelper<double>::store(double *mem, VectorType x, AlignedFlag)
{
    _mm512_store_pd(mem, x);
}

template<> inline void VectorHelper<double>::store(double *mem, VectorType x, UnalignedFlag)
{
    _mm512_packstorehi_pd(mem, x);
    _mm512_packstorelo_pd(mem, x);
}

template<> inline void VectorHelper<double>::store(double *mem, VectorType x, StreamingAndAlignedFlag)
{
    _mm512_extstore_pd(mem, x, _MM_DOWNCONV_PD_NONE, _MM_HINT_NT);
}

template<> inline void VectorHelper<double>::store(double *mem, VectorType x, StreamingAndUnalignedFlag)
{
    _mm512_extpackstorehi_pd(mem, x, _MM_DOWNCONV_PD_NONE, _MM_HINT_NT);
    _mm512_extpackstorelo_pd(mem, x, _MM_DOWNCONV_PD_NONE, _MM_HINT_NT);
}

template<> inline void VectorHelper<double>::store(double *mem, VectorType x, __mmask8 k, AlignedFlag)
{
    _mm512_mask_store_pd(mem, k, x);
}

template<> inline void VectorHelper<double>::store(double *mem, VectorType x, __mmask8 k, UnalignedFlag)
{
    _mm512_mask_packstorehi_pd(mem, k, x);
    _mm512_mask_packstorelo_pd(mem, k, x);
}

template<> inline void VectorHelper<double>::store(double *mem, VectorType x, __mmask8 k, StreamingAndAlignedFlag)
{
    _mm512_mask_extstore_pd(mem, k, x, _MM_DOWNCONV_PD_NONE, _MM_HINT_NT);
}

template<> inline void VectorHelper<double>::store(double *mem, VectorType x, __mmask8 k, StreamingAndUnalignedFlag)
{
    _mm512_mask_extpackstorehi_pd(mem, k, x, _MM_DOWNCONV_PD_NONE, _MM_HINT_NT);
    _mm512_mask_extpackstorelo_pd(mem, k, x, _MM_DOWNCONV_PD_NONE, _MM_HINT_NT);
}


#define LOAD(T1, T2, SUFFIX) \
template<> inline VectorHelper<T1>::VectorType VectorHelper<T1>::load(const T2 *x, AlignedFlag) \
{ \
    return _mm512_extload_##SUFFIX(x, UpDownConversion<T1, T2>(), _MM_BROADCAST_16X16, _MM_HINT_NONE); \
} \
template<> inline VectorHelper<T1>::VectorType VectorHelper<T1>::load(const T2 *x, UnalignedFlag) \
{ \
    return _mm512_loadu_##SUFFIX(x, UpDownConversion<T1, T2>()); \
} \
template<> inline VectorHelper<T1>::VectorType VectorHelper<T1>::load(const T2 *x, StreamingAndAlignedFlag) \
{ \
    return _mm512_extload_##SUFFIX(x, UpDownConversion<T1, T2>(), _MM_BROADCAST_16X16, _MM_HINT_NT); \
} \
template<> inline VectorHelper<T1>::VectorType VectorHelper<T1>::load(const T2 *x, StreamingAndUnalignedFlag) \
{ \
    return _mm512_loadu_##SUFFIX(x, UpDownConversion<T1, T2>(), _MM_HINT_NT); \
}

#define STORE(T1, T2, SUFFIX) \
template<> inline void VectorHelper<T1>::store(T2 *mem, VectorType x, AlignedFlag) \
{ \
    _mm512_extstore_##SUFFIX(mem, x, UpDownConversion<T1, T2>(), _MM_HINT_NONE); \
} \
template<> inline void VectorHelper<T1>::store(T2 *mem, VectorType x, UnalignedFlag) \
{ \
    _mm512_extstore_##SUFFIX(mem, x, UpDownConversion<T1, T2>(), _MM_HINT_NONE); \
} \
template<> inline void VectorHelper<T1>::store(T2 *mem, VectorType x, StreamingAndAlignedFlag) \
{ \
    _mm512_extstore_##SUFFIX(mem, x, UpDownConversion<T1, T2>(), _MM_HINT_NT); \
} \
template<> inline void VectorHelper<T1>::store(T2 *mem, VectorType x, StreamingAndUnalignedFlag) \
{ \
    _mm512_extstore_##SUFFIX(mem, x, UpDownConversion<T1, T2>(), _MM_HINT_NT); \
} \
template<> inline void VectorHelper<T1>::store(T2 *mem, VectorType x, __mmask16 k, AlignedFlag) \
{ \
    _mm512_mask_extstore_##SUFFIX(mem, k, x, UpDownConversion<T1, T2>(), _MM_HINT_NONE); \
} \
template<> inline void VectorHelper<T1>::store(T2 *mem, VectorType x, __mmask16 k, UnalignedFlag) \
{ \
    _mm512_mask_extstore_##SUFFIX(mem, k, x, UpDownConversion<T1, T2>(), _MM_HINT_NONE); \
} \
template<> inline void VectorHelper<T1>::store(T2 *mem, VectorType x, __mmask16 k, StreamingAndAlignedFlag) \
{ \
    _mm512_mask_extstore_##SUFFIX(mem, k, x, UpDownConversion<T1, T2>(), _MM_HINT_NT); \
} \
template<> inline void VectorHelper<T1>::store(T2 *mem, VectorType x, __mmask16 k, StreamingAndUnalignedFlag) \
{ \
    _mm512_mask_extstore_##SUFFIX(mem, k, x, UpDownConversion<T1, T2>(), _MM_HINT_NT); \
}

LOAD(float, float, ps)
//LOAD(float, half_float, ps)
LOAD(float, unsigned char, ps)
LOAD(float, signed char, ps)
LOAD(float, unsigned short, ps)
LOAD(float, signed short, ps)
STORE(float, float, ps)
//STORE(float, half_float, ps)
STORE(float, unsigned char, ps)
STORE(float, signed char, ps)
STORE(float, unsigned short, ps)
STORE(float, signed short, ps)

LOAD(int, int, epi32)
LOAD(int, unsigned int, epi32)
LOAD(int, signed char, epi32)
LOAD(int, signed short, epi32)
LOAD(int, unsigned short, epi32)
LOAD(int, unsigned char, epi32)
STORE(int, int, epi32)
STORE(int, signed char, epi32)
STORE(int, signed short, epi32)

LOAD(unsigned int, unsigned int, epi32)
LOAD(unsigned int, unsigned char, epi32)
LOAD(unsigned int, unsigned short, epi32)
STORE(unsigned int, unsigned int, epi32)
STORE(unsigned int, unsigned char, epi32)
STORE(unsigned int, unsigned short, epi32)

#undef LOAD
#undef STORE

#define _Vc_float_impls(A) \
template<> inline __m512 VectorHelper<float>::load(const double *x, A) \
{ \
    __m512 r = _mm512_cvtpd_pslo(VectorHelper<double>::load(&x[0], Internal::FlagObject<A>::the())); \
    return _mm512_mask_permute4f128_ps(r, 0xff00, _mm512_cvtpd_pslo(VectorHelper<double>::load(&x[8], Internal::FlagObject<A>::the())), \
                _MM_PERM_BADC); \
} \
template<> inline __m512 VectorHelper<float>::load(const int *x, A) \
{ \
    return StaticCastHelper<int, float>::cast(VectorHelper<int>::load(x, Internal::FlagObject<A>::the())); \
} \
template<> inline __m512 VectorHelper<float>::load(const unsigned int *x, A) \
{ \
    return StaticCastHelper<unsigned int, float>::cast(VectorHelper<unsigned int>::load(x, Internal::FlagObject<A>::the())); \
}
_Vc_float_impls(AlignedFlag)
_Vc_float_impls(UnalignedFlag)
_Vc_float_impls(StreamingAndAlignedFlag)
_Vc_float_impls(StreamingAndUnalignedFlag)

//}}}1
Vc_NAMESPACE_END

// vim: foldmethod=marker
