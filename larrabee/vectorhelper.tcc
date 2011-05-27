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

namespace Vc
{
namespace LRBni
{

template<> inline _M512D VectorHelper<double>::load(const double *x, AlignedFlag)
{
    return lrb_cast<VectorType>(FixedIntrinsics::_mm512_loadq(x, _MM_FULLUPC64_NONE, _MM_BROADCAST_8X8));
}

template<> inline _M512D VectorHelper<double>::load(const double *x, UnalignedFlag)
{
    return lrb_cast<VectorType>(_mm512_expandq( const_cast<double *>(x), _MM_FULLUPC64_NONE, _MM_HINT_NONE));
}

template<> inline void VectorHelper<double>::store(void *mem, VectorType x, AlignedFlag)
{
    _mm512_storeq(mem, lrb_cast<_M512>(x), _MM_DOWNC64_NONE, _MM_SUBSET64_8, _MM_HINT_NONE);
}

template<> inline void VectorHelper<double>::store(void *mem, VectorType x, UnalignedFlag)
{
    _mm512_compressq(mem, lrb_cast<_M512>(x), _MM_DOWNC64_NONE, _MM_HINT_NONE);
}

template<> inline void VectorHelper<double>::store(void *mem, VectorType x, StreamingAndAlignedFlag)
{
    _mm512_storeq(mem, lrb_cast<_M512>(x), _MM_DOWNC64_NONE, _MM_SUBSET64_8, _MM_HINT_NT);
}

template<> inline void VectorHelper<double>::store(void *mem, VectorType x, StreamingAndUnalignedFlag)
{
    _mm512_compressq(mem, lrb_cast<_M512>(x), _MM_DOWNC64_NONE, _MM_HINT_NT);
}

template<> inline void VectorHelper<double>::store(void *mem, VectorType x, __mmask k, AlignedFlag)
{
    _mm512_mask_storeq(mem, k, lrb_cast<_M512>(x), _MM_DOWNC64_NONE, _MM_SUBSET64_8, _MM_HINT_NONE);
}

template<> inline void VectorHelper<double>::store(void *mem, VectorType x, __mmask k, UnalignedFlag)
{
    _mm512_mask_compressq(mem, k, lrb_cast<_M512>(x), _MM_DOWNC64_NONE, _MM_HINT_NONE);
}

template<> inline void VectorHelper<double>::store(void *mem, VectorType x, __mmask k, StreamingAndAlignedFlag)
{
    _mm512_mask_storeq(mem, k, lrb_cast<_M512>(x), _MM_DOWNC64_NONE, _MM_SUBSET64_8, _MM_HINT_NT);
}

template<> inline void VectorHelper<double>::store(void *mem, VectorType x, __mmask k, StreamingAndUnalignedFlag)
{
    _mm512_mask_compressq(mem, k, lrb_cast<_M512>(x), _MM_DOWNC64_NONE, _MM_HINT_NT);
}


#define LOAD(T1, T2, conv) \
template<> inline VectorHelper<T1>::VectorType VectorHelper<T1>::load(const T2 *x, AlignedFlag) \
{ \
    return lrb_cast<VectorType>(FixedIntrinsics::_mm512_loadd(x, conv, _MM_BROADCAST_16X16, _MM_HINT_NONE)); \
} \
template<> inline VectorHelper<T1>::VectorType VectorHelper<T1>::load(const T2 *x, UnalignedFlag) \
{ \
    return lrb_cast<VectorType>(_mm512_expandd(const_cast<T2 *>(x), conv, _MM_HINT_NONE)); \
}

#define STORE(T1, T2, conv) \
template<> inline void VectorHelper<T1>::store(T2 *mem, VectorType x, AlignedFlag) \
{ \
    _mm512_stored(mem, lrb_cast<_M512>(x), conv, _MM_SUBSET32_16, _MM_HINT_NONE); \
} \
template<> inline void VectorHelper<T1>::store(T2 *mem, VectorType x, UnalignedFlag) \
{ \
    _mm512_compressd(mem, lrb_cast<_M512>(x), conv, _MM_HINT_NONE); \
} \
template<> inline void VectorHelper<T1>::store(T2 *mem, VectorType x, StreamingAndAlignedFlag) \
{ \
    _mm512_stored(mem, lrb_cast<_M512>(x), conv, _MM_SUBSET32_16, _MM_HINT_NT); \
} \
template<> inline void VectorHelper<T1>::store(T2 *mem, VectorType x, StreamingAndUnalignedFlag) \
{ \
    _mm512_compressd(mem, lrb_cast<_M512>(x), conv, _MM_HINT_NT); \
} \
template<> inline void VectorHelper<T1>::store(T2 *mem, VectorType x, __mmask k, AlignedFlag) \
{ \
    _mm512_mask_stored(mem, k, lrb_cast<_M512>(x), conv, _MM_SUBSET32_16, _MM_HINT_NONE); \
} \
template<> inline void VectorHelper<T1>::store(T2 *mem, VectorType x, __mmask k, UnalignedFlag) \
{ \
    _mm512_mask_compressd(mem, k, lrb_cast<_M512>(x), conv, _MM_HINT_NONE); \
} \
template<> inline void VectorHelper<T1>::store(T2 *mem, VectorType x, __mmask k, StreamingAndAlignedFlag) \
{ \
    _mm512_mask_stored(mem, k, lrb_cast<_M512>(x), conv, _MM_SUBSET32_16, _MM_HINT_NT); \
} \
template<> inline void VectorHelper<T1>::store(T2 *mem, VectorType x, __mmask k, StreamingAndUnalignedFlag) \
{ \
    _mm512_mask_compressd(mem, k, lrb_cast<_M512>(x), conv, _MM_HINT_NT); \
}

LOAD(float, float, _MM_FULLUPC_NONE)
LOAD(float, float16, _MM_FULLUPC_FLOAT16)
LOAD(float, unsigned char, _MM_FULLUPC_UINT8)
LOAD(float, signed char, _MM_FULLUPC_SINT8)
LOAD(float, char, _MM_FULLUPC_SINT8)
LOAD(float, unsigned short, _MM_FULLUPC_UINT16)
LOAD(float, signed short, _MM_FULLUPC_SINT16)
STORE(float, float, _MM_DOWNC_NONE)
STORE(float, float16, _MM_DOWNC_FLOAT16)
STORE(float, unsigned char, _MM_DOWNC_UINT8)
STORE(float, signed char, _MM_DOWNC_SINT8)
STORE(float, unsigned short, _MM_DOWNC_UINT16)
STORE(float, signed short, _MM_DOWNC_SINT16)

LOAD(int, int, _MM_FULLUPC_NONE)
LOAD(int, unsigned int, _MM_FULLUPC_NONE)
LOAD(int, signed char, _MM_FULLUPC_SINT8I)
LOAD(int, signed short, _MM_FULLUPC_SINT16I)
LOAD(int, unsigned short, _MM_FULLUPC_UINT16I)
LOAD(int, unsigned char, _MM_FULLUPC_UINT8I)
STORE(int, int, _MM_DOWNC_NONE)
STORE(int, signed char, _MM_DOWNC_SINT8I)
STORE(int, signed short, _MM_DOWNC_SINT16I)

LOAD(unsigned int, unsigned int, _MM_FULLUPC_NONE)
LOAD(unsigned int, unsigned char, _MM_FULLUPC_UINT8I)
LOAD(unsigned int, unsigned short, _MM_FULLUPC_UINT16I)
STORE(unsigned int, unsigned int, _MM_DOWNC_NONE)
STORE(unsigned int, unsigned char, _MM_DOWNC_UINT8I)
STORE(unsigned int, unsigned short, _MM_DOWNC_UINT16I)

#undef LOAD
#undef STORE

template<typename A> inline _M512 VectorHelper<float>::load(const double *x, A flag)
{
    _M512 r = _mm512_setzero_ps();
    r = _mm512_cvtl_pd2ps(r, VectorHelper<double>::load(&x[0], flag), _MM_ROUND_MODE_NEAREST);
    r = _mm512_cvth_pd2ps(r, VectorHelper<double>::load(&x[8], flag), _MM_ROUND_MODE_NEAREST);
    return r;
}
template<typename A> inline _M512 VectorHelper<float>::load(const int *x, A flag)
{
    return StaticCastHelper<int, float>::cast(VectorHelper<int>::load(x, flag));
}
template<typename A> inline _M512 VectorHelper<float>::load(const unsigned int *x, A flag)
{
    return StaticCastHelper<unsigned int, float>::cast(VectorHelper<unsigned int>::load(x, flag));
}

} // namespace LRBni
} // namespace Vc
