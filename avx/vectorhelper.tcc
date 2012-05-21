/*  This file is part of the Vc library.

    Copyright (C) 2009-2011 Matthias Kretz <kretz@kde.org>

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

#include "casts.h"
#include <cstdlib>

namespace Vc
{
namespace AVX
{

////////////////////////////////////////////////////////////////////////////////////////////////////
// float_v
////////////////////////////////////////////////////////////////////////////////////////////////////
//// loads
template<> inline __m256 VectorHelper<__m256>::load(const float *m, AlignedFlag)
{
    return _mm256_load_ps(m);
}
template<> inline __m256 VectorHelper<__m256>::load(const float *m, UnalignedFlag)
{
    return _mm256_loadu_ps(m);
}
template<> inline __m256 VectorHelper<__m256>::load(const float *m, StreamingAndAlignedFlag)
{
    return avx_cast<__m256>(concat(_mm_stream_load_si128(reinterpret_cast<__m128i *>(const_cast<float *>(m))),
                _mm_stream_load_si128(reinterpret_cast<__m128i *>(const_cast<float *>(&m[4])))));
}
template<> inline __m256
    VC_WARN("AVX does not support streaming unaligned loads. Will use non-streaming unaligned load instead.")
VectorHelper<__m256>::load(const float *m, StreamingAndUnalignedFlag)
{
    return _mm256_loadu_ps(m);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
//// stores
inline void VectorHelper<__m256>::store(float *mem, const VectorType x, AlignedFlag)
{
    _mm256_store_ps(mem, x);
}
inline void VectorHelper<__m256>::store(float *mem, const VectorType x, UnalignedFlag)
{
    _mm256_storeu_ps(mem, x);
}
inline void VectorHelper<__m256>::store(float *mem, const VectorType x, StreamingAndAlignedFlag)
{
    _mm256_stream_ps(mem, x);
}
inline void VectorHelper<__m256>::store(float *mem, const VectorType x, StreamingAndUnalignedFlag)
{
    _mm_maskmoveu_si128(avx_cast<__m128i>(x), _mm_setallone_si128(), reinterpret_cast<char *>(mem));
    _mm_maskmoveu_si128(_mm256_extractf128_si256(avx_cast<__m256i>(x), 1), _mm_setallone_si128(), reinterpret_cast<char *>(mem + 4));
}
inline void VectorHelper<__m256>::store(float *mem, const VectorType x, const VectorType m, AlignedFlag)
{
    _mm256_maskstore(mem, m, x);
}
inline void VectorHelper<__m256>::store(float *mem, const VectorType x, const VectorType m, UnalignedFlag)
{
    _mm256_maskstore(mem, m, x);
}
inline void VectorHelper<__m256>::store(float *mem, const VectorType x, const VectorType m, StreamingAndAlignedFlag)
{
    _mm_maskmoveu_si128(avx_cast<__m128i>(x), avx_cast<__m128i>(m), reinterpret_cast<char *>(mem));
    _mm_maskmoveu_si128(_mm256_extractf128_si256(avx_cast<__m256i>(x), 1), _mm256_extractf128_si256(avx_cast<__m256i>(m), 1), reinterpret_cast<char *>(mem + 4));
}
inline void VectorHelper<__m256>::store(float *mem, const VectorType x, const VectorType m, StreamingAndUnalignedFlag)
{
    _mm_maskmoveu_si128(avx_cast<__m128i>(x), avx_cast<__m128i>(m), reinterpret_cast<char *>(mem));
    _mm_maskmoveu_si128(_mm256_extractf128_si256(avx_cast<__m256i>(x), 1), _mm256_extractf128_si256(avx_cast<__m256i>(m), 1), reinterpret_cast<char *>(mem + 4));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// double_v
////////////////////////////////////////////////////////////////////////////////////////////////////
//// loads
template<> inline __m256d VectorHelper<__m256d>::load(const double *m, AlignedFlag)
{
    return _mm256_load_pd(m);
}
template<> inline __m256d VectorHelper<__m256d>::load(const double *m, UnalignedFlag)
{
    return _mm256_loadu_pd(m);
}
template<> inline __m256d VectorHelper<__m256d>::load(const double *m, StreamingAndAlignedFlag)
{
    return avx_cast<__m256d>(concat(
                _mm_stream_load_si128(reinterpret_cast<__m128i *>(const_cast<double *>(m))),
                _mm_stream_load_si128(reinterpret_cast<__m128i *>(const_cast<double *>(&m[2])))));
}
template<> inline __m256d
    VC_WARN("AVX does not support streaming unaligned loads. Will use non-streaming unaligned load instead.")
VectorHelper<__m256d>::load(const double *m, StreamingAndUnalignedFlag)
{
    return _mm256_loadu_pd(m);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
//// stores
inline void VectorHelper<__m256d>::store(double *mem, const VectorType x, AlignedFlag)
{
    _mm256_store_pd(mem, x);
}
inline void VectorHelper<__m256d>::store(double *mem, const VectorType x, UnalignedFlag)
{
    _mm256_storeu_pd(mem, x);
}
inline void VectorHelper<__m256d>::store(double *mem, const VectorType x, StreamingAndAlignedFlag)
{
    _mm256_stream_pd(mem, x);
}
inline void VectorHelper<__m256d>::store(double *mem, const VectorType x, StreamingAndUnalignedFlag)
{
    _mm_maskmoveu_si128(avx_cast<__m128i>(x), _mm_setallone_si128(), reinterpret_cast<char *>(mem));
    _mm_maskmoveu_si128(avx_cast<__m128i>(_mm256_extractf128_pd(x, 1)), _mm_setallone_si128(), reinterpret_cast<char *>(mem + 2));
}
inline void VectorHelper<__m256d>::store(double *mem, const VectorType x, const VectorType m, AlignedFlag)
{
    _mm256_maskstore(mem, m, x);
}
inline void VectorHelper<__m256d>::store(double *mem, const VectorType x, const VectorType m, UnalignedFlag)
{
    _mm256_maskstore(mem, m, x);
}
inline void VectorHelper<__m256d>::store(double *mem, const VectorType x, const VectorType m, StreamingAndAlignedFlag)
{
    _mm_maskmoveu_si128(avx_cast<__m128i>(x), avx_cast<__m128i>(m), reinterpret_cast<char *>(mem));
    _mm_maskmoveu_si128(avx_cast<__m128i>(_mm256_extractf128_pd(x, 1)), avx_cast<__m128i>(_mm256_extractf128_pd(m, 1)), reinterpret_cast<char *>(mem + 2));
}
inline void VectorHelper<__m256d>::store(double *mem, const VectorType x, const VectorType m, StreamingAndUnalignedFlag)
{
    _mm_maskmoveu_si128(avx_cast<__m128i>(x), avx_cast<__m128i>(m), reinterpret_cast<char *>(mem));
    _mm_maskmoveu_si128(avx_cast<__m128i>(_mm256_extractf128_pd(x, 1)), avx_cast<__m128i>(_mm256_extractf128_pd(m, 1)), reinterpret_cast<char *>(mem + 2));
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// (u)int_v
////////////////////////////////////////////////////////////////////////////////////////////////////
//// loads
template<typename T> inline __m256i VectorHelper<__m256i>::load(const T *m, AlignedFlag)
{
    return _mm256_load_si256(reinterpret_cast<const __m256i *>(m));
}
template<typename T> inline __m256i VectorHelper<__m256i>::load(const T *m, UnalignedFlag)
{
    return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(m));
}
template<typename T> inline __m256i VectorHelper<__m256i>::load(const T *m, StreamingAndAlignedFlag)
{
    return concat(_mm_stream_load_si128(reinterpret_cast<__m128i *>(const_cast<T *>(m))),
            _mm_stream_load_si128(reinterpret_cast<__m128i *>(const_cast<T *>(&m[4]))));
}
template<typename T> inline __m256i
    VC_WARN("AVX does not support streaming unaligned loads. Will use non-streaming unaligned load instead.")
VectorHelper<__m256i>::load(const T *m, StreamingAndUnalignedFlag)
{
    return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(m));
}
////////////////////////////////////////////////////////////////////////////////////////////////////
//// stores
template<typename T> inline void VectorHelper<__m256i>::store(T *mem, const VectorType x, AlignedFlag)
{
    _mm256_store_si256(reinterpret_cast<VectorType *>(mem), x);
}
template<typename T> inline void VectorHelper<__m256i>::store(T *mem, const VectorType x, UnalignedFlag)
{
    _mm256_storeu_si256(reinterpret_cast<VectorType *>(mem), x);
}
template<typename T> inline void VectorHelper<__m256i>::store(T *mem, const VectorType x, StreamingAndAlignedFlag)
{
    _mm256_stream_si256(reinterpret_cast<VectorType *>(mem), x);
}
template<typename T> inline void VectorHelper<__m256i>::store(T *mem, const VectorType x, StreamingAndUnalignedFlag)
{
    _mm_maskmoveu_si128(avx_cast<__m128i>(x), _mm_setallone_si128(), reinterpret_cast<char *>(mem));
    _mm_maskmoveu_si128(_mm256_extractf128_si256(x, 1), _mm_setallone_si128(), reinterpret_cast<char *>(mem + 4));
}
template<typename T> inline void VectorHelper<__m256i>::store(T *mem, const VectorType x, const VectorType m, AlignedFlag)
{
    _mm256_maskstore(mem, m, x);
}
template<typename T> inline void VectorHelper<__m256i>::store(T *mem, const VectorType x, const VectorType m, UnalignedFlag)
{
    _mm256_maskstore(mem, m, x);
}
template<typename T> inline void VectorHelper<__m256i>::store(T *mem, const VectorType x, const VectorType m, StreamingAndAlignedFlag)
{
    _mm_maskmoveu_si128(lo128(x), lo128(m), reinterpret_cast<char *>(mem));
    _mm_maskmoveu_si128(hi128(x), hi128(m), reinterpret_cast<char *>(mem + 4));
}
template<typename T> inline void VectorHelper<__m256i>::store(T *mem, const VectorType x, const VectorType m, StreamingAndUnalignedFlag)
{
    _mm_maskmoveu_si128(lo128(x), lo128(m), reinterpret_cast<char *>(mem));
    _mm_maskmoveu_si128(hi128(x), hi128(m), reinterpret_cast<char *>(mem + 4));
}
////////////////////////////////////////////////////////////////////////////////////////////////////
// (u)short_v
////////////////////////////////////////////////////////////////////////////////////////////////////
//// loads
template<typename T> inline __m128i VectorHelper<__m128i>::load(const T *m, AlignedFlag)
{
    return _mm_load_si128(reinterpret_cast<const __m128i *>(m));
}
template<typename T> inline __m128i VectorHelper<__m128i>::load(const T *m, UnalignedFlag)
{
    return _mm_loadu_si128(reinterpret_cast<const __m128i *>(m));
}
template<typename T> inline __m128i VectorHelper<__m128i>::load(const T *m, StreamingAndAlignedFlag)
{
    return _mm_stream_load_si128(reinterpret_cast<__m128i *>(const_cast<T *>(m)));
}
template<typename T> inline __m128i
    VC_WARN("AVX does not support streaming unaligned loads. Will use non-streaming unaligned load instead.")
VectorHelper<__m128i>::load(const T *m, StreamingAndUnalignedFlag)
{
    return _mm_loadu_si128(reinterpret_cast<const __m128i *>(m));
}
////////////////////////////////////////////////////////////////////////////////////////////////////
//// stores
template<typename T> inline void VectorHelper<__m128i>::store(T *mem, const VectorType x, AlignedFlag)
{
    _mm_store_si128(reinterpret_cast<VectorType *>(mem), x);
}
template<typename T> inline void VectorHelper<__m128i>::store(T *mem, const VectorType x, UnalignedFlag)
{
    _mm_storeu_si128(reinterpret_cast<VectorType *>(mem), x);
}
template<typename T> inline void VectorHelper<__m128i>::store(T *mem, const VectorType x, StreamingAndAlignedFlag)
{
    _mm_stream_si128(reinterpret_cast<VectorType *>(mem), x);
}
template<typename T> inline void VectorHelper<__m128i>::store(T *mem, const VectorType x, StreamingAndUnalignedFlag)
{
    _mm_maskmoveu_si128(x, _mm_setallone_si128(), reinterpret_cast<char *>(mem));
}
template<typename T> inline void VectorHelper<__m128i>::store(T *mem, const VectorType x, const VectorType m, AlignedFlag align)
{
    store(mem, _mm_blendv_epi8(load(mem, align), x, m), align);
}
template<typename T> inline void VectorHelper<__m128i>::store(T *mem, const VectorType x, const VectorType m, UnalignedFlag align)
{
    store(mem, _mm_blendv_epi8(load(mem, align), x, m), align);
}
template<typename T> inline void VectorHelper<__m128i>::store(T *mem, const VectorType x, const VectorType m, StreamingAndAlignedFlag)
{
    _mm_maskmoveu_si128(x, m, reinterpret_cast<char *>(mem));
}
template<typename T> inline void VectorHelper<__m128i>::store(T *mem, const VectorType x, const VectorType m, StreamingAndUnalignedFlag)
{
    _mm_maskmoveu_si128(x, m, reinterpret_cast<char *>(mem));
}

} // namespace AVX
} // namespace Vc
