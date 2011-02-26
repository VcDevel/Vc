/*  This file is part of the Vc library.

    Copyright (C) 2009-2010 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_NO_BSF_LOOPS
# ifdef VC_NO_GATHER_TRICKS
#  define VC_NO_BSF_LOOPS
# elif !defined(__x86_64__) // 32 bit x86 does not have enough registers
#  define VC_NO_BSF_LOOPS
# elif defined(_MSC_VER) // TODO: write inline asm version for MSVC
#  define VC_NO_BSF_LOOPS
# elif defined(__GNUC__) // gcc and icc work fine with the inline asm
# else
#  error "Check whether inline asm works, or define VC_NO_BSF_LOOPS"
# endif
#endif

#ifdef VC_SLOWDOWN_GATHER
#define SLOWDOWN_ASM "ror $9,%1\n\trol $1,%1\n\t" \
                     "rol $1,%1\n\trol $1,%1\n\t" \
                     "rol $1,%1\n\trol $1,%1\n\t" \
                     "rol $1,%1\n\trol $1,%1\n\t" \
                     "rol $1,%1\n\trol $1,%1\n\t"
#else //VC_SLOWDOWN_GATHER
#define SLOWDOWN_ASM
#endif //VC_SLOWDOWN_GATHER

#define ALIGN_16 "\n.align 16\n\t"

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
    _mm256_maskstore_ps(mem, m, x);
}
inline void VectorHelper<__m256>::store(float *mem, const VectorType x, const VectorType m, UnalignedFlag)
{
    _mm256_maskstore_ps(mem, m, x);
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
                _mm_stream_load_si128(reinterpret_cast<__m128i *>(const_cast<double *>(&m[4])))));
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
    _mm256_maskstore_pd(mem, m, x);
}
inline void VectorHelper<__m256d>::store(double *mem, const VectorType x, const VectorType m, UnalignedFlag)
{
    _mm256_maskstore_pd(mem, m, x);
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
    return _mm256_insertf128_si256(avx_cast<VectorType>(_mm_stream_load_si128(m)),
            _mm_stream_load_si128(&m[4]));
}
template<typename T> inline __m256i
    VC_WARN("AVX does not support streaming unaligned loads. Will use non-streaming unaligned load instead.")
VectorHelper<__m256i>::load(const T *m, StreamingAndUnalignedFlag)
{
    return _mm256_loadu_si256(m);
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
    _mm256_maskstore_ps(reinterpret_cast<float *>(mem), avx_cast<__m256>(m), avx_cast<__m256>(x));
}
template<typename T> inline void VectorHelper<__m256i>::store(T *mem, const VectorType x, const VectorType m, UnalignedFlag)
{
    _mm256_maskstore_ps(reinterpret_cast<float *>(mem), avx_cast<__m256>(m), avx_cast<__m256>(x));
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
    return _mm_stream_load_si128(m);
}
template<typename T> inline __m128i
    VC_WARN("AVX does not support streaming unaligned loads. Will use non-streaming unaligned load instead.")
VectorHelper<__m128i>::load(const T *m, StreamingAndUnalignedFlag)
{
    return _mm_loadu_si128(m);
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

// TODO
#if 0
    template<> inline _M256I SortHelper<_M256I, 8>::sort(_M256I x)
    {
        _M256I lo, hi, y;
        // sort pairs
        y = _mm256_shufflelo_epi16(_mm256_shufflehi_epi16(x, _MM_SHUFFLE(2, 3, 0, 1)), _MM_SHUFFLE(2, 3, 0, 1));
        lo = _mm256_min_epi16(x, y);
        hi = _mm256_max_epi16(x, y);
        x = _mm256_blend_epi16(lo, hi, 0xaa);

        // merge left and right quads
        y = _mm256_shufflelo_epi16(_mm256_shufflehi_epi16(x, _MM_SHUFFLE(0, 1, 2, 3)), _MM_SHUFFLE(0, 1, 2, 3));
        lo = _mm256_min_epi16(x, y);
        hi = _mm256_max_epi16(x, y);
        x = _mm256_blend_epi16(lo, hi, 0xcc);
        y = _mm256_srli_si256(x, 2);
        lo = _mm256_min_epi16(x, y);
        hi = _mm256_max_epi16(x, y);
        x = _mm256_blend_epi16(lo, _mm256_slli_si256(hi, 2), 0xaa);

        // merge quads into octs
        y = _mm256_shuffle_epi32(x, _MM_SHUFFLE(1, 0, 3, 2));
        y = _mm256_shufflelo_epi16(y, _MM_SHUFFLE(0, 1, 2, 3));
        lo = _mm256_min_epi16(x, y);
        hi = _mm256_max_epi16(x, y);

        x = _mm256_unpacklo_epi16(lo, hi);
        y = _mm256_srli_si256(x, 8);
        lo = _mm256_min_epi16(x, y);
        hi = _mm256_max_epi16(x, y);

        x = _mm256_unpacklo_epi16(lo, hi);
        y = _mm256_srli_si256(x, 8);
        lo = _mm256_min_epi16(x, y);
        hi = _mm256_max_epi16(x, y);

        return _mm256_unpacklo_epi16(lo, hi);
    }
    template<> inline _M256I SortHelper<_M256I, 4>::sort(_M256I x)
    {
        /*
        // in 16,67% of the cases the merge can be replaced by an append

        // x = [a b c d]
        // y = [c d a b]
        _M256I y = _mm256_shuffle_epi32(x, _MM_SHUFFLE(1, 0, 3, 2));
        _M256I l = _mm256_min_epi32(x, y); // min[ac bd ac bd]
        _M256I h = _mm256_max_epi32(x, y); // max[ac bd ac bd]
        if (IS_UNLIKELY(_mm256_cvtsi256_si32(h) <= l[1])) { // l[0] < h[0] < l[1] < h[1]
            return _mm256_unpacklo_epi32(l, h);
        }
        // h[0] > l[1]
        */

        // sort pairs
        _M256I y = _mm256_shuffle_epi32(x, _MM_SHUFFLE(2, 3, 0, 1));
        _M256I l = _mm256_min_epi32(x, y);
        _M256I h = _mm256_max_epi32(x, y);
        x = _mm256_unpacklo_epi32(l, h);
        y = _mm256_unpackhi_epi32(h, l);

        // sort quads
        l = _mm256_min_epi32(x, y);
        h = _mm256_max_epi32(x, y);
        x = _mm256_unpacklo_epi32(l, h);
        y = _mm256_unpackhi_epi64(x, x);

        l = _mm256_min_epi32(x, y);
        h = _mm256_max_epi32(x, y);
        return _mm256_unpacklo_epi32(l, h);
    }
    template<> inline _M256 SortHelper<_M256, 4>::sort(_M256 x)
    {
        _M256 y = _mm256_shuffle_ps(x, x, _MM_SHUFFLE(2, 3, 0, 1));
        _M256 l = _mm256_min_ps(x, y);
        _M256 h = _mm256_max_ps(x, y);
        x = _mm256_unpacklo_ps(l, h);
        y = _mm256_unpackhi_ps(h, l);

        l = _mm256_min_ps(x, y);
        h = _mm256_max_ps(x, y);
        x = _mm256_unpacklo_ps(l, h);
        y = _mm256_movehl_ps(x, x);

        l = _mm256_min_ps(x, y);
        h = _mm256_max_ps(x, y);
        return _mm256_unpacklo_ps(l, h);
//X         _M256 k = _mm256_cmpgt_ps(x, y);
//X         k = _mm256_shuffle_ps(k, k, _MM_SHUFFLE(2, 2, 0, 0));
//X         x = _mm256_blendv_ps(x, y, k);
//X         y = _mm256_shuffle_ps(x, x, _MM_SHUFFLE(1, 0, 3, 2));
//X         k = _mm256_cmpgt_ps(x, y);
//X         k = _mm256_shuffle_ps(k, k, _MM_SHUFFLE(1, 0, 1, 0));
//X         x = _mm256_blendv_ps(x, y, k);
//X         y = _mm256_shuffle_ps(x, x, _MM_SHUFFLE(3, 1, 2, 0));
//X         k = _mm256_cmpgt_ps(x, y);
//X         k = _mm256_shuffle_ps(k, k, _MM_SHUFFLE(0, 1, 1, 0));
//X         return _mm256_blendv_ps(x, y, k);
    }
    template<> inline M256 SortHelper<M256, 8>::sort(M256 x)
    {
        typedef SortHelper<_M256, 4> H;

        _M256 a, b, l, h;
        a = H::sort(x[0]);
        b = H::sort(x[1]);

        // merge
        b = _mm256_shuffle_ps(b, b, _MM_SHUFFLE(0, 1, 2, 3));
        l = _mm256_min_ps(a, b);
        h = _mm256_max_ps(a, b);

        a = _mm256_unpacklo_ps(l, h);
        b = _mm256_unpackhi_ps(l, h);
        l = _mm256_min_ps(a, b);
        h = _mm256_max_ps(a, b);

        a = _mm256_unpacklo_ps(l, h);
        b = _mm256_unpackhi_ps(l, h);
        l = _mm256_min_ps(a, b);
        h = _mm256_max_ps(a, b);

        x[0] = _mm256_unpacklo_ps(l, h);
        x[1] = _mm256_unpackhi_ps(l, h);
        return x;
    }
    template<> inline _M256D SortHelper<_M256D, 2>::sort(_M256D x)
    {
        const _M256D y = _mm256_shuffle_pd(x, x, _MM_SHUFFLE2(0, 1));
        return _mm256_unpacklo_pd(_mm256_min_sd(x, y), _mm256_max_sd(x, y));
    }

    // can be used to multiply with a constant. For some special constants it doesn't need an extra
    // vector but can use a shift instead, basically encoding the factor in the instruction.
    template<typename IndexType, unsigned int constant> inline IndexType mulConst(const IndexType &x) {
        typedef VectorHelper<typename IndexType::EntryType> H;
        switch (constant) {
            case    0: return H::zero();
            case    1: return x;
            case    2: return H::slli(x.data(),  1);
            case    4: return H::slli(x.data(),  2);
            case    8: return H::slli(x.data(),  3);
            case   16: return H::slli(x.data(),  4);
            case   32: return H::slli(x.data(),  5);
            case   64: return H::slli(x.data(),  6);
            case  128: return H::slli(x.data(),  7);
            case  256: return H::slli(x.data(),  8);
            case  512: return H::slli(x.data(),  9);
            case 1024: return H::slli(x.data(), 10);
            case 2048: return H::slli(x.data(), 11);
        }
        return H::mul(x.data(), H::set(constant));
    }
#endif

} // namespace AVX
} // namespace Vc

#undef SLOWDOWN_ASM
#undef ALIGN_16
