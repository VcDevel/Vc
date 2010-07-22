/*  This file is part of the Vc library.

    Copyright (C) 2009 Matthias Kretz <kretz@kde.org>

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
namespace SSE
{

////////////////////////////////////////////////////////////////////////////////////////////////////
// float_v
template<> inline _M128 VectorHelper<_M128>::load(const float *x, AlignedFlag)
{
    return _mm_load_ps(x);
}

template<> inline _M128 VectorHelper<_M128>::load(const float *x, UnalignedFlag)
{
    return _mm_loadu_ps(x);
}

template<> inline _M128 VectorHelper<_M128>::load(const float *x, StreamingAndAlignedFlag)
{
#ifdef VC_IMPL_SSE41
    return _mm_castsi128_ps(_mm_stream_load_si128(reinterpret_cast<_M128I *>(const_cast<float *>(x))));
#else
    return load(x, Aligned);
#endif
}

template<> inline _M128 VectorHelper<_M128>::load(const float *x, StreamingAndUnalignedFlag)
{
    return load(x, Unaligned);
}

template<> inline void VectorHelper<_M128>::store(float *mem, const VectorType &x, AlignedFlag)
{
    _mm_store_ps(mem, x);
}

template<> inline void VectorHelper<_M128>::store(float *mem, const VectorType &x, UnalignedFlag)
{
    _mm_storeu_ps(mem, x);
}

template<> inline void VectorHelper<_M128>::store(float *mem, const VectorType &x, StreamingAndAlignedFlag)
{
    _mm_stream_ps(mem, x);
}

template<> inline void VectorHelper<_M128>::store(float *mem, const VectorType &x, StreamingAndUnalignedFlag)
{
    // SSE does not support unaligned streaming stores. Since unaligned memory access is more
    // important we ignore the streaming part
    _mm_storeu_ps(mem, x);
}

template<> inline void VectorHelper<_M128>::store(float *mem, const VectorType &x, const VectorType m)
{
    _mm_maskmoveu_si128(_mm_castps_si128(x), _mm_castps_si128(m), reinterpret_cast<char *>(mem));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// sfloat_v
template<> inline M256 VectorHelper<M256>::load(const float *x, AlignedFlag)
{
    return VectorType::create(_mm_load_ps(x), _mm_load_ps(x + 4));
}

template<> inline M256 VectorHelper<M256>::load(const float *x, UnalignedFlag)
{
    return VectorType::create(_mm_loadu_ps(x), _mm_loadu_ps(x + 4));
}

template<> inline M256 VectorHelper<M256>::load(const float *x, StreamingAndAlignedFlag)
{
#ifdef VC_IMPL_SSE41
    return VectorType::create(
            _mm_castsi128_ps(_mm_stream_load_si128(reinterpret_cast<_M128I *>(const_cast<float *>(x)))),
            _mm_castsi128_ps(_mm_stream_load_si128(reinterpret_cast<_M128I *>(const_cast<float *>(x + 4)))));
#else
    return load(x, Aligned);
#endif
}

template<> inline M256 VectorHelper<M256>::load(const float *x, StreamingAndUnalignedFlag)
{
    return load(x, Unaligned);
}

template<> inline void VectorHelper<M256>::store(float *mem, const VectorType &x, AlignedFlag)
{
    _mm_store_ps(mem, x[0]);
    _mm_store_ps(mem + 4, x[1]);
}

template<> inline void VectorHelper<M256>::store(float *mem, const VectorType &x, UnalignedFlag)
{
    _mm_storeu_ps(mem, x[0]);
    _mm_storeu_ps(mem + 4, x[1]);
}

template<> inline void VectorHelper<M256>::store(float *mem, const VectorType &x, StreamingAndAlignedFlag)
{
    _mm_stream_ps(mem, x[0]);
    _mm_stream_ps(mem + 4, x[1]);
}

template<> inline void VectorHelper<M256>::store(float *mem, const VectorType &x, StreamingAndUnalignedFlag)
{
    // SSE does not support unaligned streaming stores. Since unaligned memory access is more
    // important we ignore the streaming part
    _mm_storeu_ps(mem, x[0]);
    _mm_storeu_ps(mem + 4, x[1]);
}

template<> inline void VectorHelper<M256>::store(float *mem, const VectorType &x, const VectorType m)
{
    _mm_maskmoveu_si128(_mm_castps_si128(x[0]), _mm_castps_si128(m[0]), reinterpret_cast<char *>(mem));
    _mm_maskmoveu_si128(_mm_castps_si128(x[1]), _mm_castps_si128(m[1]), reinterpret_cast<char *>(mem + 4));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// double_v
template<> inline _M128D VectorHelper<_M128D>::load(const double *x, AlignedFlag)
{
    return _mm_load_pd(x);
}

template<> inline _M128D VectorHelper<_M128D>::load(const double *x, UnalignedFlag)
{
    return _mm_loadu_pd(x);
}

template<> inline _M128D VectorHelper<_M128D>::load(const double *x, StreamingAndAlignedFlag)
{
#ifdef VC_IMPL_SSE41
    return _mm_castsi128_pd(_mm_stream_load_si128(reinterpret_cast<_M128I *>(const_cast<double *>(x))));
#else
    return load(x, Aligned);
#endif
}

template<> inline _M128D VectorHelper<_M128D>::load(const double *x, StreamingAndUnalignedFlag)
{
    return load(x, Unaligned);
}

template<> inline void VectorHelper<_M128D>::store(double *mem, const VectorType &x, AlignedFlag)
{
    _mm_store_pd(mem, x);
}

template<> inline void VectorHelper<_M128D>::store(double *mem, const VectorType &x, UnalignedFlag)
{
    _mm_storeu_pd(mem, x);
}

template<> inline void VectorHelper<_M128D>::store(double *mem, const VectorType &x, StreamingAndAlignedFlag)
{
    _mm_stream_pd(mem, x);
}

template<> inline void VectorHelper<_M128D>::store(double *mem, const VectorType &x, StreamingAndUnalignedFlag)
{
    // SSE does not support unaligned streaming stores. Since unaligned memory access is more
    // important we ignore the streaming part
    _mm_storeu_pd(mem, x);
}

template<> inline void VectorHelper<_M128D>::store(double *mem, const VectorType &x, const VectorType m)
{
    _mm_maskmoveu_si128(_mm_castpd_si128(x), _mm_castpd_si128(m), reinterpret_cast<char *>(mem));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// int_v, uint_v, short_v, ushort_v
template<typename T> inline _M128I VectorHelper<_M128I>::load(const T *x, AlignedFlag)
{
    return _mm_load_si128(reinterpret_cast<const VectorType *>(x));
}

template<typename T> inline _M128I VectorHelper<_M128I>::load(const T *x, UnalignedFlag)
{
    return _mm_loadu_si128(reinterpret_cast<const VectorType *>(x));
}

template<typename T> inline _M128I VectorHelper<_M128I>::load(const T *x, StreamingAndAlignedFlag)
{
#ifdef VC_IMPL_SSE41
    return _mm_stream_load_si128(reinterpret_cast<_M128I *>(const_cast<T *>(x)));
#else
    return load(x, Aligned);
#endif
}

template<typename T> inline _M128I VectorHelper<_M128I>::load(const T *x, StreamingAndUnalignedFlag)
{
    return load(x, Unaligned);
}

template<typename T> inline void VectorHelper<_M128I>::store(T *mem, const VectorType &x, AlignedFlag)
{
    _mm_store_si128(reinterpret_cast<VectorType *>(mem), x);
}

template<typename T> inline void VectorHelper<_M128I>::store(T *mem, const VectorType &x, UnalignedFlag)
{
    _mm_storeu_si128(reinterpret_cast<VectorType *>(mem), x);
}

template<typename T> inline void VectorHelper<_M128I>::store(T *mem, const VectorType &x, StreamingAndAlignedFlag)
{
    _mm_stream_si128(reinterpret_cast<VectorType *>(mem), x);
}

template<typename T> inline void VectorHelper<_M128I>::store(T *mem, const VectorType &x, StreamingAndUnalignedFlag)
{
    // SSE does not support unaligned streaming stores. Since unaligned memory access is more
    // important we ignore the streaming part
    _mm_storeu_si128(reinterpret_cast<VectorType *>(mem), x);
}

template<typename T> inline void VectorHelper<_M128I>::store(T *mem, const VectorType &x, const VectorType &m)
{
    _mm_maskmoveu_si128(x, m, reinterpret_cast<char *>(mem));
}

    template<> inline _M128I SortHelper<_M128I, 8>::sort(_M128I x)
    {
        _M128I lo, hi, y;
        // sort pairs
        y = _mm_shufflelo_epi16(_mm_shufflehi_epi16(x, _MM_SHUFFLE(2, 3, 0, 1)), _MM_SHUFFLE(2, 3, 0, 1));
        lo = _mm_min_epi16(x, y);
        hi = _mm_max_epi16(x, y);
        x = _mm_blend_epi16(lo, hi, 0xaa);

        // merge left and right quads
        y = _mm_shufflelo_epi16(_mm_shufflehi_epi16(x, _MM_SHUFFLE(0, 1, 2, 3)), _MM_SHUFFLE(0, 1, 2, 3));
        lo = _mm_min_epi16(x, y);
        hi = _mm_max_epi16(x, y);
        x = _mm_blend_epi16(lo, hi, 0xcc);
        y = _mm_srli_si128(x, 2);
        lo = _mm_min_epi16(x, y);
        hi = _mm_max_epi16(x, y);
        x = _mm_blend_epi16(lo, _mm_slli_si128(hi, 2), 0xaa);

        // merge quads into octs
        y = _mm_shuffle_epi32(x, _MM_SHUFFLE(1, 0, 3, 2));
        y = _mm_shufflelo_epi16(y, _MM_SHUFFLE(0, 1, 2, 3));
        lo = _mm_min_epi16(x, y);
        hi = _mm_max_epi16(x, y);

        x = _mm_unpacklo_epi16(lo, hi);
        y = _mm_srli_si128(x, 8);
        lo = _mm_min_epi16(x, y);
        hi = _mm_max_epi16(x, y);

        x = _mm_unpacklo_epi16(lo, hi);
        y = _mm_srli_si128(x, 8);
        lo = _mm_min_epi16(x, y);
        hi = _mm_max_epi16(x, y);

        return _mm_unpacklo_epi16(lo, hi);
    }
    template<> inline _M128I SortHelper<_M128I, 4>::sort(_M128I x)
    {
        /*
        // in 16,67% of the cases the merge can be replaced by an append

        // x = [a b c d]
        // y = [c d a b]
        _M128I y = _mm_shuffle_epi32(x, _MM_SHUFFLE(1, 0, 3, 2));
        _M128I l = _mm_min_epi32(x, y); // min[ac bd ac bd]
        _M128I h = _mm_max_epi32(x, y); // max[ac bd ac bd]
        if (IS_UNLIKELY(_mm_cvtsi128_si32(h) <= l[1])) { // l[0] < h[0] < l[1] < h[1]
            return _mm_unpacklo_epi32(l, h);
        }
        // h[0] > l[1]
        */

        // sort pairs
        _M128I y = _mm_shuffle_epi32(x, _MM_SHUFFLE(2, 3, 0, 1));
        _M128I l = _mm_min_epi32(x, y);
        _M128I h = _mm_max_epi32(x, y);
        x = _mm_unpacklo_epi32(l, h);
        y = _mm_unpackhi_epi32(h, l);

        // sort quads
        l = _mm_min_epi32(x, y);
        h = _mm_max_epi32(x, y);
        x = _mm_unpacklo_epi32(l, h);
        y = _mm_unpackhi_epi64(x, x);

        l = _mm_min_epi32(x, y);
        h = _mm_max_epi32(x, y);
        return _mm_unpacklo_epi32(l, h);
    }
    template<> inline _M128 SortHelper<_M128, 4>::sort(_M128 x)
    {
        _M128 y = _mm_shuffle_ps(x, x, _MM_SHUFFLE(2, 3, 0, 1));
        _M128 l = _mm_min_ps(x, y);
        _M128 h = _mm_max_ps(x, y);
        x = _mm_unpacklo_ps(l, h);
        y = _mm_unpackhi_ps(h, l);

        l = _mm_min_ps(x, y);
        h = _mm_max_ps(x, y);
        x = _mm_unpacklo_ps(l, h);
        y = _mm_movehl_ps(x, x);

        l = _mm_min_ps(x, y);
        h = _mm_max_ps(x, y);
        return _mm_unpacklo_ps(l, h);
//X         _M128 k = _mm_cmpgt_ps(x, y);
//X         k = _mm_shuffle_ps(k, k, _MM_SHUFFLE(2, 2, 0, 0));
//X         x = _mm_blendv_ps(x, y, k);
//X         y = _mm_shuffle_ps(x, x, _MM_SHUFFLE(1, 0, 3, 2));
//X         k = _mm_cmpgt_ps(x, y);
//X         k = _mm_shuffle_ps(k, k, _MM_SHUFFLE(1, 0, 1, 0));
//X         x = _mm_blendv_ps(x, y, k);
//X         y = _mm_shuffle_ps(x, x, _MM_SHUFFLE(3, 1, 2, 0));
//X         k = _mm_cmpgt_ps(x, y);
//X         k = _mm_shuffle_ps(k, k, _MM_SHUFFLE(0, 1, 1, 0));
//X         return _mm_blendv_ps(x, y, k);
    }
    template<> inline M256 SortHelper<M256, 8>::sort(M256 x)
    {
        typedef SortHelper<_M128, 4> H;

        _M128 a, b, l, h;
        a = H::sort(x[0]);
        b = H::sort(x[1]);

        // merge
        b = _mm_shuffle_ps(b, b, _MM_SHUFFLE(0, 1, 2, 3));
        l = _mm_min_ps(a, b);
        h = _mm_max_ps(a, b);

        a = _mm_unpacklo_ps(l, h);
        b = _mm_unpackhi_ps(l, h);
        l = _mm_min_ps(a, b);
        h = _mm_max_ps(a, b);

        a = _mm_unpacklo_ps(l, h);
        b = _mm_unpackhi_ps(l, h);
        l = _mm_min_ps(a, b);
        h = _mm_max_ps(a, b);

        x[0] = _mm_unpacklo_ps(l, h);
        x[1] = _mm_unpackhi_ps(l, h);
        return x;
    }
    template<> inline _M128D SortHelper<_M128D, 2>::sort(_M128D x)
    {
        const _M128D y = _mm_shuffle_pd(x, x, _MM_SHUFFLE2(0, 1));
        return _mm_unpacklo_pd(_mm_min_sd(x, y), _mm_max_sd(x, y));
    }

    struct GeneralHelpers
    {
        /**
         * There are several possible aproaches to masked gather implementations. Which one is the
         * fastest depends on many factors and is hard to benchmark. Still, finding a perfect gather
         * is an important speed issue.
         *
         * in principle there are two ways to move the data into a 128 bit register
         *
         * 1. move each value to a GPR, then to a memory location; if all values are done, move
         *    128 bit from the memory location to the XMM register
         * 2. move each value directly to an XMM register, shift and or to get result
         *
         * then there are different ways to gather the source values
         *
         * 1. iteration from 0 to Size using branching to conditionally execute one value move
         * 2. iteration from 0 to Size using conditional moves with some trickery (the cmov always
         *    accesses the source memory, thus possibly leading to invalid memory accesses if the
         *    condition is false; this can be solved by another cmov that sets the index to 0 on the
         *    inverted condition)
         * 3. extract the bit positions from the mask using bsf and btr and loop while bsf does not
         *    set the zero flag, moving one value in the loop body
         * 4. calculate the population count of the mask to use it (Size - popcnt) as a jump offset
         *    into an unrolled bsf loop
         */
        template<unsigned int scale, typename Base, typename IndexType, typename EntryType>
        static inline void maskedDoubleGatherHelper(
                Base &v, const IndexType &outer, const IndexType &inner, _ulong mask, const EntryType *const *const baseAddr
                ) {
#ifdef VC_NO_BSF_LOOPS
# ifdef VC_NO_GATHER_TRICKS
            for (int i = 0; i < Base::Size; ++i) {
                if (mask & (1 << i)) {
                    v.d.m(i) = baseAddr[outer.d.m(i) * (scale / sizeof(void *))][inner.d.m(i)];
                }
            }
# else // VC_NO_GATHER_TRICKS
            unrolled_loop16(i, 0, Base::Size,
                    if (mask & (1 << i)) v.d.m(i) = baseAddr[outer.d.m(i) * (scale / sizeof(void *))][inner.d.m(i)];
                    );
# endif // VC_NO_GATHER_TRICKS
#else // VC_NO_BSF_LOOPS
            if (sizeof(EntryType) == 2) {
                register _ulong bit;
                register _ulong outerIndex;
                register _ulong innerIndex;
                register const EntryType *array;
                register EntryType value;
                asm volatile(
                          "\t"  "bsf %1,%0"// %0 contains the index to use for outer and inner
                        "\n\t"  "jz 1f"
                        ALIGN_16
                        "\n\t"  "0:"
                        "\n\t"  "movzwq (%7,%0,2),%4" // outer index in ecx
                        "\n\t"  "movzwq (%11,%0,2),%5"// inner index in ecx
                        "\n\t"  "imul %10,%4"         // scale to become byte-offset
                        "\n\t"  "btr %0,%1"
                        "\n\t"  "mov (%8,%4,1),%6"    // rdx = baseAddr[outer[%0] * scale / sizeof(void*)]
                        "\n\t"  "movw (%6,%5,2),%2"   // value = rdx[inner[%0]]
                        "\n\t"  "movw %2,(%9,%0,2)"   // v[%0] = value
                        "\n\t"  "bsf %1,%0"           // %0 contains the index to use for outer and inner
                        "\n\t"  "jnz 0b"
                        ALIGN_16
                        "\n\t"  "1:"
                        : "=&r"(bit), "+r"(mask), "=&r"(value), "+m"(v.d),
                          "=&r"(outerIndex), "=&r"(innerIndex), "=&r"(array)
                        : "r"(&outer.d.v()), "r"(baseAddr), "r"(&v.d), "n"(scale), "r"(&inner.d.v()), "m"(outer.d.v()), "m"(inner.d.v()));
            } else if (sizeof(EntryType) == 4 && sizeof(typename IndexType::EntryType) == 4) {
                register _ulong bit;
                register _ulong innerIndex;
                register const EntryType *array;
                register EntryType value;
                asm volatile(
                          "\t"  "bsf %1,%0"// %0 contains the index to use for outer and inner
                        "\n\t"  "jz 1f"
                        ALIGN_16
                        "\n\t"  "0:"
                        "\n\t"  "imul %9,(%6,%0,4),%%ecx" // outer index * scale => byte offset
                        "\n\t"  "movslq (%10,%0,4),%4"     // inner index in ecx
                        "\n\t"  "btr %0,%1"
                        "\n\t"  "mov (%7,%%rcx,1),%5"  // rdx = baseAddr[outer[%0] * scale / sizeof(void*)]
                        "\n\t"  "mov (%5,%4,4),%2"  // value = rdx[inner[%0]]
                        "\n\t"  "mov %2,(%8,%0,4)"        // v[%0] = value
                        "\n\t"  "bsf %1,%0"           // %0 contains the index to use for outer and inner
                        "\n\t"  "jnz 0b"
                        ALIGN_16
                        "\n\t"  "1:"
                        : "=&r"(bit), "+r"(mask), "=&r"(value), "+m"(v.d),
                          "=&r"(innerIndex), "=&r"(array)
                        : "r"(&outer.d.v()), "r"(baseAddr), "r"(&v.d), "n"(scale), "r"(&inner.d.v()), "m"(outer.d.v()), "m"(inner.d.v())
                        : "rcx" );
            } else if (sizeof(EntryType) == 4 && sizeof(typename IndexType::EntryType) == 2) {
                register _ulong bit;
                register _ulong outerIndex;
                register _ulong innerIndex;
                register const EntryType *array;
                register EntryType value;
                asm volatile(
                          "\t"  "bsf %1,%0"// %0 contains the index to use for outer and inner
                        "\n\t"  "jz 1f"
                        ALIGN_16
                        "\n\t"  "0:"
                        "\n\t"  "movzwq (%7,%0,2),%4"  // outer index in ecx
                        "\n\t"  "movzwq (%11,%0,2),%5"  // inner index in ecx
                        "\n\t"  "imul %10,%4"           // scale to become byte-offset
                        "\n\t"  "btr %0,%1"
                        "\n\t"  "mov (%8,%4,1),%6"  // rdx = baseAddr[outer[%0] * scale / sizeof(void*)]
                        "\n\t"  "mov (%6,%5,4),%2"  // value = rdx[inner[%0]]
                        "\n\t"  "mov %2,(%9,%0,4)"        // v[%0] = value
                        "\n\t"  "bsf %1,%0"// %0 contains the index to use for outer and inner
                        "\n\t"  "jnz 0b"
                        ALIGN_16
                        "\n\t"  "1:"
                        : "=&r"(bit), "+r"(mask), "=&r"(value), "+m"(v.d),
                          "=&r"(outerIndex), "=&r"(innerIndex), "=&r"(array)
                        : "r"(&outer.d.v()), "r"(baseAddr), "r"(&v.d), "n"(scale), "r"(&inner.d.v()), "m"(outer.d.v()), "m"(inner.d.v()));
            } else if (sizeof(EntryType) == 8 && sizeof(typename IndexType::EntryType) == 4) {
                register _ulong bit;
                register _ulong innerIndex;
                register const EntryType *array;
                register EntryType value;
                asm volatile(
                          "\t"  "bsf %1,%0"               // %0 contains the index to use for outer and inner
                        "\n\t"  "jz 1f"
                        ALIGN_16
                        "\n\t"  "0:"
                        "\n\t"  "imul %9,(%6,%0,4),%%ecx" // outer index * scale => byte offset
                        "\n\t"  "movslq (%10,%0,4),%4"     // inner index in ecx
                        "\n\t"  "btr %0,%1"
                        "\n\t"  "mov (%7,%%rcx,1),%5"  // rdx = baseAddr[outer[%0] * scale / sizeof(void*)]
                        "\n\t"  "mov (%5,%4,8),%2"  // value = rdx[inner[%0]]
                        "\n\t"  "mov %2,(%8,%0,8)"        // v[%0] = value
                        "\n\t"  "bsf %1,%0"               // %0 contains the index to use for outer and inner
                        "\n\t"  "jnz 0b"
                        ALIGN_16
                        "\n\t"  "1:"
                        : "=&r"(bit), "+r"(mask), "=&r"(value), "+m"(v.d),
                          "=&r"(innerIndex), "=&r"(array)
                        : "r"(&outer.d.v()), "r"(baseAddr), "r"(&v.d), "n"(scale), "r"(&inner.d.v()), "m"(outer.d.v()), "m"(inner.d.v())
                        : "rcx"   );
            } else {
                abort();
            }
#endif // VC_NO_BSF_LOOPS
        }
        template<unsigned int scale, typename Base, typename IndexType, typename EntryType>
        static inline void maskedGatherStructHelper(
                Base &v, const IndexType &indexes, int mask, const EntryType *baseAddr
                ) {

#ifndef VC_NO_BSF_LOOPS
            asm volatile(""::"m"(indexes.d.v()));
            if (sizeof(EntryType) == 2) {
                register _ulong index;
                register EntryType value;
                asm volatile(
                        SLOWDOWN_ASM
                        "jmp 1f"                "\n\t"
                        ALIGN_16
                        "0:"                   "\n\t"
                        "movzwq (%[indexes],%%rdi,2),%[index]"  "\n\t"
                        "imul %[scale],%[index]"           "\n\t"
                        "btr %%edi,%[mask]"            "\n\t"
                        "movw (%[base],%[index],1),%[value]"     "\n\t"
                        "movw %[value],(%[vec],%%rdi,2)"     "\n\t"
                        ALIGN_16
                        "1:"                   "\n\t"
                        "bsf %[mask],%%edi"            "\n\t"
                        "jnz 0b"               "\n\t"
                        : [mask]"+r"(mask), [value]"=&r"(value), "+m"(v.d), [index]"=&r"(index)
                        : [indexes]"r"(&indexes.d.v()), [base]"r"(baseAddr), [vec]"r"(&v.d), [scale]"n"(scale)
                        : "rdi");
            } else if (sizeof(EntryType) == 4) {
                if (sizeof(typename IndexType::EntryType) == 4) {
                    register EntryType value;
                    asm volatile(
                            SLOWDOWN_ASM
                            "jmp 1f"                "\n\t"
                            ALIGN_16
                            "0:"                   "\n\t"
                            "imul %[scale],(%[indexes],%%rdi,4),%%ecx""\n\t"
                            "btr %%edi,%[mask]"            "\n\t"
                            "mov (%[base],%%rcx,1),%[value]"  "\n\t"
                            "mov %[value],(%[vec],%%rdi,4)"     "\n\t"
                            ALIGN_16
                            "1:"                   "\n\t"
                            "bsf %[mask],%%edi"            "\n\t"
                            "jnz 0b"               "\n\t"
                            : [mask]"+r"(mask), [value]"=&r"(value), "+m"(v.d)
                            : [indexes]"r"(&indexes.d.v()), [base]"r"(baseAddr), [vec]"r"(&v.d), [scale]"n"(scale)
                            : "rcx", "rdi"   );
                } else if (sizeof(typename IndexType::EntryType) == 2) {
                    register _ulong index;
                    register EntryType value;
                    asm volatile(
                            SLOWDOWN_ASM
                            "jmp 1f"                "\n\t"
                            ALIGN_16
                            "0:"                   "\n\t"
                            "movzwq (%[indexes],%%rdi,2),%[index]"  "\n\t"
                            "imul %[scale],%[index]"           "\n\t"
                            "btr %%edi,%[mask]"            "\n\t"
                            "mov (%[base],%[index],1),%[value]"     "\n\t"
                            "mov %[value],(%[vec],%%rdi,4)"     "\n\t"
                            ALIGN_16
                            "1:"                   "\n\t"
                            "bsf %[mask],%%edi"            "\n\t"
                            "jnz 0b"               "\n\t"
                            : [mask]"+r"(mask), [value]"=&r"(value), "+m"(v.d), [index]"=&r"(index)
                            : [indexes]"r"(&indexes.d.v()), [base]"r"(baseAddr), [vec]"r"(&v.d), [scale]"n"(scale)
                            : "rdi");
                } else {
                    abort();
                }
            } else if (sizeof(EntryType) == 8) {
                register EntryType value;
                asm volatile(
                        SLOWDOWN_ASM
                        "jmp 1f"                "\n\t"
                        ALIGN_16
                        "0:"                   "\n\t"
                        "imul %[scale],(%[indexes],%%rdi,4),%%ecx""\n\t"
                        "btr %%edi,%[mask]"            "\n\t"
                        "mov (%[base],%%rcx,1),%[value]"  "\n\t"
                        "mov %[value],(%[vec],%%rdi,8)"     "\n\t"
                        ALIGN_16
                        "1:"                   "\n\t"
                        "bsf %[mask],%%edi"            "\n\t"
                        "jnz 0b"               "\n\t"
                        : [mask]"+r"(mask), [value]"=&r"(value), "+m"(v.d)
                        : [indexes]"r"(&indexes.d.v()), [base]"r"(baseAddr), [vec]"r"(&v.d), [scale]"n"(scale)
                        : "rcx", "rdi"   );
            } else {
                abort();
            }
#else
# ifdef VC_NO_GATHER_TRICKS
            typedef const char * Memory MAY_ALIAS;
            Memory const baseAddr2 = reinterpret_cast<Memory>(baseAddr);
            for (int i = 0; i < Base::Size; ++i) {
                if (mask & (1 << i)) {
                    v.d.m(i) = *reinterpret_cast<const EntryType *>(&baseAddr2[scale * indexes.d.m(i)]);
                }
            }
# else
            if (sizeof(EntryType) <= 4) {
                unrolled_loop16(i, 0, Base::Size,
                        register EntryType tmp = v.d.m(i);
                        register _long j = scale * indexes.d.m(i);
                        asm volatile("test %[i],%[mask]\n\t"
                            "cmove %[zero],%[j]\n\t"
                            "cmovne (%[mem],%[j],1),%[tmp]"
                            : [tmp]"+r"(tmp),
                            [j]"+r"(j)
                            : [i]"i"(1 << i),
                            [mask]"r"(mask),
                            [mem]"r"(baseAddr),
                            [zero]"r"(0)
                            );
                        v.d.m(i) = tmp;
                        );
                return;
            }
#  ifdef __x86_64__
            unrolled_loop16(i, 0, Base::Size,
                    register EntryType tmp = v.d.m(i);
                    register _long j = scale * indexes.d.m(i);
                    asm volatile("test %[i],%[mask]\n\t"
                        "cmove %[zero],%[j]\n\t"
                        "cmovne (%[mem],%[j],1),%[tmp]"
                        : [tmp]"+r"(tmp),
                        [j]"+r"(j)
                        : [i]"i"(1 << i),
                        [mask]"r"(mask),
                        [mem]"r"(baseAddr),
                        [zero]"r"(0)
                        );
                    v.d.m(i) = tmp;
                    );
#  else
            // on 32 bit archs, 64 bit copies require two 32 bit registers
            unrolled_loop16(i, 0, Base::Size,
                    register EntryType tmp = v.d.m(i);
                    register _long j = scale * indexes.d.m(i);
                    asm volatile("test %[i],%[mask]\n\t"
                        "cmove %[zero],%[j]\n\t"
                        "cmovne (%[mem],%[j],1),%%eax\n\t"
                        "cmovne 4(%[mem],%[j],1),%%edx"
                        : "+A"(tmp),
                          [j]"+r"(j)
                        : [i]"i"(1 << i),
                          [mask]"r"(mask),
                          [mem]"r"(baseAddr),
                          [zero]"r"(0)
                          );
                    v.d.m(i) = tmp;
                    );
#  endif
# endif
#endif
        }

        template<typename Base, typename IndexType, typename EntryType>
        static inline void maskedGatherHelper(
                Base &v, const IndexType &indexes, int mask, const EntryType *baseAddr
                ) {
#ifndef VC_NO_BSF_LOOPS
            asm volatile(""::"m"(indexes.d.v()));
            if (sizeof(EntryType) == 2) {
                register _ulong index;
                register EntryType value;
                asm volatile(
                        SLOWDOWN_ASM
                        "jmp 1f"                                 "\n\t"
                        ALIGN_16
                        "0:"                                    "\n\t"
                        "movzwq (%[indexes],%%rdi,2),%[index]"  "\n\t"
                        "movw (%[base],%[index],2),%[value]"     "\n\t"
                        "btr %%edi,%[mask]"                     "\n\t"
                        "movw %[value],(%[vec],%%rdi,2)"         "\n\t"
                        ALIGN_16
                        "1:"                                    "\n\t"
                        "bsf %[mask],%%edi"                     "\n\t"
                        "jnz 0b"                                "\n\t"
                        : [mask]"+r"(mask), [index]"=&r"(index), [value]"=&r"(value), "+m"(v.d)
                        : [indexes]"r"(&indexes.d.v()), [base]"r"(baseAddr), [vec]"r"(&v.d)
                        : "rdi"
                        );
            } else if (sizeof(EntryType) == 4) {
                if (sizeof(typename IndexType::EntryType) == 4) {
                    register _ulong index;
                    register EntryType value;
                    asm volatile(
                            SLOWDOWN_ASM
                            "jmp 1f"                                 "\n\t"
                            ALIGN_16
                            "0:"                                    "\n\t"
                            "movslq (%[indexes],%%rdi,4),%[index]"  "\n\t"
                            "mov (%[base],%[index],4),%[value]"     "\n\t"
                            "btr %%edi,%[mask]"                     "\n\t"
                            "mov %[value],(%[vec],%%rdi,4)"         "\n\t"
                            ALIGN_16
                            "1:"                                    "\n\t"
                            "bsf %[mask],%%edi"                     "\n\t"
                            "jnz 0b"                                "\n\t"
                            : [mask]"+r"(mask), [index]"=&r"(index), [value]"=&r"(value), "+m"(v.d)
                            : [indexes]"r"(&indexes.d.v()), [base]"r"(baseAddr), [vec]"r"(&v.d)
                            : "rdi"
                            );
                } else if (sizeof(typename IndexType::EntryType) == 2) {
                    register _ulong index;
                    register EntryType value;
                    asm volatile(
                            SLOWDOWN_ASM
                            "jmp 1f"                                 "\n\t"
                            ALIGN_16
                            "0:"                                    "\n\t"
                            "movzwq (%[indexes],%%rdi,2),%[index]"  "\n\t"
                            "mov (%[base],%[index],4),%[value]"     "\n\t"
                            "btr %%edi,%[mask]"                     "\n\t"
                            "mov %[value],(%[vec],%%rdi,4)"         "\n\t"
                            ALIGN_16
                            "1:"                                    "\n\t"
                            "bsf %[mask],%%edi"                     "\n\t"
                            "jnz 0b"                                "\n\t"
                            : [mask]"+r"(mask), [index]"=&r"(index), [value]"=&r"(value), "+m"(v.d)
                            : [indexes]"r"(&indexes.d.v()), [base]"r"(baseAddr), [vec]"r"(&v.d)
                            : "rdi"
                            );
                } else {
                    abort();
                }
            } else if (sizeof(EntryType) == 8) {
                register _ulong index;
                register EntryType value;
                asm volatile(
                        SLOWDOWN_ASM
                        "jmp 1f"                                 "\n\t"
                        ALIGN_16
                        "0:"                                    "\n\t"
                        "movslq (%[indexes],%%rdi,4),%[index]"  "\n\t"
                        "mov (%[base],%[index],8),%[value]"     "\n\t"
                        "btr %%edi,%[mask]"                     "\n\t"
                        "mov %[value],(%[vec],%%rdi,8)"         "\n\t"
                        ALIGN_16
                        "1:"                                    "\n\t"
                        "bsf %[mask],%%edi"                     "\n\t"
                        "jnz 0b"                                "\n\t"
                        : [mask]"+r"(mask), [index]"=&r"(index), [value]"=&r"(value), "+m"(v.d)
                        : [indexes]"r"(&indexes.d.v()), [base]"r"(baseAddr), [vec]"r"(&v.d)
                        : "rdi"
                        );
            } else {
                abort();
            }
#else
# ifdef VC_NO_GATHER_TRICKS
            for (int i = 0; i < Base::Size; ++i) {
                if (mask & (1 << i)) {
                    v.d.m(i) = baseAddr[indexes.d.m(i)];
                }
            }
//            unrolled_loop16(i, 0, Base::Size,
//                   if (mask & (1 << i)) v.d.m(i) = baseAddr[indexes.d.m(i)];
//                  );
# else
            if (sizeof(EntryType) == 8) {
#ifdef __x86_64__
                unrolled_loop16(i, 0, Base::Size,
                    register _long j = indexes.d.m(i);
                    register _long zero = 0;
                    register EntryType tmp = v.d.m(i);
                    asm volatile(
                        "test %[i],%[mask]\n\t"
                        "cmove %[zero],%[j]\n\t"
                        "cmovne (%[mem],%[j],8),%[tmp]"
                        : [tmp]"+r"(tmp),
                          [j]"+r"(j)
                        : [i]"i"(1 << i),
                          [mask]"r"(mask),
                          [mem]"r"(baseAddr),
                          [zero]"r"(zero)
                        );
                    v.d.m(i) = tmp;
                    );
#else
                unrolled_loop16(i, 0, Base::Size,
                    register _long j = indexes.d.m(i);
                    register _long zero = 0;
                    register EntryType tmp = v.d.m(i);
                    asm volatile(
                        "test %[i],%[mask]\n\t"
                        "cmove %[zero],%[j]\n\t"
                        "cmovne (%[mem],%[j],8),%%eax\n\t"
                        "cmovne 4(%[mem],%[j],8),%%edx"
                        : "+A"(tmp),
                          [j]"+r"(j)
                        : [i]"i"(1 << i),
                          [mask]"r"(mask),
                          [mem]"r"(baseAddr),
                          [zero]"r"(zero)
                        );
                    v.d.m(i) = tmp;
                    );
#endif
                    return;
            }

            unrolled_loop16(i, 0, Base::Size,
                    register _long j = indexes.d.m(i);
                    register _long zero = 0;
                    register EntryType tmp = v.d.m(i);
                    if (sizeof(EntryType) == 2) asm volatile(
                        "test %[i],%[mask]\n\t"
                        "cmove %[zero],%[j]\n\t"
                        "cmovne (%[mem],%[j],2),%[tmp]"
                        : [tmp]"+r"(tmp),
                          [j]"+r"(j)
                        : [i]"i"(1 << i),
                          [mask]"r"(mask),
                          [mem]"r"(baseAddr),
                          [zero]"r"(zero)
                        );
                    else if (sizeof(EntryType) == 4) asm volatile(
                        "test %[i],%[mask]\n\t"
                        "cmove %[zero],%[j]\n\t"
                        "cmovne (%[mem],%[j],4),%[tmp]"
                        : [tmp]"+r"(tmp),
                          [j]"+r"(j)
                        : [i]"i"(1 << i),
                          [mask]"r"(mask),
                          [mem]"r"(baseAddr),
                          [zero]"r"(zero)
                        );
                    v.d.m(i) = tmp;
                    );
# endif
#endif
        }

        template<typename Base, typename IndexType, typename EntryType>
        static inline void maskedScatterHelper(
                const Base &v, const IndexType &indexes, _long mask, EntryType *baseAddr
                ) {
#ifndef VC_NO_BSF_LOOPS
            if (sizeof(EntryType) == 2) {
                register _ulong bit;
                register _ulong index;
                register EntryType value;
                asm volatile(
                        SLOWDOWN_ASM
                        "jmp 1f"                "\n\t"
                        ALIGN_16
                        "0:"                   "\n\t"
                        "movzwl (%5,%0,2),%%ecx""\n\t" // ecx contains the index
                        "btr %0,%1"            "\n\t"
                        "movw (%7,%0,2),%3"    "\n\t"  // %3 contains the value to copy
                        "movw %3,(%6,%%rcx,2)" "\n\t"  // store the value into baseAddr[ecx]
                        "1:"                   "\n\t"
                        "bsf %1,%0"            "\n\t"
                        "jnz 0b"               "\n\t"
                        : "=&r"(bit), "+r"(mask), "=&r"(index), "=&r"(value), "+m"(*baseAddr)
                        : "r"(&indexes.d.v()), "r"(baseAddr), "r"(&v.d), "m"(indexes.d.v())
                        : "rcx"   );
            } else if (sizeof(EntryType) == 4) {
                if (sizeof(typename IndexType::EntryType) == 4) {
                    register _ulong bit;
                    register _ulong index;
                    register EntryType value;
                    asm volatile(
                            SLOWDOWN_ASM
                            "jmp 1f"                "\n\t"
                            ALIGN_16
                            "0:"                   "\n\t"
                            "mov (%5,%0,4),%%ecx"  "\n\t" // ecx contains the index
                            "btr %0,%1"            "\n\t"
                            "mov (%7,%0,4),%3"    "\n\t"  // %3 contains the value to copy
                            "mov %3,(%6,%%rcx,4)" "\n\t"  // store the value into baseAddr[ecx]
                            "1:"                   "\n\t"
                            "bsf %1,%0"            "\n\t"
                            "jnz 0b"               "\n\t"
                            : "=&r"(bit), "+r"(mask), "=&r"(index), "=&r"(value), "+m"(*baseAddr)
                            : "r"(&indexes.d.v()), "r"(baseAddr), "r"(&v.d), "m"(indexes.d.v())
                            : "rcx"   );
                } else if (sizeof(typename IndexType::EntryType) == 2) { // sfloat_v[ushort_v]
                    register _ulong bit;
                    register _ulong index;
                    register EntryType value;
                    asm volatile(
                            SLOWDOWN_ASM
                            "jmp 1f"                "\n\t"
                            ALIGN_16
                            "0:"                   "\n\t"
                            "movzwl (%5,%0,2),%%ecx""\n\t" // ecx contains the index
                            "btr %0,%1"            "\n\t"
                            "mov (%7,%0,4),%3"    "\n\t"  // %3 contains the value to copy
                            "mov %3,(%6,%%rcx,4)" "\n\t"  // store the value into baseAddr[ecx]
                            "1:"                   "\n\t"
                            "bsf %1,%0"            "\n\t"
                            "jnz 0b"               "\n\t"
                            : "=&r"(bit), "+r"(mask), "=&r"(index), "=&r"(value), "+m"(*baseAddr)
                            : "r"(&indexes.d.v()), "r"(baseAddr), "r"(&v.d), "m"(indexes.d.v())
                            : "rcx"   );
                } else {
                    abort();
                }
            } else if (sizeof(EntryType) == 8) {
                register _ulong bit;
                register _ulong index;
                register EntryType value;
                asm volatile(
                        SLOWDOWN_ASM
                        "jmp 1f"                "\n\t"
                        ALIGN_16
                        "0:"                   "\n\t"
                        "mov (%5,%0,4),%%ecx"  "\n\t" // ecx contains the index
                        "btr %0,%1"            "\n\t"
                        "mov (%7,%0,8),%3"    "\n\t"  // %3 contains the value to copy
                        "mov %3,(%6,%%rcx,8)" "\n\t"  // store the value into baseAddr[ecx]
                        "1:"                   "\n\t"
                        "bsf %1,%0"            "\n\t"
                        "jnz 0b"               "\n\t"
                        : "=&r"(bit), "+r"(mask), "=&r"(index), "=&r"(value), "+m"(*baseAddr)
                        : "r"(&indexes.d.v()), "r"(baseAddr), "r"(&v.d), "m"(indexes.d.v())
                        : "rcx"   );
            } else {
                abort();
            }
#else
            unrolled_loop16(i, 0, Base::Size,
                    if (mask & (1 << i)) baseAddr[indexes.d.m(i)] = v.d.m(i);
                    );
#endif
        }
    };

    ////////////////////////////////////////////////////////
    // Array gathers
    template<typename T> inline void GatherHelper<T>::gather(
            Base &v, const unsigned int *indexes, const EntryType *baseAddr)
    {
        for_all_vector_entries(i,
                v.d.m(i) = baseAddr[indexes[i]];
                );
    }
    template<> inline void GatherHelper<double>::gather(
            Base &v, const unsigned int *indexes, const EntryType *baseAddr)
    {
        v.d.v() = _mm_set_pd(baseAddr[indexes[1]], baseAddr[indexes[0]]);
    }
    template<> inline void GatherHelper<float>::gather(
            Base &v, const unsigned int *indexes, const EntryType *baseAddr)
    {
        v.d.v() = _mm_set_ps(
                baseAddr[indexes[3]], baseAddr[indexes[2]],
                baseAddr[indexes[1]], baseAddr[indexes[0]]);
    }
    template<> inline void GatherHelper<float8>::gather(
            Base &v, const unsigned int *indexes, const EntryType *baseAddr)
    {
        v.d.v()[1] = _mm_set_ps(
                baseAddr[indexes[7]], baseAddr[indexes[6]],
                baseAddr[indexes[5]], baseAddr[indexes[4]]);
        v.d.v()[0] = _mm_set_ps(
                baseAddr[indexes[3]], baseAddr[indexes[2]],
                baseAddr[indexes[1]], baseAddr[indexes[0]]);
    }
    template<> inline void GatherHelper<int>::gather(
            Base &v, const unsigned int *indexes, const EntryType *baseAddr)
    {
        v.d.v() = _mm_set_epi32(
                baseAddr[indexes[3]], baseAddr[indexes[2]],
                baseAddr[indexes[1]], baseAddr[indexes[0]]);
    }
    template<> inline void GatherHelper<unsigned int>::gather(
            Base &v, const unsigned int *indexes, const EntryType *baseAddr)
    {
        v.d.v() = _mm_set_epi32(
                baseAddr[indexes[3]], baseAddr[indexes[2]],
                baseAddr[indexes[1]], baseAddr[indexes[0]]);
    }
    template<> inline void GatherHelper<short>::gather(
            Base &v, const unsigned int *indexes, const EntryType *baseAddr)
    {
        v.d.v() = _mm_set_epi16(
                baseAddr[indexes[7]], baseAddr[indexes[6]],
                baseAddr[indexes[5]], baseAddr[indexes[4]],
                baseAddr[indexes[3]], baseAddr[indexes[2]],
                baseAddr[indexes[1]], baseAddr[indexes[0]]);
    }
    template<> inline void GatherHelper<unsigned short>::gather(
            Base &v, const unsigned int *indexes, const EntryType *baseAddr)
    {
        v.d.v() = _mm_set_epi16(
                baseAddr[indexes[7]], baseAddr[indexes[6]],
                baseAddr[indexes[5]], baseAddr[indexes[4]],
                baseAddr[indexes[3]], baseAddr[indexes[2]],
                baseAddr[indexes[1]], baseAddr[indexes[0]]);
    }
    template<typename T> inline void GatherHelper<T>::gather(
            Base &v, const IndexType &indexes, const EntryType *baseAddr)
    {
        for_all_vector_entries(i,
                v.d.m(i) = baseAddr[indexes.d.m(i)];
                );
    }
    template<> inline void GatherHelper<double>::gather(
            Base &v, const IndexType &indexes, const EntryType *baseAddr)
    {
        v.d.v() = _mm_set_pd(baseAddr[indexes.d.m(1)], baseAddr[indexes.d.m(0)]);
    }
    template<> inline void GatherHelper<float>::gather(
            Base &v, const IndexType &indexes, const EntryType *baseAddr)
    {
        v.d.v() = _mm_set_ps(
                baseAddr[indexes.d.m(3)], baseAddr[indexes.d.m(2)],
                baseAddr[indexes.d.m(1)], baseAddr[indexes.d.m(0)]);
    }
    template<> inline void GatherHelper<float8>::gather(
            Base &v, const IndexType &indexes, const EntryType *baseAddr)
    {
        v.d.v()[1] = _mm_set_ps(
                baseAddr[indexes.d.m(7)], baseAddr[indexes.d.m(6)],
                baseAddr[indexes.d.m(5)], baseAddr[indexes.d.m(4)]);
        v.d.v()[0] = _mm_set_ps(
                baseAddr[indexes.d.m(3)], baseAddr[indexes.d.m(2)],
                baseAddr[indexes.d.m(1)], baseAddr[indexes.d.m(0)]);
    }
    template<> inline void GatherHelper<int>::gather(
            Base &v, const IndexType &indexes, const EntryType *baseAddr)
    {
        v.d.v() = _mm_set_epi32(
                baseAddr[indexes.d.m(3)], baseAddr[indexes.d.m(2)],
                baseAddr[indexes.d.m(1)], baseAddr[indexes.d.m(0)]);
    }
    template<> inline void GatherHelper<unsigned int>::gather(
            Base &v, const IndexType &indexes, const EntryType *baseAddr)
    {
        v.d.v() = _mm_set_epi32(
                baseAddr[indexes.d.m(3)], baseAddr[indexes.d.m(2)],
                baseAddr[indexes.d.m(1)], baseAddr[indexes.d.m(0)]);
    }
    template<> inline void GatherHelper<short>::gather(
            Base &v, const IndexType &indexes, const EntryType *baseAddr)
    {
        v.d.v() = _mm_set_epi16(
                baseAddr[indexes.d.m(7)], baseAddr[indexes.d.m(6)],
                baseAddr[indexes.d.m(5)], baseAddr[indexes.d.m(4)],
                baseAddr[indexes.d.m(3)], baseAddr[indexes.d.m(2)],
                baseAddr[indexes.d.m(1)], baseAddr[indexes.d.m(0)]);
    }
    template<> inline void GatherHelper<unsigned short>::gather(
            Base &v, const IndexType &indexes, const EntryType *baseAddr)
    {
        v.d.v() = _mm_set_epi16(
                baseAddr[indexes.d.m(7)], baseAddr[indexes.d.m(6)],
                baseAddr[indexes.d.m(5)], baseAddr[indexes.d.m(4)],
                baseAddr[indexes.d.m(3)], baseAddr[indexes.d.m(2)],
                baseAddr[indexes.d.m(1)], baseAddr[indexes.d.m(0)]);
    }

    ////////////////////////////////////////////////////////
    // Struct gathers
    template<typename T> template<typename S1> inline void GatherHelper<T>::gather(
            Base &v, const IndexType &indexes, const S1 *baseAddr, const EntryType S1::* member1)
    {
        for_all_vector_entries(i,
                v.d.m(i) = baseAddr[indexes.d.m(i)].*(member1);
                );
    }
    template<> template<typename S1> inline void GatherHelper<double>::gather(
            Base &v, const IndexType &indexes, const S1 *baseAddr, const EntryType S1::* member1)
    {
        v.d.v() = _mm_set_pd(baseAddr[indexes.d.m(1)].*(member1), baseAddr[indexes.d.m(0)].*(member1));
    }
    template<> template<typename S1> inline void GatherHelper<float>::gather(
            Base &v, const IndexType &indexes, const S1 *baseAddr, const EntryType S1::* member1)
    {
        v.d.v() = _mm_set_ps(
                baseAddr[indexes.d.m(3)].*(member1), baseAddr[indexes.d.m(2)].*(member1),
                baseAddr[indexes.d.m(1)].*(member1), baseAddr[indexes.d.m(0)].*(member1));
    }
    template<> template<typename S1> inline void GatherHelper<float8>::gather(
            Base &v, const IndexType &indexes, const S1 *baseAddr, const EntryType S1::* member1)
    {
        v.d.v()[1] = _mm_set_ps(
                baseAddr[indexes.d.m(7)].*(member1), baseAddr[indexes.d.m(6)].*(member1),
                baseAddr[indexes.d.m(5)].*(member1), baseAddr[indexes.d.m(4)].*(member1));
        v.d.v()[0] = _mm_set_ps(
                baseAddr[indexes.d.m(3)].*(member1), baseAddr[indexes.d.m(2)].*(member1),
                baseAddr[indexes.d.m(1)].*(member1), baseAddr[indexes.d.m(0)].*(member1));
    }
    template<> template<typename S1> inline void GatherHelper<int>::gather(
            Base &v, const IndexType &indexes, const S1 *baseAddr, const EntryType S1::* member1)
    {
        v.d.v() = _mm_set_epi32(
                baseAddr[indexes.d.m(3)].*(member1), baseAddr[indexes.d.m(2)].*(member1),
                baseAddr[indexes.d.m(1)].*(member1), baseAddr[indexes.d.m(0)].*(member1));
    }
    template<> template<typename S1> inline void GatherHelper<unsigned int>::gather(
            Base &v, const IndexType &indexes, const S1 *baseAddr, const EntryType S1::* member1)
    {
        v.d.v() = _mm_set_epi32(
                baseAddr[indexes.d.m(3)].*(member1), baseAddr[indexes.d.m(2)].*(member1),
                baseAddr[indexes.d.m(1)].*(member1), baseAddr[indexes.d.m(0)].*(member1));
    }
    template<> template<typename S1> inline void GatherHelper<short>::gather(
            Base &v, const IndexType &indexes, const S1 *baseAddr, const EntryType S1::* member1)
    {
        v.d.v() = _mm_set_epi16(
                baseAddr[indexes.d.m(7)].*(member1), baseAddr[indexes.d.m(6)].*(member1),
                baseAddr[indexes.d.m(5)].*(member1), baseAddr[indexes.d.m(4)].*(member1),
                baseAddr[indexes.d.m(3)].*(member1), baseAddr[indexes.d.m(2)].*(member1),
                baseAddr[indexes.d.m(1)].*(member1), baseAddr[indexes.d.m(0)].*(member1));
    }
    template<> template<typename S1> inline void GatherHelper<unsigned short>::gather(
            Base &v, const IndexType &indexes, const S1 *baseAddr, const EntryType S1::* member1)
    {
        v.d.v() = _mm_set_epi16(
                baseAddr[indexes.d.m(7)].*(member1), baseAddr[indexes.d.m(6)].*(member1),
                baseAddr[indexes.d.m(5)].*(member1), baseAddr[indexes.d.m(4)].*(member1),
                baseAddr[indexes.d.m(3)].*(member1), baseAddr[indexes.d.m(2)].*(member1),
                baseAddr[indexes.d.m(1)].*(member1), baseAddr[indexes.d.m(0)].*(member1));
    }

    ////////////////////////////////////////////////////////
    // Struct of Struct gathers
    template<typename T> template<typename S1, typename S2> inline void GatherHelper<T>::gather(
            Base &v, const IndexType &indexes, const S1 *baseAddr, const S2 S1::* member1, const EntryType S2::* member2)
    {
        for_all_vector_entries(i,
                v.d.m(i) = baseAddr[indexes.d.m(i)].*(member1).*(member2);
                );
    }
    template<> template<typename S1, typename S2> inline void GatherHelper<double>::gather(
            Base &v, const IndexType &indexes, const S1 *baseAddr, const S2 S1::* member1, const EntryType S2::* member2)
    {
        v.d.v() = _mm_set_pd(baseAddr[indexes.d.m(1)].*(member1).*(member2), baseAddr[indexes.d.m(0)].*(member1).*(member2));
    }
    template<> template<typename S1, typename S2> inline void GatherHelper<float>::gather(
            Base &v, const IndexType &indexes, const S1 *baseAddr, const S2 S1::* member1, const EntryType S2::* member2)
    {
        v.d.v() = _mm_set_ps(
                baseAddr[indexes.d.m(3)].*(member1).*(member2), baseAddr[indexes.d.m(2)].*(member1).*(member2),
                baseAddr[indexes.d.m(1)].*(member1).*(member2), baseAddr[indexes.d.m(0)].*(member1).*(member2));
    }
    template<> template<typename S1, typename S2> inline void GatherHelper<float8>::gather(
            Base &v, const IndexType &indexes, const S1 *baseAddr, const S2 S1::* member1, const EntryType S2::* member2)
    {
        v.d.v()[1] = _mm_set_ps(
                baseAddr[indexes.d.m(7)].*(member1).*(member2), baseAddr[indexes.d.m(6)].*(member1).*(member2),
                baseAddr[indexes.d.m(5)].*(member1).*(member2), baseAddr[indexes.d.m(4)].*(member1).*(member2));
        v.d.v()[0] = _mm_set_ps(
                baseAddr[indexes.d.m(3)].*(member1).*(member2), baseAddr[indexes.d.m(2)].*(member1).*(member2),
                baseAddr[indexes.d.m(1)].*(member1).*(member2), baseAddr[indexes.d.m(0)].*(member1).*(member2));
    }
    template<> template<typename S1, typename S2> inline void GatherHelper<int>::gather(
            Base &v, const IndexType &indexes, const S1 *baseAddr, const S2 S1::* member1, const EntryType S2::* member2)
    {
        v.d.v() = _mm_set_epi32(
                baseAddr[indexes.d.m(3)].*(member1).*(member2), baseAddr[indexes.d.m(2)].*(member1).*(member2),
                baseAddr[indexes.d.m(1)].*(member1).*(member2), baseAddr[indexes.d.m(0)].*(member1).*(member2));
    }
    template<> template<typename S1, typename S2> inline void GatherHelper<unsigned int>::gather(
            Base &v, const IndexType &indexes, const S1 *baseAddr, const S2 S1::* member1, const EntryType S2::* member2)
    {
        v.d.v() = _mm_set_epi32(
                baseAddr[indexes.d.m(3)].*(member1).*(member2), baseAddr[indexes.d.m(2)].*(member1).*(member2),
                baseAddr[indexes.d.m(1)].*(member1).*(member2), baseAddr[indexes.d.m(0)].*(member1).*(member2));
    }
    template<> template<typename S1, typename S2> inline void GatherHelper<short>::gather(
            Base &v, const IndexType &indexes, const S1 *baseAddr, const S2 S1::* member1, const EntryType S2::* member2)
    {
        v.d.v() = _mm_set_epi16(
                baseAddr[indexes.d.m(7)].*(member1).*(member2), baseAddr[indexes.d.m(6)].*(member1).*(member2),
                baseAddr[indexes.d.m(5)].*(member1).*(member2), baseAddr[indexes.d.m(4)].*(member1).*(member2),
                baseAddr[indexes.d.m(3)].*(member1).*(member2), baseAddr[indexes.d.m(2)].*(member1).*(member2),
                baseAddr[indexes.d.m(1)].*(member1).*(member2), baseAddr[indexes.d.m(0)].*(member1).*(member2));
    }
    template<> template<typename S1, typename S2> inline void GatherHelper<unsigned short>::gather(
            Base &v, const IndexType &indexes, const S1 *baseAddr, const S2 S1::* member1, const EntryType S2::* member2)
    {
        v.d.v() = _mm_set_epi16(
                baseAddr[indexes.d.m(7)].*(member1).*(member2), baseAddr[indexes.d.m(6)].*(member1).*(member2),
                baseAddr[indexes.d.m(5)].*(member1).*(member2), baseAddr[indexes.d.m(4)].*(member1).*(member2),
                baseAddr[indexes.d.m(3)].*(member1).*(member2), baseAddr[indexes.d.m(2)].*(member1).*(member2),
                baseAddr[indexes.d.m(1)].*(member1).*(member2), baseAddr[indexes.d.m(0)].*(member1).*(member2));
    }

    ////////////////////////////////////////////////////////
    // Scatters
    //
    // There is no equivalent to the set intrinsics. Therefore the vector entries are copied in
    // memory instead from the xmm register directly.
    //
    // TODO: With SSE 4.1 the extract intrinsics might be an interesting option, though.
    //
    template<typename T> inline void ScatterHelper<T>::scatter(
            const Base &v, const IndexType &indexes, EntryType *baseAddr) {
        for_all_vector_entries(i,
                baseAddr[indexes.d.m(i)] = v.d.m(i);
                );
    }
    template<> inline void ScatterHelper<short>::scatter(
            const Base &v, const IndexType &indexes, EntryType *baseAddr) {
        // TODO: verify that using extract is really faster
        for_all_vector_entries(i,
                baseAddr[indexes.d.m(i)] = _mm_extract_epi16(v.d.v(), i);
                );
    }

    template<typename T> inline void ScatterHelper<T>::scatter(
            const Base &v, const IndexType &indexes, int mask, EntryType *baseAddr) {
        GeneralHelpers::maskedScatterHelper(v, indexes, mask, baseAddr);
    }

    template<typename T> template<typename S1> inline void ScatterHelper<T>::scatter(
            const Base &v, const IndexType &indexes, S1 *baseAddr, EntryType S1::* member1) {
        for_all_vector_entries(i,
                baseAddr[indexes.d.m(i)].*(member1) = v.d.m(i);
                );
    }

    template<typename T> template<typename S1> inline void ScatterHelper<T>::scatter(
            const Base &v, const IndexType &indexes, int mask, S1 *baseAddr, EntryType S1::* member1) {
        for_all_vector_entries(i,
                if (mask & (1 << i)) baseAddr[indexes.d.m(i)].*(member1) = v.d.m(i);
                );
    }

    template<typename T> template<typename S1, typename S2> inline void ScatterHelper<T>::scatter(
            const Base &v, const IndexType &indexes, S1 *baseAddr, S2 S1::* member1, EntryType S2::* member2) {
        for_all_vector_entries(i,
                baseAddr[indexes.d.m(i)].*(member1).*(member2) = v.d.m(i);
                );
    }

    template<typename T> template<typename S1, typename S2> inline void ScatterHelper<T>::scatter(
            const Base &v, const IndexType &indexes, int mask, S1 *baseAddr, S2 S1::* member1,
            EntryType S2::* member2) {
        for_all_vector_entries(i,
                if (mask & (1 << i)) baseAddr[indexes.d.m(i)].*(member1).*(member2) = v.d.m(i);
                );
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
#ifndef VC_IMPL_SSE4_1
        // without SSE 4.1 int multiplication is not so nice
        if (sizeof(typename IndexType::EntryType) == 4) {
            switch (constant) {
                case    3: return H::add(        x.data()    , H::slli(x.data(),  1));
                case    5: return H::add(        x.data()    , H::slli(x.data(),  2));
                case    9: return H::add(        x.data()    , H::slli(x.data(),  3));
                case   17: return H::add(        x.data()    , H::slli(x.data(),  4));
                case   33: return H::add(        x.data()    , H::slli(x.data(),  5));
                case   65: return H::add(        x.data()    , H::slli(x.data(),  6));
                case  129: return H::add(        x.data()    , H::slli(x.data(),  7));
                case  257: return H::add(        x.data()    , H::slli(x.data(),  8));
                case  513: return H::add(        x.data()    , H::slli(x.data(),  9));
                case 1025: return H::add(        x.data()    , H::slli(x.data(), 10));
                case 2049: return H::add(        x.data()    , H::slli(x.data(), 11));
                case    6: return H::add(H::slli(x.data(), 1), H::slli(x.data(),  2));
                case   10: return H::add(H::slli(x.data(), 1), H::slli(x.data(),  3));
                case   18: return H::add(H::slli(x.data(), 1), H::slli(x.data(),  4));
                case   34: return H::add(H::slli(x.data(), 1), H::slli(x.data(),  5));
                case   66: return H::add(H::slli(x.data(), 1), H::slli(x.data(),  6));
                case  130: return H::add(H::slli(x.data(), 1), H::slli(x.data(),  7));
                case  258: return H::add(H::slli(x.data(), 1), H::slli(x.data(),  8));
                case  514: return H::add(H::slli(x.data(), 1), H::slli(x.data(),  9));
                case 1026: return H::add(H::slli(x.data(), 1), H::slli(x.data(), 10));
                case 2050: return H::add(H::slli(x.data(), 1), H::slli(x.data(), 11));
                case   12: return H::add(H::slli(x.data(), 2), H::slli(x.data(),  3));
                case   20: return H::add(H::slli(x.data(), 2), H::slli(x.data(),  4));
                case   36: return H::add(H::slli(x.data(), 2), H::slli(x.data(),  5));
                case   68: return H::add(H::slli(x.data(), 2), H::slli(x.data(),  6));
                case  132: return H::add(H::slli(x.data(), 2), H::slli(x.data(),  7));
                case  260: return H::add(H::slli(x.data(), 2), H::slli(x.data(),  8));
                case  516: return H::add(H::slli(x.data(), 2), H::slli(x.data(),  9));
                case 1028: return H::add(H::slli(x.data(), 2), H::slli(x.data(), 10));
                case 2052: return H::add(H::slli(x.data(), 2), H::slli(x.data(), 11));
                case   24: return H::add(H::slli(x.data(), 3), H::slli(x.data(),  4));
                case   40: return H::add(H::slli(x.data(), 3), H::slli(x.data(),  5));
                case   72: return H::add(H::slli(x.data(), 3), H::slli(x.data(),  6));
                case  136: return H::add(H::slli(x.data(), 3), H::slli(x.data(),  7));
                case  264: return H::add(H::slli(x.data(), 3), H::slli(x.data(),  8));
                case  520: return H::add(H::slli(x.data(), 3), H::slli(x.data(),  9));
                case 1032: return H::add(H::slli(x.data(), 3), H::slli(x.data(), 10));
                case 2056: return H::add(H::slli(x.data(), 3), H::slli(x.data(), 11));
                case   48: return H::add(H::slli(x.data(), 4), H::slli(x.data(),  5));
                case   80: return H::add(H::slli(x.data(), 4), H::slli(x.data(),  6));
                case  144: return H::add(H::slli(x.data(), 4), H::slli(x.data(),  7));
                case  272: return H::add(H::slli(x.data(), 4), H::slli(x.data(),  8));
                case  528: return H::add(H::slli(x.data(), 4), H::slli(x.data(),  9));
                case 1040: return H::add(H::slli(x.data(), 4), H::slli(x.data(), 10));
                case 2064: return H::add(H::slli(x.data(), 4), H::slli(x.data(), 11));
                case   96: return H::add(H::slli(x.data(), 5), H::slli(x.data(),  6));
                case  160: return H::add(H::slli(x.data(), 5), H::slli(x.data(),  7));
                case  288: return H::add(H::slli(x.data(), 5), H::slli(x.data(),  8));
                case  544: return H::add(H::slli(x.data(), 5), H::slli(x.data(),  9));
                case 1056: return H::add(H::slli(x.data(), 5), H::slli(x.data(), 10));
                case 2080: return H::add(H::slli(x.data(), 5), H::slli(x.data(), 11));
                case  192: return H::add(H::slli(x.data(), 6), H::slli(x.data(),  7));
                case  320: return H::add(H::slli(x.data(), 6), H::slli(x.data(),  8));
                case  576: return H::add(H::slli(x.data(), 6), H::slli(x.data(),  9));
                case 1088: return H::add(H::slli(x.data(), 6), H::slli(x.data(), 10));
                case 2112: return H::add(H::slli(x.data(), 6), H::slli(x.data(), 11));
                case  384: return H::add(H::slli(x.data(), 7), H::slli(x.data(),  8));
                case  640: return H::add(H::slli(x.data(), 7), H::slli(x.data(),  9));
                case 1152: return H::add(H::slli(x.data(), 7), H::slli(x.data(), 10));
                case 2176: return H::add(H::slli(x.data(), 7), H::slli(x.data(), 11));
                case  768: return H::add(H::slli(x.data(), 8), H::slli(x.data(),  9));
                case 1280: return H::add(H::slli(x.data(), 8), H::slli(x.data(), 10));
                case 2304: return H::add(H::slli(x.data(), 8), H::slli(x.data(), 11));
                case 1536: return H::add(H::slli(x.data(), 9), H::slli(x.data(), 10));
                case 2560: return H::add(H::slli(x.data(), 9), H::slli(x.data(), 11));
                case 3072: return H::add(H::slli(x.data(),10), H::slli(x.data(), 11));
            }
        }
#endif
        return H::mul(x.data(), H::set(constant));
    }
} // namespace SSE
} // namespace Vc

#undef SLOWDOWN_ASM
#undef ALIGN_16
