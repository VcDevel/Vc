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

#ifndef VC_AVX_SHUFFLE_H
#define VC_AVX_SHUFFLE_H

#include "../sse/shuffle.h"

namespace Vc
{
    namespace Mem
    {
        template<VecPos L, VecPos H> __m256 ALWAYS_INLINE CONST permute128(__m256 x, __m256 y) {
            VC_STATIC_ASSERT(L >= X0 && H >= X0, Incorrect_Range);
            VC_STATIC_ASSERT(L <= Y1 && H <= Y1, Incorrect_Range);
            return _mm256_permute2f128_ps(x, y, (L < Y0 ? L : L - Y0 + 2) + (H < Y0 ? H : H - Y0 + 2) * (1 << 4));
        }
        template<VecPos L, VecPos H> __m256i ALWAYS_INLINE CONST permute128(__m256i x, __m256i y) {
            VC_STATIC_ASSERT(L >= X0 && H >= X0, Incorrect_Range);
            VC_STATIC_ASSERT(L <= Y1 && H <= Y1, Incorrect_Range);
            return _mm256_permute2f128_si256(x, y, (L < Y0 ? L : L - Y0 + 2) + (H < Y0 ? H : H - Y0 + 2) * (1 << 4));
        }
        template<VecPos L, VecPos H> __m256d ALWAYS_INLINE CONST permute128(__m256d x, __m256d y) {
            VC_STATIC_ASSERT(L >= X0 && H >= X0, Incorrect_Range);
            VC_STATIC_ASSERT(L <= Y1 && H <= Y1, Incorrect_Range);
            return _mm256_permute2f128_pd(x, y, (L < Y0 ? L : L - Y0 + 2) + (H < Y0 ? H : H - Y0 + 2) * (1 << 4));
        }
        template<VecPos Dst0, VecPos Dst1, VecPos Dst2, VecPos Dst3> __m256d ALWAYS_INLINE CONST permute(__m256d x) {
            VC_STATIC_ASSERT(Dst0 >= X0 && Dst1 >= X0 && Dst2 >= X2 && Dst3 >= X2, Incorrect_Range);
            VC_STATIC_ASSERT(Dst0 <= X1 && Dst1 <= X1 && Dst2 <= X3 && Dst3 <= X3, Incorrect_Range);
            return _mm256_permute_pd(x, Dst0 + Dst1 * 2 + (Dst2 - X2) * 4 + (Dst3 - X2) * 8);
        }
        template<VecPos Dst0, VecPos Dst1, VecPos Dst2, VecPos Dst3> __m256 ALWAYS_INLINE CONST permute(__m256 x) {
            VC_STATIC_ASSERT(Dst0 >= X0 && Dst1 >= X0 && Dst2 >= X0 && Dst3 >= X0, Incorrect_Range);
            VC_STATIC_ASSERT(Dst0 <= X3 && Dst1 <= X3 && Dst2 <= X3 && Dst3 <= X3, Incorrect_Range);
            return _mm256_permute_ps(x, Dst0 + Dst1 * 4 + Dst2 * 16 + Dst3 * 64);
        }
        template<VecPos Dst0, VecPos Dst1, VecPos Dst2, VecPos Dst3> __m256d ALWAYS_INLINE CONST shuffle(__m256d x, __m256d y) {
            VC_STATIC_ASSERT(Dst0 >= X0 && Dst1 >= Y0 && Dst2 >= X2 && Dst3 >= Y2, Incorrect_Range);
            VC_STATIC_ASSERT(Dst0 <= X1 && Dst1 <= Y1 && Dst2 <= X3 && Dst3 <= Y3, Incorrect_Range);
            return _mm256_shuffle_pd(x, y, Dst0 + (Dst1 - Y0) * 2 + (Dst2 - X2) * 4 + (Dst3 - Y2) * 8);
        }
        template<VecPos Dst0, VecPos Dst1, VecPos Dst2, VecPos Dst3> __m256 ALWAYS_INLINE CONST shuffle(__m256 x, __m256 y) {
            VC_STATIC_ASSERT(Dst0 >= X0 && Dst1 >= X0 && Dst2 >= Y0 && Dst3 >= Y0, Incorrect_Range);
            VC_STATIC_ASSERT(Dst0 <= X3 && Dst1 <= X3 && Dst2 <= Y3 && Dst3 <= Y3, Incorrect_Range);
            return _mm256_shuffle_ps(x, y, Dst0 + Dst1 * 4 + (Dst2 - Y0) * 16 + (Dst3 - Y0) * 64);
        }
    } // namespace Mem

    // little endian has the lo bits on the right and high bits on the left
    // with vectors this becomes greatly confusing:
    // Mem: abcd
    // Reg: dcba
    //
    // The shuffles and permutes above use memory ordering. The ones below use register ordering:
    namespace Reg
    {
        template<VecPos H, VecPos L> __m256 ALWAYS_INLINE CONST permute128(__m256 x, __m256 y) {
            VC_STATIC_ASSERT(L >= X0 && H >= X0, Incorrect_Range);
            VC_STATIC_ASSERT(L <= Y1 && H <= Y1, Incorrect_Range);
            return _mm256_permute2f128_ps(x, y, (L < Y0 ? L : L - Y0 + 2) + (H < Y0 ? H : H - Y0 + 2) * (1 << 4));
        }
        template<VecPos H, VecPos L> __m256i ALWAYS_INLINE CONST permute128(__m256i x, __m256i y) {
            VC_STATIC_ASSERT(L >= X0 && H >= X0, Incorrect_Range);
            VC_STATIC_ASSERT(L <= Y1 && H <= Y1, Incorrect_Range);
            return _mm256_permute2f128_si256(x, y, (L < Y0 ? L : L - Y0 + 2) + (H < Y0 ? H : H - Y0 + 2) * (1 << 4));
        }
        template<VecPos H, VecPos L> __m256d ALWAYS_INLINE CONST permute128(__m256d x, __m256d y) {
            VC_STATIC_ASSERT(L >= X0 && H >= X0, Incorrect_Range);
            VC_STATIC_ASSERT(L <= Y1 && H <= Y1, Incorrect_Range);
            return _mm256_permute2f128_pd(x, y, (L < Y0 ? L : L - Y0 + 2) + (H < Y0 ? H : H - Y0 + 2) * (1 << 4));
        }
        template<VecPos Dst3, VecPos Dst2, VecPos Dst1, VecPos Dst0> __m256d ALWAYS_INLINE CONST permute(__m256d x) {
            VC_STATIC_ASSERT(Dst0 >= X0 && Dst1 >= X0 && Dst2 >= X2 && Dst3 >= X2, Incorrect_Range);
            VC_STATIC_ASSERT(Dst0 <= X1 && Dst1 <= X1 && Dst2 <= X3 && Dst3 <= X3, Incorrect_Range);
            return _mm256_permute_pd(x, Dst0 + Dst1 * 2 + (Dst2 - X2) * 4 + (Dst3 - X2) * 8);
        }
        template<VecPos Dst3, VecPos Dst2, VecPos Dst1, VecPos Dst0> __m256 ALWAYS_INLINE CONST permute(__m256 x) {
            VC_STATIC_ASSERT(Dst0 >= X0 && Dst1 >= X0 && Dst2 >= X0 && Dst3 >= X0, Incorrect_Range);
            VC_STATIC_ASSERT(Dst0 <= X3 && Dst1 <= X3 && Dst2 <= X3 && Dst3 <= X3, Incorrect_Range);
            return _mm256_permute_ps(x, Dst0 + Dst1 * 4 + Dst2 * 16 + Dst3 * 64);
        }
        template<VecPos Dst1, VecPos Dst0> __m128d ALWAYS_INLINE CONST permute(__m128d x) {
            VC_STATIC_ASSERT(Dst0 >= X0 && Dst1 >= X0, Incorrect_Range);
            VC_STATIC_ASSERT(Dst0 <= X1 && Dst1 <= X1, Incorrect_Range);
            return _mm_permute_pd(x, Dst0 + Dst1 * 2);
        }
        template<VecPos Dst3, VecPos Dst2, VecPos Dst1, VecPos Dst0> __m128 ALWAYS_INLINE CONST permute(__m128 x) {
            VC_STATIC_ASSERT(Dst0 >= X0 && Dst1 >= X0 && Dst2 >= X0 && Dst3 >= X0, Incorrect_Range);
            VC_STATIC_ASSERT(Dst0 <= X3 && Dst1 <= X3 && Dst2 <= X3 && Dst3 <= X3, Incorrect_Range);
            return _mm_permute_ps(x, Dst0 + Dst1 * 4 + Dst2 * 16 + Dst3 * 64);
        }
        template<VecPos Dst3, VecPos Dst2, VecPos Dst1, VecPos Dst0> __m256d ALWAYS_INLINE CONST shuffle(__m256d x, __m256d y) {
            VC_STATIC_ASSERT(Dst0 >= X0 && Dst1 >= Y0 && Dst2 >= X2 && Dst3 >= Y2, Incorrect_Range);
            VC_STATIC_ASSERT(Dst0 <= X1 && Dst1 <= Y1 && Dst2 <= X3 && Dst3 <= Y3, Incorrect_Range);
            return _mm256_shuffle_pd(x, y, Dst0 + (Dst1 - Y0) * 2 + (Dst2 - X2) * 4 + (Dst3 - Y2) * 8);
        }
        template<VecPos Dst3, VecPos Dst2, VecPos Dst1, VecPos Dst0> __m256 ALWAYS_INLINE CONST shuffle(__m256 x, __m256 y) {
            VC_STATIC_ASSERT(Dst0 >= X0 && Dst1 >= X0 && Dst2 >= Y0 && Dst3 >= Y0, Incorrect_Range);
            VC_STATIC_ASSERT(Dst0 <= X3 && Dst1 <= X3 && Dst2 <= Y3 && Dst3 <= Y3, Incorrect_Range);
            return _mm256_shuffle_ps(x, y, Dst0 + Dst1 * 4 + (Dst2 - Y0) * 16 + (Dst3 - Y0) * 64);
        }
    } // namespace Reg
} // namespace Vc

#endif // VC_AVX_SHUFFLE_H
