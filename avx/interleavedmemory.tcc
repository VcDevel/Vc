/*  This file is part of the Vc library. {{{

    Copyright (C) 2012 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_AVX_INTERLEAVEDMEMORY_TCC
#define VC_AVX_INTERLEAVEDMEMORY_TCC

namespace Vc
{
namespace Common
{

template<> inline void InterleavedMemoryAccessBase<float_v>::deinterleave(float_v &v0, float_v &v1) const
{
    const __m128  il0 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 const *>(&m_data[m_indexes[0]])); // a0 b0
    const __m128  il2 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 const *>(&m_data[m_indexes[2]])); // a2 b2
    const __m128  il4 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 const *>(&m_data[m_indexes[4]])); // a4 b4
    const __m128  il6 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 const *>(&m_data[m_indexes[6]])); // a6 b6
    const __m128 il01 = _mm_loadh_pi(             il0, reinterpret_cast<__m64 const *>(&m_data[m_indexes[1]])); // a0 b0 a1 b1
    const __m128 il23 = _mm_loadh_pi(             il2, reinterpret_cast<__m64 const *>(&m_data[m_indexes[3]])); // a2 b2 a3 b3
    const __m128 il45 = _mm_loadh_pi(             il4, reinterpret_cast<__m64 const *>(&m_data[m_indexes[5]])); // a4 b4 a5 b5
    const __m128 il67 = _mm_loadh_pi(             il6, reinterpret_cast<__m64 const *>(&m_data[m_indexes[7]])); // a6 b6 a7 b7

    const __m256 tmp2 = AVX::concat(il01, il45);
    const __m256 tmp3 = AVX::concat(il23, il67);

    const __m256 tmp0 = _mm256_unpacklo_ps(tmp2, tmp3);
    const __m256 tmp1 = _mm256_unpackhi_ps(tmp2, tmp3);

    v0.data() = _mm256_unpacklo_ps(tmp0, tmp1);
    v1.data() = _mm256_unpackhi_ps(tmp0, tmp1);
}

template<> inline void InterleavedMemoryAccessBase<float_v>::deinterleave(float_v &v0, float_v &v1, float_v &v2) const
{
    const __m128  il0 = _mm_loadu_ps(&m_data[m_indexes[0]]); // a0 b0 c0 d0
    const __m128  il1 = _mm_loadu_ps(&m_data[m_indexes[1]]); // a1 b1 c1 d1
    const __m128  il2 = _mm_loadu_ps(&m_data[m_indexes[2]]); // a2 b2 c2 d2
    const __m128  il3 = _mm_loadu_ps(&m_data[m_indexes[3]]); // a3 b3 c3 d3
    const __m128  il4 = _mm_loadu_ps(&m_data[m_indexes[4]]); // a4 b4 c4 d4
    const __m128  il5 = _mm_loadu_ps(&m_data[m_indexes[5]]); // a5 b5 c5 d5
    const __m128  il6 = _mm_loadu_ps(&m_data[m_indexes[6]]); // a6 b6 c6 d6
    const __m128  il7 = _mm_loadu_ps(&m_data[m_indexes[7]]); // a7 b7 c7 d7

    const __m256 il04 = AVX::concat(il0, il4);
    const __m256 il15 = AVX::concat(il1, il5);
    const __m256 il26 = AVX::concat(il2, il6);
    const __m256 il37 = AVX::concat(il3, il7);
    const __m256 ab0246 = _mm256_unpacklo_ps(il04, il26);
    const __m256 ab1357 = _mm256_unpacklo_ps(il15, il37);
    const __m256 cd0246 = _mm256_unpackhi_ps(il04, il26);
    const __m256 cd1357 = _mm256_unpackhi_ps(il15, il37);
    v0.data() = _mm256_unpacklo_ps(ab0246, ab1357);
    v1.data() = _mm256_unpackhi_ps(ab0246, ab1357);
    v2.data() = _mm256_unpacklo_ps(cd0246, cd1357);
}

template<> inline void InterleavedMemoryAccessBase<float_v>::deinterleave(float_v &v0, float_v &v1, float_v &v2, float_v &v3) const
{
    const __m128  il0 = _mm_loadu_ps(&m_data[m_indexes[0]]); // a0 b0 c0 d0
    const __m128  il1 = _mm_loadu_ps(&m_data[m_indexes[1]]); // a1 b1 c1 d1
    const __m128  il2 = _mm_loadu_ps(&m_data[m_indexes[2]]); // a2 b2 c2 d2
    const __m128  il3 = _mm_loadu_ps(&m_data[m_indexes[3]]); // a3 b3 c3 d3
    const __m128  il4 = _mm_loadu_ps(&m_data[m_indexes[4]]); // a4 b4 c4 d4
    const __m128  il5 = _mm_loadu_ps(&m_data[m_indexes[5]]); // a5 b5 c5 d5
    const __m128  il6 = _mm_loadu_ps(&m_data[m_indexes[6]]); // a6 b6 c6 d6
    const __m128  il7 = _mm_loadu_ps(&m_data[m_indexes[7]]); // a7 b7 c7 d7

    const __m256 il04 = AVX::concat(il0, il4);
    const __m256 il15 = AVX::concat(il1, il5);
    const __m256 il26 = AVX::concat(il2, il6);
    const __m256 il37 = AVX::concat(il3, il7);
    const __m256 ab0246 = _mm256_unpacklo_ps(il04, il26);
    const __m256 ab1357 = _mm256_unpacklo_ps(il15, il37);
    const __m256 cd0246 = _mm256_unpackhi_ps(il04, il26);
    const __m256 cd1357 = _mm256_unpackhi_ps(il15, il37);
    v0.data() = _mm256_unpacklo_ps(ab0246, ab1357);
    v1.data() = _mm256_unpackhi_ps(ab0246, ab1357);
    v2.data() = _mm256_unpacklo_ps(cd0246, cd1357);
    v3.data() = _mm256_unpackhi_ps(cd0246, cd1357);
}

template<> inline void InterleavedMemoryAccessBase<float_v>::deinterleave(float_v &v0, float_v &v1, float_v &v2, float_v &v3, float_v &v4) const
{
    v4.gather(m_data, m_indexes + I(4));
    deinterleave(v0, v1, v2, v3);
}

template<> inline void InterleavedMemoryAccessBase<float_v>::deinterleave(float_v &v0, float_v &v1, float_v &v2, float_v &v3, float_v &v4, float_v &v5) const
{
    deinterleave(v0, v1, v2, v3);
    const __m128  il0 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 const *>(&m_data[m_indexes[0] + 4])); // a0 b0
    const __m128  il2 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 const *>(&m_data[m_indexes[2] + 4])); // a2 b2
    const __m128  il4 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 const *>(&m_data[m_indexes[4] + 4])); // a4 b4
    const __m128  il6 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 const *>(&m_data[m_indexes[6] + 4])); // a6 b6
    const __m128 il01 = _mm_loadh_pi(             il0, reinterpret_cast<__m64 const *>(&m_data[m_indexes[1] + 4])); // a0 b0 a1 b1
    const __m128 il23 = _mm_loadh_pi(             il2, reinterpret_cast<__m64 const *>(&m_data[m_indexes[3] + 4])); // a2 b2 a3 b3
    const __m128 il45 = _mm_loadh_pi(             il4, reinterpret_cast<__m64 const *>(&m_data[m_indexes[5] + 4])); // a4 b4 a5 b5
    const __m128 il67 = _mm_loadh_pi(             il6, reinterpret_cast<__m64 const *>(&m_data[m_indexes[7] + 4])); // a6 b6 a7 b7

    const __m256 tmp2 = AVX::concat(il01, il45);
    const __m256 tmp3 = AVX::concat(il23, il67);

    const __m256 tmp0 = _mm256_unpacklo_ps(tmp2, tmp3);
    const __m256 tmp1 = _mm256_unpackhi_ps(tmp2, tmp3);

    v4.data() = _mm256_unpacklo_ps(tmp0, tmp1);
    v5.data() = _mm256_unpackhi_ps(tmp0, tmp1);
}

template<> inline void InterleavedMemoryAccessBase<float_v>::deinterleave(float_v &v0, float_v &v1, float_v &v2, float_v &v3, float_v &v4, float_v &v5, float_v &v6) const
{
    deinterleave(v0, v1, v2, v3);
    const __m128  il0 = _mm_loadu_ps(&m_data[m_indexes[0] + 4]); // a0 b0 c0 d0
    const __m128  il1 = _mm_loadu_ps(&m_data[m_indexes[1] + 4]); // a1 b1 c1 d1
    const __m128  il2 = _mm_loadu_ps(&m_data[m_indexes[2] + 4]); // a2 b2 c2 d2
    const __m128  il3 = _mm_loadu_ps(&m_data[m_indexes[3] + 4]); // a3 b3 c3 d3
    const __m128  il4 = _mm_loadu_ps(&m_data[m_indexes[4] + 4]); // a4 b4 c4 d4
    const __m128  il5 = _mm_loadu_ps(&m_data[m_indexes[5] + 4]); // a5 b5 c5 d5
    const __m128  il6 = _mm_loadu_ps(&m_data[m_indexes[6] + 4]); // a6 b6 c6 d6
    const __m128  il7 = _mm_loadu_ps(&m_data[m_indexes[7] + 4]); // a7 b7 c7 d7

    const __m256 il04 = AVX::concat(il0, il4);
    const __m256 il15 = AVX::concat(il1, il5);
    const __m256 il26 = AVX::concat(il2, il6);
    const __m256 il37 = AVX::concat(il3, il7);
    const __m256 ab0246 = _mm256_unpacklo_ps(il04, il26);
    const __m256 ab1357 = _mm256_unpacklo_ps(il15, il37);
    const __m256 cd0246 = _mm256_unpackhi_ps(il04, il26);
    const __m256 cd1357 = _mm256_unpackhi_ps(il15, il37);
    v4.data() = _mm256_unpacklo_ps(ab0246, ab1357);
    v5.data() = _mm256_unpackhi_ps(ab0246, ab1357);
    v6.data() = _mm256_unpacklo_ps(cd0246, cd1357);
}

template<> inline void InterleavedMemoryAccessBase<float_v>::deinterleave(float_v &v0, float_v &v1, float_v &v2, float_v &v3, float_v &v4, float_v &v5, float_v &v6, float_v &v7) const
{
    deinterleave(v0, v1, v2, v3);
    const __m128  il0 = _mm_loadu_ps(&m_data[m_indexes[0] + 4]); // a0 b0 c0 d0
    const __m128  il1 = _mm_loadu_ps(&m_data[m_indexes[1] + 4]); // a1 b1 c1 d1
    const __m128  il2 = _mm_loadu_ps(&m_data[m_indexes[2] + 4]); // a2 b2 c2 d2
    const __m128  il3 = _mm_loadu_ps(&m_data[m_indexes[3] + 4]); // a3 b3 c3 d3
    const __m128  il4 = _mm_loadu_ps(&m_data[m_indexes[4] + 4]); // a4 b4 c4 d4
    const __m128  il5 = _mm_loadu_ps(&m_data[m_indexes[5] + 4]); // a5 b5 c5 d5
    const __m128  il6 = _mm_loadu_ps(&m_data[m_indexes[6] + 4]); // a6 b6 c6 d6
    const __m128  il7 = _mm_loadu_ps(&m_data[m_indexes[7] + 4]); // a7 b7 c7 d7

    const __m256 il04 = AVX::concat(il0, il4);
    const __m256 il15 = AVX::concat(il1, il5);
    const __m256 il26 = AVX::concat(il2, il6);
    const __m256 il37 = AVX::concat(il3, il7);
    const __m256 ab0246 = _mm256_unpacklo_ps(il04, il26);
    const __m256 ab1357 = _mm256_unpacklo_ps(il15, il37);
    const __m256 cd0246 = _mm256_unpackhi_ps(il04, il26);
    const __m256 cd1357 = _mm256_unpackhi_ps(il15, il37);
    v4.data() = _mm256_unpacklo_ps(ab0246, ab1357);
    v5.data() = _mm256_unpackhi_ps(ab0246, ab1357);
    v6.data() = _mm256_unpacklo_ps(cd0246, cd1357);
    v7.data() = _mm256_unpackhi_ps(cd0246, cd1357);
}

} // namespace Common
} // namespace Vc

#endif // VC_AVX_INTERLEAVEDMEMORY_TCC
