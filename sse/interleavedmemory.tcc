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

#ifndef SSE_INTERLEAVEDMEMORY_TCC
#define SSE_INTERLEAVEDMEMORY_TCC

namespace Vc
{
namespace Common
{

template<> void InterleavedMemoryAccess<4, float_v>::deinterleave(float_v &v0, float_v &v1) const
{
    const __m128 a = _mm_load_ps(&m_data[4 * m_indexes[0]]);
    const __m128 b = _mm_load_ps(&m_data[4 * m_indexes[1]]);
    const __m128 c = _mm_load_ps(&m_data[4 * m_indexes[2]]);
    const __m128 d = _mm_load_ps(&m_data[4 * m_indexes[3]]);

    const __m128 tmp0 = _mm_unpacklo_ps(a, b); // [a0 a1 b0 b1]
    const __m128 tmp1 = _mm_unpacklo_ps(c, d); // [a2 a3 b2 b3]

    v0.data() = _mm_movelh_ps(tmp0, tmp1);
    v1.data() = _mm_movehl_ps(tmp1, tmp0);
}

template<> void InterleavedMemoryAccess<4, float_v>::deinterleave(float_v &v0, float_v &v1, float_v &v2) const
{
    const __m128 a = _mm_load_ps(&m_data[4 * m_indexes[0]]);
    const __m128 b = _mm_load_ps(&m_data[4 * m_indexes[1]]);
    const __m128 c = _mm_load_ps(&m_data[4 * m_indexes[2]]);
    const __m128 d = _mm_load_ps(&m_data[4 * m_indexes[3]]);

    const __m128 tmp0 = _mm_unpacklo_ps(a, b); // [a0 a1 b0 b1]
    const __m128 tmp1 = _mm_unpacklo_ps(c, d); // [a2 a3 b2 b3]
    const __m128 tmp2 = _mm_unpackhi_ps(a, b); // [c0 c1 d0 d1]
    const __m128 tmp3 = _mm_unpackhi_ps(c, d); // [c2 c3 d2 d3]

    v0.data() = _mm_movelh_ps(tmp0, tmp1);
    v1.data() = _mm_movehl_ps(tmp1, tmp0);
    v2.data() = _mm_movelh_ps(tmp2, tmp3);
}

template<> void InterleavedMemoryAccess<4, float_v>::deinterleave(float_v &v0, float_v &v1, float_v &v2, float_v &v3) const
{
    const __m128 a = _mm_load_ps(&m_data[4 * m_indexes[0]]);
    const __m128 b = _mm_load_ps(&m_data[4 * m_indexes[1]]);
    const __m128 c = _mm_load_ps(&m_data[4 * m_indexes[2]]);
    const __m128 d = _mm_load_ps(&m_data[4 * m_indexes[3]]);

    const __m128 tmp0 = _mm_unpacklo_ps(a, b); // [a0 a1 b0 b1]
    const __m128 tmp1 = _mm_unpacklo_ps(c, d); // [a2 a3 b2 b3]
    const __m128 tmp2 = _mm_unpackhi_ps(a, b); // [c0 c1 d0 d1]
    const __m128 tmp3 = _mm_unpackhi_ps(c, d); // [c2 c3 d2 d3]

    v0.data() = _mm_movelh_ps(tmp0, tmp1);
    v1.data() = _mm_movehl_ps(tmp1, tmp0);
    v2.data() = _mm_movelh_ps(tmp2, tmp3);
    v3.data() = _mm_movehl_ps(tmp3, tmp2);
}

} // namespace Common
} // namespace Vc

#endif // SSE_INTERLEAVEDMEMORY_TCC
