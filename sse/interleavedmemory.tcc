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

#ifndef VC_SSE_INTERLEAVEDMEMORY_TCC
#define VC_SSE_INTERLEAVEDMEMORY_TCC

namespace Vc
{
namespace Common
{

template<> inline void InterleavedMemoryAccessBase<float_v>::deinterleave(float_v &v0, float_v &v1) const
{
    const __m128 a = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 const *>(&m_data[m_indexes[0]]));
    const __m128 b = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 const *>(&m_data[m_indexes[1]]));
    const __m128 c = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 const *>(&m_data[m_indexes[2]]));
    const __m128 d = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 const *>(&m_data[m_indexes[3]]));

    const __m128 tmp0 = _mm_unpacklo_ps(a, b); // [a0 a1 b0 b1]
    const __m128 tmp1 = _mm_unpacklo_ps(c, d); // [a2 a3 b2 b3]

    v0.data() = _mm_movelh_ps(tmp0, tmp1);
    v1.data() = _mm_movehl_ps(tmp1, tmp0);
}

template<> inline void InterleavedMemoryAccessBase<float_v>::deinterleave(float_v &v0, float_v &v1, float_v &v2) const
{
    const __m128 a = _mm_loadu_ps(&m_data[m_indexes[0]]);
    const __m128 b = _mm_loadu_ps(&m_data[m_indexes[1]]);
    const __m128 c = _mm_loadu_ps(&m_data[m_indexes[2]]);
    const __m128 d = _mm_loadu_ps(&m_data[m_indexes[3]]);

    const __m128 tmp0 = _mm_unpacklo_ps(a, b); // [a0 a1 b0 b1]
    const __m128 tmp1 = _mm_unpacklo_ps(c, d); // [a2 a3 b2 b3]
    const __m128 tmp2 = _mm_unpackhi_ps(a, b); // [c0 c1 XX XX]
    const __m128 tmp3 = _mm_unpackhi_ps(c, d); // [c2 c3 XX XX]

    v0.data() = _mm_movelh_ps(tmp0, tmp1);
    v1.data() = _mm_movehl_ps(tmp1, tmp0);
    v2.data() = _mm_movelh_ps(tmp2, tmp3);
}

template<> inline void InterleavedMemoryAccessBase<float_v>::deinterleave(float_v &v0, float_v &v1, float_v &v2, float_v &v3) const
{
    const __m128 a = _mm_loadu_ps(&m_data[m_indexes[0]]);
    const __m128 b = _mm_loadu_ps(&m_data[m_indexes[1]]);
    const __m128 c = _mm_loadu_ps(&m_data[m_indexes[2]]);
    const __m128 d = _mm_loadu_ps(&m_data[m_indexes[3]]);

    const __m128 tmp0 = _mm_unpacklo_ps(a, b); // [a0 a1 b0 b1]
    const __m128 tmp1 = _mm_unpacklo_ps(c, d); // [a2 a3 b2 b3]
    const __m128 tmp2 = _mm_unpackhi_ps(a, b); // [c0 c1 d0 d1]
    const __m128 tmp3 = _mm_unpackhi_ps(c, d); // [c2 c3 d2 d3]

    v0.data() = _mm_movelh_ps(tmp0, tmp1);
    v1.data() = _mm_movehl_ps(tmp1, tmp0);
    v2.data() = _mm_movelh_ps(tmp2, tmp3);
    v3.data() = _mm_movehl_ps(tmp3, tmp2);
}

template<> inline void InterleavedMemoryAccessBase<float_v>::deinterleave(float_v &v0, float_v &v1, float_v &v2, float_v &v3, float_v &v4) const
{
    v4.gather(m_data, m_indexes + I(4));
    deinterleave(v0, v1, v2, v3);
}

template<> inline void InterleavedMemoryAccessBase<float_v>::deinterleave(float_v &v0, float_v &v1, float_v &v2, float_v &v3, float_v &v4, float_v &v5) const
{
    const __m128 a = _mm_loadu_ps(&m_data[m_indexes[0]]);
    const __m128 e = _mm_loadu_ps(&m_data[4 + m_indexes[0]]);
    const __m128 b = _mm_loadu_ps(&m_data[m_indexes[1]]);
    const __m128 f = _mm_loadu_ps(&m_data[4 + m_indexes[1]]);

    const __m128 tmp0 = _mm_unpacklo_ps(a, b); // [a0 a1 b0 b1]
    const __m128 tmp2 = _mm_unpackhi_ps(a, b); // [c0 c1 d0 d1]
    const __m128 tmp4 = _mm_unpacklo_ps(e, f); // [a0 a1 b0 b1]

    const __m128 c = _mm_loadu_ps(&m_data[m_indexes[2]]);
    const __m128 g = _mm_loadu_ps(&m_data[4 + m_indexes[2]]);
    const __m128 d = _mm_loadu_ps(&m_data[m_indexes[3]]);
    const __m128 h = _mm_loadu_ps(&m_data[4 + m_indexes[3]]);

    const __m128 tmp1 = _mm_unpacklo_ps(c, d); // [a2 a3 b2 b3]
    v0.data() = _mm_movelh_ps(tmp0, tmp1);
    v1.data() = _mm_movehl_ps(tmp1, tmp0);

    const __m128 tmp3 = _mm_unpackhi_ps(c, d); // [c2 c3 d2 d3]
    v2.data() = _mm_movelh_ps(tmp2, tmp3);
    v3.data() = _mm_movehl_ps(tmp3, tmp2);

    const __m128 tmp5 = _mm_unpacklo_ps(g, h); // [a2 a3 b2 b3]
    v4.data() = _mm_movelh_ps(tmp4, tmp5);
    v5.data() = _mm_movehl_ps(tmp5, tmp4);
}

template<> inline void InterleavedMemoryAccessBase<float_v>::deinterleave(float_v &v0, float_v &v1, float_v &v2, float_v &v3, float_v &v4, float_v &v5, float_v &v6) const
{
    const __m128 a = _mm_loadu_ps(&m_data[m_indexes[0]]);
    const __m128 e = _mm_loadu_ps(&m_data[4 + m_indexes[0]]);
    const __m128 b = _mm_loadu_ps(&m_data[m_indexes[1]]);
    const __m128 f = _mm_loadu_ps(&m_data[4 + m_indexes[1]]);

    const __m128 tmp0 = _mm_unpacklo_ps(a, b); // [a0 a1 b0 b1]
    const __m128 tmp2 = _mm_unpackhi_ps(a, b); // [c0 c1 d0 d1]
    const __m128 tmp4 = _mm_unpacklo_ps(e, f); // [a0 a1 b0 b1]
    const __m128 tmp6 = _mm_unpackhi_ps(e, f); // [c0 c1 d0 d1]

    const __m128 c = _mm_loadu_ps(&m_data[m_indexes[2]]);
    const __m128 g = _mm_loadu_ps(&m_data[4 + m_indexes[2]]);
    const __m128 d = _mm_loadu_ps(&m_data[m_indexes[3]]);
    const __m128 h = _mm_loadu_ps(&m_data[4 + m_indexes[3]]);

    const __m128 tmp1 = _mm_unpacklo_ps(c, d); // [a2 a3 b2 b3]
    v0.data() = _mm_movelh_ps(tmp0, tmp1);
    v1.data() = _mm_movehl_ps(tmp1, tmp0);

    const __m128 tmp3 = _mm_unpackhi_ps(c, d); // [c2 c3 d2 d3]
    v2.data() = _mm_movelh_ps(tmp2, tmp3);
    v3.data() = _mm_movehl_ps(tmp3, tmp2);

    const __m128 tmp5 = _mm_unpacklo_ps(g, h); // [a2 a3 b2 b3]
    v4.data() = _mm_movelh_ps(tmp4, tmp5);
    v5.data() = _mm_movehl_ps(tmp5, tmp4);

    const __m128 tmp7 = _mm_unpackhi_ps(g, h); // [c2 c3 d2 d3]
    v6.data() = _mm_movelh_ps(tmp6, tmp7);
}

template<> inline void InterleavedMemoryAccessBase<float_v>::deinterleave(float_v &v0, float_v &v1, float_v &v2, float_v &v3, float_v &v4, float_v &v5, float_v &v6, float_v &v7) const
{
    const __m128 a = _mm_loadu_ps(&m_data[m_indexes[0]]);
    const __m128 e = _mm_loadu_ps(&m_data[4 + m_indexes[0]]);
    const __m128 b = _mm_loadu_ps(&m_data[m_indexes[1]]);
    const __m128 f = _mm_loadu_ps(&m_data[4 + m_indexes[1]]);

    const __m128 tmp0 = _mm_unpacklo_ps(a, b); // [a0 a1 b0 b1]
    const __m128 tmp2 = _mm_unpackhi_ps(a, b); // [c0 c1 d0 d1]
    const __m128 tmp4 = _mm_unpacklo_ps(e, f); // [a0 a1 b0 b1]
    const __m128 tmp6 = _mm_unpackhi_ps(e, f); // [c0 c1 d0 d1]

    const __m128 c = _mm_loadu_ps(&m_data[m_indexes[2]]);
    const __m128 g = _mm_loadu_ps(&m_data[4 + m_indexes[2]]);
    const __m128 d = _mm_loadu_ps(&m_data[m_indexes[3]]);
    const __m128 h = _mm_loadu_ps(&m_data[4 + m_indexes[3]]);

    const __m128 tmp1 = _mm_unpacklo_ps(c, d); // [a2 a3 b2 b3]
    v0.data() = _mm_movelh_ps(tmp0, tmp1);
    v1.data() = _mm_movehl_ps(tmp1, tmp0);

    const __m128 tmp3 = _mm_unpackhi_ps(c, d); // [c2 c3 d2 d3]
    v2.data() = _mm_movelh_ps(tmp2, tmp3);
    v3.data() = _mm_movehl_ps(tmp3, tmp2);

    const __m128 tmp5 = _mm_unpacklo_ps(g, h); // [a2 a3 b2 b3]
    v4.data() = _mm_movelh_ps(tmp4, tmp5);
    v5.data() = _mm_movehl_ps(tmp5, tmp4);

    const __m128 tmp7 = _mm_unpackhi_ps(g, h); // [c2 c3 d2 d3]
    v6.data() = _mm_movelh_ps(tmp6, tmp7);
    v7.data() = _mm_movehl_ps(tmp7, tmp6);
}

} // namespace Common
} // namespace Vc

#endif // VC_SSE_INTERLEAVEDMEMORY_TCC
