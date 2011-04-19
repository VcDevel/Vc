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

template<unsigned int Size> template<typename VectorType, typename MemoryType>
inline void GSHelper<Size>::gather(VectorType &v, const MemoryType *m, _M512I i)
{
    v = lrb_cast<VectorType>(_mm512_gatherd(i, const_cast<MemoryType *>(m), _MM_FULLUPC_NONE, _MM_SCALE_4, _MM_HINT_NONE));
}

template<> template<> inline void GSHelper<16>::gather(_M512 &v, const short *m, _M512I i)
{
    v = _mm512_gatherd(i, const_cast<short *>(m), _MM_FULLUPC_SINT16, _MM_SCALE_2, _MM_HINT_NONE);
}
template<> template<> inline void GSHelper<16>::gather(_M512 &v, const unsigned short *m, _M512I i)
{
    v = _mm512_gatherd(i, const_cast<unsigned short *>(m), _MM_FULLUPC_UINT16, _MM_SCALE_2, _MM_HINT_NONE);
}

template<> template<> inline void GSHelper<16>::gather(_M512I &v, const short *m, _M512I i)
{
    v = _mm512_castps_si512(_mm512_gatherd(i, const_cast<short *>(m), _MM_FULLUPC_SINT16I, _MM_SCALE_2, _MM_HINT_NONE));
}
template<> template<> inline void GSHelper<16>::gather(_M512I &v, const unsigned short *m, _M512I i)
{
    v = _mm512_castps_si512(_mm512_gatherd(i, const_cast<unsigned short *>(m), _MM_FULLUPC_UINT16I, _MM_SCALE_2, _MM_HINT_NONE));
}

template<> template<> inline void GSHelper<8>::gather(_M512D &v, const double *m, _M512I i)
{
    i = _mm512_sll_pi(i, _mm512_set_1to16_pi(1));
    i = _mm512_castps_si512(_mm512_mask_movq(
                _mm512_shuf128x32(_mm512_castsi512_ps(i), _MM_PERM_BBAA, _MM_PERM_DDCC),
                0x33,
                _mm512_shuf128x32(_mm512_castsi512_ps(i), _MM_PERM_BBAA, _MM_PERM_BBAA)
                ));
    i = _mm512_add_pi(i, _mm512_set_4to16_pi(0, 1, 0, 1));
    v = _mm512_castps_pd(_mm512_gatherd(i, const_cast<double *>(m), _MM_FULLUPC_NONE, _MM_SCALE_4, _MM_HINT_NONE));
}

} // namespace LRBni
} // namespace Vc
