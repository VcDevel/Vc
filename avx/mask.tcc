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

namespace Vc
{
namespace AVX
{

template<> inline Mask<4, 32>::Mask(Mask<8, 32> m)
    : k(concat(_mm_unpacklo_ps(lo128(m.data()), lo128(m.data())),
                _mm_unpackhi_ps(lo128(m.data()), lo128(m.data()))))
{
}

template<> inline Mask<8, 32>::Mask(Mask<4, 32> m)
    // aabb ccdd -> abcd 0000
    : k(concat(Mem::shuffle<X0, X2, Y0, Y2>(lo128(m.data()), hi128(m.data())),
                _mm_setzero_ps()))
{
}

template<unsigned int Size> inline int Mask<Size, 32u>::shiftMask() const
{
    return _mm256_movemask_epi8(dataI());
}
template<unsigned int Size> inline int Mask<Size, 16u>::shiftMask() const
{
    return _mm_movemask_epi8(dataI());
}

template<> inline int Mask< 4, 32>::toInt() const { return _mm256_movemask_pd(dataD()); }
template<> inline int Mask< 8, 32>::toInt() const { return _mm256_movemask_ps(data ()); }
template<> inline int Mask< 8, 16>::toInt() const { return _mm_movemask_epi8(_mm_packs_epi16(dataI(), _mm_setzero_si128())); }
template<> inline int Mask<16, 16>::toInt() const { return _mm_movemask_epi8(dataI()); }

template<> inline bool Mask< 4, 32>::operator[](int index) const { return toInt() & (1 << index); }
template<> inline bool Mask< 8, 32>::operator[](int index) const { return toInt() & (1 << index); }
template<> inline bool Mask< 8, 16>::operator[](int index) const { return shiftMask() & (1 << 2 * index); }
template<> inline bool Mask<16, 16>::operator[](int index) const { return toInt() & (1 << index); }

template<unsigned int Size> inline int Mask<Size, 32u>::count() const { return _mm_popcnt_u32(toInt()); }
template<unsigned int Size> inline int Mask<Size, 16u>::count() const { return _mm_popcnt_u32(toInt()); }
template<unsigned int Size> inline int Mask<Size, 32u>::firstOne() const { return _bit_scan_forward(toInt()); }
template<unsigned int Size> inline int Mask<Size, 16u>::firstOne() const { return _bit_scan_forward(toInt()); }

} // namespace AVX
} // namespace Vc
