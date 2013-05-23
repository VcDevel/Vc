/*  This file is part of the Vc library.

    Copyright (C) 2011-2012 Matthias Kretz <kretz@kde.org>

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

Vc_NAMESPACE_BEGIN(Vc_IMPL_NAMESPACE)

template<> Vc_ALWAYS_INLINE Mask<4, 32>::Mask(const Mask<8, 32> &m)
    : k(concat(_mm_unpacklo_ps(lo128(m.data()), lo128(m.data())),
                _mm_unpackhi_ps(lo128(m.data()), lo128(m.data()))))
{
}

template<> Vc_ALWAYS_INLINE Mask<8, 32>::Mask(const Mask<4, 32> &m)
    // aabb ccdd -> abcd 0000
    : k(concat(Mem::shuffle<X0, X2, Y0, Y2>(lo128(m.data()), hi128(m.data())),
                _mm_setzero_ps()))
{
}

template<unsigned int Size> Vc_ALWAYS_INLINE Vc_PURE int Mask<Size, 32u>::shiftMask() const
{
    return _mm256_movemask_epi8(dataI());
}
template<unsigned int Size> Vc_ALWAYS_INLINE Vc_PURE int Mask<Size, 16u>::shiftMask() const
{
    return _mm_movemask_epi8(dataI());
}

template<> Vc_ALWAYS_INLINE Vc_PURE int Mask< 4, 32>::toInt() const { return _mm256_movemask_pd(dataD()); }
template<> Vc_ALWAYS_INLINE Vc_PURE int Mask< 8, 32>::toInt() const { return _mm256_movemask_ps(data ()); }
template<> Vc_ALWAYS_INLINE Vc_PURE int Mask< 8, 16>::toInt() const { return _mm_movemask_epi8(_mm_packs_epi16(dataI(), _mm_setzero_si128())); }
template<> Vc_ALWAYS_INLINE Vc_PURE int Mask<16, 16>::toInt() const { return _mm_movemask_epi8(dataI()); }

template<> Vc_ALWAYS_INLINE Vc_PURE bool Mask< 4, 32>::operator[](size_t index) const { return toInt() & (1 << index); }
template<> Vc_ALWAYS_INLINE Vc_PURE bool Mask< 8, 32>::operator[](size_t index) const { return toInt() & (1 << index); }
template<> Vc_ALWAYS_INLINE Vc_PURE bool Mask< 8, 16>::operator[](size_t index) const { return shiftMask() & (1 << 2 * index); }
template<> Vc_ALWAYS_INLINE Vc_PURE bool Mask<16, 16>::operator[](size_t index) const { return toInt() & (1 << index); }

template<> Vc_ALWAYS_INLINE void Mask< 4, 32>::setEntry(size_t index, bool value) {
    Common::VectorMemoryUnion<__m256i, unsigned long long> mask(dataI());
    mask.m(index) = value ? 0xffffffffffffffffull : 0ull;
    k = avx_cast<m256>(mask.v());
}
template<> Vc_ALWAYS_INLINE void Mask< 8, 32>::setEntry(size_t index, bool value) {
    Common::VectorMemoryUnion<__m256i, unsigned int> mask(dataI());
    mask.m(index) = value ? 0xffffffffu : 0u;
    k = avx_cast<m256>(mask.v());
}
template<> Vc_ALWAYS_INLINE void Mask< 8, 16>::setEntry(size_t index, bool value) {
    Common::VectorMemoryUnion<__m128i, unsigned short> mask(dataI());
    mask.m(index) = value ? 0xffffu : 0u;
    k = avx_cast<m128>(mask.v());
}
template<> Vc_ALWAYS_INLINE void Mask<16, 16>::setEntry(size_t index, bool value) {
    Common::VectorMemoryUnion<__m128i, unsigned short> mask(dataI());
    mask.m(index) = value ? 0xffffu: 0u;
    k = avx_cast<m128>(mask.v());
}

template<> Vc_ALWAYS_INLINE Mask< 4, 32> &Mask< 4, 32>::operator=(const std::array<bool, 4> &values) {
    static_assert(sizeof(bool) == 1, "Vc expects bool to have a sizeof 1 Byte");
    unsigned int x = *reinterpret_cast<const unsigned int *>(values.data());
    x *= 0xffu;
    m128i y = _mm_cvtsi32_si128(x); //  4 Bytes
    y = _mm_unpacklo_epi8(y, y);    //  8 Bytes
    y = _mm_unpacklo_epi16(y, y);   // 16 Bytes
    k = avx_cast<decltype(k)>(concat(_mm_unpacklo_epi32(y, y), _mm_unpackhi_epi32(y, y)));
    return *this;
}
template<> Vc_ALWAYS_INLINE Mask< 8, 32> &Mask< 8, 32>::operator=(const std::array<bool, 8> &values) {
    static_assert(sizeof(bool) == 1, "Vc expects bool to have a sizeof 1 Byte");
    unsigned long long x = *reinterpret_cast<const unsigned long long *>(values.data());
    x *= 0xffull;
    m128i y = _mm_cvtsi64_si128(x); //  8 Bytes
    y = _mm_unpacklo_epi8(y, y);   // 16 Bytes
    k = avx_cast<decltype(k)>(concat(_mm_unpacklo_epi16(y, y), _mm_unpackhi_epi16(y, y)));
    return *this;
}
template<> Vc_ALWAYS_INLINE Mask< 8, 16> &Mask< 8, 16>::operator=(const std::array<bool, 8> &values) {
    static_assert(sizeof(bool) == 1, "Vc expects bool to have a sizeof 1 Byte");
    unsigned long long x = *reinterpret_cast<const unsigned long long *>(values.data());
    x *= 0xffull;
    m128i y = _mm_cvtsi64_si128(x); //  8 Bytes
    k = avx_cast<decltype(k)>(_mm_unpacklo_epi8(y, y));
    return *this;
}
template<> Vc_ALWAYS_INLINE Mask<16, 16> &Mask<16, 16>::operator=(const std::array<bool, 16> &values) {
    static_assert(sizeof(bool) == 1, "Vc expects bool to have a sizeof 1 Byte");
    m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i *>(values.data()));
    k = _mm_andnot_ps(_mm_setallone_ps(), avx_cast<m128>(_mm_sub_epi8(x, _mm_set1_epi8(1))));
    return *this;
}

template<> Mask< 4, 32>::operator std::array<bool, 4> &&() const {
    static_assert(sizeof(bool) == 1, "Vc expects bool to have a sizeof 1 Byte");
    m128i x = _mm_packs_epi32(lo128(dataI()), hi128(dataI())); // 64bit -> 32bit
    x = _mm_packs_epi32(x, x); // 32bit -> 16bit
    x = _mm_srli_epi16(x, 15);
    x = _mm_packs_epi16(x, x); // 16bit ->  8bit
    std::array<bool, 4> r;
    asm volatile("vmovd %1,%0" : "=m"(*r.data()) : "x"(x));
    return std::move(r);
}
template<> Mask< 8, 32>::operator std::array<bool, 8> &&() const {
    static_assert(sizeof(bool) == 1, "Vc expects bool to have a sizeof 1 Byte");
    m128i x = _mm_packs_epi32(lo128(dataI()), hi128(dataI())); // 32bit -> 16bit
    x = _mm_srli_epi16(x, 15);
    x = _mm_packs_epi16(x, x); // 16bit ->  8bit
    std::array<bool, 8> r;
    asm volatile("vmovq %1,%0" : "=m"(*r.data()) : "x"(x));
    return std::move(r);
}
template<> Mask< 8, 16>::operator std::array<bool, 8> &&() const {
    static_assert(sizeof(bool) == 1, "Vc expects bool to have a sizeof 1 Byte");
    m128i x = _mm_srli_epi16(dataI(), 15);
    x = _mm_packs_epi16(x, x); // 16bit ->  8bit
    std::array<bool, 8> r;
    asm volatile("vmovq %1,%0" : "=m"(*r.data()) : "x"(x));
    return std::move(r);
}
template<> Mask<16, 16>::operator std::array<bool, 16> &&() const {
    static_assert(sizeof(bool) == 1, "Vc expects bool to have a sizeof 1 Byte");
    m128 x = _mm_and_ps(k, avx_cast<m128>(_mm_set1_epi32(0x01010101)));
    std::array<bool, 16> r;
    asm volatile("vmovups %1,%0" : "=m"(*r.data()) : "x"(x));
    return std::move(r);
}

#ifndef VC_IMPL_POPCNT
static Vc_ALWAYS_INLINE Vc_CONST unsigned int _mm_popcnt_u32(unsigned int n) {
    n = (n & 0x55555555U) + ((n >> 1) & 0x55555555U);
    n = (n & 0x33333333U) + ((n >> 2) & 0x33333333U);
    n = (n & 0x0f0f0f0fU) + ((n >> 4) & 0x0f0f0f0fU);
    //n = (n & 0x00ff00ffU) + ((n >> 8) & 0x00ff00ffU);
    //n = (n & 0x0000ffffU) + ((n >>16) & 0x0000ffffU);
    return n;
}
#endif
template<unsigned int Size> Vc_ALWAYS_INLINE Vc_PURE unsigned int Mask<Size, 32u>::count() const { return _mm_popcnt_u32(toInt()); }
template<unsigned int Size> Vc_ALWAYS_INLINE Vc_PURE unsigned int Mask<Size, 16u>::count() const { return _mm_popcnt_u32(toInt()); }
template<unsigned int Size> Vc_ALWAYS_INLINE Vc_PURE unsigned int Mask<Size, 32u>::firstOne() const { return _bit_scan_forward(toInt()); }
template<unsigned int Size> Vc_ALWAYS_INLINE Vc_PURE unsigned int Mask<Size, 16u>::firstOne() const { return _bit_scan_forward(toInt()); }

Vc_IMPL_NAMESPACE_END
