/*  This file is part of the Vc library. {{{

    Copyright (C) 2011-2013 Matthias Kretz <kretz@kde.org>

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

Vc_NAMESPACE_BEGIN(Vc_IMPL_NAMESPACE)

namespace internal
{

template<> Vc_ALWAYS_INLINE Vc_CONST m128 zero<m128>() { return _mm_setzero_ps(); }
template<> Vc_ALWAYS_INLINE Vc_CONST m256 zero<m256>() { return _mm256_setzero_ps(); }

template<> Vc_ALWAYS_INLINE Vc_CONST m128 allone<m128>() { return _mm_setallone_ps(); }
template<> Vc_ALWAYS_INLINE Vc_CONST m256 allone<m256>() { return _mm256_setallone_ps(); }
template<> Vc_ALWAYS_INLINE Vc_CONST m128i allone<m128i>() { return _mm_setallone_si128(); }
template<> Vc_ALWAYS_INLINE Vc_CONST m256i allone<m256i>() { return _mm256_setallone_si256(); }

// mask_cast/*{{{*/
template<size_t From, size_t To, typename R> Vc_ALWAYS_INLINE Vc_CONST R mask_cast(m128i k)
{
    static_assert(From == To, "Incorrect mask cast.");
    static_assert(std::is_same<R, m128>::value, "Incorrect mask cast.");
    return avx_cast<m128>(k);
}

template<size_t From, size_t To, typename R> Vc_ALWAYS_INLINE Vc_CONST R mask_cast(m256i k)
{
    static_assert(From == To, "Incorrect mask cast.");
    static_assert(std::is_same<R, m256>::value, "Incorrect mask cast.");
    return avx_cast<m256>(k);
}

template<> Vc_ALWAYS_INLINE Vc_CONST m256 mask_cast<4, 8, m256>(m256i k)
{
    // aabb ccdd -> abcd 0000
    return avx_cast<m256>(concat(_mm_packs_epi32(lo128(k), hi128(k)),
                                 _mm_setzero_si128()));
}

template<> Vc_ALWAYS_INLINE Vc_CONST m128 mask_cast<4, 8, m128>(m256i k)
{
    // aaaa bbbb cccc dddd -> abcd 0000
    return avx_cast<m128>(_mm_packs_epi16(_mm_packs_epi32(lo128(k), hi128(k)), _mm_setzero_si128()));
}

template<> Vc_ALWAYS_INLINE Vc_CONST m256 mask_cast<8, 4, m256>(m256i k)
{
    // aabb ccdd eeff gghh -> aaaa bbbb cccc dddd
    const auto lo = lo128(avx_cast<m256>(k));
    return concat(_mm_unpacklo_ps(lo, lo),
                  _mm_unpackhi_ps(lo, lo));
}

template<> Vc_ALWAYS_INLINE Vc_CONST m128 mask_cast<8, 8, m128>(m256i k)
{
    // aabb ccdd eeff gghh -> abcd efgh
    return avx_cast<m128>(_mm_packs_epi16(lo128(k), hi128(k)));
}

template<> Vc_ALWAYS_INLINE Vc_CONST m256 mask_cast<8, 4, m256>(m128i k)
{
    // abcd efgh -> aaaa bbbb cccc dddd
    const auto tmp = _mm_unpacklo_epi16(k, k); // aa bb cc dd
    return avx_cast<m256>(concat(_mm_unpacklo_epi32(tmp, tmp), // aaaa bbbb
                                 _mm_unpackhi_epi32(tmp, tmp))); // cccc dddd
}

template<> Vc_ALWAYS_INLINE Vc_CONST m256 mask_cast<8, 8, m256>(m128i k)
{
    return avx_cast<m256>(concat(_mm_unpacklo_epi16(k, k),
                                 _mm_unpackhi_epi16(k, k)));
}

/*}}}*/
// mask_to_int/*{{{*/
template<> Vc_ALWAYS_INLINE Vc_CONST int mask_to_int<4>(m256i k)
{
    return movemask(avx_cast<m256d>(k));
}
template<> Vc_ALWAYS_INLINE Vc_CONST int mask_to_int<8>(m256i k)
{
    return movemask(avx_cast<m256>(k));
}
template<> Vc_ALWAYS_INLINE Vc_CONST int mask_to_int<8>(m128i k)
{
    return movemask(avx_cast<m128i>(_mm_packs_epi16(k, _mm_setzero_si128())));
}
/*}}}*/
// mask_store/*{{{*/
template<size_t> Vc_ALWAYS_INLINE void mask_store(m256i k, bool *mem);
template<size_t> Vc_ALWAYS_INLINE void mask_store(m128i k, bool *mem);
template<> Vc_ALWAYS_INLINE void mask_store<8>(m256i k, bool *mem)
{
    const auto k2 = _mm_srli_epi16(_mm_packs_epi16(lo128(k), hi128(k)), 15);
    typedef uint64_t boolAlias Vc_MAY_ALIAS;
    const auto k3 = _mm_packs_epi16(k2, _mm_setzero_si128());
#ifdef __x86_64__
    *reinterpret_cast<boolAlias *>(mem) = _mm_cvtsi128_si64(k3);
#else
    *reinterpret_cast<boolAlias *>(mem) = _mm_cvtsi128_si32(k3);
    *reinterpret_cast<boolAlias *>(mem + 4) = _mm_extract_epi32(k3, 1);
#endif
}
template<> Vc_ALWAYS_INLINE void mask_store<8>(m128i k, bool *mem)
{
    k = _mm_srli_epi16(k, 15);
    typedef uint64_t boolAlias Vc_MAY_ALIAS;
    const auto k2 = _mm_packs_epi16(k, _mm_setzero_si128());
#ifdef __x86_64__
    *reinterpret_cast<boolAlias *>(mem) = _mm_cvtsi128_si64(k2);
#else
    *reinterpret_cast<boolAlias *>(mem) = _mm_cvtsi128_si32(k2);
    *reinterpret_cast<boolAlias *>(mem + 4) = _mm_extract_epi32(k2, 1);
#endif
}
/*}}}*/
// mask_load/*{{{*/
template<typename R, size_t> Vc_ALWAYS_INLINE R mask_load(const bool *mem);
template<> Vc_ALWAYS_INLINE m128 mask_load<m128, 8>(const bool *mem)
{
    m128i k = _mm_cvtsi64_si128(*reinterpret_cast<const int64_t *>(mem));
    return avx_cast<m128>(_mm_cmpgt_epi16(_mm_unpacklo_epi8(k, k), _mm_setzero_si128()));
}
template<> Vc_ALWAYS_INLINE m256 mask_load<m256, 8>(const bool *mem)
{
    m128i k = _mm_cvtsi64_si128(*reinterpret_cast<const int64_t *>(mem));
    k = _mm_cmpgt_epi16(_mm_unpacklo_epi8(k, k), _mm_setzero_si128());

    return avx_cast<m256>(concat(_mm_unpacklo_epi16(k, k), _mm_unpackhi_epi16(k, k)));
}
template<> Vc_ALWAYS_INLINE m256 mask_load<m256, 4>(const bool *mem)
{
    m128i k = avx_cast<m128i>(_mm_and_ps(_mm_set1_ps(*reinterpret_cast<const float *>(mem)),
                avx_cast<m128>(_mm_setr_epi32(0x1, 0x100, 0x10000, 0x1000000))
                ));
    k = _mm_cmpgt_epi32(k, _mm_setzero_si128());
    return avx_cast<m256>(concat(_mm_unpacklo_epi32(k, k), _mm_unpackhi_epi32(k, k)));
}
/*}}}*/

} // namespace internal

template<> Vc_ALWAYS_INLINE void Mask<double>::store(bool *mem) const
{
    typedef uint16_t boolAlias Vc_MAY_ALIAS;
    boolAlias *ptr = reinterpret_cast<boolAlias *>(mem);
    ptr[0] = _mm_movemask_epi8(lo128(dataI())) & 0x0101;
    ptr[1] = _mm_movemask_epi8(hi128(dataI())) & 0x0101;
}
template<typename T> Vc_ALWAYS_INLINE void Mask<T>::store(bool *mem) const
{
    internal::mask_store<Size>(dataI(), mem);
}
template<typename T> Vc_ALWAYS_INLINE void Mask<T>::load(const bool *mem)
{
    d.v() = internal::mask_load<VectorType, Size>(mem);
}

template<typename T> Vc_ALWAYS_INLINE Vc_PURE bool Mask<T>::operator[](size_t index) const { return toInt() & (1 << index); }
template<> Vc_ALWAYS_INLINE Vc_PURE bool Mask< int16_t>::operator[](size_t index) const { return shiftMask() & (1 << 2 * index); }
template<> Vc_ALWAYS_INLINE Vc_PURE bool Mask<uint16_t>::operator[](size_t index) const { return shiftMask() & (1 << 2 * index); }

/*
template<> Vc_ALWAYS_INLINE Mask< 4, 32> &Mask< 4, 32>::operator=(const std::array<bool, 4> &values) {
    static_assert(sizeof(bool) == 1, "Vc expects bool to have a sizeof 1 Byte");
    unsigned int x = *reinterpret_cast<const unsigned int *>(values.data());
    x *= 0xffu;
    m128i y = _mm_cvtsi32_si128(x); //  4 Bytes
    y = _mm_unpacklo_epi8(y, y);    //  8 Bytes
    y = _mm_unpacklo_epi16(y, y);   // 16 Bytes
    d.v() = avx_cast<m256>(concat(_mm_unpacklo_epi32(y, y), _mm_unpackhi_epi32(y, y)));
    return *this;
}
template<> Vc_ALWAYS_INLINE Mask< 8, 32> &Mask< 8, 32>::operator=(const std::array<bool, 8> &values) {
    static_assert(sizeof(bool) == 1, "Vc expects bool to have a sizeof 1 Byte");
    unsigned long long x = *reinterpret_cast<const unsigned long long *>(values.data());
    x *= 0xffull;
    m128i y = _mm_cvtsi64_si128(x); //  8 Bytes
    y = _mm_unpacklo_epi8(y, y);   // 16 Bytes
    d.v() = avx_cast<m256>(concat(_mm_unpacklo_epi16(y, y), _mm_unpackhi_epi16(y, y)));
    return *this;
}
template<> Vc_ALWAYS_INLINE Mask< 8, 16> &Mask< 8, 16>::operator=(const std::array<bool, 8> &values) {
    static_assert(sizeof(bool) == 1, "Vc expects bool to have a sizeof 1 Byte");
    unsigned long long x = *reinterpret_cast<const unsigned long long *>(values.data());
    x *= 0xffull;
    m128i y = _mm_cvtsi64_si128(x); //  8 Bytes
    d.v() = avx_cast<m128>(_mm_unpacklo_epi8(y, y));
    return *this;
}
template<> Vc_ALWAYS_INLINE Mask<16, 16> &Mask<16, 16>::operator=(const std::array<bool, 16> &values) {
    static_assert(sizeof(bool) == 1, "Vc expects bool to have a sizeof 1 Byte");
    m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i *>(values.data()));
    d.v() = _mm_andnot_ps(_mm_setallone_ps(), avx_cast<m128>(_mm_sub_epi8(x, _mm_set1_epi8(1))));
    return *this;
}

template<> Vc_ALWAYS_INLINE Mask< 4, 32>::operator std::array<bool, 4>() const {
    static_assert(sizeof(bool) == 1, "Vc expects bool to have a sizeof 1 Byte");
    m128i x = _mm_packs_epi32(lo128(dataI()), hi128(dataI())); // 64bit -> 32bit
    x = _mm_packs_epi32(x, x); // 32bit -> 16bit
    x = _mm_srli_epi16(x, 15);
    x = _mm_packs_epi16(x, x); // 16bit ->  8bit
    std::array<bool, 4> r;
    asm volatile("vmovd %1,%0" : "=m"(*r.data()) : "x"(x));
    return r;
}
template<> Vc_ALWAYS_INLINE Mask< 8, 32>::operator std::array<bool, 8>() const {
    static_assert(sizeof(bool) == 1, "Vc expects bool to have a sizeof 1 Byte");
    m128i x = _mm_packs_epi32(lo128(dataI()), hi128(dataI())); // 32bit -> 16bit
    x = _mm_srli_epi16(x, 15);
    x = _mm_packs_epi16(x, x); // 16bit ->  8bit
    std::array<bool, 8> r;
    asm volatile("vmovq %1,%0" : "=m"(*r.data()) : "x"(x));
    return r;
}
template<> Vc_ALWAYS_INLINE Mask< 8, 16>::operator std::array<bool, 8>() const {
    static_assert(sizeof(bool) == 1, "Vc expects bool to have a sizeof 1 Byte");
    m128i x = _mm_srli_epi16(dataI(), 15);
    x = _mm_packs_epi16(x, x); // 16bit ->  8bit
    std::array<bool, 8> r;
    asm volatile("vmovq %1,%0" : "=m"(*r.data()) : "x"(x));
    return r;
}
template<> Vc_ALWAYS_INLINE Mask<16, 16>::operator std::array<bool, 16>() const {
    static_assert(sizeof(bool) == 1, "Vc expects bool to have a sizeof 1 Byte");
    m128 x = _mm_and_ps(d.v(), avx_cast<m128>(_mm_set1_epi32(0x01010101)));
    std::array<bool, 16> r;
    asm volatile("vmovups %1,%0" : "=m"(*r.data()) : "x"(x));
    return r;
}
*/

Vc_IMPL_NAMESPACE_END

// vim: foldmethod=marker
