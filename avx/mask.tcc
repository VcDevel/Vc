/*  This file is part of the Vc library. {{{
Copyright © 2011-2014 Matthias Kretz <kretz@kde.org>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the names of contributing organizations nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

}}}*/

namespace Vc_VERSIONED_NAMESPACE
{
namespace Vc_IMPL_NAMESPACE
{

namespace internal
{

template<> Vc_ALWAYS_INLINE Vc_CONST m128  zero<m128 >() { return _mm_setzero_ps(); }
template<> Vc_ALWAYS_INLINE Vc_CONST m128i zero<m128i>() { return _mm_setzero_si128(); }
template<> Vc_ALWAYS_INLINE Vc_CONST m128d zero<m128d>() { return _mm_setzero_pd(); }
template<> Vc_ALWAYS_INLINE Vc_CONST m256  zero<m256 >() { return _mm256_setzero_ps(); }
template<> Vc_ALWAYS_INLINE Vc_CONST m256i zero<m256i>() { return _mm256_setzero_si256(); }
template<> Vc_ALWAYS_INLINE Vc_CONST m256d zero<m256d>() { return _mm256_setzero_pd(); }

template<> Vc_ALWAYS_INLINE Vc_CONST m128  allone<m128 >() { return _mm_setallone_ps(); }
template<> Vc_ALWAYS_INLINE Vc_CONST m128i allone<m128i>() { return _mm_setallone_si128(); }
template<> Vc_ALWAYS_INLINE Vc_CONST m128d allone<m128d>() { return _mm_setallone_pd(); }
template<> Vc_ALWAYS_INLINE Vc_CONST m256  allone<m256 >() { return setallone_ps(); }
template<> Vc_ALWAYS_INLINE Vc_CONST m256i allone<m256i>() { return setallone_si256(); }
template<> Vc_ALWAYS_INLINE Vc_CONST m256d allone<m256d>() { return setallone_pd(); }

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

// store {{{1
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
// load {{{1
template<typename T> Vc_ALWAYS_INLINE void Mask<T>::load(const bool *mem)
{
    d.v() = avx_cast<VectorType>(internal::mask_load<VectorTypeF, Size>(mem));
}
// operator[] {{{1
template<typename T> Vc_ALWAYS_INLINE Vc_PURE bool Mask<T>::operator[](size_t index) const { return toInt() & (1 << index); }
template<> Vc_ALWAYS_INLINE Vc_PURE bool Mask< int16_t>::operator[](size_t index) const { return shiftMask() & (1 << 2 * index); }
template<> Vc_ALWAYS_INLINE Vc_PURE bool Mask<uint16_t>::operator[](size_t index) const { return shiftMask() & (1 << 2 * index); }
// operator== {{{1
template <> Vc_INTRINSIC Vc_PURE bool double_m::operator==(const double_m &rhs) const
{ return internal::movemask(dataD()) == internal::movemask(rhs.dataD()); }
template <> Vc_INTRINSIC Vc_PURE bool short_m::operator==(const short_m &rhs) const
{ return internal::movemask(dataI()) == internal::movemask(rhs.dataI()); }
template <> Vc_INTRINSIC Vc_PURE bool ushort_m::operator==(const ushort_m &rhs) const
{ return internal::movemask(dataI()) == internal::movemask(rhs.dataI()); }
// generate {{{1
template <typename M, typename G>
Vc_INTRINSIC M generate_impl(G &&gen, std::integral_constant<int, 4 + 32>)
{
    return _mm256_setr_epi64x(
        gen(0) ? 0xffffffffffffffffull : 0, gen(1) ? 0xffffffffffffffffull : 0,
        gen(2) ? 0xffffffffffffffffull : 0, gen(3) ? 0xffffffffffffffffull : 0);
}
template <typename M, typename G>
Vc_INTRINSIC M generate_impl(G &&gen, std::integral_constant<int, 8 + 32>)
{
    return _mm256_setr_epi32(gen(0) ? 0xfffffffful : 0, gen(1) ? 0xfffffffful : 0,
                             gen(2) ? 0xfffffffful : 0, gen(3) ? 0xfffffffful : 0,
                             gen(4) ? 0xfffffffful : 0, gen(5) ? 0xfffffffful : 0,
                             gen(6) ? 0xfffffffful : 0, gen(7) ? 0xfffffffful : 0);
}
template <typename M, typename G>
Vc_INTRINSIC M generate_impl(G &&gen, std::integral_constant<int, 8 + 16>)
{
    return _mm_setr_epi16(gen(0) ? 0xffffu : 0, gen(1) ? 0xffffu : 0,
                          gen(2) ? 0xffffu : 0, gen(3) ? 0xffffu : 0,
                          gen(4) ? 0xffffu : 0, gen(5) ? 0xffffu : 0,
                          gen(6) ? 0xffffu : 0, gen(7) ? 0xffffu : 0);
}
template <typename T>
template <typename G>
Vc_INTRINSIC Mask<T> Mask<T>::generate(G &&gen)
{
    return generate_impl<Mask<T>>(std::forward<G>(gen),
                                  std::integral_constant<int, Size + sizeof(Storage)>());
}
// shifted {{{1
template <int amount, typename T>
Vc_INTRINSIC Vc_PURE enable_if<(sizeof(T) == 32 && amount >= 16), T> shifted_impl(T k)
{
    return zeroExtend(_mm_srli_si128(hi128(k), amount - 16));
}
template <int amount, typename T>
Vc_INTRINSIC Vc_PURE enable_if<(sizeof(T) == 32 && amount > 0 && amount < 16), T>
    shifted_impl(T k)
{
    return alignr<amount>(Mem::permute128<X1, Const0>(k), Mem::permute128<X0, X1>(k));
}
template <int amount, typename T>
Vc_INTRINSIC Vc_PURE enable_if<(sizeof(T) == 32 && amount <= -16), T> shifted_impl(T k)
{
    return Mem::permute128<Const0, X0>(avx_cast<m256i>(_mm_slli_si128(lo128(k), -16 - amount)));
}
template <int amount, typename T>
Vc_INTRINSIC Vc_PURE enable_if<(sizeof(T) == 32 && amount > -16 && amount < 0), T>
    shifted_impl(T k)
{
    return alignr<16 + amount>(k, Mem::permute128<Const0, X0>(k));
}
template <int amount, typename T>
Vc_INTRINSIC Vc_PURE enable_if<(sizeof(T) == 16 && amount > 0), T> shifted_impl(T k)
{
    return _mm_srli_si128(k, amount);
}
template <int amount, typename T>
Vc_INTRINSIC Vc_PURE enable_if<(sizeof(T) == 16 && amount < 0), T> shifted_impl(T k)
{
    return _mm_slli_si128(k, -amount);
}
template <typename T> Vc_INTRINSIC Vc_PURE Mask<T> Mask<T>::shifted(int amount) const
{
    switch (amount * int(sizeof(VectorEntryType))) {
    case   0: return *this;
    case   1: return shifted_impl<  1>(dataI());
    case   2: return shifted_impl<  2>(dataI());
    case   3: return shifted_impl<  3>(dataI());
    case   4: return shifted_impl<  4>(dataI());
    case   5: return shifted_impl<  5>(dataI());
    case   6: return shifted_impl<  6>(dataI());
    case   7: return shifted_impl<  7>(dataI());
    case   8: return shifted_impl<  8>(dataI());
    case   9: return shifted_impl<  9>(dataI());
    case  10: return shifted_impl< 10>(dataI());
    case  11: return shifted_impl< 11>(dataI());
    case  12: return shifted_impl< 12>(dataI());
    case  13: return shifted_impl< 13>(dataI());
    case  14: return shifted_impl< 14>(dataI());
    case  15: return shifted_impl< 15>(dataI());
    case  16: return shifted_impl< 16>(dataI());
    case  17: return shifted_impl< 17>(dataI());
    case  18: return shifted_impl< 18>(dataI());
    case  19: return shifted_impl< 19>(dataI());
    case  20: return shifted_impl< 20>(dataI());
    case  21: return shifted_impl< 21>(dataI());
    case  22: return shifted_impl< 22>(dataI());
    case  23: return shifted_impl< 23>(dataI());
    case  24: return shifted_impl< 24>(dataI());
    case  25: return shifted_impl< 25>(dataI());
    case  26: return shifted_impl< 26>(dataI());
    case  27: return shifted_impl< 27>(dataI());
    case  28: return shifted_impl< 28>(dataI());
    case  29: return shifted_impl< 29>(dataI());
    case  30: return shifted_impl< 30>(dataI());
    case  31: return shifted_impl< 31>(dataI());
    case  -1: return shifted_impl< -1>(dataI());
    case  -2: return shifted_impl< -2>(dataI());
    case  -3: return shifted_impl< -3>(dataI());
    case  -4: return shifted_impl< -4>(dataI());
    case  -5: return shifted_impl< -5>(dataI());
    case  -6: return shifted_impl< -6>(dataI());
    case  -7: return shifted_impl< -7>(dataI());
    case  -8: return shifted_impl< -8>(dataI());
    case  -9: return shifted_impl< -9>(dataI());
    case -10: return shifted_impl<-10>(dataI());
    case -11: return shifted_impl<-11>(dataI());
    case -12: return shifted_impl<-12>(dataI());
    case -13: return shifted_impl<-13>(dataI());
    case -14: return shifted_impl<-14>(dataI());
    case -15: return shifted_impl<-15>(dataI());
    case -16: return shifted_impl<-16>(dataI());
    case -17: return shifted_impl<-17>(dataI());
    case -18: return shifted_impl<-18>(dataI());
    case -19: return shifted_impl<-19>(dataI());
    case -20: return shifted_impl<-20>(dataI());
    case -21: return shifted_impl<-21>(dataI());
    case -22: return shifted_impl<-22>(dataI());
    case -23: return shifted_impl<-23>(dataI());
    case -24: return shifted_impl<-24>(dataI());
    case -25: return shifted_impl<-25>(dataI());
    case -26: return shifted_impl<-26>(dataI());
    case -27: return shifted_impl<-27>(dataI());
    case -28: return shifted_impl<-28>(dataI());
    case -29: return shifted_impl<-29>(dataI());
    case -30: return shifted_impl<-30>(dataI());
    case -31: return shifted_impl<-31>(dataI());
    }
    return Zero();
}
// }}}1

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

}
}

// vim: foldmethod=marker
