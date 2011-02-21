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
namespace AVX
{

inline void deinterleave(Vector<float> &a, Vector<float> &b)
{
    // a7 a6 a5 a4 a3 a2 a1 a0
    // b7 b6 b5 b4 b3 b2 b1 b0
    const _M256 tmp0 = AVX::permute128<X0, Y0>(a.data(), b.data()); // b3 b2 b1 b0 a3 a2 a1 a0
    const _M256 tmp1 = AVX::permute128<X1, Y1>(a.data(), b.data()); // b7 b6 b5 b4 a7 a6 a5 a4

    const _M256 tmp2 = _mm256_unpacklo_ps(tmp0, tmp1); // b5 b1 b4 b0 a5 a1 a4 a0
    const _M256 tmp3 = _mm256_unpackhi_ps(tmp0, tmp1); // b7 b3 b6 b2 a7 a3 a6 a2

    a.data() = _mm256_unpacklo_ps(tmp2, tmp3); // b6 b4 b2 b0 a6 a4 a2 a0
    b.data() = _mm256_unpackhi_ps(tmp2, tmp3); // b7 b5 b3 b1 a7 a5 a3 a1
}

inline void deinterleave(Vector<short> &a, Vector<short> &b)
{
    __m128i tmp0 = _mm_unpacklo_epi16(a.data(), b.data()); // a0 a4 b0 b4 a1 a5 b1 b5
    __m128i tmp1 = _mm_unpackhi_epi16(a.data(), b.data()); // a2 a6 b2 b6 a3 a7 b3 b7
    __m128i tmp2 = _mm_unpacklo_epi16(tmp0, tmp1); // a0 a2 a4 a6 b0 b2 b4 b6
    __m128i tmp3 = _mm_unpackhi_epi16(tmp0, tmp1); // a1 a3 a5 a7 b1 b3 b5 b7
    a.data() = _mm_unpacklo_epi16(tmp2, tmp3);
    b.data() = _mm_unpackhi_epi16(tmp2, tmp3);
}

inline void deinterleave(Vector<unsigned short> &a, Vector<unsigned short> &b)
{
    __m128i tmp0 = _mm_unpacklo_epi16(a.data(), b.data()); // a0 a4 b0 b4 a1 a5 b1 b5
    __m128i tmp1 = _mm_unpackhi_epi16(a.data(), b.data()); // a2 a6 b2 b6 a3 a7 b3 b7
    __m128i tmp2 = _mm_unpacklo_epi16(tmp0, tmp1); // a0 a2 a4 a6 b0 b2 b4 b6
    __m128i tmp3 = _mm_unpackhi_epi16(tmp0, tmp1); // a1 a3 a5 a7 b1 b3 b5 b7
    a.data() = _mm_unpacklo_epi16(tmp2, tmp3);
    b.data() = _mm_unpackhi_epi16(tmp2, tmp3);
}

} // namespace AVX


namespace Internal
{

template<typename A> inline void HelperImpl<Vc::AVXImpl>::deinterleave(
        float_v &a, float_v &b, const float *m, A align)
{
    a.load(m, align);
    b.load(m + float_v::Size, align);
    Vc::AVX::deinterleave(a, b);
}

template<typename A> inline void HelperImpl<Vc::AVXImpl>::deinterleave(
        float_v &a, float_v &b, const short *m, A align)
{
    const __m256i tmp = Vc::AVX::VectorHelper<__m256i>::load(m, align);
    a.data() = _mm256_cvtepi32_ps(Vc::AVX::concat(
                _mm_srai_epi32(_mm_slli_epi32(AVX::lo128(tmp), 16), 16),
                _mm_srai_epi32(_mm_slli_epi32(AVX::hi128(tmp), 16), 16)));
    b.data() = _mm256_cvtepi32_ps(Vc::AVX::concat(
                _mm_srai_epi32(AVX::lo128(tmp), 16),
                _mm_srai_epi32(AVX::hi128(tmp), 16)));
}

template<typename A> inline void HelperImpl<Vc::AVXImpl>::deinterleave(
        float_v &a, float_v &b, const unsigned short *m, A align)
{
    const __m256i tmp = Vc::AVX::VectorHelper<__m256i>::load(m, align);
    a.data() = _mm256_cvtepi32_ps(Vc::AVX::concat(
                _mm_blend_epi16(AVX::lo128(tmp), _mm_setzero_si128(), 0x55),
                _mm_blend_epi16(AVX::hi128(tmp), _mm_setzero_si128(), 0x55)));
    b.data() = _mm256_cvtepi32_ps(Vc::AVX::concat(
                _mm_blend_epi16(AVX::lo128(tmp), _mm_setzero_si128(), 0xaa),
                _mm_blend_epi16(AVX::hi128(tmp), _mm_setzero_si128(), 0xaa)));
}

template<typename A> inline void HelperImpl<Vc::AVXImpl>::deinterleave(
        double_v &a, double_v &b, const double *m, A align)
{
    a.load(m, align);
    b.load(m + double_v::Size, align);

    __m256d tmp0 = AVX::permute128<AVX::X0, AVX::Y0>(a.data(), b.data()); // b1 b0 a1 a0
    __m256d tmp1 = AVX::permute128<AVX::X1, AVX::Y1>(a.data(), b.data()); // b3 b2 a3 a2

    a.data() = _mm256_unpacklo_pd(tmp0, tmp1); // b2 b0 a2 a0
    b.data() = _mm256_unpackhi_pd(tmp0, tmp1); // b3 b1 a3 a1
}

template<typename A> inline void HelperImpl<Vc::AVXImpl>::deinterleave(
        int_v &a, int_v &b, const int *m, A align)
{
    a.load(m, align);
    b.load(m + int_v::Size, align);

    const __m256 tmp0 = AVX::avx_cast<__m256>(AVX::permute128<AVX::X0, AVX::Y0>(a.data(), b.data()));
    const __m256 tmp1 = AVX::avx_cast<__m256>(AVX::permute128<AVX::X1, AVX::Y1>(a.data(), b.data()));

    const __m256 tmp2 = _mm256_unpacklo_ps(tmp0, tmp1); // b5 b1 b4 b0 a5 a1 a4 a0
    const __m256 tmp3 = _mm256_unpackhi_ps(tmp0, tmp1); // b7 b3 b6 b2 a7 a3 a6 a2

    a.data() = AVX::avx_cast<__m256i>(_mm256_unpacklo_ps(tmp2, tmp3)); // b6 b4 b2 b0 a6 a4 a2 a0
    b.data() = AVX::avx_cast<__m256i>(_mm256_unpackhi_ps(tmp2, tmp3)); // b7 b5 b3 b1 a7 a5 a3 a1
}

template<typename A> inline void HelperImpl<Vc::AVXImpl>::deinterleave(
        int_v &a, int_v &b, const short *m, A align)
{
    const __m256i tmp = Vc::AVX::VectorHelper<__m256i>::load(m, align);
    a.data() = Vc::AVX::concat(
                _mm_srai_epi32(_mm_slli_epi32(AVX::lo128(tmp), 16), 16),
                _mm_srai_epi32(_mm_slli_epi32(AVX::hi128(tmp), 16), 16));
    b.data() = Vc::AVX::concat(
                _mm_srai_epi32(AVX::lo128(tmp), 16),
                _mm_srai_epi32(AVX::hi128(tmp), 16));
}

template<typename A> inline void HelperImpl<Vc::AVXImpl>::deinterleave(
        uint_v &a, uint_v &b, const unsigned int *m, A align)
{
    a.load(m, align);
    b.load(m + uint_v::Size, align);

    const __m256 tmp0 = AVX::avx_cast<__m256>(AVX::permute128<AVX::X0, AVX::Y0>(a.data(), b.data()));
    const __m256 tmp1 = AVX::avx_cast<__m256>(AVX::permute128<AVX::X1, AVX::Y1>(a.data(), b.data()));

    const __m256 tmp2 = _mm256_unpacklo_ps(tmp0, tmp1); // b5 b1 b4 b0 a5 a1 a4 a0
    const __m256 tmp3 = _mm256_unpackhi_ps(tmp0, tmp1); // b7 b3 b6 b2 a7 a3 a6 a2

    a.data() = AVX::avx_cast<__m256i>(_mm256_unpacklo_ps(tmp2, tmp3)); // b6 b4 b2 b0 a6 a4 a2 a0
    b.data() = AVX::avx_cast<__m256i>(_mm256_unpackhi_ps(tmp2, tmp3)); // b7 b5 b3 b1 a7 a5 a3 a1
}

template<typename A> inline void HelperImpl<Vc::AVXImpl>::deinterleave(
        uint_v &a, uint_v &b, const unsigned short *m, A align)
{
    const __m256i tmp = Vc::AVX::VectorHelper<__m256i>::load(m, align);
    a.data() = Vc::AVX::concat(
                _mm_srli_epi32(_mm_slli_epi32(AVX::lo128(tmp), 16), 16),
                _mm_srli_epi32(_mm_slli_epi32(AVX::hi128(tmp), 16), 16));
    b.data() = Vc::AVX::concat(
                _mm_srli_epi32(AVX::lo128(tmp), 16),
                _mm_srli_epi32(AVX::hi128(tmp), 16));
}

template<typename A> inline void HelperImpl<Vc::AVXImpl>::deinterleave(
        short_v &a, short_v &b, const short *m, A align)
{
    a.load(m, align);
    b.load(m + short_v::Size, align);
    Vc::AVX::deinterleave(a, b);
}

template<typename A> inline void HelperImpl<Vc::AVXImpl>::deinterleave(
        ushort_v &a, ushort_v &b, const unsigned short *m, A align)
{
    a.load(m, align);
    b.load(m + ushort_v::Size, align);
    Vc::AVX::deinterleave(a, b);
}

} // namespace Internal
} // namespace Vc
