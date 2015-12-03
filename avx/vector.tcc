/*  This file is part of the Vc library. {{{
Copyright © 2011-2015 Matthias Kretz <kretz@kde.org>
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

#include "../common/x86_prefetches.h"
#include "../common/gatherimplementation.h"
#include "../common/scatterimplementation.h"
#include "limits.h"
#include "const.h"
#include "../common/set.h"
#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
///////////////////////////////////////////////////////////////////////////////////////////
// constants {{{1
template <typename T> Vc_INTRINSIC Vector<T, VectorAbi::Avx>::Vector(VectorSpecialInitializerZero) : d{} {}

template <> Vc_INTRINSIC AVX2::double_v::Vector(VectorSpecialInitializerOne) : d(AVX::setone_pd()) {}
template <> Vc_INTRINSIC  AVX2::float_v::Vector(VectorSpecialInitializerOne) : d(AVX::setone_ps()) {}
#ifdef Vc_IMPL_AVX2
template <> Vc_INTRINSIC    AVX2::int_v::Vector(VectorSpecialInitializerOne) : d(AVX::setone_epi32()) {}
template <> Vc_INTRINSIC   AVX2::uint_v::Vector(VectorSpecialInitializerOne) : d(AVX::setone_epu32()) {}
template <> Vc_INTRINSIC  AVX2::short_v::Vector(VectorSpecialInitializerOne) : d(AVX::setone_epi16()) {}
template <> Vc_INTRINSIC AVX2::ushort_v::Vector(VectorSpecialInitializerOne) : d(AVX::setone_epu16()) {}
template <> Vc_INTRINSIC AVX2::Vector<  signed char>::Vector(VectorSpecialInitializerOne) : d(AVX::setone_epi8()) {}
template <> Vc_INTRINSIC AVX2::Vector<unsigned char>::Vector(VectorSpecialInitializerOne) : d(AVX::setone_epu8()) {}
#endif

template <typename T>
Vc_ALWAYS_INLINE Vector<T, VectorAbi::Avx>::Vector(
    VectorSpecialInitializerIndexesFromZero)
    : Vector(AVX::IndexesFromZeroData<T>::address(), Vc::Aligned)
{
}

template <>
Vc_ALWAYS_INLINE AVX2::float_v::Vector(VectorSpecialInitializerIndexesFromZero)
    : Vector(AVX::IndexesFromZeroData<int>::address(), Vc::Aligned)
{
}
template <>
Vc_ALWAYS_INLINE AVX2::double_v::Vector(VectorSpecialInitializerIndexesFromZero)
    : Vector(AVX::IndexesFromZeroData<int>::address(), Vc::Aligned)
{
}

///////////////////////////////////////////////////////////////////////////////////////////
// load member functions {{{1
// general load, implemented via LoadHelper {{{2
template <typename DstT>
template <typename SrcT, typename Flags, typename>
Vc_INTRINSIC void Vector<DstT, VectorAbi::Avx>::load(const SrcT *mem, Flags flags)
{
    Common::handleLoadPrefetches(mem, flags);
    d.v() = Detail::load<VectorType, DstT>(mem, flags);
}

///////////////////////////////////////////////////////////////////////////////////////////
// zeroing {{{1
template<typename T> Vc_INTRINSIC void Vector<T, VectorAbi::Avx>::setZero()
{
    data() = Detail::zero<VectorType>();
}
template<typename T> Vc_INTRINSIC void Vector<T, VectorAbi::Avx>::setZero(const Mask &k)
{
    data() = Detail::andnot_(AVX::avx_cast<VectorType>(k.data()), data());
}
template<typename T> Vc_INTRINSIC void Vector<T, VectorAbi::Avx>::setZeroInverted(const Mask &k)
{
    data() = Detail::and_(AVX::avx_cast<VectorType>(k.data()), data());
}

template<> Vc_INTRINSIC void AVX2::double_v::setQnan()
{
    data() = Detail::allone<VectorType>();
}
template<> Vc_INTRINSIC void AVX2::double_v::setQnan(MaskArgument k)
{
    data() = _mm256_or_pd(data(), k.dataD());
}
template<> Vc_INTRINSIC void AVX2::float_v::setQnan()
{
    data() = Detail::allone<VectorType>();
}
template<> Vc_INTRINSIC void AVX2::float_v::setQnan(MaskArgument k)
{
    data() = _mm256_or_ps(data(), k.data());
}

///////////////////////////////////////////////////////////////////////////////////////////
// stores {{{1
template <typename T>
template <typename U,
          typename Flags,
          typename>
Vc_INTRINSIC void Vector<T, VectorAbi::Avx>::store(U *mem, Flags flags) const
{
    Common::handleStorePrefetches(mem, flags);
    HV::template store<Flags>(mem, data());
}

template <typename T>
template <typename U,
          typename Flags,
          typename>
Vc_INTRINSIC void Vector<T, VectorAbi::Avx>::store(U *mem, Mask mask, Flags flags) const
{
    Common::handleStorePrefetches(mem, flags);
    HV::template store<Flags>(mem, data(), AVX::avx_cast<VectorType>(mask.data()));
}

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
// division {{{1
template<typename T> inline AVX2::Vector<T> &Vector<T, VectorAbi::Avx>::operator/=(EntryType x)
{
    if (HasVectorDivision) {
        return operator/=(AVX2::Vector<T>(x));
    }
    Common::for_all_vector_entries<Size>([&](size_t i) { d.set(i, d.m(i) / x); });
    return *this;
}
// per default fall back to scalar division
template<typename T> inline AVX2::Vector<T> &Vector<T, VectorAbi::Avx>::operator/=(Vc_ALIGNED_PARAMETER(AVX2::Vector<T>) x)
{
    Common::for_all_vector_entries<Size>([&](size_t i) { d.set(i, d.m(i) / x.d.m(i)); });
    return *this;
}

template<typename T> inline Vc_PURE AVX2::Vector<T> Vector<T, VectorAbi::Avx>::operator/(Vc_ALIGNED_PARAMETER(AVX2::Vector<T>) x) const
{
    AVX2::Vector<T> r;
    Common::for_all_vector_entries<Size>(
        [&](size_t i) { r.d.set(i, d.m(i) / x.d.m(i)); });
    return r;
}
#ifdef Vc_IMPL_AVX2
// specialize division on type
static Vc_INTRINSIC __m256i Vc_CONST divInt(__m256i a, __m256i b)
{
    using namespace AVX;
    const m256d lo1 = _mm256_cvtepi32_pd(lo128(a));
    const m256d lo2 = _mm256_cvtepi32_pd(lo128(b));
    const m256d hi1 = _mm256_cvtepi32_pd(hi128(a));
    const m256d hi2 = _mm256_cvtepi32_pd(hi128(b));
    return concat(
            _mm256_cvttpd_epi32(_mm256_div_pd(lo1, lo2)),
            _mm256_cvttpd_epi32(_mm256_div_pd(hi1, hi2))
            );
}
template<> inline AVX2::int_v &AVX2::int_v::operator/=(Vc_ALIGNED_PARAMETER(AVX2::int_v) x)
{
    d.v() = divInt(d.v(), x.d.v());
    return *this;
}
template<> inline AVX2::int_v Vc_PURE AVX2::int_v::operator/(Vc_ALIGNED_PARAMETER(AVX2::int_v) x) const
{
    return divInt(d.v(), x.d.v());
}
static inline __m256i Vc_CONST divUInt(__m256i a, __m256i b) {
    // SSE/AVX only has signed int conversion to doubles. Therefore we first adjust the input before
    // conversion and take the adjustment back after the conversion.
    // It could be argued that for b this is not really important because division by a b >= 2^31 is
    // useless. But for full correctness it cannot be ignored.
    using namespace AVX;
    const __m256i aa = add_epi32(a, set1_epi32(-2147483648));
    const __m256i bb = add_epi32(b, set1_epi32(-2147483648));
    const __m256d loa = _mm256_add_pd(_mm256_cvtepi32_pd(lo128(aa)), set1_pd(2147483648.));
    const __m256d hia = _mm256_add_pd(_mm256_cvtepi32_pd(hi128(aa)), set1_pd(2147483648.));
    const __m256d lob = _mm256_add_pd(_mm256_cvtepi32_pd(lo128(bb)), set1_pd(2147483648.));
    const __m256d hib = _mm256_add_pd(_mm256_cvtepi32_pd(hi128(bb)), set1_pd(2147483648.));
    // there is one remaining problem: a >= 2^31 and b == 1
    // in that case the return value would be 2^31
    return avx_cast<__m256i>(_mm256_blendv_ps(
        avx_cast<__m256>(concat(_mm256_cvttpd_epi32(_mm256_div_pd(loa, lob)),
                                          _mm256_cvttpd_epi32(_mm256_div_pd(hia, hib)))),
        avx_cast<__m256>(a),
        avx_cast<__m256>(cmpeq_epi32(b, setone_epi32()))));
}
template<> Vc_ALWAYS_INLINE AVX2::uint_v &AVX2::uint_v::operator/=(Vc_ALIGNED_PARAMETER(AVX2::uint_v) x)
{
    d.v() = divUInt(d.v(), x.d.v());
    return *this;
}
template<> Vc_ALWAYS_INLINE AVX2::uint_v Vc_PURE AVX2::uint_v::operator/(Vc_ALIGNED_PARAMETER(AVX2::uint_v) x) const
{
    return divUInt(d.v(), x.d.v());
}
template <typename T> static inline __m256i Vc_CONST divShort(__m256i a, __m256i b)
{
    using namespace AVX;
    const __m256 lo =
        _mm256_div_ps(convert<T, float>(lo128(a)), convert<T, float>(lo128(b)));
    const __m256 hi =
        _mm256_div_ps(convert<T, float>(hi128(a)), convert<T, float>(hi128(b)));
    if (std::is_same<T, ushort>::value) {
        const float_v threshold = 32767.f;
        const __m128i loShort = (Vc_IS_UNLIKELY((float_v(lo) > threshold).isNotEmpty()))
                                    ? convert<float, ushort>(lo)
                                    : convert<float, short>(lo);
        const __m128i hiShort = (Vc_IS_UNLIKELY((float_v(hi) > threshold).isNotEmpty()))
                                    ? convert<float, ushort>(hi)
                                    : convert<float, short>(hi);
        return concat(loShort, hiShort);
    }
    return concat(convert<float, short>(lo), convert<float, short>(hi));
}
template<> Vc_ALWAYS_INLINE AVX2::short_v &AVX2::short_v::operator/=(Vc_ALIGNED_PARAMETER(AVX2::short_v) x)
{
    d.v() = divShort<short>(d.v(), x.d.v());
    return *this;
}
template<> Vc_ALWAYS_INLINE AVX2::short_v Vc_PURE AVX2::short_v::operator/(Vc_ALIGNED_PARAMETER(AVX2::short_v) x) const
{
    return divShort<short>(d.v(), x.d.v());
}
template<> Vc_ALWAYS_INLINE AVX2::ushort_v &AVX2::ushort_v::operator/=(Vc_ALIGNED_PARAMETER(AVX2::ushort_v) x)
{
    d.v() = divShort<unsigned short>(d.v(), x.d.v());
    return *this;
}
template<> Vc_ALWAYS_INLINE AVX2::ushort_v Vc_PURE AVX2::ushort_v::operator/(Vc_ALIGNED_PARAMETER(AVX2::ushort_v) x) const
{
    return divShort<unsigned short>(d.v(), x.d.v());
}
#endif
template<> Vc_INTRINSIC AVX2::float_v &AVX2::float_v::operator/=(Vc_ALIGNED_PARAMETER(AVX2::float_v) x)
{
    d.v() = _mm256_div_ps(d.v(), x.d.v());
    return *this;
}
template<> Vc_INTRINSIC AVX2::float_v Vc_PURE AVX2::float_v::operator/(Vc_ALIGNED_PARAMETER(AVX2::float_v) x) const
{
    return _mm256_div_ps(d.v(), x.d.v());
}
template<> Vc_INTRINSIC AVX2::double_v &AVX2::double_v::operator/=(Vc_ALIGNED_PARAMETER(AVX2::double_v) x)
{
    d.v() = _mm256_div_pd(d.v(), x.d.v());
    return *this;
}
template<> Vc_INTRINSIC AVX2::double_v Vc_PURE AVX2::double_v::operator/(Vc_ALIGNED_PARAMETER(AVX2::double_v) x) const
{
    return _mm256_div_pd(d.v(), x.d.v());
}

///////////////////////////////////////////////////////////////////////////////////////////
// integer ops {{{1
#ifdef Vc_IMPL_AVX2
template <typename T>
inline Vc_PURE Vector<T, VectorAbi::Avx> Vector<T, VectorAbi::Avx>::operator%(
    const Vector &n) const
{
    return *this - n * (*this / n);
}

#define Vc_OP_IMPL(T, symbol)                                                            \
    template <>                                                                          \
    Vc_ALWAYS_INLINE AVX2::Vector<T> &Vector<T, VectorAbi::Avx>::operator symbol##=(     \
        AsArg x)                                                                         \
    {                                                                                    \
        Common::unrolled_loop<std::size_t, 0, Size>(                                     \
            [&](std::size_t i) { d.set(i, d.m(i) symbol x.d.m(i)); });                   \
        return *this;                                                                    \
    }                                                                                    \
    template <>                                                                          \
    Vc_ALWAYS_INLINE Vc_PURE AVX2::Vector<T> Vector<T, VectorAbi::Avx>::operator symbol( \
        AsArg x) const                                                                   \
    {                                                                                    \
        AVX2::Vector<T> r;                                                               \
        Common::unrolled_loop<std::size_t, 0, Size>(                                     \
            [&](std::size_t i) { r.d.set(i, d.m(i) symbol x.d.m(i)); });                 \
        return r;                                                                        \
    }
Vc_OP_IMPL(int, <<)
Vc_OP_IMPL(int, >>)
Vc_OP_IMPL(unsigned int, <<)
Vc_OP_IMPL(unsigned int, >>)
Vc_OP_IMPL(short, <<)
Vc_OP_IMPL(short, >>)
Vc_OP_IMPL(unsigned short, <<)
Vc_OP_IMPL(unsigned short, >>)
#undef Vc_OP_IMPL
#endif

template<typename T> Vc_ALWAYS_INLINE AVX2::Vector<T> &Vector<T, VectorAbi::Avx>::operator>>=(int shift) {
    d.v() = Detail::shiftRight(d.v(), shift, T());
    return *static_cast<AVX2::Vector<T> *>(this);
}
template<typename T> Vc_ALWAYS_INLINE Vc_PURE AVX2::Vector<T> Vector<T, VectorAbi::Avx>::operator>>(int shift) const {
    return Detail::shiftRight(d.v(), shift, T());
}
template<typename T> Vc_ALWAYS_INLINE AVX2::Vector<T> &Vector<T, VectorAbi::Avx>::operator<<=(int shift) {
    d.v() = Detail::shiftLeft(d.v(), shift, T());
    return *static_cast<AVX2::Vector<T> *>(this);
}
template<typename T> Vc_ALWAYS_INLINE Vc_PURE AVX2::Vector<T> Vector<T, VectorAbi::Avx>::operator<<(int shift) const {
    return Detail::shiftLeft(d.v(), shift, T());
}

#define Vc_OP_IMPL(T, symbol, fun)                                                       \
    template <>                                                                          \
    Vc_ALWAYS_INLINE AVX2::Vector<T> &Vector<T, VectorAbi::Avx>::operator symbol##=(     \
        AsArg x)                                                                         \
    {                                                                                    \
        d.v() = Detail::fun(d.v(), x.d.v());                                             \
        return *this;                                                                    \
    }                                                                                    \
    template <>                                                                          \
    Vc_ALWAYS_INLINE Vc_PURE AVX2::Vector<T> Vector<T, VectorAbi::Avx>::operator symbol( \
        AsArg x) const                                                                   \
    {                                                                                    \
        return AVX2::Vector<T>(Detail::fun(d.v(), x.d.v()));                             \
    }
#ifdef Vc_IMPL_AVX2
  Vc_OP_IMPL(int, &, and_)
  Vc_OP_IMPL(int, |, or_)
  Vc_OP_IMPL(int, ^, xor_)
  Vc_OP_IMPL(unsigned int, &, and_)
  Vc_OP_IMPL(unsigned int, |, or_)
  Vc_OP_IMPL(unsigned int, ^, xor_)
  Vc_OP_IMPL(short, &, and_)
  Vc_OP_IMPL(short, |, or_)
  Vc_OP_IMPL(short, ^, xor_)
  Vc_OP_IMPL(unsigned short, &, and_)
  Vc_OP_IMPL(unsigned short, |, or_)
  Vc_OP_IMPL(unsigned short, ^, xor_)
#endif
#ifdef Vc_ENABLE_FLOAT_BIT_OPERATORS
  Vc_OP_IMPL(float, &, and_)
  Vc_OP_IMPL(float, |, or_)
  Vc_OP_IMPL(float, ^, xor_)
  Vc_OP_IMPL(double, &, and_)
  Vc_OP_IMPL(double, |, or_)
  Vc_OP_IMPL(double, ^, xor_)
#endif
#undef Vc_OP_IMPL

// isnegative {{{1
Vc_INTRINSIC Vc_CONST AVX2::float_m isnegative(AVX2::float_v x)
{
    return AVX::avx_cast<__m256>(AVX::srai_epi32<31>(
        AVX::avx_cast<__m256i>(_mm256_and_ps(AVX::setsignmask_ps(), x.data()))));
}
Vc_INTRINSIC Vc_CONST AVX2::double_m isnegative(AVX2::double_v x)
{
    return Mem::permute<X1, X1, X3, X3>(AVX::avx_cast<__m256>(AVX::srai_epi32<31>(
        AVX::avx_cast<__m256i>(_mm256_and_pd(AVX::setsignmask_pd(), x.data())))));
}
// gathers {{{1
template <>
template <typename MT, typename IT>
inline void AVX2::double_v::gatherImplementation(const MT *mem, IT &&indexes)
{
    d.v() = _mm256_setr_pd(mem[indexes[0]], mem[indexes[1]], mem[indexes[2]], mem[indexes[3]]);
}

template <>
template <typename MT, typename IT>
inline void AVX2::float_v::gatherImplementation(const MT *mem, IT &&indexes)
{
    d.v() = _mm256_setr_ps(mem[indexes[0]],
                           mem[indexes[1]],
                           mem[indexes[2]],
                           mem[indexes[3]],
                           mem[indexes[4]],
                           mem[indexes[5]],
                           mem[indexes[6]],
                           mem[indexes[7]]);
}

#ifdef Vc_IMPL_AVX2
template <>
template <typename MT, typename IT>
inline void AVX2::int_v::gatherImplementation(const MT *mem, IT &&indexes)
{
    d.v() = _mm256_setr_epi32(mem[indexes[0]], mem[indexes[1]], mem[indexes[2]],
                              mem[indexes[3]], mem[indexes[4]], mem[indexes[5]],
                              mem[indexes[6]], mem[indexes[7]]);
}

template <>
template <typename MT, typename IT>
inline void AVX2::uint_v::gatherImplementation(const MT *mem, IT &&indexes)
{
    d.v() = _mm256_setr_epi32(mem[indexes[0]], mem[indexes[1]], mem[indexes[2]],
                              mem[indexes[3]], mem[indexes[4]], mem[indexes[5]],
                              mem[indexes[6]], mem[indexes[7]]);
}

template <>
template <typename MT, typename IT>
inline void AVX2::short_v::gatherImplementation(const MT *mem, IT &&indexes)
{
    d.v() = _mm256_setr_epi16(mem[indexes[0]], mem[indexes[1]], mem[indexes[2]],
                              mem[indexes[3]], mem[indexes[4]], mem[indexes[5]],
                              mem[indexes[6]], mem[indexes[7]], mem[indexes[8]],
                              mem[indexes[9]], mem[indexes[10]], mem[indexes[11]],
                              mem[indexes[12]], mem[indexes[13]], mem[indexes[14]],
                              mem[indexes[15]]);
}

template <>
template <typename MT, typename IT>
inline void AVX2::ushort_v::gatherImplementation(const MT *mem, IT &&indexes)
{
    d.v() = _mm256_setr_epi16(mem[indexes[0]], mem[indexes[1]], mem[indexes[2]],
                              mem[indexes[3]], mem[indexes[4]], mem[indexes[5]],
                              mem[indexes[6]], mem[indexes[7]], mem[indexes[8]],
                              mem[indexes[9]], mem[indexes[10]], mem[indexes[11]],
                              mem[indexes[12]], mem[indexes[13]], mem[indexes[14]],
                              mem[indexes[15]]);
}
#endif

template <typename T>
template <typename MT, typename IT>
inline void Vector<T, VectorAbi::Avx>::gatherImplementation(const MT *mem, IT &&indexes, MaskArgument mask)
{
    using Selector = std::integral_constant < Common::GatherScatterImplementation,
#ifdef Vc_USE_SET_GATHERS
          Traits::is_simd_vector<IT>::value ? Common::GatherScatterImplementation::SetIndexZero :
#endif
#ifdef Vc_USE_BSF_GATHERS
                                            Common::GatherScatterImplementation::BitScanLoop
#elif defined Vc_USE_POPCNT_BSF_GATHERS
              Common::GatherScatterImplementation::PopcntSwitch
#else
              Common::GatherScatterImplementation::SimpleLoop
#endif
                                                > ;
    Common::executeGather(Selector(), *this, mem, indexes, mask);
}

template <typename T>
template <typename MT, typename IT>
inline void Vector<T, VectorAbi::Avx>::scatterImplementation(MT *mem, IT &&indexes) const
{
    Common::unrolled_loop<std::size_t, 0, Size>([&](std::size_t i) { mem[indexes[i]] = d.m(i); });
}

template <typename T>
template <typename MT, typename IT>
inline void Vector<T, VectorAbi::Avx>::scatterImplementation(MT *mem, IT &&indexes, MaskArgument mask) const
{
    using Selector = std::integral_constant < Common::GatherScatterImplementation,
#ifdef Vc_USE_SET_GATHERS
          Traits::is_simd_vector<IT>::value ? Common::GatherScatterImplementation::SetIndexZero :
#endif
#ifdef Vc_USE_BSF_GATHERS
                                            Common::GatherScatterImplementation::BitScanLoop
#elif defined Vc_USE_POPCNT_BSF_GATHERS
              Common::GatherScatterImplementation::PopcntSwitch
#else
              Common::GatherScatterImplementation::SimpleLoop
#endif
                                                > ;
    Common::executeScatter(Selector(), *this, mem, indexes, mask);
}

#if defined(Vc_MSVC) && Vc_MSVC >= 170000000
// MSVC miscompiles the store mem[indexes[1]] = d.m(1) for T = (u)short
template <>
template <typename MT, typename IT>
Vc_ALWAYS_INLINE void AVX2::short_v::scatterImplementation(MT *mem, IT &&indexes) const
{
    const unsigned int tmp = d.v()._d.__m128i_u32[0];
    mem[indexes[0]] = tmp & 0xffff;
    mem[indexes[1]] = tmp >> 16;
    mem[indexes[2]] = _mm_extract_epi16(d.v(), 2);
    mem[indexes[3]] = _mm_extract_epi16(d.v(), 3);
    mem[indexes[4]] = _mm_extract_epi16(d.v(), 4);
    mem[indexes[5]] = _mm_extract_epi16(d.v(), 5);
    mem[indexes[6]] = _mm_extract_epi16(d.v(), 6);
    mem[indexes[7]] = _mm_extract_epi16(d.v(), 7);
}
template <>
template <typename MT, typename IT>
Vc_ALWAYS_INLINE void AVX2::ushort_v::scatterImplementation(MT *mem, IT &&indexes) const
{
    const unsigned int tmp = d.v()._d.__m128i_u32[0];
    mem[indexes[0]] = tmp & 0xffff;
    mem[indexes[1]] = tmp >> 16;
    mem[indexes[2]] = _mm_extract_epi16(d.v(), 2);
    mem[indexes[3]] = _mm_extract_epi16(d.v(), 3);
    mem[indexes[4]] = _mm_extract_epi16(d.v(), 4);
    mem[indexes[5]] = _mm_extract_epi16(d.v(), 5);
    mem[indexes[6]] = _mm_extract_epi16(d.v(), 6);
    mem[indexes[7]] = _mm_extract_epi16(d.v(), 7);
}
#endif

///////////////////////////////////////////////////////////////////////////////////////////
// operator- {{{1
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
template<typename T> Vc_ALWAYS_INLINE Vc_PURE AVX2::Vector<T> Vector<T, VectorAbi::Avx>::operator-() const
{
    return VectorType(-d.builtin());
}
#else
template<typename T> Vc_ALWAYS_INLINE Vc_PURE AVX2::Vector<T> Vector<T, VectorAbi::Avx>::operator-() const
{
    return Detail::negate(d.v(), std::integral_constant<std::size_t, sizeof(T)>());
}
#endif

///////////////////////////////////////////////////////////////////////////////////////////
// horizontal ops {{{1
template <typename T>
Vc_INTRINSIC std::pair<Vector<T, VectorAbi::Avx>, int>
Vector<T, VectorAbi::Avx>::minIndex() const
{
    AVX2::Vector<T> x = min();
    return std::make_pair(x, (*this == x).firstOne());
}
template <typename T>
Vc_INTRINSIC std::pair<Vector<T, VectorAbi::Avx>, int>
Vector<T, VectorAbi::Avx>::maxIndex() const
{
    AVX2::Vector<T> x = max();
    return std::make_pair(x, (*this == x).firstOne());
}
template <> Vc_INTRINSIC std::pair<AVX2::float_v, int> AVX2::float_v::minIndex() const
{
    /*
    // 28 cycles latency:
    __m256 x = _mm256_min_ps(Mem::permute128<X1, X0>(d.v()), d.v());
    x = _mm256_min_ps(x, Reg::permute<X2, X3, X0, X1>(x));
    AVX2::float_v xx = _mm256_min_ps(x, Reg::permute<X1, X0, X3, X2>(x));
    AVX2::uint_v idx = AVX2::uint_v::IndexesFromZero();
    idx = _mm256_castps_si256(
        _mm256_or_ps((*this != xx).data(), _mm256_castsi256_ps(idx.data())));
    return std::make_pair(xx, (*this == xx).firstOne());

    __m128 loData = AVX::lo128(d.v());
    __m128 hiData = AVX::hi128(d.v());
    const __m128 less2 = _mm_cmplt_ps(hiData, loData);
    loData = _mm_min_ps(loData, hiData);
    hiData = Mem::permute<X2, X3, X0, X1>(loData);
    const __m128 less1 = _mm_cmplt_ps(hiData, loData);
    loData = _mm_min_ps(loData, hiData);
    hiData = Mem::permute<X1, X0, X3, X2>(loData);
    const __m128 less0 = _mm_cmplt_ps(hiData, loData);
    unsigned bits = _mm_movemask_ps(less0) & 0x1;
    bits |= ((_mm_movemask_ps(less1) << 1) - bits) & 0x2;
    bits |= ((_mm_movemask_ps(less2) << 3) - bits) & 0x4;
    loData = _mm_min_ps(loData, hiData);
    return std::make_pair(AVX::concat(loData, loData), bits);
    */

    // 28 cycles Latency:
    __m256 x = d.v();
    __m256 idx = _mm256_castsi256_ps(
        Detail::load<AlignedTag>(AVX::IndexesFromZeroData<int>::address(), __m256i()));
    __m256 y = Mem::permute128<X1, X0>(x);
    __m256 idy = Mem::permute128<X1, X0>(idx);
    __m256 less = AVX::cmplt_ps(x, y);

    x = _mm256_blendv_ps(y, x, less);
    idx = _mm256_blendv_ps(idy, idx, less);
    y = Reg::permute<X2, X3, X0, X1>(x);
    idy = Reg::permute<X2, X3, X0, X1>(idx);
    less = AVX::cmplt_ps(x, y);

    x = _mm256_blendv_ps(y, x, less);
    idx = _mm256_blendv_ps(idy, idx, less);
    y = Reg::permute<X1, X0, X3, X2>(x);
    idy = Reg::permute<X1, X0, X3, X2>(idx);
    less = AVX::cmplt_ps(x, y);

    idx = _mm256_blendv_ps(idy, idx, less);

    const auto index = _mm_cvtsi128_si32(AVX::avx_cast<__m128i>(idx));
    __asm__ __volatile__(""); // help GCC to order the instructions better
    x = _mm256_blendv_ps(y, x, less);
    return std::make_pair(x, index);
}
template<typename T> Vc_ALWAYS_INLINE AVX2::Vector<T> Vector<T, VectorAbi::Avx>::partialSum() const
{
    //   a    b    c    d    e    f    g    h
    // +      a    b    c    d    e    f    g    -> a ab bc  cd   de    ef     fg      gh
    // +           a    ab   bc   cd   de   ef   -> a ab abc abcd bcde  cdef   defg    efgh
    // +                     a    ab   abc  abcd -> a ab abc abcd abcde abcdef abcdefg abcdefgh
    AVX2::Vector<T> tmp = *this;
    if (Size >  1) tmp += tmp.shifted(-1);
    if (Size >  2) tmp += tmp.shifted(-2);
    if (Size >  4) tmp += tmp.shifted(-4);
    if (Size >  8) tmp += tmp.shifted(-8);
    if (Size > 16) tmp += tmp.shifted(-16);
    return tmp;
}

/* This function requires correct masking because the neutral element of \p op is not necessarily 0
 *
template<typename T> template<typename BinaryOperation> Vc_ALWAYS_INLINE AVX2::Vector<T> Vector<T, VectorAbi::Avx>::partialSum(BinaryOperation op) const
{
    //   a    b    c    d    e    f    g    h
    // +      a    b    c    d    e    f    g    -> a ab bc  cd   de    ef     fg      gh
    // +           a    ab   bc   cd   de   ef   -> a ab abc abcd bcde  cdef   defg    efgh
    // +                     a    ab   abc  abcd -> a ab abc abcd abcde abcdef abcdefg abcdefgh
    AVX2::Vector<T> tmp = *this;
    Mask mask(true);
    if (Size >  1) tmp(mask) = op(tmp, tmp.shifted(-1));
    if (Size >  2) tmp(mask) = op(tmp, tmp.shifted(-2));
    if (Size >  4) tmp(mask) = op(tmp, tmp.shifted(-4));
    if (Size >  8) tmp(mask) = op(tmp, tmp.shifted(-8));
    if (Size > 16) tmp(mask) = op(tmp, tmp.shifted(-16));
    return tmp;
}
*/

template<typename T> Vc_ALWAYS_INLINE typename Vector<T, VectorAbi::Avx>::EntryType Vector<T, VectorAbi::Avx>::min(MaskArg m) const
{
    AVX2::Vector<T> tmp = std::numeric_limits<AVX2::Vector<T> >::max();
    tmp(m) = *this;
    return tmp.min();
}
template<typename T> Vc_ALWAYS_INLINE typename Vector<T, VectorAbi::Avx>::EntryType Vector<T, VectorAbi::Avx>::max(MaskArg m) const
{
    AVX2::Vector<T> tmp = std::numeric_limits<AVX2::Vector<T> >::min();
    tmp(m) = *this;
    return tmp.max();
}
template<typename T> Vc_ALWAYS_INLINE typename Vector<T, VectorAbi::Avx>::EntryType Vector<T, VectorAbi::Avx>::product(MaskArg m) const
{
    AVX2::Vector<T> tmp(Vc::One);
    tmp(m) = *this;
    return tmp.product();
}
template<typename T> Vc_ALWAYS_INLINE typename Vector<T, VectorAbi::Avx>::EntryType Vector<T, VectorAbi::Avx>::sum(MaskArg m) const
{
    AVX2::Vector<T> tmp(Vc::Zero);
    tmp(m) = *this;
    return tmp.sum();
}//}}}
// copySign {{{1
template<> Vc_INTRINSIC AVX2::float_v AVX2::float_v::copySign(AVX2::float_v::AsArg reference) const
{
    return _mm256_or_ps(
            _mm256_and_ps(reference.d.v(), AVX::setsignmask_ps()),
            _mm256_and_ps(d.v(), AVX::setabsmask_ps())
            );
}
template<> Vc_INTRINSIC AVX2::double_v AVX2::double_v::copySign(AVX2::double_v::AsArg reference) const
{
    return _mm256_or_pd(
            _mm256_and_pd(reference.d.v(), AVX::setsignmask_pd()),
            _mm256_and_pd(d.v(), AVX::setabsmask_pd())
            );
}//}}}1
// exponent {{{1
namespace Detail
{
Vc_INTRINSIC Vc_CONST __m256 exponent(__m256 v)
{
    using namespace AVX;
    __m128i tmp0 = _mm_srli_epi32(avx_cast<__m128i>(v), 23);
    __m128i tmp1 = _mm_srli_epi32(avx_cast<__m128i>(hi128(v)), 23);
    tmp0 = _mm_sub_epi32(tmp0, _mm_set1_epi32(0x7f));
    tmp1 = _mm_sub_epi32(tmp1, _mm_set1_epi32(0x7f));
    return _mm256_cvtepi32_ps(concat(tmp0, tmp1));
}
Vc_INTRINSIC Vc_CONST __m256d exponent(__m256d v)
{
    using namespace AVX;
    __m128i tmp0 = _mm_srli_epi64(avx_cast<__m128i>(v), 52);
    __m128i tmp1 = _mm_srli_epi64(avx_cast<__m128i>(hi128(v)), 52);
    tmp0 = _mm_sub_epi32(tmp0, _mm_set1_epi32(0x3ff));
    tmp1 = _mm_sub_epi32(tmp1, _mm_set1_epi32(0x3ff));
    return _mm256_cvtepi32_pd(avx_cast<__m128i>(Mem::shuffle<X0, X2, Y0, Y2>(avx_cast<__m128>(tmp0), avx_cast<__m128>(tmp1))));
}
} // namespace Detail

template<> Vc_INTRINSIC AVX2::float_v AVX2::float_v::exponent() const
{
    Vc_ASSERT((*this >= 0.f).isFull());
    return Detail::exponent(d.v());
}
template<> Vc_INTRINSIC AVX2::double_v AVX2::double_v::exponent() const
{
    Vc_ASSERT((*this >= 0.).isFull());
    return Detail::exponent(d.v());
}
// }}}1
// Random {{{1
static Vc_ALWAYS_INLINE __m256i _doRandomStep()
{
#ifdef Vc_IMPL_AVX2
    AVX2::uint_v state0(&Common::RandomState[0]);
    AVX2::uint_v state1(&Common::RandomState[AVX2::uint_v::Size]);
    (state1 * 0xdeece66du + 11).store(&Common::RandomState[AVX2::uint_v::Size]);
    AVX2::uint_v(Detail::xor_((state0 * 0xdeece66du + 11).data(), _mm256_srli_epi32(state1.data(), 16))).store(&Common::RandomState[0]);
    return state0.data();
#else
    SSE::uint_v state0(&Common::RandomState[0]);
    SSE::uint_v state1(&Common::RandomState[SSE::uint_v::Size]);
    SSE::uint_v state2(&Common::RandomState[2 * SSE::uint_v::Size]);
    SSE::uint_v state3(&Common::RandomState[3 * SSE::uint_v::Size]);
    (state2 * 0xdeece66du + 11).store(&Common::RandomState[2 * SSE::uint_v::Size]);
    (state3 * 0xdeece66du + 11).store(&Common::RandomState[3 * SSE::uint_v::Size]);
    SSE::uint_v(Detail::xor_((state0 * 0xdeece66du + 11).data(), _mm_srli_epi32(state2.data(), 16))).store(&Common::RandomState[0]);
    SSE::uint_v(Detail::xor_((state1 * 0xdeece66du + 11).data(), _mm_srli_epi32(state3.data(), 16))).store(&Common::RandomState[SSE::uint_v::Size]);
    return AVX::concat(state0.data(), state1.data());
#endif
}

#ifdef Vc_IMPL_AVX2
template<typename T> Vc_ALWAYS_INLINE AVX2::Vector<T> Vector<T, VectorAbi::Avx>::Random()
{
    return {_doRandomStep()};
}
#endif

template <> Vc_ALWAYS_INLINE AVX2::float_v AVX2::float_v::Random()
{
    return HT::sub(Detail::or_(_cast(AVX::srli_epi32<2>(_doRandomStep())), HT::one()),
                   HT::one());
}

template<> Vc_ALWAYS_INLINE AVX2::double_v AVX2::double_v::Random()
{
    const __m256i state = Detail::load<AlignedTag>(&Common::RandomState[0], __m256i());
    for (size_t k = 0; k < 8; k += 2) {
        typedef unsigned long long uint64 Vc_MAY_ALIAS;
        const uint64 stateX = *reinterpret_cast<const uint64 *>(&Common::RandomState[k]);
        *reinterpret_cast<uint64 *>(&Common::RandomState[k]) = (stateX * 0x5deece66dull + 11);
    }
    return HT::sub(Detail::or_(_cast(AVX::srli_epi64<12>(state)), HT::one()), HT::one());
}
// }}}1
// shifted / rotated {{{1
template<typename T> Vc_INTRINSIC AVX2::Vector<T> Vector<T, VectorAbi::Avx>::shifted(int amount) const
{
    return Detail::shifted<EntryType>(d.v(), amount);
}

template <typename VectorType>
Vc_INTRINSIC Vc_CONST VectorType shifted_shortcut(VectorType left, VectorType right, Common::WidthT<__m128>)
{
    return Mem::shuffle<X2, X3, Y0, Y1>(left, right);
}
template <typename VectorType>
Vc_INTRINSIC Vc_CONST VectorType shifted_shortcut(VectorType left, VectorType right, Common::WidthT<__m256>)
{
    return Mem::shuffle128<X1, Y0>(left, right);
}

template<typename T> Vc_INTRINSIC AVX2::Vector<T> Vector<T, VectorAbi::Avx>::shifted(int amount, Vector shiftIn) const
{
#ifdef __GNUC__
    if (__builtin_constant_p(amount)) {
        switch (amount * 2) {
        case int(Size):
            return shifted_shortcut(d.v(), shiftIn.d.v(), WidthT());
        case -int(Size):
            return shifted_shortcut(shiftIn.d.v(), d.v(), WidthT());
        }
    }
#endif
    return shifted(amount) | (amount > 0 ?
                              shiftIn.shifted(amount - Size) :
                              shiftIn.shifted(Size + amount));
}
template<typename T> Vc_INTRINSIC AVX2::Vector<T> Vector<T, VectorAbi::Avx>::rotated(int amount) const
{
    return Detail::rotated<EntryType, size()>(d.v(), amount);
}
// sorted {{{1
template <typename T>
Vc_ALWAYS_INLINE Vc_PURE Vector<T, VectorAbi::Avx> Vector<T, VectorAbi::Avx>::sorted()
    const
{
    return Detail::sorted(*this);
}
// interleaveLow/-High {{{1
template <> Vc_INTRINSIC AVX2::double_v AVX2::double_v::interleaveLow(AVX2::double_v x) const
{
    return Mem::shuffle128<X0, Y0>(_mm256_unpacklo_pd(data(), x.data()),
                                   _mm256_unpackhi_pd(data(), x.data()));
}
template <> Vc_INTRINSIC AVX2::double_v AVX2::double_v::interleaveHigh(AVX2::double_v x) const
{
    return Mem::shuffle128<X1, Y1>(_mm256_unpacklo_pd(data(), x.data()),
                                   _mm256_unpackhi_pd(data(), x.data()));
}
template <> Vc_INTRINSIC AVX2::float_v AVX2::float_v::interleaveLow(AVX2::float_v x) const
{
    return Mem::shuffle128<X0, Y0>(_mm256_unpacklo_ps(data(), x.data()),
                                   _mm256_unpackhi_ps(data(), x.data()));
}
template <> Vc_INTRINSIC AVX2::float_v AVX2::float_v::interleaveHigh(AVX2::float_v x) const
{
    return Mem::shuffle128<X1, Y1>(_mm256_unpacklo_ps(data(), x.data()),
                                   _mm256_unpackhi_ps(data(), x.data()));
}
#ifdef Vc_IMPL_AVX2
template <> Vc_INTRINSIC    AVX2::int_v    AVX2::int_v::interleaveLow (   AVX2::int_v x) const {
    return Mem::shuffle128<X0, Y0>(_mm256_unpacklo_epi32(data(), x.data()),
                                   _mm256_unpackhi_epi32(data(), x.data()));
}
template <> Vc_INTRINSIC    AVX2::int_v    AVX2::int_v::interleaveHigh(   AVX2::int_v x) const {
    return Mem::shuffle128<X1, Y1>(_mm256_unpacklo_epi32(data(), x.data()),
                                   _mm256_unpackhi_epi32(data(), x.data()));
}
template <> Vc_INTRINSIC   AVX2::uint_v   AVX2::uint_v::interleaveLow (  AVX2::uint_v x) const {
    return Mem::shuffle128<X0, Y0>(_mm256_unpacklo_epi32(data(), x.data()),
                                   _mm256_unpackhi_epi32(data(), x.data()));
}
template <> Vc_INTRINSIC   AVX2::uint_v   AVX2::uint_v::interleaveHigh(  AVX2::uint_v x) const {
    return Mem::shuffle128<X1, Y1>(_mm256_unpacklo_epi32(data(), x.data()),
                                   _mm256_unpackhi_epi32(data(), x.data()));
}
template <> Vc_INTRINSIC  AVX2::short_v  AVX2::short_v::interleaveLow ( AVX2::short_v x) const {
    return Mem::shuffle128<X0, Y0>(_mm256_unpacklo_epi16(data(), x.data()),
                                   _mm256_unpackhi_epi16(data(), x.data()));
}
template <> Vc_INTRINSIC  AVX2::short_v  AVX2::short_v::interleaveHigh( AVX2::short_v x) const {
    return Mem::shuffle128<X1, Y1>(_mm256_unpacklo_epi16(data(), x.data()),
                                   _mm256_unpackhi_epi16(data(), x.data()));
}
template <> Vc_INTRINSIC AVX2::ushort_v AVX2::ushort_v::interleaveLow (AVX2::ushort_v x) const {
    return Mem::shuffle128<X0, Y0>(_mm256_unpacklo_epi16(data(), x.data()),
                                   _mm256_unpackhi_epi16(data(), x.data()));
}
template <> Vc_INTRINSIC AVX2::ushort_v AVX2::ushort_v::interleaveHigh(AVX2::ushort_v x) const {
    return Mem::shuffle128<X1, Y1>(_mm256_unpacklo_epi16(data(), x.data()),
                                   _mm256_unpackhi_epi16(data(), x.data()));
}
#endif
// generate {{{1
template <> template <typename G> Vc_INTRINSIC AVX2::double_v AVX2::double_v::generate(G gen)
{
    const auto tmp0 = gen(0);
    const auto tmp1 = gen(1);
    const auto tmp2 = gen(2);
    const auto tmp3 = gen(3);
    return _mm256_setr_pd(tmp0, tmp1, tmp2, tmp3);
}
template <> template <typename G> Vc_INTRINSIC AVX2::float_v AVX2::float_v::generate(G gen)
{
    const auto tmp0 = gen(0);
    const auto tmp1 = gen(1);
    const auto tmp2 = gen(2);
    const auto tmp3 = gen(3);
    const auto tmp4 = gen(4);
    const auto tmp5 = gen(5);
    const auto tmp6 = gen(6);
    const auto tmp7 = gen(7);
    return _mm256_setr_ps(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7);
}
#ifdef Vc_IMPL_AVX2
template <> template <typename G> Vc_INTRINSIC AVX2::int_v AVX2::int_v::generate(G gen)
{
    const auto tmp0 = gen(0);
    const auto tmp1 = gen(1);
    const auto tmp2 = gen(2);
    const auto tmp3 = gen(3);
    const auto tmp4 = gen(4);
    const auto tmp5 = gen(5);
    const auto tmp6 = gen(6);
    const auto tmp7 = gen(7);
    return _mm256_setr_epi32(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7);
}
template <> template <typename G> Vc_INTRINSIC AVX2::uint_v AVX2::uint_v::generate(G gen)
{
    const auto tmp0 = gen(0);
    const auto tmp1 = gen(1);
    const auto tmp2 = gen(2);
    const auto tmp3 = gen(3);
    const auto tmp4 = gen(4);
    const auto tmp5 = gen(5);
    const auto tmp6 = gen(6);
    const auto tmp7 = gen(7);
    return _mm256_setr_epi32(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7);
}
template <> template <typename G> Vc_INTRINSIC AVX2::short_v AVX2::short_v::generate(G gen)
{
    const auto tmp0 = gen(0);
    const auto tmp1 = gen(1);
    const auto tmp2 = gen(2);
    const auto tmp3 = gen(3);
    const auto tmp4 = gen(4);
    const auto tmp5 = gen(5);
    const auto tmp6 = gen(6);
    const auto tmp7 = gen(7);
    const auto tmp8 = gen(8);
    const auto tmp9 = gen(9);
    const auto tmp10 = gen(10);
    const auto tmp11 = gen(11);
    const auto tmp12 = gen(12);
    const auto tmp13 = gen(13);
    const auto tmp14 = gen(14);
    const auto tmp15 = gen(15);
    return _mm256_setr_epi16(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12, tmp13, tmp14, tmp15);
}
template <> template <typename G> Vc_INTRINSIC AVX2::ushort_v AVX2::ushort_v::generate(G gen)
{
    const auto tmp0 = gen(0);
    const auto tmp1 = gen(1);
    const auto tmp2 = gen(2);
    const auto tmp3 = gen(3);
    const auto tmp4 = gen(4);
    const auto tmp5 = gen(5);
    const auto tmp6 = gen(6);
    const auto tmp7 = gen(7);
    const auto tmp8 = gen(8);
    const auto tmp9 = gen(9);
    const auto tmp10 = gen(10);
    const auto tmp11 = gen(11);
    const auto tmp12 = gen(12);
    const auto tmp13 = gen(13);
    const auto tmp14 = gen(14);
    const auto tmp15 = gen(15);
    return _mm256_setr_epi16(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12, tmp13, tmp14, tmp15);
}
#endif

// permutation via operator[] {{{1
template <> Vc_INTRINSIC Vc_PURE AVX2::double_v AVX2::double_v::operator[](Permutation::ReversedTag) const
{
    return Mem::permute128<X1, X0>(Mem::permute<X1, X0, X3, X2>(d.v()));
}
template <> Vc_INTRINSIC Vc_PURE AVX2::float_v AVX2::float_v::operator[](Permutation::ReversedTag) const
{
    return Mem::permute128<X1, X0>(Mem::permute<X3, X2, X1, X0>(d.v()));
}
#ifdef Vc_IMPL_AVX2
template <>
Vc_INTRINSIC Vc_PURE AVX2::int_v AVX2::int_v::operator[](Permutation::ReversedTag) const
{
    return Mem::permute128<X1, X0>(Mem::permute<X3, X2, X1, X0>(d.v()));
}
template <>
Vc_INTRINSIC Vc_PURE AVX2::uint_v AVX2::uint_v::operator[](Permutation::ReversedTag) const
{
    return Mem::permute128<X1, X0>(Mem::permute<X3, X2, X1, X0>(d.v()));
}
template <>
Vc_INTRINSIC Vc_PURE AVX2::short_v AVX2::short_v::operator[](
    Permutation::ReversedTag) const
{
    return Mem::permute128<X1, X0>(AVX::avx_cast<__m256i>(Mem::shuffle<X1, Y0, X3, Y2>(
        AVX::avx_cast<__m256d>(Mem::permuteHi<X7, X6, X5, X4>(d.v())),
        AVX::avx_cast<__m256d>(Mem::permuteLo<X3, X2, X1, X0>(d.v())))));
}
template <>
Vc_INTRINSIC Vc_PURE AVX2::ushort_v AVX2::ushort_v::operator[](
    Permutation::ReversedTag) const
{
    return Mem::permute128<X1, X0>(AVX::avx_cast<__m256i>(Mem::shuffle<X1, Y0, X3, Y2>(
        AVX::avx_cast<__m256d>(Mem::permuteHi<X7, X6, X5, X4>(d.v())),
        AVX::avx_cast<__m256d>(Mem::permuteLo<X3, X2, X1, X0>(d.v())))));
}
#endif
template <> Vc_INTRINSIC AVX2::float_v AVX2::float_v::operator[](const IndexType &/*perm*/) const
{
    // TODO
    return *this;
#ifdef Vc_IMPL_AVX2
#else
    /*
    const int_m cross128 = AVX::concat(_mm_cmpgt_epi32(AVX::lo128(perm.data()), _mm_set1_epi32(3)),
                                  _mm_cmplt_epi32(AVX::hi128(perm.data()), _mm_set1_epi32(4)));
    if (cross128.isNotEmpty()) {
    AVX2::float_v x = _mm256_permutevar_ps(d.v(), perm.data());
        x(cross128) = _mm256_permutevar_ps(Mem::permute128<X1, X0>(d.v()), perm.data());
        return x;
    } else {
    */
#endif
}

// reversed {{{1
template <typename T>
Vc_INTRINSIC Vc_PURE Vector<T, VectorAbi::Avx> Vector<T, VectorAbi::Avx>::reversed() const
{
    return (*this)[Permutation::Reversed];
}

// broadcast from constexpr index {{{1
template <> template <int Index> Vc_INTRINSIC AVX2::float_v AVX2::float_v::broadcast() const
{
    constexpr VecPos Inner = static_cast<VecPos>(Index & 0x3);
    constexpr VecPos Outer = static_cast<VecPos>((Index & 0x4) / 4);
    return Mem::permute<Inner, Inner, Inner, Inner>(Mem::permute128<Outer, Outer>(d.v()));
}
template <> template <int Index> Vc_INTRINSIC AVX2::double_v AVX2::double_v::broadcast() const
{
    constexpr VecPos Inner = static_cast<VecPos>(Index & 0x1);
    constexpr VecPos Outer = static_cast<VecPos>((Index & 0x2) / 2);
    return Mem::permute<Inner, Inner>(Mem::permute128<Outer, Outer>(d.v()));
}
// }}}1
}  // namespace Vc

// vim: foldmethod=marker
