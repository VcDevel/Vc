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

#include "../common/x86_prefetches.h"
#include "../common/gatherimplementation.h"
#include "../common/scatterimplementation.h"
#include "limits.h"
#include "const.h"
#include "../common/set.h"
#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace Vc_AVX_NAMESPACE
{

///////////////////////////////////////////////////////////////////////////////////////////
// constants {{{1
template<typename T> Vc_ALWAYS_INLINE Vector<T>::Vector(VectorSpecialInitializerZero::ZEnum) : d(HT::zero()) {}
template<typename T> Vc_ALWAYS_INLINE Vector<T>::Vector(VectorSpecialInitializerOne::OEnum) : d(HT::one()) {}
template<typename T> Vc_ALWAYS_INLINE Vector<T>::Vector(VectorSpecialInitializerIndexesFromZero::IEnum)
    : d(HV::template load<AlignedTag>(IndexesFromZeroData<T>::address())) {}

template<> Vc_ALWAYS_INLINE float_v::Vector(VectorSpecialInitializerIndexesFromZero::IEnum)
    : d(StaticCastHelper<int, float>::cast(int_v::IndexesFromZero().data())) {}
template<> Vc_ALWAYS_INLINE double_v::Vector(VectorSpecialInitializerIndexesFromZero::IEnum)
    : d(_mm256_cvtepi32_pd(_mm_load_si128(reinterpret_cast<const __m128i *>(_IndexesFromZero32)))) {}

///////////////////////////////////////////////////////////////////////////////////////////
// load member functions {{{1
// LoadHelper {{{2
template<typename DstT, typename SrcT, typename Flags> struct LoadHelper;

// equal types {{{2
template <typename T, typename Flags> struct LoadHelper<T, T, Flags>
{
    using VectorType = typename Vector<T>::VectorType;
    using HV = VectorHelper<VectorType>;
    static VectorType load(const T *mem, Flags flags)
    {
        Common::handleLoadPrefetches(mem, flags);
        return HV::template load<Flags>(mem);
    }
};

// float {{{2
template<typename Flags> struct LoadHelper<float, double, Flags> {
    static m256 load(const double *mem, Flags)
    {
        return concat(_mm256_cvtpd_ps(VectorHelper<m256d>::load<Flags>(&mem[0])),
                      _mm256_cvtpd_ps(VectorHelper<m256d>::load<Flags>(&mem[4])));
    }
};
template<typename Flags> struct LoadHelper<float, unsigned int, Flags> {
    static m256 load(const unsigned int *mem, Flags)
    {
        return StaticCastHelper<unsigned int, float>::cast(VectorHelper<m256i>::load<Flags>(mem));
    }
};
template<typename Flags> struct LoadHelper<float, int, Flags> {
    static m256 load(const int *mem, Flags)
    {
        return StaticCastHelper<int, float>::cast(VectorHelper<m256i>::load<Flags>(mem));
    }
};
template<typename Flags> struct LoadHelper<float, unsigned short, Flags> {
    static m256 load(const unsigned short *mem, Flags)
    {
        return StaticCastHelper<unsigned short, float>::cast(VectorHelper<m128i>::load<Flags>(mem));
    }
};
template<typename Flags> struct LoadHelper<float, short, Flags> {
    static m256 load(const short *mem, Flags)
    {
        return StaticCastHelper<short, float>::cast(VectorHelper<m128i>::load<Flags>(mem));
    }
};
template<typename Flags> struct LoadHelper<float, unsigned char, Flags> {
    static m256 load(const unsigned char *mem, Flags f)
    {
        return StaticCastHelper<unsigned int, float>::cast(LoadHelper<unsigned int, unsigned char, Flags>::load(mem, f));
    }
};
template<typename Flags> struct LoadHelper<float, signed char, Flags> {
    static m256 load(const signed char *mem, Flags f)
    {
        return StaticCastHelper<int, float>::cast(LoadHelper<int, signed char, Flags>::load(mem, f));
    }
};

// int {{{2
template<typename Flags> struct LoadHelper<int, unsigned int, Flags> {
    static m256i load(const unsigned int *mem, Flags)
    {
        return VectorHelper<m256i>::load<Flags>(mem);
    }
};
template<typename Flags> struct LoadHelper<int, unsigned short, Flags> {
    static m256i load(const unsigned short *mem, Flags)
    {
        return StaticCastHelper<unsigned short, unsigned int>::cast(VectorHelper<m128i>::load<Flags>(mem));
    }
};
template<typename Flags> struct LoadHelper<int, short, Flags> {
    static m256i load(const short *mem, Flags)
    {
        return StaticCastHelper<short, int>::cast(VectorHelper<m128i>::load<Flags>(mem));
    }
};
template<typename Flags> struct LoadHelper<int, unsigned char, Flags> {
    static m256i load(const unsigned char *mem, Flags)
    {
        // the only available streaming load loads 16 bytes - twice as much as we need => can't use
        // it, or we risk an out-of-bounds read and an unaligned load exception
        const m128i epu8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(mem));
        const m128i epu16 = _mm_cvtepu8_epi16(epu8);
        return StaticCastHelper<unsigned short, unsigned int>::cast(epu16);
    }
};
template<typename Flags> struct LoadHelper<int, signed char, Flags> {
    static m256i load(const signed char *mem, Flags)
    {
        // the only available streaming load loads 16 bytes - twice as much as we need => can't use
        // it, or we risk an out-of-bounds read and an unaligned load exception
        const m128i epi8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(mem));
        const m128i epi16 = _mm_cvtepi8_epi16(epi8);
        return StaticCastHelper<short, int>::cast(epi16);
    }
};

// unsigned int {{{2
template<typename Flags> struct LoadHelper<unsigned int, unsigned short, Flags> {
    static m256i load(const unsigned short *mem, Flags)
    {
        return StaticCastHelper<unsigned short, unsigned int>::cast(VectorHelper<m128i>::load<Flags>(mem));
    }
};
template<typename Flags> struct LoadHelper<unsigned int, unsigned char, Flags> {
    static m256i load(const unsigned char *mem, Flags)
    {
        // the only available streaming load loads 16 bytes - twice as much as we need => can't use
        // it, or we risk an out-of-bounds read and an unaligned load exception
        const m128i epu8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(mem));
        const m128i epu16 = _mm_cvtepu8_epi16(epu8);
        return StaticCastHelper<unsigned short, unsigned int>::cast(epu16);
    }
};

// short {{{2
template<typename Flags> struct LoadHelper<short, unsigned short, Flags> {
    static m128i load(const unsigned short *mem, Flags)
    {
        return StaticCastHelper<unsigned short, short>::cast(VectorHelper<m128i>::load<Flags>(mem));
    }
};
template<typename Flags> struct LoadHelper<short, unsigned char, Flags> {
    static m128i load(const unsigned char *mem, Flags)
    {
        // the only available streaming load loads 16 bytes - twice as much as we need => can't use
        // it, or we risk an out-of-bounds read and an unaligned load exception
        const m128i epu8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(mem));
        return _mm_cvtepu8_epi16(epu8);
    }
};
template<typename Flags> struct LoadHelper<short, signed char, Flags> {
    static m128i load(const signed char *mem, Flags)
    {
        // the only available streaming load loads 16 bytes - twice as much as we need => can't use
        // it, or we risk an out-of-bounds read and an unaligned load exception
        const m128i epi8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(mem));
        return _mm_cvtepi8_epi16(epi8);
    }
};

// unsigned short {{{2
template<typename Flags> struct LoadHelper<unsigned short, unsigned char, Flags> {
    static m128i load(const unsigned char *mem, Flags)
    {
        // the only available streaming load loads 16 bytes - twice as much as we need => can't use
        // it, or we risk an out-of-bounds read and an unaligned load exception
        const m128i epu8 = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(mem));
        return _mm_cvtepu8_epi16(epu8);
    }
};

// general load, implemented via LoadHelper {{{2
template <typename DstT>
template <typename SrcT,
          typename Flags,
          typename>
Vc_INTRINSIC void Vector<DstT>::load(const SrcT *mem, Flags flags)
{
    Common::handleLoadPrefetches(mem, flags);
    d.v() = LoadHelper<DstT, SrcT, Flags>::load(mem, flags);
}

///////////////////////////////////////////////////////////////////////////////////////////
// zeroing {{{1
template<typename T> Vc_INTRINSIC void Vector<T>::setZero()
{
    data() = HV::zero();
}
template<typename T> Vc_INTRINSIC void Vector<T>::setZero(const Mask &k)
{
    data() = HV::andnot_(avx_cast<VectorType>(k.data()), data());
}
template<typename T> Vc_INTRINSIC void Vector<T>::setZeroInverted(const Mask &k)
{
    data() = HV::and_(avx_cast<VectorType>(k.data()), data());
}

template<> Vc_INTRINSIC void Vector<double>::setQnan()
{
    data() = setallone_pd();
}
template<> Vc_INTRINSIC void Vector<double>::setQnan(MaskArg k)
{
    data() = _mm256_or_pd(data(), k.dataD());
}
template<> Vc_INTRINSIC void Vector<float>::setQnan()
{
    data() = setallone_ps();
}
template<> Vc_INTRINSIC void Vector<float>::setQnan(MaskArg k)
{
    data() = _mm256_or_ps(data(), k.data());
}

///////////////////////////////////////////////////////////////////////////////////////////
// stores {{{1
template <typename T>
template <typename U,
          typename Flags,
          typename>
Vc_INTRINSIC void Vector<T>::store(U *mem, Flags flags) const
{
    Common::handleStorePrefetches(mem, flags);
    HV::template store<Flags>(mem, data());
}

template <typename T>
template <typename U,
          typename Flags,
          typename>
Vc_INTRINSIC void Vector<T>::store(U *mem, Mask mask, Flags flags) const
{
    Common::handleStorePrefetches(mem, flags);
    HV::template store<Flags>(mem, data(), avx_cast<VectorType>(mask.data()));
}

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
// swizzles {{{1
template<typename T> Vc_INTRINSIC const Vector<T> Vc_PURE &Vector<T>::abcd() const { return *this; }
template<typename T> Vc_INTRINSIC const Vector<T> Vc_PURE  Vector<T>::cdab() const { return Mem::permute<X2, X3, X0, X1>(data()); }
template<typename T> Vc_INTRINSIC const Vector<T> Vc_PURE  Vector<T>::badc() const { return Mem::permute<X1, X0, X3, X2>(data()); }
template<typename T> Vc_INTRINSIC const Vector<T> Vc_PURE  Vector<T>::aaaa() const { return Mem::permute<X0, X0, X0, X0>(data()); }
template<typename T> Vc_INTRINSIC const Vector<T> Vc_PURE  Vector<T>::bbbb() const { return Mem::permute<X1, X1, X1, X1>(data()); }
template<typename T> Vc_INTRINSIC const Vector<T> Vc_PURE  Vector<T>::cccc() const { return Mem::permute<X2, X2, X2, X2>(data()); }
template<typename T> Vc_INTRINSIC const Vector<T> Vc_PURE  Vector<T>::dddd() const { return Mem::permute<X3, X3, X3, X3>(data()); }
template<typename T> Vc_INTRINSIC const Vector<T> Vc_PURE  Vector<T>::bcad() const { return Mem::permute<X1, X2, X0, X3>(data()); }
template<typename T> Vc_INTRINSIC const Vector<T> Vc_PURE  Vector<T>::bcda() const { return Mem::permute<X1, X2, X3, X0>(data()); }
template<typename T> Vc_INTRINSIC const Vector<T> Vc_PURE  Vector<T>::dabc() const { return Mem::permute<X3, X0, X1, X2>(data()); }
template<typename T> Vc_INTRINSIC const Vector<T> Vc_PURE  Vector<T>::acbd() const { return Mem::permute<X0, X2, X1, X3>(data()); }
template<typename T> Vc_INTRINSIC const Vector<T> Vc_PURE  Vector<T>::dbca() const { return Mem::permute<X3, X1, X2, X0>(data()); }
template<typename T> Vc_INTRINSIC const Vector<T> Vc_PURE  Vector<T>::dcba() const { return Mem::permute<X3, X2, X1, X0>(data()); }

template<> Vc_INTRINSIC const double_v Vc_PURE Vector<double>::cdab() const { return Mem::shuffle128<X1, X0>(data(), data()); }
template<> Vc_INTRINSIC const double_v Vc_PURE Vector<double>::badc() const { return Mem::permute<X1, X0, X3, X2>(data()); }
template<> Vc_INTRINSIC const double_v Vc_PURE Vector<double>::aaaa() const { const double &tmp = d.m(0); return _mm256_broadcast_sd(&tmp); }
template<> Vc_INTRINSIC const double_v Vc_PURE Vector<double>::bbbb() const { const double &tmp = d.m(1); return _mm256_broadcast_sd(&tmp); }
template<> Vc_INTRINSIC const double_v Vc_PURE Vector<double>::cccc() const { const double &tmp = d.m(2); return _mm256_broadcast_sd(&tmp); }
template<> Vc_INTRINSIC const double_v Vc_PURE Vector<double>::dddd() const { const double &tmp = d.m(3); return _mm256_broadcast_sd(&tmp); }
template<> Vc_INTRINSIC const double_v Vc_PURE Vector<double>::bcad() const { return Mem::shuffle<X1, Y0, X2, Y3>(Mem::shuffle128<X0, X0>(data(), data()), Mem::shuffle128<X1, X1>(data(), data())); }
template<> Vc_INTRINSIC const double_v Vc_PURE Vector<double>::bcda() const { return Mem::shuffle<X1, Y0, X3, Y2>(data(), Mem::shuffle128<X1, X0>(data(), data())); }
template<> Vc_INTRINSIC const double_v Vc_PURE Vector<double>::dabc() const { return Mem::shuffle<X1, Y0, X3, Y2>(Mem::shuffle128<X1, X0>(data(), data()), data()); }
template<> Vc_INTRINSIC const double_v Vc_PURE Vector<double>::acbd() const { return Mem::shuffle<X0, Y0, X3, Y3>(Mem::shuffle128<X0, X0>(data(), data()), Mem::shuffle128<X1, X1>(data(), data())); }
template<> Vc_INTRINSIC const double_v Vc_PURE Vector<double>::dbca() const { return Mem::shuffle<X1, Y1, X2, Y2>(Mem::shuffle128<X1, X1>(data(), data()), Mem::shuffle128<X0, X0>(data(), data())); }
template<> Vc_INTRINSIC const double_v Vc_PURE Vector<double>::dcba() const { return cdab().badc(); }

#define VC_SWIZZLES_16BIT_IMPL(T) \
template<> Vc_INTRINSIC const Vector<T> Vc_PURE Vector<T>::cdab() const { return Mem::permute<X2, X3, X0, X1, X6, X7, X4, X5>(data()); } \
template<> Vc_INTRINSIC const Vector<T> Vc_PURE Vector<T>::badc() const { return Mem::permute<X1, X0, X3, X2, X5, X4, X7, X6>(data()); } \
template<> Vc_INTRINSIC const Vector<T> Vc_PURE Vector<T>::aaaa() const { return Mem::permute<X0, X0, X0, X0, X4, X4, X4, X4>(data()); } \
template<> Vc_INTRINSIC const Vector<T> Vc_PURE Vector<T>::bbbb() const { return Mem::permute<X1, X1, X1, X1, X5, X5, X5, X5>(data()); } \
template<> Vc_INTRINSIC const Vector<T> Vc_PURE Vector<T>::cccc() const { return Mem::permute<X2, X2, X2, X2, X6, X6, X6, X6>(data()); } \
template<> Vc_INTRINSIC const Vector<T> Vc_PURE Vector<T>::dddd() const { return Mem::permute<X3, X3, X3, X3, X7, X7, X7, X7>(data()); } \
template<> Vc_INTRINSIC const Vector<T> Vc_PURE Vector<T>::bcad() const { return Mem::permute<X1, X2, X0, X3, X5, X6, X4, X7>(data()); } \
template<> Vc_INTRINSIC const Vector<T> Vc_PURE Vector<T>::bcda() const { return Mem::permute<X1, X2, X3, X0, X5, X6, X7, X4>(data()); } \
template<> Vc_INTRINSIC const Vector<T> Vc_PURE Vector<T>::dabc() const { return Mem::permute<X3, X0, X1, X2, X7, X4, X5, X6>(data()); } \
template<> Vc_INTRINSIC const Vector<T> Vc_PURE Vector<T>::acbd() const { return Mem::permute<X0, X2, X1, X3, X4, X6, X5, X7>(data()); } \
template<> Vc_INTRINSIC const Vector<T> Vc_PURE Vector<T>::dbca() const { return Mem::permute<X3, X1, X2, X0, X7, X5, X6, X4>(data()); } \
template<> Vc_INTRINSIC const Vector<T> Vc_PURE Vector<T>::dcba() const { return Mem::permute<X3, X2, X1, X0, X7, X6, X5, X4>(data()); }
VC_SWIZZLES_16BIT_IMPL(short)
VC_SWIZZLES_16BIT_IMPL(unsigned short)
#undef VC_SWIZZLES_16BIT_IMPL

///////////////////////////////////////////////////////////////////////////////////////////
// division {{{1
template<typename T> inline Vector<T> &Vector<T>::operator/=(EntryType x)
{
    if (HasVectorDivision) {
        return operator/=(Vector<T>(x));
    }
    for_all_vector_entries(i,
            d.m(i) /= x;
            );
    return *this;
}
// per default fall back to scalar division
template<typename T> inline Vector<T> &Vector<T>::operator/=(VC_ALIGNED_PARAMETER(Vector<T>) x)
{
    for_all_vector_entries(i,
            d.m(i) /= x.d.m(i);
            );
    return *this;
}

template<typename T> inline Vc_PURE Vector<T> Vector<T>::operator/(VC_ALIGNED_PARAMETER(Vector<T>) x) const
{
    Vector<T> r;
    for_all_vector_entries(i,
            r.d.set(i, d.m(i) / x.d.m(i));
            );
    return r;
}
// specialize division on type
static Vc_INTRINSIC m256i Vc_CONST divInt(param256i a, param256i b) {
    const m256d lo1 = _mm256_cvtepi32_pd(lo128(a));
    const m256d lo2 = _mm256_cvtepi32_pd(lo128(b));
    const m256d hi1 = _mm256_cvtepi32_pd(hi128(a));
    const m256d hi2 = _mm256_cvtepi32_pd(hi128(b));
    return concat(
            _mm256_cvttpd_epi32(_mm256_div_pd(lo1, lo2)),
            _mm256_cvttpd_epi32(_mm256_div_pd(hi1, hi2))
            );
}
template<> inline Vector<int> &Vector<int>::operator/=(VC_ALIGNED_PARAMETER(Vector<int>) x)
{
    d.v() = divInt(d.v(), x.d.v());
    return *this;
}
template<> inline Vector<int> Vc_PURE Vector<int>::operator/(VC_ALIGNED_PARAMETER(Vector<int>) x) const
{
    return divInt(d.v(), x.d.v());
}
static inline m256i Vc_CONST divUInt(param256i a, param256i b) {
    // SSE/AVX only has signed int conversion to doubles. Therefore we first adjust the input before
    // conversion and take the adjustment back after the conversion.
    // It could be argued that for b this is not really important because division by a b >= 2^31 is
    // useless. But for full correctness it cannot be ignored.
#ifdef VC_IMPL_AVX2
    const m256i aa = add_epi32(a, set1_epi32(-2147483648));
    const m256i bb = add_epi32(b, set1_epi32(-2147483648));
    const m256d loa = _mm256_add_pd(_mm256_cvtepi32_pd(lo128(aa)), set1_pd(2147483648.));
    const m256d hia = _mm256_add_pd(_mm256_cvtepi32_pd(hi128(aa)), set1_pd(2147483648.));
    const m256d lob = _mm256_add_pd(_mm256_cvtepi32_pd(lo128(bb)), set1_pd(2147483648.));
    const m256d hib = _mm256_add_pd(_mm256_cvtepi32_pd(hi128(bb)), set1_pd(2147483648.));
#else
    const auto a0 = _mm_add_epi32(lo128(a), _mm_set1_epi32(-2147483648));
    const auto a1 = _mm_add_epi32(hi128(a), _mm_set1_epi32(-2147483648));
    const auto b0 = _mm_add_epi32(lo128(b), _mm_set1_epi32(-2147483648));
    const auto b1 = _mm_add_epi32(hi128(b), _mm_set1_epi32(-2147483648));
    const m256d loa = _mm256_add_pd(_mm256_cvtepi32_pd(a0), set1_pd(2147483648.));
    const m256d hia = _mm256_add_pd(_mm256_cvtepi32_pd(a1), set1_pd(2147483648.));
    const m256d lob = _mm256_add_pd(_mm256_cvtepi32_pd(b0), set1_pd(2147483648.));
    const m256d hib = _mm256_add_pd(_mm256_cvtepi32_pd(b1), set1_pd(2147483648.));
#endif
    // there is one remaining problem: a >= 2^31 and b == 1
    // in that case the return value would be 2^31
    return avx_cast<m256i>(_mm256_blendv_ps(avx_cast<m256>(concat(
                        _mm256_cvttpd_epi32(_mm256_div_pd(loa, lob)),
                        _mm256_cvttpd_epi32(_mm256_div_pd(hia, hib))
                        )), avx_cast<m256>(a), avx_cast<m256>(
                            cmpeq_epi32(b, setone_epi32())
                            )));
}
template<> Vc_ALWAYS_INLINE Vector<unsigned int> &Vector<unsigned int>::operator/=(VC_ALIGNED_PARAMETER(Vector<unsigned int>) x)
{
    d.v() = divUInt(d.v(), x.d.v());
    return *this;
}
template<> Vc_ALWAYS_INLINE Vector<unsigned int> Vc_PURE Vector<unsigned int>::operator/(VC_ALIGNED_PARAMETER(Vector<unsigned int>) x) const
{
    return divUInt(d.v(), x.d.v());
}
template<typename T> static inline m128i Vc_CONST divShort(param128i a, param128i b)
{
    const m256 r = _mm256_div_ps(StaticCastHelper<T, float>::cast(a),
            StaticCastHelper<T, float>::cast(b));
    return StaticCastHelper<float, T>::cast(r);
}
template<> Vc_ALWAYS_INLINE Vector<short> &Vector<short>::operator/=(VC_ALIGNED_PARAMETER(Vector<short>) x)
{
    d.v() = divShort<short>(d.v(), x.d.v());
    return *this;
}
template<> Vc_ALWAYS_INLINE Vector<short> Vc_PURE Vector<short>::operator/(VC_ALIGNED_PARAMETER(Vector<short>) x) const
{
    return divShort<short>(d.v(), x.d.v());
}
template<> Vc_ALWAYS_INLINE Vector<unsigned short> &Vector<unsigned short>::operator/=(VC_ALIGNED_PARAMETER(Vector<unsigned short>) x)
{
    d.v() = divShort<unsigned short>(d.v(), x.d.v());
    return *this;
}
template<> Vc_ALWAYS_INLINE Vector<unsigned short> Vc_PURE Vector<unsigned short>::operator/(VC_ALIGNED_PARAMETER(Vector<unsigned short>) x) const
{
    return divShort<unsigned short>(d.v(), x.d.v());
}
template<> Vc_INTRINSIC float_v &float_v::operator/=(VC_ALIGNED_PARAMETER(float_v) x)
{
    d.v() = _mm256_div_ps(d.v(), x.d.v());
    return *this;
}
template<> Vc_INTRINSIC float_v Vc_PURE float_v::operator/(VC_ALIGNED_PARAMETER(float_v) x) const
{
    return _mm256_div_ps(d.v(), x.d.v());
}
template<> Vc_INTRINSIC double_v &double_v::operator/=(VC_ALIGNED_PARAMETER(double_v) x)
{
    d.v() = _mm256_div_pd(d.v(), x.d.v());
    return *this;
}
template<> Vc_INTRINSIC double_v Vc_PURE double_v::operator/(VC_ALIGNED_PARAMETER(double_v) x) const
{
    return _mm256_div_pd(d.v(), x.d.v());
}

///////////////////////////////////////////////////////////////////////////////////////////
// integer ops {{{1
template <> inline Vc_PURE int_v int_v::operator%(const int_v &n) const
{
    return HT::sub(
        data(),
        HT::mul(n.data(),
                concat(_mm256_cvttpd_epi32(_mm256_div_pd(_mm256_cvtepi32_pd(lo128(data())),
                                                         _mm256_cvtepi32_pd(lo128(n.data())))),
                       _mm256_cvttpd_epi32(_mm256_div_pd(_mm256_cvtepi32_pd(hi128(data())),
                                                         _mm256_cvtepi32_pd(hi128(n.data())))))));
}
template <> inline Vc_PURE uint_v uint_v::operator%(const uint_v &n) const
{
    auto cvt = [](m128i v) {
        return _mm256_add_pd(
            _mm256_cvtepi32_pd(_mm_sub_epi32(v, _mm_setmin_epi32())),
            set1_pd(1u << 31));
    };
    auto cvt2 = [](m256d v) {
        return m128i(_mm256_cvttpd_epi32(
                                 _mm256_sub_pd(_mm256_floor_pd(v), set1_pd(0x80000000u))));
    };
    return HT::sub(
        data(),
        HT::mul(
            n.data(),
            add_epi32(concat(cvt2(_mm256_div_pd(cvt(lo128(data())), cvt(lo128(n.data())))),
                                    cvt2(_mm256_div_pd(cvt(hi128(data())), cvt(hi128(n.data()))))),
                             set2power31_epu32())));
}
template <> inline Vc_PURE short_v short_v::operator%(const short_v &n) const
{
    return *this - n * static_cast<short_v>(static_cast<float_v>(*this) / static_cast<float_v>(n));
}
template <> inline Vc_PURE ushort_v ushort_v::operator%(const ushort_v &n) const
{
    return *this - n * static_cast<ushort_v>(static_cast<float_v>(*this) / static_cast<float_v>(n));
}

#define OP_IMPL(T, symbol)                                                               \
    template <> Vc_ALWAYS_INLINE Vector<T> &Vector<T>::operator symbol##=(AsArg x)       \
    {                                                                                    \
        Common::unrolled_loop<std::size_t, 0, Size>(                                     \
            [&](std::size_t i) { d.set(i, d.m(i) symbol x.d.m(i)); });                   \
        return *this;                                                                    \
    }                                                                                    \
    template <>                                                                          \
    Vc_ALWAYS_INLINE Vc_PURE Vector<T> Vector<T>::operator symbol(AsArg x) const         \
    {                                                                                    \
        Vector<T> r;                                                                     \
        Common::unrolled_loop<std::size_t, 0, Size>(                                     \
            [&](std::size_t i) { r.d.set(i, d.m(i) symbol x.d.m(i)); });                 \
        return r;                                                                        \
    }
OP_IMPL(int, <<)
OP_IMPL(int, >>)
OP_IMPL(unsigned int, <<)
OP_IMPL(unsigned int, >>)
OP_IMPL(short, <<)
OP_IMPL(short, >>)
OP_IMPL(unsigned short, <<)
OP_IMPL(unsigned short, >>)
#undef OP_IMPL

template<typename T> Vc_ALWAYS_INLINE Vector<T> &Vector<T>::operator>>=(int shift) {
    d.v() = VectorHelper<T>::shiftRight(d.v(), shift);
    return *static_cast<Vector<T> *>(this);
}
template<typename T> Vc_ALWAYS_INLINE Vc_PURE Vector<T> Vector<T>::operator>>(int shift) const {
    return VectorHelper<T>::shiftRight(d.v(), shift);
}
template<typename T> Vc_ALWAYS_INLINE Vector<T> &Vector<T>::operator<<=(int shift) {
    d.v() = VectorHelper<T>::shiftLeft(d.v(), shift);
    return *static_cast<Vector<T> *>(this);
}
template<typename T> Vc_ALWAYS_INLINE Vc_PURE Vector<T> Vector<T>::operator<<(int shift) const {
    return VectorHelper<T>::shiftLeft(d.v(), shift);
}

#define OP_IMPL(T, symbol, fun) \
  template<> Vc_ALWAYS_INLINE Vector<T> &Vector<T>::operator symbol##=(AsArg x) { d.v() = HV::fun(d.v(), x.d.v()); return *this; } \
  template<> Vc_ALWAYS_INLINE Vc_PURE Vector<T>  Vector<T>::operator symbol(AsArg x) const { return Vector<T>(HV::fun(d.v(), x.d.v())); }
  OP_IMPL(int, &, and_)
  OP_IMPL(int, |, or_)
  OP_IMPL(int, ^, xor_)
  OP_IMPL(unsigned int, &, and_)
  OP_IMPL(unsigned int, |, or_)
  OP_IMPL(unsigned int, ^, xor_)
  OP_IMPL(short, &, and_)
  OP_IMPL(short, |, or_)
  OP_IMPL(short, ^, xor_)
  OP_IMPL(unsigned short, &, and_)
  OP_IMPL(unsigned short, |, or_)
  OP_IMPL(unsigned short, ^, xor_)
#ifdef VC_ENABLE_FLOAT_BIT_OPERATORS
  OP_IMPL(float, &, and_)
  OP_IMPL(float, |, or_)
  OP_IMPL(float, ^, xor_)
  OP_IMPL(double, &, and_)
  OP_IMPL(double, |, or_)
  OP_IMPL(double, ^, xor_)
#endif
#undef OP_IMPL

// isNegative {{{1
template<> Vc_INTRINSIC Vc_PURE float_m float_v::isNegative() const
{
    return avx_cast<m256>(srai_epi32<31>(avx_cast<m256i>(_mm256_and_ps(setsignmask_ps(), d.v()))));
}
template<> Vc_INTRINSIC Vc_PURE double_m double_v::isNegative() const
{
    return Mem::permute<X1, X1, X3, X3>(avx_cast<m256>(
                srai_epi32<31>(avx_cast<m256i>(_mm256_and_pd(setsignmask_pd(), d.v())))
                ));
}
// gathers {{{1
template <>
template <typename MT, typename IT>
inline void double_v::gatherImplementation(const MT *mem, IT &&indexes)
{
    d.v() = _mm256_setr_pd(mem[indexes[0]], mem[indexes[1]], mem[indexes[2]], mem[indexes[3]]);
}

template <>
template <typename MT, typename IT>
inline void float_v::gatherImplementation(const MT *mem, IT &&indexes)
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

template <>
template <typename MT, typename IT>
inline void int_v::gatherImplementation(const MT *mem, IT &&indexes)
{
    d.v() = _mm256_setr_epi32(mem[indexes[0]],
                              mem[indexes[1]],
                              mem[indexes[2]],
                              mem[indexes[3]],
                              mem[indexes[4]],
                              mem[indexes[5]],
                              mem[indexes[6]],
                              mem[indexes[7]]);
}

template <>
template <typename MT, typename IT>
inline void uint_v::gatherImplementation(const MT *mem, IT &&indexes)
{
    d.v() = _mm256_setr_epi32(mem[indexes[0]],
                              mem[indexes[1]],
                              mem[indexes[2]],
                              mem[indexes[3]],
                              mem[indexes[4]],
                              mem[indexes[5]],
                              mem[indexes[6]],
                              mem[indexes[7]]);
}

template <>
template <typename MT, typename IT>
inline void short_v::gatherImplementation(const MT *mem, IT &&indexes)
{
    d.v() = set(mem[indexes[0]],
                mem[indexes[1]],
                mem[indexes[2]],
                mem[indexes[3]],
                mem[indexes[4]],
                mem[indexes[5]],
                mem[indexes[6]],
                mem[indexes[7]]);
}

template <>
template <typename MT, typename IT>
inline void ushort_v::gatherImplementation(const MT *mem, IT &&indexes)
{
    d.v() = set(mem[indexes[0]],
                mem[indexes[1]],
                mem[indexes[2]],
                mem[indexes[3]],
                mem[indexes[4]],
                mem[indexes[5]],
                mem[indexes[6]],
                mem[indexes[7]]);
}

template <typename T>
template <typename MT, typename IT>
inline void Vector<T>::gatherImplementation(const MT *mem, IT &&indexes, MaskArgument mask)
{
    using Selector = std::integral_constant < Common::GatherScatterImplementation,
#ifdef VC_USE_SET_GATHERS
          Traits::is_simd_vector<IT>::value ? Common::GatherScatterImplementation::SetIndexZero :
#endif
#ifdef VC_USE_BSF_GATHERS
                                            Common::GatherScatterImplementation::BitScanLoop
#elif defined VC_USE_POPCNT_BSF_GATHERS
              Common::GatherScatterImplementation::PopcntSwitch
#else
              Common::GatherScatterImplementation::SimpleLoop
#endif
                                                > ;
    Common::executeGather(Selector(), *this, mem, indexes, mask);
}

template <typename T>
template <typename MT, typename IT>
inline void Vector<T>::scatterImplementation(MT *mem, IT &&indexes) const
{
    Common::unrolled_loop<std::size_t, 0, Size>([&](std::size_t i) { mem[indexes[i]] = d.m(i); });
}

template <typename T>
template <typename MT, typename IT>
inline void Vector<T>::scatterImplementation(MT *mem, IT &&indexes, MaskArgument mask) const
{
    using Selector = std::integral_constant < Common::GatherScatterImplementation,
#ifdef VC_USE_SET_GATHERS
          Traits::is_simd_vector<IT>::value ? Common::GatherScatterImplementation::SetIndexZero :
#endif
#ifdef VC_USE_BSF_GATHERS
                                            Common::GatherScatterImplementation::BitScanLoop
#elif defined VC_USE_POPCNT_BSF_GATHERS
              Common::GatherScatterImplementation::PopcntSwitch
#else
              Common::GatherScatterImplementation::SimpleLoop
#endif
                                                > ;
    Common::executeScatter(Selector(), *this, mem, indexes, mask);
}

#if defined(VC_MSVC) && VC_MSVC >= 170000000
// MSVC miscompiles the store mem[indexes[1]] = d.m(1) for T = (u)short
template <>
template <typename MT, typename IT>
Vc_ALWAYS_INLINE void short_v::scatterImplementation(MT *mem, IT &&indexes) const
{
    const unsigned int tmp = d.v()._d.m128i_u32[0];
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
Vc_ALWAYS_INLINE void ushort_v::scatterImplementation(MT *mem, IT &&indexes) const
{
    const unsigned int tmp = d.v()._d.m128i_u32[0];
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
#ifdef VC_USE_BUILTIN_VECTOR_TYPES
template<typename T> Vc_ALWAYS_INLINE Vc_PURE Vector<T> Vector<T>::operator-() const
{
    return VectorType(-d.builtin());
}
#else
namespace Internal
{
Vc_ALWAYS_INLINE Vc_CONST __m256 negate(__m256 v, std::integral_constant<std::size_t, 4>)
{
    return _mm256_xor_ps(v, setsignmask_ps());
}
Vc_ALWAYS_INLINE Vc_CONST __m256d negate(__m256d v, std::integral_constant<std::size_t, 8>)
{
    return _mm256_xor_pd(v, setsignmask_pd());
}
Vc_ALWAYS_INLINE Vc_CONST __m256i negate(__m256i v, std::integral_constant<std::size_t, 4>)
{
    return sign_epi32(v, setallone_si256());
}
Vc_ALWAYS_INLINE Vc_CONST __m128i negate(__m128i v, std::integral_constant<std::size_t, 2>)
{
    return _mm_sign_epi16(v, _mm_setallone_si128());
}
}  // namespace

template<typename T> Vc_ALWAYS_INLINE Vc_PURE Vector<T> Vector<T>::operator-() const
{
    return Internal::negate(d.v(), std::integral_constant<std::size_t, sizeof(T)>());
}
#endif

///////////////////////////////////////////////////////////////////////////////////////////
// horizontal ops {{{1
template <typename T> Vc_INTRINSIC std::pair<Vector<T>, int> Vector<T>::minIndex() const
{
    Vector<T> x = min();
    return std::make_pair(x, (*this == x).firstOne());
}
template <typename T> Vc_INTRINSIC std::pair<Vector<T>, int> Vector<T>::maxIndex() const
{
    Vector<T> x = max();
    return std::make_pair(x, (*this == x).firstOne());
}
template <> Vc_INTRINSIC std::pair<float_v, int> float_v::minIndex() const
{
    /*
    // 28 cycles latency:
    __m256 x = _mm256_min_ps(Mem::permute128<X1, X0>(d.v()), d.v());
    x = _mm256_min_ps(x, Reg::permute<X2, X3, X0, X1>(x));
    float_v xx = _mm256_min_ps(x, Reg::permute<X1, X0, X3, X2>(x));
    uint_v idx = uint_v::IndexesFromZero();
    idx = _mm256_castps_si256(
        _mm256_or_ps((*this != xx).data(), _mm256_castsi256_ps(idx.data())));
    return std::make_pair(xx, (*this == xx).firstOne());

    __m128 loData = lo128(d.v());
    __m128 hiData = hi128(d.v());
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
    return std::make_pair(concat(loData, loData), bits);
    */

    // 28 cycles Latency:
    __m256 x = d.v();
    __m256 idx = _mm256_castsi256_ps(
        VectorHelper<__m256i>::load<AlignedTag>(IndexesFromZeroData<int>::address()));
    __m256 y = Mem::permute128<X1, X0>(x);
    __m256 idy = Mem::permute128<X1, X0>(idx);
    __m256 less = cmplt_ps(x, y);

    x = _mm256_blendv_ps(y, x, less);
    idx = _mm256_blendv_ps(idy, idx, less);
    y = Reg::permute<X2, X3, X0, X1>(x);
    idy = Reg::permute<X2, X3, X0, X1>(idx);
    less = cmplt_ps(x, y);

    x = _mm256_blendv_ps(y, x, less);
    idx = _mm256_blendv_ps(idy, idx, less);
    y = Reg::permute<X1, X0, X3, X2>(x);
    idy = Reg::permute<X1, X0, X3, X2>(idx);
    less = cmplt_ps(x, y);

    idx = _mm256_blendv_ps(idy, idx, less);

    const auto index = _mm_cvtsi128_si32(avx_cast<__m128i>(idx));
    __asm__ __volatile__(""); // help GCC to order the instructions better
    x = _mm256_blendv_ps(y, x, less);
    return std::make_pair(x, index);
}
template<typename T> Vc_ALWAYS_INLINE Vector<T> Vector<T>::partialSum() const
{
    //   a    b    c    d    e    f    g    h
    // +      a    b    c    d    e    f    g    -> a ab bc  cd   de    ef     fg      gh
    // +           a    ab   bc   cd   de   ef   -> a ab abc abcd bcde  cdef   defg    efgh
    // +                     a    ab   abc  abcd -> a ab abc abcd abcde abcdef abcdefg abcdefgh
    Vector<T> tmp = *this;
    if (Size >  1) tmp += tmp.shifted(-1);
    if (Size >  2) tmp += tmp.shifted(-2);
    if (Size >  4) tmp += tmp.shifted(-4);
    if (Size >  8) tmp += tmp.shifted(-8);
    if (Size > 16) tmp += tmp.shifted(-16);
    return tmp;
}

/* This function requires correct masking because the neutral element of \p op is not necessarily 0
 *
template<typename T> template<typename BinaryOperation> Vc_ALWAYS_INLINE Vector<T> Vector<T>::partialSum(BinaryOperation op) const
{
    //   a    b    c    d    e    f    g    h
    // +      a    b    c    d    e    f    g    -> a ab bc  cd   de    ef     fg      gh
    // +           a    ab   bc   cd   de   ef   -> a ab abc abcd bcde  cdef   defg    efgh
    // +                     a    ab   abc  abcd -> a ab abc abcd abcde abcdef abcdefg abcdefgh
    Vector<T> tmp = *this;
    Mask mask(true);
    if (Size >  1) tmp(mask) = op(tmp, tmp.shifted(-1));
    if (Size >  2) tmp(mask) = op(tmp, tmp.shifted(-2));
    if (Size >  4) tmp(mask) = op(tmp, tmp.shifted(-4));
    if (Size >  8) tmp(mask) = op(tmp, tmp.shifted(-8));
    if (Size > 16) tmp(mask) = op(tmp, tmp.shifted(-16));
    return tmp;
}
*/

template<typename T> Vc_ALWAYS_INLINE typename Vector<T>::EntryType Vector<T>::min(MaskArg m) const
{
    Vector<T> tmp = std::numeric_limits<Vector<T> >::max();
    tmp(m) = *this;
    return tmp.min();
}
template<typename T> Vc_ALWAYS_INLINE typename Vector<T>::EntryType Vector<T>::max(MaskArg m) const
{
    Vector<T> tmp = std::numeric_limits<Vector<T> >::min();
    tmp(m) = *this;
    return tmp.max();
}
template<typename T> Vc_ALWAYS_INLINE typename Vector<T>::EntryType Vector<T>::product(MaskArg m) const
{
    Vector<T> tmp(VectorSpecialInitializerOne::One);
    tmp(m) = *this;
    return tmp.product();
}
template<typename T> Vc_ALWAYS_INLINE typename Vector<T>::EntryType Vector<T>::sum(MaskArg m) const
{
    Vector<T> tmp(VectorSpecialInitializerZero::Zero);
    tmp(m) = *this;
    return tmp.sum();
}//}}}
// copySign {{{1
template<> Vc_INTRINSIC Vector<float> Vector<float>::copySign(Vector<float>::AsArg reference) const
{
    return _mm256_or_ps(
            _mm256_and_ps(reference.d.v(), setsignmask_ps()),
            _mm256_and_ps(d.v(), setabsmask_ps())
            );
}
template<> Vc_INTRINSIC Vector<double> Vector<double>::copySign(Vector<double>::AsArg reference) const
{
    return _mm256_or_pd(
            _mm256_and_pd(reference.d.v(), setsignmask_pd()),
            _mm256_and_pd(d.v(), setabsmask_pd())
            );
}//}}}1
// exponent {{{1
template<> Vc_INTRINSIC Vector<float> Vector<float>::exponent() const
{
    VC_ASSERT((*this >= 0.f).isFull());
    return Internal::exponent(d.v());
}
template<> Vc_INTRINSIC Vector<double> Vector<double>::exponent() const
{
    VC_ASSERT((*this >= 0.).isFull());
    return Internal::exponent(d.v());
}
// }}}1
// Random {{{1
static Vc_ALWAYS_INLINE uint_v _doRandomStep()
{
    Vector<unsigned int> state0, state1;
    state0.load(&Common::RandomState[0]);
    state1.load(&Common::RandomState[uint_v::Size]);
    (state1 * 0xdeece66du + 11).store(&Common::RandomState[uint_v::Size]);
    uint_v(xor_si256((state0 * 0xdeece66du + 11).data(), srli_epi32<16>(state1.data()))).store(&Common::RandomState[0]);
    return state0;
}

template<typename T> Vc_ALWAYS_INLINE Vector<T> Vector<T>::Random()
{
    const Vector<unsigned int> state0 = _doRandomStep();
    return state0.reinterpretCast<Vector<T> >();
}

template<> Vc_ALWAYS_INLINE Vector<float> Vector<float>::Random()
{
    const Vector<unsigned int> state0 = _doRandomStep();
    return HT::sub(HV::or_(_cast(srli_epi32<2>(state0.data())), HT::one()), HT::one());
}

template<> Vc_ALWAYS_INLINE Vector<double> Vector<double>::Random()
{
    const m256i state = VectorHelper<m256i>::load<AlignedTag>(&Common::RandomState[0]);
    for (size_t k = 0; k < 8; k += 2) {
        typedef unsigned long long uint64 Vc_MAY_ALIAS;
        const uint64 stateX = *reinterpret_cast<const uint64 *>(&Common::RandomState[k]);
        *reinterpret_cast<uint64 *>(&Common::RandomState[k]) = (stateX * 0x5deece66dull + 11);
    }
    return HT::sub(HV::or_(_cast(srli_epi64<12>(state)), HT::one()), HT::one());
}
// }}}1
// shifted / rotated {{{1
template<size_t SIMDWidth, size_t Size, typename VectorType, typename EntryType> struct VectorShift;
template<> struct VectorShift<32, 4, m256d, double>
{
    static Vc_INTRINSIC m256d shifted(param256d v, int amount)
    {
        switch (amount) {
        case  0: return v;
        case  1: return avx_cast<m256d>(srli_si256<1 * sizeof(double)>(avx_cast<m256i>(v)));
        case  2: return avx_cast<m256d>(srli_si256<2 * sizeof(double)>(avx_cast<m256i>(v)));
        case  3: return avx_cast<m256d>(srli_si256<3 * sizeof(double)>(avx_cast<m256i>(v)));
        case -1: return avx_cast<m256d>(slli_si256<1 * sizeof(double)>(avx_cast<m256i>(v)));
        case -2: return avx_cast<m256d>(slli_si256<2 * sizeof(double)>(avx_cast<m256i>(v)));
        case -3: return avx_cast<m256d>(slli_si256<3 * sizeof(double)>(avx_cast<m256i>(v)));
        }
        return _mm256_setzero_pd();
    }
};
template<typename VectorType, typename EntryType> struct VectorShift<32, 8, VectorType, EntryType>
{
    typedef typename SseVectorType<VectorType>::Type SmallV;
    static Vc_INTRINSIC VectorType shifted(VC_ALIGNED_PARAMETER(VectorType) v, int amount)
    {
        switch (amount) {
        case  0: return v;
        case  1: return avx_cast<VectorType>(srli_si256<1 * sizeof(EntryType)>(avx_cast<m256i>(v)));
        case  2: return avx_cast<VectorType>(srli_si256<2 * sizeof(EntryType)>(avx_cast<m256i>(v)));
        case  3: return avx_cast<VectorType>(srli_si256<3 * sizeof(EntryType)>(avx_cast<m256i>(v)));
        case  4: return avx_cast<VectorType>(srli_si256<4 * sizeof(EntryType)>(avx_cast<m256i>(v)));
        case  5: return avx_cast<VectorType>(srli_si256<5 * sizeof(EntryType)>(avx_cast<m256i>(v)));
        case  6: return avx_cast<VectorType>(srli_si256<6 * sizeof(EntryType)>(avx_cast<m256i>(v)));
        case  7: return avx_cast<VectorType>(srli_si256<7 * sizeof(EntryType)>(avx_cast<m256i>(v)));
        case -1: return avx_cast<VectorType>(slli_si256<1 * sizeof(EntryType)>(avx_cast<m256i>(v)));
        case -2: return avx_cast<VectorType>(slli_si256<2 * sizeof(EntryType)>(avx_cast<m256i>(v)));
        case -3: return avx_cast<VectorType>(slli_si256<3 * sizeof(EntryType)>(avx_cast<m256i>(v)));
        case -4: return avx_cast<VectorType>(slli_si256<4 * sizeof(EntryType)>(avx_cast<m256i>(v)));
        case -5: return avx_cast<VectorType>(slli_si256<5 * sizeof(EntryType)>(avx_cast<m256i>(v)));
        case -6: return avx_cast<VectorType>(slli_si256<6 * sizeof(EntryType)>(avx_cast<m256i>(v)));
        case -7: return avx_cast<VectorType>(slli_si256<7 * sizeof(EntryType)>(avx_cast<m256i>(v)));
        }
        return avx_cast<VectorType>(_mm256_setzero_ps());
    }
};
template<typename VectorType, typename EntryType> struct VectorShift<16, 8, VectorType, EntryType>
{
    enum {
        EntryTypeSizeof = sizeof(EntryType)
    };
    static Vc_INTRINSIC VectorType shifted(VC_ALIGNED_PARAMETER(VectorType) v, int amount)
    {
        switch (amount) {
        case  0: return v;
        case  1: return avx_cast<VectorType>(_mm_srli_si128(avx_cast<m128i>(v), 1 * EntryTypeSizeof));
        case  2: return avx_cast<VectorType>(_mm_srli_si128(avx_cast<m128i>(v), 2 * EntryTypeSizeof));
        case  3: return avx_cast<VectorType>(_mm_srli_si128(avx_cast<m128i>(v), 3 * EntryTypeSizeof));
        case  4: return avx_cast<VectorType>(_mm_srli_si128(avx_cast<m128i>(v), 4 * EntryTypeSizeof));
        case  5: return avx_cast<VectorType>(_mm_srli_si128(avx_cast<m128i>(v), 5 * EntryTypeSizeof));
        case  6: return avx_cast<VectorType>(_mm_srli_si128(avx_cast<m128i>(v), 6 * EntryTypeSizeof));
        case  7: return avx_cast<VectorType>(_mm_srli_si128(avx_cast<m128i>(v), 7 * EntryTypeSizeof));
        case -1: return avx_cast<VectorType>(_mm_slli_si128(avx_cast<m128i>(v), 1 * EntryTypeSizeof));
        case -2: return avx_cast<VectorType>(_mm_slli_si128(avx_cast<m128i>(v), 2 * EntryTypeSizeof));
        case -3: return avx_cast<VectorType>(_mm_slli_si128(avx_cast<m128i>(v), 3 * EntryTypeSizeof));
        case -4: return avx_cast<VectorType>(_mm_slli_si128(avx_cast<m128i>(v), 4 * EntryTypeSizeof));
        case -5: return avx_cast<VectorType>(_mm_slli_si128(avx_cast<m128i>(v), 5 * EntryTypeSizeof));
        case -6: return avx_cast<VectorType>(_mm_slli_si128(avx_cast<m128i>(v), 6 * EntryTypeSizeof));
        case -7: return avx_cast<VectorType>(_mm_slli_si128(avx_cast<m128i>(v), 7 * EntryTypeSizeof));
        }
        return _mm_setzero_si128();
    }
};
template<typename T> Vc_INTRINSIC Vector<T> Vector<T>::shifted(int amount) const
{
    return VectorShift<sizeof(VectorType), Size, VectorType, EntryType>::shifted(d.v(), amount);
}

template <typename VectorType>
Vc_INTRINSIC Vc_CONST VectorType shifted_shortcut(VectorType left, VectorType right, Common::WidthT<m128>)
{
    return Mem::shuffle<X2, X3, Y0, Y1>(left, right);
}
template <typename VectorType>
Vc_INTRINSIC Vc_CONST VectorType shifted_shortcut(VectorType left, VectorType right, Common::WidthT<m256>)
{
    return Mem::shuffle128<X1, Y0>(left, right);
}

template<typename T> Vc_INTRINSIC Vector<T> Vector<T>::shifted(int amount, Vector shiftIn) const
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
template<size_t SIMDWidth, size_t Size, typename VectorType, typename EntryType> struct VectorRotate;
template<typename VectorType, typename EntryType> struct VectorRotate<32, 4, VectorType, EntryType>
{
    typedef typename SseVectorType<VectorType>::Type SmallV;
    enum {
        EntryTypeSizeof = sizeof(EntryType)
    };
    static Vc_INTRINSIC VectorType rotated(VC_ALIGNED_PARAMETER(VectorType) v, int amount)
    {
        const m128i vLo = avx_cast<m128i>(lo128(v));
        const m128i vHi = avx_cast<m128i>(hi128(v));
        switch (static_cast<unsigned int>(amount) % 4) {
        case  0: return v;
        case  1: return concat(avx_cast<SmallV>(_mm_alignr_epi8(vHi, vLo, 1 * EntryTypeSizeof)), avx_cast<SmallV>(_mm_alignr_epi8(vLo, vHi, 1 * EntryTypeSizeof)));
        case  2: return Mem::permute128<X1, X0>(v);
        case  3: return concat(avx_cast<SmallV>(_mm_alignr_epi8(vLo, vHi, 1 * EntryTypeSizeof)), avx_cast<SmallV>(_mm_alignr_epi8(vHi, vLo, 1 * EntryTypeSizeof)));
        }
        return _mm256_setzero_pd();
    }
};
template<typename VectorType, typename EntryType> struct VectorRotate<32, 8, VectorType, EntryType>
{
    typedef typename SseVectorType<VectorType>::Type SmallV;
    enum {
        EntryTypeSizeof = sizeof(EntryType)
    };
    static Vc_INTRINSIC VectorType rotated(VC_ALIGNED_PARAMETER(VectorType) v, int amount)
    {
        const m128i vLo = avx_cast<m128i>(lo128(v));
        const m128i vHi = avx_cast<m128i>(hi128(v));
        switch (static_cast<unsigned int>(amount) % 8) {
        case  0: return v;
        case  1: return concat(avx_cast<SmallV>(_mm_alignr_epi8(vHi, vLo, 1 * EntryTypeSizeof)), avx_cast<SmallV>(_mm_alignr_epi8(vLo, vHi, 1 * EntryTypeSizeof)));
        case  2: return concat(avx_cast<SmallV>(_mm_alignr_epi8(vHi, vLo, 2 * EntryTypeSizeof)), avx_cast<SmallV>(_mm_alignr_epi8(vLo, vHi, 2 * EntryTypeSizeof)));
        case  3: return concat(avx_cast<SmallV>(_mm_alignr_epi8(vHi, vLo, 3 * EntryTypeSizeof)), avx_cast<SmallV>(_mm_alignr_epi8(vLo, vHi, 3 * EntryTypeSizeof)));
        case  4: return Mem::permute128<X1, X0>(v);
        case  5: return concat(avx_cast<SmallV>(_mm_alignr_epi8(vLo, vHi, 1 * EntryTypeSizeof)), avx_cast<SmallV>(_mm_alignr_epi8(vHi, vLo, 1 * EntryTypeSizeof)));
        case  6: return concat(avx_cast<SmallV>(_mm_alignr_epi8(vLo, vHi, 2 * EntryTypeSizeof)), avx_cast<SmallV>(_mm_alignr_epi8(vHi, vLo, 2 * EntryTypeSizeof)));
        case  7: return concat(avx_cast<SmallV>(_mm_alignr_epi8(vLo, vHi, 3 * EntryTypeSizeof)), avx_cast<SmallV>(_mm_alignr_epi8(vHi, vLo, 3 * EntryTypeSizeof)));
        }
        return avx_cast<VectorType>(_mm256_setzero_ps());
    }
};
template<typename VectorType, typename EntryType> struct VectorRotate<16, 8, VectorType, EntryType>
{
    enum {
        EntryTypeSizeof = sizeof(EntryType)
    };
    static Vc_INTRINSIC VectorType rotated(VC_ALIGNED_PARAMETER(VectorType) v, int amount)
    {
        switch (static_cast<unsigned int>(amount) % 8) {
        case  0: return v;
        case  1: return avx_cast<VectorType>(_mm_alignr_epi8(v, v, 1 * EntryTypeSizeof));
        case  2: return avx_cast<VectorType>(_mm_alignr_epi8(v, v, 2 * EntryTypeSizeof));
        case  3: return avx_cast<VectorType>(_mm_alignr_epi8(v, v, 3 * EntryTypeSizeof));
        case  4: return avx_cast<VectorType>(_mm_alignr_epi8(v, v, 4 * EntryTypeSizeof));
        case  5: return avx_cast<VectorType>(_mm_alignr_epi8(v, v, 5 * EntryTypeSizeof));
        case  6: return avx_cast<VectorType>(_mm_alignr_epi8(v, v, 6 * EntryTypeSizeof));
        case  7: return avx_cast<VectorType>(_mm_alignr_epi8(v, v, 7 * EntryTypeSizeof));
        }
        return _mm_setzero_si128();
    }
};
template<typename T> Vc_INTRINSIC Vector<T> Vector<T>::rotated(int amount) const
{
    return VectorRotate<sizeof(VectorType), Size, VectorType, EntryType>::rotated(d.v(), amount);
    /*
    const m128i v0 = avx_cast<m128i>(d.v()[0]);
    const m128i v1 = avx_cast<m128i>(d.v()[1]);
    switch (static_cast<unsigned int>(amount) % Size) {
    case  0: return *this;
    case  1: return concat(avx_cast<m128>(_mm_alignr_epi8(v1, v0, 1 * sizeof(EntryType))), avx_cast<m128>(_mm_alignr_epi8(v0, v1, 1 * sizeof(EntryType))));
    case  2: return concat(avx_cast<m128>(_mm_alignr_epi8(v1, v0, 2 * sizeof(EntryType))), avx_cast<m128>(_mm_alignr_epi8(v0, v1, 2 * sizeof(EntryType))));
    case  3: return concat(avx_cast<m128>(_mm_alignr_epi8(v1, v0, 3 * sizeof(EntryType))), avx_cast<m128>(_mm_alignr_epi8(v0, v1, 3 * sizeof(EntryType))));
    case  4: return concat(d.v()[1], d.v()[0]);
    case  5: return concat(avx_cast<m128>(_mm_alignr_epi8(v0, v1, 1 * sizeof(EntryType))), avx_cast<m128>(_mm_alignr_epi8(v1, v0, 1 * sizeof(EntryType))));
    case  6: return concat(avx_cast<m128>(_mm_alignr_epi8(v0, v1, 2 * sizeof(EntryType))), avx_cast<m128>(_mm_alignr_epi8(v1, v0, 2 * sizeof(EntryType))));
    case  7: return concat(avx_cast<m128>(_mm_alignr_epi8(v0, v1, 3 * sizeof(EntryType))), avx_cast<m128>(_mm_alignr_epi8(v1, v0, 3 * sizeof(EntryType))));
    }
    */
}
// interleaveLow/-High {{{1
template <> Vc_INTRINSIC double_v double_v::interleaveLow(double_v x) const
{
    return Mem::shuffle128<X0, Y0>(_mm256_unpacklo_pd(data(), x.data()),
                                   _mm256_unpackhi_pd(data(), x.data()));
}
template <> Vc_INTRINSIC double_v double_v::interleaveHigh(double_v x) const
{
    return Mem::shuffle128<X1, Y1>(_mm256_unpacklo_pd(data(), x.data()),
                                   _mm256_unpackhi_pd(data(), x.data()));
}
template <> Vc_INTRINSIC float_v float_v::interleaveLow(float_v x) const
{
    return Mem::shuffle128<X0, Y0>(_mm256_unpacklo_ps(data(), x.data()),
                                   _mm256_unpackhi_ps(data(), x.data()));
}
template <> Vc_INTRINSIC float_v float_v::interleaveHigh(float_v x) const
{
    return Mem::shuffle128<X1, Y1>(_mm256_unpacklo_ps(data(), x.data()),
                                   _mm256_unpackhi_ps(data(), x.data()));
}
#ifdef VC_IMPL_AVX2
template <> Vc_INTRINSIC    int_v    int_v::interleaveLow (   int_v x) const { return unpacklo_epi32(data(), x.data()); }
template <> Vc_INTRINSIC    int_v    int_v::interleaveHigh(   int_v x) const { return unpackhi_epi32(data(), x.data()); }
template <> Vc_INTRINSIC   uint_v   uint_v::interleaveLow (  uint_v x) const { return unpacklo_epi32(data(), x.data()); }
template <> Vc_INTRINSIC   uint_v   uint_v::interleaveHigh(  uint_v x) const { return unpackhi_epi32(data(), x.data()); }
// TODO:
//template <> Vc_INTRINSIC  short_v  short_v::interleaveLow ( short_v x) const { return unpacklo_epi16(data(), x.data()); }
//template <> Vc_INTRINSIC  short_v  short_v::interleaveHigh( short_v x) const { return unpackhi_epi16(data(), x.data()); }
//template <> Vc_INTRINSIC ushort_v ushort_v::interleaveLow (ushort_v x) const { return unpacklo_epi16(data(), x.data()); }
//template <> Vc_INTRINSIC ushort_v ushort_v::interleaveHigh(ushort_v x) const { return unpackhi_epi16(data(), x.data()); }
template <> Vc_INTRINSIC  short_v  short_v::interleaveLow ( short_v x) const { return _mm_unpacklo_epi16(data(), x.data()); }
template <> Vc_INTRINSIC  short_v  short_v::interleaveHigh( short_v x) const { return _mm_unpackhi_epi16(data(), x.data()); }
template <> Vc_INTRINSIC ushort_v ushort_v::interleaveLow (ushort_v x) const { return _mm_unpacklo_epi16(data(), x.data()); }
template <> Vc_INTRINSIC ushort_v ushort_v::interleaveHigh(ushort_v x) const { return _mm_unpackhi_epi16(data(), x.data()); }
#else
template <> Vc_INTRINSIC    int_v    int_v::interleaveLow (   int_v x) const {
    return concat(_mm_unpacklo_epi32(lo128(data()), lo128(x.data())),
                  _mm_unpackhi_epi32(lo128(data()), lo128(x.data())));
}
template <> Vc_INTRINSIC int_v int_v::interleaveHigh(int_v x) const
{
    return concat(_mm_unpacklo_epi32(hi128(data()), hi128(x.data())),
                  _mm_unpackhi_epi32(hi128(data()), hi128(x.data())));
}
template <> Vc_INTRINSIC uint_v uint_v::interleaveLow(uint_v x) const
{
    return concat(_mm_unpacklo_epi32(lo128(data()), lo128(x.data())),
                  _mm_unpackhi_epi32(lo128(data()), lo128(x.data())));
}
template <> Vc_INTRINSIC uint_v uint_v::interleaveHigh(uint_v x) const
{
    return concat(_mm_unpacklo_epi32(hi128(data()), hi128(x.data())),
                  _mm_unpackhi_epi32(hi128(data()), hi128(x.data())));
}
template <> Vc_INTRINSIC  short_v  short_v::interleaveLow ( short_v x) const { return _mm_unpacklo_epi16(data(), x.data()); }
template <> Vc_INTRINSIC  short_v  short_v::interleaveHigh( short_v x) const { return _mm_unpackhi_epi16(data(), x.data()); }
template <> Vc_INTRINSIC ushort_v ushort_v::interleaveLow (ushort_v x) const { return _mm_unpacklo_epi16(data(), x.data()); }
template <> Vc_INTRINSIC ushort_v ushort_v::interleaveHigh(ushort_v x) const { return _mm_unpackhi_epi16(data(), x.data()); }
#endif
// generate {{{1
template <> template <typename G> Vc_INTRINSIC double_v double_v::generate(G gen)
{
    const auto tmp0 = gen(0);
    const auto tmp1 = gen(1);
    const auto tmp2 = gen(2);
    const auto tmp3 = gen(3);
    return _mm256_setr_pd(tmp0, tmp1, tmp2, tmp3);
}
template <> template <typename G> Vc_INTRINSIC float_v float_v::generate(G gen)
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
template <> template <typename G> Vc_INTRINSIC int_v int_v::generate(G gen)
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
template <> template <typename G> Vc_INTRINSIC uint_v uint_v::generate(G gen)
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
template <> template <typename G> Vc_INTRINSIC short_v short_v::generate(G gen)
{
    const auto tmp0 = gen(0);
    const auto tmp1 = gen(1);
    const auto tmp2 = gen(2);
    const auto tmp3 = gen(3);
    const auto tmp4 = gen(4);
    const auto tmp5 = gen(5);
    const auto tmp6 = gen(6);
    const auto tmp7 = gen(7);
    return _mm_setr_epi16(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7);
}
template <> template <typename G> Vc_INTRINSIC ushort_v ushort_v::generate(G gen)
{
    const auto tmp0 = gen(0);
    const auto tmp1 = gen(1);
    const auto tmp2 = gen(2);
    const auto tmp3 = gen(3);
    const auto tmp4 = gen(4);
    const auto tmp5 = gen(5);
    const auto tmp6 = gen(6);
    const auto tmp7 = gen(7);
    return _mm_setr_epi16(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7);
}
// }}}1
// reversed {{{1
template <> Vc_INTRINSIC Vc_PURE double_v double_v::reversed() const
{
    return Mem::permute128<X1, X0>(Mem::permute<X1, X0, X3, X2>(d.v()));
}
template <> Vc_INTRINSIC Vc_PURE float_v float_v::reversed() const
{
    return Mem::permute128<X1, X0>(Mem::permute<X3, X2, X1, X0>(d.v()));
}
#ifdef VC_IMPL_AVX2
#else
template <> Vc_INTRINSIC Vc_PURE int_v int_v::reversed() const
{
    return Mem::permute128<X1, X0>(Mem::permute<X3, X2, X1, X0>(d.v()));
}
template <> Vc_INTRINSIC Vc_PURE uint_v uint_v::reversed() const
{
    return Mem::permute128<X1, X0>(Mem::permute<X3, X2, X1, X0>(d.v()));
}
template <> Vc_INTRINSIC Vc_PURE short_v short_v::reversed() const
{
    return avx_cast<__m128i>(
        Mem::shuffle<X1, Y0>(avx_cast<__m128d>(Mem::permuteHi<X7, X6, X5, X4>(d.v())),
                             avx_cast<__m128d>(Mem::permuteLo<X3, X2, X1, X0>(d.v()))));
}
template <> Vc_INTRINSIC Vc_PURE ushort_v ushort_v::reversed() const
{
    return avx_cast<__m128i>(
        Mem::shuffle<X1, Y0>(avx_cast<__m128d>(Mem::permuteHi<X7, X6, X5, X4>(d.v())),
                             avx_cast<__m128d>(Mem::permuteLo<X3, X2, X1, X0>(d.v()))));
}
#endif
// permutation via operator[] {{{1
template <> Vc_INTRINSIC float_v float_v::operator[](int_v perm) const
{
    /*
    const int_m cross128 = concat(_mm_cmpgt_epi32(lo128(perm.data()), _mm_set1_epi32(3)),
                                  _mm_cmplt_epi32(hi128(perm.data()), _mm_set1_epi32(4)));
    if (cross128.isNotEmpty()) {
        float_v x = _mm256_permutevar_ps(d.v(), perm.data());
        x(cross128) = _mm256_permutevar_ps(Mem::permute128<X1, X0>(d.v()), perm.data());
        return x;
    } else {
    */
        return _mm256_permutevar_ps(d.v(), perm.data());
}
// broadcast from constexpr index {{{1
template <> template <int Index> Vc_INTRINSIC float_v float_v::broadcast() const
{
    constexpr VecPos Inner = static_cast<VecPos>(Index & 0x3);
    constexpr VecPos Outer = static_cast<VecPos>((Index & 0x4) / 4);
    return Mem::permute<Inner, Inner, Inner, Inner>(Mem::permute128<Outer, Outer>(d.v()));
}
template <> template <int Index> Vc_INTRINSIC double_v double_v::broadcast() const
{
    constexpr VecPos Inner = static_cast<VecPos>(Index & 0x1);
    constexpr VecPos Outer = static_cast<VecPos>((Index & 0x2) / 2);
    return Mem::permute<Inner, Inner>(Mem::permute128<Outer, Outer>(d.v()));
}
// }}}1
}
}

#include "undomacros.h"

// vim: foldmethod=marker
