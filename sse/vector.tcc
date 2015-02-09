/*  This file is part of the Vc library. {{{
Copyright © 2010-2014 Matthias Kretz <kretz@kde.org>
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
#include "limits.h"
#include "../common/bitscanintrinsics.h"
#include "../common/set.h"
#include "../common/gatherimplementation.h"
#include "../common/scatterimplementation.h"
#include "../common/transpose.h"
#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace SSE
{

// constants {{{1
template<typename T, int Size> Vc_ALWAYS_INLINE Vc_CONST const T *_IndexesFromZero() {
    if (Size == 4) {
        return reinterpret_cast<const T *>(_IndexesFromZero4);
    } else if (Size == 8) {
        return reinterpret_cast<const T *>(_IndexesFromZero8);
    } else if (Size == 16) {
        return reinterpret_cast<const T *>(_IndexesFromZero16);
    }
    return 0;
}

template<typename T> Vc_INTRINSIC Vector<T>::Vector(VectorSpecialInitializerZero::ZEnum)
    : d(VectorHelper<VectorType>::zero())
{
}

template<typename T> Vc_INTRINSIC Vector<T>::Vector(VectorSpecialInitializerOne::OEnum)
    : d(VectorHelper<T>::one())
{
}

template<typename T> Vc_INTRINSIC Vector<T>::Vector(VectorSpecialInitializerIndexesFromZero::IEnum)
    : d(VectorHelper<VectorType>::template load<AlignedTag>(_IndexesFromZero<EntryType, Size>()))
{
}

template<> Vc_INTRINSIC float_v::Vector(VectorSpecialInitializerIndexesFromZero::IEnum)
    : d(StaticCastHelper<int, float>::cast(int_v::IndexesFromZero().data()))
{
}

template<> Vc_INTRINSIC double_v::Vector(VectorSpecialInitializerIndexesFromZero::IEnum)
    : d(StaticCastHelper<int, double>::cast(int_v::IndexesFromZero().data()))
{
}

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
    static Vc_ALWAYS_INLINE Vc_PURE __m128 load(const double *mem, Flags)
    {
        return _mm_movelh_ps(_mm_cvtpd_ps(VectorHelper<__m128d>::load<Flags>(&mem[0])),
                             _mm_cvtpd_ps(VectorHelper<__m128d>::load<Flags>(&mem[2])));
    }
};
template<typename Flags> struct LoadHelper<float, unsigned int, Flags> {
    static Vc_ALWAYS_INLINE Vc_PURE __m128 load(const unsigned int *mem, Flags)
    {
        return StaticCastHelper<unsigned int, float>::cast(VectorHelper<__m128i>::load<Flags>(mem));
    }
};
template<typename Flags> struct LoadHelper<float, int, Flags> {
    static Vc_ALWAYS_INLINE Vc_PURE __m128 load(const int *mem, Flags)
    {
        return StaticCastHelper<int, float>::cast(VectorHelper<__m128i>::load<Flags>(mem));
    }
};
template<typename Flags> struct LoadHelper<float, unsigned short, Flags> {
    static Vc_ALWAYS_INLINE Vc_PURE __m128 load(const unsigned short *mem, Flags f)
    {
        return _mm_cvtepi32_ps(LoadHelper<int, unsigned short, Flags>::load(mem, f));
    }
};
template<typename Flags> struct LoadHelper<float, short, Flags> {
    static Vc_ALWAYS_INLINE Vc_PURE __m128 load(const short *mem, Flags f)
    {
        return _mm_cvtepi32_ps(LoadHelper<int, short, Flags>::load(mem, f));
    }
};
template<typename Flags> struct LoadHelper<float, unsigned char, Flags> {
    static Vc_ALWAYS_INLINE Vc_PURE __m128 load(const unsigned char *mem, Flags f)
    {
        return _mm_cvtepi32_ps(LoadHelper<int, unsigned char, Flags>::load(mem, f));
    }
};
template<typename Flags> struct LoadHelper<float, signed char, Flags> {
    static Vc_ALWAYS_INLINE Vc_PURE __m128 load(const signed char *mem, Flags f)
    {
        return _mm_cvtepi32_ps(LoadHelper<int, signed char, Flags>::load(mem, f));
    }
};


// int {{{2
template<typename Flags> struct LoadHelper<int, unsigned int, Flags> {
    static Vc_ALWAYS_INLINE Vc_PURE __m128i load(const unsigned int *mem, Flags)
    {
        return VectorHelper<__m128i>::load<Flags>(mem);
    }
};
// no difference between streaming and alignment, because the
// 32/64 bit loads are not available as streaming loads, and can always be unaligned
template<typename Flags> struct LoadHelper<int, unsigned short, Flags> {
    static Vc_ALWAYS_INLINE Vc_PURE __m128i load(const unsigned short *mem, Flags)
    {
        return cvtepu16_epi32( _mm_loadl_epi64(reinterpret_cast<const __m128i *>(mem)));
    }
};
template<typename Flags> struct LoadHelper<int, short, Flags> {
    static Vc_ALWAYS_INLINE Vc_PURE __m128i load(const short *mem, Flags)
    {
        return cvtepi16_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(mem)));
    }
};
template<typename Flags> struct LoadHelper<int, unsigned char, Flags> {
    static Vc_ALWAYS_INLINE Vc_PURE __m128i load(const unsigned char *mem, Flags)
    {
        return cvtepu8_epi32(_mm_cvtsi32_si128(*reinterpret_cast<const int *>(mem)));
    }
};
template<typename Flags> struct LoadHelper<int, signed char, Flags> {
    static Vc_ALWAYS_INLINE Vc_PURE __m128i load(const signed char *mem, Flags)
    {
        return cvtepi8_epi32(_mm_cvtsi32_si128(*reinterpret_cast<const int *>(mem)));
    }
};

// unsigned int {{{2
template<typename Flags> struct LoadHelper<unsigned int, unsigned short, Flags> {
    static Vc_ALWAYS_INLINE Vc_PURE __m128i load(const unsigned short *mem, Flags)
    {
        return cvtepu16_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(mem)));
    }
};
template<typename Flags> struct LoadHelper<unsigned int, unsigned char, Flags> {
    static Vc_ALWAYS_INLINE Vc_PURE __m128i load(const unsigned char *mem, Flags)
    {
        return cvtepu8_epi32(_mm_cvtsi32_si128(*reinterpret_cast<const int *>(mem)));
    }
};

// short {{{2
template<typename Flags> struct LoadHelper<short, unsigned short, Flags> {
    static Vc_ALWAYS_INLINE Vc_PURE __m128i load(const unsigned short *mem, Flags)
    {
        return VectorHelper<__m128i>::load<Flags>(mem);
    }
};
template<typename Flags> struct LoadHelper<short, unsigned char, Flags> {
    static Vc_ALWAYS_INLINE Vc_PURE __m128i load(const unsigned char *mem, Flags)
    {
        return cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(mem)));
    }
};
template<typename Flags> struct LoadHelper<short, signed char, Flags> {
    static Vc_ALWAYS_INLINE Vc_PURE __m128i load(const signed char *mem, Flags)
    {
        return cvtepi8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(mem)));
    }
};

// unsigned short {{{2
template<typename Flags> struct LoadHelper<unsigned short, unsigned char, Flags> {
    static Vc_ALWAYS_INLINE Vc_PURE __m128i load(const unsigned char *mem, Flags)
    {
        return cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(mem)));
    }
};

// general load, implemented via LoadHelper {{{2
template <typename DstT>
template <typename SrcT, typename Flags, typename>
Vc_INTRINSIC void Vector<DstT>::load(const SrcT *mem, Flags flags)
{
    Common::handleLoadPrefetches(mem, flags);
    d.v() = LoadHelper<DstT, SrcT, Flags>::load(mem, flags);
}

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
// zeroing {{{1
template<typename T> Vc_INTRINSIC void Vector<T>::setZero()
{
    data() = VectorHelper<VectorType>::zero();
}

template<typename T> Vc_INTRINSIC void Vector<T>::setZero(const Mask &k)
{
    data() = VectorHelper<VectorType>::andnot_(mm128_reinterpret_cast<VectorType>(k.data()), data());
}

template<typename T> Vc_INTRINSIC void Vector<T>::setZeroInverted(const Mask &k)
{
    data() = VectorHelper<VectorType>::and_(mm128_reinterpret_cast<VectorType>(k.data()), data());
}

template<> Vc_INTRINSIC void Vector<double>::setQnan()
{
    data() = _mm_setallone_pd();
}
template<> Vc_INTRINSIC void Vector<double>::setQnan(const Mask &k)
{
    data() = _mm_or_pd(data(), k.dataD());
}
template<> Vc_INTRINSIC void Vector<float>::setQnan()
{
    data() = _mm_setallone_ps();
}
template<> Vc_INTRINSIC void Vector<float>::setQnan(const Mask &k)
{
    data() = _mm_or_ps(data(), k.data());
}

///////////////////////////////////////////////////////////////////////////////////////////
// stores {{{1
template <typename T>
template <typename U, typename Flags, typename>
Vc_INTRINSIC void Vector<T>::store(U *mem, Flags flags) const
{
    Common::handleStorePrefetches(mem, flags);
    HV::template store<Flags>(mem, data());
}

template <typename T>
template <typename U, typename Flags, typename>
Vc_INTRINSIC void Vector<T>::store(U *mem, Mask mask, Flags flags) const
{
    Common::handleStorePrefetches(mem, flags);
    HV::template store<Flags>(mem, data(), sse_cast<VectorType>(mask.data()));
}

///////////////////////////////////////////////////////////////////////////////////////////
// division {{{1
template<typename T> inline Vector<T> &Vector<T>::operator/=(EntryType x)
{
    if (VectorTraits<T>::HasVectorDivision) {
        return operator/=(Vector<T>(x));
    }
    for_all_vector_entries(i,
            d.m(i) /= x;
            );
    return *this;
}

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
            r.d.m(i) = d.m(i) / x.d.m(i);
            );
    return r;
}

template<> inline Vector<short> &Vector<short>::operator/=(VC_ALIGNED_PARAMETER(Vector<short>) x)
{
    __m128 lo = _mm_cvtepi32_ps(VectorHelper<short>::expand0(d.v()));
    __m128 hi = _mm_cvtepi32_ps(VectorHelper<short>::expand1(d.v()));
    lo = _mm_div_ps(lo, _mm_cvtepi32_ps(VectorHelper<short>::expand0(x.d.v())));
    hi = _mm_div_ps(hi, _mm_cvtepi32_ps(VectorHelper<short>::expand1(x.d.v())));
    d.v() = HT::concat(_mm_cvttps_epi32(lo), _mm_cvttps_epi32(hi));
    return *this;
}

template<> inline Vc_PURE Vector<short> Vector<short>::operator/(VC_ALIGNED_PARAMETER(Vector<short>) x) const
{
    __m128 lo = _mm_cvtepi32_ps(VectorHelper<short>::expand0(d.v()));
    __m128 hi = _mm_cvtepi32_ps(VectorHelper<short>::expand1(d.v()));
    lo = _mm_div_ps(lo, _mm_cvtepi32_ps(VectorHelper<short>::expand0(x.d.v())));
    hi = _mm_div_ps(hi, _mm_cvtepi32_ps(VectorHelper<short>::expand1(x.d.v())));
    return HT::concat(_mm_cvttps_epi32(lo), _mm_cvttps_epi32(hi));
}

template<> inline Vector<unsigned short> &Vector<unsigned short>::operator/=(VC_ALIGNED_PARAMETER(Vector<unsigned short>) x)
{
    __m128 lo = _mm_cvtepi32_ps(VectorHelper<unsigned short>::expand0(d.v()));
    __m128 hi = _mm_cvtepi32_ps(VectorHelper<unsigned short>::expand1(d.v()));
    lo = _mm_div_ps(lo, _mm_cvtepi32_ps(VectorHelper<unsigned short>::expand0(x.d.v())));
    hi = _mm_div_ps(hi, _mm_cvtepi32_ps(VectorHelper<unsigned short>::expand1(x.d.v())));
    d.v() = HT::concat(_mm_cvttps_epi32(lo), _mm_cvttps_epi32(hi));
    return *this;
}

template<> Vc_ALWAYS_INLINE Vc_PURE Vector<unsigned short> Vector<unsigned short>::operator/(VC_ALIGNED_PARAMETER(Vector<unsigned short>) x) const
{
    __m128 lo = _mm_cvtepi32_ps(VectorHelper<unsigned short>::expand0(d.v()));
    __m128 hi = _mm_cvtepi32_ps(VectorHelper<unsigned short>::expand1(d.v()));
    lo = _mm_div_ps(lo, _mm_cvtepi32_ps(VectorHelper<unsigned short>::expand0(x.d.v())));
    hi = _mm_div_ps(hi, _mm_cvtepi32_ps(VectorHelper<unsigned short>::expand1(x.d.v())));
    return HT::concat(_mm_cvttps_epi32(lo), _mm_cvttps_epi32(hi));
}

template<> Vc_ALWAYS_INLINE Vector<float> &Vector<float>::operator/=(VC_ALIGNED_PARAMETER(Vector<float>) x)
{
    d.v() = _mm_div_ps(d.v(), x.d.v());
    return *this;
}

template<> Vc_ALWAYS_INLINE Vc_PURE Vector<float> Vector<float>::operator/(VC_ALIGNED_PARAMETER(Vector<float>) x) const
{
    return _mm_div_ps(d.v(), x.d.v());
}

template<> Vc_ALWAYS_INLINE Vector<double> &Vector<double>::operator/=(VC_ALIGNED_PARAMETER(Vector<double>) x)
{
    d.v() = _mm_div_pd(d.v(), x.d.v());
    return *this;
}

template<> Vc_ALWAYS_INLINE Vc_PURE Vector<double> Vector<double>::operator/(VC_ALIGNED_PARAMETER(Vector<double>) x) const
{
    return _mm_div_pd(d.v(), x.d.v());
}

///////////////////////////////////////////////////////////////////////////////////////////
// operator- {{{1
namespace Internal
{
Vc_ALWAYS_INLINE Vc_CONST __m128 negate(__m128 v, std::integral_constant<std::size_t, 4>)
{
    return _mm_xor_ps(v, _mm_setsignmask_ps());
}
Vc_ALWAYS_INLINE Vc_CONST __m128d negate(__m128d v, std::integral_constant<std::size_t, 8>)
{
    return _mm_xor_pd(v, _mm_setsignmask_pd());
}
Vc_ALWAYS_INLINE Vc_CONST __m128i negate(__m128i v, std::integral_constant<std::size_t, 4>)
{
#ifdef VC_IMPL_SSSE3
    return _mm_sign_epi32(v, _mm_setallone_si128());
#else
    return _mm_sub_epi32(_mm_setzero_si128(), v);
#endif
}
Vc_ALWAYS_INLINE Vc_CONST __m128i negate(__m128i v, std::integral_constant<std::size_t, 2>)
{
#ifdef VC_IMPL_SSSE3
    return _mm_sign_epi16(v, _mm_setallone_si128());
#else
    return _mm_sub_epi16(_mm_setzero_si128(), v);
#endif
}
}  // namespace

template<typename T> Vc_ALWAYS_INLINE Vc_PURE Vector<T> Vector<T>::operator-() const
{
    return Internal::negate(d.v(), std::integral_constant<std::size_t, sizeof(T)>());
}
///////////////////////////////////////////////////////////////////////////////////////////
// integer ops {{{1
template <typename T> inline Vc_PURE Vector<T> Vector<T>::operator%(const Vector<T> &n) const
{
    return *this - *this / n * n;
}

#define OP_IMPL(T, symbol, fun) \
template<> Vc_ALWAYS_INLINE Vector<T> &Vector<T>::operator symbol##=(const Vector<T> &x) \
{ \
    d.v() = VectorHelper<T>::fun(d.v(), x.d.v()); \
    return *this; \
} \
template<> Vc_ALWAYS_INLINE Vc_PURE Vector<T>  Vector<T>::operator symbol(const Vector<T> &x) const \
{ \
    return VectorHelper<T>::fun(d.v(), x.d.v()); \
}
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

#ifdef VC_IMPL_XOP
static Vc_INTRINSIC Vc_CONST __m128i shiftLeft (const    int_v &value, const    int_v &count) { return _mm_sha_epi32(value.data(), count.data()); }
static Vc_INTRINSIC Vc_CONST __m128i shiftLeft (const   uint_v &value, const   uint_v &count) { return _mm_shl_epi32(value.data(), count.data()); }
static Vc_INTRINSIC Vc_CONST __m128i shiftLeft (const  short_v &value, const  short_v &count) { return _mm_sha_epi16(value.data(), count.data()); }
static Vc_INTRINSIC Vc_CONST __m128i shiftLeft (const ushort_v &value, const ushort_v &count) { return _mm_shl_epi16(value.data(), count.data()); }
static Vc_INTRINSIC Vc_CONST __m128i shiftRight(const    int_v &value, const    int_v &count) { return shiftLeft(value,          -count ); }
static Vc_INTRINSIC Vc_CONST __m128i shiftRight(const   uint_v &value, const   uint_v &count) { return shiftLeft(value,   uint_v(-count)); }
static Vc_INTRINSIC Vc_CONST __m128i shiftRight(const  short_v &value, const  short_v &count) { return shiftLeft(value,          -count ); }
static Vc_INTRINSIC Vc_CONST __m128i shiftRight(const ushort_v &value, const ushort_v &count) { return shiftLeft(value, ushort_v(-count)); }

#define _VC_OP(T, symbol, impl) \
template<> Vc_INTRINSIC T &T::operator symbol##=(T::AsArg shift) \
{ \
    d.v() = impl(*this, shift); \
    return *this; \
} \
template<> Vc_INTRINSIC Vc_PURE T  T::operator symbol   (T::AsArg shift) const \
{ \
    return impl(*this, shift); \
}
VC_APPLY_2(VC_LIST_INT_VECTOR_TYPES, _VC_OP, <<, shiftLeft)
VC_APPLY_2(VC_LIST_INT_VECTOR_TYPES, _VC_OP, >>, shiftRight)
#undef _VC_OP
#else
#if defined(VC_GCC) && VC_GCC == 0x40600 && defined(VC_IMPL_XOP)
#define VC_WORKAROUND __attribute__((optimize("no-tree-vectorize"),weak))
#else
#define VC_WORKAROUND Vc_INTRINSIC
#endif

#define OP_IMPL(T, symbol) \
template<> VC_WORKAROUND Vector<T> &Vector<T>::operator symbol##=(Vector<T>::AsArg x) \
{ \
    for_all_vector_entries(i, \
            d.m(i) symbol##= x.d.m(i); \
            ); \
    return *this; \
} \
template<> inline Vc_PURE Vector<T>  Vector<T>::operator symbol(Vector<T>::AsArg x) const \
{ \
    Vector<T> r; \
    for_all_vector_entries(i, \
            r.d.m(i) = d.m(i) symbol x.d.m(i); \
            ); \
    return r; \
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
#undef VC_WORKAROUND
#endif

template<typename T> Vc_ALWAYS_INLINE Vector<T> &Vector<T>::operator>>=(int shift) {
    d.v() = VectorHelper<T>::shiftRight(d.v(), shift);
    return *this;
}
template<typename T> Vc_ALWAYS_INLINE Vc_PURE Vector<T> Vector<T>::operator>>(int shift) const {
    return VectorHelper<T>::shiftRight(d.v(), shift);
}
template<typename T> Vc_ALWAYS_INLINE Vector<T> &Vector<T>::operator<<=(int shift) {
    d.v() = VectorHelper<T>::shiftLeft(d.v(), shift);
    return *this;
}
template<typename T> Vc_ALWAYS_INLINE Vc_PURE Vector<T> Vector<T>::operator<<(int shift) const {
    return VectorHelper<T>::shiftLeft(d.v(), shift);
}

///////////////////////////////////////////////////////////////////////////////////////////
// swizzles {{{1
template<typename T> Vc_INTRINSIC Vc_PURE const Vector<T> &Vector<T>::abcd() const { return *this; }
template<typename T> Vc_INTRINSIC Vc_PURE const Vector<T>  Vector<T>::cdab() const { return Mem::permute<X2, X3, X0, X1>(data()); }
template<typename T> Vc_INTRINSIC Vc_PURE const Vector<T>  Vector<T>::badc() const { return Mem::permute<X1, X0, X3, X2>(data()); }
template<typename T> Vc_INTRINSIC Vc_PURE const Vector<T>  Vector<T>::aaaa() const { return Mem::permute<X0, X0, X0, X0>(data()); }
template<typename T> Vc_INTRINSIC Vc_PURE const Vector<T>  Vector<T>::bbbb() const { return Mem::permute<X1, X1, X1, X1>(data()); }
template<typename T> Vc_INTRINSIC Vc_PURE const Vector<T>  Vector<T>::cccc() const { return Mem::permute<X2, X2, X2, X2>(data()); }
template<typename T> Vc_INTRINSIC Vc_PURE const Vector<T>  Vector<T>::dddd() const { return Mem::permute<X3, X3, X3, X3>(data()); }
template<typename T> Vc_INTRINSIC Vc_PURE const Vector<T>  Vector<T>::bcad() const { return Mem::permute<X1, X2, X0, X3>(data()); }
template<typename T> Vc_INTRINSIC Vc_PURE const Vector<T>  Vector<T>::bcda() const { return Mem::permute<X1, X2, X3, X0>(data()); }
template<typename T> Vc_INTRINSIC Vc_PURE const Vector<T>  Vector<T>::dabc() const { return Mem::permute<X3, X0, X1, X2>(data()); }
template<typename T> Vc_INTRINSIC Vc_PURE const Vector<T>  Vector<T>::acbd() const { return Mem::permute<X0, X2, X1, X3>(data()); }
template<typename T> Vc_INTRINSIC Vc_PURE const Vector<T>  Vector<T>::dbca() const { return Mem::permute<X3, X1, X2, X0>(data()); }
template<typename T> Vc_INTRINSIC Vc_PURE const Vector<T>  Vector<T>::dcba() const { return Mem::permute<X3, X2, X1, X0>(data()); }

#define VC_SWIZZLES_16BIT_IMPL(T) \
template<> Vc_INTRINSIC Vc_PURE const Vector<T> Vector<T>::cdab() const { return Mem::permute<X2, X3, X0, X1, X6, X7, X4, X5>(data()); } \
template<> Vc_INTRINSIC Vc_PURE const Vector<T> Vector<T>::badc() const { return Mem::permute<X1, X0, X3, X2, X5, X4, X7, X6>(data()); } \
template<> Vc_INTRINSIC Vc_PURE const Vector<T> Vector<T>::aaaa() const { return Mem::permute<X0, X0, X0, X0, X4, X4, X4, X4>(data()); } \
template<> Vc_INTRINSIC Vc_PURE const Vector<T> Vector<T>::bbbb() const { return Mem::permute<X1, X1, X1, X1, X5, X5, X5, X5>(data()); } \
template<> Vc_INTRINSIC Vc_PURE const Vector<T> Vector<T>::cccc() const { return Mem::permute<X2, X2, X2, X2, X6, X6, X6, X6>(data()); } \
template<> Vc_INTRINSIC Vc_PURE const Vector<T> Vector<T>::dddd() const { return Mem::permute<X3, X3, X3, X3, X7, X7, X7, X7>(data()); } \
template<> Vc_INTRINSIC Vc_PURE const Vector<T> Vector<T>::bcad() const { return Mem::permute<X1, X2, X0, X3, X5, X6, X4, X7>(data()); } \
template<> Vc_INTRINSIC Vc_PURE const Vector<T> Vector<T>::bcda() const { return Mem::permute<X1, X2, X3, X0, X5, X6, X7, X4>(data()); } \
template<> Vc_INTRINSIC Vc_PURE const Vector<T> Vector<T>::dabc() const { return Mem::permute<X3, X0, X1, X2, X7, X4, X5, X6>(data()); } \
template<> Vc_INTRINSIC Vc_PURE const Vector<T> Vector<T>::acbd() const { return Mem::permute<X0, X2, X1, X3, X4, X6, X5, X7>(data()); } \
template<> Vc_INTRINSIC Vc_PURE const Vector<T> Vector<T>::dbca() const { return Mem::permute<X3, X1, X2, X0, X7, X5, X6, X4>(data()); } \
template<> Vc_INTRINSIC Vc_PURE const Vector<T> Vector<T>::dcba() const { return Mem::permute<X3, X2, X1, X0, X7, X6, X5, X4>(data()); }
VC_SWIZZLES_16BIT_IMPL(short)
VC_SWIZZLES_16BIT_IMPL(unsigned short)
#undef VC_SWIZZLES_16BIT_IMPL

// isNegative {{{1
template<> Vc_INTRINSIC Vc_PURE float_m float_v::isNegative() const
{
    return sse_cast<__m128>(_mm_srai_epi32(sse_cast<__m128i>(_mm_and_ps(_mm_setsignmask_ps(), d.v())), 31));
}
template<> Vc_INTRINSIC Vc_PURE double_m double_v::isNegative() const
{
    return Mem::permute<X1, X1, X3, X3>(sse_cast<__m128>(
                _mm_srai_epi32(sse_cast<__m128i>(_mm_and_pd(_mm_setsignmask_pd(), d.v())), 31)
                ));
}
// gathers {{{1
template <>
template <typename MT, typename IT>
Vc_ALWAYS_INLINE void double_v::gatherImplementation(const MT *mem, IT &&indexes)
{
    d.v() = _mm_setr_pd(mem[indexes[0]], mem[indexes[1]]);
}

template <>
template <typename MT, typename IT>
Vc_ALWAYS_INLINE void float_v::gatherImplementation(const MT *mem, IT &&indexes)
{
    d.v() = _mm_setr_ps(mem[indexes[0]], mem[indexes[1]], mem[indexes[2]], mem[indexes[3]]);
}

template <>
template <typename MT, typename IT>
Vc_ALWAYS_INLINE void int_v::gatherImplementation(const MT *mem, IT &&indexes)
{
    d.v() = _mm_setr_epi32(mem[indexes[0]], mem[indexes[1]], mem[indexes[2]], mem[indexes[3]]);
}

template <>
template <typename MT, typename IT>
Vc_ALWAYS_INLINE void uint_v::gatherImplementation(const MT *mem, IT &&indexes)
{
    d.v() = _mm_setr_epi32(mem[indexes[0]], mem[indexes[1]], mem[indexes[2]], mem[indexes[3]]);
}

template <>
template <typename MT, typename IT>
Vc_ALWAYS_INLINE void short_v::gatherImplementation(const MT *mem, IT &&indexes)
{
    d.v() = set(mem[indexes[0]], mem[indexes[1]], mem[indexes[2]], mem[indexes[3]],
            mem[indexes[4]], mem[indexes[5]], mem[indexes[6]], mem[indexes[7]]);
}

template <>
template <typename MT, typename IT>
Vc_ALWAYS_INLINE void ushort_v::gatherImplementation(const MT *mem, IT &&indexes)
{
    d.v() = set(mem[indexes[0]], mem[indexes[1]], mem[indexes[2]], mem[indexes[3]],
                mem[indexes[4]], mem[indexes[5]], mem[indexes[6]], mem[indexes[7]]);
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

// scatters {{{1
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

///////////////////////////////////////////////////////////////////////////////////////////
// operator[] {{{1
template<typename T> Vc_INTRINSIC typename Vector<T>::EntryType Vc_PURE Vector<T>::operator[](size_t index) const
{
#ifdef VC_CLANG
    typedef typename Common::GccTypeHelper<T, VectorType>::Type TV;
    return static_cast<TV>(d.v())[index];
#else
    return d.m(index);
#endif
}
#ifdef VC_GCC
template<> Vc_INTRINSIC double Vc_PURE Vector<double>::operator[](size_t index) const
{
    if (__builtin_constant_p(index)) {
        return extract_double_imm(d.v(), index);
    }
    return d.m(index);
}
template<> Vc_INTRINSIC float Vc_PURE Vector<float>::operator[](size_t index) const
{
    return extract_float(d.v(), index);
}
template<> Vc_INTRINSIC int Vc_PURE Vector<int>::operator[](size_t index) const
{
    if (__builtin_constant_p(index)) {
#ifdef __x86_64__
        if (index == 0) return _mm_cvtsi128_si64(d.v()) & 0xFFFFFFFFull;
        if (index == 1) return _mm_cvtsi128_si64(d.v()) >> 32;
#else
        if (index == 0) return _mm_cvtsi128_si32(d.v());
#endif
#ifdef VC_IMPL_SSE4_1
        return _mm_extract_epi32(d.v(), index);
#endif
    }
    return d.m(index);
}
template<> Vc_INTRINSIC unsigned int Vc_PURE Vector<unsigned int>::operator[](size_t index) const
{
    if (__builtin_constant_p(index)) {
#ifdef __x86_64__
        if (index == 0) return _mm_cvtsi128_si64(d.v()) & 0xFFFFFFFFull;
        if (index == 1) return _mm_cvtsi128_si64(d.v()) >> 32;
#else
        if (index == 0) return _mm_cvtsi128_si32(d.v());
#endif
#ifdef VC_IMPL_SSE4_1
        return _mm_extract_epi32(d.v(), index);
#endif
    }
    return d.m(index);
}
template<> Vc_INTRINSIC short Vc_PURE Vector<short>::operator[](size_t index) const
{
    if (__builtin_constant_p(index)) {
        return _mm_extract_epi16(d.v(), index);
    }
    return d.m(index);
}
template<> Vc_INTRINSIC unsigned short Vc_PURE Vector<unsigned short>::operator[](size_t index) const
{
    if (__builtin_constant_p(index)) {
        return _mm_extract_epi16(d.v(), index);
    }
    return d.m(index);
}
#endif // GCC
///////////////////////////////////////////////////////////////////////////////////////////
// horizontal ops {{{1
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
#ifndef VC_IMPL_SSE4_1
// without SSE4.1 integer multiplication is slow and we rather multiply the scalars
template<> Vc_INTRINSIC Vc_PURE int Vector<int>::product() const
{
    return (d.m(0) * d.m(1)) * (d.m(2) * d.m(3));
}
template<> Vc_INTRINSIC Vc_PURE unsigned int Vector<unsigned int>::product() const
{
    return (d.m(0) * d.m(1)) * (d.m(2) * d.m(3));
}
#endif
template<typename T> Vc_ALWAYS_INLINE Vc_PURE typename Vector<T>::EntryType Vector<T>::min(MaskArg m) const
{
    Vector<T> tmp = std::numeric_limits<Vector<T> >::max();
    tmp(m) = *this;
    return tmp.min();
}
template<typename T> Vc_ALWAYS_INLINE Vc_PURE typename Vector<T>::EntryType Vector<T>::max(MaskArg m) const
{
    Vector<T> tmp = std::numeric_limits<Vector<T> >::min();
    tmp(m) = *this;
    return tmp.max();
}
template<typename T> Vc_ALWAYS_INLINE Vc_PURE typename Vector<T>::EntryType Vector<T>::product(MaskArg m) const
{
    Vector<T> tmp(VectorSpecialInitializerOne::One);
    tmp(m) = *this;
    return tmp.product();
}
template<typename T> Vc_ALWAYS_INLINE Vc_PURE typename Vector<T>::EntryType Vector<T>::sum(MaskArg m) const
{
    Vector<T> tmp(VectorSpecialInitializerZero::Zero);
    tmp(m) = *this;
    return tmp.sum();
}

///////////////////////////////////////////////////////////////////////////////////////////
// copySign {{{1
template<> Vc_INTRINSIC Vc_PURE Vector<float> Vector<float>::copySign(Vector<float>::AsArg reference) const
{
    return _mm_or_ps(
            _mm_and_ps(reference.d.v(), _mm_setsignmask_ps()),
            _mm_and_ps(d.v(), _mm_setabsmask_ps())
            );
}
template<> Vc_INTRINSIC Vc_PURE Vector<double> Vector<double>::copySign(Vector<double>::AsArg reference) const
{
    return _mm_or_pd(
            _mm_and_pd(reference.d.v(), _mm_setsignmask_pd()),
            _mm_and_pd(d.v(), _mm_setabsmask_pd())
            );
}//}}}1
// exponent {{{1
template<> Vc_INTRINSIC Vc_PURE Vector<float> Vector<float>::exponent() const
{
    VC_ASSERT((*this >= 0.f).isFull());
    return Internal::exponent(d.v());
}
template<> Vc_INTRINSIC Vc_PURE Vector<double> Vector<double>::exponent() const
{
    VC_ASSERT((*this >= 0.).isFull());
    return Internal::exponent(d.v());
}
// }}}1
// Random {{{1
static void _doRandomStep(Vector<unsigned int> &state0,
        Vector<unsigned int> &state1)
{
    state0.load(&Common::RandomState[0]);
    state1.load(&Common::RandomState[uint_v::Size]);
    (state1 * 0xdeece66du + 11).store(&Common::RandomState[uint_v::Size]);
    uint_v(_mm_xor_si128((state0 * 0xdeece66du + 11).data(), _mm_srli_epi32(state1.data(), 16))).store(&Common::RandomState[0]);
}

template<typename T> Vc_ALWAYS_INLINE Vector<T> Vector<T>::Random()
{
    Vector<unsigned int> state0, state1;
    _doRandomStep(state0, state1);
    return state0.reinterpretCast<Vector<T> >();
}

template<> Vc_ALWAYS_INLINE Vector<float> Vector<float>::Random()
{
    Vector<unsigned int> state0, state1;
    _doRandomStep(state0, state1);
    return _mm_sub_ps(_mm_or_ps(_mm_castsi128_ps(_mm_srli_epi32(state0.data(), 2)), HT::one()), HT::one());
}

template<> Vc_ALWAYS_INLINE Vector<double> Vector<double>::Random()
{
    typedef unsigned long long uint64 Vc_MAY_ALIAS;
    uint64 state0 = *reinterpret_cast<const uint64 *>(&Common::RandomState[8]);
    uint64 state1 = *reinterpret_cast<const uint64 *>(&Common::RandomState[10]);
    const __m128i state = _mm_load_si128(reinterpret_cast<const __m128i *>(&Common::RandomState[8]));
    *reinterpret_cast<uint64 *>(&Common::RandomState[ 8]) = (state0 * 0x5deece66dull + 11);
    *reinterpret_cast<uint64 *>(&Common::RandomState[10]) = (state1 * 0x5deece66dull + 11);
    return _mm_sub_pd(_mm_or_pd(_mm_castsi128_pd(_mm_srli_epi64(state, 12)), HT::one()), HT::one());
}
// shifted / rotated {{{1
template<typename T> Vc_INTRINSIC Vc_PURE Vector<T> Vector<T>::shifted(int amount) const
{
    enum {
        EntryTypeSizeof = sizeof(EntryType)
    };
    switch (amount) {
    case  0: return *this;
    case  1: return mm128_reinterpret_cast<VectorType>(_mm_srli_si128(mm128_reinterpret_cast<__m128i>(d.v()), 1 * EntryTypeSizeof));
    case  2: return mm128_reinterpret_cast<VectorType>(_mm_srli_si128(mm128_reinterpret_cast<__m128i>(d.v()), 2 * EntryTypeSizeof));
    case  3: return mm128_reinterpret_cast<VectorType>(_mm_srli_si128(mm128_reinterpret_cast<__m128i>(d.v()), 3 * EntryTypeSizeof));
    case  4: return mm128_reinterpret_cast<VectorType>(_mm_srli_si128(mm128_reinterpret_cast<__m128i>(d.v()), 4 * EntryTypeSizeof));
    case  5: return mm128_reinterpret_cast<VectorType>(_mm_srli_si128(mm128_reinterpret_cast<__m128i>(d.v()), 5 * EntryTypeSizeof));
    case  6: return mm128_reinterpret_cast<VectorType>(_mm_srli_si128(mm128_reinterpret_cast<__m128i>(d.v()), 6 * EntryTypeSizeof));
    case  7: return mm128_reinterpret_cast<VectorType>(_mm_srli_si128(mm128_reinterpret_cast<__m128i>(d.v()), 7 * EntryTypeSizeof));
    case  8: return mm128_reinterpret_cast<VectorType>(_mm_srli_si128(mm128_reinterpret_cast<__m128i>(d.v()), 8 * EntryTypeSizeof));
    case -1: return mm128_reinterpret_cast<VectorType>(_mm_slli_si128(mm128_reinterpret_cast<__m128i>(d.v()), 1 * EntryTypeSizeof));
    case -2: return mm128_reinterpret_cast<VectorType>(_mm_slli_si128(mm128_reinterpret_cast<__m128i>(d.v()), 2 * EntryTypeSizeof));
    case -3: return mm128_reinterpret_cast<VectorType>(_mm_slli_si128(mm128_reinterpret_cast<__m128i>(d.v()), 3 * EntryTypeSizeof));
    case -4: return mm128_reinterpret_cast<VectorType>(_mm_slli_si128(mm128_reinterpret_cast<__m128i>(d.v()), 4 * EntryTypeSizeof));
    case -5: return mm128_reinterpret_cast<VectorType>(_mm_slli_si128(mm128_reinterpret_cast<__m128i>(d.v()), 5 * EntryTypeSizeof));
    case -6: return mm128_reinterpret_cast<VectorType>(_mm_slli_si128(mm128_reinterpret_cast<__m128i>(d.v()), 6 * EntryTypeSizeof));
    case -7: return mm128_reinterpret_cast<VectorType>(_mm_slli_si128(mm128_reinterpret_cast<__m128i>(d.v()), 7 * EntryTypeSizeof));
    case -8: return mm128_reinterpret_cast<VectorType>(_mm_slli_si128(mm128_reinterpret_cast<__m128i>(d.v()), 8 * EntryTypeSizeof));
    }
    return Zero();
}
template<typename T> Vc_INTRINSIC Vector<T> Vector<T>::shifted(int amount, Vector shiftIn) const
{
    return shifted(amount) | (amount > 0 ?
                              shiftIn.shifted(amount - Size) :
                              shiftIn.shifted(Size + amount));
}
template<typename T> Vc_INTRINSIC Vc_PURE Vector<T> Vector<T>::rotated(int amount) const
{
    enum {
        EntryTypeSizeof = sizeof(EntryType)
    };
    const __m128i v = mm128_reinterpret_cast<__m128i>(d.v());
    switch (static_cast<unsigned int>(amount) % Size) {
    case  0: return *this;
    case  1: return mm128_reinterpret_cast<VectorType>(alignr_epi8<1 * EntryTypeSizeof>(v, v));
    case  2: return mm128_reinterpret_cast<VectorType>(alignr_epi8<2 * EntryTypeSizeof>(v, v));
    case  3: return mm128_reinterpret_cast<VectorType>(alignr_epi8<3 * EntryTypeSizeof>(v, v));
             // warning "Immediate parameter to intrinsic call too large" disabled in VcMacros.cmake.
             // ICC fails to see that the modulo operation (Size == sizeof(VectorType) / sizeof(EntryType))
             // disables the following four calls unless sizeof(EntryType) == 2.
    case  4: return mm128_reinterpret_cast<VectorType>(alignr_epi8<4 * EntryTypeSizeof>(v, v));
    case  5: return mm128_reinterpret_cast<VectorType>(alignr_epi8<5 * EntryTypeSizeof>(v, v));
    case  6: return mm128_reinterpret_cast<VectorType>(alignr_epi8<6 * EntryTypeSizeof>(v, v));
    case  7: return mm128_reinterpret_cast<VectorType>(alignr_epi8<7 * EntryTypeSizeof>(v, v));
    }
    return Zero();
}
// sorted specializations {{{1
template<> inline Vc_PURE uint_v uint_v::sorted() const
{
    __m128i x = data();
    __m128i y = _mm_shuffle_epi32(x, _MM_SHUFFLE(2, 3, 0, 1));
    __m128i l = min_epu32(x, y);
    __m128i h = max_epu32(x, y);
    x = _mm_unpacklo_epi32(l, h);
    y = _mm_unpackhi_epi32(h, l);

    // sort quads
    l = min_epu32(x, y);
    h = max_epu32(x, y);
    x = _mm_unpacklo_epi32(l, h);
    y = _mm_unpackhi_epi64(x, x);

    l = min_epu32(x, y);
    h = max_epu32(x, y);
    return _mm_unpacklo_epi32(l, h);
}
template<> inline Vc_PURE ushort_v ushort_v::sorted() const
{
    __m128i lo, hi, y, x = data();
    // sort pairs
    y = Mem::permute<X1, X0, X3, X2, X5, X4, X7, X6>(x);
    lo = min_epu16(x, y);
    hi = max_epu16(x, y);
    x = blend_epi16<0xaa>(lo, hi);

    // merge left and right quads
    y = Mem::permute<X3, X2, X1, X0, X7, X6, X5, X4>(x);
    lo = min_epu16(x, y);
    hi = max_epu16(x, y);
    x = blend_epi16<0xcc>(lo, hi);
    y = _mm_srli_si128(x, 2);
    lo = min_epu16(x, y);
    hi = max_epu16(x, y);
    x = blend_epi16<0xaa>(lo, _mm_slli_si128(hi, 2));

    // merge quads into octs
    y = _mm_shuffle_epi32(x, _MM_SHUFFLE(1, 0, 3, 2));
    y = _mm_shufflelo_epi16(y, _MM_SHUFFLE(0, 1, 2, 3));
    lo = min_epu16(x, y);
    hi = max_epu16(x, y);

    x = _mm_unpacklo_epi16(lo, hi);
    y = _mm_srli_si128(x, 8);
    lo = min_epu16(x, y);
    hi = max_epu16(x, y);

    x = _mm_unpacklo_epi16(lo, hi);
    y = _mm_srli_si128(x, 8);
    lo = min_epu16(x, y);
    hi = max_epu16(x, y);

    return _mm_unpacklo_epi16(lo, hi);
}
// interleaveLow/-High {{{1
template <> Vc_INTRINSIC double_v double_v::interleaveLow (double_v x) const { return _mm_unpacklo_pd(data(), x.data()); }
template <> Vc_INTRINSIC double_v double_v::interleaveHigh(double_v x) const { return _mm_unpackhi_pd(data(), x.data()); }
template <> Vc_INTRINSIC  float_v  float_v::interleaveLow ( float_v x) const { return _mm_unpacklo_ps(data(), x.data()); }
template <> Vc_INTRINSIC  float_v  float_v::interleaveHigh( float_v x) const { return _mm_unpackhi_ps(data(), x.data()); }
template <> Vc_INTRINSIC    int_v    int_v::interleaveLow (   int_v x) const { return _mm_unpacklo_epi32(data(), x.data()); }
template <> Vc_INTRINSIC    int_v    int_v::interleaveHigh(   int_v x) const { return _mm_unpackhi_epi32(data(), x.data()); }
template <> Vc_INTRINSIC   uint_v   uint_v::interleaveLow (  uint_v x) const { return _mm_unpacklo_epi32(data(), x.data()); }
template <> Vc_INTRINSIC   uint_v   uint_v::interleaveHigh(  uint_v x) const { return _mm_unpackhi_epi32(data(), x.data()); }
template <> Vc_INTRINSIC  short_v  short_v::interleaveLow ( short_v x) const { return _mm_unpacklo_epi16(data(), x.data()); }
template <> Vc_INTRINSIC  short_v  short_v::interleaveHigh( short_v x) const { return _mm_unpackhi_epi16(data(), x.data()); }
template <> Vc_INTRINSIC ushort_v ushort_v::interleaveLow (ushort_v x) const { return _mm_unpacklo_epi16(data(), x.data()); }
template <> Vc_INTRINSIC ushort_v ushort_v::interleaveHigh(ushort_v x) const { return _mm_unpackhi_epi16(data(), x.data()); }
// }}}1
// generate {{{1
template <> template <typename G> Vc_INTRINSIC double_v double_v::generate(G gen)
{
    const auto tmp0 = gen(0);
    const auto tmp1 = gen(1);
    return _mm_setr_pd(tmp0, tmp1);
}
template <> template <typename G> Vc_INTRINSIC float_v float_v::generate(G gen)
{
    const auto tmp0 = gen(0);
    const auto tmp1 = gen(1);
    const auto tmp2 = gen(2);
    const auto tmp3 = gen(3);
    return _mm_setr_ps(tmp0, tmp1, tmp2, tmp3);
}
template <> template <typename G> Vc_INTRINSIC int_v int_v::generate(G gen)
{
    const auto tmp0 = gen(0);
    const auto tmp1 = gen(1);
    const auto tmp2 = gen(2);
    const auto tmp3 = gen(3);
    return _mm_setr_epi32(tmp0, tmp1, tmp2, tmp3);
}
template <> template <typename G> Vc_INTRINSIC uint_v uint_v::generate(G gen)
{
    const auto tmp0 = gen(0);
    const auto tmp1 = gen(1);
    const auto tmp2 = gen(2);
    const auto tmp3 = gen(3);
    return _mm_setr_epi32(tmp0, tmp1, tmp2, tmp3);
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
    return Mem::permute<X1, X0>(d.v());
}
template <> Vc_INTRINSIC Vc_PURE float_v float_v::reversed() const
{
    return Mem::permute<X3, X2, X1, X0>(d.v());
}
template <> Vc_INTRINSIC Vc_PURE int_v int_v::reversed() const
{
    return Mem::permute<X3, X2, X1, X0>(d.v());
}
template <> Vc_INTRINSIC Vc_PURE uint_v uint_v::reversed() const
{
    return Mem::permute<X3, X2, X1, X0>(d.v());
}
template <> Vc_INTRINSIC Vc_PURE short_v short_v::reversed() const
{
    return sse_cast<__m128i>(
        Mem::shuffle<X1, Y0>(sse_cast<__m128d>(Mem::permuteHi<X7, X6, X5, X4>(d.v())),
                             sse_cast<__m128d>(Mem::permuteLo<X3, X2, X1, X0>(d.v()))));
}
template <> Vc_INTRINSIC Vc_PURE ushort_v ushort_v::reversed() const
{
    return sse_cast<__m128i>(
        Mem::shuffle<X1, Y0>(sse_cast<__m128d>(Mem::permuteHi<X7, X6, X5, X4>(d.v())),
                             sse_cast<__m128d>(Mem::permuteLo<X3, X2, X1, X0>(d.v()))));
}
// }}}1
// permutation via operator[] {{{1
template <>
Vc_INTRINSIC float_v float_v::operator[](int_v
#ifdef VC_IMPL_AVX
                                             perm
#endif
                                         ) const
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
#ifdef VC_IMPL_AVX
    return _mm_permutevar_ps(d.v(), perm.data());
#else
    return *this;//TODO
#endif
}
// broadcast from constexpr index {{{1
template <> template <int Index> Vc_INTRINSIC float_v float_v::broadcast() const
{
    constexpr VecPos Inner = static_cast<VecPos>(Index & 0x3);
    return Mem::permute<Inner, Inner, Inner, Inner>(d.v());
}
template <> template <int Index> Vc_INTRINSIC double_v double_v::broadcast() const
{
    constexpr VecPos Inner = static_cast<VecPos>(Index & 0x1);
    return Mem::permute<Inner, Inner>(d.v());
}
// }}}1
}

namespace Common
{
// transpose_impl {{{1
template <int L>
Vc_ALWAYS_INLINE enable_if<L == 4, void> transpose_impl(
    SSE::float_v *VC_RESTRICT r[],
    const TransposeProxy<SSE::float_v, SSE::float_v, SSE::float_v, SSE::float_v> &proxy)
{
    const auto in0 = std::get<0>(proxy.in).data();
    const auto in1 = std::get<1>(proxy.in).data();
    const auto in2 = std::get<2>(proxy.in).data();
    const auto in3 = std::get<3>(proxy.in).data();
    const auto tmp0 = _mm_unpacklo_ps(in0, in2);
    const auto tmp1 = _mm_unpacklo_ps(in1, in3);
    const auto tmp2 = _mm_unpackhi_ps(in0, in2);
    const auto tmp3 = _mm_unpackhi_ps(in1, in3);
    *r[0] = _mm_unpacklo_ps(tmp0, tmp1);
    *r[1] = _mm_unpackhi_ps(tmp0, tmp1);
    *r[2] = _mm_unpacklo_ps(tmp2, tmp3);
    *r[3] = _mm_unpackhi_ps(tmp2, tmp3);
}
// }}}1
}  // namespace Common
}

#include "undomacros.h"

// vim: foldmethod=marker
