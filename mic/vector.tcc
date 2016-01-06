/*  This file is part of the Vc library. {{{
Copyright © 2010-2015 Matthias Kretz <kretz@kde.org>

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

#include <type_traits>
#include "../common/x86_prefetches.h"
#include "debug.h"
#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace Detail
{
// bitwise operators {{{1
template <typename T>
Vc_INTRINSIC MIC::Vector<T> operator^(MIC::Vector<T> a, MIC::Vector<T> b)
{
    return xor_(a.data(), b.data());
}
template <typename T>
Vc_INTRINSIC MIC::Vector<T> operator&(MIC::Vector<T> a, MIC::Vector<T> b)
{
    return and_(a.data(), b.data());
}
template <typename T>
Vc_INTRINSIC MIC::Vector<T> operator|(MIC::Vector<T> a, MIC::Vector<T> b)
{
    return or_(a.data(), b.data());
}

// compare operators {{{1
Vc_INTRINSIC MIC::double_m operator==(MIC::double_v a, MIC::double_v b) { return _mm512_cmp_pd_mask(a.data(), b.data(), _CMP_EQ_OQ); }
Vc_INTRINSIC MIC:: float_m operator==(MIC:: float_v a, MIC:: float_v b) { return _mm512_cmp_ps_mask(a.data(), b.data(), _CMP_EQ_OQ); }
Vc_INTRINSIC MIC::   int_m operator==(MIC::   int_v a, MIC::   int_v b) { return _mm512_cmp_epi32_mask(a.data(), b.data(), _MM_CMPINT_EQ); }
Vc_INTRINSIC MIC::  uint_m operator==(MIC::  uint_v a, MIC::  uint_v b) { return _mm512_cmp_epu32_mask(a.data(), b.data(), _MM_CMPINT_EQ); }
Vc_INTRINSIC MIC:: short_m operator==(MIC:: short_v a, MIC:: short_v b) { return _mm512_cmp_epi32_mask(a.data(), b.data(), _MM_CMPINT_EQ); }

Vc_INTRINSIC MIC::double_m operator!=(MIC::double_v a, MIC::double_v b) { return _mm512_cmp_pd_mask(a.data(), b.data(), _CMP_NEQ_UQ); }
Vc_INTRINSIC MIC:: float_m operator!=(MIC:: float_v a, MIC:: float_v b) { return _mm512_cmp_ps_mask(a.data(), b.data(), _CMP_NEQ_UQ); }
Vc_INTRINSIC MIC::   int_m operator!=(MIC::   int_v a, MIC::   int_v b) { return _mm512_cmp_epi32_mask(a.data(), b.data(), _MM_CMPINT_NE); }
Vc_INTRINSIC MIC::  uint_m operator!=(MIC::  uint_v a, MIC::  uint_v b) { return _mm512_cmp_epu32_mask(a.data(), b.data(), _MM_CMPINT_NE); }
Vc_INTRINSIC MIC:: short_m operator!=(MIC:: short_v a, MIC:: short_v b) { return _mm512_cmp_epi32_mask(a.data(), b.data(), _MM_CMPINT_NE); }

Vc_INTRINSIC MIC::double_m operator>=(MIC::double_v a, MIC::double_v b) { return _mm512_cmp_pd_mask(a.data(), b.data(), _CMP_NLT_US); }
Vc_INTRINSIC MIC:: float_m operator>=(MIC:: float_v a, MIC:: float_v b) { return _mm512_cmp_ps_mask(a.data(), b.data(), _CMP_NLT_US); }
Vc_INTRINSIC MIC::   int_m operator>=(MIC::   int_v a, MIC::   int_v b) { return _mm512_cmp_epi32_mask(a.data(), b.data(), _MM_CMPINT_NLT); }
Vc_INTRINSIC MIC::  uint_m operator>=(MIC::  uint_v a, MIC::  uint_v b) { return _mm512_cmp_epu32_mask(a.data(), b.data(), _MM_CMPINT_NLT); }
Vc_INTRINSIC MIC:: short_m operator>=(MIC:: short_v a, MIC:: short_v b) { return _mm512_cmp_epi32_mask(a.data(), b.data(), _MM_CMPINT_NLT); }

Vc_INTRINSIC MIC::double_m operator<=(MIC::double_v a, MIC::double_v b) { return _mm512_cmp_pd_mask(a.data(), b.data(), _CMP_LE_OS); }
Vc_INTRINSIC MIC:: float_m operator<=(MIC:: float_v a, MIC:: float_v b) { return _mm512_cmp_ps_mask(a.data(), b.data(), _CMP_LE_OS); }
Vc_INTRINSIC MIC::   int_m operator<=(MIC::   int_v a, MIC::   int_v b) { return _mm512_cmp_epi32_mask(a.data(), b.data(), _MM_CMPINT_LE); }
Vc_INTRINSIC MIC::  uint_m operator<=(MIC::  uint_v a, MIC::  uint_v b) { return _mm512_cmp_epu32_mask(a.data(), b.data(), _MM_CMPINT_LE); }
Vc_INTRINSIC MIC:: short_m operator<=(MIC:: short_v a, MIC:: short_v b) { return _mm512_cmp_epi32_mask(a.data(), b.data(), _MM_CMPINT_LE); }

Vc_INTRINSIC MIC::double_m operator> (MIC::double_v a, MIC::double_v b) { return _mm512_cmp_pd_mask(a.data(), b.data(), _CMP_NLE_US); }
Vc_INTRINSIC MIC:: float_m operator> (MIC:: float_v a, MIC:: float_v b) { return _mm512_cmp_ps_mask(a.data(), b.data(), _CMP_NLE_US); }
Vc_INTRINSIC MIC::   int_m operator> (MIC::   int_v a, MIC::   int_v b) { return _mm512_cmp_epi32_mask(a.data(), b.data(), _MM_CMPINT_NLE); }
Vc_INTRINSIC MIC::  uint_m operator> (MIC::  uint_v a, MIC::  uint_v b) { return _mm512_cmp_epu32_mask(a.data(), b.data(), _MM_CMPINT_NLE); }
Vc_INTRINSIC MIC:: short_m operator> (MIC:: short_v a, MIC:: short_v b) { return _mm512_cmp_epi32_mask(a.data(), b.data(), _MM_CMPINT_NLE); }

Vc_INTRINSIC MIC::double_m operator< (MIC::double_v a, MIC::double_v b) { return _mm512_cmp_pd_mask(a.data(), b.data(), _CMP_LT_OS); }
Vc_INTRINSIC MIC:: float_m operator< (MIC:: float_v a, MIC:: float_v b) { return _mm512_cmp_ps_mask(a.data(), b.data(), _CMP_LT_OS); }
Vc_INTRINSIC MIC::   int_m operator< (MIC::   int_v a, MIC::   int_v b) { return _mm512_cmp_epi32_mask(a.data(), b.data(), _MM_CMPINT_LT); }
Vc_INTRINSIC MIC::  uint_m operator< (MIC::  uint_v a, MIC::  uint_v b) { return _mm512_cmp_epu32_mask(a.data(), b.data(), _MM_CMPINT_LT); }
Vc_INTRINSIC MIC:: short_m operator< (MIC:: short_v a, MIC:: short_v b) { return _mm512_cmp_epi32_mask(a.data(), b.data(), _MM_CMPINT_LT); }

// only unsigned integers have well-defined behavior on over-/underflow
Vc_INTRINSIC MIC::ushort_m operator==(MIC::ushort_v a, MIC::ushort_v b) { return _mm512_cmpeq_epu32_mask(and_(a.data(), MIC::_set1(0xffff)), and_(b.data(), MIC::_set1(0xffff))); }
Vc_INTRINSIC MIC::ushort_m operator!=(MIC::ushort_v a, MIC::ushort_v b) { return _mm512_cmpneq_epu32_mask(and_(a.data(), MIC::_set1(0xffff)), and_(b.data(), MIC::_set1(0xffff))); }
Vc_INTRINSIC MIC::ushort_m operator>=(MIC::ushort_v a, MIC::ushort_v b) { return _mm512_cmpge_epu32_mask(and_(a.data(), MIC::_set1(0xffff)), and_(b.data(), MIC::_set1(0xffff))); }
Vc_INTRINSIC MIC::ushort_m operator<=(MIC::ushort_v a, MIC::ushort_v b) { return _mm512_cmple_epu32_mask(and_(a.data(), MIC::_set1(0xffff)), and_(b.data(), MIC::_set1(0xffff))); }
Vc_INTRINSIC MIC::ushort_m operator> (MIC::ushort_v a, MIC::ushort_v b) { return _mm512_cmpgt_epu32_mask(and_(a.data(), MIC::_set1(0xffff)), and_(b.data(), MIC::_set1(0xffff))); }
Vc_INTRINSIC MIC::ushort_m operator< (MIC::ushort_v a, MIC::ushort_v b) { return _mm512_cmplt_epu32_mask(and_(a.data(), MIC::_set1(0xffff)), and_(b.data(), MIC::_set1(0xffff))); }

// only unsigned integers have well-defined behavior on over-/underflow
Vc_INTRINSIC MIC::uchar_m operator==(MIC::uchar_v a, MIC::uchar_v b) { return _mm512_cmpeq_epu32_mask (and_(a.data(), MIC::_set1(0xff)), and_(b.data(), MIC::_set1(0xff))); }
Vc_INTRINSIC MIC::uchar_m operator!=(MIC::uchar_v a, MIC::uchar_v b) { return _mm512_cmpneq_epu32_mask(and_(a.data(), MIC::_set1(0xff)), and_(b.data(), MIC::_set1(0xff))); }
Vc_INTRINSIC MIC::uchar_m operator>=(MIC::uchar_v a, MIC::uchar_v b) { return _mm512_cmpge_epu32_mask (and_(a.data(), MIC::_set1(0xff)), and_(b.data(), MIC::_set1(0xff))); }
Vc_INTRINSIC MIC::uchar_m operator> (MIC::uchar_v a, MIC::uchar_v b) { return _mm512_cmpgt_epu32_mask (and_(a.data(), MIC::_set1(0xff)), and_(b.data(), MIC::_set1(0xff))); }
Vc_INTRINSIC MIC::uchar_m operator<=(MIC::uchar_v a, MIC::uchar_v b) { return _mm512_cmple_epu32_mask (and_(a.data(), MIC::_set1(0xff)), and_(b.data(), MIC::_set1(0xff))); }
Vc_INTRINSIC MIC::uchar_m operator< (MIC::uchar_v a, MIC::uchar_v b) { return _mm512_cmplt_epu32_mask (and_(a.data(), MIC::_set1(0xff)), and_(b.data(), MIC::_set1(0xff))); }

// arithmetic operators {{{1
template <typename T>
Vc_INTRINSIC MIC::Vector<T> operator+(MIC::Vector<T> a, MIC::Vector<T> b)
{
    return add(a.data(), b.data(), T());
}
template <typename T>
Vc_INTRINSIC MIC::Vector<T> operator-(MIC::Vector<T> a, MIC::Vector<T> b)
{
    return sub(a.data(), b.data(), T());
}
template <typename T>
Vc_INTRINSIC MIC::Vector<T> operator*(MIC::Vector<T> a, MIC::Vector<T> b)
{
    return mul(a.data(), b.data(), T());
}
template <typename T>
Vc_INTRINSIC MIC::Vector<T> operator/(MIC::Vector<T> a, MIC::Vector<T> b)
{
    return div(a.data(), b.data(), T());
}
template <typename T>
Vc_INTRINSIC enable_if<std::is_integral<T>::value, MIC::Vector<T>> operator%(
    MIC::Vector<T> a, MIC::Vector<T> b)
{
    return rem(a.data(), b.data(), T());
    //return a - a / b * b;
}
// }}}1
}  // namespace Detail

// LoadHelper {{{1
namespace
{

template<typename V, typename T> struct LoadHelper2;

template<typename V> struct LoadHelper/*{{{*/
{
    typedef typename V::VectorType VectorType;

    static Vc_INTRINSIC VectorType _load(const void *m, _MM_UPCONV_PS_ENUM upconv, int memHint) {
        return _mm512_extload_ps(m, upconv, _MM_BROADCAST32_NONE, memHint);
    }
    static Vc_INTRINSIC VectorType _load(const void *m, _MM_UPCONV_PD_ENUM upconv, int memHint) {
        return _mm512_extload_pd(m, upconv, _MM_BROADCAST64_NONE, memHint);
    }
    static Vc_INTRINSIC VectorType _load(const void *m, _MM_UPCONV_EPI32_ENUM upconv, int memHint) {
        return _mm512_extload_epi32(m, upconv, _MM_BROADCAST32_NONE, memHint);
    }

    static Vc_INTRINSIC VectorType _loadu(const void *m, _MM_UPCONV_PS_ENUM upconv, int memHint) {
        return MIC::mm512_loadu_ps(m, upconv, memHint);
    }
    static Vc_INTRINSIC VectorType _loadu(const void *m, _MM_UPCONV_PD_ENUM upconv, int memHint) {
        return MIC::mm512_loadu_pd(m, upconv, memHint);
    }
    static Vc_INTRINSIC VectorType _loadu(const void *m, _MM_UPCONV_EPI32_ENUM upconv, int memHint) {
        return MIC::mm512_loadu_epi32(m, upconv, memHint);
    }

    template<typename T, typename Flags> static Vc_INTRINSIC VectorType load(const T *mem, Flags)
    {
        return LoadHelper2<V, T>::template load<Flags>(mem);
    }
};/*}}}*/

template<typename V, typename T = typename V::VectorEntryType> struct LoadHelper2
{
    typedef typename V::VectorType VectorType;

    template<typename Flags> static Vc_INTRINSIC VectorType genericLoad(const T *mem, typename Flags::EnableIfAligned = nullptr) {
        return LoadHelper<V>::_load(mem, MIC::UpDownConversion<typename V::VectorEntryType, T>(), _MM_HINT_NONE);
    }
    template<typename Flags> static Vc_INTRINSIC VectorType genericLoad(const T *mem, typename Flags::EnableIfUnalignedNotStreaming = nullptr) {
        return LoadHelper<V>::_loadu(mem, MIC::UpDownConversion<typename V::VectorEntryType, T>(), _MM_HINT_NONE);
    }
    template<typename Flags> static Vc_INTRINSIC VectorType genericLoad(const T *mem, typename Flags::EnableIfStreaming = nullptr) {
        return LoadHelper<V>::_load(mem, MIC::UpDownConversion<typename V::VectorEntryType, T>(), _MM_HINT_NT);
    }
    template<typename Flags> static Vc_INTRINSIC VectorType genericLoad(const T *mem, typename Flags::EnableIfUnalignedAndStreaming = nullptr) {
        return LoadHelper<V>::_loadu(mem, MIC::UpDownConversion<typename V::VectorEntryType, T>(), _MM_HINT_NT);
    }

    template<typename Flags> static Vc_INTRINSIC VectorType load(const T *mem) {
        return genericLoad<Flags>(mem);
    }
};

template<> template<typename Flags> Vc_INTRINSIC __m512 LoadHelper2<MIC::float_v, double>::load(const double *mem)
{
    return MIC::mic_cast<__m512>(_mm512_mask_permute4f128_epi32(MIC::mic_cast<__m512i>(
                    _mm512_cvt_roundpd_pslo(LoadHelper<MIC::double_v>::load(&mem[0], Flags()), _MM_FROUND_CUR_DIRECTION)),
                0xff00, MIC::mic_cast<__m512i>(
                    _mm512_cvt_roundpd_pslo(LoadHelper<MIC::double_v>::load(&mem[MIC::double_v::Size], Flags()), _MM_FROUND_CUR_DIRECTION)),
                _MM_PERM_BABA));
}
template<> template<typename Flags> Vc_INTRINSIC __m512 LoadHelper2<MIC::float_v, int>::load(const int *mem)
{
    return MIC::convert<int, float>(LoadHelper<MIC::int_v>::load(mem, Flags()));
}
template<> template<typename Flags> Vc_INTRINSIC __m512 LoadHelper2<MIC::float_v, unsigned int>::load(const unsigned int *mem)
{
    return MIC::convert<unsigned int, float>(LoadHelper<MIC::uint_v>::load(mem, Flags()));
}

} // anonymous namespace

// constants {{{1
template <typename T>
Vc_ALWAYS_INLINE Vector<T, VectorAbi::Mic>::Vector(VectorSpecialInitializerZero)
    : d(Detail::zero<VectorType>())
{
}
template <typename T>
Vc_ALWAYS_INLINE Vector<T, VectorAbi::Mic>::Vector(VectorSpecialInitializerOne)
    : d(Detail::one(EntryType()))
{
}
template <typename T>
Vc_ALWAYS_INLINE Vector<T, VectorAbi::Mic>::Vector(
    VectorSpecialInitializerIndexesFromZero)
    : d(LoadHelper<Vector<T, VectorAbi::Mic>>::load(MIC::IndexesFromZeroHelper<T>(),
                                                    Aligned))
{
}

template <>
Vc_ALWAYS_INLINE MIC::float_v::Vector(VectorSpecialInitializerIndexesFromZero)
    : d(_mm512_extload_ps(&MIC::_IndexesFromZero, _MM_UPCONV_PS_SINT8,
                          _MM_BROADCAST32_NONE, _MM_HINT_NONE))
{
}

template <>
Vc_ALWAYS_INLINE MIC::double_v::Vector(VectorSpecialInitializerIndexesFromZero)
    : d(MIC::convert<int, double>(MIC::int_v::IndexesFromZero().data()))
{
}

// loads {{{1
template <typename T>
template <typename U, typename Flags, typename>
Vc_INTRINSIC void Vector<T, VectorAbi::Mic>::load(const U *x, Flags flags)
{
    Common::handleLoadPrefetches(x, flags);
    d.v() = LoadHelper<Vector<T, VectorAbi::Mic>>::load(x, flags);
}

///////////////////////////////////////////////////////////////////////////////////////////
// zeroing {{{1
template<typename T> Vc_INTRINSIC void Vector<T, VectorAbi::Mic>::setZero()
{
    data() = Detail::zero<VectorType>();
}
template<typename T> Vc_INTRINSIC void Vector<T, VectorAbi::Mic>::setZero(MaskArgument k)
{
    data() = MIC::_xor(data(), k.data(), data(), data());
}
template<typename T> Vc_INTRINSIC void Vector<T, VectorAbi::Mic>::setZeroInverted(MaskArgument k)
{
    data() = MIC::_xor(data(), (!k).data(), data(), data());
}

template<typename T> Vc_INTRINSIC void Vector<T, VectorAbi::Mic>::setQnan()
{
    data() = MIC::allone<VectorType>();
}
template<typename T> Vc_INTRINSIC void Vector<T, VectorAbi::Mic>::setQnan(MaskArgument k)
{
    data() = MIC::mask_mov(data(), k.data(), MIC::allone<VectorType>());
}

///////////////////////////////////////////////////////////////////////////////////////////
// assign {{{1
template<> Vc_INTRINSIC void MIC::double_v::assign(MIC::double_v v, MIC::double_m m)
{
    d.v() = _mm512_mask_mov_pd(d.v(), m.data(), v.d.v());
}
template<> Vc_INTRINSIC void MIC::float_v::assign(MIC::float_v v, MIC::float_m m)
{
    d.v() = _mm512_mask_mov_ps(d.v(), m.data(), v.d.v());
}
template<> Vc_INTRINSIC void MIC::int_v::assign(MIC::int_v v, MIC::int_m m)
{
    d.v() = _mm512_mask_mov_epi32(d.v(), m.data(), v.d.v());
}
template<> Vc_INTRINSIC void MIC::uint_v::assign(MIC::uint_v v, MIC::uint_m m)
{
    d.v() = _mm512_mask_mov_epi32(d.v(), m.data(), v.d.v());
}
template<> Vc_INTRINSIC void MIC::short_v::assign(MIC::short_v v, MIC::short_m m)
{
    d.v() = _mm512_mask_mov_epi32(d.v(), m.data(), v.d.v());
}
template<> Vc_INTRINSIC void MIC::ushort_v::assign(MIC::ushort_v v, MIC::ushort_m m)
{
    d.v() = _mm512_mask_mov_epi32(d.v(), m.data(), v.d.v());
}
// stores {{{1
template <typename V> Vc_INTRINSIC V foldAfterOverflow(V vector)
{
    return vector;
}
Vc_INTRINSIC MIC::ushort_v foldAfterOverflow(MIC::ushort_v vector)
{
    using Detail::operator&;
    return vector & MIC::ushort_v(0xffffu);
}

namespace MIC
{
template <typename Parent, typename T>
template <typename U,
          typename Flags,
          typename std::enable_if<std::is_arithmetic<U>::value, int>::type>
Vc_INTRINSIC void StoreMixin<Parent, T>::store(U *mem, Flags flags) const
{
    Common::handleStorePrefetches(mem, flags);
    MicIntrinsics::store<Flags>(mem, foldAfterOverflow(*static_cast<const Parent *>(this)).data(), UpDownC<U>());
}
}  // namespace MIC

constexpr int alignrCount(int rotationStride, int offset)
{
    return 16 - (rotationStride * offset > 16 ? 16 : rotationStride * offset);
}

template<int rotationStride>
inline __m512i rotationHelper(__m512i v, int offset, std::integral_constant<int, rotationStride>)
{
    switch (offset) {
    case  1: return _mm512_alignr_epi32(v, v, alignrCount(rotationStride,  1));
    case  2: return _mm512_alignr_epi32(v, v, alignrCount(rotationStride,  2));
    case  3: return _mm512_alignr_epi32(v, v, alignrCount(rotationStride,  3));
    case  4: return _mm512_alignr_epi32(v, v, alignrCount(rotationStride,  4));
    case  5: return _mm512_alignr_epi32(v, v, alignrCount(rotationStride,  5));
    case  6: return _mm512_alignr_epi32(v, v, alignrCount(rotationStride,  6));
    case  7: return _mm512_alignr_epi32(v, v, alignrCount(rotationStride,  7));
    case  8: return _mm512_alignr_epi32(v, v, alignrCount(rotationStride,  8));
    case  9: return _mm512_alignr_epi32(v, v, alignrCount(rotationStride,  9));
    case 10: return _mm512_alignr_epi32(v, v, alignrCount(rotationStride, 10));
    case 11: return _mm512_alignr_epi32(v, v, alignrCount(rotationStride, 11));
    case 12: return _mm512_alignr_epi32(v, v, alignrCount(rotationStride, 12));
    case 13: return _mm512_alignr_epi32(v, v, alignrCount(rotationStride, 13));
    case 14: return _mm512_alignr_epi32(v, v, alignrCount(rotationStride, 14));
    case 15: return _mm512_alignr_epi32(v, v, alignrCount(rotationStride, 15));
    default: return v;
    }
}

template <typename V,
          typename Mem,
          typename Mask,
          typename Flags,
          typename Flags::EnableIfUnaligned = nullptr>
inline void storeDispatch(V vector, Mem *address, Mask mask, Flags flags)
{
    constexpr std::ptrdiff_t alignment = sizeof(Mem) * V::Size;
    constexpr std::ptrdiff_t alignmentMask = ~(alignment - 1);

    const auto loAddress = reinterpret_cast<Mem *>((reinterpret_cast<char *>(address) - static_cast<const char*>(0)) & alignmentMask);

    const auto offset = address - loAddress;
    if (offset == 0) {
        MicIntrinsics::store<Flags::UnalignedRemoved>(mask.data(), address, vector.data(), V::template UpDownC<Mem>());
        return;
    }

    constexpr int rotationStride = sizeof(typename V::VectorEntryType) / 4; // palignr shifts by 4 Bytes per "count"
    auto v = rotationHelper(MIC::mic_cast<__m512i>(vector.data()), offset, std::integral_constant<int, rotationStride>());
    auto rotated = MIC::mic_cast<typename V::VectorType>(v);

    MicIntrinsics::store<Flags::UnalignedRemoved>(mask.data() << offset, loAddress, rotated, V::template UpDownC<Mem>());
    MicIntrinsics::store<Flags::UnalignedRemoved>(mask.data() >> (V::Size - offset), loAddress + V::Size, rotated, V::template UpDownC<Mem>());
}

template <typename V,
          typename Mem,
          typename Mask,
          typename Flags,
          typename Flags::EnableIfNotUnaligned = nullptr>
Vc_INTRINSIC void storeDispatch(V vector, Mem *address, Mask mask, Flags flags)
{
    MicIntrinsics::store<Flags>(mask.data(), address, vector.data(), V::template UpDownC<Mem>());
}

namespace MIC
{
template <typename Parent, typename T>
template <typename U,
          typename Flags,
          typename std::enable_if<std::is_arithmetic<U>::value, int>::type>
Vc_INTRINSIC void StoreMixin<Parent, T>::store(U *mem, Mask mask, Flags flags) const
{
    Common::handleStorePrefetches(mem, flags);
    // MicIntrinsics::store does not exist for Flags containing Vc::Unaligned
    storeDispatch(foldAfterOverflow(*static_cast<const Parent *>(this)), mem, mask, flags);
}

template<typename Parent, typename T> Vc_INTRINSIC void StoreMixin<Parent, T>::store(VectorEntryType *mem, decltype(Vc::Streaming)) const
{
    // NR = No-Read hint, NGO = Non-Globally Ordered hint
    // It is not clear whether we will get issues with nrngo if users only expected nr
    //_mm512_storenr_ps(mem, MIC::mic_cast<__m512>(data()));
    _mm512_storenrngo_ps(mem, MIC::mic_cast<__m512>(data()));

    // the ICC auto-vectorizer adds clevict after storenrngo, but testing shows this to be slower...
    //_mm_clevict(mem, _MM_HINT_T1);
}
}  // namespace MIC

// negation {{{1
template<typename T> Vc_ALWAYS_INLINE Vc_PURE Vector<T, VectorAbi::Mic> Vector<T, VectorAbi::Mic>::operator-() const
{
    return Zero() - *this;
}
template<> Vc_ALWAYS_INLINE Vc_PURE MIC::double_v MIC::double_v::operator-() const
{
    return MIC::_xor(d.v(), MIC::mic_cast<VectorType>(MIC::_set1(0x8000000000000000ull)));
}
template<> Vc_ALWAYS_INLINE Vc_PURE MIC::float_v MIC::float_v::operator-() const
{
    return MIC::_xor(d.v(), MIC::mic_cast<VectorType>(MIC::_set1(0x80000000u)));
}
// horizontal ops {{{1
template<typename T> Vc_ALWAYS_INLINE Vector<T, VectorAbi::Mic> Vector<T, VectorAbi::Mic>::partialSum() const
{
    //   a    b    c    d    e    f    g    h
    // +      a    b    c    d    e    f    g    -> a ab bc  cd   de    ef     fg      gh
    // +           a    ab   bc   cd   de   ef   -> a ab abc abcd bcde  cdef   defg    efgh
    // +                     a    ab   abc  abcd -> a ab abc abcd abcde abcdef abcdefg abcdefgh
    Vector<T, VectorAbi::Mic> tmp = *this;
    if (Size >  1) tmp += tmp.shifted(-1);
    if (Size >  2) tmp += tmp.shifted(-2);
    if (Size >  4) tmp += tmp.shifted(-4);
    if (Size >  8) tmp += tmp.shifted(-8);
    if (Size > 16) tmp += tmp.shifted(-16);
    return tmp;
}
template<typename T> inline typename Vector<T, VectorAbi::Mic>::EntryType Vector<T, VectorAbi::Mic>::min(MaskArgument m) const
{
    return _mm512_mask_reduce_min_epi32(m.data(), data());
}
template<> inline float Vector<float>::min(MaskArgument m) const
{
    return _mm512_mask_reduce_min_ps(m.data(), data());
}
template<> inline double Vector<double>::min(MaskArgument m) const
{
    return _mm512_mask_reduce_min_pd(m.data(), data());
}
template<typename T> inline typename Vector<T, VectorAbi::Mic>::EntryType Vector<T, VectorAbi::Mic>::max(MaskArgument m) const
{
    return _mm512_mask_reduce_max_epi32(m.data(), data());
}
template<> inline float Vector<float>::max(MaskArgument m) const
{
    return _mm512_mask_reduce_max_ps(m.data(), data());
}
template<> inline double Vector<double>::max(MaskArgument m) const
{
    return _mm512_mask_reduce_max_pd(m.data(), data());
}
template<typename T> inline typename Vector<T, VectorAbi::Mic>::EntryType Vector<T, VectorAbi::Mic>::product(MaskArgument m) const
{
    return _mm512_mask_reduce_mul_epi32(m.data(), data());
}
template<> inline float Vector<float>::product(MaskArgument m) const
{
    return _mm512_mask_reduce_mul_ps(m.data(), data());
}
template<> inline double Vector<double>::product(MaskArgument m) const
{
    return _mm512_mask_reduce_mul_pd(m.data(), data());
}
template<typename T> inline typename Vector<T, VectorAbi::Mic>::EntryType Vector<T, VectorAbi::Mic>::sum(MaskArgument m) const
{
    return _mm512_mask_reduce_add_epi32(m.data(), data());
}
template<> inline float Vector<float>::sum(MaskArgument m) const
{
    return _mm512_mask_reduce_add_ps(m.data(), data());
}
template<> inline double Vector<double>::sum(MaskArgument m) const
{
    return _mm512_mask_reduce_add_pd(m.data(), data());
}

// integer ops {{{1
template<> Vc_ALWAYS_INLINE    MIC::int_v    MIC::int_v::operator<<(   MIC::int_v::AsArg x) const { return _mm512_sllv_epi32(d.v(), x.d.v()); }
template<> Vc_ALWAYS_INLINE   MIC::uint_v   MIC::uint_v::operator<<(  MIC::uint_v::AsArg x) const { return _mm512_sllv_epi32(d.v(), x.d.v()); }
template<> Vc_ALWAYS_INLINE  MIC::short_v  MIC::short_v::operator<<( MIC::short_v::AsArg x) const { return _mm512_sllv_epi32(d.v(), x.d.v()); }
template<> Vc_ALWAYS_INLINE MIC::ushort_v MIC::ushort_v::operator<<(MIC::ushort_v::AsArg x) const { return _mm512_sllv_epi32(d.v(), x.d.v()); }
template<> Vc_ALWAYS_INLINE    MIC::int_v    MIC::int_v::operator>>(   MIC::int_v::AsArg x) const { return _mm512_srav_epi32(d.v(), x.d.v()); }
template<> Vc_ALWAYS_INLINE   MIC::uint_v   MIC::uint_v::operator>>(  MIC::uint_v::AsArg x) const { return _mm512_srlv_epi32(d.v(), x.d.v()); }
template<> Vc_ALWAYS_INLINE  MIC::short_v  MIC::short_v::operator>>( MIC::short_v::AsArg x) const { return _mm512_srav_epi32(d.v(), x.d.v()); }
template<> Vc_ALWAYS_INLINE MIC::ushort_v MIC::ushort_v::operator>>(MIC::ushort_v::AsArg x) const { return _mm512_srlv_epi32(d.v(), x.d.v()); }
template<typename T> Vc_ALWAYS_INLINE Vector<T, VectorAbi::Mic> &Vector<T, VectorAbi::Mic>::operator<<=(AsArg x) { return *this = *this << x; }
template<typename T> Vc_ALWAYS_INLINE Vector<T, VectorAbi::Mic> &Vector<T, VectorAbi::Mic>::operator>>=(AsArg x) { return *this = *this >> x; }

template<> Vc_ALWAYS_INLINE    MIC::int_v    MIC::int_v::operator<<(unsigned int x) const { return _mm512_slli_epi32(d.v(), x); }
template<> Vc_ALWAYS_INLINE   MIC::uint_v   MIC::uint_v::operator<<(unsigned int x) const { return _mm512_slli_epi32(d.v(), x); }
template<> Vc_ALWAYS_INLINE  MIC::short_v  MIC::short_v::operator<<(unsigned int x) const { return _mm512_slli_epi32(d.v(), x); }
template<> Vc_ALWAYS_INLINE MIC::ushort_v MIC::ushort_v::operator<<(unsigned int x) const { return _mm512_slli_epi32(d.v(), x); }
template<> Vc_ALWAYS_INLINE  MIC::schar_v  MIC::schar_v::operator<<(unsigned int x) const { return _mm512_slli_epi32(d.v(), x); }
template<> Vc_ALWAYS_INLINE  MIC::uchar_v  MIC::uchar_v::operator<<(unsigned int x) const { return _mm512_slli_epi32(d.v(), x); }
template<> Vc_ALWAYS_INLINE    MIC::int_v    MIC::int_v::operator>>(unsigned int x) const { return _mm512_srai_epi32(d.v(), x); }
template<> Vc_ALWAYS_INLINE   MIC::uint_v   MIC::uint_v::operator>>(unsigned int x) const { return _mm512_srli_epi32(d.v(), x); }
template<> Vc_ALWAYS_INLINE  MIC::short_v  MIC::short_v::operator>>(unsigned int x) const { return _mm512_srai_epi32(d.v(), x); }
template<> Vc_ALWAYS_INLINE MIC::ushort_v MIC::ushort_v::operator>>(unsigned int x) const { return _mm512_srli_epi32(d.v(), x); }
template<> Vc_ALWAYS_INLINE  MIC::schar_v  MIC::schar_v::operator>>(unsigned int x) const { return _mm512_srai_epi32(d.v(), x); }
template<> Vc_ALWAYS_INLINE  MIC::uchar_v  MIC::uchar_v::operator>>(unsigned int x) const { return _mm512_srli_epi32(d.v(), x); }
template<typename T> Vc_ALWAYS_INLINE Vector<T, VectorAbi::Mic> &Vector<T, VectorAbi::Mic>::operator<<=(unsigned int x) { return *this = *this << x; }
template<typename T> Vc_ALWAYS_INLINE Vector<T, VectorAbi::Mic> &Vector<T, VectorAbi::Mic>::operator>>=(unsigned int x) { return *this = *this >> x; }

// isnegative {{{1
Vc_INTRINSIC Vc_CONST MIC::float_m isnegative(MIC::float_v x)
{
    return _mm512_cmpge_epu32_mask(MIC::mic_cast<__m512i>(x.data()),
                                   MIC::_set1(MIC::c_general::signMaskFloat[1]));
}
Vc_INTRINSIC Vc_CONST MIC::double_m isnegative(MIC::double_v x)
{
    return _mm512_cmpge_epu32_mask(MIC::mic_cast<__m512i>(_mm512_cvtpd_pslo(x.data())),
                                   MIC::_set1(MIC::c_general::signMaskFloat[1]));
}

// ensureVector helper {{{1
namespace
{
template <typename IT>
Vc_ALWAYS_INLINE enable_if<MIC::is_vector<IT>::value, __m512i> ensureVector(IT indexes)
{ return indexes.data(); }

template <typename IT>
Vc_ALWAYS_INLINE enable_if<Traits::isAtomicSimdArray<IT>::value, __m512i> ensureVector(
    IT indexes)
{ return internal_data(indexes).data(); }

template <typename IT>
Vc_ALWAYS_INLINE
    enable_if<(!MIC::is_vector<IT>::value && !Traits::isAtomicSimdArray<IT>::value &&
               Traits::has_subscript_operator<IT>::value &&
               Traits::has_contiguous_storage<IT>::value),
              __m512i>
    ensureVector(IT indexes)
{
    return MIC::int_v(std::addressof(indexes[0]), Vc::Unaligned).data();
}

Vc_ALWAYS_INLINE __m512i ensureVector(const SimdArray<int, 8> &indexes)
{
    return _mm512_mask_loadunpacklo_epi32(_mm512_setzero_epi32(), 0x00ff, &indexes);
}

template <typename IT>
Vc_ALWAYS_INLINE
    enable_if<(!MIC::is_vector<IT>::value && !Traits::isAtomicSimdArray<IT>::value &&
               !(Traits::has_subscript_operator<IT>::value &&
                 Traits::has_contiguous_storage<IT>::value)),
              __m512i> ensureVector(IT) = delete;
} // anonymous namespace

// gathers {{{1
template <typename T>
template <typename MT, typename IT>
Vc_INTRINSIC Vc_PURE void Vector<T, VectorAbi::Mic>::gatherImplementation(const MT *mem, IT &&indexes)
{
    d.v() = MicIntrinsics::gather(ensureVector(std::forward<IT>(indexes)), mem,
                                  UpDownC<MT>());
}

template <typename T>
template <typename MT, typename IT>
Vc_INTRINSIC Vc_PURE void Vector<T, VectorAbi::Mic>::gatherImplementation(const MT *mem, IT &&indexes,
                                                          MaskArgument mask)
{
    d.v() = MicIntrinsics::gather(
        d.v(), mask.data(), ensureVector(std::forward<IT>(indexes)), mem, UpDownC<MT>());
}

// scatters {{{1
template <typename T>
template <typename MT, typename IT>
Vc_INTRINSIC void Vector<T, VectorAbi::Mic>::scatterImplementation(MT *mem, IT &&indexes) const
{
    using namespace Detail;
    const auto v =
        std::is_same<T, ushort>::value ? (*this & Vector(0xffff)).data() : d.v();
    MicIntrinsics::scatter(mem, ensureVector(std::forward<IT>(indexes)), v, UpDownC<MT>(),
                           sizeof(MT));
}
template <typename T>
template <typename MT, typename IT>
Vc_INTRINSIC void Vector<T, VectorAbi::Mic>::scatterImplementation(MT *mem, IT &&indexes,
                                                   MaskArgument mask) const
{
    using namespace Detail;
    const auto v =
        std::is_same<T, ushort>::value ? (*this & Vector(0xffff)).data() : d.v();
    MicIntrinsics::scatter(mask.data(), mem, ensureVector(std::forward<IT>(indexes)),
                           d.v(), UpDownC<MT>(), sizeof(MT));
}

// exponent {{{1
Vc_INTRINSIC Vc_CONST MIC::float_v exponent(MIC::float_v x)
{
    using Detail::operator>=;
    Vc_ASSERT((x >= x.Zero()).isFull());
    return _mm512_getexp_ps(x.data());
}
Vc_INTRINSIC Vc_CONST MIC::double_v exponent(MIC::double_v x)
{
    using Detail::operator>=;
    Vc_ASSERT((x >= x.Zero()).isFull());
    return _mm512_getexp_pd(x.data());
}

// Random {{{1
static Vc_ALWAYS_INLINE void _doRandomStep(Vector<unsigned int> &state0,
        Vector<unsigned int> &state1)
{
    using MIC::uint_v;
    using namespace Detail;
    state0.load(&Common::RandomState[0]);
    state1.load(&Common::RandomState[uint_v::Size]);
    (state1 * uint_v(0xdeece66du) + uint_v(11)).store(&Common::RandomState[uint_v::Size]);
    uint_v(xor_((state0 * uint_v(0xdeece66du) + uint_v(11)).data(),
                _mm512_srli_epi32(state1.data(), 16))).store(&Common::RandomState[0]);
}

template<typename T> Vc_ALWAYS_INLINE Vector<T, VectorAbi::Mic> Vector<T, VectorAbi::Mic>::Random()
{
    Vector<unsigned int> state0, state1;
    _doRandomStep(state0, state1);
    if (std::is_same<T, short>::value) {
        // short and ushort vectors would hold values that are outside of their range
        // for ushort this doesn't matter because overflow behavior is defined in the compare
        // operators
        return state0.reinterpretCast<Vector<T, VectorAbi::Mic>>() >> 16;
    }
    return state0.reinterpretCast<Vector<T, VectorAbi::Mic> >();
}

template<> Vc_ALWAYS_INLINE Vector<float> Vector<float>::Random()
{
    Vector<unsigned int> state0, state1;
    _doRandomStep(state0, state1);
    using namespace Detail;
    return (reinterpret_components_cast<Vector<float>>(state0 >> 2) | One()) - One();
}

// _mm512_srli_epi64 is neither documented nor defined in any header, here's what it does:
//Vc_INTRINSIC __m512i _mm512_srli_epi64(__m512i v, int n) {
//    return _mm512_mask_mov_epi32(
//            _mm512_srli_epi32(v, n),
//            0x5555,
//            _mm512_swizzle_epi32(_mm512_slli_epi32(v, 32 - n), _MM_SWIZ_REG_CDAB)
//            );
//}

template<> Vc_ALWAYS_INLINE Vector<double> Vector<double>::Random()
{
    using MicIntrinsics::swizzle;
    const auto state = LoadHelper<MIC::uint_v>::load(&Common::RandomState[0], Vc::Aligned);
    const auto factor = MIC::_set1(0x5deece66dull);
    _mm512_store_epi32(&Common::RandomState[0],
            _mm512_add_epi64(
                // the following is not _mm512_mullo_epi64, but something close...
                _mm512_add_epi32(_mm512_mullo_epi32(state, factor), swizzle(_mm512_mulhi_epu32(state, factor), _MM_SWIZ_REG_CDAB)),
                MIC::_set1(11ull)));

    using Detail::operator|;
    using Detail::operator-;
    return (Vector<double>(_cast(_mm512_srli_epi64(MIC::mic_cast<__m512i>(state), 12))) | One()) - One();
}
// }}}1
// shifted / rotated {{{1
namespace
{
template<size_t SIMDWidth, size_t Size> struct VectorShift;
template<> struct VectorShift<64, 8>/*{{{*/
{
    typedef __m512i VectorType;
    static Vc_INTRINSIC VectorType shifted(VectorType v, int amount,
                                           VectorType z = _mm512_setzero_epi32())
    {
        switch (amount) {
        case 15: return _mm512_alignr_epi32(_mm512_setzero_epi32(), z, 14);
        case 14: return _mm512_alignr_epi32(_mm512_setzero_epi32(), z, 12);
        case 13: return _mm512_alignr_epi32(_mm512_setzero_epi32(), z, 10);
        case 12: return _mm512_alignr_epi32(_mm512_setzero_epi32(), z,  8);
        case 11: return _mm512_alignr_epi32(_mm512_setzero_epi32(), z,  6);
        case 10: return _mm512_alignr_epi32(_mm512_setzero_epi32(), z,  4);
        case  9: return _mm512_alignr_epi32(_mm512_setzero_epi32(), z,  2);
        case  8: return z;
        case  7: return _mm512_alignr_epi32(z, v, 14);
        case  6: return _mm512_alignr_epi32(z, v, 12);
        case  5: return _mm512_alignr_epi32(z, v, 10);
        case  4: return _mm512_alignr_epi32(z, v,  8);
        case  3: return _mm512_alignr_epi32(z, v,  6);
        case  2: return _mm512_alignr_epi32(z, v,  4);
        case  1: return _mm512_alignr_epi32(z, v,  2);
        case  0: return v;
        case -1: return _mm512_alignr_epi32(v, z, 14);
        case -2: return _mm512_alignr_epi32(v, z, 12);
        case -3: return _mm512_alignr_epi32(v, z, 10);
        case -4: return _mm512_alignr_epi32(v, z,  8);
        case -5: return _mm512_alignr_epi32(v, z,  6);
        case -6: return _mm512_alignr_epi32(v, z,  4);
        case -7: return _mm512_alignr_epi32(v, z,  2);
        case -8: return z;
        case -9: return _mm512_alignr_epi32(z, _mm512_setzero_epi32(), 14);
        case-10: return _mm512_alignr_epi32(z, _mm512_setzero_epi32(), 12);
        case-11: return _mm512_alignr_epi32(z, _mm512_setzero_epi32(), 10);
        case-12: return _mm512_alignr_epi32(z, _mm512_setzero_epi32(),  8);
        case-13: return _mm512_alignr_epi32(z, _mm512_setzero_epi32(),  6);
        case-14: return _mm512_alignr_epi32(z, _mm512_setzero_epi32(),  4);
        case-15: return _mm512_alignr_epi32(z, _mm512_setzero_epi32(),  2);
        }
        return z;
    }
};/*}}}*/
template<> struct VectorShift<64, 16>/*{{{*/
{
    typedef __m512i VectorType;
    static Vc_INTRINSIC VectorType shifted(VectorType v, int amount,
                                           VectorType z = _mm512_setzero_epi32())
    {
        switch (amount) {
        case 31: return _mm512_alignr_epi32(_mm512_setzero_epi32(), z, 15);
        case 30: return _mm512_alignr_epi32(_mm512_setzero_epi32(), z, 14);
        case 29: return _mm512_alignr_epi32(_mm512_setzero_epi32(), z, 13);
        case 28: return _mm512_alignr_epi32(_mm512_setzero_epi32(), z, 12);
        case 27: return _mm512_alignr_epi32(_mm512_setzero_epi32(), z, 11);
        case 26: return _mm512_alignr_epi32(_mm512_setzero_epi32(), z, 10);
        case 25: return _mm512_alignr_epi32(_mm512_setzero_epi32(), z,  9);
        case 24: return _mm512_alignr_epi32(_mm512_setzero_epi32(), z,  8);
        case 23: return _mm512_alignr_epi32(_mm512_setzero_epi32(), z,  7);
        case 22: return _mm512_alignr_epi32(_mm512_setzero_epi32(), z,  6);
        case 21: return _mm512_alignr_epi32(_mm512_setzero_epi32(), z,  5);
        case 20: return _mm512_alignr_epi32(_mm512_setzero_epi32(), z,  4);
        case 19: return _mm512_alignr_epi32(_mm512_setzero_epi32(), z,  3);
        case 18: return _mm512_alignr_epi32(_mm512_setzero_epi32(), z,  2);
        case 17: return _mm512_alignr_epi32(_mm512_setzero_epi32(), z,  1);
        case 16: return z;
        case 15: return _mm512_alignr_epi32(z, v, 15);
        case 14: return _mm512_alignr_epi32(z, v, 14);
        case 13: return _mm512_alignr_epi32(z, v, 13);
        case 12: return _mm512_alignr_epi32(z, v, 12);
        case 11: return _mm512_alignr_epi32(z, v, 11);
        case 10: return _mm512_alignr_epi32(z, v, 10);
        case  9: return _mm512_alignr_epi32(z, v,  9);
        case  8: return _mm512_alignr_epi32(z, v,  8);
        case  7: return _mm512_alignr_epi32(z, v,  7);
        case  6: return _mm512_alignr_epi32(z, v,  6);
        case  5: return _mm512_alignr_epi32(z, v,  5);
        case  4: return _mm512_alignr_epi32(z, v,  4);
        case  3: return _mm512_alignr_epi32(z, v,  3);
        case  2: return _mm512_alignr_epi32(z, v,  2);
        case  1: return _mm512_alignr_epi32(z, v,  1);
        case  0: return v;
        case -1: return _mm512_alignr_epi32(v, z, 15);
        case -2: return _mm512_alignr_epi32(v, z, 14);
        case -3: return _mm512_alignr_epi32(v, z, 13);
        case -4: return _mm512_alignr_epi32(v, z, 12);
        case -5: return _mm512_alignr_epi32(v, z, 11);
        case -6: return _mm512_alignr_epi32(v, z, 10);
        case -7: return _mm512_alignr_epi32(v, z,  9);
        case -8: return _mm512_alignr_epi32(v, z,  8);
        case -9: return _mm512_alignr_epi32(v, z,  7);
        case-10: return _mm512_alignr_epi32(v, z,  6);
        case-11: return _mm512_alignr_epi32(v, z,  5);
        case-12: return _mm512_alignr_epi32(v, z,  4);
        case-13: return _mm512_alignr_epi32(v, z,  3);
        case-14: return _mm512_alignr_epi32(v, z,  2);
        case-15: return _mm512_alignr_epi32(v, z,  1);
        case-16: return z;
        case-17: return _mm512_alignr_epi32(z, _mm512_setzero_epi32(), 15);
        case-18: return _mm512_alignr_epi32(z, _mm512_setzero_epi32(), 14);
        case-19: return _mm512_alignr_epi32(z, _mm512_setzero_epi32(), 13);
        case-20: return _mm512_alignr_epi32(z, _mm512_setzero_epi32(), 12);
        case-21: return _mm512_alignr_epi32(z, _mm512_setzero_epi32(), 11);
        case-22: return _mm512_alignr_epi32(z, _mm512_setzero_epi32(), 10);
        case-23: return _mm512_alignr_epi32(z, _mm512_setzero_epi32(),  9);
        case-24: return _mm512_alignr_epi32(z, _mm512_setzero_epi32(),  8);
        case-25: return _mm512_alignr_epi32(z, _mm512_setzero_epi32(),  7);
        case-26: return _mm512_alignr_epi32(z, _mm512_setzero_epi32(),  6);
        case-27: return _mm512_alignr_epi32(z, _mm512_setzero_epi32(),  5);
        case-28: return _mm512_alignr_epi32(z, _mm512_setzero_epi32(),  4);
        case-29: return _mm512_alignr_epi32(z, _mm512_setzero_epi32(),  3);
        case-30: return _mm512_alignr_epi32(z, _mm512_setzero_epi32(),  2);
        case-31: return _mm512_alignr_epi32(z, _mm512_setzero_epi32(),  1);
        }
        return z;
    }
};/*}}}*/
} // anonymous namespace
template<typename T> Vc_INTRINSIC Vector<T, VectorAbi::Mic> Vector<T, VectorAbi::Mic>::shifted(int amount) const
{
    typedef VectorShift<sizeof(VectorType), Size> VS;
    return _cast(VS::shifted(MIC::mic_cast<typename VS::VectorType>(d.v()), amount));
}
template<typename T> Vc_INTRINSIC Vector<T, VectorAbi::Mic> Vector<T, VectorAbi::Mic>::shifted(int amount, Vector shiftIn) const
{
    typedef VectorShift<sizeof(VectorType), Size> VS;
    return _cast(VS::shifted(MIC::mic_cast<typename VS::VectorType>(d.v()), amount,
                MIC::mic_cast<typename VS::VectorType>(shiftIn.d.v())));
}

namespace
{
template<size_t SIMDWidth, size_t Size> struct VectorRotate;
template<> struct VectorRotate<64, 8>/*{{{*/
{
    typedef __m512i VectorType;
    static Vc_INTRINSIC VectorType rotated(VectorType v, int amount)
    {
        switch (static_cast<unsigned int>(amount) % 8) {
        case  0: return v;
        case  1: return _mm512_alignr_epi32(v, v,  2);
        case  2: return _mm512_alignr_epi32(v, v,  4);
        case  3: return _mm512_alignr_epi32(v, v,  6);
        case  4: return _mm512_alignr_epi32(v, v,  8);
        case  5: return _mm512_alignr_epi32(v, v, 10);
        case  6: return _mm512_alignr_epi32(v, v, 12);
        case  7: return _mm512_alignr_epi32(v, v, 14);
        }
        return _mm512_setzero_epi32();
    }
};/*}}}*/
template<> struct VectorRotate<64, 16>/*{{{*/
{
    typedef __m512i VectorType;
    static Vc_INTRINSIC VectorType rotated(VectorType v, int amount)
    {
        switch (static_cast<unsigned int>(amount) % 16) {
        case 15: return _mm512_alignr_epi32(v, v, 15);
        case 14: return _mm512_alignr_epi32(v, v, 14);
        case 13: return _mm512_alignr_epi32(v, v, 13);
        case 12: return _mm512_alignr_epi32(v, v, 12);
        case 11: return _mm512_alignr_epi32(v, v, 11);
        case 10: return _mm512_alignr_epi32(v, v, 10);
        case  9: return _mm512_alignr_epi32(v, v,  9);
        case  8: return _mm512_alignr_epi32(v, v,  8);
        case  7: return _mm512_alignr_epi32(v, v,  7);
        case  6: return _mm512_alignr_epi32(v, v,  6);
        case  5: return _mm512_alignr_epi32(v, v,  5);
        case  4: return _mm512_alignr_epi32(v, v,  4);
        case  3: return _mm512_alignr_epi32(v, v,  3);
        case  2: return _mm512_alignr_epi32(v, v,  2);
        case  1: return _mm512_alignr_epi32(v, v,  1);
        case  0: return v;
        }
        return _mm512_setzero_epi32();
    }
};/*}}}*/
} // anonymous namespace
template<typename T> Vc_INTRINSIC Vector<T, VectorAbi::Mic> Vector<T, VectorAbi::Mic>::rotated(int amount) const
{
    typedef VectorRotate<sizeof(VectorType), Size> VR;
    return _cast(VR::rotated(MIC::mic_cast<typename VR::VectorType>(d.v()), amount));
}
// interleaveLow/-High {{{1
template <typename T>
Vc_INTRINSIC Vector<T, VectorAbi::Mic> Vector<T, VectorAbi::Mic>::interleaveLow(
    Vector<T, VectorAbi::Mic> x) const
{
    using namespace MIC;
    __m512i lo = mic_cast<__m512i>(d.v());
    __m512i hi = mic_cast<__m512i>(x.d.v());
    lo = _mm512_permute4f128_epi32(lo, _MM_PERM_BBAA);
    lo = _mm512_mask_swizzle_epi32(lo, 0xf0f0, lo, _MM_SWIZ_REG_BADC);
    lo = _mm512_shuffle_epi32(lo, _MM_PERM_BBAA);
    hi = _mm512_permute4f128_epi32(hi, _MM_PERM_BBAA);
    hi = _mm512_mask_swizzle_epi32(hi, 0xf0f0, hi, _MM_SWIZ_REG_BADC);
    return mic_cast<VectorType>(_mm512_mask_shuffle_epi32(lo, 0xaaaa, hi, _MM_PERM_BBAA));
}
template <typename T>
Vc_INTRINSIC Vector<T, VectorAbi::Mic> Vector<T, VectorAbi::Mic>::interleaveHigh(
    Vector<T, VectorAbi::Mic> x) const
{
    using namespace MIC;
    __m512i lo = mic_cast<__m512i>(d.v());
    __m512i hi = mic_cast<__m512i>(x.d.v());
    lo = _mm512_permute4f128_epi32(lo, _MM_PERM_DDCC);
    lo = _mm512_mask_swizzle_epi32(lo, 0xf0f0, lo, _MM_SWIZ_REG_BADC);
    lo = _mm512_shuffle_epi32(lo, _MM_PERM_BBAA);
    hi = _mm512_permute4f128_epi32(hi, _MM_PERM_DDCC);
    hi = _mm512_mask_swizzle_epi32(hi, 0xf0f0, hi, _MM_SWIZ_REG_BADC);
    return mic_cast<VectorType>(_mm512_mask_shuffle_epi32(lo, 0xaaaa, hi, _MM_PERM_BBAA));
}
template <>
Vc_INTRINSIC Vector<double, VectorAbi::Mic> Vector<double, VectorAbi::Mic>::interleaveLow(
    Vector<double, VectorAbi::Mic> x) const
{
    using namespace MIC;
    __m512i lo = mic_cast<__m512i>(d.v());
    __m512i hi = mic_cast<__m512i>(x.d.v());
    lo = _mm512_permute4f128_epi32(lo, _MM_PERM_BBAA);
    lo = _mm512_mask_swizzle_epi32(lo, 0xf0f0, lo, _MM_SWIZ_REG_BADC);
    lo = _mm512_shuffle_epi32(lo, _MM_PERM_BABA);
    hi = _mm512_permute4f128_epi32(hi, _MM_PERM_BBAA);
    hi = _mm512_mask_swizzle_epi32(hi, 0xf0f0, hi, _MM_SWIZ_REG_BADC);
    return mic_cast<VectorType>(_mm512_mask_shuffle_epi32(lo, 0xcccc, hi, _MM_PERM_BABA));
}
template <>
Vc_INTRINSIC Vector<double, VectorAbi::Mic>
Vector<double, VectorAbi::Mic>::interleaveHigh(Vector<double, VectorAbi::Mic> x) const
{
    using namespace MIC;
    __m512i lo = mic_cast<__m512i>(d.v());
    __m512i hi = mic_cast<__m512i>(x.d.v());
    lo = _mm512_permute4f128_epi32(lo, _MM_PERM_DDCC);
    lo = _mm512_mask_swizzle_epi32(lo, 0xf0f0, lo, _MM_SWIZ_REG_BADC);
    lo = _mm512_shuffle_epi32(lo, _MM_PERM_BABA);
    hi = _mm512_permute4f128_epi32(hi, _MM_PERM_DDCC);
    hi = _mm512_mask_swizzle_epi32(hi, 0xf0f0, hi, _MM_SWIZ_REG_BADC);
    return mic_cast<VectorType>(_mm512_mask_shuffle_epi32(lo, 0xcccc, hi, _MM_PERM_BABA));
}

// reversed {{{1
template <typename T>
Vc_INTRINSIC Vc_PURE Vector<T, VectorAbi::Mic> Vector<T, VectorAbi::Mic>::reversed() const
{
    return MIC::mic_cast<VectorType>(MIC::permute128(
        _mm512_shuffle_epi32(MIC::mic_cast<__m512i>(data()), _MM_PERM_ABCD),
        _MM_PERM_ABCD));
}
template <> Vc_INTRINSIC Vc_PURE MIC::double_v MIC::double_v::reversed() const
{
    return _mm512_castps_pd(MIC::permute128(
        _mm512_swizzle_ps(_mm512_castpd_ps(d.v()), _MM_SWIZ_REG_BADC), _MM_PERM_ABCD));
}

// }}}1

}  // namespace Vc

// vim: foldmethod=marker
