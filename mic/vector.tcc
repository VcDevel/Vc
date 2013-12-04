/*  This file is part of the Vc library.

    Copyright (C) 2010-2013 Matthias Kretz <kretz@kde.org>

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

#include <type_traits>
#include "../common/x86_prefetches.h"
#include "helperimpl.h"
#include "debug.h"
#include "macros.h"

Vc_NAMESPACE_BEGIN(Vc_IMPL_NAMESPACE)

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
        return _mm512_loadu_ps(m, upconv, memHint);
    }
    static Vc_INTRINSIC VectorType _loadu(const void *m, _MM_UPCONV_PD_ENUM upconv, int memHint) {
        return _mm512_loadu_pd(m, upconv, memHint);
    }
    static Vc_INTRINSIC VectorType _loadu(const void *m, _MM_UPCONV_EPI32_ENUM upconv, int memHint) {
        return _mm512_loadu_epi32(m, upconv, memHint);
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
        return LoadHelper<V>::_load(mem, UpDownConversion<typename V::VectorEntryType, T>(), _MM_HINT_NONE);
    }
    template<typename Flags> static Vc_INTRINSIC VectorType genericLoad(const T *mem, typename Flags::EnableIfUnalignedNotStreaming = nullptr) {
        return LoadHelper<V>::_loadu(mem, UpDownConversion<typename V::VectorEntryType, T>(), _MM_HINT_NONE);
    }
    template<typename Flags> static Vc_INTRINSIC VectorType genericLoad(const T *mem, typename Flags::EnableIfStreaming = nullptr) {
        return LoadHelper<V>::_load(mem, UpDownConversion<typename V::VectorEntryType, T>(), _MM_HINT_NT);
    }
    template<typename Flags> static Vc_INTRINSIC VectorType genericLoad(const T *mem, typename Flags::EnableIfUnalignedAndStreaming = nullptr) {
        return LoadHelper<V>::_loadu(mem, UpDownConversion<typename V::VectorEntryType, T>(), _MM_HINT_NT);
    }

    template<typename Flags> static Vc_INTRINSIC VectorType load(const T *mem) {
        return genericLoad<Flags>(mem);
    }
};

template<> template<typename Flags> Vc_INTRINSIC __m512 LoadHelper2<float_v, double>::load(const double *mem)
{
    return mic_cast<__m512>(_mm512_mask_permute4f128_epi32(mic_cast<__m512i>(
                    _mm512_cvt_roundpd_pslo(LoadHelper<double_v>::load(&mem[0], Flags()), _MM_FROUND_CUR_DIRECTION)),
                0xff00, mic_cast<__m512i>(
                    _mm512_cvt_roundpd_pslo(LoadHelper<double_v>::load(&mem[double_v::Size], Flags()), _MM_FROUND_CUR_DIRECTION)),
                _MM_PERM_BABA));
}
template<> template<typename Flags> Vc_INTRINSIC __m512 LoadHelper2<float_v, int>::load(const int *mem)
{
    return StaticCastHelper<int, float>::cast(LoadHelper<int_v>::load(mem, Flags()));
}
template<> template<typename Flags> Vc_INTRINSIC __m512 LoadHelper2<float_v, unsigned int>::load(const unsigned int *mem)
{
    return StaticCastHelper<unsigned int, float>::cast(LoadHelper<uint_v>::load(mem, Flags()));
}

} // anonymous namespace

// constants {{{1
template<typename T> Vc_ALWAYS_INLINE Vector<T>::Vector(VectorSpecialInitializerZero::ZEnum) : d(HV::zero()) {}
template<typename T> Vc_ALWAYS_INLINE Vector<T>::Vector(VectorSpecialInitializerOne::OEnum) : d(HV::one()) {}
template<typename T> Vc_ALWAYS_INLINE Vector<T>::Vector(VectorSpecialInitializerIndexesFromZero::IEnum)
    : d(LoadHelper<Vector<T>>::load(IndexesFromZeroHelper<T>(), Aligned)) {}

template<> Vc_ALWAYS_INLINE float_v::Vector(VectorSpecialInitializerIndexesFromZero::IEnum)
    : d(_mm512_extload_ps(&_IndexesFromZero, _MM_UPCONV_PS_SINT8, _MM_BROADCAST32_NONE, _MM_HINT_NONE)) {}

template<> Vc_ALWAYS_INLINE double_v::Vector(VectorSpecialInitializerIndexesFromZero::IEnum)
    : d(StaticCastHelper<int, double>::cast(int_v::IndexesFromZero().data())) {}

template<typename T> Vc_INTRINSIC Vc_CONST Vector<T> Vector<T>::Zero() { return HV::zero(); }
template<typename T> Vc_INTRINSIC Vc_CONST Vector<T> Vector<T>::One() { return HV::one(); }
template<typename T> Vc_INTRINSIC Vc_CONST Vector<T> Vector<T>::IndexesFromZero() {
    return Vector<T>(VectorSpecialInitializerIndexesFromZero::IndexesFromZero);
}

// loads {{{1
template<typename T> template<typename Flags> Vc_INTRINSIC void Vector<T>::load(const EntryType *x, Flags flags) {
    handleLoadPrefetches(x, flags);
    d.v() = LoadHelper<Vector<T>>::load(x, flags);
}
template<typename T> template<typename OtherT, typename Flags> Vc_INTRINSIC void Vector<T>::load(const OtherT *x, Flags flags) {
    handleLoadPrefetches(x, flags);
    d.v() = LoadHelper<Vector<T>>::load(x, flags);
}

///////////////////////////////////////////////////////////////////////////////////////////
// zeroing {{{1
template<typename T> Vc_INTRINSIC void Vector<T>::setZero()
{
    data() = HV::zero();
}
template<typename T> Vc_INTRINSIC void Vector<T>::setZero(MaskArg k)
{
    data() = _xor(data(), k.data(), data(), data());
}

template<typename T> Vc_INTRINSIC void Vector<T>::setQnan()
{
    data() = _setallone<VectorType>();
}
template<typename T> Vc_INTRINSIC void Vector<T>::setQnan(MaskArg k)
{
    data() = _mask_mov(data(), k.data(), _setallone<VectorType>());
}

///////////////////////////////////////////////////////////////////////////////////////////
// assign {{{1
template<> Vc_INTRINSIC void double_v::assign(double_v v, double_m m)
{
    d.v() = _mm512_mask_mov_pd(d.v(), m.data(), v.d.v());
}
template<> Vc_INTRINSIC void float_v::assign(float_v v, float_m m)
{
    d.v() = _mm512_mask_mov_ps(d.v(), m.data(), v.d.v());
}
template<> Vc_INTRINSIC void int_v::assign(int_v v, int_m m)
{
    d.v() = _mm512_mask_mov_epi32(d.v(), m.data(), v.d.v());
}
template<> Vc_INTRINSIC void uint_v::assign(uint_v v, uint_m m)
{
    d.v() = _mm512_mask_mov_epi32(d.v(), m.data(), v.d.v());
}
template<> Vc_INTRINSIC void short_v::assign(short_v v, short_m m)
{
    d.v() = _mm512_mask_mov_epi32(d.v(), m.data(), v.d.v());
}
template<> Vc_INTRINSIC void ushort_v::assign(ushort_v v, ushort_m m)
{
    d.v() = _mm512_mask_mov_epi32(d.v(), m.data(), v.d.v());
}
// stores {{{1
template<typename Parent, typename T> template<typename T2, typename Flags>
Vc_INTRINSIC void StoreMixin<Parent, T>::store(T2 *mem, Flags flags) const
{
    handleStorePrefetches(mem, flags);
    MicIntrinsics::store<Flags>(mem, data(), UpDownC<T2>());
}

template<typename Parent, typename T> template<typename T2, typename Flags>
Vc_INTRINSIC void StoreMixin<Parent, T>::store(T2 *mem, Mask mask, Flags flags) const
{
    handleStorePrefetches(mem, flags);
    MicIntrinsics::store<Flags>(mask.data(), mem, data(), UpDownC<T2>());
}

template<typename Parent, typename T> Vc_INTRINSIC void StoreMixin<Parent, T>::store(VectorEntryType *mem, decltype(Vc::Streaming)) const
{
    // NR = No-Read hint, NGO = Non-Globally Ordered hint
    // It is not clear whether we will get issues with nrngo if users only expected nr
    //_mm512_storenr_ps(mem, mic_cast<__m512>(data()));
    _mm512_storenrngo_ps(mem, mic_cast<__m512>(data()));

    // the ICC auto-vectorizer adds clevict after storenrngo, but testing shows this to be slower...
    //_mm_clevict(mem, _MM_HINT_T1);
}

// swizzles {{{1
template<typename T> Vc_INTRINSIC Vc_CONST const Vector<T> &Vector<T>::abcd() const { return *this; }
template<typename T> Vc_INTRINSIC Vc_CONST Vector<T> Vector<T>::cdab() const { return MicIntrinsics::swizzle(d.v(), _MM_SWIZ_REG_BADC); }
template<typename T> Vc_INTRINSIC Vc_CONST Vector<T> Vector<T>::badc() const { return MicIntrinsics::swizzle(d.v(), _MM_SWIZ_REG_CDAB); }
template<typename T> Vc_INTRINSIC Vc_CONST Vector<T> Vector<T>::aaaa() const { return MicIntrinsics::swizzle(d.v(), _MM_SWIZ_REG_AAAA); }
template<typename T> Vc_INTRINSIC Vc_CONST Vector<T> Vector<T>::bbbb() const { return MicIntrinsics::swizzle(d.v(), _MM_SWIZ_REG_BBBB); }
template<typename T> Vc_INTRINSIC Vc_CONST Vector<T> Vector<T>::cccc() const { return MicIntrinsics::swizzle(d.v(), _MM_SWIZ_REG_CCCC); }
template<typename T> Vc_INTRINSIC Vc_CONST Vector<T> Vector<T>::dddd() const { return MicIntrinsics::swizzle(d.v(), _MM_SWIZ_REG_DDDD); }
template<typename T> Vc_INTRINSIC Vc_CONST Vector<T> Vector<T>::bcad() const { return MicIntrinsics::swizzle(d.v(), _MM_SWIZ_REG_DACB); }
template<typename T> Vc_INTRINSIC Vc_CONST Vector<T> Vector<T>::bcda() const { return MicIntrinsics::shuffle(d.v(), _MM_PERM_ADCB); }
template<typename T> Vc_INTRINSIC Vc_CONST Vector<T> Vector<T>::dabc() const { return MicIntrinsics::shuffle(d.v(), _MM_PERM_CBAD); }
template<typename T> Vc_INTRINSIC Vc_CONST Vector<T> Vector<T>::acbd() const { return MicIntrinsics::shuffle(d.v(), _MM_PERM_DBCA); }
template<typename T> Vc_INTRINSIC Vc_CONST Vector<T> Vector<T>::dbca() const { return MicIntrinsics::shuffle(d.v(), _MM_PERM_ACBD); }
template<typename T> Vc_INTRINSIC Vc_CONST Vector<T> Vector<T>::dcba() const { return MicIntrinsics::shuffle(d.v(), _MM_PERM_ABCD); }

template<> Vc_INTRINSIC Vc_CONST double_v double_v::bcda() const {
    //ADCB
    auto &&tmp = _mm512_swizzle_pd(d.v(), _MM_SWIZ_REG_DACB);
    return _mm512_mask_swizzle_pd(tmp, 0xcc, tmp, _MM_SWIZ_REG_CDAB);
}
template<> Vc_INTRINSIC Vc_CONST double_v double_v::dabc() const {
    //CBAD
    auto &&tmp = _mm512_mask_swizzle_pd(d.v(), 0xaa, d.v(), _MM_SWIZ_REG_BADC); // BCDA
    return _mm512_swizzle_pd(tmp, _MM_SWIZ_REG_CDAB);
}
template<> Vc_INTRINSIC Vc_CONST double_v double_v::acbd() const {
    //DBCA
    auto &&tmp = _mm512_swizzle_pd(d.v(), _MM_SWIZ_REG_BADC); // BXXC
    return _mm512_mask_swizzle_pd(d.v(), 0x66, tmp, _MM_SWIZ_REG_CDAB); // XBCX
}
template<> Vc_INTRINSIC Vc_CONST double_v double_v::dbca() const {
    //ACBD
    auto &&tmp = _mm512_swizzle_pd(d.v(), _MM_SWIZ_REG_BADC); // XADX
    return _mm512_mask_swizzle_pd(d.v(), 0x99, tmp, _MM_SWIZ_REG_CDAB); // AXXD
}
template<> Vc_INTRINSIC Vc_CONST double_v double_v::dcba() const {
    //ABCD
    return _mm512_swizzle_pd(_mm512_swizzle_pd(d.v(), _MM_SWIZ_REG_CDAB), _MM_SWIZ_REG_BADC);
}
// expand/merge 1 float_v <=> 2 double_v          XXX rationale? remove it for release? XXX {{{1
template<typename T> Vc_ALWAYS_INLINE Vc_FLATTEN Vector<T>::Vector(const Vector<typename ConcatTypeHelper<T>::Type> *a)
    : d(a[0].data())
{
}
template<> Vc_ALWAYS_INLINE Vc_FLATTEN float_v::Vector(const double_v *a)
    : d(mic_cast<__m512>(_mm512_mask_permute4f128_epi32(
                    mic_cast<__m512i>(_mm512_cvtpd_pslo(a[0].data())), 0xff00,
                    mic_cast<__m512i>(_mm512_cvtpd_pslo(a[1].data())), _MM_PERM_BABA)))
{
}

template<typename T> Vc_ALWAYS_INLINE void Vc_FLATTEN Vector<T>::expand(Vector<typename ConcatTypeHelper<T>::Type> *x) const
{
    x[0].data() = data();
}
template<> Vc_ALWAYS_INLINE void Vc_FLATTEN float_v::expand(double_v *x) const
{
    x[0].data() = _mm512_cvtpslo_pd(d.v());
    x[1].data() = _mm512_cvtpslo_pd(_mm512_permute4f128_ps(d.v(), _MM_PERM_DCDC));
}

///////////////////////////////////////////////////////////////////////////////////////////
// negation {{{1
template<> Vc_PURE Vc_ALWAYS_INLINE Vc_FLATTEN Vector<double> Vector<double>::operator-() const
{
    return _xor(d.v(), mic_cast<VectorType>(_set1(0x8000000000000000ull)));
}
template<> Vc_PURE Vc_ALWAYS_INLINE Vc_FLATTEN Vector<float> Vector<float>::operator-() const
{
    return _xor(d.v(), mic_cast<VectorType>(_set1(0x80000000u)));
}
template<> Vc_PURE Vc_ALWAYS_INLINE Vc_FLATTEN Vector<int> Vector<int>::operator-() const
{
    return (~(*this)) + 1;
}
template<> Vc_PURE Vc_ALWAYS_INLINE Vc_FLATTEN Vector<int> Vector<unsigned int>::operator-() const
{
    return Vector<int>(~(*this)) + 1;
}
template<> Vc_PURE Vc_ALWAYS_INLINE Vc_FLATTEN Vector<short> Vector<short>::operator-() const
{
    return (~(*this)) + EntryType(1);
}
template<> Vc_PURE Vc_ALWAYS_INLINE Vc_FLATTEN Vector<short> Vector<unsigned short>::operator-() const
{
    return Vector<short>(~(*this)) + short(1);
}
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
template<typename T> inline typename Vector<T>::EntryType Vector<T>::min(MaskArg m) const
{
    return _mm512_mask_reduce_min_epi32(m.data(), data());
}
template<> inline float Vector<float>::min(MaskArg m) const
{
    return _mm512_mask_reduce_min_ps(m.data(), data());
}
template<> inline double Vector<double>::min(MaskArg m) const
{
    return _mm512_mask_reduce_min_pd(m.data(), data());
}
template<typename T> inline typename Vector<T>::EntryType Vector<T>::max(MaskArg m) const
{
    return _mm512_mask_reduce_max_epi32(m.data(), data());
}
template<> inline float Vector<float>::max(MaskArg m) const
{
    return _mm512_mask_reduce_max_ps(m.data(), data());
}
template<> inline double Vector<double>::max(MaskArg m) const
{
    return _mm512_mask_reduce_max_pd(m.data(), data());
}
template<typename T> inline typename Vector<T>::EntryType Vector<T>::product(MaskArg m) const
{
    return _mm512_mask_reduce_mul_epi32(m.data(), data());
}
template<> inline float Vector<float>::product(MaskArg m) const
{
    return _mm512_mask_reduce_mul_ps(m.data(), data());
}
template<> inline double Vector<double>::product(MaskArg m) const
{
    return _mm512_mask_reduce_mul_pd(m.data(), data());
}
template<typename T> inline typename Vector<T>::EntryType Vector<T>::sum(MaskArg m) const
{
    return _mm512_mask_reduce_add_epi32(m.data(), data());
}
template<> inline float Vector<float>::sum(MaskArg m) const
{
    return _mm512_mask_reduce_add_ps(m.data(), data());
}
template<> inline double Vector<double>::sum(MaskArg m) const
{
    return _mm512_mask_reduce_add_pd(m.data(), data());
}

// copySign {{{1
template<> Vc_INTRINSIC float_v float_v::copySign(float_v::AsArg reference) const
{
    return _or(
            _and(reference.d.v(), _mm512_setsignmask_ps()),
            _and(d.v(), _mm512_setabsmask_ps())
            );
}
template<> Vc_INTRINSIC double_v double_v::copySign(double_v::AsArg reference) const
{
    return _or(
            _and(reference.d.v(), _mm512_setsignmask_pd()),
            _and(d.v(), _mm512_setabsmask_pd())
            );
}//}}}1
// integer ops {{{1
// only unsigned integers have well-defined behavior on over-/underflow
template<> Vc_ALWAYS_INLINE ushort_m ushort_v::operator==(ushort_v::AsArg x) const {
    return _mm512_cmpeq_epu32_mask(_and(d.v(), _set1(0xffff)), _and(x.d.v(), _set1(0xffff)));
}
template<> Vc_ALWAYS_INLINE ushort_m ushort_v::operator!=(ushort_v::AsArg x) const {
    return _mm512_cmpneq_epu32_mask(_and(d.v(), _set1(0xffff)), _and(x.d.v(), _set1(0xffff)));
}
template<> Vc_ALWAYS_INLINE ushort_m ushort_v::operator>=(ushort_v::AsArg x) const {
    return _mm512_cmpge_epu32_mask(_and(d.v(), _set1(0xffff)), _and(x.d.v(), _set1(0xffff)));
}
template<> Vc_ALWAYS_INLINE ushort_m ushort_v::operator> (ushort_v::AsArg x) const {
    return _mm512_cmpgt_epu32_mask(_and(d.v(), _set1(0xffff)), _and(x.d.v(), _set1(0xffff)));
}
template<> Vc_ALWAYS_INLINE ushort_m ushort_v::operator<=(ushort_v::AsArg x) const {
    return _mm512_cmple_epu32_mask(_and(d.v(), _set1(0xffff)), _and(x.d.v(), _set1(0xffff)));
}
template<> Vc_ALWAYS_INLINE ushort_m ushort_v::operator< (ushort_v::AsArg x) const {
    return _mm512_cmplt_epu32_mask(_and(d.v(), _set1(0xffff)), _and(x.d.v(), _set1(0xffff)));
}

template<> Vc_ALWAYS_INLINE    int_v    int_v::operator<<(   int_v::AsArg x) const { return _mm512_sllv_epi32(d.v(), x.d.v()); }
template<> Vc_ALWAYS_INLINE   uint_v   uint_v::operator<<(  uint_v::AsArg x) const { return _mm512_sllv_epi32(d.v(), x.d.v()); }
template<> Vc_ALWAYS_INLINE  short_v  short_v::operator<<( short_v::AsArg x) const { return _mm512_sllv_epi32(d.v(), x.d.v()); }
template<> Vc_ALWAYS_INLINE ushort_v ushort_v::operator<<(ushort_v::AsArg x) const { return _mm512_sllv_epi32(d.v(), x.d.v()); }
template<> Vc_ALWAYS_INLINE    int_v    int_v::operator>>(   int_v::AsArg x) const { return _mm512_srav_epi32(d.v(), x.d.v()); }
template<> Vc_ALWAYS_INLINE   uint_v   uint_v::operator>>(  uint_v::AsArg x) const { return _mm512_srlv_epi32(d.v(), x.d.v()); }
template<> Vc_ALWAYS_INLINE  short_v  short_v::operator>>( short_v::AsArg x) const { return _mm512_srav_epi32(d.v(), x.d.v()); }
template<> Vc_ALWAYS_INLINE ushort_v ushort_v::operator>>(ushort_v::AsArg x) const { return _mm512_srlv_epi32(d.v(), x.d.v()); }
template<typename T> Vc_ALWAYS_INLINE Vector<T> &Vector<T>::operator<<=(AsArg x) { return *this = *this << x; }
template<typename T> Vc_ALWAYS_INLINE Vector<T> &Vector<T>::operator>>=(AsArg x) { return *this = *this >> x; }

template<> Vc_ALWAYS_INLINE    int_v    int_v::operator<<(unsigned int x) const { return _mm512_slli_epi32(d.v(), x); }
template<> Vc_ALWAYS_INLINE   uint_v   uint_v::operator<<(unsigned int x) const { return _mm512_slli_epi32(d.v(), x); }
template<> Vc_ALWAYS_INLINE  short_v  short_v::operator<<(unsigned int x) const { return _mm512_slli_epi32(d.v(), x); }
template<> Vc_ALWAYS_INLINE ushort_v ushort_v::operator<<(unsigned int x) const { return _mm512_slli_epi32(d.v(), x); }
template<> Vc_ALWAYS_INLINE    int_v    int_v::operator>>(unsigned int x) const { return _mm512_srai_epi32(d.v(), x); }
template<> Vc_ALWAYS_INLINE   uint_v   uint_v::operator>>(unsigned int x) const { return _mm512_srli_epi32(d.v(), x); }
template<> Vc_ALWAYS_INLINE  short_v  short_v::operator>>(unsigned int x) const { return _mm512_srai_epi32(d.v(), x); }
template<> Vc_ALWAYS_INLINE ushort_v ushort_v::operator>>(unsigned int x) const { return _mm512_srli_epi32(d.v(), x); }
template<typename T> Vc_ALWAYS_INLINE Vector<T> &Vector<T>::operator<<=(unsigned int x) { return *this = *this << x; }
template<typename T> Vc_ALWAYS_INLINE Vector<T> &Vector<T>::operator>>=(unsigned int x) { return *this = *this >> x; }

// operators {{{1
#include "../common/operators.h"

// this specialization is required because overflow is well defined (mod 2^16) for unsigned short,
// but a / b is not independent of the high bits (in contrast to mul, add, and sub)
template<> ushort_v &ushort_v::operator/=(ushort_v::AsArg x)
{
    d.v() = _div<VectorEntryType>(_and(d.v(), _set1(0xffff)), _and(x.d.v(), _set1(0xffff)));
    return *this;
}
template<> ushort_v ushort_v::operator/(ushort_v::AsArg x) const
{
    return ushort_v(_div<VectorEntryType>(_and(d.v(), _set1(0xffff)), _and(x.d.v(), _set1(0xffff))));
}
// isNegative {{{1
template<> Vc_INTRINSIC Vc_PURE float_m float_v::isNegative() const
{
    return _mm512_cmpge_epu32_mask(mic_cast<__m512i>(d.v()), _set1(c_general::signMaskFloat[1]));
}
template<> Vc_INTRINSIC Vc_PURE double_m double_v::isNegative() const
{
    return _mm512_cmpge_epu32_mask(mic_cast<__m512i>(_mm512_cvtpd_pslo(d.v())), _set1(c_general::signMaskFloat[1]));
}
// gathers {{{1
template<typename T> template<typename IndexT> Vc_ALWAYS_INLINE Vector<T>::Vector(const EntryType *mem, const IndexT *indexes)
{
    gather(mem, indexes);
}
template<typename T> template<typename IndexT> Vc_ALWAYS_INLINE Vector<T>::Vector(const EntryType *mem, Vector<IndexT> indexes)
{
    gather(mem, indexes);
}

template<typename T> template<typename IndexT> Vc_ALWAYS_INLINE Vector<T>::Vector(const EntryType *mem, const IndexT *indexes, MaskArg mask)
    : d(HV::zero())
{
    gather(mem, indexes, mask);
}

template<typename T> template<typename IndexT> Vc_ALWAYS_INLINE Vector<T>::Vector(const EntryType *mem, Vector<IndexT> indexes, MaskArg mask)
    : d(HV::zero())
{
    gather(mem, indexes, mask);
}

template<typename T> template<typename S1, typename IT> Vc_ALWAYS_INLINE Vector<T>::Vector(const S1 *array, const EntryType S1::* member1, IT indexes)
{
    gather(array, member1, indexes);
}
template<typename T> template<typename S1, typename IT> Vc_ALWAYS_INLINE Vector<T>::Vector(const S1 *array, const EntryType S1::* member1, IT indexes, MaskArg mask)
    : d(HV::zero())
{
    gather(array, member1, indexes, mask);
}
template<typename T> template<typename S1, typename S2, typename IT> Vc_ALWAYS_INLINE Vector<T>::Vector(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes)
{
    gather(array, member1, member2, indexes);
}
template<typename T> template<typename S1, typename S2, typename IT> Vc_ALWAYS_INLINE Vector<T>::Vector(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes, MaskArg mask)
    : d(HV::zero())
{
    gather(array, member1, member2, indexes, mask);
}
template<typename T> template<typename S1, typename IT1, typename IT2> Vc_ALWAYS_INLINE Vector<T>::Vector(const S1 *array, const EntryType *const S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes)
{
    gather(array, ptrMember1, outerIndexes, innerIndexes);
}
template<typename T> template<typename S1, typename IT1, typename IT2> Vc_ALWAYS_INLINE Vector<T>::Vector(const S1 *array, const EntryType *const S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes, MaskArg mask)
    : d(HV::zero())
{
    gather(array, ptrMember1, outerIndexes, innerIndexes, mask);
}

template<typename T, size_t Size> struct IndexSizeChecker { static void check() {} };
template<typename T, size_t Size> struct IndexSizeChecker<Vector<T>, Size>
{
    static void check() {
        static_assert(Vector<T>::Size >= Size, "IndexVector must have greater or equal number of entries");
    }
};

namespace
{
template<typename IT> struct EnsureVector
{
    static Vc_ALWAYS_INLINE Vc_PURE __m512i convert(IT indexes) { return int_v(indexes, Vc::Unaligned).data(); }
};
template<typename T> struct EnsureVector<Vector<T>>
{
    static Vc_ALWAYS_INLINE Vc_CONST __m512i convert(Vector<T> indexes) { return indexes.data(); }
};
template<typename IT> static Vc_ALWAYS_INLINE __m512i ensureVector(IT indexes) { return EnsureVector<IT>::convert(indexes); }
} // anonymous namespace

template<typename T> template<typename Index> Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<T>::gather(const EntryType *mem, Index indexes)
{
    IndexSizeChecker<Index, Size>::check();
    d.v() = MicIntrinsics::gather(ensureVector(indexes), mem, UpDownC<EntryType>());
}

template<typename T> template<typename Index> Vc_INTRINSIC void Vector<T>::gather(const EntryType *mem, Index indexes, MaskArg mask)
{
    IndexSizeChecker<Index, Size>::check();
    d.v() = MicIntrinsics::gather(d.v(), mask.data(), ensureVector(indexes), mem, UpDownC<EntryType>());
}

template<typename T> template<typename S1, typename IT>
Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<T>::gather(const S1 *array, const EntryType S1::* member1, IT indexes)
{
    IndexSizeChecker<IT, Size>::check();
    const char *start = reinterpret_cast<const char *>(array);
    const char *offset = reinterpret_cast<const char *>(&(array->*(member1)));
    d.v() = MicIntrinsics::gather(((indexes * sizeof(S1)) + (offset - start)).data(), array, UpDownC<EntryType>(), _MM_SCALE_1);
}

template<> template<typename S1, typename IT>
Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<double>::gather(const S1 *array, const EntryType S1::* member1, IT indexes, MaskArg mask)
{
    IndexSizeChecker<IT, Size>::check();
    const char *start = reinterpret_cast<const char *>(array);
    const char *offset = reinterpret_cast<const char *>(&(array->*(member1)));
    d.v() = _mm512_mask_i32logather_pd(d.v(), mask.data(), ((indexes * sizeof(S1)) + (offset - start)).data(), array, _MM_SCALE_1);
}
template<> template<typename S1, typename IT>
Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<float>::gather(const S1 *array, const EntryType S1::* member1, IT indexes, MaskArg mask)
{
    IndexSizeChecker<IT, Size>::check();
    const char *start = reinterpret_cast<const char *>(array);
    const char *offset = reinterpret_cast<const char *>(&(array->*(member1)));
    d.v() = _mm512_mask_i32gather_ps(d.v(), mask.data(), ((indexes * sizeof(S1)) + (offset - start)).data(), array, _MM_SCALE_1);
}
template<> template<typename S1, typename IT>
Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<int>::gather(const S1 *array, const EntryType S1::* member1, IT indexes, MaskArg mask)
{
    IndexSizeChecker<IT, Size>::check();
    const char *start = reinterpret_cast<const char *>(array);
    const char *offset = reinterpret_cast<const char *>(&(array->*(member1)));
    d.v() = _mm512_mask_i32gather_epi32(d.v(), mask.data(), ((indexes * sizeof(S1)) + (offset - start)).data(), array, _MM_SCALE_1);
}
template<> template<typename S1, typename IT>
Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<unsigned int>::gather(const S1 *array, const EntryType S1::* member1, IT indexes, MaskArg mask)
{
    IndexSizeChecker<IT, Size>::check();
    const char *start = reinterpret_cast<const char *>(array);
    const char *offset = reinterpret_cast<const char *>(&(array->*(member1)));
    d.v() = _mm512_mask_i32gather_epi32(d.v(), mask.data(), ((indexes * sizeof(S1)) + (offset - start)).data(), array, _MM_SCALE_1);
}
template<> template<typename S1, typename IT>
Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<short>::gather(const S1 *array, const EntryType S1::* member1, IT indexes, MaskArg mask)
{
    IndexSizeChecker<IT, Size>::check();
    const char *start = reinterpret_cast<const char *>(array);
    const char *offset = reinterpret_cast<const char *>(&(array->*(member1)));
    d.v() = _mm512_mask_i32extgather_epi32(d.v(), mask.data(), ((indexes * sizeof(S1)) + (offset - start)).data(), array, UpDownC<EntryType>(), _MM_SCALE_1, _MM_HINT_NONE);
}
template<> template<typename S1, typename IT>
Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<unsigned short>::gather(const S1 *array, const EntryType S1::* member1, IT indexes, MaskArg mask)
{
    IndexSizeChecker<IT, Size>::check();
    const char *start = reinterpret_cast<const char *>(array);
    const char *offset = reinterpret_cast<const char *>(&(array->*(member1)));
    d.v() = _mm512_mask_i32extgather_epi32(d.v(), mask.data(), ((indexes * sizeof(S1)) + (offset - start)).data(), array, UpDownC<EntryType>(), _MM_SCALE_1, _MM_HINT_NONE);
}

template<> template<typename S1, typename S2, typename IT>
Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<double>::gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes)
{
    IndexSizeChecker<IT, Size>::check();
    const char *start = reinterpret_cast<const char *>(array);
    const char *offset = reinterpret_cast<const char *>(&(array->*(member1).*(member2)));
    d.v() = _mm512_i32extlogather_pd(((indexes * sizeof(S1)) + (offset - start)).data(), array, UpDownC<EntryType>(), _MM_SCALE_1, _MM_HINT_NONE);
}
template<> template<typename S1, typename S2, typename IT>
Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<float>::gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes)
{
    IndexSizeChecker<IT, Size>::check();
    const char *start = reinterpret_cast<const char *>(array);
    const char *offset = reinterpret_cast<const char *>(&(array->*(member1).*(member2)));
    d.v() = _mm512_i32extgather_ps(((indexes * sizeof(S1)) + (offset - start)).data(), array, UpDownC<EntryType>(), _MM_SCALE_1, _MM_HINT_NONE);
}
template<> template<typename S1, typename S2, typename IT>
Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<int>::gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes)
{
    IndexSizeChecker<IT, Size>::check();
    const char *start = reinterpret_cast<const char *>(array);
    const char *offset = reinterpret_cast<const char *>(&(array->*(member1).*(member2)));
    d.v() = _mm512_i32extgather_epi32(((indexes * sizeof(S1)) + (offset - start)).data(), array, UpDownC<EntryType>(), _MM_SCALE_1, _MM_HINT_NONE);
}
template<> template<typename S1, typename S2, typename IT>
Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<unsigned int>::gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes)
{
    IndexSizeChecker<IT, Size>::check();
    const char *start = reinterpret_cast<const char *>(array);
    const char *offset = reinterpret_cast<const char *>(&(array->*(member1).*(member2)));
    d.v() = _mm512_i32extgather_epi32(((indexes * sizeof(S1)) + (offset - start)).data(), array, UpDownC<EntryType>(), _MM_SCALE_1, _MM_HINT_NONE);
}
template<> template<typename S1, typename S2, typename IT>
Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<short>::gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes)
{
    IndexSizeChecker<IT, Size>::check();
    const char *start = reinterpret_cast<const char *>(array);
    const char *offset = reinterpret_cast<const char *>(&(array->*(member1).*(member2)));
    d.v() = _mm512_i32extgather_epi32(((indexes * sizeof(S1)) + (offset - start)).data(), array, UpDownC<EntryType>(), _MM_SCALE_1, _MM_HINT_NONE);
}
template<> template<typename S1, typename S2, typename IT>
Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<unsigned short>::gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes)
{
    IndexSizeChecker<IT, Size>::check();
    const char *start = reinterpret_cast<const char *>(array);
    const char *offset = reinterpret_cast<const char *>(&(array->*(member1).*(member2)));
    d.v() = _mm512_i32extgather_epi32(((indexes * sizeof(S1)) + (offset - start)).data(), array, UpDownC<EntryType>(), _MM_SCALE_1, _MM_HINT_NONE);
}

template<> template<typename S1, typename S2, typename IT>
Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<double>::gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes, MaskArg mask)
{
    IndexSizeChecker<IT, Size>::check();
    const char *start = reinterpret_cast<const char *>(array);
    const char *offset = reinterpret_cast<const char *>(&(array->*(member1).*(member2)));
    d.v() = _mm512_mask_i32loextgather_pd(d.v(), mask.data(), ((indexes * sizeof(S1)) + (offset - start)).data(), array, UpDownC<EntryType>(), _MM_SCALE_1, _MM_HINT_NONE);
}
template<> template<typename S1, typename S2, typename IT>
Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<float>::gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes, MaskArg mask)
{
    IndexSizeChecker<IT, Size>::check();
    const char *start = reinterpret_cast<const char *>(array);
    const char *offset = reinterpret_cast<const char *>(&(array->*(member1).*(member2)));
    d.v() = _mm512_mask_i32extgather_ps(d.v(), mask.data(), ((indexes * sizeof(S1)) + (offset - start)).data(), array, UpDownC<EntryType>(), _MM_SCALE_1, _MM_HINT_NONE);
}
template<> template<typename S1, typename S2, typename IT>
Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<int>::gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes, MaskArg mask)
{
    IndexSizeChecker<IT, Size>::check();
    const char *start = reinterpret_cast<const char *>(array);
    const char *offset = reinterpret_cast<const char *>(&(array->*(member1).*(member2)));
    d.v() = _mm512_mask_i32extgather_epi32(d.v(), mask.data(), ((indexes * sizeof(S1)) + (offset - start)).data(), array, UpDownC<EntryType>(), _MM_SCALE_1, _MM_HINT_NONE);
}
template<> template<typename S1, typename S2, typename IT>
Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<unsigned int>::gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes, MaskArg mask)
{
    IndexSizeChecker<IT, Size>::check();
    const char *start = reinterpret_cast<const char *>(array);
    const char *offset = reinterpret_cast<const char *>(&(array->*(member1).*(member2)));
    d.v() = _mm512_mask_i32extgather_epi32(d.v(), mask.data(), ((indexes * sizeof(S1)) + (offset - start)).data(), array, UpDownC<EntryType>(), _MM_SCALE_1, _MM_HINT_NONE);
}
template<> template<typename S1, typename S2, typename IT>
Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<short>::gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes, MaskArg mask)
{
    IndexSizeChecker<IT, Size>::check();
    const char *start = reinterpret_cast<const char *>(array);
    const char *offset = reinterpret_cast<const char *>(&(array->*(member1).*(member2)));
    d.v() = _mm512_mask_i32extgather_epi32(d.v(), mask.data(), ((indexes * sizeof(S1)) + (offset - start)).data(), array, UpDownC<EntryType>(), _MM_SCALE_1, _MM_HINT_NONE);
}
template<> template<typename S1, typename S2, typename IT>
Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<unsigned short>::gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes, MaskArg mask)
{
    IndexSizeChecker<IT, Size>::check();
    const char *start = reinterpret_cast<const char *>(array);
    const char *offset = reinterpret_cast<const char *>(&(array->*(member1).*(member2)));
    d.v() = _mm512_mask_i32extgather_epi32(d.v(), mask.data(), ((indexes * sizeof(S1)) + (offset - start)).data(), array, UpDownC<EntryType>(), _MM_SCALE_1, _MM_HINT_NONE);
}

template<> template<typename S1, typename IT1, typename IT2>
Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<double>::gather(const S1 *array, const EntryType *const S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes)
{
    IndexSizeChecker<IT1, Size>::check();
    IndexSizeChecker<IT2, Size>::check();
    d.v() = _mm512_setr_pd((array[outerIndexes[0]].*(ptrMember1))[innerIndexes[0]], (array[outerIndexes[1]].*(ptrMember1))[innerIndexes[1]],
            (array[outerIndexes[2]].*(ptrMember1))[innerIndexes[2]], (array[outerIndexes[3]].*(ptrMember1))[innerIndexes[3]],
            (array[outerIndexes[4]].*(ptrMember1))[innerIndexes[4]], (array[outerIndexes[5]].*(ptrMember1))[innerIndexes[5]],
            (array[outerIndexes[6]].*(ptrMember1))[innerIndexes[6]], (array[outerIndexes[7]].*(ptrMember1))[innerIndexes[7]]);
}
template<> template<typename S1, typename IT1, typename IT2>
Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<float>::gather(const S1 *array, const EntryType *const S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes)
{
    IndexSizeChecker<IT1, Size>::check();
    IndexSizeChecker<IT2, Size>::check();
    d.v() = _mm512_setr_ps((array[outerIndexes[0]].*(ptrMember1))[innerIndexes[0]], (array[outerIndexes[1]].*(ptrMember1))[innerIndexes[1]],
            (array[outerIndexes[ 2]].*(ptrMember1))[innerIndexes[ 2]], (array[outerIndexes[ 3]].*(ptrMember1))[innerIndexes[ 3]],
            (array[outerIndexes[ 4]].*(ptrMember1))[innerIndexes[ 4]], (array[outerIndexes[ 5]].*(ptrMember1))[innerIndexes[ 5]],
            (array[outerIndexes[ 6]].*(ptrMember1))[innerIndexes[ 6]], (array[outerIndexes[ 7]].*(ptrMember1))[innerIndexes[ 7]],
            (array[outerIndexes[ 8]].*(ptrMember1))[innerIndexes[ 8]], (array[outerIndexes[ 9]].*(ptrMember1))[innerIndexes[ 9]],
            (array[outerIndexes[10]].*(ptrMember1))[innerIndexes[10]], (array[outerIndexes[11]].*(ptrMember1))[innerIndexes[11]],
            (array[outerIndexes[12]].*(ptrMember1))[innerIndexes[12]], (array[outerIndexes[13]].*(ptrMember1))[innerIndexes[13]],
            (array[outerIndexes[14]].*(ptrMember1))[innerIndexes[14]], (array[outerIndexes[15]].*(ptrMember1))[innerIndexes[15]]);
}
template<typename T> template<typename S1, typename IT1, typename IT2>
Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<T>::gather(const S1 *array, const EntryType *const S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes)
{
    IndexSizeChecker<IT1, Size>::check();
    IndexSizeChecker<IT2, Size>::check();
    d.v() = _mm512_setr_epi32((array[outerIndexes[0]].*(ptrMember1))[innerIndexes[0]], (array[outerIndexes[1]].*(ptrMember1))[innerIndexes[1]],
            (array[outerIndexes[ 2]].*(ptrMember1))[innerIndexes[ 2]], (array[outerIndexes[ 3]].*(ptrMember1))[innerIndexes[ 3]],
            (array[outerIndexes[ 4]].*(ptrMember1))[innerIndexes[ 4]], (array[outerIndexes[ 5]].*(ptrMember1))[innerIndexes[ 5]],
            (array[outerIndexes[ 6]].*(ptrMember1))[innerIndexes[ 6]], (array[outerIndexes[ 7]].*(ptrMember1))[innerIndexes[ 7]],
            (array[outerIndexes[ 8]].*(ptrMember1))[innerIndexes[ 8]], (array[outerIndexes[ 9]].*(ptrMember1))[innerIndexes[ 9]],
            (array[outerIndexes[10]].*(ptrMember1))[innerIndexes[10]], (array[outerIndexes[11]].*(ptrMember1))[innerIndexes[11]],
            (array[outerIndexes[12]].*(ptrMember1))[innerIndexes[12]], (array[outerIndexes[13]].*(ptrMember1))[innerIndexes[13]],
            (array[outerIndexes[14]].*(ptrMember1))[innerIndexes[14]], (array[outerIndexes[15]].*(ptrMember1))[innerIndexes[15]]);
}
template<typename T> template<typename S1, typename IT1, typename IT2>
Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<T>::gather(const S1 *array, const EntryType *const S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes, MaskArg mask)
{
    IndexSizeChecker<IT1, Size>::check();
    IndexSizeChecker<IT2, Size>::check();
    Vc_foreach_bit (size_t i, mask) {
        d.m(i) = (array[outerIndexes[i]].*(ptrMember1))[innerIndexes[i]];
    }
}

///////////////////////////////////////////////////////////////////////////////////////////
// scatters {{{1
template<typename T> template<typename Index> Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<T>::scatter(EntryType *mem, Index indexes) const
{
    MicIntrinsics::scatter(mem, ensureVector(indexes), d.v(), UpDownC<EntryType>(), sizeof(EntryType));
}
template<typename T> template<typename Index> Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<T>::scatter(EntryType *mem, Index indexes, MaskArg mask) const
{
    MicIntrinsics::scatter(mask.data(), mem, ensureVector(indexes), d.v(), UpDownC<EntryType>(), sizeof(EntryType));
}
template<typename T> template<typename S1, typename IT> Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<T>::scatter(S1 *array, EntryType S1::* member1, IT indexes) const
{
    const char *start = reinterpret_cast<const char *>(array);
    const char *offset = reinterpret_cast<const char *>(&(array->*(member1)));
    MicIntrinsics::scatter(mem, ((indexes * sizeof(S1)) + (offset - start)).data(), d.v(), UpDownC<EntryType>(), _MM_SCALE_1);
}
template<typename T> template<typename S1, typename IT> Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<T>::scatter(S1 *array, EntryType S1::* member1, IT indexes, MaskArg mask) const
{
    const char *start = reinterpret_cast<const char *>(array);
    const char *offset = reinterpret_cast<const char *>(&(array->*(member1)));
    MicIntrinsics::scatter(mask.data(), array, ((indexes * sizeof(S1)) + (offset - start)).data(), d.v(), UpDownC<EntryType>(), _MM_SCALE_1);
}
template<typename T> template<typename S1, typename S2, typename IT> Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<T>::scatter(S1 *array, S2 S1::* member1, EntryType S2::* member2, IT indexes) const
{
    const char *start = reinterpret_cast<const char *>(array);
    const char *offset = reinterpret_cast<const char *>(&(array->*(member1).*(member2)));
    MicIntrinsics::scatter(array, ((indexes * sizeof(S1)) + (offset - start)).data(), d.v(), UpDownC<EntryType>(), _MM_SCALE_1);
}
template<typename T> template<typename S1, typename S2, typename IT> Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<T>::scatter(S1 *array, S2 S1::* member1, EntryType S2::* member2, IT indexes, MaskArg mask) const
{
    const char *start = reinterpret_cast<const char *>(array);
    const char *offset = reinterpret_cast<const char *>(&(array->*(member1).*(member2)));
    MicIntrinsics::scatter(mask.data(), array, ((indexes * sizeof(S1)) + (offset - start)).data(), d.v(), UpDownC<EntryType>(), _MM_SCALE_1);
}
template<typename T> template<typename S1, typename IT1, typename IT2> Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<T>::scatter(S1 *array, EntryType *S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes) const
{
    for_all_vector_entries(i,
            (array[innerIndexes[i]].*(ptrMember1))[outerIndexes[i]] = d.m(i);
            );
}
template<typename T> template<typename S1, typename IT1, typename IT2> Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<T>::scatter(S1 *array, EntryType *S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes, MaskArg mask) const
{
    Vc_foreach_bit (size_t i, mask) {
        (array[outerIndexes[i]].*(ptrMember1))[innerIndexes[i]] = d.m(i);
    }
}
//}}}1
// exponent {{{1
template<typename T> Vc_INTRINSIC Vector<T> Vector<T>::exponent() const
{
    VC_ASSERT((*this >= Zero()).isFull());
    return _mm512_getexp_ps(d.v());
}
template<> Vc_INTRINSIC double_v double_v::exponent() const
{
    VC_ASSERT((*this >= Zero()).isFull());
    return _mm512_getexp_pd(d.v());
}
// }}}1
// Random {{{1
static Vc_ALWAYS_INLINE void _doRandomStep(Vector<unsigned int> &state0,
        Vector<unsigned int> &state1)
{
    state0.load(&Common::RandomState[0]);
    state1.load(&Common::RandomState[uint_v::Size]);
    (state1 * 0xdeece66du + 11).store(&Common::RandomState[uint_v::Size]);
    uint_v(_xor((state0 * 0xdeece66du + 11).data(), _mm512_srli_epi32(state1.data(), 16))).store(&Common::RandomState[0]);
}

template<typename T> Vc_ALWAYS_INLINE Vector<T> Vector<T>::Random()
{
    Vector<unsigned int> state0, state1;
    _doRandomStep(state0, state1);
    if (std::is_same<T, short>::value) {
        // short and ushort vectors would hold values that are outside of their range
        // for ushort this doesn't matter because overflow behavior is defined in the compare
        // operators
        return state0.reinterpretCast<Vector<T>>() >> 16;
    }
    return state0.reinterpretCast<Vector<T> >();
}

template<> Vc_ALWAYS_INLINE Vector<float> Vector<float>::Random()
{
    Vector<unsigned int> state0, state1;
    _doRandomStep(state0, state1);
    return HT::sub(_or(_cast(_mm512_srli_epi32(state0.data(), 2)), HV::one()), HV::one());
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
    const auto state = LoadHelper<uint_v>::load(&Common::RandomState[0], Vc::Aligned);
    const auto factor = _set1(0x5deece66dull);
    _mm512_store_epi32(&Common::RandomState[0],
            _mm512_add_epi64(
                // the following is not _mm512_mullo_epi64, but something close...
                _mm512_add_epi32(_mm512_mullo_epi32(state, factor), swizzle(_mm512_mulhi_epu32(state, factor), _MM_SWIZ_REG_CDAB)),
                _set1(11ull)));

    return (Vector<double>(_cast(_mm512_srli_epi64(mic_cast<__m512i>(state), 12))) | One()) - One();
}
// }}}1
// shifted / rotated {{{1
namespace
{
template<size_t SIMDWidth, size_t Size> struct VectorShift;
template<> struct VectorShift<64, 8>/*{{{*/
{
    typedef __m512i VectorType;
    static Vc_INTRINSIC VectorType shifted(VC_ALIGNED_PARAMETER(VectorType) v, int amount,
            VC_ALIGNED_PARAMETER(VectorType) z = _mm512_setzero_epi32())
    {
        switch (amount) {
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
        }
        return z;
    }
};/*}}}*/
template<> struct VectorShift<64, 16>/*{{{*/
{
    typedef __m512i VectorType;
    static Vc_INTRINSIC VectorType shifted(VC_ALIGNED_PARAMETER(VectorType) v, int amount,
            VC_ALIGNED_PARAMETER(VectorType) z = _mm512_setzero_epi32())
    {
        switch (amount) {
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
        }
        return z;
    }
};/*}}}*/
} // anonymous namespace
template<typename T> Vc_INTRINSIC Vector<T> Vector<T>::shifted(int amount) const
{
    typedef VectorShift<sizeof(VectorType), Size> VS;
    return _cast(VS::shifted(mic_cast<typename VS::VectorType>(d.v()), amount));
}
template<typename T> Vc_INTRINSIC Vector<T> Vector<T>::shifted(int amount, Vector shiftIn) const
{
    typedef VectorShift<sizeof(VectorType), Size> VS;
    return _cast(VS::shifted(mic_cast<typename VS::VectorType>(d.v()), amount,
                mic_cast<typename VS::VectorType>(shiftIn.d.v())));
}

namespace
{
template<size_t SIMDWidth, size_t Size> struct VectorRotate;
template<> struct VectorRotate<64, 8>/*{{{*/
{
    typedef __m512i VectorType;
    static Vc_INTRINSIC VectorType rotated(VC_ALIGNED_PARAMETER(VectorType) v, int amount)
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
    static Vc_INTRINSIC VectorType rotated(VC_ALIGNED_PARAMETER(VectorType) v, int amount)
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
template<typename T> Vc_INTRINSIC Vector<T> Vector<T>::rotated(int amount) const
{
    typedef VectorRotate<sizeof(VectorType), Size> VR;
    return _cast(VR::rotated(mic_cast<typename VR::VectorType>(d.v()), amount));
}
// }}}1

Vc_NAMESPACE_END

// vim: foldmethod=marker
