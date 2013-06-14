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

#include <type_traits>
#include "debug.h"

Vc_NAMESPACE_BEGIN(Vc_IMPL_NAMESPACE)

// LoadHelper {{{1
namespace
{
template<typename V, typename T> struct LoadHelper2;

template<typename V> struct LoadHelper
{
    typedef typename V::VectorType VectorType;

    static Vc_ALWAYS_INLINE VectorType _load(const void *m, _MM_UPCONV_PS_ENUM upconv, int memHint) {
        return _mm512_extload_ps(m, upconv, _MM_BROADCAST32_NONE, memHint);
    }
    static Vc_ALWAYS_INLINE VectorType _load(const void *m, _MM_UPCONV_PD_ENUM upconv, int memHint) {
        return _mm512_extload_pd(m, upconv, _MM_BROADCAST64_NONE, memHint);
    }
    static Vc_ALWAYS_INLINE VectorType _load(const void *m, _MM_UPCONV_EPI32_ENUM upconv, int memHint) {
        return _mm512_extload_epi32(m, upconv, _MM_BROADCAST32_NONE, memHint);
    }

    static Vc_ALWAYS_INLINE VectorType _loadu(const void *m, _MM_UPCONV_PS_ENUM upconv, int memHint) {
        return _mm512_loadu_ps(m, upconv, memHint);
    }
    static Vc_ALWAYS_INLINE VectorType _loadu(const void *m, _MM_UPCONV_PD_ENUM upconv, int memHint) {
        return _mm512_loadu_pd(m, upconv, memHint);
    }
    static Vc_ALWAYS_INLINE VectorType _loadu(const void *m, _MM_UPCONV_EPI32_ENUM upconv, int memHint) {
        return _mm512_loadu_epi32(m, upconv, memHint);
    }

    template<typename T, typename Flags> static Vc_ALWAYS_INLINE VectorType load(const T *mem, Flags f)
    {
        return LoadHelper2<V, T>::load(mem, f);
    }
};

template<typename V, typename T = typename V::VectorEntryType> struct LoadHelper2
{
    typedef typename V::VectorType VectorType;
    template<typename Flags> static Vc_ALWAYS_INLINE VectorType load(const T *mem, Flags f)
    {
        if (std::is_same<Flags, Vc::AlignedFlag>::value) {
            return LoadHelper<V>::_load(mem, UpDownConversion<typename V::VectorEntryType, T>(), _MM_HINT_NONE);
        } else if (std::is_same<Flags, Vc::UnalignedFlag>::value) {
            return LoadHelper<V>::_loadu(mem, UpDownConversion<typename V::VectorEntryType, T>(), _MM_HINT_NONE);
        } else if (std::is_same<Flags, Vc::StreamingAndAlignedFlag>::value) {
            return LoadHelper<V>::_load(mem, UpDownConversion<typename V::VectorEntryType, T>(), _MM_HINT_NT);
        } else if (std::is_same<Flags, Vc::StreamingAndUnalignedFlag>::value) {
            return LoadHelper<V>::_loadu(mem, UpDownConversion<typename V::VectorEntryType, T>(), _MM_HINT_NT);
        }
        return VectorType();
    }
};

template<> template<typename A> Vc_ALWAYS_INLINE __m512 LoadHelper2<float_v, double>::load(const double *mem, A align)
{
    return mic_cast<__m512>(_mm512_mask_permute4f128_epi32(mic_cast<__m512i>(
                    _mm512_cvt_roundpd_pslo(LoadHelper<double_v>::load(&mem[0], align), _MM_FROUND_CUR_DIRECTION)),
                0xff00, mic_cast<__m512i>(
                    _mm512_cvt_roundpd_pslo(LoadHelper<double_v>::load(&mem[double_v::Size], align), _MM_FROUND_CUR_DIRECTION)),
                _MM_PERM_BABA));
}
template<> template<typename A> Vc_ALWAYS_INLINE __m512 LoadHelper2<float_v, int>::load(const int *mem, A align)
{
    return StaticCastHelper<int, float>::cast(LoadHelper<int_v>::load(mem, align));
}
template<> template<typename A> Vc_ALWAYS_INLINE __m512 LoadHelper2<float_v, unsigned int>::load(const unsigned int *mem, A align)
{
    return StaticCastHelper<unsigned int, float>::cast(LoadHelper<uint_v>::load(mem, align));
}

template<> template<typename A> Vc_ALWAYS_INLINE __m512 LoadHelper2<sfloat_v, double>::load(const double *mem, A align)
{
    return mic_cast<__m512>(_mm512_mask_permute4f128_epi32(mic_cast<__m512i>(
                    _mm512_cvt_roundpd_pslo(LoadHelper<double_v>::load(&mem[0], align), _MM_FROUND_CUR_DIRECTION)),
                0xff00, mic_cast<__m512i>(
                    _mm512_cvt_roundpd_pslo(LoadHelper<double_v>::load(&mem[double_v::Size], align), _MM_FROUND_CUR_DIRECTION)),
                _MM_PERM_BABA));
}
template<> template<typename A> Vc_ALWAYS_INLINE __m512 LoadHelper2<sfloat_v, int>::load(const int *mem, A align)
{
    return StaticCastHelper<int, float>::cast(LoadHelper<int_v>::load(mem, align));
}
template<> template<typename A> Vc_ALWAYS_INLINE __m512 LoadHelper2<sfloat_v, unsigned int>::load(const unsigned int *mem, A align)
{
    return StaticCastHelper<unsigned int, float>::cast(LoadHelper<uint_v>::load(mem, align));
}
} // anonymous namespace

// constants {{{1
template<typename T> Vc_ALWAYS_INLINE Vector<T>::Vector(VectorSpecialInitializerZero::ZEnum) : d(HV::zero()) {}
template<typename T> Vc_ALWAYS_INLINE Vector<T>::Vector(VectorSpecialInitializerOne::OEnum) : d(HV::one()) {}
template<typename T> Vc_ALWAYS_INLINE Vector<T>::Vector(VectorSpecialInitializerIndexesFromZero::IEnum)
    : d(LoadHelper<Vector<T>>::load(IndexesFromZeroHelper<T>(), Aligned)) {}

template<typename T> Vc_INTRINSIC Vc_CONST Vector<T> Vector<T>::Zero() { return HV::zero(); }
template<typename T> Vc_INTRINSIC Vc_CONST Vector<T> Vector<T>::One() { return HV::one(); }
template<typename T> Vc_INTRINSIC Vc_CONST Vector<T> Vector<T>::IndexesFromZero() {
    return LoadHelper<Vector<T>>::load(IndexesFromZeroHelper<T>(), Aligned);
}

// static cast {{{1
template<typename T> template<typename OtherT> Vc_ALWAYS_INLINE Vector<T>::Vector(Vector<OtherT> x)
    : d(StaticCastHelper<OtherT, T>::cast(x.data())) {}
//template<typename T> template<typename OtherT> Vc_ALWAYS_INLINE Vector<T>::Vector(VectorMultiplication<OtherT> x)
//    : d(StaticCastHelper<OtherT, T>::cast(x.data())) {}

// broadcast {{{1
template<typename T> Vc_ALWAYS_INLINE Vector<T>::Vector(EntryType x) : d(_set1(x)) {}

// loads {{{1
template<typename T> Vc_INTRINSIC Vector<T>::Vector(const EntryType *x) { load(x); }
template<typename T> template<typename A> Vc_INTRINSIC Vector<T>::Vector(const EntryType *x, A a) { load(x, a); }
template<typename T> template<typename OtherT> Vc_INTRINSIC Vector<T>::Vector(const OtherT *x) { load(x); }
template<typename T> template<typename OtherT, typename A> Vc_INTRINSIC Vector<T>::Vector(const OtherT *x, A a) { load(x, a); }

template<typename T> Vc_INTRINSIC void Vector<T>::load(const EntryType *x) { load(x, Aligned); }
template<typename T> template<typename A> Vc_INTRINSIC void Vector<T>::load(const EntryType *x, A align) {
    d.v() = LoadHelper<Vector<T>>::load(x, align);
}
template<typename T> template<typename OtherT> Vc_INTRINSIC void Vector<T>::load(const OtherT *x) {
    d.v() = LoadHelper<Vector<T>>::load(x, Aligned);
}
template<typename T> template<typename OtherT, typename A> Vc_INTRINSIC void Vector<T>::load(const OtherT *x, A align) {
    d.v() = LoadHelper<Vector<T>>::load(x, align);
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
template<> Vc_INTRINSIC void sfloat_v::assign(sfloat_v v, sfloat_m m)
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
template<typename Parent, typename T> template<typename T2> inline void StoreMixin<Parent, T>::store(T2 *mem) const
{
    MicIntrinsics::store(mem, data(), UpDownC<T2>(), Aligned);
}

template<typename Parent, typename T> template<typename T2> inline void StoreMixin<Parent, T>::store(T2 *mem, Mask mask) const
{
    MicIntrinsics::store(mask.data(), mem, data(), UpDownC<T2>(), Aligned);
}

template<typename Parent, typename T> template<typename T2, typename A> inline void StoreMixin<Parent, T>::store(T2 *mem, A align) const
{
    MicIntrinsics::store(mem, data(), UpDownC<T2>(), align);
}

template<typename Parent, typename T> template<typename T2, typename A> inline void StoreMixin<Parent, T>::store(T2 *mem, Mask mask, A align) const
{
    MicIntrinsics::store(mask.data(), mem, data(), UpDownC<T2>(), align);
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
template<> Vc_ALWAYS_INLINE Vc_FLATTEN sfloat_v::Vector(const double_v *a)
    : d(_mm512_mask_permute4f128_ps(_mm512_cvtpd_pslo(a[0].data()), 0xff00,
                _mm512_cvtpd_pslo(a[1].data()), _MM_PERM_BABA))
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
template<> Vc_ALWAYS_INLINE void Vc_FLATTEN sfloat_v::expand(double_v *x) const
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
template<> Vc_PURE Vc_ALWAYS_INLINE Vc_FLATTEN Vector<sfloat> Vector<sfloat>::operator-() const
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
template<> inline float Vector<sfloat>::min(MaskArg m) const
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
template<> inline float Vector<sfloat>::max(MaskArg m) const
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
template<> inline float Vector<sfloat>::product(MaskArg m) const
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
template<> inline float Vector<sfloat>::sum(MaskArg m) const
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
template<> Vc_INTRINSIC sfloat_v sfloat_v::copySign(sfloat_v::AsArg reference) const
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
template<> template<typename TT> Vc_ALWAYS_INLINE VC_EXACT_TYPE(TT, unsigned short, ushort_m) ushort_v::operator==(TT x) const {
    return _mm512_cmpeq_epu32_mask(_and(d.v(), _set1(0xffff)), _set1(x));
}
template<> template<typename TT> Vc_ALWAYS_INLINE VC_EXACT_TYPE(TT, unsigned short, ushort_m) ushort_v::operator!=(TT x) const {
    return _mm512_cmpneq_epu32_mask(_and(d.v(), _set1(0xffff)), _set1(x));
}
template<> template<typename TT> Vc_ALWAYS_INLINE VC_EXACT_TYPE(TT, unsigned short, ushort_m) ushort_v::operator>=(TT x) const {
    return _mm512_cmpge_epu32_mask(_and(d.v(), _set1(0xffff)), _set1(x));
}
template<> template<typename TT> Vc_ALWAYS_INLINE VC_EXACT_TYPE(TT, unsigned short, ushort_m) ushort_v::operator> (TT x) const {
    return _mm512_cmpgt_epu32_mask(_and(d.v(), _set1(0xffff)), _set1(x));
}
template<> template<typename TT> Vc_ALWAYS_INLINE VC_EXACT_TYPE(TT, unsigned short, ushort_m) ushort_v::operator<=(TT x) const {
    return _mm512_cmple_epu32_mask(_and(d.v(), _set1(0xffff)), _set1(x));
}
template<> template<typename TT> Vc_ALWAYS_INLINE VC_EXACT_TYPE(TT, unsigned short, ushort_m) ushort_v::operator< (TT x) const {
    return _mm512_cmplt_epu32_mask(_and(d.v(), _set1(0xffff)), _set1(x));
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
// isNegative {{{1
template<> Vc_INTRINSIC Vc_PURE float_m float_v::isNegative() const
{
    return _mm512_cmpge_epu32_mask(mic_cast<__m512i>(d.v()), _set1(c_general::signMaskFloat[1]));
}
template<> Vc_INTRINSIC Vc_PURE sfloat_m sfloat_v::isNegative() const
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

template<typename T> template<typename Index> Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<T>::gather(const EntryType *mem, Index indexes)
{
    IndexSizeChecker<Index, Size>::check();
    d.v() = MicIntrinsics::gather(indexes.data(), mem, UpDownC<EntryType>());
}

template<typename T> template<typename Index> Vc_INTRINSIC void Vector<T>::gather(const EntryType *mem, Index indexes, MaskArg mask)
{
    IndexSizeChecker<Index, Size>::check();
    d.v() = MicIntrinsics::gather(d.v(), mask.data(), indexes.data(), mem, UpDownC<EntryType>());
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
Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<sfloat>::gather(const S1 *array, const EntryType S1::* member1, IT indexes, MaskArg mask)
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
    d.v() = _mm512_i32extlogather_pd(((indexes * sizeof(S2)) + (offset - start)).data(), array, UpDownC<EntryType>(), _MM_SCALE_1, _MM_HINT_NONE);
}
template<> template<typename S1, typename S2, typename IT>
Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<float>::gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes)
{
    IndexSizeChecker<IT, Size>::check();
    const char *start = reinterpret_cast<const char *>(array);
    const char *offset = reinterpret_cast<const char *>(&(array->*(member1).*(member2)));
    d.v() = _mm512_i32extgather_ps(((indexes * sizeof(S2)) + (offset - start)).data(), array, UpDownC<EntryType>(), _MM_SCALE_1, _MM_HINT_NONE);
}
template<> template<typename S1, typename S2, typename IT>
Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<sfloat>::gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes)
{
    IndexSizeChecker<IT, Size>::check();
    const char *start = reinterpret_cast<const char *>(array);
    const char *offset = reinterpret_cast<const char *>(&(array->*(member1).*(member2)));
    d.v() = _mm512_i32extgather_ps(((indexes * sizeof(S2)) + (offset - start)).data(), array, UpDownC<EntryType>(), _MM_SCALE_1, _MM_HINT_NONE);
}
template<> template<typename S1, typename S2, typename IT>
Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<int>::gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes)
{
    IndexSizeChecker<IT, Size>::check();
    const char *start = reinterpret_cast<const char *>(array);
    const char *offset = reinterpret_cast<const char *>(&(array->*(member1).*(member2)));
    d.v() = _mm512_i32extgather_epi32(((indexes * sizeof(S2)) + (offset - start)).data(), array, UpDownC<EntryType>(), _MM_SCALE_1, _MM_HINT_NONE);
}
template<> template<typename S1, typename S2, typename IT>
Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<unsigned int>::gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes)
{
    IndexSizeChecker<IT, Size>::check();
    const char *start = reinterpret_cast<const char *>(array);
    const char *offset = reinterpret_cast<const char *>(&(array->*(member1).*(member2)));
    d.v() = _mm512_i32extgather_epi32(((indexes * sizeof(S2)) + (offset - start)).data(), array, UpDownC<EntryType>(), _MM_SCALE_1, _MM_HINT_NONE);
}
template<> template<typename S1, typename S2, typename IT>
Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<short>::gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes)
{
    IndexSizeChecker<IT, Size>::check();
    const char *start = reinterpret_cast<const char *>(array);
    const char *offset = reinterpret_cast<const char *>(&(array->*(member1).*(member2)));
    d.v() = _mm512_i32extgather_epi32(((indexes * sizeof(S2)) + (offset - start)).data(), array, UpDownC<EntryType>(), _MM_SCALE_1, _MM_HINT_NONE);
}
template<> template<typename S1, typename S2, typename IT>
Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<unsigned short>::gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes)
{
    IndexSizeChecker<IT, Size>::check();
    const char *start = reinterpret_cast<const char *>(array);
    const char *offset = reinterpret_cast<const char *>(&(array->*(member1).*(member2)));
    d.v() = _mm512_i32extgather_epi32(((indexes * sizeof(S2)) + (offset - start)).data(), array, UpDownC<EntryType>(), _MM_SCALE_1, _MM_HINT_NONE);
}

template<> template<typename S1, typename S2, typename IT>
Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<double>::gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes, MaskArg mask)
{
    IndexSizeChecker<IT, Size>::check();
    const char *start = reinterpret_cast<const char *>(array);
    const char *offset = reinterpret_cast<const char *>(&(array->*(member1).*(member2)));
    d.v() = _mm512_mask_i32loextgather_pd(d.v(), mask.data(), ((indexes * sizeof(S2)) + (offset - start)).data(), array, UpDownC<EntryType>(), _MM_SCALE_1, _MM_HINT_NONE);
}
template<> template<typename S1, typename S2, typename IT>
Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<float>::gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes, MaskArg mask)
{
    IndexSizeChecker<IT, Size>::check();
    const char *start = reinterpret_cast<const char *>(array);
    const char *offset = reinterpret_cast<const char *>(&(array->*(member1).*(member2)));
    d.v() = _mm512_mask_i32extgather_ps(d.v(), mask.data(), ((indexes * sizeof(S2)) + (offset - start)).data(), array, UpDownC<EntryType>(), _MM_SCALE_1, _MM_HINT_NONE);
}
template<> template<typename S1, typename S2, typename IT>
Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<sfloat>::gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes, MaskArg mask)
{
    IndexSizeChecker<IT, Size>::check();
    const char *start = reinterpret_cast<const char *>(array);
    const char *offset = reinterpret_cast<const char *>(&(array->*(member1).*(member2)));
    d.v() = _mm512_mask_i32extgather_ps(d.v(), mask.data(), ((indexes * sizeof(S2)) + (offset - start)).data(), array, UpDownC<EntryType>(), _MM_SCALE_1, _MM_HINT_NONE);
}
template<> template<typename S1, typename S2, typename IT>
Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<int>::gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes, MaskArg mask)
{
    IndexSizeChecker<IT, Size>::check();
    const char *start = reinterpret_cast<const char *>(array);
    const char *offset = reinterpret_cast<const char *>(&(array->*(member1).*(member2)));
    d.v() = _mm512_mask_i32extgather_epi32(d.v(), mask.data(), ((indexes * sizeof(S2)) + (offset - start)).data(), array, UpDownC<EntryType>(), _MM_SCALE_1, _MM_HINT_NONE);
}
template<> template<typename S1, typename S2, typename IT>
Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<unsigned int>::gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes, MaskArg mask)
{
    IndexSizeChecker<IT, Size>::check();
    const char *start = reinterpret_cast<const char *>(array);
    const char *offset = reinterpret_cast<const char *>(&(array->*(member1).*(member2)));
    d.v() = _mm512_mask_i32extgather_epi32(d.v(), mask.data(), ((indexes * sizeof(S2)) + (offset - start)).data(), array, UpDownC<EntryType>(), _MM_SCALE_1, _MM_HINT_NONE);
}
template<> template<typename S1, typename S2, typename IT>
Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<short>::gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes, MaskArg mask)
{
    IndexSizeChecker<IT, Size>::check();
    const char *start = reinterpret_cast<const char *>(array);
    const char *offset = reinterpret_cast<const char *>(&(array->*(member1).*(member2)));
    d.v() = _mm512_mask_i32extgather_epi32(d.v(), mask.data(), ((indexes * sizeof(S2)) + (offset - start)).data(), array, UpDownC<EntryType>(), _MM_SCALE_1, _MM_HINT_NONE);
}
template<> template<typename S1, typename S2, typename IT>
Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<unsigned short>::gather(const S1 *array, const S2 S1::* member1, const EntryType S2::* member2, IT indexes, MaskArg mask)
{
    IndexSizeChecker<IT, Size>::check();
    const char *start = reinterpret_cast<const char *>(array);
    const char *offset = reinterpret_cast<const char *>(&(array->*(member1).*(member2)));
    d.v() = _mm512_mask_i32extgather_epi32(d.v(), mask.data(), ((indexes * sizeof(S2)) + (offset - start)).data(), array, UpDownC<EntryType>(), _MM_SCALE_1, _MM_HINT_NONE);
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
template<> template<typename S1, typename IT1, typename IT2>
Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<sfloat>::gather(const S1 *array, const EntryType *const S1::* ptrMember1, IT1 outerIndexes, IT2 innerIndexes)
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
    MicIntrinsics::scatter(mem, indexes.data(), d.v(), UpDownC<EntryType>(), sizeof(EntryType));
}
template<typename T> template<typename Index> Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<T>::scatter(EntryType *mem, Index indexes, MaskArg mask) const
{
    MicIntrinsics::scatter(mask.data(), mem, indexes.data(), d.v(), UpDownC<EntryType>(), sizeof(EntryType));
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
    MicIntrinsics::scatter(array, ((indexes * sizeof(S2)) + (offset - start)).data(), d.v(), UpDownC<EntryType>(), _MM_SCALE_1);
}
template<typename T> template<typename S1, typename S2, typename IT> Vc_ALWAYS_INLINE Vc_FLATTEN void Vector<T>::scatter(S1 *array, S2 S1::* member1, EntryType S2::* member2, IT indexes, MaskArg mask) const
{
    const char *start = reinterpret_cast<const char *>(array);
    const char *offset = reinterpret_cast<const char *>(&(array->*(member1).*(member2)));
    MicIntrinsics::scatter(mask.data(), array, ((indexes * sizeof(S2)) + (offset - start)).data(), d.v(), UpDownC<EntryType>(), _MM_SCALE_1);
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

template<> Vc_ALWAYS_INLINE Vector<sfloat> Vector<sfloat>::Random()
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

    return (Vector<double>(_cast(_mm512_srli_epi64(mic_cast<__m512>(state), 12))) | One()) - One();
}
// }}}1
// shifted / rotated {{{1
template<size_t SIMDWidth, size_t Size> struct VectorShift;
template<> struct VectorShift<64, 8>
{
    typedef __m512 VectorType;
    static Vc_INTRINSIC VectorType shifted(VC_ALIGNED_PARAMETER(VectorType) _v, int amount)
    {
        const __m512 z = _mm512_setzero_ps();
        const __m512d v = mic_cast<__m512d>(_v);
        // in memory  : ABCD
        // in register: DCBA
        switch (amount) {
        case  7: return mic_cast<VectorType>(_mm512_mask_permute4f128_ps(z, 0x0003,
                             mic_cast<__m512>(_mm512_swizzle_pd(v, _MM_SWIZ_REG_DDDD)), _MM_PERM_DDDD));
        case  6: return mic_cast<VectorType>(_mm512_mask_permute4f128_ps(z, 0x000f, mic_cast<__m512>(v), _MM_PERM_CBAD));
        case  5: {auto &&tmp1 = _mm512_swizzle_ps(mic_cast<__m512>(v), _MM_SWIZ_REG_BADC); // ghef cdab
                 auto &&tmp2 = mic_cast<__m512i>(_mm512_mask_permute4f128_ps(_mm512_permute4f128_ps(tmp1, _MM_PERM_DDDC),
                             0x000c, tmp1, _MM_PERM_DDDD));
                 return mic_cast<VectorType>(_mm512_mask_xor_epi32(tmp2, 0xffc0, tmp2, tmp2));}
        case  4: return mic_cast<VectorType>(_mm512_mask_permute4f128_ps(z, 0x00ff, mic_cast<__m512>(v), _MM_PERM_BADC));
        case  3: {auto &&tmp1 = _mm512_swizzle_ps(mic_cast<__m512>(v), _MM_SWIZ_REG_BADC); // ghef cdab
                 auto &&tmp2 = mic_cast<__m512i>(_mm512_mask_permute4f128_ps(_mm512_permute4f128_ps(tmp1, _MM_PERM_DDCB),
                             0x00cc, tmp1, _MM_PERM_DDDC));
                 return mic_cast<VectorType>(_mm512_mask_xor_epi32(tmp2, 0xfc00, tmp2, tmp2));}
        case  2: return mic_cast<VectorType>(_mm512_mask_permute4f128_ps(z, 0x0fff, mic_cast<__m512>(v), _MM_PERM_ADCB));
        case  1: {auto &&tmp1 = _mm512_swizzle_ps(mic_cast<__m512>(v), _MM_SWIZ_REG_BADC); // ghef cdab
                 auto &&tmp2 = mic_cast<__m512i>(_mm512_mask_permute4f128_ps(tmp1,
                             0x0ccc, tmp1, _MM_PERM_DDCB));
                 return mic_cast<VectorType>(_mm512_mask_xor_epi32(tmp2, 0xc000, tmp2, tmp2));}
        case  0: return _v;
        case -1: {auto &&tmp1 = _mm512_swizzle_ps(mic_cast<__m512>(v), _MM_SWIZ_REG_BADC); // ghef cdab
                 auto &&tmp2 = mic_cast<__m512i>(_mm512_mask_permute4f128_ps(tmp1,
                             0x3330, tmp1, _MM_PERM_CBAA));
                 return mic_cast<VectorType>(_mm512_mask_xor_epi32(tmp2, 0x0003, tmp2, tmp2));}
        case -2: return mic_cast<VectorType>(_mm512_mask_permute4f128_ps(z, 0xfff0, mic_cast<__m512>(v), _MM_PERM_CBAD));
        case -3: {auto &&tmp1 = _mm512_swizzle_ps(mic_cast<__m512>(v), _MM_SWIZ_REG_BADC); // ghef cdab
                 auto &&tmp2 = mic_cast<__m512i>(_mm512_mask_permute4f128_ps(_mm512_permute4f128_ps(tmp1, _MM_PERM_CBAA),
                             0x3300, tmp1, _MM_PERM_BAAA));
                 return mic_cast<VectorType>(_mm512_mask_xor_epi32(tmp2, 0x003f, tmp2, tmp2));}
        case -4: return mic_cast<VectorType>(_mm512_mask_permute4f128_ps(z, 0xff00, mic_cast<__m512>(v), _MM_PERM_BADC));
        case -5: return mic_cast<VectorType>(_mm512_mask_permute4f128_ps(z, 0xfc00, mic_cast<__m512>(_mm512_mask_mov_pd(
                                     _mm512_swizzle_pd(v, _MM_SWIZ_REG_AAAA), 0x07,
                                     _mm512_swizzle_pd(v, _MM_SWIZ_REG_DACB))), _MM_PERM_ABAA));
        case -6: return mic_cast<VectorType>(_mm512_mask_permute4f128_ps(z, 0xf000, mic_cast<__m512>(v), _MM_PERM_ADCB));
        case -7: return mic_cast<VectorType>(_mm512_mask_permute4f128_ps(z, 0xc000,
                             mic_cast<__m512>(_mm512_swizzle_pd(v, _MM_SWIZ_REG_AAAA)), _MM_PERM_AAAA));
        }
        return _mm512_setzero_ps();
    }
};
template<> struct VectorShift<64, 16>
{
    typedef __m512i VectorType;
    static Vc_INTRINSIC __m512i shifted(VC_ALIGNED_PARAMETER(VectorType) v, int amount)
    {
        const __m512i z = _mm512_setzero_epi32();
        switch (amount) {
        case 15: return _mm512_permute4f128_epi32(_mm512_mask_mov_epi32(z, 0x1000, _mm512_swizzle_epi32(v, _MM_SWIZ_REG_DDDD)), _MM_PERM_CBAD);
        case 14: return _mm512_permute4f128_epi32(_mm512_mask_mov_epi32(z, 0x3000, _mm512_swizzle_epi32(v, _MM_SWIZ_REG_BADC)), _MM_PERM_CBAD);
        case 13: return _mm512_permute4f128_epi32(_mm512_mask_shuffle_epi32(z, 0x7000, v, _MM_PERM_ADCB), _MM_PERM_CBAD);
        case 12: return mic_cast<__m512i>(_mm512_mask_permute4f128_epi32(z, 0x000f, v, _MM_PERM_CBAD));
        case 11: {auto &&tmp = _mm512_shuffle_epi32(v, _MM_PERM_CBAD);
                 return mic_cast<__m512i>(_mm512_mask_mov_epi32(
                             _mm512_mask_permute4f128_epi32(z, 0x000e, tmp, _MM_PERM_CBAD),
                             0x0011, _mm512_permute4f128_epi32(tmp, _MM_PERM_BADC)
                             ));}
        case 10: {auto &&tmp = _mm512_swizzle_epi32(v, _MM_SWIZ_REG_BADC);
                 return mic_cast<__m512i>(_mm512_mask_mov_epi32(
                             _mm512_mask_permute4f128_epi32(z, 0x000c, tmp, _MM_PERM_CBAD),
                             0x0033, _mm512_permute4f128_epi32(tmp, _MM_PERM_BADC)
                             ));}
        case  9: {auto &&tmp = _mm512_shuffle_epi32(v, _MM_PERM_ADCB);
                 return mic_cast<__m512i>(_mm512_mask_mov_epi32(
                             _mm512_mask_permute4f128_epi32(z, 0x0008, tmp, _MM_PERM_CBAD),
                             0x0077, _mm512_permute4f128_epi32(tmp, _MM_PERM_BADC)
                             ));}
        case  8: return mic_cast<__m512i>(_mm512_mask_permute4f128_epi32(z, 0x00ff, v, _MM_PERM_BADC));
        case  7: {auto &&tmp = _mm512_shuffle_epi32(v, _MM_PERM_CBAD);
                 return mic_cast<__m512i>(_mm512_mask_mov_epi32(
                             _mm512_mask_permute4f128_epi32(z, 0x00ee, tmp, _MM_PERM_BADC),
                             0x0111, _mm512_permute4f128_epi32(tmp, _MM_PERM_ADCB)
                             ));}
        case  6: {auto &&tmp = _mm512_swizzle_epi32(v, _MM_SWIZ_REG_BADC);
                 return mic_cast<__m512i>(_mm512_mask_mov_epi32(
                             _mm512_mask_permute4f128_epi32(z, 0x00cc, tmp, _MM_PERM_BADC),
                             0x0333, _mm512_permute4f128_epi32(tmp, _MM_PERM_ADCB)
                             ));}
        case  5: {auto &&tmp = _mm512_shuffle_epi32(v, _MM_PERM_ADCB);
                 return mic_cast<__m512i>(_mm512_mask_mov_epi32(
                             _mm512_mask_permute4f128_epi32(z, 0x0088, tmp, _MM_PERM_BADC),
                             0x0777, _mm512_permute4f128_epi32(tmp, _MM_PERM_ADCB)
                             ));}
        case  4: return mic_cast<__m512i>(_mm512_mask_permute4f128_epi32(z, 0x0fff, v, _MM_PERM_ADCB));
        case  3: {auto &&tmp = _mm512_shuffle_epi32(v, _MM_PERM_CBAD);
                 return mic_cast<__m512i>(_mm512_mask_mov_epi32(
                             _mm512_mask_permute4f128_epi32(z, 0x0eee, tmp, _MM_PERM_ADCB),
                             0x1111, tmp
                             ));}
        case  2: {auto &&tmp = _mm512_swizzle_epi32(v, _MM_SWIZ_REG_BADC);
                 return mic_cast<__m512i>(_mm512_mask_mov_epi32(
                             _mm512_mask_permute4f128_epi32(z, 0x0ccc, tmp, _MM_PERM_ADCB),
                             0x3333, tmp
                             ));}
        case  1: {auto &&tmp = _mm512_shuffle_epi32(v, _MM_PERM_ADCB);
                 return mic_cast<__m512i>(_mm512_mask_mov_epi32(
                             _mm512_mask_permute4f128_epi32(z, 0x0888, tmp, _MM_PERM_ADCB),
                             0x7777, tmp
                             ));}
        case  0: return mic_cast<__m512i>(v);
        case -1: {auto &&tmp = _mm512_shuffle_epi32(v, _MM_PERM_CBAD);
                 return mic_cast<__m512i>(_mm512_mask_mov_epi32(
                             _mm512_mask_permute4f128_epi32(z, 0x1110, tmp, _MM_PERM_CBAA),
                             0xeeee, tmp
                             ));}
        case -2: {auto &&tmp = _mm512_swizzle_epi32(v, _MM_SWIZ_REG_BADC);
                 return mic_cast<__m512i>(_mm512_mask_mov_epi32(
                             _mm512_mask_permute4f128_epi32(z, 0x3330, tmp, _MM_PERM_CBAA),
                             0xcccc, tmp
                             ));}
        case -3: {auto &&tmp = _mm512_shuffle_epi32(v, _MM_PERM_ADCB);
                 return mic_cast<__m512i>(_mm512_mask_mov_epi32(
                             _mm512_mask_permute4f128_epi32(z, 0x7770, tmp, _MM_PERM_CBAA),
                             0x8888, tmp
                             ));}
        case -4: return mic_cast<__m512i>(_mm512_mask_permute4f128_epi32(z, 0xfff0, v, _MM_PERM_CBAD));
        case -5: {auto &&tmp = _mm512_shuffle_epi32(v, _MM_PERM_CBAD);
                 return mic_cast<__m512i>(_mm512_mask_mov_epi32(
                             _mm512_mask_permute4f128_epi32(z, 0xeee0, tmp, _MM_PERM_CBAD),
                             0x1100, _mm512_permute4f128_epi32(tmp, _MM_PERM_BAAA)
                             ));}
        case -6: {auto &&tmp = _mm512_swizzle_epi32(v, _MM_SWIZ_REG_BADC);
                 return mic_cast<__m512i>(_mm512_mask_mov_epi32(
                             _mm512_mask_permute4f128_epi32(z, 0xccc0, tmp, _MM_PERM_CBAD),
                             0x3300, _mm512_permute4f128_epi32(tmp, _MM_PERM_BAAA)
                             ));}
        case -7: // ponm lkji hgfe dcba
                 // mpon ilkj ehgf adcb (tmp)
                 // i000 e000 a000 0000
                 // 0hgf 0dcb 0000 0000
                 // ihgf edcb a000 0000
                 {auto &&tmp = _mm512_shuffle_epi32(v, _MM_PERM_ADCB);
                 return mic_cast<__m512i>(_mm512_mask_mov_epi32(
                             _mm512_mask_permute4f128_epi32(z, 0x8880, tmp, _MM_PERM_CBAD),
                             0x7700, _mm512_permute4f128_epi32(tmp, _MM_PERM_BAAA)
                             ));}
        case -8: return mic_cast<__m512i>(_mm512_mask_permute4f128_epi32(z, 0xff00, v, _MM_PERM_BADC));
        case -9: // ponm lkji hgfe dcba
                 // onmp kjil gfeh cbad (tmp)
                 // gfe0 cba0 0000 0000
                 // 000d 0000 0000 0000
                 // gfed cba0 0000 0000
                 {auto &&tmp = _mm512_shuffle_epi32(v, _MM_PERM_CBAD);
                 return mic_cast<__m512i>(_mm512_mask_mov_epi32(
                             _mm512_mask_permute4f128_epi32(z, 0xee00, tmp, _MM_PERM_BADC),
                             0x1000, _mm512_permute4f128_epi32(tmp, _MM_PERM_AAAA)
                             ));}
        case-10: // ponm lkji hgfe dcba
                 // nmpo jilk fehg badc (tmp)
                 // fe00 ba00 0000 0000
                 // 00dc 0000 0000 0000
                 // fedc ba00 0000 0000
                 {auto &&tmp = _mm512_swizzle_epi32(v, _MM_SWIZ_REG_BADC);
                 return mic_cast<__m512i>(_mm512_mask_mov_epi32(
                             _mm512_mask_permute4f128_epi32(z, 0xcc00, tmp, _MM_PERM_BADC),
                             0x3000, _mm512_permute4f128_epi32(tmp, _MM_PERM_AAAA)
                             ));}
        case-11: // ponm lkji hgfe dcba
                 // mpon ilkj ehgf adcb (tmp)
                 // e000 a000 0000 0000
                 // 0dcb 0000 0000 0000
                 // edcb a000 0000 0000
                 {auto &&tmp = _mm512_shuffle_epi32(v, _MM_PERM_ADCB);
                 return mic_cast<__m512i>(_mm512_mask_mov_epi32(
                             _mm512_mask_permute4f128_epi32(z, 0x8800, tmp, _MM_PERM_BADC),
                             0x7000, _mm512_permute4f128_epi32(tmp, _MM_PERM_AAAA)
                             ));}
        case-12: return mic_cast<__m512i>(_mm512_mask_permute4f128_epi32(z, 0xf000, v, _MM_PERM_ABCD));
        case-13: return mic_cast<__m512i>(_mm512_permute4f128_epi32(_mm512_mask_shuffle_epi32(z, 0x000e, v, _MM_PERM_CBAD),
                             _MM_PERM_ABCD));
        case-14: return mic_cast<__m512i>(_mm512_permute4f128_epi32(_mm512_mask_mov_epi32(z,
                             0x000c, _mm512_swizzle_epi32(v, _MM_SWIZ_REG_BADC)), _MM_PERM_ABCD));
        case-15: return mic_cast<__m512i>(_mm512_permute4f128_epi32(_mm512_mask_mov_epi32(z,
                             0x0008, _mm512_swizzle_epi32(v, _MM_SWIZ_REG_AAAA)), _MM_PERM_ABCD));
        }
        return z;
    }
};
template<typename T> Vc_INTRINSIC Vector<T> Vector<T>::shifted(int amount) const
{
    typedef VectorShift<sizeof(VectorType), Size> VS;
    return _cast(VS::shifted(mic_cast<typename VS::VectorType>(d.v()), amount));
}

namespace
{
template<size_t SIMDWidth, size_t Size> struct VectorRotate;
template<> struct VectorRotate<64, 8>
{
    typedef __m512 VectorType;
    static Vc_INTRINSIC VectorType rotated(VC_ALIGNED_PARAMETER(VectorType) v, int amount)
    {
        switch (static_cast<unsigned int>(amount) % 8) {
        case  0: return v;
        case  1:{auto &&tmp1 = _mm512_swizzle_ps(v, _MM_SWIZ_REG_BADC);
                 return _mm512_mask_permute4f128_ps(tmp1, 0xcccc, tmp1, _MM_PERM_ADCB);}
        case  2: return _mm512_permute4f128_ps(v, _MM_PERM_ADCB);
        case  3:{auto &&tmp1 = _mm512_swizzle_ps(v, _MM_SWIZ_REG_BADC);
                 return _mm512_mask_permute4f128_ps(_mm512_permute4f128_ps(tmp1, _MM_PERM_ADCB), 0xcccc, tmp1, _MM_PERM_BADC);}
        case  4: return _mm512_permute4f128_ps(v, _MM_PERM_BADC);
        case  5:{auto &&tmp1 = _mm512_swizzle_ps(v, _MM_SWIZ_REG_BADC);
                 return _mm512_mask_permute4f128_ps(_mm512_permute4f128_ps(tmp1, _MM_PERM_BADC), 0xcccc, tmp1, _MM_PERM_CBAD);}
        case  6: return _mm512_permute4f128_ps(v, _MM_PERM_CBAD);
        case  7:{auto &&tmp1 = _mm512_swizzle_ps(v, _MM_SWIZ_REG_BADC);
                 return _mm512_mask_permute4f128_ps(_mm512_permute4f128_ps(tmp1, _MM_PERM_CBAD), 0xcccc, tmp1, _MM_PERM_DCBA);}
        }
        return _mm512_setzero_ps();
    }
};
template<> struct VectorRotate<64, 16>
{
    typedef __m512i VectorType;
    static Vc_INTRINSIC VectorType rotated(VC_ALIGNED_PARAMETER(VectorType) v, int amount)
    {
        switch (static_cast<unsigned int>(amount) % 16) {
        case 15: {auto &&tmp = _mm512_shuffle_epi32(v, _MM_PERM_CBAD);
                 return _mm512_mask_permute4f128_epi32(_mm512_permute4f128_epi32(tmp, _MM_PERM_CBAD), 0xeeee, tmp, _MM_PERM_DCBA);}
        case 14: {auto &&tmp = _mm512_swizzle_epi32(v, _MM_SWIZ_REG_BADC);
                 return _mm512_mask_permute4f128_epi32(_mm512_permute4f128_epi32(tmp, _MM_PERM_CBAD), 0xcccc, tmp, _MM_PERM_DCBA);}
        case 13: {auto &&tmp = _mm512_shuffle_epi32(v, _MM_PERM_ADCB);
                 return _mm512_mask_permute4f128_epi32(_mm512_permute4f128_epi32(tmp, _MM_PERM_CBAD), 0x8888, tmp, _MM_PERM_DCBA);}
        case 12: return _mm512_permute4f128_epi32(v, _MM_PERM_CBAD);
        case 11: {auto &&tmp = _mm512_shuffle_epi32(v, _MM_PERM_CBAD);
                 return _mm512_mask_permute4f128_epi32(_mm512_permute4f128_epi32(tmp, _MM_PERM_BADC), 0xeeee, tmp, _MM_PERM_CBAD);}
        case 10: {auto &&tmp = _mm512_swizzle_epi32(v, _MM_SWIZ_REG_BADC);
                 return _mm512_mask_permute4f128_epi32(_mm512_permute4f128_epi32(tmp, _MM_PERM_BADC), 0xcccc, tmp, _MM_PERM_CBAD);}
        case  9: {auto &&tmp = _mm512_shuffle_epi32(v, _MM_PERM_ADCB);
                 return _mm512_mask_permute4f128_epi32(_mm512_permute4f128_epi32(tmp, _MM_PERM_BADC), 0x8888, tmp, _MM_PERM_CBAD);}
        case  8: return _mm512_permute4f128_epi32(v, _MM_PERM_BADC);
        case  7: {auto &&tmp = _mm512_shuffle_epi32(v, _MM_PERM_CBAD);
                 return _mm512_mask_permute4f128_epi32(_mm512_permute4f128_epi32(tmp, _MM_PERM_ADCB), 0xeeee, tmp, _MM_PERM_BADC);}
        case  6: {auto &&tmp = _mm512_swizzle_epi32(v, _MM_SWIZ_REG_BADC);
                 return _mm512_mask_permute4f128_epi32(_mm512_permute4f128_epi32(tmp, _MM_PERM_ADCB), 0xcccc, tmp, _MM_PERM_BADC);}
        case  5: {auto &&tmp = _mm512_shuffle_epi32(v, _MM_PERM_ADCB);
                 return _mm512_mask_permute4f128_epi32(_mm512_permute4f128_epi32(tmp, _MM_PERM_ADCB), 0x8888, tmp, _MM_PERM_BADC);}
        case  4: return _mm512_permute4f128_epi32(v, _MM_PERM_ADCB);
        case  3: {auto &&tmp = _mm512_shuffle_epi32(v, _MM_PERM_CBAD);
                 return _mm512_mask_permute4f128_epi32(tmp, 0xeeee, tmp, _MM_PERM_ADCB);}
        case  2: {auto &&tmp = _mm512_swizzle_epi32(v, _MM_SWIZ_REG_BADC);
                 return _mm512_mask_permute4f128_epi32(tmp, 0xcccc, tmp, _MM_PERM_ADCB);}
        case  1: {auto &&tmp = _mm512_shuffle_epi32(v, _MM_PERM_ADCB);
                 return _mm512_mask_permute4f128_epi32(tmp, 0x8888, tmp, _MM_PERM_ADCB);}
        case  0: return v;
        }
        return _mm512_setzero_epi32();
    }
};
} // anonymous namespace
template<typename T> Vc_INTRINSIC Vector<T> Vector<T>::rotated(int amount) const
{
    typedef VectorRotate<sizeof(VectorType), Size> VR;
    return _cast(VR::rotated(mic_cast<typename VR::VectorType>(d.v()), amount));
}
// }}}1

Vc_NAMESPACE_END

// vim: foldmethod=marker
