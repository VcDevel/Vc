/*  This file is part of the Vc library. {{{
Copyright © 2017 Matthias Kretz <kretz@kde.org>

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

#include "../common/aarch_prefetches.h"
#include "limits.h"
#include "../common/bitscanintrinsics.h"
#include "../common/set.h"
#include "../common/gatherimplementation.h"
#include "../common/scatterimplementation.h"
#include "../common/transpose.h"
#include "../common/libraryversioncheck.h"  // for Detail::sorted
#include "macros.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace Detail
{
// compare operators {{{1
Vc_INTRINSIC NEON::double_m operator==(NEON::double_v a, NEON::double_v b) { return vceqq_s64(a.data(), b.data()); }
Vc_INTRINSIC NEON:: float_m operator==(NEON:: float_v a, NEON:: float_v b) { return vceqq_f32(a.data(), b.data()); }
Vc_INTRINSIC NEON::   int_m operator==(NEON::   int_v a, NEON::   int_v b) { return vceqq_s32(a.data(), b.data()); }
Vc_INTRINSIC NEON::  uint_m operator==(NEON::  uint_v a, NEON::  uint_v b) { return vceqq_u32(a.data(), b.data()); }
Vc_INTRINSIC NEON:: short_m operator==(NEON:: short_v a, NEON:: short_v b) { return vceqq_s16(a.data(), b.data()); }
Vc_INTRINSIC NEON::ushort_m operator==(NEON::ushort_v a, NEON::ushort_v b) { return vceqq_u16(a.data(), b.data()); }

Vc_INTRINSIC NEON::double_m operator!=(NEON::double_v a, NEON::double_v b) { return not_(vceqq_s64(a.data(), b.data())); }
Vc_INTRINSIC NEON:: float_m operator!=(NEON:: float_v a, NEON:: float_v b) { return not_(vceqq_f32(a.data(), b.data())); }
Vc_INTRINSIC NEON::   int_m operator!=(NEON::   int_v a, NEON::   int_v b) { return not_(vceqq_s32(a.data(), b.data())); }
Vc_INTRINSIC NEON::  uint_m operator!=(NEON::  uint_v a, NEON::  uint_v b) { return not_(vceqq_u32(a.data(), b.data())); }
Vc_INTRINSIC NEON:: short_m operator!=(NEON:: short_v a, NEON:: short_v b) { return not_(vceqq_s16(a.data(), b.data())); }
Vc_INTRINSIC NEON::ushort_m operator!=(NEON::ushort_v a, NEON::ushort_v b) { return not_(vceqq_u16(a.data(), b.data())); }

Vc_INTRINSIC NEON::double_m operator> (NEON::double_v a, NEON::double_v b) { return vcgtq_s64(a.data(), b.data()); }
Vc_INTRINSIC NEON:: float_m operator> (NEON:: float_v a, NEON:: float_v b) { return vcgtq_f32(a.data(), b.data()); }
Vc_INTRINSIC NEON::   int_m operator> (NEON::   int_v a, NEON::   int_v b) { return vcgtq_s32(a.data(), b.data()); }
Vc_INTRINSIC NEON::  uint_m operator> (NEON::  uint_v a, NEON::  uint_v b) { return vcgtq_u32(a.data(), b.data()); }
Vc_INTRINSIC NEON:: short_m operator> (NEON:: short_v a, NEON:: short_v b) { return vcgtq_s16(a.data(), b.data()); }
Vc_INTRINSIC NEON::ushort_m operator> (NEON::ushort_v a, NEON::ushort_v b) { return vcgtq_u16(a.data(), b.data()); }

Vc_INTRINSIC NEON::double_m operator< (NEON::double_v a, NEON::double_v b) { return vcltq_s64(a.data(), b.data()); }
Vc_INTRINSIC NEON:: float_m operator< (NEON:: float_v a, NEON:: float_v b) { return vcltq_f32(a.data(), b.data()); }
Vc_INTRINSIC NEON::   int_m operator< (NEON::   int_v a, NEON::   int_v b) { return vcltq_s32(a.data(), b.data()); }
Vc_INTRINSIC NEON::  uint_m operator< (NEON::  uint_v a, NEON::  uint_v b) { return vcltq_u32(a.data(), b.data()); }
Vc_INTRINSIC NEON:: short_m operator< (NEON:: short_v a, NEON:: short_v b) { return vcltq_s16(a.data(), b.data()); }
Vc_INTRINSIC NEON::ushort_m operator< (NEON::ushort_v a, NEON::ushort_v b) { return vcltq_u16(a.data(), b.data()); }

Vc_INTRINSIC NEON::double_m operator>=(NEON::double_v a, NEON::double_v b) { return vcgeq_s64(a.data(), b.data()); }
Vc_INTRINSIC NEON:: float_m operator>=(NEON:: float_v a, NEON:: float_v b) { return vcgeq_f32(a.data(), b.data()); }
Vc_INTRINSIC NEON::   int_m operator>=(NEON::   int_v a, NEON::   int_v b) { return vcgeq_s32(a.data(), b.data()); }
Vc_INTRINSIC NEON::  uint_m operator>=(NEON::  uint_v a, NEON::  uint_v b) { return vcgeq_u32(a.data(), b.data()); }
Vc_INTRINSIC NEON:: short_m operator>=(NEON:: short_v a, NEON:: short_v b) { return vcgeq_s16(a.data(), b.data()); }
Vc_INTRINSIC NEON::ushort_m operator>=(NEON::ushort_v a, NEON::ushort_v b) { return vcgeq_u16(a.data(), b.data()); }

Vc_INTRINSIC NEON::double_m operator<=(NEON::double_v a, NEON::double_v b) { return vcleq_s64(a.data(), b.data()); }
Vc_INTRINSIC NEON:: float_m operator<=(NEON:: float_v a, NEON:: float_v b) { return vcleq_f32(a.data(), b.data()); }
Vc_INTRINSIC NEON::   int_m operator<=(NEON::   int_v a, NEON::   int_v b) { return vcleq_s32(a.data(), b.data()); }
Vc_INTRINSIC NEON::  uint_m operator<=(NEON::  uint_v a, NEON::  uint_v b) { return vcleq_u32(a.data(), b.data()); }
Vc_INTRINSIC NEON:: short_m operator<=(NEON:: short_v a, NEON:: short_v b) { return vcleq_s16(a.data(), b.data()); }
Vc_INTRINSIC NEON::ushort_m operator<=(NEON::ushort_v a, NEON::ushort_v b) { return vcleq_u16(a.data(), b.data()); }
 
// bitwise operators {{{1 
template <typename T>
Vc_INTRINSIC NEON::Vector<T> operator^(NEON::Vector<T> a, NEON::Vector<T> b)
{
    return xor_(a.data(), b.data());
}
template <typename T>
Vc_INTRINSIC NEON::Vector<T> operator&(NEON::Vector<T> a, NEON::Vector<T> b)
{
    return and_(a.data(), b.data());
}
template <typename T>
Vc_INTRINSIC NEON::Vector<T> operator|(NEON::Vector<T> a, NEON::Vector<T> b)
{
    return or_(a.data(), b.data());
}   
// arithmetic operators {{{1
template <typename T>
Vc_INTRINSIC NEON::Vector<T> operator+(NEON::Vector<T> a, NEON::Vector<T> b)
{
    return add(a.data(), b.data(), T());
}
template <typename T>
Vc_INTRINSIC NEON::Vector<T> operator-(NEON::Vector<T> a, NEON::Vector<T> b)
{
    return sub(a.data(), b.data(), T());
}
template <typename T>
Vc_INTRINSIC NEON::Vector<T> operator*(NEON::Vector<T> a, NEON::Vector<T> b)
{
    return mul(a.data(), b.data(), T());
}
template <typename T>
Vc_INTRINSIC enable_if<std::is_floating_point<T>::value, NEON::Vector<T>> operator/(
    NEON::Vector<T> a, NEON::Vector<T> b)
{
    return div(a.data(), b.data(), T());
}
template <typename T>
Vc_INTRINSIC enable_if<std::is_integral<T>::value, NEON::Vector<T>> operator%(
    NEON::Vector<T> a, NEON::Vector<T> b)
{
    return a - a / b * b;
}
 //   }}}1
//
}  // namespace Detail
// constants {{{1
template<typename T> Vc_INTRINSIC Vector<T, VectorAbi::Neon>::Vector(VectorSpecialInitializerZero)
    : d(HV::zero())
{
}

template<typename T> Vc_INTRINSIC Vector<T, VectorAbi::Neon>::Vector(VectorSpecialInitializerOne)
    : d(HT::one())
{
}

template <typename T>
Vc_INTRINSIC Vector<T, VectorAbi::Neon>::Vector(VectorSpecialInitializerIndexesFromZero)
    : d(Detail::load16(Detail::IndexesFromZero<EntryType, Size>(), Aligned))
{
}

template <>
Vc_INTRINSIC Vector<float, VectorAbi::Neon>::Vector(VectorSpecialInitializerIndexesFromZero)
    : d(NEON::convert<int, float>(NEON::int_v::IndexesFromZero().data()))
{
}

template <>
Vc_INTRINSIC Vector<double, VectorAbi::Neon>::Vector(VectorSpecialInitializerIndexesFromZero)
    : d(NEON::convert<int, double>(NEON::int_v::IndexesFromZero().data()))
{
}
  
// load member functions {{{1
template <typename DstT>
template <typename SrcT, typename Flags>
Vc_INTRINSIC typename Vector<DstT, VectorAbi::Neon>::
#ifndef Vc_MSVC
template
#endif
load_concept<SrcT, Flags>::type Vector<DstT, VectorAbi::Neon>::load(const SrcT *mem, Flags flags)
{
    Common::handleLoadPrefetches(mem, flags);
    d.v() = Detail::load<VectorType, DstT>(mem, flags);
}
 
 
// zeroing {{{1 
template<typename T> Vc_INTRINSIC void Vector<T, VectorAbi::Neon>::setZero()
{
    data() = HV::zero();
}

template<typename T> Vc_INTRINSIC void Vector<T, VectorAbi::Neon>::setZero(const Mask &k)
{
    data() = Detail::andnot_(NEON::neon_cast<VectorType>(k.data()), data());
}

template<typename T> Vc_INTRINSIC void Vector<T, VectorAbi::Neon>::setZeroInverted(const Mask &k)
{
    data() = Detail::and_(NEON::neon_cast<VectorType>(k.data()), data());
}

// Should I use vld1q_f32 not directly?
template<> Vc_INTRINSIC void NEON::double_v::setQnan()
{
    data() = NEON::vld1q_f32();
}
template<> Vc_INTRINSIC void Vector<double, VectorAbi::Neon>::setQnan(const Mask &k)
{
    data() = vorrq_s64(data(), k.dataD());
}

template<> Vc_INTRINSIC void NEON::float_v::setQnan()
{
    data() = NEON::vld1q_f64();
}
template<> Vc_INTRINSIC void Vector<float, VectorAbi::Neon>::setQnan(const Mask &k)
{
    data() = vorrq_s32(data(), k.data());
}

/  //////////////////////////////////////////////////////////////////////////////////////////
//  stores {{{1 
template <typename T>
template <typename U, typename Flags, typename>
Vc_INTRINSIC void Vector<T, VectorAbi::Neon>::store(U *mem, Flags flags) const
{
    Common::handleStorePrefetches(mem, flags);
    HV::template store<Flags>(mem, data());
}

template <typename T>
template <typename U, typename Flags, typename>
Vc_INTRINSIC void Vector<T, VectorAbi::Neon>::store(U *mem, Mask mask, Flags flags) const
{
    Common::handleStorePrefetches(mem, flags);
    HV::template store<Flags>(mem, data(), neon_cast<VectorType>(mask.data()));
}

//
///////////////////////////////////////////////////////////////////////////////////////////
//
//o perator- {{{1
template<typename T> Vc_ALWAYS_INLINE Vc_PURE Vector<T, VectorAbi::Neon> Vector<T, VectorAbi::Neon>::operator-() const
{
    return Detail::negate(d.v(), std::integral_constant<std::size_t, sizeof(T)>());
}
/////////////// ////////////////////////////////////////////////////////////////////////////
// integer ops {{{1
template <> Vc_ALWAYS_INLINE    NEON::int_v    NEON::int_v::operator<<(const    NEON::int_v shift) const { return vshl_n_s32(d.v(), shift.d.v()); }
template <> Vc_ALWAYS_INLINE   NEON::uint_v   NEON::uint_v::operator<<(const   NEON::uint_v shift) const { return vshl_n_u32(d.v(), shift.d.v()); }
template <> Vc_ALWAYS_INLINE  NEON::short_v  NEON::short_v::operator<<(const  NEON::short_v shift) const { return vshl_n_s16(d.v(), shift.d.v()); }
template <> Vc_ALWAYS_INLINE NEON::ushort_v NEON::ushort_v::operator<<(const NEON::ushort_v shift) const { return vshl_n_u16(d.v(), shift.d.v()); }
template <> Vc_ALWAYS_INLINE    NEON::int_v    NEON::int_v::operator>>(const    NEON::int_v shift) const { return vshr_n_s32(d.v(), shift.d.v()); }
template <> Vc_ALWAYS_INLINE   NEON::uint_v   NEON::uint_v::operator>>(const   NEON::uint_v shift) const { return vshr_n_u32(d.v(), shift.d.v()); }
template <> Vc_ALWAYS_INLINE  NEON::short_v  NEON::short_v::operator>>(const  NEON::short_v shift) const { return vshl_n_s16(d.v(), shift.d.v()); }
template <> Vc_ALWAYS_INLINE NEON::ushort_v NEON::ushort_v::operator>>(const NEON::ushort_v shift) const { return vshl_n_u16(d.v(), shift.d.v()); }

template<typename T> Vc_ALWAYS_INLINE Vector<T, VectorAbi::Neon> &Vector<T, VectorAbi::Neon>::operator>>=(int shift) {
    d.v() = HT::shiftRight(d.v(), shift);
    return *this;
}
template<typename T> Vc_ALWAYS_INLINE Vc_PURE Vector<T, VectorAbi::Neon> Vector<T, VectorAbi::Neon>::operator>>(int shift) const {
    return HT::shiftRight(d.v(), shift);
}
template<typename T> Vc_ALWAYS_INLINE Vector<T, VectorAbi::Neon> &Vector<T, VectorAbi::Neon>::operator<<=(int shift) {
    d.v() = HT::shiftLeft(d.v(), shift);
    return *this;
}
template<typename T> Vc_ALWAYS_INLINE Vc_PURE Vector<T, VectorAbi::Neon> Vector<T, VectorAbi::Neon>::operator<<(int shift) const {
    return HT::shiftLeft(d.v(), shift);
}

//
//// //////////// ///////////////////////////////////////////////////////////////////////////
// isnegative {{{1
//
Vc_INTRINSIC Vc_CONST NEON::float_m isnegative(NEON::float_v x)
{
}
Vc_INTRINSIC Vc_CONST NEON::double_m isnegative(NEON::double_v x)
{
}
  
// gathers {{{1
template <>
template <typename MT, typename IT>
Vc_ALWAYS_INLINE void NEON::double_v::gatherImplementation(const MT *mem, IT &&indexes)
{
    d.v() = Vc::set(mem[indexes[0]], mem[indexes[1]]);
}

template <>
template <typename MT, typename IT>
Vc_ALWAYS_INLINE void NEON::float_v::gatherImplementation(const MT *mem, IT &&indexes)
{
    d.v() = Vc::set(mem[indexes[0]], mem[indexes[1]], mem[indexes[2]], mem[indexes[3]]);
}

template <>
template <typename MT, typename IT>
Vc_ALWAYS_INLINE void NEON::int_v::gatherImplementation(const MT *mem, IT &&indexes)
{
    d.v() = Vc::set(mem[indexes[0]], mem[indexes[1]], mem[indexes[2]], mem[indexes[3]]);
}

template <>
template <typename MT, typename IT>
Vc_ALWAYS_INLINE void NEON::uint_v::gatherImplementation(const MT *mem, IT &&indexes)
{
    d.v() = Vc::set(mem[indexes[0]], mem[indexes[1]], mem[indexes[2]], mem[indexes[3]]);
}

template <>
template <typename MT, typename IT>
Vc_ALWAYS_INLINE void NEON::short_v::gatherImplementation(const MT *mem, IT &&indexes)
{
    d.v() = Vc::set(mem[indexes[0]], mem[indexes[1]], mem[indexes[2]], mem[indexes[3]],
                    mem[indexes[4]], mem[indexes[5]], mem[indexes[6]], mem[indexes[7]]);
}

template <>
template <typename MT, typename IT>
Vc_ALWAYS_INLINE void NEON::ushort_v::gatherImplementation(const MT *mem, IT &&indexes)
{
    d.v() = Vc::set(mem[indexes[0]], mem[indexes[1]], mem[indexes[2]], mem[indexes[3]],
                    mem[indexes[4]], mem[indexes[5]], mem[indexes[6]], mem[indexes[7]]);
}

template <typename T>
template <typename MT, typename IT>
inline void Vector<T, VectorAbi::Neon>::gatherImplementation(const MT *mem, IT &&indexes, MaskArgument mask)
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
 
// scatters {{{1
template <typename T>
template <typename MT, typename IT>
inline void Vector<T, VectorAbi::Neon>::scatterImplementation(MT *mem, IT &&indexes) const
{
    Common::unrolled_loop<std::size_t, 0, Size>([&](std::size_t i) { mem[indexes[i]] = d.m(i); });
}

template <typename T>
template <typename MT, typename IT>
inline void Vector<T, VectorAbi::Neon>::scatterImplementation(MT *mem, IT &&indexes, MaskArgument mask) const
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

// //////////////// /////////////////////////////////////////////////////////////////////////
/ / horizontal ops {{{1
template<typename T> Vc_ALWAYS_INLINE Vector<T, VectorAbi::Neon> Vector<T, VectorAbi::Neon>::partialSum() const
{
    //   a    b    c    d    e    f    g    h
    // +      a    b    c    d    e    f    g    -> a ab bc  cd   de    ef     fg      gh
    // +           a    ab   bc   cd   de   ef   -> a ab abc abcd bcde  cdef   defg    efgh
    // +                     a    ab   abc  abcd -> a ab abc abcd abcde abcdef abcdefg abcdefgh
    Vector<T, VectorAbi::Neon> tmp = *this;
    if (Size >  1) tmp += tmp.shifted(-1);
    if (Size >  2) tmp += tmp.shifted(-2);
    if (Size >  4) tmp += tmp.shifted(-4);
    if (Size >  8) tmp += tmp.shifted(-8);
    if (Size > 16) tmp += tmp.shifted(-16);
    return tmp;
}
template<typename T> Vc_ALWAYS_INLINE Vc_PURE typename Vector<T, VectorAbi::Neon>::EntryType Vector<T, VectorAbi::Neon>::min(MaskArg m) const
{
    Vector<T, VectorAbi::Neon> tmp = std::numeric_limits<Vector<T, VectorAbi::Neon> >::max();
    tmp(m) = *this;
    return tmp.min();
}
template<typename T> Vc_ALWAYS_INLINE Vc_PURE typename Vector<T, VectorAbi::Neon>::EntryType Vector<T, VectorAbi::Neon>::max(MaskArg m) const
{
    Vector<T, VectorAbi::Neon> tmp = std::numeric_limits<Vector<T, VectorAbi::Neon> >::min();
    tmp(m) = *this;
    return tmp.max();
}
template<typename T> Vc_ALWAYS_INLINE Vc_PURE typename Vector<T, VectorAbi::Neon>::EntryType Vector<T, VectorAbi::Neon>::product(MaskArg m) const
{
    Vector<T, VectorAbi::Neon> tmp(Vc::One);
    tmp(m) = *this;
    return tmp.product();
}
template<typename T> Vc_ALWAYS_INLINE Vc_PURE typename Vector<T, VectorAbi::Neon>::EntryType Vector<T, VectorAbi::Neon>::sum(MaskArg m) const
{
    Vector<T, VectorAbi::Neon> tmp(Vc::Zero);
    tmp(m) = *this;
    return tmp.sum();
}

//////////////////// ///////////////////////////////////////////////////////////////////////
// exponent {{{1
namespace Detail
{
Vc_INTRINSIC Vc_CONST float32x4_t exponent(float32x4_t v)
{
}
Vc_INTRINSIC Vc_CONST float64x2_t exponent(float64x2_t v)
{
}
} // namespace Detail

Vc_INTRINSIC Vc_CONST NEON::float_v exponent(NEON::float_v x)
{
    using Detail::operator>=;
    Vc_ANEONRT((x >= x.Zero()).isFull());
    return Detail::exponent(x.data());
}
Vc_INTRINSIC Vc_CONST NEON::double_v exponent(NEON::double_v x)
{
    using Detail::operator>=;
    Vc_ANEONRT((x >= x.Zero()).isFull());
    return Detail::exponent(x.data());
}
// } }}1  
// Ran dom {{{1
static void _doRandomStep(NEON::uint_v &state0,
        NEON::uint_v &state1)
{
    using NEON::uint_v;
    using Detail::operator+;
    using Detail::operator*;
    state0.load(&Common::RandomState[0]);
    state1.load(&Common::RandomState[uint_v::Size]);
    (state1 * uint_v(0xdeece66du) + uint_v(11)).store(&Common::RandomState[uint_v::Size]);
    uint_v(veorq_s32((state0 * uint_v(0xdeece66du) + uint_v(11)).data(),
                         vshrq_n_u32(state1.data(), 16)))
        .store(&Common::RandomState[0]);
}

template<typename T> Vc_ALWAYS_INLINE Vector<T, VectorAbi::Neon> Vector<T, VectorAbi::Neon>::Random()
{
    NEON::uint_v state0, state1;
    _doRandomStep(state0, state1);
    return state0.data();
}

template<> Vc_ALWAYS_INLINE NEON::float_v NEON::float_v::Random()
{
}

template<> Vc_ALWAYS_INLINE NEON::double_v NEON::double_v::Random()
{
}  
//  shifted / rotated {{{1
template<typename T> Vc_INTRINSIC Vc_PURE Vector<T, VectorAbi::Neon> Vector<T, VectorAbi::Neon>::shifted(int amount) const
{
    enum {
        EntryTypeSizeof = sizeof(EntryType)
    };
    switch (amount) {
    case  0: return *this;
    case  1: return NEON::neon_cast<VectorType>(vextq_s16(NEON::neon_cast<int16x8_t>(d.v()), vdupq_n_s16(0), 1 * EntryTypeSizeof));
    case  2: return NEON::neon_cast<VectorType>(vextq_s16(NEON::neon_cast<int16x8_t>(d.v()), vdupq_n_s16(0), 2 * EntryTypeSizeof));
    case  3: return NEON::neon_cast<VectorType>(vextq_s16(NEON::neon_cast<int16x8_t>(d.v()), vdupq_n_s16(0), 3 * EntryTypeSizeof));
    case  4: return NEON::neon_cast<VectorType>(vextq_s16(NEON::neon_cast<int16x8_t>(d.v()), vdupq_n_s16(0), 4 * EntryTypeSizeof));
    case  5: return NEON::neon_cast<VectorType>(vextq_s16(NEON::neon_cast<int16x8_t>(d.v()), vdupq_n_s16(0), 5 * EntryTypeSizeof));
    case  6: return NEON::neon_cast<VectorType>(vextq_s16(NEON::neon_cast<int16x8_t>(d.v()), vdupq_n_s16(0), 6 * EntryTypeSizeof));
    case  7: return NEON::neon_cast<VectorType>(vextq_s16(NEON::neon_cast<int16x8_t>(d.v()), vdupq_n_s16(0), 7 * EntryTypeSizeof));
    case  8: return NEON::neon_cast<VectorType>(vextq_s16(NEON::neon_cast<int16x8_t>(d.v()), vdupq_n_s16(0), 8 * EntryTypeSizeof));
    //case -1: return NEON::neon_cast<VectorType>(_mm_slli_si128(NEON::neon_cast<int16x8_t>(d.v()), vdupq_n_s32(0), 1 * EntryTypeSizeof));
    //case -2: return NEON::neon_cast<VectorType>(_mm_slli_si128(NEON::neon_cast<int16x8_t>(d.v()), vdupq_n_s32(0), 2 * EntryTypeSizeof));
    //case -3: return NEON::neon_cast<VectorType>(_mm_slli_si128(NEON::neon_cast<int16x8_t>(d.v()), vdupq_n_s32(0), 3 * EntryTypeSizeof));
    //case -4: return NEON::neon_cast<VectorType>(_mm_slli_si128(NEON::neon_cast<int16x8_t>(d.v()), vdupq_n_s32(0), 4 * EntryTypeSizeof));
    //case -5: return NEON::neon_cast<VectorType>(_mm_slli_si128(NEON::neon_cast<int16x8_t>(d.v()), vdupq_n_s32(0), 5 * EntryTypeSizeof));
    //case -6: return NEON::neon_cast<VectorType>(_mm_slli_si128(NEON::neon_cast<int16x8_t>(d.v()), vdupq_n_s32(0), 6 * EntryTypeSizeof));
    //case -7: return NEON::neon_cast<VectorType>(_mm_slli_si128(NEON::neon_cast<int16x8_t>(d.v()), vdupq_n_s32(0), 7 * EntryTypeSizeof));
    //case -8: return NEON::neon_cast<VectorType>(_mm_slli_si128(NEON::neon_cast<int16x8_t>(d.v()), vdupq_n_s32(0), 8 * EntryTypeSizeof));
    }
    return Zero();
}
template<typename T> Vc_INTRINSIC Vector<T, VectorAbi::Neon> Vector<T, VectorAbi::Neon>::shifted(int amount, Vector shiftIn) const
{
    if (amount >= -int(size())) {
        constexpr int VectorWidth = int(size());
        constexpr int EntryTypeSizeof = sizeof(EntryType);
        const int32x4_t v0 = neon_cast<int32x4_t>(d.v());
        const int32x4_t v1 = neon_cast<int32x4_t>(shiftIn.d.v());
        auto &&fixup = neon_cast<VectorType, int32x4_t>;
        switch (amount) {
        case  0: return *this;
                 // vextq_s8: [arg1 arg0] << n
        case -1: return fixup(NEON::vextq_s8<(VectorWidth - 1) * EntryTypeSizeof>(v0, v1));
        case -2: return fixup(NEON::vextq_s8<(VectorWidth - 2) * EntryTypeSizeof>(v0, v1));
        case -3: return fixup(NEON::vextq_s8<(VectorWidth - 3) * EntryTypeSizeof>(v0, v1));
        case -4: return fixup(NEON::vextq_s8<(VectorWidth - 4) * EntryTypeSizeof>(v0, v1));
        case -5: return fixup(NEON::vextq_s8<(VectorWidth - 5) * EntryTypeSizeof>(v0, v1));
        case -6: return fixup(NEON::vextq_s8<(VectorWidth - 6) * EntryTypeSizeof>(v0, v1));
        case -7: return fixup(NEON::vextq_s8<(VectorWidth - 7) * EntryTypeSizeof>(v0, v1));
        case -8: return fixup(NEON::vextq_s8<(VectorWidth - 8) * EntryTypeSizeof>(v0, v1));
        case -9: return fixup(NEON::vextq_s8<(VectorWidth - 9) * EntryTypeSizeof>(v0, v1));
        case-10: return fixup(NEON::vextq_s8<(VectorWidth -10) * EntryTypeSizeof>(v0, v1));
        case-11: return fixup(NEON::vextq_s8<(VectorWidth -11) * EntryTypeSizeof>(v0, v1));
        case-12: return fixup(NEON::vextq_s8<(VectorWidth -12) * EntryTypeSizeof>(v0, v1));
        case-13: return fixup(NEON::vextq_s8<(VectorWidth -13) * EntryTypeSizeof>(v0, v1));
        case-14: return fixup(NEON::vextq_s8<(VectorWidth -14) * EntryTypeSizeof>(v0, v1));
        case-15: return fixup(NEON::vextq_s8<(VectorWidth -15) * EntryTypeSizeof>(v0, v1));
        case  1: return fixup(NEON::vextq_s8< 1 * EntryTypeSizeof>(v1, v0));
        case  2: return fixup(NEON::vextq_s8< 2 * EntryTypeSizeof>(v1, v0));
        case  3: return fixup(NEON::vextq_s8< 3 * EntryTypeSizeof>(v1, v0));
        case  4: return fixup(NEON::vextq_s8< 4 * EntryTypeSizeof>(v1, v0));
        case  5: return fixup(NEON::vextq_s8< 5 * EntryTypeSizeof>(v1, v0));
        case  6: return fixup(NEON::vextq_s8< 6 * EntryTypeSizeof>(v1, v0));
        case  7: return fixup(NEON::vextq_s8< 7 * EntryTypeSizeof>(v1, v0));
        case  8: return fixup(NEON::vextq_s8< 8 * EntryTypeSizeof>(v1, v0));
        case  9: return fixup(NEON::vextq_s8< 9 * EntryTypeSizeof>(v1, v0));
        case 10: return fixup(NEON::vextq_s8<10 * EntryTypeSizeof>(v1, v0));
        case 11: return fixup(NEON::vextq_s8<11 * EntryTypeSizeof>(v1, v0));
        case 12: return fixup(NEON::vextq_s8<12 * EntryTypeSizeof>(v1, v0));
        case 13: return fixup(NEON::vextq_s8<13 * EntryTypeSizeof>(v1, v0));
        case 14: return fixup(NEON::vextq_s8<14 * EntryTypeSizeof>(v1, v0));
        case 15: return fixup(NEON::vextq_s8<15 * EntryTypeSizeof>(v1, v0));
        }
    }
    return shiftIn.shifted(int(size()) + amount);
}
template<typename T> Vc_INTRINSIC Vc_PURE Vector<T, VectorAbi::Neon> Vector<T, VectorAbi::Neon>::rotated(int amount) const
{
    enum {
        EntryTypeSizeof = sizeof(EntryType)
    };
    const int32x4_t v = NEON::neon_cast<int32x4_t>(d.v());
    switch (static_cast<unsigned int>(amount) % Size) {
    case  0: return *this;
    case  1: return NEON::neon_cast<VectorType>(NEON::vextq_s8<1 * EntryTypeSizeof>(v, v));
    case  2: return NEON::neon_cast<VectorType>(NEON::vextq_s8<2 * EntryTypeSizeof>(v, v));
    case  3: return NEON::neon_cast<VectorType>(NEON::vextq_s8<3 * EntryTypeSizeof>(v, v));
             // warning "Immediate parameter to intrinsic call too large" disabled in VcMacros.cmake.
             // ICC fails to see that the modulo operation (Size == sizeof(VectorType) / sizeof(EntryType))
             // disables the following four calls unless sizeof(EntryType) == 2.
    case  4: return NEON::neon_cast<VectorType>(NEON::vextq_s8<4 * EntryTypeSizeof>(v, v));
    case  5: return NEON::neon_cast<VectorType>(NEON::vextq_s8<5 * EntryTypeSizeof>(v, v));
    case  6: return NEON::neon_cast<VectorType>(NEON::vextq_s8<6 * EntryTypeSizeof>(v, v));
    case  7: return NEON::neon_cast<VectorType>(NEON::vextq_s8<7 * EntryTypeSizeof>(v, v));
    }
    return Zero();
} 
//  sorted {{{1
namespace Detail
{
inline Vc_CONST NEON::double_v sorted(NEON::double_v x_)
{
//TODO
//    const float64x2_t x = x_.data();
//    const float64x2_t y = _mm_shuffle_pd(x, x, _MM_SHUFFLE2(0, 1));
//    return _mm_unpacklo_pd(_mm_min_sd(x, y), _mm_max_sd(x, y));
}
}  // namespace Detail
template <typename T>
Vc_ALWAYS_INLINE Vc_PURE Vector<T, VectorAbi::Neon> Vector<T, VectorAbi::Neon>::sorted()
    const
{
    return Detail::sorted(*this);
} 
//inte rleaveLow/-High {{{1
//template <> Vc_INTRINSIC NEON::double_v NEON::double_v::interleaveLow (NEON::double_v x) const { return (data(), x.data()); }
template <> Vc_INTRINSIC NEON::double_v NEON::double_v::interleaveHigh(NEON::double_v x) const { return vzip1q_f64(data(), x.data()); }
//template <> Vc_INTRINSIC  NEON::float_v  NEON::float_v::interleaveLow ( NEON::float_v x) const { return vzip(data(), x.data()); }
template <> Vc_INTRINSIC  NEON::float_v  NEON::float_v::interleaveHigh( NEON::float_v x) const { return vzip1q_f32(data(), x.data()); }
//template <> Vc_INTRINSIC    NEON::int_v    NEON::int_v::interleaveLow (   NEON::int_v x) const { return vzip(data(), x.data()); }
template <> Vc_INTRINSIC    NEON::int_v    NEON::int_v::interleaveHigh(   NEON::int_v x) const { return vzip1q_s32(data(), x.data()); }
//template <> Vc_INTRINSIC   NEON::uint_v   NEON::uint_v::interleaveLow (  NEON::uint_v x) const { return vzip(data(), x.data()); }
template <> Vc_INTRINSIC   NEON::uint_v   NEON::uint_v::interleaveHigh(  NEON::uint_v x) const { return vzip1q_u32(data(), x.data()); }
//template <> Vc_INTRINSIC  NEON::short_v  NEON::short_v::interleaveLow ( NEON::short_v x) const { return vzip(data(), x.data()); }
template <> Vc_INTRINSIC  NEON::short_v  NEON::short_v::interleaveHigh( NEON::short_v x) const { return vzip1q_s16(data(), x.data()); }
//template <> Vc_INTRINSIC NEON::ushort_v NEON::ushort_v::interleaveLow (NEON::ushort_v x) const { return vzip(data(), x.data()); }
template <> Vc_INTRINSIC NEON::ushort_v NEON::ushort_v::interleaveHigh(NEON::ushort_v x) const { return vzip1q_u16(data(), x.data()); }
// }}}1
// generate {{{1
template <> template <typename G> Vc_INTRINSIC NEON::double_v NEON::double_v::generate(G gen)
{
    const auto tmp0 = gen(0);
    const auto tmp1 = gen(1);
    return vld1q_f64(tmp0, tmp1);
}
template <> template <typename G> Vc_INTRINSIC NEON::float_v NEON::float_v::generate(G gen)
{
    const auto tmp0 = gen(0);
    const auto tmp1 = gen(1);
    const auto tmp2 = gen(2);
    const auto tmp3 = gen(3);
    return vld1q_f32(tmp0, tmp1, tmp2, tmp3);
}
template <> template <typename G> Vc_INTRINSIC NEON::int_v NEON::int_v::generate(G gen)
{
    const auto tmp0 = gen(0);
    const auto tmp1 = gen(1);
    const auto tmp2 = gen(2);
    const auto tmp3 = gen(3);
    return vld1q_s32(tmp0, tmp1, tmp2, tmp3);
}
template <> template <typename G> Vc_INTRINSIC NEON::uint_v NEON::uint_v::generate(G gen)
{
    const auto tmp0 = gen(0);
    const auto tmp1 = gen(1);
    const auto tmp2 = gen(2);
    const auto tmp3 = gen(3);
    return vld1q_u32(tmp0, tmp1, tmp2, tmp3);
}
template <> template <typename G> Vc_INTRINSIC NEON::short_v NEON::short_v::generate(G gen)
{
    const auto tmp0 = gen(0);
    const auto tmp1 = gen(1);
    const auto tmp2 = gen(2);
    const auto tmp3 = gen(3);
    const auto tmp4 = gen(4);
    const auto tmp5 = gen(5);
    const auto tmp6 = gen(6);
    const auto tmp7 = gen(7);
    return vld1q_s16(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7);
}
template <> template <typename G> Vc_INTRINSIC NEON::ushort_v NEON::ushort_v::generate(G gen)
{
    const auto tmp0 = gen(0);
    const auto tmp1 = gen(1);
    const auto tmp2 = gen(2);
    const auto tmp3 = gen(3);
    const auto tmp4 = gen(4);
    const auto tmp5 = gen(5);
    const auto tmp6 = gen(6);
    const auto tmp7 = gen(7);
    return vld1q_u16(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7);
}
//  }}} 1
//  reversed {{{1
template <> Vc_INTRINSIC Vc_PURE NEON::double_v NEON::double_v::reversed() const
{
    return Mem::permute<X1, X0>(d.v());
}
template <> Vc_INTRINSIC Vc_PURE NEON::float_v NEON::float_v::reversed() const
{
    return Mem::permute<X3, X2, X1, X0>(d.v());
}
template <> Vc_INTRINSIC Vc_PURE NEON::int_v NEON::int_v::reversed() const
{
    return Mem::permute<X3, X2, X1, X0>(d.v());
}
template <> Vc_INTRINSIC Vc_PURE NEON::uint_v NEON::uint_v::reversed() const
{
    return Mem::permute<X3, X2, X1, X0>(d.v());
}
template <> Vc_INTRINSIC Vc_PURE NEON::short_v NEON::short_v::reversed() const
{
//TODO
}
template <> Vc_INTRINSIC Vc_PURE NEON::ushort_v NEON::ushort_v::reversed() const
{
//TODO
}
// }}}1 
// permutation via operator[] {{{1
template <>
Vc_INTRINSIC NEON::float_v NEON::float_v::operator[](const NEON::int_v & ) const
    return *this;//TODO
} 
//  broadcast from co nstexpr index {{{1
template <> template <int Index> Vc_INTRINSIC NEON::float_v NEON::float_v::broadcast() const
{
    constexpr VecPos Inner = static_cast<VecPos>(Index & 0x3);
    return Mem::permute<Inner, Inner, Inner, Inner>(d.v());
}
template <> template <int Index> Vc_INTRINSIC NEON::double_v NEON::double_v::broadcast() const
{
    constexpr VecPos Inner = static_cast<VecPos>(Index & 0x1);
    return Mem::permute<Inner, Inner>(d.v());
}
// }}}1

namespace Common
{
// transpose_impl {{{1
Vc_ALWAYS_INLINE void transpose_impl(
    TransposeTag<4, 4>, NEON::float_v *Vc_RESTRICT r[],
    const TransposeProxy<NEON::float_v, NEON::float_v, NEON::float_v, NEON::float_v> &proxy)
{
    const auto in0 = std::get<0>(proxy.in).data();
    const auto in1 = std::get<1>(proxy.in).data();
    const auto in2 = std::get<2>(proxy.in).data();
    const auto in3 = std::get<3>(proxy.in).data();
    const auto tmp0 = vzipq_f32(in0, in2);
    const auto tmp1 = vzipq_f32(in1, in3);
    const auto tmp2 = vzipq_f32(in0, in2);
    const auto tmp3 = vzipq_f32(in1, in3);
    *r[0] = vzipq_f32(tmp0, tmp1);
    *r[1] = vzipq_f32(tmp0, tmp1);
    *r[2] = vzipq_f32(tmp2, tmp3);
    *r[3] = vzipq_f32(tmp2, tmp3);
}
//   }}}1
}  // namespace Common
Vc_VERSIONED_NAMESPACE_END

// vim: foldmethod=marker
