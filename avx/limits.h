/*  This file is part of the Vc library. {{{
Copyright © 2009-2014 Matthias Kretz <kretz@kde.org>
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

#ifndef VC_AVX_LIMITS_H
#define VC_AVX_LIMITS_H

#include "intrinsics.h"
#include "types.h"
#include "macros.h"

namespace std
{
#define _VC_NUM_LIM(T, _max, _min) \
template<> struct numeric_limits< ::Vc::Vc_IMPL_NAMESPACE::Vector<T> > : public numeric_limits<T> \
{ \
    static Vc_INTRINSIC Vc_CONST ::Vc::Vc_IMPL_NAMESPACE::Vector<T> max()           Vc_NOEXCEPT { return _max; } \
    static Vc_INTRINSIC Vc_CONST ::Vc::Vc_IMPL_NAMESPACE::Vector<T> min()           Vc_NOEXCEPT { return _min; } \
    static Vc_INTRINSIC Vc_CONST ::Vc::Vc_IMPL_NAMESPACE::Vector<T> lowest()        Vc_NOEXCEPT { return min(); } \
    static Vc_INTRINSIC Vc_CONST ::Vc::Vc_IMPL_NAMESPACE::Vector<T> epsilon()       Vc_NOEXCEPT { return ::Vc::Vc_IMPL_NAMESPACE::Vector<T>::Zero(); } \
    static Vc_INTRINSIC Vc_CONST ::Vc::Vc_IMPL_NAMESPACE::Vector<T> round_error()   Vc_NOEXCEPT { return ::Vc::Vc_IMPL_NAMESPACE::Vector<T>::Zero(); } \
    static Vc_INTRINSIC Vc_CONST ::Vc::Vc_IMPL_NAMESPACE::Vector<T> infinity()      Vc_NOEXCEPT { return ::Vc::Vc_IMPL_NAMESPACE::Vector<T>::Zero(); } \
    static Vc_INTRINSIC Vc_CONST ::Vc::Vc_IMPL_NAMESPACE::Vector<T> quiet_NaN()     Vc_NOEXCEPT { return ::Vc::Vc_IMPL_NAMESPACE::Vector<T>::Zero(); } \
    static Vc_INTRINSIC Vc_CONST ::Vc::Vc_IMPL_NAMESPACE::Vector<T> signaling_NaN() Vc_NOEXCEPT { return ::Vc::Vc_IMPL_NAMESPACE::Vector<T>::Zero(); } \
    static Vc_INTRINSIC Vc_CONST ::Vc::Vc_IMPL_NAMESPACE::Vector<T> denorm_min()    Vc_NOEXCEPT { return ::Vc::Vc_IMPL_NAMESPACE::Vector<T>::Zero(); } \
}

_VC_NUM_LIM(unsigned short, ::Vc::Vc_IMPL_NAMESPACE::_mm_setallone_si128(), _mm_setzero_si128());
_VC_NUM_LIM(         short, _mm_srli_epi16(::Vc::Vc_IMPL_NAMESPACE::_mm_setallone_si128(), 1), ::Vc::Vc_IMPL_NAMESPACE::_mm_setmin_epi16());
_VC_NUM_LIM(  unsigned int, Vc::AvxIntrinsics::_mm_setallone_si128(), _mm_setzero_si128());
_VC_NUM_LIM(           int, _mm_srli_epi32(::Vc::AvxIntrinsics::_mm_setallone_si128(), 1), Vc::Vc_IMPL_NAMESPACE::_mm_setmin_epi32());
#undef _VC_NUM_LIM

} // namespace std

#include "undomacros.h"

#endif // VC_AVX_LIMITS_H
