/*  This file is part of the Vc library.

    Copyright (C) 2009-2011 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_IMPL_AVX2
namespace {
    using ::Vc::Vc_IMPL_NAMESPACE::_mm256_srli_epi32;
}
#endif
_VC_NUM_LIM(unsigned short, ::Vc::Vc_IMPL_NAMESPACE::_mm_setallone_si128(), _mm_setzero_si128());
_VC_NUM_LIM(         short, _mm_srli_epi16(::Vc::Vc_IMPL_NAMESPACE::_mm_setallone_si128(), 1), ::Vc::Vc_IMPL_NAMESPACE::_mm_setmin_epi16());
_VC_NUM_LIM(  unsigned int, ::Vc::Vc_IMPL_NAMESPACE::_mm256_setallone_si256(), _mm256_setzero_si256());
_VC_NUM_LIM(           int, _mm256_srli_epi32(::Vc::Vc_IMPL_NAMESPACE::_mm256_setallone_si256(), 1), ::Vc::Vc_IMPL_NAMESPACE::_mm256_setmin_epi32());
#undef _VC_NUM_LIM

} // namespace std

#include "undomacros.h"

#endif // VC_AVX_LIMITS_H
