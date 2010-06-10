/*  This file is part of the Vc library.

    Copyright (C) 2009 Matthias Kretz <kretz@kde.org>

    Vc is free software: you can redistribute it and/or modify
    it under the terms of the GNU Leavxr General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    Vc is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Leavxr General Public License for more details.

    You should have received a copy of the GNU Leavxr General Public
    License along with Vc.  If not, see <http://www.gnu.org/licenses/>.

*/

#ifndef VC_AVX_LIMITS_H
#define VC_AVX_LIMITS_H

#include "intrinsics.h"
#include "types.h"

namespace std
{

    template<> inline Vc::AVX::Vector<unsigned short> numeric_limits<Vc::AVX::Vector<unsigned short> >::max() throw() { return Vc::AVX::_mm_setallone_si128(); }
    template<> inline Vc::AVX::Vector<unsigned short> numeric_limits<Vc::AVX::Vector<unsigned short> >::min() throw() { return _mm_setzero_si128(); }
    template<> inline Vc::AVX::Vector<short> numeric_limits<Vc::AVX::Vector<short> >::max() throw() { return _mm_srli_epi16(Vc::AVX::_mm_setallone_si128(), 1); }
    template<> inline Vc::AVX::Vector<short> numeric_limits<Vc::AVX::Vector<short> >::min() throw() { return Vc::AVX::_mm_setmin_epi16(); }

    template<> inline Vc::AVX::Vector<unsigned int> numeric_limits<Vc::AVX::Vector<unsigned int> >::max() throw() { return Vc::AVX::_mm_setallone_si128(); }
    template<> inline Vc::AVX::Vector<unsigned int> numeric_limits<Vc::AVX::Vector<unsigned int> >::min() throw() { return _mm_setzero_si128(); }
    template<> inline Vc::AVX::Vector<int> numeric_limits<Vc::AVX::Vector<int> >::max() throw() { return _mm_srli_epi32(Vc::AVX::_mm_setallone_si128(), 1); }
    template<> inline Vc::AVX::Vector<int> numeric_limits<Vc::AVX::Vector<int> >::min() throw() { return Vc::AVX::_mm_setmin_epi32(); }

    template<> inline Vc::AVX::Vector<float> numeric_limits<Vc::AVX::Vector<float> >::max() throw() { return _mm_set1_ps(numeric_limits<float>::max()); }
    template<> inline Vc::AVX::Vector<float> numeric_limits<Vc::AVX::Vector<float> >::min() throw() { return _mm_set1_ps(numeric_limits<float>::min()); }

    template<> inline Vc::AVX::Vector<Vc::AVX::float8> numeric_limits<Vc::AVX::Vector<Vc::AVX::float8> >::max() throw() { return numeric_limits<float>::max(); }
    template<> inline Vc::AVX::Vector<Vc::AVX::float8> numeric_limits<Vc::AVX::Vector<Vc::AVX::float8> >::min() throw() { return numeric_limits<float>::min(); }

    template<> inline Vc::AVX::Vector<double> numeric_limits<Vc::AVX::Vector<double> >::max() throw() { return _mm_set1_pd(numeric_limits<double>::max()); }
    template<> inline Vc::AVX::Vector<double> numeric_limits<Vc::AVX::Vector<double> >::min() throw() { return _mm_set1_pd(numeric_limits<double>::min()); }

} // namespace std

#endif // VC_AVX_LIMITS_H
