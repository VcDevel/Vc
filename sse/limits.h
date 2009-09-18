/*  This file is part of the Vc library.

    Copyright (C) 2009 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_SSE_LIMITS_H
#define VC_SSE_LIMITS_H

#include "intrinsics.h"
#include "types.h"

namespace std
{

    template<> inline Vc::SSE::Vector<unsigned short> numeric_limits<Vc::SSE::Vector<unsigned short> >::max() throw() { return Vc::SSE::_mm_setallone_si128(); }
    template<> inline Vc::SSE::Vector<unsigned short> numeric_limits<Vc::SSE::Vector<unsigned short> >::min() throw() { return _mm_setzero_si128(); }
    template<> inline Vc::SSE::Vector<short> numeric_limits<Vc::SSE::Vector<short> >::max() throw() { return _mm_srli_epi16(Vc::SSE::_mm_setallone_si128(), 1); }
    template<> inline Vc::SSE::Vector<short> numeric_limits<Vc::SSE::Vector<short> >::min() throw() { return Vc::SSE::_mm_setmin_epi16(); }

    template<> inline Vc::SSE::Vector<unsigned int> numeric_limits<Vc::SSE::Vector<unsigned int> >::max() throw() { return Vc::SSE::_mm_setallone_si128(); }
    template<> inline Vc::SSE::Vector<unsigned int> numeric_limits<Vc::SSE::Vector<unsigned int> >::min() throw() { return _mm_setzero_si128(); }
    template<> inline Vc::SSE::Vector<int> numeric_limits<Vc::SSE::Vector<int> >::max() throw() { return _mm_srli_epi32(Vc::SSE::_mm_setallone_si128(), 1); }
    template<> inline Vc::SSE::Vector<int> numeric_limits<Vc::SSE::Vector<int> >::min() throw() { return Vc::SSE::_mm_setmin_epi32(); }

    template<> inline Vc::SSE::Vector<float> numeric_limits<Vc::SSE::Vector<float> >::max() throw() { return _mm_set1_ps(numeric_limits<float>::max()); }
    template<> inline Vc::SSE::Vector<float> numeric_limits<Vc::SSE::Vector<float> >::min() throw() { return _mm_set1_ps(numeric_limits<float>::min()); }

    template<> inline Vc::SSE::Vector<Vc::SSE::float8> numeric_limits<Vc::SSE::Vector<Vc::SSE::float8> >::max() throw() { return numeric_limits<float>::max(); }
    template<> inline Vc::SSE::Vector<Vc::SSE::float8> numeric_limits<Vc::SSE::Vector<Vc::SSE::float8> >::min() throw() { return numeric_limits<float>::min(); }

    template<> inline Vc::SSE::Vector<double> numeric_limits<Vc::SSE::Vector<double> >::max() throw() { return _mm_set1_pd(numeric_limits<double>::max()); }
    template<> inline Vc::SSE::Vector<double> numeric_limits<Vc::SSE::Vector<double> >::min() throw() { return _mm_set1_pd(numeric_limits<double>::min()); }

} // namespace std

#endif // VC_SSE_LIMITS_H
