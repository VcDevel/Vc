/*  This file is part of the Vc library.

    Copyright (C) 2009-2010 Matthias Kretz <kretz@kde.org>

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

namespace std
{

#define NUM_LIM(type) \
    template<> inline Vc::AVX::Vector<type> numeric_limits<Vc::AVX::Vector<type> >

    NUM_LIM(unsigned short)::max() throw() { return Vc::AVX::_mm_setallone_si128(); }
    NUM_LIM(unsigned short)::min() throw() { return _mm_setzero_si128(); }
    NUM_LIM(short         )::max() throw() { return _mm_srli_epi16(Vc::AVX::_mm_setallone_si128(), 1); }
    NUM_LIM(short         )::min() throw() { return Vc::AVX::_mm_setmin_epi16(); }

    NUM_LIM(unsigned int  )::max() throw() { return Vc::AVX::_mm256_setallone_si256(); }
    NUM_LIM(unsigned int  )::min() throw() { return _mm256_setzero_si256(); }
    NUM_LIM(int           )::max() throw() { const __m128i tmp =  _mm_srli_epi32(Vc::AVX::_mm_setallone_si128(), 1); return Vc::AVX::concat(tmp, tmp); }
    NUM_LIM(int           )::min() throw() { return Vc::AVX::_mm256_setmin_epi32(); }

    NUM_LIM(float         )::max() throw() { return _mm256_set1_ps(numeric_limits<float>::max()); }
    NUM_LIM(float         )::min() throw() { return _mm256_set1_ps(numeric_limits<float>::min()); }

    NUM_LIM(double        )::max() throw() { return _mm256_set1_pd(numeric_limits<double>::max()); }
    NUM_LIM(double        )::min() throw() { return _mm256_set1_pd(numeric_limits<double>::min()); }
#undef NUM_LIM

} // namespace std

#endif // VC_AVX_LIMITS_H
