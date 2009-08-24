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

namespace std
{

    template<> SSE::Vector<unsigned short> numeric_limits<SSE::Vector<unsigned short> >::max() { return _mm_setallone_si128(); }
    template<> SSE::Vector<unsigned short> numeric_limits<SSE::Vector<unsigned short> >::min() { return _mm_setzero_si128(); }
    template<> SSE::Vector<short> numeric_limits<SSE::Vector<short> >::max() { return _mm_srli_epi16(_mm_setallone_si128(), 1); }
    template<> SSE::Vector<short> numeric_limits<SSE::Vector<short> >::min() { return _mm_setmin_epi16(); }

    template<> SSE::Vector<unsigned int> numeric_limits<SSE::Vector<unsigned int> >::max() { return _mm_setallone_si128(); }
    template<> SSE::Vector<unsigned int> numeric_limits<SSE::Vector<unsigned int> >::min() { return _mm_setzero_si128(); }
    template<> SSE::Vector<int> numeric_limits<SSE::Vector<int> >::max() { return _mm_srli_epi32(_mm_setallone_si128(), 1); }
    template<> SSE::Vector<int> numeric_limits<SSE::Vector<int> >::min() { return _mm_setmin_epi32(); }

    template<> SSE::Vector<float> numeric_limits<SSE::Vector<float> >::max() { return _mm_set1_ps(numeric_limits<float>::max()); }
    template<> SSE::Vector<float> numeric_limits<SSE::Vector<float> >::min() { return _mm_set1_ps(numeric_limits<float>::min()); }

    template<> SSE::Vector<float8> numeric_limits<SSE::Vector<float8> >::max() { return _mm_set1_ps(numeric_limits<float>::max()); }
    template<> SSE::Vector<float8> numeric_limits<SSE::Vector<float8> >::min() { return _mm_set1_ps(numeric_limits<float>::min()); }

    template<> SSE::Vector<double> numeric_limits<SSE::Vector<double> >::max() { return _mm_set1_pd(numeric_limits<double>::max()); }
    template<> SSE::Vector<double> numeric_limits<SSE::Vector<double> >::min() { return _mm_set1_pd(numeric_limits<double>::min()); }

} // namespace std

#endif // VC_SSE_LIMITS_H
