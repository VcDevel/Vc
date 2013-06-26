/*  This file is part of the Vc library. {{{

    Copyright (C) 2010-2012 Matthias Kretz <kretz@kde.org>

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

}}}*/

Vc_NAMESPACE_BEGIN(Vc_IMPL_NAMESPACE)
// zero, one {{{1
template<> inline __m512  VectorHelper<__m512 >::zero() { return _mm512_setzero_ps(); }
template<> inline __m512d VectorHelper<__m512d>::zero() { return _mm512_setzero_pd(); }
template<> inline __m512i VectorHelper<__m512i>::zero() { return _mm512_setzero_epi32(); }

template<> inline __m512  VectorHelper<__m512 >::one() { return _mm512_set1_ps(1.f); }
template<> inline __m512d VectorHelper<__m512d>::one() { return _mm512_set1_pd(1.); }
template<> inline __m512i VectorHelper<__m512i>::one() { return _mm512_set1_epi32(1); }

//}}}1
Vc_NAMESPACE_END

// vim: foldmethod=marker
