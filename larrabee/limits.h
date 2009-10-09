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

#ifndef VC_LARRABEE_LIMITS_H
#define VC_LARRABEE_LIMITS_H

#include <limits>
#include "intrinsics.h"

namespace std
{
#define SPECIALIZATION(T) template<> Vc::LRBni::Vector<T> numeric_limits<Vc::LRBni::Vector<T> >
    SPECIALIZATION(unsigned int)::max() throw() { return Vc::LRBni::_mm512_setallone_pi(); }
    SPECIALIZATION(unsigned int)::min() throw() { return _mm512_setzero_pi(); }
    SPECIALIZATION(int)::max() throw() { return _mm512_set_1to16_pi(std::numeric_limits<int>::max()); }
    SPECIALIZATION(int)::min() throw() { return _mm512_set_1to16_pi(std::numeric_limits<int>::min()); }

    SPECIALIZATION(float)::max() throw() { return _mm512_set_1to16_ps(std::numeric_limits<float>::max()); }
    SPECIALIZATION(float)::min() throw() { return _mm512_set_1to16_ps(std::numeric_limits<float>::min()); }

    SPECIALIZATION(double)::max() throw() { return _mm512_set_1to8_pd(std::numeric_limits<double>::max()); }
    SPECIALIZATION(double)::min() throw() { return _mm512_set_1to8_pd(std::numeric_limits<double>::min()); }
} // namespace std
#endif // VC_LARRABEE_LIMITS_H
