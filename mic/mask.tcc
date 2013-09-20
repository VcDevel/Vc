/*  This file is part of the Vc library. {{{

    Copyright (C) 2013 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_MIC_MASK_TCC
#define VC_MIC_MASK_TCC

#include "macros.h"

Vc_NAMESPACE_BEGIN(Vc_IMPL_NAMESPACE)

template<> inline void Mask<double>::store(bool *mem) const
{
    const __m512i zero = _mm512_setzero_epi32();
    const __m512i one = _mm512_set1_epi32(1);
    const __m512i tmp = _and(zero, static_cast<__mmask16>(k), one, one);
    _mm512_mask_extstore_epi32(mem, 0xff, tmp, UpDownConversion<unsigned int, unsigned char>(), _MM_HINT_NONE);
}

Vc_NAMESPACE_END

#include "undomacros.h"

#endif // VC_MIC_MASK_TCC
