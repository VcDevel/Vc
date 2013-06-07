/*  This file is part of the Vc library.

    Copyright (C) 2010 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_MIC_HELPERIMPL_TCC
#define VC_MIC_HELPERIMPL_TCC

#include "types.h"

Vc_NAMESPACE_BEGIN(Internal)

template<size_t X>
static inline size_t nextMultipleOf(size_t value)
{
    const size_t offset = value % X;
    if ( offset > 0 ) {
        return value + X - offset;
    }
    return value;
}

template<Vc::MallocAlignment A>
inline void *HelperImpl<MICImpl>::malloc(size_t n)
{
    void *ptr = 0;
    switch (A) {
        case Vc::AlignOnVector:
        case Vc::AlignOnCacheline:
            return _mm_malloc(nextMultipleOf<MIC::VectorAlignment>(n), MIC::VectorAlignment);
        case Vc::AlignOnPage:
            // TODO: hardcoding 4096 is not such a great idea
            return _mm_malloc(nextMultipleOf<4096>(n), 4096);
    }
    return 0;
}

inline void HelperImpl<MICImpl>::free(void *p)
{
    _mm_free(p);
}

Vc_NAMESPACE_END

#endif // VC_MIC_HELPERIMPL_TCC
