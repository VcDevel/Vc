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

#ifndef VC_SSE_HELPERIMPL_TCC
#define VC_SSE_HELPERIMPL_TCC

namespace Vc
{
namespace Internal
{

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
inline void *HelperImpl<LRBniImpl>::malloc(size_t n)
{
    void *ptr = 0;
    switch (A) {
        case Vc::AlignOnVector:
        case Vc::AlignOnCacheline:
            // TODO: hardcoding 64 is not such a great idea
            if (0 == posix_memalign(&ptr, 64, nextMultipleOf<64>(n))) {
                return ptr;
            }
            break;
        case Vc::AlignOnPage:
            // TODO: hardcoding 4096 is not such a great idea
            if (0 == posix_memalign(&ptr, 4096, nextMultipleOf<4096>(n))) {
                return ptr;
            }
    }
    return 0;
}

inline void HelperImpl<LRBniImpl>::free(void *p)
{
    _mm_free(p);
}

} // namespace Internal
} // namespace Vc

#endif // VC_SSE_HELPERIMPL_TCC
