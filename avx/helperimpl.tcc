/*  This file is part of the Vc library.

    Copyright (C) 2011-2013 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_AVX_HELPERIMPL_TCC
#define VC_AVX_HELPERIMPL_TCC

#include "../common/malloc.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace Internal
{

template<Vc::MallocAlignment A>
inline void *HelperImpl<VC_IMPL>::malloc(size_t n)
{
    return Common::malloc<A>(n);
}

inline void HelperImpl<VC_IMPL>::free(void *p)
{
    Common::free(p);
}

}
}

#endif // VC_AVX_HELPERIMPL_TCC
