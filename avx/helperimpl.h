/*  This file is part of the Vc library.

    Copyright (C) 2011 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_AVX_HELPERIMPL_H
#define VC_AVX_HELPERIMPL_H

#include "macros.h"

namespace Vc
{
namespace Internal
{

template<> struct HelperImpl<Vc::AVXImpl>
{
    typedef AVX::Vector<float> float_v;
    typedef AVX::Vector<double> double_v;
    typedef AVX::Vector<int> int_v;
    typedef AVX::Vector<unsigned int> uint_v;
    typedef AVX::Vector<short> short_v;
    typedef AVX::Vector<unsigned short> ushort_v;

    // TODO: deinterleave

    static inline void prefetchForOneRead(const void *addr) ALWAYS_INLINE;
    static inline void prefetchForModify(const void *addr) ALWAYS_INLINE;
    static inline void prefetchClose(const void *addr) ALWAYS_INLINE;
    static inline void prefetchMid(const void *addr) ALWAYS_INLINE;
    static inline void prefetchFar(const void *addr) ALWAYS_INLINE;

    template<Vc::MallocAlignment A>
    static inline void *malloc(size_t n) ALWAYS_INLINE;
    static inline void free(void *p) ALWAYS_INLINE;
};
} // namespace Internal
} // namespace Vc
#include "undomacros.h"

#endif // VC_AVX_HELPERIMPL_H
