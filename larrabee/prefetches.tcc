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

#ifndef VC_LRBni_PREFETCH_TCC
#define VC_LRBni_PREFETCH_TCC

namespace Vc
{
namespace Internal
{

// TODO: I don't know what the hints really mean
inline void HelperImpl<Vc::LRBniImpl>::prefetchForOneRead(void *addr)
{
    _mm_vprefetch1(addr, _MM_PFHINT_NT);
}
inline void HelperImpl<Vc::LRBniImpl>::prefetchForModify(void *addr)
{
    _mm_vprefetch1(addr, _MM_PFHINT_EX);
}
inline void HelperImpl<Vc::LRBniImpl>::prefetchClose(void *addr)
{
    _mm_vprefetch1(addr, _MM_PFHINT_NONE);
}
inline void HelperImpl<Vc::LRBniImpl>::prefetchMid(void *addr)
{
    _mm_vprefetch2(addr, _MM_PFHINT_NONE);
}
inline void HelperImpl<Vc::LRBniImpl>::prefetchFar(void *addr)
{
    _mm_vprefetch2(addr, _MM_PFHINT_NONE);
}

} // namespace Internal
} // namespace Vc

#endif // VC_LRBni_PREFETCH_TCC
