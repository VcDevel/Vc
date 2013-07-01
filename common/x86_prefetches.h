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

#ifndef VC_COMMON_X86_PREFETCHES_H
#define VC_COMMON_X86_PREFETCHES_H

#include <xmmintrin.h>
#include "macros.h"

Vc_NAMESPACE_BEGIN(Vc_IMPL_NAMESPACE)

#if !defined(VC_IMPL_MIC) && !defined(_MM_HINT_ENTA)
#define VC__NO_SUPPORT_FOR_EXCLUSIVE_HINT 1
#define _MM_HINT_ENTA _MM_HINT_NTA
#define _MM_HINT_ET0 _MM_HINT_T0
#define _MM_HINT_ET1 _MM_HINT_T1
#define _MM_HINT_ET2 _MM_HINT_T2
#endif

// TODO: support AMD's prefetchw with correct flags and checks via cpuid

template<typename ExclusiveOrShared = Vc::Shared>
Vc_INTRINSIC void prefetchForOneRead(const void *addr)
{
    if (std::is_same<ExclusiveOrShared, Vc::Shared>::value) {
        _mm_prefetch(static_cast<char *>(const_cast<void *>(addr)), _MM_HINT_NTA);
    } else {
        _mm_prefetch(static_cast<char *>(const_cast<void *>(addr)), _MM_HINT_ENTA);
    }
}
template<typename ExclusiveOrShared = Vc::Shared>
Vc_INTRINSIC void prefetchClose(const void *addr)
{
    if (std::is_same<ExclusiveOrShared, Vc::Shared>::value) {
        _mm_prefetch(static_cast<char *>(const_cast<void *>(addr)), _MM_HINT_T0);
    } else {
        _mm_prefetch(static_cast<char *>(const_cast<void *>(addr)), _MM_HINT_ET0);
    }
}
template<typename ExclusiveOrShared = Vc::Shared>
Vc_INTRINSIC void prefetchMid(const void *addr)
{
    if (std::is_same<ExclusiveOrShared, Vc::Shared>::value) {
        _mm_prefetch(static_cast<char *>(const_cast<void *>(addr)), _MM_HINT_T1);
    } else {
        _mm_prefetch(static_cast<char *>(const_cast<void *>(addr)), _MM_HINT_ET1);
    }
}
template<typename ExclusiveOrShared = Vc::Shared>
Vc_INTRINSIC void prefetchFar(const void *addr)
{
    if (std::is_same<ExclusiveOrShared, Vc::Shared>::value) {
        _mm_prefetch(static_cast<char *>(const_cast<void *>(addr)), _MM_HINT_T2);
    } else {
        _mm_prefetch(static_cast<char *>(const_cast<void *>(addr)), _MM_HINT_ET2);
    }
}

#ifdef VC__NO_SUPPORT_FOR_EXCLUSIVE_HINT
#undef VC__NO_SUPPORT_FOR_EXCLUSIVE_HINT
#undef _MM_HINT_ENTA
#undef _MM_HINT_ET0
#undef _MM_HINT_ET1
#undef _MM_HINT_ET2
#endif

/*handlePrefetch/handleLoadPrefetches/handleStorePrefetches{{{*/
namespace
{
template<int L1, int L2> Vc_INTRINSIC void handlePrefetch(const void *addr_, Vc::PrefetchFlag<L1, L2, Shared>)
{
    const char *addr = static_cast<const char *>(addr_);
    prefetchClose(addr + L1);
    prefetchMid  (addr + L2);
}
template<int L1> Vc_INTRINSIC void handlePrefetch(const void *addr_, Vc::PrefetchFlag<L1, 0, Shared>)
{
    const char *addr = static_cast<const char *>(addr_);
    prefetchClose(addr + L1);
}
template<int L2> Vc_INTRINSIC void handlePrefetch(const void *addr_, Vc::PrefetchFlag<0, L2, Shared>)
{
    const char *addr = static_cast<const char *>(addr_);
    prefetchMid  (addr + L2);
}
template<int L1, int L2> Vc_INTRINSIC void handlePrefetch(const void *addr_, Vc::PrefetchFlag<L1, L2, Exclusive>)
{
    const char *addr = static_cast<const char *>(addr_);
    prefetchClose<Vc::Exclusive>(addr + L1);
    prefetchMid  <Vc::Exclusive>(addr + L2);
}
template<int L1> Vc_INTRINSIC void handlePrefetch(const void *addr_, Vc::PrefetchFlag<L1, 0, Exclusive>)
{
    const char *addr = static_cast<const char *>(addr_);
    prefetchClose<Vc::Exclusive>(addr + L1);
}
template<int L2> Vc_INTRINSIC void handlePrefetch(const void *addr_, Vc::PrefetchFlag<0, L2, Exclusive>)
{
    const char *addr = static_cast<const char *>(addr_);
    prefetchMid  <Vc::Exclusive>(addr + L2);
}
Vc_INTRINSIC void handleLoadPrefetches(const void *) {}
template<int L1, int L2, typename... Flags> Vc_INTRINSIC void handleLoadPrefetches(const void *addr, Vc::PrefetchFlag<L1, L2, void>, Flags... flags)
{
    handlePrefetch(addr, Vc::PrefetchFlag<L1, L2, Shared>());
    handleLoadPrefetches(addr, flags...);
}
template<int L1, int L2, typename SharedOrExclusive, typename... Flags> Vc_INTRINSIC void handleLoadPrefetches(const void *addr, Vc::PrefetchFlag<L1, L2, SharedOrExclusive>, Flags... flags)
{
    handlePrefetch(addr, Vc::PrefetchFlag<L1, L2, SharedOrExclusive>());
    handleLoadPrefetches(addr, flags...);
}
template<typename F, typename... Flags> Vc_INTRINSIC void handleLoadPrefetches(const void *addr, F otherFlag, Flags... flags)
{
    handleLoadPrefetches(addr, flags...);
}
Vc_INTRINSIC void handleStorePrefetches(const void *) {}
template<int L1, int L2, typename... Flags> Vc_INTRINSIC void handleStorePrefetches(const void *addr, Vc::PrefetchFlag<L1, L2, void>, Flags... flags)
{
    handlePrefetch(addr, Vc::PrefetchFlag<L1, L2, Exclusive>());
    handleStorePrefetches(addr, flags...);
}
template<int L1, int L2, typename SharedOrExclusive, typename... Flags> Vc_INTRINSIC void handleStorePrefetches(const void *addr, Vc::PrefetchFlag<L1, L2, SharedOrExclusive>, Flags... flags)
{
    handlePrefetch(addr, Vc::PrefetchFlag<L1, L2, SharedOrExclusive>());
    handleStorePrefetches(addr, flags...);
}
template<typename F, typename... Flags> Vc_INTRINSIC void handleStorePrefetches(const void *addr, F otherFlag, Flags... flags)
{
    handleStorePrefetches(addr, flags...);
}
} // anonymous namespace
/*}}}*/

Vc_NAMESPACE_END

Vc_PUBLIC_NAMESPACE_BEGIN
using Vc_IMPL_NAMESPACE::prefetchForOneRead;
using Vc_IMPL_NAMESPACE::prefetchClose;
using Vc_IMPL_NAMESPACE::prefetchMid;
using Vc_IMPL_NAMESPACE::prefetchFar;
Vc_NAMESPACE_END

#include "undomacros.h"

#endif // VC_COMMON_X86_PREFETCHES_H

// vim: foldmethod=marker
