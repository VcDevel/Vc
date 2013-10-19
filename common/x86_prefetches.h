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
template<size_t L1, size_t L2, bool UseExclusivePrefetch> Vc_INTRINSIC void handlePrefetch(const void *addr_, typename std::enable_if<L1 != 0 && L2 != 0, void *>::type = nullptr)
{
    const char *addr = static_cast<const char *>(addr_);
    prefetchClose<typename std::conditional<UseExclusivePrefetch, Vc::Exclusive, Vc::Shared>::type>(addr + L1);
    prefetchMid  <typename std::conditional<UseExclusivePrefetch, Vc::Exclusive, Vc::Shared>::type>(addr + L2);
}
template<size_t L1, size_t L2, bool UseExclusivePrefetch> Vc_INTRINSIC void handlePrefetch(const void *addr_, typename std::enable_if<L1 == 0 && L2 != 0, void *>::type = nullptr)
{
    const char *addr = static_cast<const char *>(addr_);
    prefetchMid  <typename std::conditional<UseExclusivePrefetch, Vc::Exclusive, Vc::Shared>::type>(addr + L2);
}
template<size_t L1, size_t L2, bool UseExclusivePrefetch> Vc_INTRINSIC void handlePrefetch(const void *addr_, typename std::enable_if<L1 != 0 && L2 == 0, void *>::type = nullptr)
{
    const char *addr = static_cast<const char *>(addr_);
    prefetchClose<typename std::conditional<UseExclusivePrefetch, Vc::Exclusive, Vc::Shared>::type>(addr + L1);
}
template<size_t L1, size_t L2, bool UseExclusivePrefetch> Vc_INTRINSIC void handlePrefetch(const void *, typename std::enable_if<L1 == 0 && L2 == 0, void *>::type = nullptr)
{
}

template<typename Flags> Vc_INTRINSIC void handleLoadPrefetches(const void *    , Flags, typename Flags::EnableIfNotPrefetch = nullptr) {}
template<typename Flags> Vc_INTRINSIC void handleLoadPrefetches(const void *addr, Flags, typename Flags::EnableIfPrefetch    = nullptr)
{
    // load prefetches default to Shared unless Exclusive was explicitely selected
    handlePrefetch<Flags::L1Stride, Flags::L2Stride, Flags::IsExclusivePrefetch>(addr);
}

template<typename Flags> Vc_INTRINSIC void handleStorePrefetches(const void *    , Flags, typename Flags::EnableIfNotPrefetch = nullptr) {}
template<typename Flags> Vc_INTRINSIC void handleStorePrefetches(const void *addr, Flags, typename Flags::EnableIfPrefetch    = nullptr)
{
    // store prefetches default to Exclusive unless Shared was explicitely selected
    handlePrefetch<Flags::L1Stride, Flags::L2Stride, !Flags::IsSharedPrefetch>(addr);
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
