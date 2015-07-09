/*  This file is part of the Vc library. {{{
Copyright © 2010-2013 Matthias Kretz <kretz@kde.org>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the names of contributing organizations nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

}}}*/

#ifndef VC_SSE_PREFETCHES_TCC
#define VC_SSE_PREFETCHES_TCC

namespace Vc_VERSIONED_NAMESPACE
{
namespace Internal
{

Vc_ALWAYS_INLINE void HelperImpl<Vc::SSE2Impl>::prefetchForOneRead(const void *addr)
{
    _mm_prefetch(static_cast<char *>(const_cast<void *>(addr)), _MM_HINT_NTA);
}
Vc_ALWAYS_INLINE void HelperImpl<Vc::SSE2Impl>::prefetchClose(const void *addr)
{
    _mm_prefetch(static_cast<char *>(const_cast<void *>(addr)), _MM_HINT_T0);
}
Vc_ALWAYS_INLINE void HelperImpl<Vc::SSE2Impl>::prefetchMid(const void *addr)
{
    _mm_prefetch(static_cast<char *>(const_cast<void *>(addr)), _MM_HINT_T1);
}
Vc_ALWAYS_INLINE void HelperImpl<Vc::SSE2Impl>::prefetchFar(const void *addr)
{
    _mm_prefetch(static_cast<char *>(const_cast<void *>(addr)), _MM_HINT_T2);
}
Vc_ALWAYS_INLINE void HelperImpl<Vc::SSE2Impl>::prefetchForModify(const void *addr)
{
#ifdef __3dNOW__
    _m_prefetchw(const_cast<void *>(addr));
#else
    _mm_prefetch(static_cast<char *>(const_cast<void *>(addr)), _MM_HINT_T0);
#endif
}

}
}

#endif // VC_SSE_PREFETCHES_TCC
