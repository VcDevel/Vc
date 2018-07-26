/*  This file is part of the Vc library. {{{
Copyright © 2017-2018 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_FWDDECL_H_
#define VC_FWDDECL_H_

#include <cstddef>

namespace Vc
{
inline namespace v2
{
}  // namespace v2
}  // namespace Vc
#define Vc_VERSIONED_NAMESPACE Vc::v2
#define Vc_VERSIONED_NAMESPACE_BEGIN namespace Vc { inline namespace v2 {
#define Vc_VERSIONED_NAMESPACE_END }}

Vc_VERSIONED_NAMESPACE_BEGIN
namespace simd_abi
{
// implementation details:
struct __scalar_abi;
template <int N> struct __fixed_abi;

template <int Bytes = 16> struct __sse_abi;
template <int Bytes = 32> struct __avx_abi;
template <int Bytes = 64> struct __avx512_abi;
template <int Bytes = 16> struct __neon_abi;

template <int N, class Abi> struct __combine;

// implementation-defined:
template <int NRegisters> using __sse_x = __combine<NRegisters, __sse_abi<>>;
template <int NRegisters> using __avx_x = __combine<NRegisters, __avx_abi<>>;
template <int NRegisters> using __avx512_x = __combine<NRegisters, __avx512_abi<>>;
template <int NRegisters> using __neon_x = __combine<NRegisters, __neon_abi<>>;

template <class T, int N> using __sse_n = __sse_abi<sizeof(T) * N>;
template <class T, int N> using __avx_n = __avx_abi<sizeof(T) * N>;
template <class T, int N> using __avx512_n = __avx512_abi<sizeof(T) * N>;
template <class T, int N> using __neon_n = __neon_abi<sizeof(T) * N>;

using __sse = __sse_abi<>;
using __avx = __avx_abi<>;
using __avx512 = __avx512_abi<>;
using __neon = __neon_abi<>;

using __neon128 = __neon_abi<16>;
using __neon64 = __neon_abi<8>;

// standard:
template <class T, std::size_t N, class... > struct deduce;
template <int N> using fixed_size = __fixed_abi<N>;
using scalar = __scalar_abi;
}

template <class T> struct is_simd;
template <class T> struct is_simd_mask;
template <class T, class Abi> class simd;
template <class T, class Abi> class simd_mask;

template <class T, class Abi> struct simd_size;

Vc_VERSIONED_NAMESPACE_END

#endif  // VC_FWDDECL_H_
