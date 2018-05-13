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
namespace detail
{
struct scalar_abi;
template <int N> struct fixed_abi;
template <int Bytes> struct sse_abi;
template <int Bytes> struct avx_abi;
template <int Bytes> struct avx512_abi;
template <int Bytes> struct neon_abi;
}  // namespace detail

namespace simd_abi
{
template <class T, std::size_t N, class... > struct deduce;
template <int N> using fixed_size = Vc::detail::fixed_abi<N>;
using scalar = Vc::detail::scalar_abi;

#ifdef __INTEL_COMPILER
#define Vc_DEPRECATED_
#else
#define Vc_DEPRECATED_ [[deprecated("Capitalize the first letter of non-std ABI tags")]]
#endif

using sse Vc_DEPRECATED_ = Vc::detail::sse_abi<16>;
using avx Vc_DEPRECATED_ = Vc::detail::avx_abi<32>;
using avx512 Vc_DEPRECATED_ = Vc::detail::avx512_abi<64>;
using neon Vc_DEPRECATED_ = Vc::detail::neon_abi<16>;

#undef Vc_DEPRECATED_

using __sse = Vc::detail::sse_abi<16>;
using __avx = Vc::detail::avx_abi<32>;
using __avx512 = Vc::detail::avx512_abi<64>;
using __neon128 = Vc::detail::neon_abi<16>;
using __neon64 = Vc::detail::neon_abi<8>;
using __neon = __neon128;
}

template <class T> struct is_simd;
template <class T> struct is_simd_mask;
template <class T, class Abi> class simd;
template <class T, class Abi> class simd_mask;

template <class T, class Abi> struct simd_size;

Vc_VERSIONED_NAMESPACE_END

#endif  // VC_FWDDECL_H_
