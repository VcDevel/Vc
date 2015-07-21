/*  This file is part of the Vc library. {{{
Copyright © 2014 Matthias Kretz <kretz@kde.org>
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

#ifndef VC_TRAITS_IS_FUNCTOR_ARGUMENT_IMMUTABLE_H_
#define VC_TRAITS_IS_FUNCTOR_ARGUMENT_IMMUTABLE_H_

namespace Vc_VERSIONED_NAMESPACE
{
namespace Traits
{
namespace is_functor_argument_immutable_impl
{
template <typename F, typename A,
#if defined(VC_ICC) || defined(VC_NVCC)
          // this is wrong, but then again ICC is broken - and better it compiles and
          // returns the wrong answer than not compiling at all
          typename MemberPtr = decltype(&F::operator()),
#else
          // this fails for cudafe++ (as of CUDA v7.0) as well
          typename MemberPtr = decltype(&F::template operator() <A>),
#endif
          typename foo = decltype((std::declval<F &>().*
                               (std::declval<MemberPtr>()))(std::declval<const A &>()))>
std::true_type test(int);
template <typename F, typename A> std::false_type test(...);
}  // namespace is_functor_argument_immutable_impl

// CUDA's C++11 support is broken (v7.0) - this workaround replaces the previous alias template
// declaration for is_functor_argument_immutable
template<typename F, typename A>
struct nvcc_alias_template_workaround
{
    using type = decltype(is_functor_argument_immutable_impl::test<F, A>(1));
};

template <typename F, typename A>
using is_functor_argument_immutable = typename nvcc_alias_template_workaround<F, A>::type;


}  // namespace Traits
}  // namespace Vc

#endif  // VC_TRAITS_IS_FUNCTOR_ARGUMENT_IMMUTABLE_H_

// vim: foldmethod=marker
