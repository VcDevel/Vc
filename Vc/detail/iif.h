/*  This file is part of the Vc library. {{{
Copyright © 2018 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_DETAIL_IIF_H_
#define VC_DETAIL_IIF_H_

Vc_VERSIONED_NAMESPACE_BEGIN
namespace __proposed
{
template <class MT, class MA, class T0, class A0, class T1, class A1,
          class RT = std::common_type<T0, T1>,
          class RA = Vc::simd_abi::deduce_t<RT, Vc::simd_size_v<T0, A0>, A0, A1>>
std::enable_if_t<
    Vc::detail::all<std::is_convertible<Vc::simd_mask<MT, MA>, Vc::simd_mask<RT, RA>>,
                    std::is_convertible<Vc::simd<T0, A0>, Vc::simd<RT, RA>>,
                    std::is_convertible<Vc::simd<T1, A1>, Vc::simd<RT, RA>>>::value,
    Vc::simd<RT, RA>>
iif(const Vc::simd_mask<MT, MA> &mask, const Vc::simd<T0, A0> &a,
    const Vc::simd<T1, A1> &b)
{
    Vc::simd<RT, RA> r = b;
    where(mask, r) = a;
    return r;
}

template <class MT, class MA, class T0, class T1, class A1>
std::enable_if_t<
    Vc::detail::all<std::negation<Vc::is_simd<T0>>,
                    std::is_convertible<Vc::simd_mask<MT, MA>, Vc::simd_mask<T1, A1>>,
                    std::is_convertible<T0, Vc::simd<T1, A1>>>::value,
    Vc::simd<T1, A1>>
iif(const Vc::simd_mask<MT, MA> &mask, const T0 &a, const Vc::simd<T1, A1> &b)
{
    Vc::simd<T1, A1> r = b;
    where(mask, r) = a;
    return r;
}

template <class MT, class MA, class T0, class A0, class T1>
std::enable_if_t<
    Vc::detail::all<std::negation<Vc::is_simd<T1>>,
                    std::is_convertible<Vc::simd_mask<MT, MA>, Vc::simd_mask<T0, A0>>,
                    std::is_convertible<T1, Vc::simd<T0, A0>>>::value,
    Vc::simd<T0, A0>>
iif(const Vc::simd_mask<MT, MA> &mask, const Vc::simd<T0, A0> &a, const T1 &b)
{
    Vc::simd<T0, A0> r = b;
    where(mask, r) = a;
    return r;
}

template <class MT, class MA, class T0, class A0, class T1, class A1,
          class RT = std::common_type<T0, T1>,
          class RA = Vc::simd_abi::deduce_t<RT, Vc::simd_size_v<T0, A0>, A0, A1>>
std::enable_if_t<
    Vc::detail::all<
        std::is_convertible<Vc::simd_mask<MT, MA>, Vc::simd_mask<RT, RA>>,
        std::is_convertible<Vc::simd_mask<T0, A0>, Vc::simd_mask<RT, RA>>,
        std::is_convertible<Vc::simd_mask<T1, A1>, Vc::simd_mask<RT, RA>>>::value,
    Vc::simd_mask<RT, RA>>
iif(const Vc::simd_mask<MT, MA> &mask, const Vc::simd_mask<T0, A0> &a,
    const Vc::simd_mask<T1, A1> &b)
{
    Vc::simd_mask<RT, RA> r = b;
    where(mask, r) = a;
    return r;
}

template <class MT, class MA, class T0, class T1, class A1>
std::enable_if_t<
    Vc::detail::all<std::negation<Vc::is_simd_mask<T0>>,
                    std::is_convertible<Vc::simd_mask<MT, MA>, Vc::simd_mask<T1, A1>>,
                    std::is_convertible<T0, Vc::simd_mask<T1, A1>>>::value,
    Vc::simd_mask<T1, A1>>
iif(const Vc::simd_mask<MT, MA> &mask, const T0 &a, const Vc::simd_mask<T1, A1> &b)
{
    Vc::simd_mask<T1, A1> r = b;
    where(mask, r) = a;
    return r;
}

template <class MT, class MA, class T0, class A0, class T1>
std::enable_if_t<
    Vc::detail::all<std::negation<Vc::is_simd_mask<T1>>,
                    std::is_convertible<Vc::simd_mask<MT, MA>, Vc::simd_mask<T0, A0>>,
                    std::is_convertible<T1, Vc::simd_mask<T0, A0>>>::value,
    Vc::simd_mask<T0, A0>>
iif(const Vc::simd_mask<MT, MA> &mask, const Vc::simd_mask<T0, A0> &a, const T1 &b)
{
    Vc::simd_mask<T0, A0> r = b;
    where(mask, r) = a;
    return r;
}

}  // namespace __proposed
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_DETAIL_IIF_H_

// vim: foldmethod=marker
