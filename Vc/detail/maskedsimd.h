/*  This file is part of the Vc library. {{{
Copyright © 2017 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_DETAIL_MASKEDSIMD_H_
#define VC_DETAIL_MASKEDSIMD_H_

Vc_VERSIONED_NAMESPACE_BEGIN
#ifdef Vc_EXPERIMENTAL
namespace detail {

template <class T> struct is_masked_simd : public std::false_type {};
template <class T, class A> class masked_simd_impl;
template <class T, class A>
struct is_masked_simd<masked_simd_impl<T, A>> : public std::true_type {
};

template <class T, class A> class masked_simd_impl {
public:
    using value_type = T;
    using abi_type = A;
    using simd_type = simd<T, A>;
    using mask_type = mask<T, A>;

    // C++17: use 'simd<T, A>' to enable deduction
    masked_simd_impl(const mask_type &kk, simd<T, A> &vv) : k(kk), v(vv) {}
    masked_simd_impl &operator=(const masked_simd_impl &rhs)
    {
        Vc::detail::get_impl_t<simd_type>::masked_assign(
            Vc::detail::data(k), Vc::detail::data(v), Vc::detail::data(rhs.v));
        return *this;
    }
    template <class U>
    std::enable_if_t<!is_masked_simd<std::decay_t<U>>::value, masked_simd_impl &>
    operator=(U &&rhs)
    {
        Vc::detail::get_impl_t<simd_type>::masked_assign(
            Vc::detail::data(k), Vc::detail::data(v),
            detail::to_value_type_or_member_type<simd_type>(std::forward<U>(rhs)));
        return *this;
    }

private:
    const mask_type &k;
    simd_type &v;
};

template <class T, class A>
masked_simd_impl<T, A> masked_simd(const typename simd<T, A>::mask_type &k,
                                         simd<T, A> &v)
{
    return {k, v};
}

}  // namespace detail

/*
template <class T, class A, class OnTrue, class OnFalse, class... Vs>
// TODO: require mask<T, A> to be convertible to Vs::mask_type forall Vs
std::enable_if_t<
detail::all<std::is_same<decltype(declval<OnTrue>()(detail::masked_simd(
                                 declval<mask<T, A> &>(), declval<Vs>())...)),
                             void>,
                std::is_same<decltype(declval<OnFalse>()(detail::masked_simd(
                                 declval<mask<T, A> &>(), declval<Vs>())...)),
                             void>>::value,
    void>
where(mask<T, A> k, OnTrue &&on_true, OnFalse &&on_false, Vs &&... vs)
{
    std::forward<OnTrue>(on_true)(detail::masked_simd(k, std::forward<Vs>(vs))...);
    std::forward<OnFalse>(on_false)(detail::masked_simd(!k, std::forward<Vs>(vs))...);
}

template <class T, class A, class OnTrue, class... Vs>
// TODO: require mask<T, A> to be convertible to Vs::mask_type forall Vs
std::enable_if_t<
detail::all<std::is_same<decltype(declval<OnTrue>()(detail::masked_simd(
                                 declval<mask<T, A> &>(), declval<Vs>())...)),
                             void>>::value,
    void>
where(mask<T, A> k, OnTrue &&on_true, Vs &&... vs)
{
    std::forward<OnTrue>(on_true)(detail::masked_simd(k, std::forward<Vs>(vs))...);
}
*/

#endif  // Vc_EXPERIMENTAL
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_DETAIL_MASKEDSIMD_H_
