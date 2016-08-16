/*  This file is part of the Vc library. {{{
Copyright © 2016 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_DATAPAR_DATAPAR_H_
#define VC_DATAPAR_DATAPAR_H_

namespace Vc_VERSIONED_NAMESPACE
{
namespace detail
{
template <class T, class Abi> struct traits;
}  // namespace detail

template <class T, class Abi> class datapar
{
    using traits = detail::traits<T, Abi>;
    using impl = typename traits::datapar_impl_type;
    static constexpr std::integral_constant<size_t, traits::size()> size_tag = {};
    static constexpr T *type_tag = nullptr;
    friend impl;

public:
    using value_type = T;
    using reference = detail::smart_reference<datapar, impl>;
    using mask_type = mask<T, Abi>;
    using size_type = size_t;
    using abi_type = Abi;

    static constexpr size_type size() { return traits::size(); }
    datapar() = default;
    datapar(const datapar &) = default;
    datapar(datapar &&) = default;
    datapar &operator=(const datapar &) = default;
    datapar &operator=(datapar &&) = default;

    // implicit broadcast constructor
    datapar(value_type x) : d{impl::broadcast(x)} {}

    // scalar access
    reference operator[](size_type i) { return {*this, int(i)}; }
    value_type operator[](size_type i) const { return impl::get(*this, int(i)); }

    // negation
    mask_type operator!() const
    {
        return {detail::private_init, impl::negate(d, type_tag)};
    }

    // access to internal representation (suggested extension)
    explicit operator typename traits::datapar_cast_type() const { return d; }
    explicit datapar(const typename traits::datapar_cast_type &init) : d{init} {}

private:
    datapar(detail::private_init_t, const typename traits::datapar_member_type &init) : d{init} {}
    alignas(traits::datapar_member_alignment) typename traits::datapar_member_type d = {};
};
}  // namespace Vc_VERSIONED_NAMESPACE

#endif  // VC_DATAPAR_DATAPAR_H_

// vim: foldmethod=marker
