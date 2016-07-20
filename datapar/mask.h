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

#ifndef VC_DATAPAR_MASK_H_
#define VC_DATAPAR_MASK_H_

#include "smart_reference.h"

namespace Vc::v2
{
template <class T, class Abi> class mask
{
    using traits = detail::traits<T, Abi>;
    using impl = typename traits::mask_impl_type;
    static constexpr T *tag = nullptr;
    friend impl;

public:
    using value_type = bool;
    using reference = detail::smart_reference<mask, impl>;
    using datapar_type = datapar<T, Abi>;
    using size_type = size_t;
    using abi_type = Abi;

    static constexpr size_type size() { return traits::size(); }
    mask() = default;
    mask(const mask &) = default;
    mask(mask &&) = default;
    mask &operator=(const mask &) = default;
    mask &operator=(mask &&) = default;

    // implicit broadcast constructor
    mask(value_type x) : d{impl::broadcast(x, tag)} {}

    // implicit type conversion constructor
    template <class U, class Abi2> mask(mask<U, Abi2> x) : d{impl::convert(x, tag)} {}

    // load constructor
    template <class Flags>
    mask(const value_type *mem, Flags flags)
        : d{impl::load(mem, flags, tag)}
    {
    }

    // scalar access
    reference operator[](size_type i) { return {*this, int(i)}; }
    value_type operator[](size_type i) const { return impl::get(*this, int(i)); }

    // negation
    mask operator!() const { return {nullptr, impl::negate(d, tag)}; }

    // access to internal representation (suggested extension)
    explicit operator typename traits::mask_cast_type() const { return d; }
    explicit mask(const typename traits::mask_cast_type &init) : d{init} {}

private:
    mask(nullptr_t, const typename traits::mask_member_type &init) : d{init} {}
    alignas(traits::mask_member_alignment) typename traits::mask_member_type d = {};
};
}  // namespace Vc::v2
#endif  // VC_DATAPAR_MASK_H_

// vim: foldmethod=marker
