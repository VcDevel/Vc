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
    datapar(value_type x) : d{impl::broadcast(x, size_tag)} {}

    // load constructor
    template <class U, class Flags>
    datapar(const U *mem, Flags f)
        : d{impl::load(mem, f, type_tag)}
    {
    }
    template <class U, class Flags> datapar(const U *mem, mask_type k, Flags f) : d{}
    {
        impl::masked_load(d, k, mem, f);
    }

    // loads [datapar.load]
    template <class U, class Flags> void copy_from(const U *mem, Flags f)
    {
        d = static_cast<decltype(d)>(impl::load(mem, f, type_tag));
    }
    template <class U, class Flags> void copy_from(const U *mem, mask_type k, Flags f)
    {
        impl::masked_load(d, k, mem, f);
    }

    // stores [datapar.store]
    template <class U, class Flags> void copy_to(U *mem, Flags f) const
    {
        impl::store(d, mem, f, type_tag);
    }
    template <class U, class Flags> void copy_to(U *mem, mask_type k, Flags f) const
    {
        impl::masked_store(d, mem, f, k);
    }

    // scalar access
    reference operator[](size_type i) { return {*this, int(i)}; }
    value_type operator[](size_type i) const { return impl::get(*this, int(i)); }

    // increment and decrement:
    //datapar &operator++();
    //datapar operator++(int);
    //datapar &operator--();
    //datapar operator--(int);

    // unary operators (for integral T)
    mask_type operator!() const { return impl::negate(*this); }
    //datapar operator~() const;

    // unary operators (for any T)
    datapar operator+() const { return *this; }
    //datapar operator-() const;

    // reductions
    //value_type sum() const;
    //value_type sum(mask_type) const;
    //value_type product() const;
    //value_type product(mask_type) const;
    //value_type min() const;
    //value_type min(mask_type) const;
    //value_type max() const;
    //value_type max(mask_type) const;

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
