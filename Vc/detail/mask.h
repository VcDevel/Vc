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

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
template <class T, class A> Vc_INTRINSIC_L auto data(const mask<T, A> &x) Vc_INTRINSIC_R;
}  // namespace detail

template <class T, class Abi> class mask
{
    using traits = detail::traits<T, Abi>;
    using impl = typename traits::mask_impl_type;
    static constexpr std::integral_constant<size_t, traits::size()> size_tag = {};
    static constexpr T *type_tag = nullptr;
    friend impl;
    friend typename traits::datapar_impl_type;  // to construct masks on return and
                                                // inspect data on masked operations

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

    // non-std; required to work around ICC ICEs
    static constexpr size_type size_v = traits::size();

    // explicit broadcast constructor
    explicit mask(value_type x) : d(impl::broadcast(x, type_tag)) {}

    // implicit type conversion constructor
    template <class U>
    mask(const mask<U, datapar_abi::fixed_size<size_v>> &x,
         enable_if<conjunction<std::is_same<abi_type, datapar_abi::fixed_size<size_v>>,
                               std::is_same<U, U>>::value> = nullarg)
        : mask{static_cast<const std::array<bool, size()> &>(x).data(),
               flags::vector_aligned}
    {
    }
    /* reference implementation for explicit mask casts
    template <class U>
    mask(const mask<U, Abi> &x,
         enable_if<
             (size() == mask<U, Abi>::size()) &&
             conjunction<std::is_integral<T>, std::is_integral<U>,
                         negation<std::is_same<Abi, datapar_abi::fixed_size<size_v>>>,
                         negation<std::is_same<T, U>>>::value> = nullarg)
        : d{x.d}
    {
    }
    template <class U, class Abi2>
    mask(const mask<U, Abi2> &x,
         enable_if<conjunction<
             negation<std::is_same<abi_type, Abi2>>,
             std::is_same<abi_type, datapar_abi::fixed_size<size_v>>>::value> = nullarg)
    {
        x.memstore(&d[0], flags::vector_aligned);
    }
    */


    // load constructor
    template <class Flags>
    mask(const value_type *mem, Flags f)
        : d(impl::load(mem, f, size_tag))
    {
    }
    template <class Flags> mask(const value_type *mem, mask k, Flags f) : d{}
    {
        impl::masked_load(d, k.d, mem, f, size_tag);
    }

    // loads [mask.load]
    template <class Flags> void memload(const value_type *mem, Flags f)
    {
        d = static_cast<decltype(d)>(impl::load(mem, f, size_tag));
    }
    template <class Flags> void Vc_VDECL memload(const value_type *mem, mask k, Flags f)
    {
        impl::masked_load(d, k.d, mem, f, size_tag);
    }

    // stores [mask.store]
    template <class Flags> void memstore(value_type *mem, Flags f) const
    {
        impl::store(d, mem, f, size_tag);
    }
    template <class Flags> void Vc_VDECL memstore(value_type *mem, mask k, Flags f) const
    {
        impl::masked_store(d, mem, f, k.d, size_tag);
    }

    // scalar access
    reference operator[](size_type i) { return {*this, int(i)}; }
    value_type operator[](size_type i) const { return impl::get(*this, int(i)); }

    // negation
    mask operator!() const { return {detail::private_init, impl::negate(d, size_tag)}; }

    // access to internal representation (suggested extension)
    explicit operator typename traits::mask_cast_type() const { return d; }
    explicit mask(const typename traits::mask_cast_type &init) : d{init} {}

    // mask binary operators [mask.binary]
    friend mask operator&&(const mask &x, const mask &y)
    {
        return impl::logical_and(x, y);
    }
    friend mask operator||(const mask &x, const mask &y)
    {
        return impl::logical_or(x, y);
    }

    friend mask operator&(const mask &x, const mask &y) { return impl::bit_and(x, y); }
    friend mask operator|(const mask &x, const mask &y) { return impl::bit_or(x, y); }
    friend mask operator^(const mask &x, const mask &y) { return impl::bit_xor(x, y); }

    friend mask &operator&=(mask &x, const mask &y) { return x = impl::bit_and(x, y); }
    friend mask &operator|=(mask &x, const mask &y) { return x = impl::bit_or (x, y); }
    friend mask &operator^=(mask &x, const mask &y) { return x = impl::bit_xor(x, y); }

    // mask compares [mask.comparison]
    friend bool operator==(const mask &x, const mask &y) { return std::equal_to<mask>{}(x, y); }
    friend bool operator!=(const mask &x, const mask &y) { return !operator==(x, y); }

private:
#ifdef Vc_MSVC
    // Work around "warning C4396: the inline specifier cannot be used when a friend
    // declaration refers to a specialization of a function template"
    template <class U, class A> friend auto detail::data(const mask<U, A> &);
#else
    friend auto detail::data<T, abi_type>(const mask &);
#endif
    mask(detail::private_init_t, const typename traits::mask_member_type &init) : d(init) {}
//#ifndef Vc_MSVC
    // MSVC refuses by value mask arguments, even if vectorcall__ is used:
    // error C2719: 'k': formal parameter with requested alignment of 16 won't be aligned
    alignas(traits::mask_member_alignment)
//#endif
        typename traits::mask_member_type d = {};
};

namespace detail
{
template <class T, class A> Vc_INTRINSIC auto data(const mask<T, A> &x) { return x.d; }
}  // namespace detail

Vc_VERSIONED_NAMESPACE_END
#endif  // VC_DATAPAR_MASK_H_

// vim: foldmethod=marker
