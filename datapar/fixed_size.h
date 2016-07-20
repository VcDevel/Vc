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

#ifndef VC_DATAPAR_FIXED_SIZE_H_
#define VC_DATAPAR_FIXED_SIZE_H_

#include "detail.h"
#include <array>

namespace Vc::v2::detail {
// datapar impl {{{1
template <int N> struct fixed_size_datapar_impl {
};

// mask impl {{{1
template <int N> struct fixed_size_mask_impl {
    // member types {{{2
    using index_seq = std::make_index_sequence<N>;
    using mask_member_type = std::array<bool, N>;
    template <class T> using mask = Vc::mask<T, datapar_abi::fixed_size<N>>;

    // broadcast {{{2
    template <size_t... I>
    static constexpr mask_member_type broadcast_impl(
        bool x, std::index_sequence<I...>) noexcept
    {
        return {((void)I, x)...};
    }
    template <class T> static constexpr mask_member_type broadcast(bool x, T *) noexcept
    {
        return broadcast_impl(x, index_seq{});
    }

    // convert {{{2
    template <class U, class Abi2, size_t... I>
    static constexpr mask_member_type convert_impl(const Vc::mask<U, Abi2> &x,
                                                   std::index_sequence<I...>) noexcept
    {
        return { x[I]... };
    }
    template <class U, class Abi2, class T>
    static constexpr mask_member_type convert(const Vc::mask<U, Abi2> &x, T *) noexcept
    {
        return convert_impl(x, index_seq{});
    }

    // load {{{2
    template <size_t... I>
    static constexpr mask_member_type load_impl(const bool *mem,
                                                std::index_sequence<I...>) noexcept
    {
        return {mem[I]...};
    }
    template <class F, class T>
    static constexpr mask_member_type load(const bool *mem, F, T *) noexcept
    {
        return load_impl(mem, index_seq{});
    }

    // negation {{{2
    template <size_t... I>
    static constexpr mask_member_type negate_impl(const mask_member_type &x,
                                                  std::index_sequence<I...>) noexcept
    {
        return {!x[I]...};
    }
    template <class T>
    static constexpr mask_member_type negate(const mask_member_type &x, T *) noexcept
    {
        return negate_impl(x, index_seq{});
    }

    // smart_reference access {{{2
    template <class T> static bool get(const mask<T> &k, int i) noexcept
    {
        return k.d[i];
    }
    template <class T> static void set(mask<T> &k, int i, bool x) noexcept { k.d[i] = x; }
    // }}}2
};

// traits {{{1
template <class T, int N> struct traits<T, datapar_abi::fixed_size<N>> {
    static constexpr size_t size() noexcept { return N; }

    using datapar_impl_type = fixed_size_datapar_impl<N>;
    using datapar_member_type = std::array<T, N>;
    static constexpr size_t datapar_member_alignment = next_power_of_2(N * sizeof(T));
    using datapar_cast_type = datapar_member_type;

    using mask_impl_type = fixed_size_mask_impl<N>;
    using mask_member_type = typename mask_impl_type::mask_member_type;
    static constexpr size_t mask_member_alignment = next_power_of_2(N);
    using mask_cast_type = mask_member_type;
};
// }}}1
}  // namespace Vc::v2::detail

namespace std
{
// datapar operators {{{1
template <class T, int N>
struct equal_to<Vc::datapar<T, Vc::datapar_abi::fixed_size<N>>> {
private:
    using V = Vc::datapar<T, Vc::datapar_abi::fixed_size<N>>;
    using M = typename V::mask_type;

    template <size_t... I> M impl(const V &x, const V &y, index_sequence<I...>) const
    {
        return {(x[I] == y[I])...};
    }

public:
    M operator()(const V &x, const V &y) const
    {
        return impl(x, y, make_index_sequence<N>());
    }
};

// mask operators {{{1
template <class T, int N>
struct equal_to<Vc::mask<T, Vc::datapar_abi::fixed_size<N>>> {
private:
    using M = Vc::mask<T, Vc::datapar_abi::fixed_size<N>>;

public:
    bool operator()(const M &x, const M &y) const
    {
        bool r = x[0] == y[0];
        for (int i = 1; i < N; ++i) {
            r = r && x[i] == y[i];
        }
        return r;
    }
};
// }}}1
}  // namespace std

#endif  // VC_DATAPAR_FIXED_SIZE_H_

// vim: foldmethod=marker
