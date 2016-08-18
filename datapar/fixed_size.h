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

namespace Vc_VERSIONED_NAMESPACE::detail {
// datapar impl {{{1
template <int N> struct fixed_size_datapar_impl {
    // member types {{{2
    static constexpr std::make_index_sequence<N> index_seq = {};
    using mask_member_type = std::array<bool, N>;
    template <class T> using datapar_member_type = std::array<T, N>;
    template <class T> using datapar = Vc::datapar<T, datapar_abi::fixed_size<N>>;
    template <class T> using mask = Vc::mask<T, datapar_abi::fixed_size<N>>;
    using size_tag = std::integral_constant<size_t, N>;
    template <class T> using type_tag = T *;

    template <typename T> static constexpr void unused(T &&) {}

    // broadcast {{{2
    template <class T, size_t... I>
    static constexpr datapar_member_type<T> broadcast_impl(
        T x, std::index_sequence<I...>) noexcept
    {
        return {((void)I, x)...};
    }
    template <class T>
    static constexpr datapar_member_type<T> broadcast(T x, size_tag) noexcept
    {
        return broadcast_impl(x, index_seq);
    }

    // load {{{2
    template <class T, class U, size_t... I>
    static constexpr datapar_member_type<T> load_impl(const U *mem,
                                                      std::index_sequence<I...>) noexcept
    {
        return {static_cast<T>(mem[I])...};
    }
    template <class T, class U, class F>
    static constexpr datapar_member_type<T> load(const U *mem, F, type_tag<T>) noexcept
    {
        return load_impl<T>(mem, index_seq);
    }

    // masked load {{{2
    template <class T, class U, size_t... I>
    static constexpr void masked_load_impl(datapar_member_type<T> &merge,
                                           const mask_member_type &mask, const U *mem,
                                           std::index_sequence<I...>) noexcept
    {
        auto &&x = {(merge[I] = mask[I] ? static_cast<T>(mem[I]) : merge[I])...};
        unused(x);
    }
    template <class T, class U, class F>
    static constexpr void masked_load(datapar_member_type<T> &merge, const mask<T> &k,
                                      const U *mem, F) noexcept
    {
        masked_load_impl(merge, k.d, mem, index_seq);
    }

    // store {{{2
    template <class T, class U, size_t... I>
    static constexpr void store_impl(const datapar_member_type<T> &v, U *mem,
                                     std::index_sequence<I...>) noexcept
    {
        auto &&x = {(mem[I] = static_cast<U>(v[I]))...};
        unused(x);
    }
    template <class T, class U, class F>
    static constexpr void store(const datapar_member_type<T> &v, U *mem, F, type_tag<T>) noexcept
    {
        return store_impl(v, mem, index_seq);
    }

    // masked store {{{2
    template <class T, class U, size_t... I>
    static constexpr void masked_store_impl(const datapar_member_type<T> &v, U *mem,
                                            std::index_sequence<I...>,
                                            const mask_member_type &k) noexcept
    {
        auto &&x = {(k[I] ? mem[I] = static_cast<U>(v[I]) : false)...};
        unused(x);
    }
    template <class T, class U, class F>
    static constexpr void masked_store(const datapar_member_type<T> &v, U *mem, F,
                                       const mask<T> &k) noexcept
    {
        return masked_store_impl(v, mem, index_seq, k.d);
    }

    // negation {{{2
    template <class T, size_t... I>
    static constexpr mask_member_type negate_impl(
        const datapar_member_type<T> &x, std::index_sequence<I...>) noexcept
    {
        return {!x[I]...};
    }
    template <class T, class A>
    static constexpr Vc::mask<T, A> negate(const Vc::datapar<T, A> &x) noexcept
    {
        return {private_init, negate_impl(x.d, index_seq)};
    }

    // compares {{{2
    template <template <typename> class Cmp, class T, size_t... I>
    static constexpr mask_member_type cmp_impl(const datapar_member_type<T> &x,
                                               const datapar_member_type<T> &y,
                                               std::index_sequence<I...>)
    {
        Cmp<T> cmp;
        return {cmp(x[I], y[I])...};
    }
#define Vc_CMP_OPERATIONS(cmp_)                                                          \
    template <class V>                                                                   \
    static constexpr typename V::mask_type cmp_(const V &x, const V &y)                  \
    {                                                                                    \
        return {private_init, cmp_impl<std::cmp_>(x.d, y.d, index_seq)};                 \
    }                                                                                    \
    Vc_NOTHING_EXPECTING_SEMICOLON
    Vc_CMP_OPERATIONS(equal_to);
    Vc_CMP_OPERATIONS(not_equal_to);
    Vc_CMP_OPERATIONS(less);
    Vc_CMP_OPERATIONS(greater);
    Vc_CMP_OPERATIONS(less_equal);
    Vc_CMP_OPERATIONS(greater_equal);

    // smart_reference access {{{2
    template <class T, class A>
    static T get(const Vc::datapar<T, A> &v, int i) noexcept
    {
        return v.d[i];
    }
    template <class T, class A, class U>
    static void set(Vc::datapar<T, A> &v, int i, U &&x) noexcept
    {
        v.d[i] = std::forward<U>(x);
    }
    // }}}2
};

// mask impl {{{1
template <int N> struct fixed_size_mask_impl {
    // member types {{{2
    static constexpr std::make_index_sequence<N> index_seq = {};
    using mask_member_type = std::array<bool, N>;
    template <class T> using mask = Vc::mask<T, datapar_abi::fixed_size<N>>;
    using size_tag = std::integral_constant<size_t, N>;

    template <typename T> static constexpr void unused(T &&) {}

    // broadcast {{{2
    template <size_t... I>
    static constexpr mask_member_type broadcast_impl(
        bool x, std::index_sequence<I...>) noexcept
    {
        return {((void)I, x)...};
    }
    static constexpr mask_member_type broadcast(bool x, size_tag) noexcept
    {
        return broadcast_impl(x, index_seq);
    }

    // load {{{2
    template <size_t... I>
    static constexpr mask_member_type load_impl(const bool *mem,
                                                std::index_sequence<I...>) noexcept
    {
        return {mem[I]...};
    }
    template <class F>
    static constexpr mask_member_type load(const bool *mem, F, size_tag) noexcept
    {
        return load_impl(mem, index_seq);
    }

    // masked load {{{2
    template <size_t... I>
    static constexpr void masked_load_impl(mask_member_type &merge,
                                           const mask_member_type &mask, const bool *mem,
                                           std::index_sequence<I...>) noexcept
    {
        auto &&x = {(merge[I] = mask[I] ? mem[I] : merge[I])...};
        unused(x);
    }
    template <class F>
    static constexpr void masked_load(mask_member_type &merge,
                                      const mask_member_type &mask, const bool *mem, F,
                                      size_tag) noexcept
    {
        masked_load_impl(merge, mask, mem, index_seq);
    }

    // store {{{2
    template <size_t... I>
    static constexpr void store_impl(const mask_member_type &v, bool *mem,
                                     std::index_sequence<I...>) noexcept
    {
        auto &&x = {(mem[I] = v[I])...};
        unused(x);
    }
    template <class F>
    static constexpr void store(const mask_member_type &v, bool *mem, F, size_tag) noexcept
    {
        return store_impl(v, mem, index_seq);
    }

    // masked store {{{2
    template <size_t... I>
    static constexpr void masked_store_impl(const mask_member_type &v, bool *mem,
                                            std::index_sequence<I...>,
                                            const mask_member_type &k) noexcept
    {
        auto &&x = {(k[I] ? mem[I] = v[I] : false)...};
        unused(x);
    }
    template <class F>
    static constexpr void masked_store(const mask_member_type &v, bool *mem, F,
                                       const mask_member_type &k, size_tag) noexcept
    {
        return masked_store_impl(v, mem, index_seq, k);
    }

    // negation {{{2
    template <size_t... I>
    static constexpr mask_member_type negate_impl(const mask_member_type &x,
                                                  std::index_sequence<I...>) noexcept
    {
        return {!x[I]...};
    }
    static constexpr mask_member_type negate(const mask_member_type &x, size_tag) noexcept
    {
        return negate_impl(x, index_seq);
    }

    // smart_reference access {{{2
    template <class T, class A> static bool get(const Vc::mask<T, A> &k, int i) noexcept
    {
        return k.d[i];
    }
    template <class T, class A> static void set(Vc::mask<T, A> &k, int i, bool x) noexcept
    {
        k.d[i] = x;
    }
    // }}}2
};

// traits {{{1
template <class T, int N> struct traits<T, datapar_abi::fixed_size<N>> {
    static constexpr size_t size() noexcept { return N; }

    using datapar_impl_type = fixed_size_datapar_impl<N>;
    using datapar_member_type = std::array<T, N>;
    static constexpr size_t datapar_member_alignment =
        std::min(size_t(
#ifdef __AVX__
                        256
#else
                        128
#endif
                        ), next_power_of_2(N * sizeof(T)));
    using datapar_cast_type = datapar_member_type;

    using mask_impl_type = fixed_size_mask_impl<N>;
    using mask_member_type = typename mask_impl_type::mask_member_type;
    static constexpr size_t mask_member_alignment = next_power_of_2(N);
    using mask_cast_type = const mask_member_type &;
};
// }}}1
}  // namespace Vc_VERSIONED_NAMESPACE::detail

namespace std
{
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
