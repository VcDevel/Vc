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

#ifndef VC_DATAPAR_SSE_H_
#define VC_DATAPAR_SSE_H_

#include "../common/storage.h"

namespace Vc::v2::detail {
// datapar impl {{{1
struct sse_datapar_impl {
    using abi = datapar_abi::sse;
    template <class T> static constexpr size_t size = datapar_size_v<T, abi>;
    template <class T> using datapar_member_type = Vc::Common::Storage<T, size<T>>;
};

// mask impl {{{1
struct sse_mask_impl {
    // member types {{{2
    using abi = datapar_abi::sse;
    template <class T> static constexpr size_t size = datapar_size_v<T, abi>;
    template <class T> using mask_member_type = Vc::Common::Storage<T, size<T>>;
    template <class T> using mask = Vc::mask<T, datapar_abi::sse>;
    template <class T> using mask_bool = Common::MaskBool<sizeof(T)>;

    // broadcast {{{2
    static auto broadcast(bool x, double *) noexcept
    {
        return _mm_set1_pd(mask_bool<double>{x});
    }
    static auto broadcast(bool x, float *) noexcept
    {
        return _mm_set1_ps(mask_bool<float>{x});
    }
    static auto broadcast(bool x, std::int64_t *) noexcept
    {
        return _mm_set1_epi64x(mask_bool<std::int64_t>{x});
    }
    static auto broadcast(bool x, std::int32_t *) noexcept
    {
        return _mm_set1_epi32(mask_bool<std::int32_t>{x});
    }
    static auto broadcast(bool x, std::int16_t *) noexcept
    {
        return _mm_set1_epi16(mask_bool<std::int16_t>{x});
    }
    static auto broadcast(bool x, std::int8_t *) noexcept
    {
        return _mm_set1_epi8(mask_bool<std::int8_t>{x});
    }

    // convert {{{2
    template <class U, class T>
    static constexpr const mask_member_type<T> &convert(const mask<U> &x, T *) noexcept
    {
        return x.d;
    }
    template <class U, class Abi2, class T>
    static constexpr mask_member_type<T> convert(const Vc::mask<U, Abi2> &x, T *) noexcept
    {
        mask_member_type<T> r;
        for (std::size_t i = 0; i < size<T>; ++i) {
            r.set(i, mask_bool<T>{x[i]});
        }
        return r;
    }

    // load {{{2
    template <class T, class F>
    static constexpr mask_member_type<T> load(const bool *mem, F, T *) noexcept
    {
        mask_member_type<T> r;
        for (std::size_t i = 0; i < size<T>; ++i) {
            r.set(i, mask_bool<T>{mem[i]});
        }
        return r;
    }

    // negation {{{2
    template <class T>
    static constexpr mask_member_type<T> negate(const mask_member_type<T> &x, T *) noexcept
    {
        return !x.builtin();
    }

    // smart_reference access {{{2
    template <class T> static bool get(const mask<T> &k, int i) noexcept
    {
        return k.d.m(i);
    }
    template <class T> static void set(mask<T> &k, int i, bool x) noexcept
    {
        k.d.set(i, x);
    }
    // }}}2
};

// traits {{{1
template <class T> struct traits<T, datapar_abi::sse> {
    static constexpr size_t size() noexcept
    {
        return sizeof(T) <= 8 ? 16 / sizeof(T) : 1;
    }

    using datapar_impl_type = sse_datapar_impl;
    using datapar_member_type =
        typename datapar_impl_type::template datapar_member_type<T>;
    static constexpr size_t datapar_member_alignment = alignof(datapar_member_type);

    using mask_impl_type = sse_mask_impl;
    using mask_member_type = typename mask_impl_type::template mask_member_type<T>;
    static constexpr size_t mask_member_alignment = alignof(mask_member_type);
    using mask_cast_type = typename mask_member_type::VectorType;
};

// mask compare base {{{1
struct sse_compare_base {
protected:
    template <class T> using M = Vc::mask<T, Vc::datapar_abi::sse>;
    template <class T>
    using S = typename Vc::detail::traits<T, Vc::datapar_abi::sse>::mask_cast_type;
};
// }}}1
}  // namespace Vc::v2::detail

namespace std
{
// datapar operators {{{1
template <class T>
struct equal_to<Vc::datapar<T, Vc::datapar_abi::sse>> {
private:
    using V = Vc::datapar<T, Vc::datapar_abi::sse>;
    using M = typename V::mask_type;


public:
    M operator()(const V &x, const V &y) const
    {
        return {};
    }
};

// mask operators {{{1
template <class T>
struct equal_to<Vc::mask<T, Vc::datapar_abi::sse>>
    : private Vc::detail::sse_compare_base {
public:
    bool operator()(const M<T> &x, const M<T> &y) const
    {
        return Vc::Detail::is_equal<M<T>::size()>(
            Vc::sse_cast<__m128>(static_cast<S<T>>(x)),
            Vc::sse_cast<__m128>(static_cast<S<T>>(y)));
    }
};
// }}}1
}  // namespace std

#endif  // VC_DATAPAR_SSE_H_

// vim: foldmethod=marker
