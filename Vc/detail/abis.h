/*  This file is part of the Vc library. {{{
Copyright © 2017-2018 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_DETAIL_ABIS_H_
#define VC_DETAIL_ABIS_H_

#include "macros.h"
#include "detail.h" // for fixed_size_storage<T, N>

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
// gnu_traits {{{1
template <class T, class MT, class Abi, size_t N> struct gnu_traits {
    using is_valid = std::true_type;
    using simd_impl_type = typename Abi::simd_impl_type;
    using mask_impl_type = typename Abi::mask_impl_type;

    // simd and simd_mask member types {{{2
    using simd_member_type = Storage<T, N>;
    using mask_member_type = Storage<MT, N>;
    static constexpr size_t simd_member_alignment = alignof(simd_member_type);
    static constexpr size_t mask_member_alignment = alignof(mask_member_type);

    // simd_base / base class for simd, providing extra conversions {{{2
    struct simd_base2 {
        explicit operator detail::intrinsic_type_t<T, N>() const
        {
            return static_cast<const simd<T, Abi> *>(this)->d.v();
        }
        explicit operator detail::builtin_type_t<T, N>() const
        {
            return static_cast<const simd<T, Abi> *>(this)->d.builtin();
        }
    };
    struct simd_base1 {
        explicit operator detail::intrinsic_type_t<T, N>() const
        {
            return detail::data(*static_cast<const simd<T, Abi> *>(this));
        }
    };
    using simd_base = std::conditional_t<
        std::is_same<detail::intrinsic_type_t<T, N>, detail::builtin_type_t<T, N>>::value,
        simd_base1, simd_base2>;

    // mask_base {{{2
    struct mask_base2 {
        explicit operator detail::intrinsic_type_t<T, N>() const
        {
            return static_cast<const simd_mask<T, Abi> *>(this)->d.v();
        }
        explicit operator detail::builtin_type_t<T, N>() const
        {
            return static_cast<const simd_mask<T, Abi> *>(this)->d.builtin();
        }
    };
    struct mask_base1 {
        explicit operator detail::intrinsic_type_t<T, N>() const
        {
            return detail::data(*static_cast<const simd_mask<T, Abi> *>(this));
        }
    };
    using mask_base = std::conditional_t<
        std::is_same<detail::intrinsic_type_t<T, N>, detail::builtin_type_t<T, N>>::value,
        mask_base1, mask_base2>;

    // mask_cast_type {{{2
    // parameter type of one explicit simd_mask constructor
    class mask_cast_type
    {
        using U = detail::intrinsic_type_t<T, N>;
        U d;

    public:
        mask_cast_type(U x) : d(x) {}
        operator mask_member_type() const { return d; }
    };

    // simd_cast_type {{{2
    // parameter type of one explicit simd constructor
    class simd_cast_type1
    {
        using A = detail::intrinsic_type_t<T, N>;
        A d;

    public:
        simd_cast_type1(A a) : d(a) {}
        //simd_cast_type1(simd_member_type x) : d(x) {}
        operator simd_member_type() const { return d; }
    };

    class simd_cast_type2
    {
        using A = detail::intrinsic_type_t<T, N>;
        using B = detail::builtin_type_t<T, N>;
        A d;

    public:
        simd_cast_type2(A a) : d(a) {}
        simd_cast_type2(B b) : d(x86::intrin_cast<A>(b)) {}
        //simd_cast_type2(simd_member_type x) : d(x) {}
        operator simd_member_type() const { return d; }
    };

    using simd_cast_type = std::conditional_t<
        std::is_same<detail::intrinsic_type_t<T, N>, detail::builtin_type_t<T, N>>::value,
        simd_cast_type1, simd_cast_type2>;
    //}}}2
};

// neon_is_vectorizable {{{1
#ifdef Vc_HAVE_NEON_ABI
template <class T> struct neon_is_vectorizable : detail::is_vectorizable<T> {};
template <> struct neon_is_vectorizable<long double> : std::false_type {};
#ifndef Vc_HAVE_FULL_NEON_ABI
template <> struct neon_is_vectorizable<double> : std::false_type {};
#endif
#else
template <class T> struct neon_is_vectorizable : std::false_type {};
#endif

// neon_abi {{{1
struct neon_simd_impl;
struct neon_mask_impl;
template <int Bytes> struct neon_abi {
    template <class T> using size_tag = size_constant<Bytes / sizeof(T)>;
    // validity traits {{{2
    struct is_valid_abi_tag : detail::bool_constant<(Bytes > 0 && Bytes <= 16)> {
    };
    template <class T>
    struct is_valid_size_for : detail::bool_constant<(Bytes % sizeof(T) == 0)> {
    };
    template <class T>
    struct is_valid
        : detail::all<is_valid_abi_tag, neon_is_vectorizable<T>, is_valid_size_for<T>> {
    };
    template <class T> static constexpr bool is_valid_v = is_valid<T>::value;

    // simd/mask_impl_type {{{2
    using simd_impl_type = neon_simd_impl;
    using mask_impl_type = neon_mask_impl;

    // traits {{{2
    template <class T>
    using traits =
        std::conditional_t<is_valid_v<T>, gnu_traits<T, T, neon_abi, size_tag<T>::value>,
                           detail::invalid_traits>;
    //}}}2
};

// sse_is_vectorizable {{{1
#ifdef Vc_HAVE_FULL_SSE_ABI
template <class T> struct sse_is_vectorizable : detail::is_vectorizable<T> {};
template <> struct sse_is_vectorizable<long double> : std::false_type {};
#elif defined Vc_HAVE_SSE_ABI
template <class T> struct sse_is_vectorizable : detail::is_same<T, float> {};
#else
template <class T> struct sse_is_vectorizable : std::false_type {};
#endif

// sse_abi {{{1
struct sse_mask_impl;
struct sse_simd_impl;
template <int Bytes> struct sse_abi {
    template <class T> using size_tag = size_constant<Bytes / sizeof(T)>;
    // validity traits {{{2
    struct is_valid_abi_tag : detail::bool_constant<(Bytes > 0 && Bytes <= 16)> {
    };
    template <class T>
    struct is_valid_size_for : detail::bool_constant<(Bytes % sizeof(T) == 0)> {
    };

    template <class T>
    struct is_valid
        : detail::all<is_valid_abi_tag, sse_is_vectorizable<T>, is_valid_size_for<T>> {
    };
    template <class T> static constexpr bool is_valid_v = is_valid<T>::value;

    // simd/mask_impl_type {{{2
    using simd_impl_type = sse_simd_impl;
    using mask_impl_type = sse_mask_impl;

    // traits {{{2
    template <class T>
    using traits = std::conditional_t<is_valid_v<T>,
                                      gnu_traits<T, T, sse_abi, Bytes / sizeof(T)>,
                                      detail::invalid_traits>;
    //}}}2
};

// avx_is_vectorizable {{{1
#ifdef Vc_HAVE_FULL_AVX_ABI
template <class T> struct avx_is_vectorizable : detail::is_vectorizable<T> {};
#elif defined Vc_HAVE_AVX_ABI
template <class T> struct avx_is_vectorizable : std::is_floating_point<T> {};
#else
template <class T> struct avx_is_vectorizable : std::false_type {};
#endif
template <> struct avx_is_vectorizable<long double> : std::false_type {};

// avx_abi {{{1
struct avx_mask_impl;
struct avx_simd_impl;
template <int Bytes> struct avx_abi {
    template <class T> using size_tag = size_constant<Bytes / sizeof(T)>;
    // validity traits {{{2
    struct is_valid_abi_tag : detail::bool_constant<(Bytes > 0 && Bytes <= 32)> {
    };
    template <class T>
    struct is_valid_size_for : detail::bool_constant<(Bytes % sizeof(T) == 0)> {
    };
    template <class T>
    struct is_valid
        : detail::all<is_valid_abi_tag, avx_is_vectorizable<T>, is_valid_size_for<T>> {
    };
    template <class T> static constexpr bool is_valid_v = is_valid<T>::value;

    // simd/mask_impl_type {{{2
    using simd_impl_type = avx_simd_impl;
    using mask_impl_type = avx_mask_impl;

    // traits {{{2
    template <class T>
    using traits = std::conditional_t<is_valid_v<T>,
                                      gnu_traits<T, T, avx_abi, Bytes / sizeof(T)>,
                                      detail::invalid_traits>;
    //}}}2
};

// avx512_is_vectorizable {{{1
#ifdef Vc_HAVE_AVX512_ABI
template <class T> struct avx512_is_vectorizable : detail::is_vectorizable<T> {};
template <> struct avx512_is_vectorizable<long double> : std::false_type {};
#ifndef Vc_HAVE_FULL_AVX512_ABI
template <> struct avx512_is_vectorizable<  char> : std::false_type {};
template <> struct avx512_is_vectorizable< uchar> : std::false_type {};
template <> struct avx512_is_vectorizable< schar> : std::false_type {};
template <> struct avx512_is_vectorizable< short> : std::false_type {};
template <> struct avx512_is_vectorizable<ushort> : std::false_type {};
template <> struct avx512_is_vectorizable<char16_t> : std::false_type {};
template <> struct avx512_is_vectorizable<wchar_t> : detail::bool_constant<sizeof(wchar_t) == 2> {};
#endif
#else
template <class T> struct avx512_is_vectorizable : std::false_type {};
#endif

// avx512_abi {{{1
struct avx512_mask_impl;
struct avx512_simd_impl;
template <int Bytes> struct avx512_abi {
    template <class T> using size_tag = size_constant<Bytes / sizeof(T)>;
    // validity traits {{{2
    struct is_valid_abi_tag : detail::bool_constant<(Bytes > 0 && Bytes <= 64)> {
    };
    template <class T>
    struct is_valid_size_for : detail::bool_constant<(Bytes % sizeof(T) == 0)> {
    };
    template <class T>
    struct is_valid
        : detail::all<is_valid_abi_tag, avx512_is_vectorizable<T>, is_valid_size_for<T>> {
    };
    template <class T> static constexpr bool is_valid_v = is_valid<T>::value;

    // simd/mask_impl_type {{{2
    using simd_impl_type = avx512_simd_impl;
    using mask_impl_type = avx512_mask_impl;

    // traits {{{2
    template <class T>
    using traits = std::conditional_t<is_valid_v<T>,
                                      gnu_traits<T, bool, avx512_abi, Bytes / sizeof(T)>,
                                      detail::invalid_traits>;
    //}}}2
};

// scalar_abi {{{1
struct scalar_simd_impl;
struct scalar_mask_impl;
struct scalar_abi {
    template <class T> using size_tag = size_constant<1>;
    struct is_valid_abi_tag : std::true_type {};
    template <class T> struct is_valid_size_for : std::true_type {};
    template <class T> struct is_valid : is_vectorizable<T> {};
    template <class T> static constexpr bool is_valid_v = is_valid<T>::value;

    using simd_impl_type = scalar_simd_impl;
    using mask_impl_type = scalar_mask_impl;

    template <class T, bool = is_valid_v<T>> struct traits : detail::invalid_traits {
    };

    template <class T> struct traits<T, true> {
        using is_valid = std::true_type;
        using simd_impl_type = scalar_simd_impl;
        using mask_impl_type = scalar_mask_impl;
        using simd_member_type = T;
        using mask_member_type = bool;
        static constexpr size_t simd_member_alignment = alignof(simd_member_type);
        static constexpr size_t mask_member_alignment = alignof(mask_member_type);

        // nothing the user can spell converts to/from simd/simd_mask
        struct simd_cast_type {
            simd_cast_type() = delete;
        };
        struct mask_cast_type {
            mask_cast_type() = delete;
        };
        struct simd_base {};
        struct mask_base {};
    };
};

// fixed_abi {{{1
template <int N> struct fixed_size_simd_impl;
template <int N> struct fixed_size_mask_impl;
template <int N> struct fixed_abi {
    template <class T> using size_tag = size_constant<N>;
    // validity traits {{{2
    struct is_valid_abi_tag
        : public detail::bool_constant<(N > 0)> {
    };
    template <class T>
    struct is_valid_size_for
        : detail::bool_constant<((N <= simd_abi::max_fixed_size<T>) ||
                                 (simd_abi::Neon::is_valid_v<char> &&
                                  N == simd_size_v<char, simd_abi::Neon>) ||
                                 (simd_abi::Sse::is_valid_v<char> &&
                                  N == simd_size_v<char, simd_abi::Sse>) ||
                                 (simd_abi::Avx::is_valid_v<char> &&
                                  N == simd_size_v<char, simd_abi::Avx>) ||
                                 (simd_abi::Avx512::is_valid_v<char> &&
                                  N == simd_size_v<char, simd_abi::Avx512>))> {
    };
    template <class T>
    struct is_valid
        : detail::all<is_valid_abi_tag, is_vectorizable<T>, is_valid_size_for<T>> {
    };
    template <class T> static constexpr bool is_valid_v = is_valid<T>::value;

    // simd/mask_impl_type {{{2
    using simd_impl_type = fixed_size_simd_impl<N>;
    using mask_impl_type = fixed_size_mask_impl<N>;

    // traits {{{2
    template <class T, bool = is_valid_v<T>> struct traits : detail::invalid_traits {
    };

    template <class T> struct traits<T, true> {
        using is_valid = std::true_type;
        using simd_impl_type = fixed_size_simd_impl<N>;
        using mask_impl_type = fixed_size_mask_impl<N>;

        // simd and simd_mask member types {{{2
        using simd_member_type = fixed_size_storage<T, N>;
        using mask_member_type = std::bitset<N>;
        static constexpr size_t simd_member_alignment =
#ifdef Vc_GCC
            std::min(size_t(
#ifdef __AVX__
                         256
#else
                         128
#endif
                         ),
#else
            (
#endif
                     next_power_of_2(N * sizeof(T)));
        static constexpr size_t mask_member_alignment = alignof(mask_member_type);

        // simd_base / base class for simd, providing extra conversions {{{2
        struct simd_base {
            explicit operator const simd_member_type &() const
            {
                return static_cast<const simd<T, fixed_abi> *>(this)->d;
            }
            explicit operator std::array<T, N>() const
            {
                std::array<T, N> r;
                // simd_member_type can be larger because of higher alignment
                static_assert(sizeof(r) <= sizeof(simd_member_type), "");
                std::memcpy(r.data(), &static_cast<const simd_member_type &>(*this),
                            sizeof(r));
                return r;
            }
        };

        // mask_base {{{2
        // empty. The std::bitset interface suffices
        struct mask_base {};

        // simd_cast_type {{{2
        struct simd_cast_type {
            simd_cast_type(const std::array<T, N> &);
            simd_cast_type(const simd_member_type &dd) : d(dd) {}
            explicit operator const simd_member_type &() const { return d; }

        private:
            const simd_member_type &d;
        };

        // mask_cast_type {{{2
        class mask_cast_type
        {
            mask_cast_type() = delete;
        };
        //}}}2
    };
};

// valid traits specialization {{{1
template <class T, class Abi>
struct traits<T, Abi, void_t<typename Abi::template is_valid<T>>>
    : Abi::template traits<T> {
};

//}}}1
}  // namespace detail
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_DETAIL_ABIS_H_

// vim: foldmethod=marker
