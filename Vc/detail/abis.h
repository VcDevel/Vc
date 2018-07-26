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
struct neon_simd_impl;
struct neon_mask_impl;
struct sse_mask_impl;
struct sse_simd_impl;
struct avx_mask_impl;
struct avx_simd_impl;
struct avx512_mask_impl;
struct avx512_simd_impl;
struct scalar_simd_impl;
struct scalar_mask_impl;
template <int N> struct fixed_size_simd_impl;
template <int N> struct fixed_size_mask_impl;
template <int N, class Abi> struct combine_simd_impl;
template <int N, class Abi> struct combine_mask_impl;

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
            return static_cast<const simd_mask<T, Abi> *>(this)->d.intrin();
        }
        explicit operator detail::builtin_type_t<T, N>() const
        {
            return static_cast<const simd_mask<T, Abi> *>(this)->d.d;
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
        simd_cast_type2(B b) : d(detail::intrin_cast<A>(b)) {}
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

// sse_is_vectorizable {{{1
#ifdef Vc_HAVE_FULL_SSE_ABI
template <class T> struct sse_is_vectorizable : detail::is_vectorizable<T> {};
template <> struct sse_is_vectorizable<long double> : std::false_type {};
#elif defined Vc_HAVE_SSE_ABI
template <class T> struct sse_is_vectorizable : detail::is_same<T, float> {};
#else
template <class T> struct sse_is_vectorizable : std::false_type {};
#endif

// avx_is_vectorizable {{{1
#ifdef Vc_HAVE_FULL_AVX_ABI
template <class T> struct avx_is_vectorizable : detail::is_vectorizable<T> {};
#elif defined Vc_HAVE_AVX_ABI
template <class T> struct avx_is_vectorizable : std::is_floating_point<T> {};
#else
template <class T> struct avx_is_vectorizable : std::false_type {};
#endif
template <> struct avx_is_vectorizable<long double> : std::false_type {};

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
template <> struct avx512_is_vectorizable<wchar_t> : detail::bool_constant<sizeof(wchar_t) >= 4> {};
#endif
#else
template <class T> struct avx512_is_vectorizable : std::false_type {};
#endif

// }}}
// implicit_mask_abi_base {{{
template <int Bytes, class Abi> struct implicit_mask_abi_base {
    template <class T>
    using implicit_mask_type =
        detail::builtin_type_t<detail::int_for_sizeof_t<T>, simd_size_v<T, Abi>>;

    template <class T>
    static constexpr implicit_mask_type<T> implicit_mask =
        Abi::is_partial ? detail::generate_builtin<implicit_mask_type<T>>([](auto i) {
            return i < Bytes / sizeof(T) ? -1 : 0;
        })
                        : ~implicit_mask_type<T>();

    template <class T, class Trait = detail::builtin_traits<T>>
    static constexpr auto masked(T x)
    {
        using U = typename Trait::value_type;
        return builtin_cast<detail::int_for_sizeof_t<U>>(x) & implicit_mask<U>;
    }
};

// }}}
}  // namespace detail
namespace simd_abi
{
// __combine {{{1
template <int N, class Abi> struct __combine {
    template <class T>
    using size_tag = detail::size_constant<N * Abi::template size_tag<T>::value>;

    static constexpr int factor = N;
    using member_abi = Abi;

    // validity traits {{{2
    // allow 2x, 3x, and 4x "unroll"
    struct is_valid_abi_tag
        : detail::bool_constant<(N > 1 && N <= 4) && Abi::is_valid_abi_tag> {
    };
    template <class T> struct is_valid_size_for : Abi::template is_valid_size_for<T> {
    };
    template <class T>
    struct is_valid : detail::all<is_valid_abi_tag, typename Abi::template is_valid<T>> {
    };
    template <class T> static constexpr bool is_valid_v = is_valid<T>::value;

    // simd/mask_impl_type {{{2
    using simd_impl_type = detail::combine_simd_impl<N, Abi>;
    using mask_impl_type = detail::combine_mask_impl<N, Abi>;

    // traits {{{2
    template <class T, bool = is_valid_v<T>> struct traits : detail::invalid_traits {
    };

    template <class T> struct traits<T, true> {
        using is_valid = std::true_type;
        using simd_impl_type = detail::combine_simd_impl<N, Abi>;
        using mask_impl_type = detail::combine_mask_impl<N, Abi>;

        // simd and simd_mask member types {{{2
        using simd_member_type =
            std::array<typename Abi::template traits<T>::simd_member_type, N>;
        using mask_member_type =
            std::array<typename Abi::template traits<T>::mask_member_type, N>;
        static constexpr size_t simd_member_alignment =
            Abi::template traits<T>::simd_member_alignment;
        static constexpr size_t mask_member_alignment =
            Abi::template traits<T>::mask_member_alignment;

        // simd_base / base class for simd, providing extra conversions {{{2
        struct simd_base {
            explicit operator const simd_member_type &() const
            {
                return static_cast<const simd<T, __combine> *>(this)->d;
            }
        };

        // mask_base {{{2
        // empty. The std::bitset interface suffices
        struct mask_base {
            explicit operator const mask_member_type &() const
            {
                return static_cast<const simd_mask<T, __combine> *>(this)->d;
            }
        };

        // simd_cast_type {{{2
        struct simd_cast_type {
            simd_cast_type(const simd_member_type &dd) : d(dd) {}
            explicit operator const simd_member_type &() const { return d; }

        private:
            const simd_member_type &d;
        };

        // mask_cast_type {{{2
        struct mask_cast_type {
            mask_cast_type(const mask_member_type &dd) : d(dd) {}
            explicit operator const mask_member_type &() const { return d; }

        private:
            const mask_member_type &d;
        };
        //}}}2
    };
    //}}}2
};
// __neon_abi {{{1
template <int Bytes>
struct __neon_abi : detail::implicit_mask_abi_base<Bytes, __neon_abi<Bytes>> {
    template <class T> using size_tag = detail::size_constant<Bytes / sizeof(T)>;
    static constexpr bool is_partial = Bytes < 16;

    // validity traits {{{2
    struct is_valid_abi_tag : detail::bool_constant<(Bytes > 0 && Bytes <= 16)> {
    };
    template <class T>
    struct is_valid_size_for
        : detail::bool_constant<(Bytes / sizeof(T) > 1 && Bytes % sizeof(T) == 0)> {
    };
    template <class T>
    struct is_valid : detail::all<is_valid_abi_tag, detail::neon_is_vectorizable<T>,
                                  is_valid_size_for<T>> {
    };
    template <class T> static constexpr bool is_valid_v = is_valid<T>::value;

    // simd/mask_impl_type {{{2
    using simd_impl_type = detail::neon_simd_impl;
    using mask_impl_type = detail::neon_mask_impl;

    // traits {{{2
    template <class T>
    using traits =
        std::conditional_t<is_valid_v<T>,
                           detail::gnu_traits<T, T, __neon_abi, size_tag<T>::value>,
                           detail::invalid_traits>;
    //}}}2
};

// __sse_abi {{{1
template <int Bytes>
struct __sse_abi : detail::implicit_mask_abi_base<Bytes, __sse_abi<Bytes>> {
    template <class T> using size_tag = detail::size_constant<Bytes / sizeof(T)>;
    static constexpr bool is_partial = Bytes < 16;

    // validity traits {{{2
    // allow 2x, 3x, and 4x "unroll"
    struct is_valid_abi_tag : detail::bool_constant<Bytes == 16> {};
    /* TODO:
    struct is_valid_abi_tag
        : detail::bool_constant<((Bytes > 0 && Bytes <= 16) || Bytes == 32 ||
                                 Bytes == 48 || Bytes == 64)> {
    };
    */
    template <class T>
    struct is_valid_size_for
        : detail::bool_constant<(Bytes / sizeof(T) > 1 && Bytes % sizeof(T) == 0)> {
    };

    template <class T>
    struct is_valid
        : detail::all<is_valid_abi_tag, detail::sse_is_vectorizable<T>, is_valid_size_for<T>> {
    };
    template <class T> static constexpr bool is_valid_v = is_valid<T>::value;

    // simd/mask_impl_type {{{2
    using simd_impl_type = detail::sse_simd_impl;
    using mask_impl_type = detail::sse_mask_impl;

    // traits {{{2
    template <class T>
    using traits =
        std::conditional_t<is_valid_v<T>,
                           detail::gnu_traits<T, T, __sse_abi, Bytes / sizeof(T)>,
                           detail::invalid_traits>;
    //}}}2
};

// __avx_abi {{{1
template <int Bytes>
struct __avx_abi : detail::implicit_mask_abi_base<Bytes, __avx_abi<Bytes>> {
    template <class T> using size_tag = detail::size_constant<Bytes / sizeof(T)>;
    static constexpr bool is_partial = Bytes < 32;

    // validity traits {{{2
    // - allow 2x, 3x, and 4x "unroll"
    // - disallow <= 16 Bytes as that's covered by __sse_abi
    struct is_valid_abi_tag : detail::bool_constant<Bytes == 32> {};
    /* TODO:
    struct is_valid_abi_tag
        : detail::bool_constant<((Bytes > 16 && Bytes <= 32) || Bytes == 64 ||
                                 Bytes == 96 || Bytes == 128)> {
    };
    */
    template <class T>
    struct is_valid_size_for : detail::bool_constant<(Bytes % sizeof(T) == 0)> {
    };
    template <class T>
    struct is_valid
        : detail::all<is_valid_abi_tag, detail::avx_is_vectorizable<T>, is_valid_size_for<T>> {
    };
    template <class T> static constexpr bool is_valid_v = is_valid<T>::value;

    // simd/mask_impl_type {{{2
    using simd_impl_type = detail::avx_simd_impl;
    using mask_impl_type = detail::avx_mask_impl;

    // traits {{{2
    template <class T>
    using traits =
        std::conditional_t<is_valid_v<T>,
                           detail::gnu_traits<T, T, __avx_abi, Bytes / sizeof(T)>,
                           detail::invalid_traits>;
    //}}}2
};

// __avx512_abi {{{1
template <int Bytes> struct __avx512_abi {
    template <class T> using size_tag = detail::size_constant<Bytes / sizeof(T)>;
    static constexpr bool is_partial = Bytes < 64;

    // validity traits {{{2
    // - disallow <= 32 Bytes as that's covered by __sse_abi and __avx_abi
    // TODO: consider AVX512VL
    struct is_valid_abi_tag : detail::bool_constant<Bytes == 64> {};
    /* TODO:
    struct is_valid_abi_tag
        : detail::bool_constant<(Bytes > 32 && Bytes <= 64)> {
    };
    */
    template <class T>
    struct is_valid_size_for : detail::bool_constant<(Bytes % sizeof(T) == 0)> {
    };
    template <class T>
    struct is_valid
        : detail::all<is_valid_abi_tag, detail::avx512_is_vectorizable<T>, is_valid_size_for<T>> {
    };
    template <class T> static constexpr bool is_valid_v = is_valid<T>::value;

    // implicit mask {{{2
    template <class T>
    using implicit_mask_type = detail::bool_storage_member_type_t<64 / sizeof(T)>;

    template <class T>
    static constexpr implicit_mask_type<T> implicit_mask =
        Bytes == 64 ? ~implicit_mask_type<T>()
                    : (implicit_mask_type<T>(1) << (Bytes / sizeof(T))) - 1;

    template <class T, class = std::enable_if_t<detail::is_bitmask_v<T>>>
    static constexpr T masked(T x)
    {
        if constexpr (is_partial) {
            constexpr size_t N = sizeof(T) * 8;
            return x &
                   ((detail::bool_storage_member_type_t<N>(1) << (Bytes * N / 64)) - 1);
        } else {
            return x;
        }
    }

    // simd/mask_impl_type {{{2
    using simd_impl_type = detail::avx512_simd_impl;
    using mask_impl_type = detail::avx512_mask_impl;

    // traits {{{2
    template <class T>
    using traits =
        std::conditional_t<is_valid_v<T>,
                           detail::gnu_traits<T, bool, __avx512_abi, Bytes / sizeof(T)>,
                           detail::invalid_traits>;
    //}}}2
};

// __scalar_abi {{{1
struct __scalar_abi {
    template <class T> using size_tag = detail::size_constant<1>;
    struct is_valid_abi_tag : std::true_type {};
    template <class T> struct is_valid_size_for : std::true_type {};
    template <class T> struct is_valid : detail::is_vectorizable<T> {};
    template <class T> static constexpr bool is_valid_v = is_valid<T>::value;

    using simd_impl_type = detail::scalar_simd_impl;
    using mask_impl_type = detail::scalar_mask_impl;

    template <class T, bool = is_valid_v<T>> struct traits : detail::invalid_traits {
    };

    template <class T> struct traits<T, true> {
        using is_valid = std::true_type;
        using simd_impl_type = detail::scalar_simd_impl;
        using mask_impl_type = detail::scalar_mask_impl;
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

// __fixed_abi {{{1
template <int N> struct __fixed_abi {
    template <class T> using size_tag = detail::size_constant<N>;
    // validity traits {{{2
    struct is_valid_abi_tag
        : public detail::bool_constant<(N > 0)> {
    };
    template <class T>
    struct is_valid_size_for
        : detail::bool_constant<((N <= simd_abi::max_fixed_size<T>) ||
                                 (simd_abi::__neon::is_valid_v<char> &&
                                  N == simd_size_v<char, simd_abi::__neon>) ||
                                 (simd_abi::__sse::is_valid_v<char> &&
                                  N == simd_size_v<char, simd_abi::__sse>) ||
                                 (simd_abi::__avx::is_valid_v<char> &&
                                  N == simd_size_v<char, simd_abi::__avx>) ||
                                 (simd_abi::__avx512::is_valid_v<char> &&
                                  N == simd_size_v<char, simd_abi::__avx512>))> {
    };
    template <class T>
    struct is_valid
        : detail::all<is_valid_abi_tag, detail::is_vectorizable<T>, is_valid_size_for<T>> {
    };
    template <class T> static constexpr bool is_valid_v = is_valid<T>::value;

    // simd/mask_impl_type {{{2
    using simd_impl_type = detail::fixed_size_simd_impl<N>;
    using mask_impl_type = detail::fixed_size_mask_impl<N>;

    // traits {{{2
    template <class T, bool = is_valid_v<T>> struct traits : detail::invalid_traits {
    };

    template <class T> struct traits<T, true> {
        using is_valid = std::true_type;
        using simd_impl_type = detail::fixed_size_simd_impl<N>;
        using mask_impl_type = detail::fixed_size_mask_impl<N>;

        // simd and simd_mask member types {{{2
        using simd_member_type = detail::fixed_size_storage<T, N>;
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
                     detail::next_power_of_2(N * sizeof(T)));
        static constexpr size_t mask_member_alignment = alignof(mask_member_type);

        // simd_base / base class for simd, providing extra conversions {{{2
        struct simd_base {
            // The following ensures, function arguments are passed via the stack. This is
            // important for ABI compatibility across TU boundaries
            simd_base(const simd_base &) {}
            simd_base() = default;

            explicit operator const simd_member_type &() const
            {
                return static_cast<const simd<T, __fixed_abi> *>(this)->d;
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

//}}}
}  // namespace simd_abi
namespace detail
{
// scalar_abi_wrapper {{{1
template <int Bytes> struct scalar_abi_wrapper : simd_abi::__scalar_abi {
    template <class T>
    static constexpr bool is_valid_v = simd_abi::__scalar_abi::is_valid<T>::value &&
                                       sizeof(T) == Bytes;
};

// decay_abi metafunction {{{1
template <class T> struct decay_abi {
    using type = T;
};
template <int Bytes> struct decay_abi<scalar_abi_wrapper<Bytes>> {
    using type = simd_abi::scalar;
};

// full_abi metafunction {{{1
template <template <int> class ATemp> struct full_abi;
template <> struct full_abi<simd_abi::__neon_abi> { using type = simd_abi::__neon128; };
template <> struct full_abi<simd_abi::__sse_abi> { using type = simd_abi::__sse; };
template <> struct full_abi<simd_abi::__avx_abi> { using type = simd_abi::__avx; };
template <> struct full_abi<simd_abi::__avx512_abi> { using type = simd_abi::__avx512; };
template <> struct full_abi<scalar_abi_wrapper> {
    using type = simd_abi::scalar;
};

// abi_list {{{1
template <template <int> class...> struct abi_list {
    template <class, int> static constexpr bool has_valid_abi = false;
    template <class, int> using first_valid_abi = void;
    template <class, int> using best_abi = void;
};

template <template <int> class A0, template <int> class... Rest>
struct abi_list<A0, Rest...> {
    template <class T, int N>
    static constexpr bool has_valid_abi = A0<sizeof(T) * N>::template is_valid_v<T> ||
                                          abi_list<Rest...>::template has_valid_abi<T, N>;
    template <class T, int N>
    using first_valid_abi =
        std::conditional_t<A0<sizeof(T) * N>::template is_valid_v<T>,
                           typename decay_abi<A0<sizeof(T) * N>>::type,
                           typename abi_list<Rest...>::template first_valid_abi<T, N>>;
    using B = typename full_abi<A0>::type;
    template <class T, int N>
    using best_abi = std::conditional_t<
        A0<sizeof(T) * N>::template is_valid_v<T>,
        typename decay_abi<A0<sizeof(T) * N>>::type,
        std::conditional_t<(B::template is_valid_v<T> &&
                            B::template size_tag<T>::value <= N),
                           B, typename abi_list<Rest...>::template best_abi<T, N>>>;
};

// }}}1

// the following lists all native ABIs, which makes them accessible to simd_abi::deduce
// and select_best_vector_type_t (for fixed_size). Order matters: Whatever comes first has
// higher priority.
using all_native_abis =
    abi_list<simd_abi::__avx512_abi, simd_abi::__avx_abi, simd_abi::__sse_abi,
             simd_abi::__neon_abi, scalar_abi_wrapper>;

// valid traits specialization {{{1
template <class T, class Abi>
struct traits<T, Abi, void_t<typename Abi::template is_valid<T>>>
    : Abi::template traits<T> {
};

// deduce_impl specializations {{{1
// try all native ABIs (including scalar) first
template <class T, std::size_t N>
struct deduce_impl<T, N,
                   std::enable_if_t<all_native_abis::template has_valid_abi<T, N>>> {
    using type = all_native_abis::first_valid_abi<T, N>;
};

// fall back to fixed_size only if scalar and native ABIs don't match
template <class T, std::size_t N, class = void> struct deduce_fixed_size_fallback {};
template <class T, std::size_t N>
struct deduce_fixed_size_fallback<
    T, N, std::enable_if_t<simd_abi::fixed_size<N>::template is_valid_v<T>>> {
    using type = simd_abi::fixed_size<N>;
};
template <class T, std::size_t N, class>
struct deduce_impl : public deduce_fixed_size_fallback<T, N> {
};

//}}}1
// is_<abi> {{{
template <template <int> class Abi, int Bytes> constexpr int abi_bytes_impl(Abi<Bytes> *)
{
    return Bytes;
}
template <class T> constexpr int abi_bytes_impl(T *) { return -1; }
template <class Abi>
inline constexpr int abi_bytes = abi_bytes_impl(static_cast<Abi *>(nullptr));

template <class Abi0, class Abi1> constexpr bool is_abi()
{
    return std::is_same_v<Abi0, Abi1>;
}
template <template <int> class Abi0, class Abi1> constexpr bool is_abi()
{
    return std::is_same_v<Abi0<abi_bytes<Abi1>>, Abi1>;
}
template <class Abi0, template <int> class Abi1> constexpr bool is_abi()
{
    return std::is_same_v<Abi1<abi_bytes<Abi0>>, Abi0>;
}
template <template <int> class Abi0, template <int> class Abi1> constexpr bool is_abi()
{
    return std::is_same_v<Abi0<0>, Abi1<0>>;
}

// }}}
// is_combined_abi{{{
template <template <int, class> class Combine, int N, class Abi>
constexpr bool is_combined_abi(Combine<N, Abi> *)
{
    return std::is_same_v<Combine<N, Abi>, simd_abi::__combine<N, Abi>>;
}
template <class Abi> constexpr bool is_combined_abi(Abi *)
{
    return false;
}

template <class Abi> constexpr bool is_combined_abi()
{
    return is_combined_abi(static_cast<Abi *>(nullptr));
}

// }}}
}  // namespace detail
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_DETAIL_ABIS_H_

// vim: foldmethod=marker
