/*  This file is part of the Vc library. {{{
Copyright © 2016-2017 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_DATAPAR_SYNOPSIS_H_
#define VC_DATAPAR_SYNOPSIS_H_

#include "global.h"
#include "macros.h"
#include "declval.h"
#include "macros.h"
#include "detail.h"
#include "where.h"
#include "concepts.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace datapar_abi
{
constexpr int max_fixed_size = 32;
template <int N> struct fixed_size {};
struct scalar {};
struct sse {};
struct avx {};
struct avx512 {};
struct knc {};
struct neon {};

template <int N> struct partial_sse {};
template <int N> struct partial_avx {};
template <int N> struct partial_avx512 {};
template <int N> struct partial_knc {};

namespace detail
{
template <class T, class A0, class A1> struct fallback_abi_for_long_double {
    using type = A0;
};
template <class A0, class A1> struct fallback_abi_for_long_double<long double, A0, A1> {
    using type = A1;
};
template <class T, class A0, class A1>
using fallback_abi_for_long_double_t =
    typename fallback_abi_for_long_double<T, A0, A1>::type;
}  // namespace detail

#if defined Vc_IS_AMD64
#if !defined Vc_HAVE_SSE2
#error "Use of SSE2 is required on AMD64"
#endif
template <typename T>
using compatible = detail::fallback_abi_for_long_double_t<T, sse, scalar>;
#elif defined Vc_HAVE_FULL_KNC_ABI
template <typename T>
using compatible = detail::fallback_abi_for_long_double_t<T, knc, scalar>;
#elif defined Vc_IS_AARCH64
template <typename T>
using compatible = detail::fallback_abi_for_long_double_t<T, neon, scalar>;
#else
template <typename> using compatible = scalar;
#endif

#if defined Vc_HAVE_FULL_AVX512_ABI
template <typename T>
using native = detail::fallback_abi_for_long_double_t<T, avx512, scalar>;
#elif defined Vc_HAVE_AVX512_ABI
template <typename T>
using native =
    std::conditional_t<(sizeof(T) >= 4),
                       detail::fallback_abi_for_long_double_t<T, avx512, scalar>, avx>;
#elif defined Vc_HAVE_FULL_AVX_ABI
template <typename T> using native = detail::fallback_abi_for_long_double_t<T, avx, scalar>;
#elif defined Vc_HAVE_AVX_ABI
template <typename T>
using native =
    std::conditional_t<std::is_floating_point<T>::value,
                       detail::fallback_abi_for_long_double_t<T, avx, scalar>, sse>;
#elif defined Vc_HAVE_FULL_SSE_ABI
template <typename T>
using native = detail::fallback_abi_for_long_double_t<T, sse, scalar>;
#elif defined Vc_HAVE_SSE_ABI
template <typename T>
using native = std::conditional_t<std::is_same<float, T>::value, sse, scalar>;
#elif defined Vc_HAVE_FULL_KNC_ABI
template <typename T>
using native = detail::fallback_abi_for_long_double_t<T, knc, scalar>;
#elif defined Vc_HAVE_FULL_NEON_ABI
template <typename T>
using native = detail::fallback_abi_for_long_double_t<T, neon, scalar>;
#else
template <typename> using native = scalar;
#endif

namespace detail
{
#if defined Vc_DEFAULT_ABI
template <typename T> using default_abi = Vc_DEFAULT_ABI<T>;
#else
template <typename T> using default_abi = compatible<T>;
#endif
}  // namespace detail
}  // namespace datapar_abi

template <class T> struct is_datapar : public std::false_type {};
template <class T> constexpr bool is_datapar_v = is_datapar<T>::value;

template <class T> struct is_mask : public std::false_type {};
template <class T> constexpr bool is_mask_v = is_mask<T>::value;

template <class T, class Abi = datapar_abi::detail::default_abi<T>> struct datapar_size;
template <class T> struct datapar_size<T, datapar_abi::scalar> : public detail::size_constant<1> {};
template <class T> struct datapar_size<T, datapar_abi::sse   > : public detail::size_constant<16 / sizeof(T)> {};
template <class T> struct datapar_size<T, datapar_abi::avx   > : public detail::size_constant<32 / sizeof(T)> {};
template <class T> struct datapar_size<T, datapar_abi::avx512> : public detail::size_constant<64 / sizeof(T)> {};
template <class T> struct datapar_size<T, datapar_abi::neon  > : public detail::size_constant<16 / sizeof(T)> {};
template <class T, int N> struct datapar_size<T, datapar_abi::fixed_size<N>> : public detail::size_constant<N> {};
template <class T, class Abi = datapar_abi::detail::default_abi<T>>
constexpr size_t datapar_size_v = datapar_size<T, Abi>::value;

namespace detail
{
template <class T, size_t N, bool, bool> struct abi_for_size_impl;
template <class T, size_t N> struct abi_for_size_impl<T, N, true, true> {
    using type = datapar_abi::fixed_size<N>;
};
template <class T> struct abi_for_size_impl<T, 1, true, true> {
    using type = datapar_abi::scalar;
};
#ifdef Vc_HAVE_SSE_ABI
template <> struct abi_for_size_impl<float, 4, true, true> { using type = datapar_abi::sse; };
#endif
#ifdef Vc_HAVE_FULL_SSE_ABI
template <> struct abi_for_size_impl<double, 2, true, true> { using type = datapar_abi::sse; };
template <> struct abi_for_size_impl< llong, 2, true, true> { using type = datapar_abi::sse; };
template <> struct abi_for_size_impl<ullong, 2, true, true> { using type = datapar_abi::sse; };
template <> struct abi_for_size_impl<  long, 16 / sizeof(long), true, true> { using type = datapar_abi::sse; };
template <> struct abi_for_size_impl< ulong, 16 / sizeof(long), true, true> { using type = datapar_abi::sse; };
template <> struct abi_for_size_impl<   int, 4, true, true> { using type = datapar_abi::sse; };
template <> struct abi_for_size_impl<  uint, 4, true, true> { using type = datapar_abi::sse; };
template <> struct abi_for_size_impl< short, 8, true, true> { using type = datapar_abi::sse; };
template <> struct abi_for_size_impl<ushort, 8, true, true> { using type = datapar_abi::sse; };
template <> struct abi_for_size_impl< schar, 16, true, true> { using type = datapar_abi::sse; };
template <> struct abi_for_size_impl< uchar, 16, true, true> { using type = datapar_abi::sse; };
#endif
#ifdef Vc_HAVE_AVX_ABI
template <> struct abi_for_size_impl<double, 4, true, true> { using type = datapar_abi::avx; };
template <> struct abi_for_size_impl<float, 8, true, true> { using type = datapar_abi::avx; };
#endif
#ifdef Vc_HAVE_FULL_AVX_ABI
template <> struct abi_for_size_impl< llong,  4, true, true> { using type = datapar_abi::avx; };
template <> struct abi_for_size_impl<ullong,  4, true, true> { using type = datapar_abi::avx; };
template <> struct abi_for_size_impl<  long, 32 / sizeof(long), true, true> { using type = datapar_abi::avx; };
template <> struct abi_for_size_impl< ulong, 32 / sizeof(long), true, true> { using type = datapar_abi::avx; };
template <> struct abi_for_size_impl<   int,  8, true, true> { using type = datapar_abi::avx; };
template <> struct abi_for_size_impl<  uint,  8, true, true> { using type = datapar_abi::avx; };
template <> struct abi_for_size_impl< short, 16, true, true> { using type = datapar_abi::avx; };
template <> struct abi_for_size_impl<ushort, 16, true, true> { using type = datapar_abi::avx; };
template <> struct abi_for_size_impl< schar, 32, true, true> { using type = datapar_abi::avx; };
template <> struct abi_for_size_impl< uchar, 32, true, true> { using type = datapar_abi::avx; };
#endif
#ifdef Vc_HAVE_AVX512_ABI
template <> struct abi_for_size_impl<double, 8, true, true> { using type = datapar_abi::avx512; };
template <> struct abi_for_size_impl<float, 16, true, true> { using type = datapar_abi::avx512; };
template <> struct abi_for_size_impl< llong,  8, true, true> { using type = datapar_abi::avx512; };
template <> struct abi_for_size_impl<ullong,  8, true, true> { using type = datapar_abi::avx512; };
template <> struct abi_for_size_impl<  long, 64 / sizeof(long), true, true> { using type = datapar_abi::avx512; };
template <> struct abi_for_size_impl< ulong, 64 / sizeof(long), true, true> { using type = datapar_abi::avx512; };
template <> struct abi_for_size_impl<   int, 16, true, true> { using type = datapar_abi::avx512; };
template <> struct abi_for_size_impl<  uint, 16, true, true> { using type = datapar_abi::avx512; };
#endif
#ifdef Vc_HAVE_FULL_AVX512_ABI
template <> struct abi_for_size_impl< short, 32, true, true> { using type = datapar_abi::avx512; };
template <> struct abi_for_size_impl<ushort, 32, true, true> { using type = datapar_abi::avx512; };
template <> struct abi_for_size_impl< schar, 64, false, true> { using type = datapar_abi::avx512; };
template <> struct abi_for_size_impl< uchar, 64, false, true> { using type = datapar_abi::avx512; };
// fixed_size must support 64 entries because schar and uchar have 64 entries. Everything in
// between max_fixed_size and 64 doesn't have to be supported.
template <class T> struct abi_for_size_impl<T, 64, false, true> {
    using type = datapar_abi::fixed_size<64>;
};
#endif
#ifdef Vc_HAVE_FULL_KNC_ABI
template <class T> struct abi_for_size_impl<T, datapar_size_v<T, datapar_abi::knc>, true, true> {
    using type = datapar_abi::knc;
};
#endif
#ifdef Vc_HAVE_FULL_NEON_ABI
template <> struct abi_for_size_impl<double,  2, true, true> { using type = datapar_abi::neon; };
template <> struct abi_for_size_impl< float,  4, true, true> { using type = datapar_abi::neon; };
template <> struct abi_for_size_impl< llong,  2, true, true> { using type = datapar_abi::neon; };
template <> struct abi_for_size_impl<ullong,  2, true, true> { using type = datapar_abi::neon; };
template <> struct abi_for_size_impl<  long, 16 / sizeof(long), true, true> { using type = datapar_abi::neon; };
template <> struct abi_for_size_impl< ulong, 16 / sizeof(long), true, true> { using type = datapar_abi::neon; };
template <> struct abi_for_size_impl<   int,  4, true, true> { using type = datapar_abi::neon; };
template <> struct abi_for_size_impl<  uint,  4, true, true> { using type = datapar_abi::neon; };
template <> struct abi_for_size_impl< short,  8, true, true> { using type = datapar_abi::neon; };
template <> struct abi_for_size_impl<ushort,  8, true, true> { using type = datapar_abi::neon; };
template <> struct abi_for_size_impl< schar, 16, true, true> { using type = datapar_abi::neon; };
template <> struct abi_for_size_impl< uchar, 16, true, true> { using type = datapar_abi::neon; };
#endif
}  // namespace detail
template <class T, size_t N>
struct abi_for_size
    : public detail::abi_for_size_impl<T, N, (N <= datapar_abi::max_fixed_size),
                                       std::is_arithmetic<T>::value> {
};
template <size_t N> struct abi_for_size<bool, N> {
};
template <class T> struct abi_for_size<T, 0> {
};
template <class T, size_t N> using abi_for_size_t = typename abi_for_size<T, N>::type;

template <class T, class U = typename T::value_type>
struct memory_alignment
    : public detail::size_constant<detail::next_power_of_2(sizeof(U) * T::size())> {
};
template <class T, class U = typename T::value_type>
constexpr size_t memory_alignment_v = memory_alignment<T, U>::value;

// class template datapar [datapar]
template <class T, class Abi = datapar_abi::detail::default_abi<T>> class datapar;
template <class T, class Abi> struct is_datapar<datapar<T, Abi>> : public std::true_type {};
template <class T> using native_datapar = datapar<T, datapar_abi::native<T>>;
template <class T, int N> using fixed_size_datapar = datapar<T, datapar_abi::fixed_size<N>>;

// class template mask [mask]
template <class T, class Abi = datapar_abi::detail::default_abi<T>> class mask;
template <class T, class Abi> struct is_mask<mask<T, Abi>> : public std::true_type {};
template <class T> using native_mask = mask<T, datapar_abi::native<T>>;
template <class T, int N> using fixed_size_mask = mask<T, datapar_abi::fixed_size<N>>;

namespace detail
{
template <class T, class Abi> struct get_impl<Vc::mask<T, Abi>> {
    using type = typename traits<T, Abi>::mask_impl_type;
};
template <class T, class Abi> struct get_impl<Vc::datapar<T, Abi>> {
    using type = typename traits<T, Abi>::datapar_impl_type;
};
}  // namespace detail

// casts [datapar.casts]
template <class T, class U, class A>
Vc_INTRINSIC auto
static_datapar_cast(const datapar<U, A> &x)
{
    return datapar<
        T, std::conditional_t<
               detail::any<std::is_same<T, U>,
                           detail::all<std::is_integral<T>, std::is_integral<U>,
                                       std::is_same<std::make_unsigned_t<T>,
                                                    std::make_unsigned_t<U>>>>::value,
               A, datapar_abi::fixed_size<datapar<U, A>::size()>>>(
        [&x](auto i) { return static_cast<T>(x[i]); });
}

template <class T, int N>
Vc_INTRINSIC fixed_size_datapar<T, N> to_fixed_size(const fixed_size_datapar<T, N> &x)
{
    return x;
}

template <class T, int N>
Vc_INTRINSIC fixed_size_mask<T, N> to_fixed_size(const fixed_size_mask<T, N> &x)
{
    return x;
}

template <class T, class A> Vc_INTRINSIC auto to_fixed_size(const datapar<T, A> &x)
{
    return datapar<T, datapar_abi::fixed_size<datapar_size_v<T, A>>>(
        [&x](auto i) { return x[i]; });
}

template <class T, class A> Vc_INTRINSIC auto to_fixed_size(const mask<T, A> &x)
{
    constexpr int N = mask<T, A>::size();
    fixed_size_mask<T, N> r;
    detail::execute_n_times<N>([&](auto i) { r[i] = x[i]; });
    return r;
}

template <class T, int N>
Vc_INTRINSIC std::enable_if_t<(N == native_datapar<T>::size()), native_datapar<T>>
to_native(const fixed_size_datapar<T, N> &x)
{
    alignas(memory_alignment_v<native_datapar<T>>) T mem[N];
    x.memstore(mem, flags::vector_aligned);
    return {mem, flags::vector_aligned};
}

template <class T, size_t N>
Vc_INTRINSIC std::enable_if_t<(N == native_mask<T>::size()), native_mask<T>> to_native(
    const fixed_size_mask<T, N> &x)
{
    return native_mask<T>([&](auto i) { return x[i]; });
}

template <class T, size_t N>
Vc_INTRINSIC std::enable_if_t<(N == datapar<T>::size()), datapar<T>> to_compatible(
    const datapar<T, datapar_abi::fixed_size<N>> &x)
{
    alignas(memory_alignment_v<datapar<T>>) T mem[N];
    x.memstore(mem, flags::vector_aligned);
    return {mem, flags::vector_aligned};
}

template <class T, size_t N>
Vc_INTRINSIC std::enable_if_t<(N == mask<T>::size()), mask<T>> to_compatible(
    const mask<T, datapar_abi::fixed_size<N>> &x)
{
    return mask<T>([&](auto i) { return x[i]; });
}

#if defined __cpp_fold_expressions && defined Vc_EXPERIMENTAL
template <size_t... Sizes, class T, class A>
std::enable_if_t<((Sizes + ...) == datapar<T, A>::size()),
                 std::tuple<datapar<T, abi_for_size_t<T, Sizes>>...>>
split(const datapar<T, A> &x)
{
    std::tuple<datapar<T, abi_for_size_t<T, Sizes>>...> tup;
    size_t offset = 0;
    detail::execute_n_times<sizeof...(Sizes)>([&](auto i) {
        auto &v_i = std::get<i>(tup);
        constexpr size_t N = std::decay_t<decltype(v_i)>::size();
        detail::execute_n_times<N>([&](auto j) { v_i[j] = x[j + offset]; });
        offset += N;
    });
    return tup;
}

template <class V, class T, class A>
std::enable_if_t<(is_datapar<V>::value && datapar_size_v<T, A> % V::size() == 0),
                 std::array<V, datapar_size_v<T, A> / V::size()>>
split(const datapar<T, A> &x)
{
    constexpr size_t Parts = datapar_size_v<T, A> / V::size();
    std::array<V, Parts> r;
    size_t offset = 0;
    detail::execute_n_times<Parts>([&](auto i) {
        detail::execute_n_times<V::size()>([&](auto j) { r[i][j] = x[j + offset]; });
        offset += V::size();
    });
    return r;
}

namespace detail
{
template <class T, class...> struct typelist
{
    using first_type = T;
};

template <size_t N, class T, class List,
          bool = (N < datapar_size_v<T, typename List::first_type>)>
struct subscript_in_pack;

template <size_t N, class T, class A, class... As>
struct subscript_in_pack<N, T, detail::typelist<A, As...>, true> {
    static Vc_INTRINSIC T get(const datapar<T, A> &x, const datapar<T, As> &...)
    {
        return x[N];
    }
};
template <size_t N, class T, class A, class... As>
struct subscript_in_pack<N, T, detail::typelist<A, As...>, false> {
    static Vc_INTRINSIC T get(const datapar<T, A> &, const datapar<T, As> &... xs)
    {
        return subscript_in_pack<N - datapar<T, A>::size(), T,
                                 detail::typelist<As...>>::get(xs...);
    }
};
}  // namespace detail

template <class T, class... As>
datapar<T, abi_for_size_t<T, (datapar_size_v<T, As> + ...)>> concat(
    const datapar<T, As> &... xs)
{
    return datapar<T, abi_for_size_t<T, (datapar_size_v<T, As> + ...)>>([&](auto i) {
        return detail::subscript_in_pack<i, T, detail::typelist<As...>>::get(xs...);
    });
}
#endif  // defined __cpp_fold_expressions && defined Vc_EXPERIMENTAL

// reductions [mask.reductions]
// implementation per ABI in fixed_size.h, sse.h, avx.h, etc.
template <class T, class Abi> inline bool all_of(const mask<T, Abi> &k);
template <class T, class Abi> inline bool any_of(const mask<T, Abi> &k);
template <class T, class Abi> inline bool none_of(const mask<T, Abi> &k);
template <class T, class Abi> inline bool some_of(const mask<T, Abi> &k);
template <class T, class Abi> inline int popcount(const mask<T, Abi> &k);
template <class T, class Abi> inline int find_first_set(const mask<T, Abi> &k);
template <class T, class Abi> inline int find_last_set(const mask<T, Abi> &k);

constexpr bool all_of(detail::exact_bool x) { return x; }
constexpr bool any_of(detail::exact_bool x) { return x; }
constexpr bool none_of(detail::exact_bool x) { return !x; }
constexpr bool some_of(detail::exact_bool) { return false; }
constexpr int popcount(detail::exact_bool x) { return x; }
constexpr int find_first_set(detail::exact_bool) { return 0; }
constexpr int find_last_set(detail::exact_bool) { return 0; }

// masked assignment [mask.where]
#ifdef Vc_EXPERIMENTAL
namespace detail {
template <class T, class A> class masked_datapar_impl;
template <class T, class A>
masked_datapar_impl<T, A> masked_datapar(const typename datapar<T, A>::mask_type &k,
                                         datapar<T, A> &v);
}  // namespace detail
#endif  // Vc_EXPERIMENTAL

template <typename M, typename T> class const_where_expression
{
    using V = std::remove_const_t<T>;
    struct Wrapper {
        using value_type = V;
    };

protected:
    using value_type =
        typename std::conditional_t<std::is_arithmetic<V>::value, Wrapper, V>::value_type;
    friend Vc_INTRINSIC const M &get_mask(const const_where_expression &x) { return x.k; }
    friend Vc_INTRINSIC T &get_lvalue(const_where_expression &x) { return x.d; }
    friend Vc_INTRINSIC const T &get_lvalue(const const_where_expression &x) { return x.d; }
    const M &k;
    T &d;

public:
    const_where_expression(const const_where_expression &) = delete;
    const_where_expression &operator=(const const_where_expression &) = delete;

    Vc_INTRINSIC const_where_expression(const M &kk, T &dd) : k(kk), d(dd) {}

    Vc_INTRINSIC V operator-() const &&
    {
        return V(detail::get_impl_t<V>::template masked_unary<std::negate>(
            detail::data(k), detail::data(d)));
    }

    template <class U, class Flags>
    Vc_NODISCARD Vc_INTRINSIC V
    memload(const detail::loadstore_ptr_type<U, value_type> *mem, Flags f) const &&
    {
        V r = d;
        detail::get_impl_t<V>::masked_load(detail::data(r), detail::data(k), mem, f);
        return r;
    }

    template <class U, class Flags>
    Vc_INTRINSIC void memstore(detail::loadstore_ptr_type<U, value_type> *mem,
                               Flags f) const &&
    {
        detail::get_impl_t<V>::masked_store(detail::data(d), mem, f, detail::data(k));
    }
};

template <typename T> class const_where_expression<bool, T>
{
    using M = bool;
    using V = std::remove_const_t<T>;
    struct Wrapper {
        using value_type = V;
    };

protected:
    using value_type =
        typename std::conditional_t<std::is_arithmetic<V>::value, Wrapper, V>::value_type;
    friend Vc_INTRINSIC const M &get_mask(const const_where_expression &x) { return x.k; }
    friend Vc_INTRINSIC T &get_lvalue(const_where_expression &x) { return x.d; }
    friend Vc_INTRINSIC const T &get_lvalue(const const_where_expression &x) { return x.d; }
    const bool k;
    T &d;

public:
    const_where_expression(const const_where_expression &) = delete;
    const_where_expression &operator=(const const_where_expression &) = delete;

    Vc_INTRINSIC const_where_expression(const bool kk, T &dd) : k(kk), d(dd) {}

    Vc_INTRINSIC V operator-() const && { return k ? -d : d; }

    template <class U, class Flags>
    Vc_NODISCARD Vc_INTRINSIC V
    memload(const detail::loadstore_ptr_type<U, value_type> *mem, Flags) const &&
    {
        return k ? static_cast<V>(mem[0]) : d;
    }

    template <class U, class Flags>
    Vc_INTRINSIC void memstore(detail::loadstore_ptr_type<U, value_type> *mem,
                               Flags) const &&
    {
        if (k) {
            mem[0] = d;
        }
    }
};

template <typename M, typename T>
class where_expression : public const_where_expression<M, T>
{
    static_assert(!std::is_const<T>::value, "where_expression may only be instantiated with a non-const T parameter");
    using typename const_where_expression<M, T>::value_type;
    using const_where_expression<M, T>::k;
    using const_where_expression<M, T>::d;
    static_assert(std::is_same<typename M::abi_type, typename T::abi_type>::value, "");
    static_assert(M::size() == T::size(), "");

public:
    where_expression(const where_expression &) = delete;
    where_expression &operator=(const where_expression &) = delete;

    Vc_INTRINSIC where_expression(const M &kk, T &dd)
        : const_where_expression<M, T>(kk, dd)
    {
    }

    template <class U> Vc_INTRINSIC void operator=(U &&x)
    {
        Vc::detail::get_impl_t<T>::masked_assign(
            detail::data(k), detail::data(d),
            detail::to_value_type_or_member_type<T>(std::forward<U>(x)));
    }

#define Vc_OP_(op_, name_)                                                               \
    template <class U> Vc_INTRINSIC void operator op_##=(U &&x)                          \
    {                                                                                    \
        Vc::detail::get_impl_t<T>::template masked_cassign<name_>(                       \
            detail::data(k), detail::data(d),                                            \
            detail::to_value_type_or_member_type<T>(std::forward<U>(x)));                \
    }                                                                                    \
    Vc_NOTHING_EXPECTING_SEMICOLON
    Vc_OP_(+, std::plus);
    Vc_OP_(-, std::minus);
    Vc_OP_(*, std::multiplies);
    Vc_OP_(/, std::divides);
    Vc_OP_(%, std::modulus);
    Vc_OP_(&, std::bit_and);
    Vc_OP_(|, std::bit_or);
    Vc_OP_(^, std::bit_xor);
    Vc_OP_(<<, detail::shift_left);
    Vc_OP_(>>, detail::shift_right);
#undef Vc_OP_

    Vc_INTRINSIC void operator++()
    {
        detail::data(d) = detail::get_impl_t<T>::template masked_unary<detail::increment>(
            detail::data(k), detail::data(d));
    }
    Vc_INTRINSIC void operator++(int)
    {
        detail::data(d) = detail::get_impl_t<T>::template masked_unary<detail::increment>(
            detail::data(k), detail::data(d));
    }
    Vc_INTRINSIC void operator--()
    {
        detail::data(d) = detail::get_impl_t<T>::template masked_unary<detail::decrement>(
            detail::data(k), detail::data(d));
    }
    Vc_INTRINSIC void operator--(int)
    {
        detail::data(d) = detail::get_impl_t<T>::template masked_unary<detail::decrement>(
            detail::data(k), detail::data(d));
    }

    // intentionally hides const_where_expression::memload
    template <class U, class Flags>
    Vc_INTRINSIC void memload(const detail::loadstore_ptr_type<U, value_type> *mem,
                              Flags f)
    {
        detail::get_impl_t<T>::masked_load(detail::data(d), detail::data(k), mem, f);
    }

#ifdef Vc_EXPERIMENTAL
    template <class F>
    Vc_INTRINSIC std::enable_if_t<
        detail::all<std::is_same<decltype(declval<F>()(detail::masked_datapar(
                                     declval<const M &>(), declval<T &>()))),
                                 void>>::value,
        where_expression &&>
    apply(F &&f) &&
    {
        std::forward<F>(f)(detail::masked_datapar(k, d));
        return std::move(*this);
    }

    template <class F>
    Vc_INTRINSIC std::enable_if_t<
        detail::all<std::is_same<decltype(declval<F>()(detail::masked_datapar(
                                     declval<const M &>(), declval<T &>()))),
                                 void>>::value,
        where_expression &&>
    apply_inv(F &&f) &&
    {
        std::forward<F>(f)(detail::masked_datapar(!k, d));
        return std::move(*this);
    }
#endif  // Vc_EXPERIMENTAL
};

template <typename T>
class where_expression<bool, T> : public const_where_expression<bool, T>
{
    using M = bool;
    using typename const_where_expression<M, T>::value_type;
    using const_where_expression<M, T>::k;
    using const_where_expression<M, T>::d;

public:
    where_expression(const where_expression &) = delete;
    where_expression &operator=(const where_expression &) = delete;

    Vc_INTRINSIC where_expression(const M &kk, T &dd)
        : const_where_expression<M, T>(kk, dd)
    {
    }

#define Vc_OP_(op_)                                                                      \
    template <class U> Vc_INTRINSIC void operator op_(U &&x)                             \
    {                                                                                    \
        if (k) {                                                                         \
            d op_ std::forward<U>(x);                                                    \
        }                                                                                \
    }                                                                                    \
    Vc_NOTHING_EXPECTING_SEMICOLON
    Vc_OP_(=);
    Vc_OP_(+=);
    Vc_OP_(-=);
    Vc_OP_(*=);
    Vc_OP_(/=);
    Vc_OP_(%=);
    Vc_OP_(&=);
    Vc_OP_(|=);
    Vc_OP_(^=);
    Vc_OP_(<<=);
    Vc_OP_(>>=);
#undef Vc_OP_
    Vc_INTRINSIC void operator++()    { if (k) { ++d; } }
    Vc_INTRINSIC void operator++(int) { if (k) { ++d; } }
    Vc_INTRINSIC void operator--()    { if (k) { --d; } }
    Vc_INTRINSIC void operator--(int) { if (k) { --d; } }

    // intentionally hides const_where_expression::memload
    template <class U, class Flags>
    Vc_INTRINSIC void memload(const detail::loadstore_ptr_type<U, value_type> *mem, Flags)
    {
        if (k) {
            d = mem[0];
        }
    }
};

#ifdef Vc_EXPERIMENTAL
template <typename M, typename... Ts> class where_expression<M, std::tuple<Ts &...>>
{
    const M &k;
    std::tuple<Ts &...> d;

public:
    where_expression(const where_expression &) = delete;
    where_expression &operator=(const where_expression &) = delete;

    Vc_INTRINSIC where_expression(const M &kk, std::tuple<Ts &...> &&dd) : k(kk), d(dd) {}

private:
    template <class F, std::size_t... Is>
    Vc_INTRINSIC void apply_helper(F &&f, const M &mask, std::index_sequence<Is...>)
    {
        return std::forward<F>(f)(detail::masked_datapar(mask, std::get<Is>(d))...);
    }

public:
    template <class F>
    Vc_INTRINSIC std::enable_if_t<
        detail::all<std::is_same<decltype(declval<F>()(detail::masked_datapar(
                                     declval<const M &>(), declval<Ts &>())...)),
                                 void>>::value,
        where_expression &&>
    apply(F &&f) &&
    {
        apply_helper(std::forward<F>(f), k, std::make_index_sequence<sizeof...(Ts)>());
        return std::move(*this);
    }

    template <class F>
    Vc_INTRINSIC std::enable_if_t<
        detail::all<std::is_same<decltype(declval<F>()(detail::masked_datapar(
                                     declval<const M &>(), declval<Ts &>())...)),
                                 void>>::value,
        where_expression &&>
    apply_inv(F &&f) &&
    {
        apply_helper(std::forward<F>(f), !k, std::make_index_sequence<sizeof...(Ts)>());
        return std::move(*this);
    }
};

template <class T, class A, class... Vs>
Vc_INTRINSIC where_expression<mask<T, A>, std::tuple<datapar<T, A> &, Vs &...>> where(
    const typename datapar<T, A>::mask_type &k, datapar<T, A> &v0, Vs &... vs)
{
    return {k, {v0, vs...}};
}
#endif  // Vc_EXPERIMENTAL

template <class T, class A>
Vc_INTRINSIC where_expression<mask<T, A>, datapar<T, A>> where(
    const typename datapar<T, A>::mask_type &k, datapar<T, A> &d)
{
    return {k, d};
}
template <class T, class A>
Vc_INTRINSIC const_where_expression<mask<T, A>, const datapar<T, A>> where(
    const typename datapar<T, A>::mask_type &k, const datapar<T, A> &d)
{
    return {k, d};
}
template <class T, class A>
Vc_INTRINSIC where_expression<mask<T, A>, mask<T, A>> where(
    const std::remove_const_t<mask<T, A>> &k, mask<T, A> &d)
{
    return {k, d};
}
template <class T, class A>
Vc_INTRINSIC const_where_expression<mask<T, A>, const mask<T, A>> where(
    const std::remove_const_t<mask<T, A>> &k, const mask<T, A> &d)
{
    return {k, d};
}
template <class T>
Vc_INTRINSIC where_expression<bool, T> where(detail::exact_bool k, T &d)
{
    return {k, d};
}
template <class T>
Vc_INTRINSIC const_where_expression<bool, const T> where(detail::exact_bool k, const T &d)
{
    return {k, d};
}
template <class T, class A> void where(bool k, datapar<T, A> &d) = delete;
template <class T, class A> void where(bool k, const datapar<T, A> &d) = delete;

// reductions [datapar.reductions]
template <class BinaryOperation = std::plus<>, class T, class Abi>
Vc_INTRINSIC T reduce(const datapar<T, Abi> &v,
                      BinaryOperation binary_op = BinaryOperation())
{
    using V = datapar<T, Abi>;
    return detail::get_impl_t<V>::reduce(detail::size_tag<V::size()>, v, binary_op);
}
template <class BinaryOperation = std::plus<>, class M, class V>
Vc_INTRINSIC typename V::value_type reduce(
    const const_where_expression<M, V> &x,
    typename V::value_type neutral_element =
        detail::default_neutral_element<typename V::value_type, BinaryOperation>::value,
    BinaryOperation binary_op = BinaryOperation())
{
    using VV = std::remove_cv_t<V>;
    VV tmp = neutral_element;
    detail::get_impl_t<VV>::masked_assign(detail::data(get_mask(x)), detail::data(tmp),
                                          detail::data(get_lvalue(x)));
    return reduce(tmp, binary_op);
}

// algorithms [datapar.alg]
template <class T, class A>
Vc_INTRINSIC datapar<T, A> min(const datapar<T, A> &a, const datapar<T, A> &b)
{
    return detail::get_impl_t<datapar<T, A>>::min(a, b);
}
template <class T, class A>
Vc_INTRINSIC datapar<T, A> max(const datapar<T, A> &a, const datapar<T, A> &b)
{
    return detail::get_impl_t<datapar<T, A>>::max(a, b);
}
template <class T, class A>
Vc_INTRINSIC std::pair<datapar<T, A>, datapar<T, A>> minmax(const datapar<T, A> &a,
                                                            const datapar<T, A> &b)
{
    return detail::get_impl_t<datapar<T, A>>::minmax(a, b);
}
template <class T, class A>
Vc_INTRINSIC datapar<T, A> clamp(const datapar<T, A> &v, const datapar<T, A> &lo,
                                 const datapar<T, A> &hi)
{
    return min(hi, max(lo, v));
}

// math functions
template <class T, class Abi>
Vc_INTRINSIC datapar<T, Abi> sqrt(const datapar<T, Abi> &x)
{
    return static_cast<datapar<T, Abi>>(
        detail::get_impl_t<datapar<T, Abi>>::sqrt(data(x)));
}

template <class T, class Abi>
Vc_INTRINSIC datapar<T, Abi> abs(const datapar<detail::SignedArithmetic<T>, Abi> &x)
{
    return static_cast<datapar<T, Abi>>(
        detail::get_impl_t<datapar<T, Abi>>::abs(data(x)));
}

Vc_VERSIONED_NAMESPACE_END

#endif  // VC_DATAPAR_SYNOPSIS_H_

// vim: foldmethod=marker
