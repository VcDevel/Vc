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

#ifndef VC_DATAPAR_SYNOPSIS_H_
#define VC_DATAPAR_SYNOPSIS_H_

#include "../common/macros.h"
#include "../common/declval.h"
#include "macros.h"
#include "detail.h"
#include "where.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace datapar_abi
{
constexpr int max_fixed_size = 32;
template <int N> struct fixed_size {};
using scalar = fixed_size<1>;
struct sse {};
struct avx {};
struct avx512 {};
struct knc {};

template <int N> struct partial_sse {};
template <int N> struct partial_avx {};
template <int N> struct partial_avx512 {};
template <int N> struct partial_knc {};

#if defined Vc_IS_AMD64
template <typename T>
using compatible = std::conditional_t<(sizeof(T) <= 8), sse, scalar>;
#elif defined Vc_HAVE_FULL_KNC_ABI
template <typename T>
using compatible = std::conditional_t<(sizeof(T) <= 8), knc, scalar>;
#else
template <typename> using compatible = scalar;
#endif

#if defined Vc_HAVE_FULL_AVX512_ABI
template <typename T> using native = std::conditional_t<(sizeof(T) <= 8), avx512, scalar>;
#elif defined Vc_HAVE_AVX512_ABI
template <typename T>
using native =
    std::conditional_t<(sizeof(T) >= 4),
                       std::conditional_t<(sizeof(T) > 8), scalar, avx512>, avx>;
#elif defined Vc_HAVE_FULL_AVX_ABI
template <typename T> using native = std::conditional_t<(sizeof(T) > 8), scalar, avx>;
#elif defined Vc_HAVE_AVX_ABI
template <typename T>
using native = std::conditional_t<std::is_floating_point<T>::value,
                                  std::conditional_t<(sizeof(T) > 8), scalar, avx>, sse>;
#elif defined Vc_HAVE_FULL_SSE_ABI
template <typename T> using native = std::conditional_t<(sizeof(T) > 8), scalar, sse>;
#elif defined Vc_HAVE_SSE_ABI
template <typename T>
using native = std::conditional_t<std::is_same<float, T>::value, sse, scalar>;
#elif defined Vc_HAVE_FULL_KNC_ABI
template <typename T> using native = std::conditional_t<(sizeof(T) > 8), scalar, knc>;
#else
template <typename> using native = scalar;
#endif
}  // namespace datapar_abi

template <class T> struct is_datapar : public std::false_type {};
template <class T> constexpr bool is_datapar_v = is_datapar<T>::value;

template <class T> struct is_mask : public std::false_type {};
template <class T> constexpr bool is_mask_v = is_mask<T>::value;

template <class T, class Abi = datapar_abi::compatible<T>>
struct datapar_size
    : public std::integral_constant<size_t, detail::traits<T, Abi>::size()> {
};
template <class T, class Abi = datapar_abi::compatible<T>>
constexpr size_t datapar_size_v = datapar_size<T, Abi>::value;

namespace detail
{
template <class T, size_t N, bool, class> struct abi_for_size_impl;
template <class T, size_t N> struct abi_for_size_impl<T, N, true, std::true_type> {
    using type = datapar_abi::fixed_size<N>;
};
template <class T> struct abi_for_size_impl<T, 1, true, std::true_type> {
    using type = datapar_abi::scalar;
};
#ifdef Vc_HAVE_SSE_ABI
template <> struct abi_for_size_impl<float, 4, true, std::true_type> { using type = datapar_abi::sse; };
#endif
#ifdef Vc_HAVE_FULL_SSE_ABI
template <> struct abi_for_size_impl<double, 2, true, std::true_type> { using type = datapar_abi::sse; };
template <> struct abi_for_size_impl<std:: int64_t, 2, true, std::true_type> { using type = datapar_abi::sse; };
template <> struct abi_for_size_impl<std::uint64_t, 2, true, std::true_type> { using type = datapar_abi::sse; };
template <> struct abi_for_size_impl<std:: int32_t, 4, true, std::true_type> { using type = datapar_abi::sse; };
template <> struct abi_for_size_impl<std::uint32_t, 4, true, std::true_type> { using type = datapar_abi::sse; };
template <> struct abi_for_size_impl<std:: int16_t, 8, true, std::true_type> { using type = datapar_abi::sse; };
template <> struct abi_for_size_impl<std::uint16_t, 8, true, std::true_type> { using type = datapar_abi::sse; };
template <> struct abi_for_size_impl<std:: int8_t, 16, true, std::true_type> { using type = datapar_abi::sse; };
template <> struct abi_for_size_impl<std::uint8_t, 16, true, std::true_type> { using type = datapar_abi::sse; };
#endif
#ifdef Vc_HAVE_AVX_ABI
template <> struct abi_for_size_impl<double, 4, true, std::true_type> { using type = datapar_abi::avx; };
template <> struct abi_for_size_impl<float, 8, true, std::true_type> { using type = datapar_abi::avx; };
#endif
#ifdef Vc_HAVE_FULL_AVX_ABI
template <> struct abi_for_size_impl<std:: int64_t, 4, true, std::true_type> { using type = datapar_abi::avx; };
template <> struct abi_for_size_impl<std::uint64_t, 4, true, std::true_type> { using type = datapar_abi::avx; };
template <> struct abi_for_size_impl<std:: int32_t, 8, true, std::true_type> { using type = datapar_abi::avx; };
template <> struct abi_for_size_impl<std::uint32_t, 8, true, std::true_type> { using type = datapar_abi::avx; };
template <> struct abi_for_size_impl<std:: int16_t, 16, true, std::true_type> { using type = datapar_abi::avx; };
template <> struct abi_for_size_impl<std::uint16_t, 16, true, std::true_type> { using type = datapar_abi::avx; };
template <> struct abi_for_size_impl<std:: int8_t, 32, true, std::true_type> { using type = datapar_abi::avx; };
template <> struct abi_for_size_impl<std::uint8_t, 32, true, std::true_type> { using type = datapar_abi::avx; };
#endif
#ifdef Vc_HAVE_AVX512_ABI
template <> struct abi_for_size_impl<double, 8, true, std::true_type> { using type = datapar_abi::avx512; };
template <> struct abi_for_size_impl<float, 16, true, std::true_type> { using type = datapar_abi::avx512; };
template <> struct abi_for_size_impl<std:: int64_t, 8, true, std::true_type> { using type = datapar_abi::avx512; };
template <> struct abi_for_size_impl<std::uint64_t, 8, true, std::true_type> { using type = datapar_abi::avx512; };
template <> struct abi_for_size_impl<std:: int32_t, 16, true, std::true_type> { using type = datapar_abi::avx512; };
template <> struct abi_for_size_impl<std::uint32_t, 16, true, std::true_type> { using type = datapar_abi::avx512; };
#endif
#ifdef Vc_HAVE_FULL_AVX512_ABI
template <> struct abi_for_size_impl<std:: int16_t, 32, true, std::true_type> { using type = datapar_abi::avx512; };
template <> struct abi_for_size_impl<std::uint16_t, 32, true, std::true_type> { using type = datapar_abi::avx512; };
template <> struct abi_for_size_impl<std:: int8_t, 64, true, std::true_type> { using type = datapar_abi::avx512; };
template <> struct abi_for_size_impl<std::uint8_t, 64, true, std::true_type> { using type = datapar_abi::avx512; };
#endif
#ifdef Vc_HAVE_FULL_KNC_ABI
template <class T> struct abi_for_size_impl<T, datapar_size_v<T, datapar_abi::knc>, true, std::true_type> {
    using type = datapar_abi::knc;
};
#endif
}  // namespace detail
template <class T, size_t N>
struct abi_for_size
    : public detail::abi_for_size_impl<T, N, (N <= datapar_abi::max_fixed_size),
                                       std::is_arithmetic<T>> {
};
template <size_t N> struct abi_for_size<bool, N> {
};
template <class T> struct abi_for_size<T, 0> {
};

template <class T, class U = typename T::value_type>
constexpr size_t memory_alignment = detail::next_power_of_2(sizeof(U) * T::size());

// class template datapar [datapar]
template <class T, class Abi = datapar_abi::compatible<T>> class datapar;
template <class T, class Abi> struct is_datapar<datapar<T, Abi>> : public std::true_type {};
template <class T> using native_datapar = datapar<T, datapar_abi::native<T>>;

// class template mask [mask]
template <class T, class Abi = datapar_abi::compatible<T>> class mask;
template <class T, class Abi> struct is_mask<mask<T, Abi>> : public std::true_type {};
template <class T> using native_mask = mask<T, datapar_abi::native<T>>;

namespace detail
{
template <class T, class Abi> struct get_impl<Vc::mask<T, Abi>> {
    using type = typename traits<T, Abi>::mask_impl_type;
};
template <class T, class Abi> struct get_impl<Vc::datapar<T, Abi>> {
    using type = typename traits<T, Abi>::datapar_impl_type;
};
}  // namespace detail

// compound assignment [datapar.cassign]
template <class T, class Abi, class U> datapar<T, Abi> &operator +=(datapar<T, Abi> &, const U &);
template <class T, class Abi, class U> datapar<T, Abi> &operator -=(datapar<T, Abi> &, const U &);
template <class T, class Abi, class U> datapar<T, Abi> &operator *=(datapar<T, Abi> &, const U &);
template <class T, class Abi, class U> datapar<T, Abi> &operator /=(datapar<T, Abi> &, const U &);
template <class T, class Abi, class U> datapar<T, Abi> &operator %=(datapar<T, Abi> &, const U &);
template <class T, class Abi, class U> datapar<T, Abi> &operator &=(datapar<T, Abi> &, const U &);
template <class T, class Abi, class U> datapar<T, Abi> &operator |=(datapar<T, Abi> &, const U &);
template <class T, class Abi, class U> datapar<T, Abi> &operator ^=(datapar<T, Abi> &, const U &);
template <class T, class Abi, class U> datapar<T, Abi> &operator<<=(datapar<T, Abi> &, const U &);
template <class T, class Abi, class U> datapar<T, Abi> &operator>>=(datapar<T, Abi> &, const U &);

// binary operators [datapar.binary]
template <class T, class Abi, class U>
inline detail::return_type<datapar<T, Abi>, U> operator+ (datapar<T, Abi>, const U &);
template <class T, class Abi, class U>
inline detail::return_type<datapar<T, Abi>, U> operator- (datapar<T, Abi>, const U &);
template <class T, class Abi, class U>
inline detail::return_type<datapar<T, Abi>, U> operator* (datapar<T, Abi>, const U &);
template <class T, class Abi, class U>
inline detail::return_type<datapar<T, Abi>, U> operator/ (datapar<T, Abi>, const U &);
template <class T, class Abi, class U>
inline detail::return_type<datapar<T, Abi>, U> operator% (datapar<T, Abi>, const U &);
template <class T, class Abi, class U>
inline detail::return_type<datapar<T, Abi>, U> operator& (datapar<T, Abi>, const U &);
template <class T, class Abi, class U>
inline detail::return_type<datapar<T, Abi>, U> operator| (datapar<T, Abi>, const U &);
template <class T, class Abi, class U>
inline detail::return_type<datapar<T, Abi>, U> operator^ (datapar<T, Abi>, const U &);
template <class T, class Abi, class U>
inline detail::return_type<datapar<T, Abi>, U> operator<<(datapar<T, Abi>, const U &);
template <class T, class Abi, class U>
inline detail::return_type<datapar<T, Abi>, U> operator>>(datapar<T, Abi>, const U &);
template <class T, class Abi, class U>
inline detail::return_type<datapar<T, Abi>, U> operator+ (const U &, datapar<T, Abi>);
template <class T, class Abi, class U>
inline detail::return_type<datapar<T, Abi>, U> operator- (const U &, datapar<T, Abi>);
template <class T, class Abi, class U>
inline detail::return_type<datapar<T, Abi>, U> operator* (const U &, datapar<T, Abi>);
template <class T, class Abi, class U>
inline detail::return_type<datapar<T, Abi>, U> operator/ (const U &, datapar<T, Abi>);
template <class T, class Abi, class U>
inline detail::return_type<datapar<T, Abi>, U> operator% (const U &, datapar<T, Abi>);
template <class T, class Abi, class U>
inline detail::return_type<datapar<T, Abi>, U> operator& (const U &, datapar<T, Abi>);
template <class T, class Abi, class U>
inline detail::return_type<datapar<T, Abi>, U> operator| (const U &, datapar<T, Abi>);
template <class T, class Abi, class U>
inline detail::return_type<datapar<T, Abi>, U> operator^ (const U &, datapar<T, Abi>);
template <class T, class Abi, class U>
inline detail::return_type<datapar<T, Abi>, U> operator<<(const U &, datapar<T, Abi>);
template <class T, class Abi, class U>
inline detail::return_type<datapar<T, Abi>, U> operator>>(const U &, datapar<T, Abi>);

// compares [datapar.comparison]
template <class T, class A, class U>
inline detail::cmp_return_type<datapar<T, A>, U> operator==(const datapar<T, A> &x,
                                                            const U &y)
{
    return detail::get_impl_t<detail::return_type<datapar<T, A>, U>>::equal_to(x, y);
}
template <class T, class A, class U>
inline detail::cmp_return_type<datapar<T, A>, U> operator!=(const datapar<T, A> &x,
                                                            const U &y)
{
    return detail::get_impl_t<detail::return_type<datapar<T, A>, U>>::not_equal_to(x, y);
}
template <class T, class A, class U>
inline detail::cmp_return_type<datapar<T, A>, U> operator<=(const datapar<T, A> &x,
                                                            const U &y)
{
    return detail::get_impl_t<detail::return_type<datapar<T, A>, U>>::less_equal(x, y);
}
template <class T, class A, class U>
inline detail::cmp_return_type<datapar<T, A>, U> operator>=(const datapar<T, A> &x,
                                                            const U &y)
{
    return detail::get_impl_t<detail::return_type<datapar<T, A>, U>>::less_equal(y, x);
}
template <class T, class A, class U>
inline detail::cmp_return_type<datapar<T, A>, U> operator<(const datapar<T, A> &x,
                                                           const U &y)
{
    return detail::get_impl_t<detail::return_type<datapar<T, A>, U>>::less(x, y);
}
template <class T, class A, class U>
inline detail::cmp_return_type<datapar<T, A>, U> operator>(const datapar<T, A> &x,
                                                           const U &y)
{
    return detail::get_impl_t<detail::return_type<datapar<T, A>, U>>::less(y, x);
}
template <class T, class A, class U,
          class = enable_if<!std::is_same<U, datapar<T, A>>::value>>
inline detail::cmp_return_type<datapar<T, A>, U> operator==(const U &x,
                                                            const datapar<T, A> &y)
{
    return detail::get_impl_t<detail::return_type<datapar<T, A>, U>>::equal_to(x, y);
}
template <class T, class A, class U,
          class = enable_if<!std::is_same<U, datapar<T, A>>::value>>
inline detail::cmp_return_type<datapar<T, A>, U> operator!=(const U &x,
                                                            const datapar<T, A> &y)
{
    return detail::get_impl_t<detail::return_type<datapar<T, A>, U>>::not_equal_to(x, y);
}
template <class T, class A, class U,
          class = enable_if<!std::is_same<U, datapar<T, A>>::value>>
inline detail::cmp_return_type<datapar<T, A>, U> operator<=(const U &x,
                                                            const datapar<T, A> &y)
{
    return detail::get_impl_t<detail::return_type<datapar<T, A>, U>>::less_equal(x, y);
}
template <class T, class A, class U,
          class = enable_if<!std::is_same<U, datapar<T, A>>::value>>
inline detail::cmp_return_type<datapar<T, A>, U> operator>=(const U &x,
                                                            const datapar<T, A> &y)
{
    return detail::get_impl_t<detail::return_type<datapar<T, A>, U>>::less_equal(y, x);
}
template <class T, class A, class U,
          class = enable_if<!std::is_same<U, datapar<T, A>>::value>>
inline detail::cmp_return_type<datapar<T, A>, U> operator<(const U &x,
                                                           const datapar<T, A> &y)
{
    return detail::get_impl_t<detail::return_type<datapar<T, A>, U>>::less(x, y);
}
template <class T, class A, class U,
          class = enable_if<!std::is_same<U, datapar<T, A>>::value>>
inline detail::cmp_return_type<datapar<T, A>, U> operator>(const U &x,
                                                           const datapar<T, A> &y)
{
    return detail::get_impl_t<detail::return_type<datapar<T, A>, U>>::less(y, x);
}

// casts [datapar.casts]
#ifndef Vc_CLANG
template <class T, class U, class... Us, size_t NN = U::size() + Us::size()...>
inline std::conditional_t<(T::size() == NN), T, std::array<T, NN / T::size()>>
    datapar_cast(U, Us...);
#endif

// mask binary operators [mask.binary]
namespace detail
{
template <class T0, class A0, class T1, class A1> struct mask_return_type_impl {

};
template <class T0, class A0, class T1, class A1>
using mask_return_type = typename mask_return_type_impl<T0, A0, T1, A1>::type;
}  // namespace detail
template <class T0, class A0, class T1, class A1>
inline detail::mask_return_type<T0, A0, T1, A1> operator&&(const mask<T0, A0> &,
                                                           const mask<T1, A1> &);
template <class T0, class A0, class T1, class A1>
inline detail::mask_return_type<T0, A0, T1, A1> operator||(const mask<T0, A0> &,
                                                           const mask<T1, A1> &);
template <class T0, class A0, class T1, class A1>
inline detail::mask_return_type<T0, A0, T1, A1> operator&(const mask<T0, A0> &,
                                                          const mask<T1, A1> &);
template <class T0, class A0, class T1, class A1>
inline detail::mask_return_type<T0, A0, T1, A1> operator|(const mask<T0, A0> &,
                                                          const mask<T1, A1> &);
template <class T0, class A0, class T1, class A1>
inline detail::mask_return_type<T0, A0, T1, A1> operator^(const mask<T0, A0> &,
                                                          const mask<T1, A1> &);

// mask compares [mask.comparison]
template <class T0, class A0, class T1, class A1>
inline std::enable_if_t<
    std::disjunction<std::is_convertible<mask<T0, A0>, mask<T1, A1>>,
                     std::is_convertible<mask<T1, A1>, mask<T0, A0>>>::value,
    bool>
operator==(const mask<T0, A0> &x, const mask<T1, A1> &y)
{
    return std::equal_to<mask<T0, A0>>{}(x, y);
}
template <class T0, class A0, class T1, class A1>
inline auto operator!=(const mask<T0, A0> &x, const mask<T1, A1> &y)
{
    return !operator==(x, y);
}

// reductions [mask.reductions]
template <class T, class Abi> inline bool all_of(const mask<T, Abi> &k)
{
    constexpr int N = datapar_size_v<T, Abi>;
    for (int i = 0; i < N; ++i) {
        if (!k[i]) {
            return false;
        }
    }
    return true;
}

template <class T, class Abi> inline bool any_of(const mask<T, Abi> &k)
{
    constexpr int N = datapar_size_v<T, Abi>;
    for (int i = 0; i < N; ++i) {
        if (k[i]) {
            return true;
        }
    }
    return false;
}

template <class T, class Abi> inline bool none_of(const mask<T, Abi> &k)
{
    constexpr int N = datapar_size_v<T, Abi>;
    for (int i = 0; i < N; ++i) {
        if (k[i]) {
            return false;
        }
    }
    return true;
}

template <class T, class Abi> inline bool some_of(const mask<T, Abi> &k)
{
    constexpr int N = datapar_size_v<T, Abi>;
    for (int i = 1; i < N; ++i) {
        if (k[i] != k[i - 1]) {
            return true;
        }
    }
    return false;
}

template <class T, class Abi> inline int popcount(const mask<T, Abi> &k)
{
    constexpr int N = datapar_size_v<T, Abi>;
    int n = k[0];
    for (int i = 1; i < N; ++i) {
        n += k[i];
    }
    return n;
}

template <class T, class Abi> inline int find_first_set(const mask<T, Abi> &k)
{
    constexpr int N = datapar_size_v<T, Abi>;
    for (int i = 0; i < N; ++i) {
        if (k[i]) {
            return i;
        }
    }
    return -1;
}

template <class T, class Abi> inline int find_last_set(const mask<T, Abi> &k)
{
    constexpr int N = datapar_size_v<T, Abi>;
    for (int i = N - 1; i >= 0; --i) {
        if (k[i]) {
            return i;
        }
    }
    return -1;
}

#if !defined VC_COMMON_ALGORITHMS_H_
constexpr bool all_of(bool x) { return x; }
constexpr bool any_of(bool x) { return x; }
constexpr bool none_of(bool x) { return !x; }
constexpr bool some_of(bool) { return false; }
#endif
constexpr int popcount(bool x) { return x; }
constexpr int find_first_set(bool) { return 0; }
constexpr int find_last_set(bool) { return 0; }

// masked assignment [mask.where]
template <class T0, class A0, class T1, class A1>
inline detail::where_proxy<mask<T1, A1>, datapar<T1, A1>> where(const mask<T0, A0> &k,
                                                                datapar<T1, A1> &d)
{
    return {k, d};
}
template <class T> inline detail::where_proxy<bool, T> where(bool k, T &d)
{
    return {k, d};
}
}  // namespace Vc_VERSIONED_NAMESPACE

#endif  // VC_DATAPAR_SYNOPSIS_H_

// vim: foldmethod=marker
