/*  This file is part of the Vc library. {{{
Copyright © 2010-2017 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_SIMD_STORAGE_H_
#define VC_SIMD_STORAGE_H_

#include <iosfwd>

#include "macros.h"
#include "builtins.h"
#include "detail.h"
#include "const.h"
#ifdef Vc_HAVE_NEON_ABI
#include "aarch/intrinsics.h"
#elif defined Vc_HAVE_SSE_ABI
#include "x86/types.h"
#endif

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
// Storage<bool>{{{1
template <size_t Width>
struct Storage<bool, Width, std::void_t<typename bool_storage_member_type<Width>::type>> {
    using register_type = typename bool_storage_member_type<Width>::type;
    using value_type = bool;
    static constexpr size_t width = Width;
    [[deprecated]] static constexpr size_t size() { return Width; }

    constexpr Vc_INTRINSIC Storage() = default;
    constexpr Vc_INTRINSIC Storage(register_type k) : d(k){};

    Vc_INTRINSIC Vc_PURE operator const register_type &() const { return d; }
    Vc_INTRINSIC Vc_PURE operator register_type &() { return d; }

    Vc_INTRINSIC register_type intrin() const { return d; }

    Vc_INTRINSIC Vc_PURE value_type operator[](size_t i) const
    {
        return d & (register_type(1) << i);
    }
    Vc_INTRINSIC void set(size_t i, value_type x)
    {
        if (x) {
            d |= (register_type(1) << i);
        } else {
            d &= ~(register_type(1) << i);
        }
    }

    register_type d;
};

// StorageBase{{{1
template <class T, size_t Width, class RegisterType = builtin_type_t<T, Width>,
          bool = std::disjunction_v<
              std::is_same<builtin_type_t<T, Width>, intrinsic_type_t<T, Width>>,
              std::is_same<RegisterType, intrinsic_type_t<T, Width>>>>
struct StorageBase;

template <class T, size_t Width, class RegisterType>
struct StorageBase<T, Width, RegisterType, true> {
    RegisterType d;
    constexpr Vc_INTRINSIC StorageBase() = default;
    constexpr Vc_INTRINSIC StorageBase(builtin_type_t<T, Width> x)
        : d(reinterpret_cast<RegisterType>(x))
    {
    }
};

template <class T, size_t Width, class RegisterType>
struct StorageBase<T, Width, RegisterType, false> {
    using intrin_type = intrinsic_type_t<T, Width>;
    RegisterType d;

    constexpr Vc_INTRINSIC StorageBase() = default;
    constexpr Vc_INTRINSIC StorageBase(builtin_type_t<T, Width> x)
        : d(reinterpret_cast<RegisterType>(x))
    {
    }
    constexpr Vc_INTRINSIC StorageBase(intrin_type x)
        : d(reinterpret_cast<RegisterType>(x))
    {
    }

    constexpr Vc_INTRINSIC operator intrin_type() const
    {
        return reinterpret_cast<intrin_type>(d);
    }
};

// StorageEquiv {{{1
template <typename T, size_t Width, bool = detail::has_same_value_representation_v<T>>
struct StorageEquiv : StorageBase<T, Width> {
    using StorageBase<T, Width>::d;
    constexpr Vc_INTRINSIC StorageEquiv() = default;
    template <class U, class = decltype(StorageBase<T, Width>(std::declval<U>()))>
    constexpr Vc_INTRINSIC StorageEquiv(U &&x) : StorageBase<T, Width>(std::forward<U>(x))
    {
    }
    // I want to use ctor inheritance, but it breaks always_inline. Having a function that
    // does a single movaps is stupid.
    //using StorageBase<T, Width>::StorageBase;
};

// This base class allows conversion to & from
// * builtin_type_t<equal_int_type_t<T>, Width>, and
// * Storage<equal_int_type_t<T>, Width>
// E.g. Storage<long, 4> is convertible to & from
// * builtin_type_t<long long, 4>, and
// * Storage<long long, 4>
// on LP64
// * builtin_type_t<int, 4>, and
// * Storage<int, 4>
// on ILP32, and LLP64
template <class T, size_t Width>
struct StorageEquiv<T, Width, true>
    : StorageBase<equal_int_type_t<T>, Width, builtin_type_t<T, Width>> {
    using Base = StorageBase<equal_int_type_t<T>, Width, builtin_type_t<T, Width>>;
    using Base::d;
    template <class U,
              class = decltype(StorageBase<equal_int_type_t<T>, Width,
                                           builtin_type_t<T, Width>>(std::declval<U>()))>
    constexpr Vc_INTRINSIC StorageEquiv(U &&x)
        : StorageBase<equal_int_type_t<T>, Width, builtin_type_t<T, Width>>(
              std::forward<U>(x))
    {
    }
    // I want to use ctor inheritance, but it breaks always_inline. Having a function that
    // does a single movaps is stupid.
    //using Base::StorageBase;

    constexpr Vc_INTRINSIC StorageEquiv() = default;

    // convertible from intrin_type, builtin_type_t<equal_int_type_t<T>, Width> and
    // builtin_type_t<T, Width>, and Storage<equal_int_type_t<T>, Width>
    constexpr Vc_INTRINSIC StorageEquiv(builtin_type_t<T, Width> x)
        : Base(reinterpret_cast<builtin_type_t<equal_int_type_t<T>, Width>>(x))
    {
    }
    constexpr Vc_INTRINSIC StorageEquiv(Storage<equal_int_type_t<T>, Width> x)
        : Base(x.d)
    {
    }

    // convertible to intrin_type, builtin_type_t<equal_int_type_t<T>, Width> and
    // builtin_type_t<T, Width> (in Storage), and Storage<equal_int_type_t<T>, Width>
    //
    // intrin_type<T> is handled by StorageBase
    // builtin_type_t<T> is handled by Storage
    // builtin_type_t<equal_int_type_t<T>> is handled in StorageEquiv, i.e. here:
    constexpr Vc_INTRINSIC operator builtin_type_t<equal_int_type_t<T>, Width>() const
    {
        return reinterpret_cast<builtin_type_t<equal_int_type_t<T>, Width>>(d);
    }
    constexpr Vc_INTRINSIC operator Storage<equal_int_type_t<T>, Width>() const
    {
        return reinterpret_cast<builtin_type_t<equal_int_type_t<T>, Width>>(d);
    }

    constexpr Vc_INTRINSIC Storage<equal_int_type_t<T>, Width> equiv() const
    {
        return reinterpret_cast<builtin_type_t<equal_int_type_t<T>, Width>>(d);
    }
};

// StorageBroadcast{{{1
template <class T, size_t Width> struct StorageBroadcast;
template <class T> struct StorageBroadcast<T, 2> {
    static constexpr Vc_INTRINSIC Storage<T, 2> broadcast(T x)
    {
        return builtin_type_t<T, 2>{x, x};
    }
};
template <class T> struct StorageBroadcast<T, 4> {
    static constexpr Vc_INTRINSIC Storage<T, 4> broadcast(T x)
    {
        return builtin_type_t<T, 4>{x, x, x, x};
    }
};
template <class T> struct StorageBroadcast<T, 8> {
    static constexpr Vc_INTRINSIC Storage<T, 8> broadcast(T x)
    {
        return builtin_type_t<T, 8>{x, x, x, x, x, x, x, x};
    }
};
template <class T> struct StorageBroadcast<T, 16> {
    static constexpr Vc_INTRINSIC Storage<T, 16> broadcast(T x)
    {
        return builtin_type_t<T, 16>{x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x};
    }
};
template <class T> struct StorageBroadcast<T, 32> {
    static constexpr Vc_INTRINSIC Storage<T, 32> broadcast(T x)
    {
        return builtin_type_t<T, 32>{x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x,
                                     x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x};
    }
};
template <class T> struct StorageBroadcast<T, 64> {
    static constexpr Vc_INTRINSIC Storage<T, 64> broadcast(T x)
    {
        return builtin_type_t<T, 64>{x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x,
                                     x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x,
                                     x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x,
                                     x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x};
    }
};

// Storage{{{1
template <typename T, size_t Width>
struct Storage<T, Width,
               std::void_t<builtin_type_t<T, Width>, intrinsic_type_t<T, Width>>>
    : StorageEquiv<T, Width>, StorageBroadcast<T, Width> {
    static_assert(is_vectorizable_v<T>);
    static_assert(Width >= 2);  // 1 doesn't make sense, use T directly then
    using register_type = builtin_type_t<T, Width>;
    using value_type = T;
    static constexpr size_t width = Width;
    [[deprecated("use width instead")]] static constexpr size_t size() { return Width; }

    constexpr Vc_INTRINSIC Storage() = default;
    template <class U, class = decltype(StorageEquiv<T, Width>(std::declval<U>()))>
    constexpr Vc_INTRINSIC Storage(U &&x) : StorageEquiv<T, Width>(std::forward<U>(x))
    {
    }
    // I want to use ctor inheritance, but it breaks always_inline. Having a function that
    // does a single movaps is stupid.
    //using StorageEquiv<T, Width>::StorageEquiv;
    using StorageEquiv<T, Width>::d;

    template <class... As,
              class = std::enable_if_t<((std::is_same_v<simd_abi::scalar, As> && ...) &&
                                        sizeof...(As) <= Width)>>
    constexpr Vc_INTRINSIC operator simd_tuple<T, As...>() const
    {
        const auto &dd = d;  // workaround for GCC7 ICE
        return detail::generate_from_n_evaluations<sizeof...(As), simd_tuple<T, As...>>(
            [&](auto i) { return dd[int(i)]; });
    }

    constexpr Vc_INTRINSIC operator const register_type &() const { return d; }
    constexpr Vc_INTRINSIC operator register_type &() { return d; }

    [[deprecated("use .d instead")]] constexpr Vc_INTRINSIC const register_type &builtin() const { return d; }
    [[deprecated("use .d instead")]] constexpr Vc_INTRINSIC register_type &builtin() { return d; }

    template <class U = intrinsic_type_t<T, Width>>
    constexpr Vc_INTRINSIC U intrin() const
    {
        return reinterpret_cast<U>(d);
    }
    [[deprecated(
        "use intrin() instead")]] constexpr Vc_INTRINSIC intrinsic_type_t<T, Width>
    v() const
    {
        return intrin();
    }

    constexpr Vc_INTRINSIC T operator[](size_t i) const { return d[i]; }
    [[deprecated("use operator[] instead")]] constexpr Vc_INTRINSIC T m(size_t i) const
    {
        return d[i];
    }

    Vc_INTRINSIC void set(size_t i, T x) { d[i] = x; }
};

// to_storage {{{1
template <class T> class to_storage
{
    static_assert(is_builtin_vector_v<T>);
    T d;

public:
    constexpr to_storage(T x) : d(x) {}
    template <class U, size_t N> constexpr operator Storage<U, N>() const
    {
        static_assert(sizeof(builtin_type_t<U, N>) == sizeof(T));
        return {reinterpret_cast<builtin_type_t<U, N>>(d)};
    }
};

// to_storage_unsafe {{{1
template <class T> class to_storage_unsafe
{
    T d;
public:
    constexpr to_storage_unsafe(T x) : d(x) {}
    template <class U, size_t N> constexpr operator Storage<U, N>() const
    {
        static_assert(sizeof(builtin_type_t<U, N>) <= sizeof(T));
        return {reinterpret_cast<builtin_type_t<U, N>>(d)};
    }
};

// storage_bitcast{{{1
template <class T, class U, size_t M, size_t N = sizeof(U) * M / sizeof(T)>
constexpr Vc_INTRINSIC Storage<T, N> storage_bitcast(Storage<U, M> x)
{
    static_assert(sizeof(builtin_type_t<T, N>) == sizeof(builtin_type_t<U, M>));
    return reinterpret_cast<builtin_type_t<T, N>>(x.d);
}

// make_storage{{{1
template <class T, class... Args>
constexpr Vc_INTRINSIC Storage<T, sizeof...(Args)> make_storage(Args &&... args)
{
    return {typename Storage<T, sizeof...(Args)>::register_type{static_cast<T>(args)...}};
}

// generate_storage{{{1
template <class T, size_t N, class G>
constexpr Vc_INTRINSIC Storage<T, N> generate_storage(G &&gen)
{
    return generate_builtin<T, N>(std::forward<G>(gen));
}

// work around clang miscompilation on set{{{1
#if defined Vc_CLANG && !defined Vc_HAVE_SSE4_1
#if Vc_CLANG <= 0x60000
template <> void Storage<double, 2, AliasStrategy::VectorBuiltin>::set(size_t i, double x)
{
    if (x == 0. && i == 1)
        asm("" : "+g"(x));  // make clang forget that x is 0
    d[i] = x;
}
#else
#warning "clang 5 failed operators_sse2_vectorbuiltin_ldouble_float_double_schar_uchar in operator<simd<double, __sse>> and required a workaround. Is this still the case for newer clang versions?"
#endif
#endif

// Storage ostream operators{{{1
template <class CharT, class T, size_t N>
inline std::basic_ostream<CharT> &operator<<(std::basic_ostream<CharT> & s,
                                             const Storage<T, N> &v)
{
    s << '[' << v[0];
    for (size_t i = 1; i < N; ++i) {
        s << ((i % 4) ? " " : " | ") << v[i];
    }
    return s << ']';
}

//}}}1
}  // namespace detail
Vc_VERSIONED_NAMESPACE_END

#ifdef Vc_HAVE_NEON_ABI
#include "aarch/storage.h"
#elif defined Vc_HAVE_SSE_ABI
#include "x86/storage.h"
#endif

#endif  // VC_SIMD_STORAGE_H_

// vim: foldmethod=marker
