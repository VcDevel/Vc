/*  This file is part of the Vc library. {{{
Copyright © 2017 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_SIMD_AARCH_CONVERT_H_
#define VC_SIMD_AARCH_CONVERT_H_

#include "storage.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
namespace aarch
{
// convert_builtin{{{1
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
template <typename To, typename From, size_t... I>
Vc_INTRINSIC To convert_builtin(From v0, std::index_sequence<I...>)
{
    using T = typename To::EntryType;
    return typename To::Builtin{static_cast<T>(v0[I])...};
}

template <typename To, typename From, size_t... I>
Vc_INTRINSIC To convert_builtin(From v0, From v1, std::index_sequence<I...>)
{
    using T = typename To::EntryType;
    return typename To::Builtin{static_cast<T>(v0[I])..., static_cast<T>(v1[I])...};
}

template <typename To, typename From, size_t... I>
Vc_INTRINSIC To convert_builtin(From v0, From v1, From v2, From v3,
                                std::index_sequence<I...>)
{
    using T = typename To::EntryType;
    return typename To::Builtin{static_cast<T>(v0[I])..., static_cast<T>(v1[I])...,
                                static_cast<T>(v2[I])..., static_cast<T>(v3[I])...};
}

template <typename To, typename From, size_t... I>
Vc_INTRINSIC To convert_builtin(From v0, From v1, From v2, From v3, From v4, From v5,
                                From v6, From v7, std::index_sequence<I...>)
{
    using T = typename To::EntryType;
    return typename To::Builtin{static_cast<T>(v0[I])..., static_cast<T>(v1[I])...,
                                static_cast<T>(v2[I])..., static_cast<T>(v3[I])...,
                                static_cast<T>(v4[I])..., static_cast<T>(v5[I])...,
                                static_cast<T>(v6[I])..., static_cast<T>(v7[I])...};
}

template <typename To, typename From, size_t... I0, size_t... I1>
Vc_INTRINSIC To convert_builtin(From v0, From v1, std::index_sequence<I0...>,
                                std::index_sequence<I1...>)
{
    using T = typename To::EntryType;
    return typename To::Builtin{static_cast<T>(v0[I0])..., static_cast<T>(v1[I0])...,
                                (I1, T{})...};
}

template <typename To, typename From, size_t... I0, size_t... I1>
Vc_INTRINSIC To convert_builtin(From v0, From v1, From v2, From v3,
                                std::index_sequence<I0...>, std::index_sequence<I1...>)
{
    using T = typename To::EntryType;
    return typename To::Builtin{static_cast<T>(v0[I0])..., static_cast<T>(v1[I0])...,
                                static_cast<T>(v2[I0])..., static_cast<T>(v3[I0])...,
                                (I1, T{})...};
}

template <typename To, typename From, size_t... I0, size_t... I1>
Vc_INTRINSIC To convert_builtin(From v0, From v1, From v2, From v3, From v4, From v5,
                                From v6, From v7, std::index_sequence<I0...>,
                                std::index_sequence<I1...>)
{
    using T = typename To::EntryType;
    return typename To::Builtin{
        static_cast<T>(v0[I0])..., static_cast<T>(v1[I0])..., static_cast<T>(v2[I0])...,
        static_cast<T>(v3[I0])..., static_cast<T>(v4[I0])..., static_cast<T>(v5[I0])...,
        static_cast<T>(v6[I0])..., static_cast<T>(v7[I0])..., (I1, T{})...};
}
#endif  // Vc_USE_BUILTIN_VECTOR_TYPES

// convert_to declarations{{{1
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_f32 v0);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_f32 v0, x_f32 v1);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_f32 v0, x_f32 v1, x_f32 v2, x_f32 v3);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_f64 v0);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_f64 v0, x_f64 v1);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_f64 v0, x_f64 v1, x_f64 v2, x_f64 v3);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_f64 v0, x_f64 v1, x_f64 v2, x_f64 v3, x_f64 v4, x_f64 v5, x_f64 v6, x_f64 v7);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_s08);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_u08);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_s16);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_s16, x_s16);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_u16);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_u16, x_u16);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_s32);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_s32, x_s32);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_s32, x_s32, x_s32, x_s32);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_u32);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_u32, x_u32);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_u32, x_u32, x_u32, x_u32);
//}}}1

// generic (u)long forwarding to (u)(llong|int){{{1

template <typename To, size_t N> Vc_INTRINSIC To Vc_VDECL convert_to(Storage<long, N> v)
{
    return convert_to<To>(Storage<equal_int_type_t<long>, N>(v));
}
template <typename To, size_t N>
Vc_INTRINSIC To Vc_VDECL convert_to(Storage<long, N> v0, Storage<long, N> v1)
{
    return convert_to<To>(Storage<equal_int_type_t<long>, N>(v0),
                          Storage<equal_int_type_t<long>, N>(v1));
}
template <typename To, size_t N>
Vc_INTRINSIC To Vc_VDECL convert_to(Storage<long, N> v0, Storage<long, N> v1, Storage<long, N> v2,
                           Storage<long, N> v3)
{
    return convert_to<To>(
        Storage<equal_int_type_t<long>, N>(v0), Storage<equal_int_type_t<long>, N>(v1),
        Storage<equal_int_type_t<long>, N>(v2), Storage<equal_int_type_t<long>, N>(v3));
}
template <typename To, size_t N>
Vc_INTRINSIC To Vc_VDECL convert_to(Storage<long, N> v0, Storage<long, N> v1, Storage<long, N> v2,
                           Storage<long, N> v3, Storage<long, N> v4, Storage<long, N> v5,
                           Storage<long, N> v6, Storage<long, N> v7)
{
    return convert_to<To>(
        Storage<equal_int_type_t<long>, N>(v0), Storage<equal_int_type_t<long>, N>(v1),
        Storage<equal_int_type_t<long>, N>(v2), Storage<equal_int_type_t<long>, N>(v3),
        Storage<equal_int_type_t<long>, N>(v4), Storage<equal_int_type_t<long>, N>(v5),
        Storage<equal_int_type_t<long>, N>(v6), Storage<equal_int_type_t<long>, N>(v7));
}

template <typename To, size_t N> Vc_INTRINSIC To Vc_VDECL convert_to(Storage<ulong, N> v)
{
    return convert_to<To>(Storage<equal_int_type_t<ulong>, N>(v));
}
template <typename To, size_t N>
Vc_INTRINSIC To Vc_VDECL convert_to(Storage<ulong, N> v0, Storage<ulong, N> v1)
{
    return convert_to<To>(Storage<equal_int_type_t<ulong>, N>(v0),
                          Storage<equal_int_type_t<ulong>, N>(v1));
}
template <typename To, size_t N>
Vc_INTRINSIC To Vc_VDECL convert_to(Storage<ulong, N> v0, Storage<ulong, N> v1, Storage<ulong, N> v2,
                           Storage<ulong, N> v3)
{
    return convert_to<To>(
        Storage<equal_int_type_t<ulong>, N>(v0), Storage<equal_int_type_t<ulong>, N>(v1),
        Storage<equal_int_type_t<ulong>, N>(v2), Storage<equal_int_type_t<ulong>, N>(v3));
}
template <typename To, size_t N>
Vc_INTRINSIC To Vc_VDECL convert_to(Storage<ulong, N> v0, Storage<ulong, N> v1, Storage<ulong, N> v2,
                           Storage<ulong, N> v3, Storage<ulong, N> v4, Storage<ulong, N> v5,
                           Storage<ulong, N> v6, Storage<ulong, N> v7)
{
    return convert_to<To>(
        Storage<equal_int_type_t<ulong>, N>(v0), Storage<equal_int_type_t<ulong>, N>(v1),
        Storage<equal_int_type_t<ulong>, N>(v2), Storage<equal_int_type_t<ulong>, N>(v3),
        Storage<equal_int_type_t<ulong>, N>(v4), Storage<equal_int_type_t<ulong>, N>(v5),
        Storage<equal_int_type_t<ulong>, N>(v6), Storage<equal_int_type_t<ulong>, N>(v7));
}

// generic forwarding for down-conversions to unsigned int{{{1
struct scalar_conversion_fallback_tag {};
template <typename T> struct fallback_int_type { using type = scalar_conversion_fallback_tag; };
template <> struct fallback_int_type< uchar> { using type = schar; };
template <> struct fallback_int_type<ushort> { using type = short; };
template <> struct fallback_int_type<  uint> { using type = int; };

template <typename T>
using equivalent_storage_t =
    Storage<typename fallback_int_type<typename T::EntryType>::type, T::size()>;

template <typename To, typename From>
Vc_INTRINSIC std::conditional_t<
    (std::is_integral<typename To::EntryType>::value &&
     sizeof(typename To::EntryType) <= sizeof(typename From::EntryType)),
    Storage<std::make_signed_t<typename From::EntryType>, From::size()>, From>
    Vc_VDECL maybe_make_signed(From v)
{
    static_assert(
        std::is_unsigned<typename From::EntryType>::value,
        "maybe_make_signed must only be used with unsigned integral Storage types");
    return std::conditional_t<
        (std::is_integral<typename To::EntryType>::value &&
         sizeof(typename To::EntryType) <= sizeof(typename From::EntryType)),
        Storage<std::make_signed_t<typename From::EntryType>, From::size()>, From>{v};
}

template <typename To,
          typename Fallback = typename fallback_int_type<typename To::EntryType>::type>
struct equivalent_conversion {
    template <size_t N, typename... From>
    static Vc_INTRINSIC Vc_CONST To Vc_VDECL convert(Storage<uchar, N> v0, From... vs)
    {
        using S = Storage<Fallback, To::size()>;
        return convert_to<S>(maybe_make_signed<To>(v0), maybe_make_signed<To>(vs)...).v();
    }

    template <size_t N, typename... From>
    static Vc_INTRINSIC Vc_CONST To Vc_VDECL convert(Storage<ushort, N> v0, From... vs)
    {
        using S = Storage<Fallback, To::size()>;
        return convert_to<S>(maybe_make_signed<To>(v0), maybe_make_signed<To>(vs)...).v();
    }

    template <size_t N, typename... From>
    static Vc_INTRINSIC Vc_CONST To Vc_VDECL convert(Storage<uint, N> v0, From... vs)
    {
        using S = Storage<Fallback, To::size()>;
        return convert_to<S>(maybe_make_signed<To>(v0), maybe_make_signed<To>(vs)...).v();
    }

    template <size_t N, typename... From>
    static Vc_INTRINSIC Vc_CONST To Vc_VDECL convert(Storage<ulong, N> v0, From... vs)
    {
        using S = Storage<Fallback, To::size()>;
        return convert_to<S>(maybe_make_signed<To>(v0), maybe_make_signed<To>(vs)...).v();
    }

    template <size_t N, typename... From>
    static Vc_INTRINSIC Vc_CONST To Vc_VDECL convert(Storage<ullong, N> v0, From... vs)
    {
        using S = Storage<Fallback, To::size()>;
        return convert_to<S>(maybe_make_signed<To>(v0), maybe_make_signed<To>(vs)...).v();
    }

    template <typename F0, typename... From>
    static Vc_INTRINSIC Vc_CONST To Vc_VDECL convert(F0 v0, From... vs)
    {
        static_assert(!std::is_unsigned<typename F0::EntryType>::value, "overload error");
        using S = Storage<Fallback, To::size()>;
        return convert_to<S>(v0, vs...).v();
    }
};

// fallback: scalar aggregate conversion{{{1
template <typename To> struct equivalent_conversion<To, scalar_conversion_fallback_tag> {
    template <typename From, typename... Fs>
    static Vc_INTRINSIC Vc_CONST To Vc_VDECL convert(From v0, Fs... vs)
    {
        using F = typename From::value_type;
        using T = typename To::value_type;
        static_assert(sizeof(F) >= sizeof(T) && std::is_integral<T>::value &&
                          std::is_unsigned<F>::value,
                      "missing an implementation for convert<To>(From, Fs...)");
        using S = Storage<typename fallback_int_type<F>::type, From::size()>;
        return convert_to<To>(S(v0), S(vs)...);
    }
};

// convert_to implementations invoking the fallbacks{{{1
/*
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_f32 v0) { return equivalent_conversion<To>::convert(v0); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_f32 v0, x_f32 v1) { return equivalent_conversion<To>::convert(v0, v1); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_f32 v0, x_f32 v1, x_f32 v2, x_f32 v3) { return equivalent_conversion<To>::convert(v0, v1, v2, v3); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_f64 v0) { return equivalent_conversion<To>::convert(v0); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_f64 v0, x_f64 v1) { return equivalent_conversion<To>::convert(v0, v1); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_f64 v0, x_f64 v1, x_f64 v2, x_f64 v3) { return equivalent_conversion<To>::convert(v0, v1, v2, v3); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_f64 v0, x_f64 v1, x_f64 v2, x_f64 v3, x_f64 v4, x_f64 v5, x_f64 v6, x_f64 v7) { return equivalent_conversion<To>::convert(v0, v1, v2, v3, v4, v5, v6, v7); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_i08 v0) { return equivalent_conversion<To>::convert(v0); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_u08 v0) { return equivalent_conversion<To>::convert(v0); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_i16 v0) { return equivalent_conversion<To>::convert(v0); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_i16 v0, x_i16 v1) { return equivalent_conversion<To>::convert(v0, v1); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_u16 v0) { return equivalent_conversion<To>::convert(v0); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_u16 v0, x_u16 v1) { return equivalent_conversion<To>::convert(v0, v1); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_i32 v0) { return equivalent_conversion<To>::convert(v0); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_i32 v0, x_i32 v1) { return equivalent_conversion<To>::convert(v0, v1); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_i32 v0, x_i32 v1, x_i32 v2, x_i32 v3) { return equivalent_conversion<To>::convert(v0, v1, v2, v3); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_u32 v0) { return equivalent_conversion<To>::convert(v0); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_u32 v0, x_u32 v1) { return equivalent_conversion<To>::convert(v0, v1); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_u32 v0, x_u32 v1, x_u32 v2, x_u32 v3) { return equivalent_conversion<To>::convert(v0, v1, v2, v3); }
*/
// convert function{{{1
template <typename From, typename To> Vc_INTRINSIC To Vc_VDECL convert(From v)
{
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
    constexpr auto N = From::size() < To::size() ? From::size() : To::size();
    return convert_builtin<To>(v.builtin(), std::make_index_sequence<N>());
#else
    return convert_to<To>(v);
#endif
}

template <typename From, typename To> Vc_INTRINSIC To Vc_VDECL convert(From v0, From v1)
{
    static_assert(To::size() >= 2 * From::size(),
                  "convert(v0, v1) requires the input to fit into the output");
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
    return convert_builtin<To>(
        v0.builtin(), v1.builtin(), std::make_index_sequence<From::size()>(),
        std::make_index_sequence<To::size() - 2 * From::size()>());
#else
    return convert_to<To>(v0, v1);
#endif
}

template <typename From, typename To>
Vc_INTRINSIC To Vc_VDECL convert(From v0, From v1, From v2, From v3)
{
    static_assert(To::size() >= 4 * From::size(),
                  "convert(v0, v1, v2, v3) requires the input to fit into the output");
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
    return convert_builtin<To>(
        v0.builtin(), v1.builtin(), v2.builtin(), v3.builtin(),
        std::make_index_sequence<From::size()>(),
        std::make_index_sequence<To::size() - 4 * From::size()>());
#else
    return convert_to<To>(v0, v1, v2, v3);
#endif
}

template <typename From, typename To>
Vc_INTRINSIC To Vc_VDECL convert(From v0, From v1, From v2, From v3, From v4, From v5, From v6,
                        From v7)
{
    static_assert(To::size() >= 8 * From::size(),
                  "convert(v0, v1, v2, v3, v4, v5, v6, v7) "
                  "requires the input to fit into the output");
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
    return convert_builtin<To>(
        v0.builtin(), v1.builtin(), v2.builtin(), v3.builtin(), v4.builtin(),
        v5.builtin(), v6.builtin(), v7.builtin(),
        std::make_index_sequence<From::size()>(),
        std::make_index_sequence<To::size() - 8 * From::size()>());
#else
    return convert_to<To>(v0, v1, v2, v3, v4, v5, v6, v7);
#endif
}

/*
// convert_all function{{{1
template <typename To, typename From>
Vc_INTRINSIC auto Vc_VDECL convert_all_impl(From v, std::true_type)
{
    constexpr size_t N = From::size() / To::size();
    return generate_from_n_evaluations<N, std::array<To, N>>([&](auto i) {
        using namespace Vc::detail::x86;  // ICC needs this to find convert and
                                          // shift_right below.
        constexpr int shift = decltype(i)::value  // MSVC needs this instead of a simple
                                                  // `i`, apparently their conversion
                                                  // operator is not (considered)
                                                  // constexpr.
                              * To::size() * sizeof(From) / From::size();
        return convert<From, To>(shift_right<shift>(v));
    });
}

*/

template <typename To, typename From>
Vc_INTRINSIC To Vc_VDECL convert_all_impl(From v, std::false_type)
{
    return convert<From, To>(v);
}

template <typename To, typename From> Vc_INTRINSIC auto Vc_VDECL convert_all(From v)
{
    return convert_all_impl<To, From>(
        v, std::integral_constant<bool, (From::size() > To::size())>());
}

// }}}1
}}
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_SIMD_AARCH_CONVERT_H_
