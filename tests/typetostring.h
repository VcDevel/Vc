/*  This file is part of the Vc library. {{{
Copyright © 2014-2015 Matthias Kretz <kretz@kde.org>

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
#ifndef VC_TO_STRING_H
#define VC_TO_STRING_H
#include <sstream>
#include "typelist.h"
#include <Vc/Vc>
#ifdef HAVE_CXX_ABI_H
#include <cxxabi.h>
#endif

template <typename T> inline std::string typeToString();
// std::array<T, N> {{{2
template <typename T, std::size_t N>
inline std::string typeToString_impl(std::array<T, N> const &)
{
    std::stringstream s;
    s << "array<" << typeToString<T>() << ", " << N << '>';
    return s.str();
}
// std::vector<T> {{{2
template <typename T> inline std::string typeToString_impl(std::vector<T> const &)
{
    std::stringstream s;
    s << "vector<" << typeToString<T>() << '>';
    return s.str();
}
// std::integral_constant<T, N> {{{2
template <typename T, T N>
inline std::string typeToString_impl(std::integral_constant<T, N> const &)
{
    std::stringstream s;
    s << "integral_constant<" << N << '>';
    return s.str();
}
// SimdArray to string {{{2
template <typename T, std::size_t N, typename V, std::size_t M>
inline std::string typeToString_impl(Vc::SimdArray<T, N, V, M> const &)
{
    std::stringstream s;
    s << "SimdArray<" << typeToString<T>() << ", " << N << '>';
    return s.str();
}
template <typename T, std::size_t N, typename V, std::size_t M>
inline std::string typeToString_impl(Vc::SimdMaskArray<T, N, V, M> const &)
{
    std::stringstream s;
    s << "SimdMaskArray<" << typeToString<T>() << ", " << N << ", " << typeToString<V>()
      << '>';
    return s.str();
}
// template parameter pack to a comma separated string {{{2
template <typename T0, typename... Ts>
std::string typeToString_impl(Typelist<T0, Ts...> const &)
{
    std::stringstream s;
    s << '{' << typeToString<T0>();
    auto &&x = {(s << ", " << typeToString<Ts>(), 0)...};
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Waddress"
#endif
    if (&x == nullptr) {}  // avoid warning about unused 'x'
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
    s << '}';
    return s.str();
}
// Vc::<Impl>::Vector<T> to string {{{2
template <typename V>
inline std::string typeToString_impl(
    V const &, typename std::enable_if<Vc::is_simd_vector<V>::value, int>::type = 0)
{
    using T = typename V::EntryType;
    std::stringstream s;
    if (std::is_same<V, Vc::Scalar::Vector<T>>::value) {
        s << "Scalar::";
    } else if (std::is_same<V, Vc::SSE::Vector<T>>::value) {
        s << "SSE::";
    } else if (std::is_same<V, Vc::AVX2::Vector<T>>::value) {
        s << "AVX::";
    } else if (std::is_same<V, Vc::MIC::Vector<T>>::value) {
        s << "MIC::";
    }
    s << typeToString<T>() << "_v";
    return s.str();
}
template <typename V>
inline std::string typeToString_impl(
    V const &, typename std::enable_if<Vc::is_simd_mask<V>::value, int>::type = 0)
{
    using T = typename V::EntryType;
    std::stringstream s;
    if (std::is_same<V, Vc::Scalar::Mask<T>>::value) {
        s << "Scalar::";
    } else if (std::is_same<V, Vc::SSE::Mask<T>>::value) {
        s << "SSE::";
    } else if (std::is_same<V, Vc::AVX2::Mask<T>>::value) {
        s << "AVX::";
    } else if (std::is_same<V, Vc::MIC::Mask<T>>::value) {
        s << "MIC::";
    }
    s << typeToString<T>() << "_m";
    return s.str();
}

// Vc::datapar to string {{{2
template <int N>
inline std::string typeToString_impl(const Vc::datapar_abi::fixed_size<N> &)
{
    return std::to_string(N);
}
inline std::string typeToString_impl(const Vc::datapar_abi::scalar &) { return "scalar"; }
inline std::string typeToString_impl(const Vc::datapar_abi::sse &) { return "sse"; }
inline std::string typeToString_impl(const Vc::datapar_abi::avx &) { return "avx"; }
inline std::string typeToString_impl(const Vc::datapar_abi::avx512 &) { return "avx512"; }
inline std::string typeToString_impl(const Vc::datapar_abi::knc &) { return "knc"; }

template <class T, class A>
inline std::string typeToString_impl(const Vc::datapar<T, A> &)
{
    return "datapar<" + typeToString<T>() + ", " + typeToString<A>() + '>';
}

template <class T, class A>
inline std::string typeToString_impl(const Vc::mask<T, A> &)
{
    return "mask<" + typeToString<T>() + ", " + typeToString<A>() + '>';
}

// generic fallback (typeid::name) {{{2
template <typename T>
inline std::string typeToString_impl(
    T const &,
    typename std::enable_if<!Vc::is_simd_vector<T>::value && !Vc::is_simd_mask<T>::value,
                            int>::type = 0)
{
#ifdef HAVE_CXX_ABI_H
    char buf[1024];
    size_t size = 1024;
    abi::__cxa_demangle(typeid(T).name(), buf, &size, nullptr);
    return std::string{buf};
#else
    return typeid(T).name();
#endif
}
// typeToString specializations {{{2
template <typename T> inline std::string typeToString() { return typeToString_impl(T()); }
template <> inline std::string typeToString<void>() { return ""; }
template <> inline std::string typeToString<long double>() { return "long double"; }
template <> inline std::string typeToString<double>() { return "double"; }
template <> inline std::string typeToString<float>() { return " float"; }
template <> inline std::string typeToString<long long>() { return " llong"; }
template <> inline std::string typeToString<unsigned long long>() { return "ullong"; }
template <> inline std::string typeToString<long>() { return "  long"; }
template <> inline std::string typeToString<unsigned long>() { return " ulong"; }
template <> inline std::string typeToString<int>() { return "   int"; }
template <> inline std::string typeToString<unsigned int>() { return "  uint"; }
template <> inline std::string typeToString<short>() { return " short"; }
template <> inline std::string typeToString<unsigned short>() { return "ushort"; }
template <> inline std::string typeToString<char>() { return "  char"; }
template <> inline std::string typeToString<unsigned char>() { return " uchar"; }
template <> inline std::string typeToString<signed char>() { return " schar"; }
#endif

// vim: foldmethod=marker
