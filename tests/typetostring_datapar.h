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

#ifndef VC_TESTS_TYPETOSTRING_DATAPAR_H_
#define VC_TESTS_TYPETOSTRING_DATAPAR_H_

#include <string>
#include <Vc/datapar>

// Vc::datapar to string
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

#endif  // VC_TESTS_TYPETOSTRING_DATAPAR_H_
