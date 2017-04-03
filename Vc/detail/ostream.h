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

#ifndef VC_DATAPAR_OSTREAM_H_
#define VC_DATAPAR_OSTREAM_H_

#include <ostream>
#include "datapar.h"

#if defined(__GNUC__) && !defined(_WIN32) && defined(_GLIBCXX_OSTREAM)
#define Vc_HACK_OSTREAM_FOR_TTY 1
#endif

#ifdef Vc_HACK_OSTREAM_FOR_TTY
#include <unistd.h>
#include <ext/stdio_sync_filebuf.h>
#endif

Vc_VERSIONED_NAMESPACE_BEGIN
// color{{{1
namespace detail
{
#ifdef Vc_HACK_OSTREAM_FOR_TTY
static bool isATty(const std::ostream &os)
{
    __gnu_cxx::stdio_sync_filebuf<char> *hack =
        dynamic_cast<__gnu_cxx::stdio_sync_filebuf<char> *>(os.rdbuf());
    if (!hack) {
        return false;
    }
    FILE *file = hack->file();
    return 1 == isatty(fileno(file));
}
Vc_ALWAYS_INLINE Vc_CONST bool mayUseColor(const std::ostream &os)
{
    static int result = -1;
    if (Vc_IS_UNLIKELY(result == -1)) {
        result = isATty(os);
    }
    return result;
}
#else
constexpr bool mayUseColor(const std::ostream &) { return false; }
#endif

namespace color
{
struct Color {
    const char *data;
};

static constexpr Color red = {"\033[1;40;31m"};
static constexpr Color green = {"\033[1;40;32m"};
static constexpr Color yellow = {"\033[1;40;33m"};
static constexpr Color blue = {"\033[1;40;34m"};
static constexpr Color normal = {"\033[0m"};

inline std::ostream &operator<<(std::ostream &out, const Color &c)
{
    if (mayUseColor(out)) {
        out << c.data;
    }
    return out;
}
}  // namespace color
}  // namespace detail

// datapar output{{{1
template <class T, class Abi>
std::ostream &operator<<(std::ostream &out, const datapar<T, Abi> &v)
{
    using U = std::conditional_t<sizeof(T) == 1, int, T>;
    out << detail::color::green << "v⃗[" << U(v[0]);
    for (size_t i = 1; i < v.size(); ++i) {
        out << (i % 4 == 0 ? " | " : ", ") << U(v[i]);
    }
    return out << ']' << detail::color::normal;
}

// mask output{{{1
template <class T, class Abi>
std::ostream &operator<<(std::ostream &out, const mask<T, Abi> &k)
{
    auto &&printBool = [&](bool b) {
        if (b) {
            out << detail::color::yellow << '1';
        } else {
            out << detail::color::blue << '0';
        }
    };
    out << detail::color::blue << "m⃗[";
    printBool(k[0]);
    for (size_t i = 1; i < k.size(); ++i) {
        if (i % 4 == 0) {
            out << ' ';
        }
        printBool(k[i]);
    }
    return out << ']' << detail::color::normal;
}

//}}}1
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_DATAPAR_OSTREAM_H_
