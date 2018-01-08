/*{{{
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

#ifndef VC_DETAIL_DEBUG_H_
#define VC_DETAIL_DEBUG_H_

#include "global.h"
#include <iostream>
#include <sstream>

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
#ifdef Vc_MSVC
#define Vc_PRETTY_FUNCTION __FUNCSIG__
#else
#define Vc_PRETTY_FUNCTION __PRETTY_FUNCTION__
#endif

enum class area : unsigned {
    _disabled = 0,
    _enabled = 1,

#ifdef Vc_DEBUG
#define Vc_ENABLE_DEBUG 1

#define sine       0x0000000000000001ull
#define cosine     0x0000000000000002ull
#define simd_tuple 0x0000000000000004ull

    _sine       = ((Vc_DEBUG) &       sine) ? _enabled : _disabled,
    _cosine     = ((Vc_DEBUG) &     cosine) ? _enabled : _disabled,
    _simd_tuple = ((Vc_DEBUG) & simd_tuple) ? _enabled : _disabled,
#undef sine
#undef cosine
#undef simd_tuple

#undef Vc_DEBUG

#else // Vc_DEBUG
    _sine = _disabled,
    _cosine = _disabled,
    _simd_tuple = _disabled,
#endif // Vc_DEBUG
};

#define Vc_DEBUG(area_)                                                                  \
    Vc::detail::debug_stream<Vc::detail::area::_##area_>(Vc_PRETTY_FUNCTION, __FILE__,   \
                                                         __LINE__)

#define Vc_PRETTY_PRINT(var_) std::setw(16), #var_ " = ", (var_)

#ifdef Vc_ENABLE_DEBUG
#define Vc_DEBUG_DEFERRED(area_, ...)                                                    \
    const auto &Vc_CONCAT(Vc_deferred_, __LINE__, _) =                                   \
        detail::defer([&]() { Vc_DEBUG(area_)(__VA_ARGS__); });
#else   // Vc_ENABLE_DEBUG
#define Vc_DEBUG_DEFERRED(area_, ...)
#endif  // Vc_ENABLE_DEBUG

template <area> class debug_stream;

#ifdef Vc_ENABLE_DEBUG
template <> class debug_stream<area::_enabled>
{
    std::stringstream buffer;
    int color = 31;

public:
    debug_stream(const char *func, const char *file, int line)
    {
        buffer << "\033[1;40;" << color << "mDEBUG: " << file << ':' << line
               << "\n       " << func;
    }

    ~debug_stream()
    {
        buffer << "\033[0m\n";
        std::cout << buffer.str() << std::flush;
    }

    template <class... Ts> debug_stream &operator()(const Ts &... args)
    {
        color = color > 37 ? 30 : color + 1;
        buffer << "\n\033[1;40;" << color << "m       ";
        //buffer << "\n        ";
#if 0 // __cpp_fold_expressions
        buffer << ... << std::forward<Ts>(args);
#else
        [](const std::initializer_list<int> &) {}({(print(args, int()), 0)...});
#endif
        return *this;
    }

private:
    template <class T, class = decltype(buffer << std::declval<const T &>())>
    void print(const T &x, int)
    {
        buffer << x;
    }

    static char hexChar(char x) { return x + (x > 9 ? 87 : 48); }
    template <class T> void print(const T &x, float)
    {
        using Bytes = char[sizeof(T)];
        auto &&bytes = reinterpret_cast<const Bytes &>(x);
        int i = -1;
        for (const unsigned char b : bytes) {
            if (++i && (i & 0x3) == 0) {
                buffer.put('\'');
            }
            buffer.put(hexChar(b >> 4));
            buffer.put(hexChar(b & 0xf));
        }
    }
};
#endif  // Vc_ENABLE_DEBUG

template <> class debug_stream<area::_disabled>
{
public:
    debug_stream(const char *, const char *, int) {}
    template <class... Ts> const debug_stream &operator()(Ts &&...) const { return *this; }
};

template <typename F> class defer_raii
{
public:
    // construct the object from the given callable
    template <typename FF> defer_raii(FF &&f) : cleanup_function(std::forward<FF>(f)) {}

    // when the object goes out of scope call the cleanup function
    ~defer_raii() { cleanup_function(); }

private:
    F cleanup_function;
};

template <typename F> detail::defer_raii<F> defer(F && f) { return {std::forward<F>(f)}; }

}  // namespace detail
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_DETAIL_DEBUG_H_
