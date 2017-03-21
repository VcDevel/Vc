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

#include <limits>
#include <type_traits>

template <typename T, typename F>
inline typename std::enable_if<std::is_floating_point<T>::value>::type
iterateNumericRange(F &&f)
{
    using L = std::numeric_limits<T>;
    constexpr int nsteps = 17;
    const T step = L::max() / nsteps;
    for (int i = 0; i < nsteps; ++i) {
        f(T(L::lowest() + i * step));
    }
    for (int i = 0; i < nsteps; ++i) {
        f(T(L::max() - i * step));
    }
    f(T(-1));
    f(-L::min());
    f(-L::denorm_min());
    f(T(0));
    f(L::denorm_min());
    f(L::min());
    f(T(1));
}

template <typename T, typename F>
inline typename std::enable_if<std::is_integral<T>::value>::type iterateNumericRange(
    F &&f)
{
    using L = std::numeric_limits<T>;
    constexpr int nsteps = 17;
    const T step = L::max() / nsteps;
    for (int i = 0; i < nsteps; ++i) {
        f(T(L::max() - i * step));
    }
    f(T(1));
    f(T(0));
    if (std::is_signed<T>::value) {
        f(T(-1));
        for (int i = 0; i < nsteps; ++i) {
            f(T(L::min() + i * step));
        }
    }
}

// vim: foldmethod=marker
