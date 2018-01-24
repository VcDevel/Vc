/*  This file is part of the Vc library. {{{
Copyright © 2018 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_TESTS_SIMD_VIEW_H_
#define VC_TESTS_SIMD_VIEW_H_

#include <Vc/simd>

Vc_VERSIONED_NAMESPACE_BEGIN
namespace experimental
{
namespace imported_begin_end
{
    using std::begin;
    using std::end;
    template <class T> using begin_type = decltype(begin(std::declval<T>()));
    template <class T> using end_type = decltype(end(std::declval<T>()));
}  // namespace imported_begin_end

template <class V, class It, class End> class viewer
{
    It it;
    const End end;

    template <class F> void for_each_impl(F &&fun, std::index_sequence<0, 1, 2>)
    {
        for (; it + V::size() <= end; it += V::size()) {
            fun(V([&](auto i) { return std::get<0>(it[i].as_tuple()); }),
                V([&](auto i) { return std::get<1>(it[i].as_tuple()); }),
                V([&](auto i) { return std::get<2>(it[i].as_tuple()); }));
        }
        if (it != end) {
            fun(V([&](auto i) {
                    auto ii = it + i < end ? i + 0 : 0;
                    return std::get<0>(it[ii].as_tuple());
                }),
                V([&](auto i) {
                    auto ii = it + i < end ? i + 0 : 0;
                    return std::get<1>(it[ii].as_tuple());
                }),
                V([&](auto i) {
                    auto ii = it + i < end ? i + 0 : 0;
                    return std::get<2>(it[ii].as_tuple());
                }));
        }
    }

    template <class F> void for_each_impl(F &&fun, std::index_sequence<0, 1>)
    {
        for (; it + V::size() <= end; it += V::size()) {
            fun(V([&](auto i) { return std::get<0>(it[i].as_tuple()); }),
                V([&](auto i) { return std::get<1>(it[i].as_tuple()); }));
        }
        if (it != end) {
            fun(V([&](auto i) {
                    auto ii = it + i < end ? i + 0 : 0;
                    return std::get<0>(it[ii].as_tuple());
                }),
                V([&](auto i) {
                    auto ii = it + i < end ? i + 0 : 0;
                    return std::get<1>(it[ii].as_tuple());
                }));
        }
    }

public:
    viewer(It _it, End _end) : it(_it), end(_end) {}

    template <class F> void for_each(F &&fun) {
        constexpr size_t N =
            std::tuple_size<std::decay_t<decltype(it->as_tuple())>>::value;
        for_each_impl(std::forward<F>(fun), std::make_index_sequence<N>());
    }
};

template <class V, class Cont>
viewer<V, imported_begin_end::begin_type<const Cont &>,
       imported_begin_end::end_type<const Cont &>>
simd_view(const Cont &data)
{
    using std::begin;
    using std::end;
    return {begin(data), end(data)};
}
}  // namespace experimental
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_TESTS_SIMD_VIEW_H_
