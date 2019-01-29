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

#include <experimental/simd>
#include <vir/test.h>
#include <iostream>
#include <iomanip>

_GLIBCXX_SIMD_BEGIN_NAMESPACE
template <typename T, typename Abi>
inline std::ostream &operator<<(std::ostream &out, const simd<T, Abi> &v)
{
    using namespace vir::detail::color;
    if constexpr (std::is_floating_point_v<T>) {
        out << green << '[';
        out << v[0] << blue << " (" << std::hexfloat << v[0] << std::defaultfloat << ')';
        for (size_t i = 1; i < v.size(); ++i) {
            out << green << ", " << v[i] << blue << " (" << std::hexfloat << v[i]
                << std::defaultfloat << ')';
        }
        return out << ']' << normal;
    } else {
        using TT = std::conditional_t<(sizeof(T) < sizeof(int)), int, T>;
        out << green << '[';
        out << TT(v[0]);
        for (size_t i = 1; i < v.size(); ++i) {
            out << ", " << TT(v[i]);
        }
        return out << ']' << normal;
    }
}

template <typename T, typename Abi>
inline std::ostream &operator<<(std::ostream &out, const simd_mask<T, Abi> &v)
{
    using namespace vir::detail::color;
    out << blue << "m[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i > 0 && (i % 4) == 0) {
            out << ' ';
        }
        if (v[i]) {
            out << yellow << '1';
        } else {
            out << blue << '0';
        }
    }
    return out << blue << ']' << normal;
}
_GLIBCXX_SIMD_END_NAMESPACE

template <class T> struct help {
    using type = T;
};
template <class T, class A> struct help<std::experimental::simd_mask<T, A>> {
    using type = std::experimental::simd<T, A>;
};

// vir::test::compare_traits specialization {{{
template <class Lhs, class Rhs, class = void> struct vc2_compare_traits {
    static constexpr bool is_mask =
        std::experimental::is_simd_mask_v<Lhs> || std::experimental::is_simd_mask_v<Rhs>;
    using lhs_t = typename help<Lhs>::type;
    using rhs_t = typename help<Rhs>::type;
    using common_simd_type = decltype(std::declval<lhs_t>() + std::declval<rhs_t>());
    using common_type = std::conditional_t<is_mask, typename common_simd_type::mask_type,
                                           common_simd_type>;
    using value_type = typename common_type::value_type;
    static constexpr bool use_memcompare = false;
    static constexpr bool is_fuzzy_comparable = std::is_floating_point<value_type>::value;
    static inline bool is_equal(const common_type &a, const common_type &b)
    {
        return all_of(a == b);
    }

    static inline common_type ulp_distance(const common_type &a, const common_type &b)
    {
        static_assert(std::is_floating_point<value_type>::value, "");
        return vir::detail::ulpDiffToReference(a, b);
    }

    static inline common_type ulp_distance_signed(const common_type &a,
                                                  const common_type &b)
    {
        static_assert(std::is_floating_point<value_type>::value, "");
        return vir::detail::ulpDiffToReferenceSigned(a, b);
    }

    static inline bool ulp_compare_and_log(const common_type &ulp,
                                           const common_type &allowed_distance)
    {
        using delegate_traits = vir::test::compare_traits<value_type, value_type>;
        for (std::size_t i = 0; i < common_type::size(); ++i) {
            if (!delegate_traits::ulp_compare_and_log(ulp[i], allowed_distance[i])) {
                return false;
            }
        }
        return true;
    }

    template <class... Ts>
    static inline std::string to_datafile_string(const common_type &d0,
                                                 const Ts &... data)
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < common_type::size(); ++i) {
            ss << std::setprecision(50) << d0[i];
            auto unused = {((ss << '\t' << data[i]), 0)...};
            ss << ((void)unused, '\n');
        }
        return ss.str();
    }
};

// SFINAE for invalid Vector<T, Abi>
/*
template <class Lhs, class Rhs>
struct vc2_compare_traits<
    Lhs, Rhs,
    typename std::enable_if<!Vc::is_simd_vector<Lhs>::value &&
                            !Vc::is_simd_vector<Rhs>::value>::type> {
};
*/

namespace vir
{
namespace test
{
template <class T0, class A0, class T1, class A1>
struct compare_traits<std::experimental::simd<T0, A0>, std::experimental::simd<T1, A1>>
    : vc2_compare_traits<std::experimental::simd<T0, A0>, std::experimental::simd<T1, A1>> {
};
template <class Lhs, class T1, class A1>
struct compare_traits<Lhs, std::experimental::simd<T1, A1>>
    : vc2_compare_traits<Lhs, std::experimental::simd<T1, A1>> {
};
template <class T0, class A0, class Rhs>
struct compare_traits<std::experimental::simd<T0, A0>, Rhs>
    : vc2_compare_traits<std::experimental::simd<T0, A0>, Rhs> {
};

template <class T0, class A0, class T1, class A1>
struct compare_traits<std::experimental::simd_mask<T0, A0>, std::experimental::simd_mask<T1, A1>>
    : vc2_compare_traits<std::experimental::simd_mask<T0, A0>, std::experimental::simd_mask<T1, A1>> {
};
}  // namespace test
}  // namespace vir

// }}}

// vim: foldmethod=marker
