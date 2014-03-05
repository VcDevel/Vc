/*  This file is part of the Vc library. {{{

    Copyright (C) 2013 Matthias Kretz <kretz@kde.org>

    Vc is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    Vc is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with Vc.  If not, see <http://www.gnu.org/licenses/>.

}}}*/

#ifndef VC_COMMON_SIMD_MASK_ARRAY_H
#define VC_COMMON_SIMD_MASK_ARRAY_H

#include <type_traits>
#include <array>
#include "simd_array_data.h"

#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{

template <
    typename T,
    std::size_t N,
    typename VectorType = typename Common::select_best_vector_type<N,
#ifdef VC_IMPL_AVX
                                                                   Vc::Vector<T>,
                                                                   Vc::SSE::Vector<T>,
                                                                   Vc::Scalar::Vector<T>
#elif defined(VC_IMPL_Scalar)
                                                                   Vc::Vector<T>
#else
                                                                   Vc::Vector<T>,
                                                                   Vc::Scalar::Vector<T>
#endif
                                                                   >::type>
class simd_mask_array
{
    using vector_type = VectorType;

public:
    typedef typename vector_type::Mask mask_type;
    static constexpr std::size_t size() { return N; }
    static constexpr std::size_t Size = size();
    static constexpr std::size_t register_count = size() > mask_type::Size ? size() / mask_type::Size : 1;

    typedef Common::MaskData<mask_type, register_count> storage_type;

    // zero init
    simd_mask_array() = default;

    Vc_ALWAYS_INLINE explicit simd_mask_array(VectorSpecialInitializerZero::ZEnum) : d(false) {}
    Vc_ALWAYS_INLINE explicit simd_mask_array(VectorSpecialInitializerOne::OEnum) : d(true) {}
    Vc_ALWAYS_INLINE simd_mask_array(bool x) : d(x) {}

    // default copy ctor/operator
    simd_mask_array(const simd_mask_array &) = default;
    simd_mask_array(simd_mask_array &&) = default;
    template <typename U> simd_mask_array(const simd_mask_array<U, N> &x) : d(x.d)
    {
    }
    simd_mask_array &operator=(const simd_mask_array &) = default;

    Vc_ALWAYS_INLINE Vc_PURE bool isFull() const { return d.isFull(); }
    Vc_ALWAYS_INLINE Vc_PURE bool isEmpty() const { return d.isEmpty(); }

#define VC_COMPARE_IMPL(op)                                                                        \
    Vc_ALWAYS_INLINE Vc_PURE bool operator op(const simd_mask_array &x) const                      \
    {                                                                                              \
        return d.apply([](bool l, bool r) { return l && r; },                                      \
                       [](mask_type l, mask_type r) { return l op r; },                            \
                       x.d);                                                                       \
    }
    VC_ALL_COMPARES(VC_COMPARE_IMPL)
#undef VC_COMPARE_IMPL

    bool operator[](std::size_t i) const {
        const auto m = d.cbegin();
        return m[i / mask_type::Size][i % mask_type::Size];
    }

    simd_mask_array operator!() const
    {
        simd_mask_array r;
        r.d.assign(d, &mask_type::operator!);
        return r;
    }

    unsigned int count() const
    {
        return d.count();
    }

//private:
    storage_type d;

    friend const decltype(d) & simd_mask_array_data(const simd_mask_array &x) { return x.d; }
    friend decltype(d) & simd_mask_array_data(simd_mask_array &x) { return x.d; }
    friend decltype(std::move(d)) simd_mask_array_data(simd_mask_array &&x) { return std::move(x.d); }
};

}

#include "undomacros.h"

#endif // VC_COMMON_SIMD_MASK_ARRAY_H
