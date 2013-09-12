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

#ifndef VC_COMMON_SIMD_ARRAY_H
#define VC_COMMON_SIMD_ARRAY_H

#include <type_traits>
#include <array>

#include "simd_array_data.h"
#include "simd_mask_array.h"
#include "macros.h"

Vc_PUBLIC_NAMESPACE_BEGIN

template<typename T, std::size_t N> class simd_array
{
    static_assert(std::is_same<T,   double>::value ||
                  std::is_same<T,    float>::value ||
                  std::is_same<T,  int32_t>::value ||
                  std::is_same<T, uint32_t>::value ||
                  std::is_same<T,  int16_t>::value ||
                  std::is_same<T, uint16_t>::value, "simd_array<T, N> may only be used with T = { double, float, int32_t, uint32_t, int16_t, uint16_t }");

    static_assert((N & (N - 1)) == 0, "simd_array<T, N> must be used with a power of two value for N.");

public:
    typedef Vc::Vector<T> vector_type;
    typedef T value_type;
    typedef simd_mask_array<T, N> mask_type;

    static constexpr std::size_t size = N;
    static constexpr std::size_t register_count = size > vector_type::Size ? size / vector_type::Size : 1;

    // Vc compat:
    typedef mask_type Mask;
    typedef value_type EntryType;
    static constexpr std::size_t Size = size;

    // zero init
    simd_array() = default;

    // default copy ctor/operator
    simd_array(const simd_array &) = default;
    simd_array(simd_array &&) = default;
    simd_array &operator=(const simd_array &) = default;

    // broadcast
    Vc_ALWAYS_INLINE simd_array(value_type a) : d(a) {}

    // load ctors
    explicit Vc_ALWAYS_INLINE simd_array(const value_type *x) : d(x) {}
    template<typename Flags = AlignedT> explicit Vc_ALWAYS_INLINE simd_array(const value_type *x, Flags flags = Flags())
        : d(x, flags) {}
    template<typename OtherT, typename Flags = AlignedT> explicit Vc_ALWAYS_INLINE simd_array(const OtherT *x, Flags flags = Flags())
        : d(x, flags) {}

    ///////////////////////////////////////////////////////////////////////////////////////////
    // load member functions
    Vc_ALWAYS_INLINE void load(const value_type *x) {
        d.call(static_cast<void (vector_type::*)(const value_type *)>(&vector_type::load), x);
    }
    template<typename Flags>
    Vc_ALWAYS_INLINE void load(const value_type *x, Flags f) {
        d.call(static_cast<void (vector_type::*)(const value_type *, Flags)>(&vector_type::load), x, f);
    }
    template<typename U, typename Flags>
    Vc_ALWAYS_INLINE void load(const U *x, Flags f) {
        d.call(static_cast<void (vector_type::*)(const U *, Flags)>(&vector_type::load), x, f);
    }

    // implicit casts
    template<typename U> Vc_ALWAYS_INLINE simd_array(const simd_array<U, N> &x) {
        d[0] = x.data(0);
    }

#define VC_COMPARE_IMPL(op) \
    Vc_ALWAYS_INLINE Vc_PURE mask_type operator op(const simd_array &x) const { \
        mask_type r; \
        r.d.assign(d, x.d, &vector_type::operator op); \
        return r; \
    }
    VC_ALL_COMPARES(VC_COMPARE_IMPL)
#undef VC_COMPARE_IMPL

#define VC_OPERATOR_IMPL(op) \
    Vc_ALWAYS_INLINE simd_array &operator op##=(const simd_array &x) { \
        for (std::size_t i = 0; i < register_count; ++i) { \
            d[i] op##= x.d[i]; \
        } \
        return *this; \
    } \
    inline simd_array operator op(const simd_array &x) const { \
        simd_array r; \
        for (std::size_t i = 0; i < register_count; ++i) { \
            r.data(i) = d[i] op x.d[i]; \
        } \
        return r; \
    }
    VC_ALL_BINARY     (VC_OPERATOR_IMPL)
    VC_ALL_ARITHMETICS(VC_OPERATOR_IMPL)
    VC_ALL_SHIFTS     (VC_OPERATOR_IMPL)
#undef VC_OPERATOR_IMPL

    value_type operator[](std::size_t i) const {
        typedef value_type TT Vc_MAY_ALIAS;
        auto m = reinterpret_cast<const TT *>(&d);
        return m[i];
    }

private:
    Common::ArrayData<vector_type, register_count> d;
};

Vc_NAMESPACE_END

#include "undomacros.h"

#endif // VC_COMMON_SIMD_ARRAY_H
